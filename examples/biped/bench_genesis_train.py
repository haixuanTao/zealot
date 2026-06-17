#!/usr/bin/env python3
"""Genesis FULL training-iteration throughput (rollout + GAE + PPO update).

Matches iter_e2e_bench.rs: one iteration = T-step rollout + (epochs x minibatches)
PPO update; reports env-control-steps/s = N*T / iter_time (Isaac's "Computation"
unit). Config mirrors iter_e2e_bench defaults: T=32, epochs=5, minibatches=16,
actor [43,256,256,128,12], critic [52,512,256,128,1], clip 0.2, entropy 0.005,
value_coef 0.5, gamma 0.99, lambda 0.95, lr 1e-3.

Run one point: ~/genesis-venv/bin/python bench_genesis_train.py [N] [T] [epochs] [minibatches]
"""
import sys
import time

import torch
import torch.nn as nn

import biped_genesis_common as C
from genesis_biped_env import GenesisBipedEnv

CRITIC_OBS = 52  # obs(43) + base_lin_vel(3) + base_ang_vel(3)
GAMMA, LAM = 0.99, 0.95
CLIP, ENT_COEF, VAL_COEF, LR = 0.2, 0.005, 0.5, 1e-3


def mlp(dims, dev):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ELU())
    return nn.Sequential(*layers).to(dev)


def critic_obs(env, obs):
    # Privileged critic obs: policy obs(43) + base lin vel(3) + ang vel(3), padded
    # to CRITIC_OBS=52 so the critic net matches the nexus [52,512,...] for FLOPs
    # parity (exact privileged content is irrelevant to throughput).
    lin = env.robot.get_vel()
    ang = env.robot.get_ang() if hasattr(env.robot, "get_ang") else torch.zeros_like(lin)
    pad = torch.zeros(obs.shape[0], CRITIC_OBS - C.OBS_DIM - 6, device=obs.device)
    return torch.cat([obs, lin, ang, pad], dim=1)


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
    T = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    MB = int(sys.argv[4]) if len(sys.argv) > 4 else 16

    env = GenesisBipedEnv(N, on_gpu=True)
    dev = env.device
    actor = mlp([C.OBS_DIM, *C.HIDDEN, C.ACT_DIM], dev)
    critic = mlp([CRITIC_OBS, 512, 256, 128, 1], dev)
    log_std = torch.zeros(C.ACT_DIM, device=dev, requires_grad=True)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()) + [log_std], lr=LR)

    def run_iter():
        # ---- rollout (T steps), store transitions ----
        obs_buf = torch.zeros(T, N, C.OBS_DIM, device=dev)
        cobs_buf = torch.zeros(T, N, CRITIC_OBS, device=dev)
        act_buf = torch.zeros(T, N, C.ACT_DIM, device=dev)
        logp_buf = torch.zeros(T, N, device=dev)
        val_buf = torch.zeros(T, N, device=dev)
        rew_buf = torch.zeros(T, N, device=dev)
        done_buf = torch.zeros(T, N, device=dev)

        obs = env.get_obs()
        for t in range(T):
            cobs = critic_obs(env, obs)
            with torch.no_grad():
                mean = actor(obs)
                std = log_std.exp()
                action = mean + torch.randn_like(mean) * std
                logp = (-0.5 * ((action - mean) / std) ** 2 - log_std - 0.9189385).sum(1)
                val = critic(cobs).squeeze(1)
            obs_buf[t], cobs_buf[t], act_buf[t] = obs, cobs, action
            logp_buf[t], val_buf[t] = logp, val
            obs, rew, z = env.step(action)
            rew_buf[t] = rew
            done_buf[t] = (z < 0.3).float()

        with torch.no_grad():
            last_val = critic(critic_obs(env, obs)).squeeze(1)

        # ---- GAE ----
        adv = torch.zeros(T, N, device=dev)
        gae = torch.zeros(N, device=dev)
        nextval = last_val
        for t in reversed(range(T)):
            nonterm = 1.0 - done_buf[t]
            delta = rew_buf[t] + GAMMA * nextval * nonterm - val_buf[t]
            gae = delta + GAMMA * LAM * nonterm * gae
            adv[t] = gae
            nextval = val_buf[t]
        ret = adv + val_buf
        # flatten
        b_obs = obs_buf.reshape(T * N, C.OBS_DIM)
        b_cobs = cobs_buf.reshape(T * N, CRITIC_OBS)
        b_act = act_buf.reshape(T * N, C.ACT_DIM)
        b_logp = logp_buf.reshape(T * N)
        b_adv = adv.reshape(T * N)
        b_ret = ret.reshape(T * N)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        # ---- PPO update: epochs x minibatches ----
        total = T * N
        mb_size = total // MB
        for _ in range(EPOCHS):
            perm = torch.randperm(total, device=dev)
            for m in range(MB):
                idx = perm[m * mb_size:(m + 1) * mb_size]
                mean = actor(b_obs[idx])
                std = log_std.exp()
                logp = (-0.5 * ((b_act[idx] - mean) / std) ** 2 - log_std - 0.9189385).sum(1)
                ratio = (logp - b_logp[idx]).exp()
                a = b_adv[idx]
                pg = -torch.min(ratio * a, torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * a).mean()
                v = critic(b_cobs[idx]).squeeze(1)
                vloss = ((v - b_ret[idx]) ** 2).mean()
                ent = (log_std + 0.5 + 0.9189385).sum()
                loss = pg + VAL_COEF * vloss - ENT_COEF * ent
                opt.zero_grad()
                loss.backward()
                opt.step()

    # warmup iter (kernel compile / autotune), then time one full iter.
    run_iter()
    torch.cuda.synchronize()
    t0 = time.time()
    run_iter()
    torch.cuda.synchronize()
    wall = time.time() - t0
    env_ctrl = N * T / wall
    print(
        f"GENESIS_TRAIN N={N} T={T} epochs={EPOCHS} mb={MB} iter_s={wall:.3f} "
        f"env_ctrl_per_s={env_ctrl:.1f}"
    )


if __name__ == "__main__":
    main()

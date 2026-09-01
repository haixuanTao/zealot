#!/usr/bin/env python3
"""Ablation: the zealot PPO update, reimplemented in PyTorch.

The rollout stays in nexus (GPU physics) — it is not replaceable by torch.
What IS replaceable is the update: policy/value forward, PPO clipped loss,
backward, Adam. This script runs *exactly* the update the Rust trainer runs,
at the same shapes, and times it against the measured Rust numbers.

Reference config (biped_train_gpu, N=4096, G1 29-DOF, BIPED_ROBOT default):
    actor   395 -> 256 -> 256 -> 128 -> 12     (ELU, tanh-free, diagonal gaussian)
    critic   90 -> 512 -> 256 -> 128 -> 1
    batch   N*T = 4096*24 = 98_304 samples, mirror-augmented -> 196_608
    update  EPOCHS=5 x (MINIBATCHES=4 doubled by mirror aug) = 40 optimizer steps
    Adam lr 1e-3, global grad-norm clip 1.0, PPO clip 0.2

Usage:
    python bench/torch_ppo_update.py [--envs 4096] [--iters 10] [--compile]
"""
import argparse
import time

import torch
import torch.nn as nn

P = argparse.ArgumentParser()
P.add_argument("--envs", type=int, default=4096)
P.add_argument("--steps", type=int, default=24)
P.add_argument("--epochs", type=int, default=5)
P.add_argument("--minibatches", type=int, default=4)
P.add_argument("--mirror", type=int, default=1, help="mirror augmentation doubles the batch")
P.add_argument("--iters", type=int, default=10, help="timed iterations")
P.add_argument("--obs", type=int, default=395)
P.add_argument("--cobs", type=int, default=90)
P.add_argument("--act", type=int, default=12)
P.add_argument("--compile", action="store_true")
P.add_argument("--tf32", type=int, default=1)
P.add_argument("--transfer", action="store_true", help="also time the host round-trip a torch pipeline would pay")
A = P.parse_args()

dev = torch.device("cuda")
if A.tf32:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def mlp(dims):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


actor = mlp([A.obs, 256, 256, 128, A.act]).to(dev)
critic = mlp([A.cobs, 512, 256, 128, 1]).to(dev)
log_std = nn.Parameter(torch.zeros(A.act, device=dev))
opt = torch.optim.Adam(
    list(actor.parameters()) + list(critic.parameters()) + [log_std], lr=1e-3
)

if A.compile:
    actor = torch.compile(actor)
    critic = torch.compile(critic)

total = A.envs * A.steps
batch = total * (2 if A.mirror else 1)
mb = total // A.minibatches                      # minibatch SIZE is held fixed
n_mb = batch // mb                               # ...so mirror doubles the COUNT
steps_per_iter = A.epochs * n_mb

# One rollout batch, resident on device (the Rust trainer stages it there too).
obs = torch.randn(batch, A.obs, device=dev)
cobs = torch.randn(batch, A.cobs, device=dev)
act = torch.randn(batch, A.act, device=dev)
old_logp = torch.randn(batch, device=dev)
adv = torch.randn(batch, device=dev)
ret = torch.randn(batch, device=dev)

CLIP = 0.2
LOG_SQRT_2PI = 0.9189385


def logprob(mean, std, a):
    z = (a - mean) / std
    return (-0.5 * z * z - torch.log(std) - LOG_SQRT_2PI).sum(-1)


def one_update():
    """Exactly one iteration's worth of PPO: epochs x minibatches."""
    for _ in range(A.epochs):
        perm = torch.randperm(batch, device=dev)
        for k in range(n_mb):
            idx = perm[k * mb:(k + 1) * mb]
            o, c, a = obs[idx], cobs[idx], act[idx]
            mean = actor(o)
            std = log_std.exp().expand_as(mean)
            lp = logprob(mean, std, a)
            ratio = (lp - old_logp[idx]).exp()
            adv_mb = adv[idx]
            adv_mb = (adv_mb - adv_mb.mean()) / (adv_mb.std() + 1e-8)
            pg = -torch.min(ratio * adv_mb,
                            ratio.clamp(1 - CLIP, 1 + CLIP) * adv_mb).mean()
            v = critic(c).squeeze(-1)
            vloss = (v - ret[idx]).pow(2).mean()
            ent = (log_std + 0.5 * (1 + torch.log(torch.tensor(2 * torch.pi)))).sum()
            loss = pg + 0.5 * vloss - 0.0 * ent
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(actor.parameters()) + list(critic.parameters()) + [log_std], 1.0)
            opt.step()


def timed(fn, n, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


print(f"device        : {torch.cuda.get_device_name(0)}")
print(f"torch         : {torch.__version__}  (cuda {torch.version.cuda}, tf32={bool(A.tf32)},"
      f" compile={A.compile})")
print(f"batch         : {batch:,} samples ({total:,} x{2 if A.mirror else 1} mirror)")
print(f"update        : {A.epochs} epochs x {n_mb} minibatches of {mb:,} = {steps_per_iter} steps")
print(f"actor         : {A.obs} -> 256 -> 256 -> 128 -> {A.act}")
print(f"critic        : {A.cobs} -> 512 -> 256 -> 128 -> 1")

ms = timed(one_update, A.iters)
print(f"\nPPO update    : {ms:8.1f} ms / iteration   ({ms / steps_per_iter:.2f} ms / optimizer step)")

if A.transfer:
    # What a torch pipeline pays to get the nexus rollout out of the Rust env
    # and back: the batch crosses the bus once per iteration.
    host = {k: torch.empty_like(v, device="cpu").pin_memory()
            for k, v in [("obs", obs), ("cobs", cobs), ("act", act)]}

    def roundtrip():
        for k, v in [("obs", obs), ("cobs", cobs), ("act", act)]:
            host[k].copy_(v, non_blocking=True)          # D2H
        for k, v in [("obs", obs), ("cobs", cobs), ("act", act)]:
            v.copy_(host[k], non_blocking=True)          # H2D
    tms = timed(roundtrip, A.iters)
    nbytes = sum(v.numel() * 4 for v in (obs, cobs, act))
    print(f"host round-trip: {tms:8.1f} ms / iteration   "
          f"({nbytes / 1e6:.0f} MB each way, pinned)")

#!/usr/bin/env python3
"""Critic-value divergence: nexus (training engine) vs sim2sim (MuJoCo).

The critic V(s) is a learned scalar = "how good is this state" under the nexus
training distribution. Running the SAME policy from the SAME start in nexus and
in MuJoCo, the two state trajectories diverge as soon as the engines' dynamics
differ — and V tracks that as a single number. Where V_mujoco peels away from
V_nexus pinpoints WHEN MuJoCo leaves the nexus training manifold (the sim-to-sim
gap), much earlier/cleaner than waiting for a fall.

Critic-obs (51) is reconstructed IDENTICALLY for both trajectories (45 actor-obs
+ 6 base lin/ang vel via finite-diff of the base pose, body frame) so the V gap
reflects state divergence, not a methodology asymmetry.

Usage: python3 critic_divergence.py <nexus_rollout.json> <policy.safetensors> [out.png] [steps]
"""
import os, sys, json
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from safetensors.numpy import load_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim2sim_xval as X

ROLLOUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/v56_final_rollout.json"
POLICY  = sys.argv[2] if len(sys.argv) > 2 else "/tmp/v56_net.safetensors"
OUT     = sys.argv[3] if len(sys.argv) > 3 else "/tmp/v56_critic_divergence.png"
N_STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 300
CMD = np.array([0.4, 0.0, 0.0, 0.0])


class Critic:
    """critic.w_l/b_l ELU stack (linear scalar output) + critic_norm."""
    def __init__(self, path):
        sd = load_file(path)
        self.W, self.b = [], []
        l = 0
        while f"critic.w_{l}" in sd:
            self.W.append(sd[f"critic.w_{l}"].astype(np.float64))
            self.b.append(sd[f"critic.b_{l}"].astype(np.float64))
            l += 1
        self.n = l
        self.mean = sd["critic_norm.mean"].astype(np.float64)
        self.m2 = sd["critic_norm.m2"].astype(np.float64)
        self.count = float(sd["critic_norm.count"].reshape(-1)[0])

    def value(self, cobs):
        var = np.maximum(self.m2 / self.count, 1e-8)
        a = np.clip((np.asarray(cobs) - self.mean) / np.sqrt(var), -5.0, 5.0)
        for i in range(self.n):
            z = self.W[i] @ a + self.b[i]
            a = np.where(z > 0, z, np.exp(z) - 1.0) if i < self.n - 1 else z
        return float(a[0])


def qmul(a, b):  # xyzw
    ax, ay, az, aw = a; bx, by, bz, bw = b
    return np.array([aw*bx+ax*bw+ay*bz-az*by,
                     aw*by-ax*bz+ay*bw+az*bx,
                     aw*bz+ax*by-ay*bx+az*bw,
                     aw*bw-ax*bx-ay*by-az*bz])


def qconj(q):  # xyzw
    return np.array([-q[0], -q[1], -q[2], q[3]])


def body_vels(p_prev, q_prev, p_cur, q_cur, dt):
    """base-frame linear [fwd,lat,up] + angular [roll,pitch,yaw] via finite diff."""
    v_world = (p_cur - p_prev) / dt
    v_body = X.quat_rotate_inv(q_cur, v_world)
    rel = qmul(qconj(q_prev), q_cur)        # body-frame incremental rotation
    if rel[3] < 0:
        rel = -rel
    w_body = 2.0 * rel[:3] / dt
    return v_body, w_body


def critic_obs(obs45, p_prev, q_prev, p_cur, q_cur, dt):
    v, w = body_vels(p_prev, q_prev, p_cur, q_cur, dt)
    return np.concatenate([obs45, v, w])


# ---------------- NEXUS trajectory (from JSON) ----------------
crit = Critic(POLICY)
gt = json.load(open(ROLLOUT))
dt = gt["dt"]
nbase = np.array(gt["base"], dtype=np.float64)     # (T,7) pos+quat(xyzw)
nobs = np.array(gt["obs"], dtype=np.float64)       # (T,45)
T = min(N_STEPS, len(nobs))
V_nexus = np.zeros(T)
for t in range(T):
    pp, qp = (nbase[t-1, :3], nbase[t-1, 3:7]) if t > 0 else (nbase[0, :3], nbase[0, 3:7])
    V_nexus[t] = crit.value(critic_obs(nobs[t], pp, qp, nbase[t, :3], nbase[t, 3:7], dt))

# ---------------- MuJoCo trajectory (closed loop, same start) ----------------
policy = X.Policy(POLICY)
jnames = gt["joint_names"]
joints0 = np.array(gt["joints"], dtype=np.float64)
model = X.build_mujoco_model(jnames)
model.opt.timestep = X.PHYS_DT
data = mujoco.MjData(model)
free_q = model.joint(X.FREEJOINT).qposadr[0]
hinge_q = {n: int(model.joint(n).qposadr[0]) for n in jnames}
hinge_d = {n: int(model.joint(n).dofadr[0]) for n in jnames}
act_id = {n: int(model.actuator(f"act_{n}").id) for n in jnames}
scale = np.array([X.pick_by_prefix(X.ACTION_SCALE, n) for n in jnames])

# init to nexus step-0 pose (xyzw -> wxyz for MuJoCo)
data.qpos[free_q:free_q+3] = nbase[0, :3]
data.qpos[free_q+3:free_q+7] = [nbase[0, 6], nbase[0, 3], nbase[0, 4], nbase[0, 5]]
for k, n in enumerate(jnames):
    data.qpos[hinge_q[n]] = joints0[0, k]
data.qvel[:] = 0.0
mujoco.mj_forward(model, data)

act_hist = [np.zeros(12), np.zeros(12)]
prev_p = nbase[0, :3].copy(); prev_q = nbase[0, 3:7].copy()
V_mj = np.zeros(T)
fell_at = None
for t in range(T):
    last_action = act_hist[0] if t >= 2 else np.zeros(12)
    jvel = np.array([data.qvel[hinge_d[n]] for n in jnames]) if t >= 2 else np.zeros(12)
    qj = np.array([data.qpos[hinge_q[n]] for n in jnames])
    qw_, qx_, qy_, qz_ = data.qpos[free_q+3:free_q+7]
    phase = (max(0, t-1) * X.CONTROL_DT / X.GAIT_PERIOD) % 1.0
    obs = np.zeros(45)
    obs[0:12] = last_action; obs[12:16] = CMD; obs[16:28] = qj; obs[28:40] = jvel
    obs[40:43] = X.projected_gravity((qx_, qy_, qz_, qw_))
    obs[43] = np.sin(2*np.pi*phase); obs[44] = np.cos(2*np.pi*phase)
    cur_p = data.qpos[free_q:free_q+3].copy()
    cur_q = np.array([qx_, qy_, qz_, qw_])
    V_mj[t] = crit.value(critic_obs(obs, prev_p, prev_q, cur_p, cur_q, dt))
    if fell_at is None and data.qpos[free_q+2] < X.FALL_Z:
        fell_at = t
    prev_p, prev_q = cur_p, cur_q
    action = policy.act(obs)
    act_hist = [act_hist[1], action.copy()]
    target = scale * action
    for k, n in enumerate(jnames):
        data.ctrl[act_id[n]] = target[k]
    for _ in range(X.DECIMATION):
        mujoco.mj_step(model, data)

# ---------------- divergence metric ----------------
absdiff = np.abs(V_nexus - V_mj)
scaleV = (np.abs(V_nexus).mean() + 1e-9)
onset = next((t for t in range(T) if absdiff[t] > 0.25 * (abs(V_nexus[t]) + 1.0)), None)
print(f"nexus V: mean {V_nexus.mean():.3f}  range [{V_nexus.min():.3f},{V_nexus.max():.3f}]")
print(f"mujoco V: mean {V_mj.mean():.3f}  range [{V_mj.min():.3f},{V_mj.max():.3f}]")
print(f"mean |V_nexus - V_mujoco| = {absdiff.mean():.3f} ({100*absdiff.mean()/scaleV:.0f}% of |V_nexus|)")
print(f"divergence onset (|ΔV| > 25% of |V|+1): step {onset}"
      + (f" (~{onset*dt:.2f}s)" if onset is not None else " — never (stayed matched)"))
print(f"MuJoCo first fall (torso<{X.FALL_Z}): "
      + (f"step {fell_at} (~{fell_at*dt:.2f}s)" if fell_at is not None else "no fall in window"))

# ---------------- plot ----------------
ts = np.arange(T) * dt
fig, ax = plt.subplots(2, 1, figsize=(9, 6), height_ratios=[2, 1], sharex=True)
ax[0].plot(ts, V_nexus, label="nexus  V(s)  (training engine)", color="#1f77b4", lw=2)
ax[0].plot(ts, V_mj, label="MuJoCo V(s)  (sim2sim)", color="#d62728", lw=2)
if onset is not None:
    ax[0].axvline(onset*dt, color="gray", ls="--", lw=1)
    ax[0].annotate(f"divergence ~{onset*dt:.2f}s", (onset*dt, ax[0].get_ylim()[1]),
                   fontsize=9, color="gray", va="top")
if fell_at is not None:
    ax[0].axvline(fell_at*dt, color="#d62728", ls=":", lw=1)
    ax[0].annotate(f"MuJoCo fall ~{fell_at*dt:.2f}s", (fell_at*dt, ax[0].get_ylim()[0]),
                   fontsize=9, color="#d62728", va="bottom")
ax[0].set_ylabel("critic value V(s)"); ax[0].legend(loc="best"); ax[0].grid(alpha=0.3)
ax[0].set_title("Critic value along matched rollouts — nexus vs MuJoCo (v56 NET+asymDR)")
ax[1].plot(ts, absdiff, color="#555", lw=1.5)
ax[1].set_ylabel("|ΔV|"); ax[1].set_xlabel("time (s)"); ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(OUT, dpi=110)
print(f"wrote {OUT}")

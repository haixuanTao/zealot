#!/usr/bin/env python3
"""Open-loop physics-divergence diagnostic: nexus vs MuJoCo.

The closed-loop sim2sim test mixes two things: the policy reacting to MuJoCo's
(different) state, AND the underlying physics differing. To isolate the PHYSICS,
this replays the EXACT action sequence the policy emitted in nexus (recorded in
the rollout JSON) OPEN-LOOP into MuJoCo — same start state, same per-step joint
targets, NO policy feedback. Any divergence is then pure dynamics (actuator model
+ contact + integrator), and the first DOF to split is the culprit.

Reuses the faithful MuJoCo model + constants from sim2sim_xval.

  python3 examples/biped/sim2sim_divergence.py <rollout.json> [out_prefix]
"""
import os, sys, json
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import sim2sim_xval as X

ROLLOUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/v21_stand.json"
OUTPRE  = sys.argv[2] if len(sys.argv) > 2 else "/tmp/v21_divergence"

gt = json.load(open(ROLLOUT))
jnames = gt["joint_names"]
base_n = np.array(gt["base"], dtype=np.float64)      # [T, 7] xyzw
joints_n = np.array(gt["joints"], dtype=np.float64)  # [T, 12]
actions = np.array(gt["actions"], dtype=np.float64)  # [T, 12]
T = len(base_n)
scale = np.array([X.pick_by_prefix(X.ACTION_SCALE, n) for n in jnames])


def tilt_deg(quat_xyzw):
    """Torso tilt from vertical (deg): angle of body +z from world +z."""
    x, y, z, w = quat_xyzw
    r22 = 1.0 - 2.0 * (x * x + y * y)
    return np.degrees(np.arccos(np.clip(r22, -1.0, 1.0)))


# ---- open-loop replay of the nexus actions in MuJoCo ----
model = X.build_mujoco_model(jnames)
model.opt.timestep = X.PHYS_DT
data = mujoco.MjData(model)
fq = model.joint(X.FREEJOINT).qposadr[0]
hq = {n: int(model.joint(n).qposadr[0]) for n in jnames}
aid = {n: int(model.actuator(f"act_{n}").id) for n in jnames}

data.qpos[fq:fq + 3] = base_n[0, :3]
qx, qy, qz, qw = base_n[0, 3:7]
data.qpos[fq + 3:fq + 7] = [qw, qx, qy, qz]  # XYZW -> WXYZ
for k, n in enumerate(jnames):
    data.qpos[hq[n]] = joints_n[0, k]
data.qvel[:] = 0.0
mujoco.mj_forward(model, data)

base_m = np.zeros((T, 7))
joints_m = np.zeros((T, 12))
base_m[0] = base_n[0]
joints_m[0] = joints_n[0]
for t in range(T):
    target = X.ACTION_SCALE_DEFAULT_POS + scale * actions[t]
    for k, n in enumerate(jnames):
        data.ctrl[aid[n]] = target[k]
    for _ in range(X.DECIMATION):
        mujoco.mj_step(model, data)
    p = data.qpos[fq:fq + 3].copy()
    w_, x_, y_, z_ = data.qpos[fq + 3:fq + 7]
    base_m[t] = [p[0], p[1], p[2], x_, y_, z_, w_]
    joints_m[t] = [data.qpos[hq[n]] for n in jnames]

# ---- divergence metrics ----
dt = X.CONTROL_DT
z_n = base_n[:, 2]; z_m = base_m[:, 2]
tilt_n = np.array([tilt_deg(base_n[t, 3:7]) for t in range(T)])
tilt_m = np.array([tilt_deg(base_m[t, 3:7]) for t in range(T)])
djoint = np.abs(joints_m - joints_n)  # [T,12]

# first step MuJoCo tilt exceeds 15 deg (clearly diverged from upright)
TILT_THRESH = 15.0
div_step = next((t for t in range(T) if tilt_m[t] > TILT_THRESH), None)
# rank joints by mean |Δ| over the first 25 control steps (0.5 s) — the early
# movers, before the whole-body fall swamps everything.
early = min(25, T)
joint_rank = sorted(range(12), key=lambda k: -djoint[:early, k].mean())

print(f"rollout: {ROLLOUT}   T={T} control steps ({T*dt:.2f}s)")
print(f"nexus: torso z stays {z_n.min():.3f}-{z_n.max():.3f}, max tilt {tilt_n.max():.1f} deg (STABLE)")
print(f"mujoco (open-loop, nexus actions): torso z min {z_m.min():.3f}, max tilt {tilt_m.max():.1f} deg")
if div_step is not None:
    print(f"  -> MuJoCo torso tilt crosses {TILT_THRESH} deg at step {div_step} (t={div_step*dt:.2f}s)")
else:
    print(f"  -> MuJoCo never exceeded {TILT_THRESH} deg tilt (stayed upright)")
print(f"\njoints ranked by mean |Δq| over first {early} steps (0.5s) — the EARLY movers:")
for k in joint_rank:
    print(f"  {jnames[k]:14s} mean|Δ|={djoint[:early, k].mean():.4f} rad   max|Δ|(all)={djoint[:, k].max():.4f} rad")

# ---- overlay plots ----
fig, ax = plt.subplots(3, 1, figsize=(11, 12))
ts = np.arange(T) * dt
ax[0].plot(ts, z_n, "C0-", label="nexus")
ax[0].plot(ts, z_m, "C3--", label="mujoco (open-loop, nexus actions)")
ax[0].axhline(X.FALL_Z, color="gray", ls=":", label="fall threshold")
ax[0].set_ylabel("torso height z (m)"); ax[0].legend(); ax[0].set_title("Base height")
ax[1].plot(ts, tilt_n, "C0-", label="nexus")
ax[1].plot(ts, tilt_m, "C3--", label="mujoco")
ax[1].axhline(TILT_THRESH, color="gray", ls=":")
ax[1].set_ylabel("torso tilt from vertical (deg)"); ax[1].legend(); ax[1].set_title("Base tilt")
# top-4 diverging joints
for k in joint_rank[:4]:
    l = ax[2].plot(ts, joints_n[:, k], "-", label=f"{jnames[k]} nexus")[0]
    ax[2].plot(ts, joints_m[:, k], "--", color=l.get_color(), label=f"{jnames[k]} mujoco")
ax[2].set_ylabel("joint angle (rad)"); ax[2].set_xlabel("time (s)")
ax[2].legend(fontsize=8, ncol=2); ax[2].set_title("Top-4 earliest-diverging joints (solid=nexus, dashed=mujoco)")
fig.tight_layout()
fig.savefig(f"{OUTPRE}.png", dpi=110)
print(f"\nwrote {OUTPRE}.png")

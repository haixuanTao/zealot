#!/usr/bin/env python3
"""Validate the SENSED obs against the trained policy's obs configuration.

Takes a record_lowstate.py .npz, downsamples it to the 50 Hz control rate,
rebuilds the zealot 45-dim observation frame with the deploy wrapper's exact
conventions (q - default, finite-diff joint velocity, projected gravity,
gait clock; last_action/cmd = 0 as when standing idle), then normalizes each
dimension with the CHECKPOINT's Welford statistics.

Interpretation: a config mismatch (joint order/sign, deg-vs-rad, quat
convention) shows up as |z| in the tens or hundreds on specific dims. Values
within a few sigma mean the sensed obs land inside the training distribution.

Usage: python3 check_obs_vs_policy.py recording.npz checkpoint.safetensors
"""
import sys

import numpy as np
from safetensors.numpy import load_file

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common.g1_constants import G1_MOTOR_NAMES
from common.rotation_helper import projected_gravity

CONTROL_DT = 0.02
OBS_FRAME = 45
DEFAULT_POS = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0] * 2)
GAIT_PERIOD = 0.7

BLOCKS = [
    ("last_action (zeros: idle)", 0, 12),
    ("command (zeros: stand)", 12, 16),
    ("joint_pos - default", 16, 28),
    ("joint_vel (finite-diff)", 28, 40),
    ("projected_gravity", 40, 43),
    ("gait clock sin/cos", 43, 45),
]

rec = np.load(sys.argv[1], allow_pickle=True)
sd = load_file(sys.argv[2])

mean = sd["obs_norm.mean"].astype(np.float64)
m2 = sd["obs_norm.m2"].astype(np.float64)
count = float(sd["obs_norm.count"].reshape(-1)[0])
std = np.sqrt(np.maximum(m2 / count, 1e-8))
hist = sd["actor.w_0"].shape[1] // OBS_FRAME
# per-frame stats: history stacks frames oldest->newest; use the newest slot
f_mean = mean[(hist - 1) * OBS_FRAME:]
f_std = std[(hist - 1) * OBS_FRAME:]

# downsample the recording to 50 Hz
t, q_all, quat_all = rec["t"], rec["q"], rec["quat"]
idx = np.searchsorted(t, np.arange(t[0], t[-1], CONTROL_DT))
q50 = q_all[idx][:, :12]
quat50 = quat_all[idx]

frames = []
prev_q = None
for k in range(len(idx)):
    o = np.zeros(OBS_FRAME)
    o[16:28] = q50[k] - DEFAULT_POS
    o[28:40] = 0.0 if prev_q is None else (q50[k] - prev_q) / CONTROL_DT
    o[40:43] = projected_gravity(quat50[k])
    ph = (max(0, k - 1) * CONTROL_DT / GAIT_PERIOD) % 1.0
    o[43], o[44] = np.sin(2 * np.pi * ph), np.cos(2 * np.pi * ph)
    frames.append(o)
    prev_q = q50[k].copy()
obs = np.array(frames[1:])  # drop the FD-warmup frame

z = (obs - f_mean) / f_std
zmax = np.abs(z).max(axis=0)
zmean = np.abs(z).mean(axis=0)

print(f"{len(obs)} frames @50 Hz | checkpoint history {hist} | "
      f"z = (sensed - train_mean)/train_std, newest-frame slot\n")
labels = ([f"act.{n.split('_joint')[0]}" for n in G1_MOTOR_NAMES[:12]]
          + ["cmd.vx", "cmd.vy", "cmd.yaw", "cmd.res"]
          + [f"q.{n.split('_joint')[0]}" for n in G1_MOTOR_NAMES[:12]]
          + [f"dq.{n.split('_joint')[0]}" for n in G1_MOTOR_NAMES[:12]]
          + ["grav.x", "grav.y", "grav.z", "clock.sin", "clock.cos"])

verdict_bad = []
for name, a, b in BLOCKS:
    print(f"-- {name}")
    for d in range(a, b):
        flag = ""
        if zmax[d] > 8:
            flag = "  <<< SUSPECT (config mismatch?)"
            verdict_bad.append(labels[d])
        elif zmax[d] > 4:
            flag = "  < borderline"
        print(f"   {labels[d]:18s} sensed[{obs[:, d].min():+7.2f},{obs[:, d].max():+7.2f}] "
              f"train m={f_mean[d]:+6.2f} s={f_std[d]:5.2f}  |z|max {zmax[d]:6.1f}"
              f"  |z|mean {zmean[d]:5.1f}{flag}")
print()
if verdict_bad:
    print(f"SUSPECT DIMS: {', '.join(verdict_bad)}")
else:
    print("OBS CONFIGURATION CONSISTENT: all sensed dims within training range.")

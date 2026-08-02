#!/usr/bin/env python3
"""Posture + cost-of-transport metrics from a biped_render_nexus rollout JSON.

Exists because these were being recomputed inline, ad hoc, every tracking tick
-- and one of those copies used the WRONG action scale for days.

    ACTION SCALE IS 0.5, NOT 0.25.

The base `unitree_g1()` robot spec says 0.25, but every G1 generation trains
with BIPED_ROBOT=g1_29dof_agile, which chains through `unitree_g1_agile()` and
overwrites every joint with 0.5. Using 0.25 does not merely rescale the answer:
it places the PD target at half its intended offset from the measured joint
angle, fabricating a tracking error that INFLATES torque and power. Measured on
v26 @ iter 37180 walking: CoT 1.26 at 0.25 vs 0.82 at 0.5, power 161 W vs 105 W.
The wrong value made a healthy gait look like a degrading one.

Usage:
    python3 scripts/posture_metrics.py <rollout.json> [action_scale]
"""
import json
import sys

import numpy as np

# Canonical leg order x2 (left, right): hip_pitch, hip_roll, hip_yaw, knee,
# ankle_pitch, ankle_roll.
LO = np.array([-2.5307, -0.5236, -2.7576, -0.087267, -0.87267, -0.2618] * 2)
HI = np.array([2.8798, 2.9671, 2.7576, 2.8798, 0.5236, 0.2618] * 2)
KP = np.array([100, 100, 100, 200, 40, 40] * 2)      # v19 ankle package: 40, not 20
KD = np.array([2.5, 2.5, 2.5, 5, 2.0, 2.0] * 2)      # ankle 2.0, not 0.2
DEFAULT_POS = np.array([-0.1, 0, 0, 0.3, -0.2, 0] * 2)
KNEE_LIMIT = -0.087267                                # rad, the e-stop the knee must stay off
MASS_KG = 33.34
G = 9.81
ACTION_SCALE = 0.5


def metrics(path, action_scale=ACTION_SCALE):
    d = json.load(open(path))
    base = np.array(d["base"], float)
    q = np.array(d["joints"], float)
    act = np.array(d["actions"], float)
    frames = np.array(d["frames"], float)
    dt = d["dt"]
    names = d["joint_names"]
    feet = d["feet"]

    knee = [i for i, n in enumerate(names) if "knee" in n]
    ankle = [i for i, n in enumerate(names) if "ankle_pitch" in n]
    # Settled window: drop the first third, which is the spawn transient.
    s = slice(len(base) // 3, None)

    dq = np.gradient(q, dt, axis=0)
    target = np.clip(act * action_scale + DEFAULT_POS, LO, HI)
    tau = KP * (target - q) - KD * dq
    power = np.abs(tau * dq).sum(axis=1)

    vel = np.gradient(base[:, :3], dt, axis=0)
    speed = np.linalg.norm(vel[:, :2], axis=1).mean()

    fz = frames[:, feet, 2]
    clearance = (fz - fz.min()).max(axis=0) * 1000.0

    return {
        "height": float(base[s, 2].mean()),
        "knee_deg": float(np.degrees(q[s][:, knee]).mean()),
        "knee_max_deg": float(np.degrees(q[s][:, knee]).max()),
        "knee_near_limit_pct": float((q[s][:, knee] <= KNEE_LIMIT + 0.05).mean() * 100),
        "ankle_peak_deg": float(np.degrees(q[s][:, ankle]).min()),
        "clearance_mm": [float(clearance[0]), float(clearance[1])],
        "speed_mps": float(speed),
        "mech_power_w": float(power.mean()),
        "cot": float(power.mean() / (MASS_KG * G * max(speed, 1e-6))),
        "action_scale": action_scale,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    scale = float(sys.argv[2]) if len(sys.argv) > 2 else ACTION_SCALE
    m = metrics(sys.argv[1], scale)
    print(f"h {m['height']:.3f} | knee {m['knee_deg']:.1f} (max {m['knee_max_deg']:.1f}) "
          f"| near-lim {m['knee_near_limit_pct']:.1f}% "
          f"| clear {m['clearance_mm'][0]:.0f}/{m['clearance_mm'][1]:.0f} mm "
          f"| ankle {m['ankle_peak_deg']:+.1f} | spd {m['speed_mps']:.2f} "
          f"| {m['mech_power_w']:.0f} W | CoT {m['cot']:.2f} (scale {scale})")

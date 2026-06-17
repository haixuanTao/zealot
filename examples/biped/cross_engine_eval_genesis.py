#!/usr/bin/env python3
"""Fidelity gate: replay a nexus-trained policy's joint-target trajectory through
Genesis physics (same PD gains/effort) and compare the resulting base trajectory
to nexus's. Mirrors cross_engine_eval.py (which does this for MuJoCo) so the two
reference engines are directly comparable.

The question: does Genesis track the same stream of PD targets without the robot
falling / wildly diverging? If yes, Genesis simulates the same robot faithfully
and the throughput comparison is apples-to-apples. MuJoCo's divergence
(cross_engine_eval.py) is the reference bar.

Input: /tmp/biped_rollout.json from `biped_render_nexus` (base, joints,
joint_names, dt, resets). Run: ~/genesis-venv/bin/python cross_engine_eval_genesis.py [rollout.json]
"""
import json
import sys

import numpy as np
import torch

import biped_genesis_common as C
from genesis_biped_env import GenesisBipedEnv


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "/tmp/biped_rollout.json"
    roll = json.load(open(src))
    base = np.array(roll["base"])        # (T,7) pos + quat xyzw
    joints = np.array(roll["joints"])    # (T,12)
    jnames = roll["joint_names"]
    dt = roll["dt"]
    resets = set(roll.get("resets", []))
    T = len(base)
    print(f"Loaded {T} frames at {dt:.4f}s control dt; nexus resets at {sorted(resets)}")

    # Reorder the dump's joint columns into the env's canonical dof order.
    col = [jnames.index(n) for n in C.JOINT_NAMES]

    env = GenesisBipedEnv(1, on_gpu=True)
    dev = env.device

    def t32(x):
        return torch.tensor([x], device=dev, dtype=torch.float32)

    def set_state(t):
        p = base[t, :3]
        qx, qy, qz, qw = base[t, 3:7]
        env.robot.set_pos(t32([p[0], p[1], p[2]]))
        env.robot.set_quat(t32([qw, qx, qy, qz]))  # Genesis wxyz
        env.robot.set_dofs_position(t32(list(joints[t, col])), env.dof_idx)
        env.robot.zero_all_dofs_velocity()

    set_state(0)
    gen_base = np.zeros((T, 7))
    gen_base[0] = base[0]
    fell_at = None
    for t in range(1, T):
        if (t - 1) in resets:
            set_state(t)
        env.robot.control_dofs_position(t32(list(joints[t, col])), env.dof_idx)
        for _ in range(C.DECIMATION):
            env.scene.step()
        pos = env.robot.get_pos()[0].detach().cpu().numpy()
        q = env.robot.get_quat()[0].detach().cpu().numpy()  # wxyz
        gen_base[t] = [pos[0], pos[1], pos[2], q[1], q[2], q[3], q[0]]
        if fell_at is None and pos[2] < 0.40:
            fell_at = t * dt

    print("\n=== nexus joint targets, replayed under Genesis physics ===\n")
    print(f"  duration {T * dt:.2f}s")
    if fell_at is None:
        print("  Genesis torso z stayed above 0.40 m for the full rollout.")
    else:
        print(f"  *** Genesis torso fell below 0.40 m at t = {fell_at:.2f} s ***")
    print(f"\n  final base pos:  nexus ({base[-1,0]:+.3f},{base[-1,1]:+.3f},{base[-1,2]:+.3f})"
          f"  genesis ({gen_base[-1,0]:+.3f},{gen_base[-1,1]:+.3f},{gen_base[-1,2]:+.3f})")
    xy = np.linalg.norm(base[:, :2] - gen_base[:, :2], axis=1)
    print("\n  base XY divergence (nexus vs genesis):")
    for ts in (0.5, 1.0, 2.0):
        i = int(ts / dt)
        if i < T:
            print(f"    t={ts:.1f}s: {xy[i]*100:.1f} cm")
    print(f"    t=end:  {xy[-1]*100:.1f} cm")
    print("\n  torso z over time:")
    for ts in (0.0, 0.5, 1.0, 2.0):
        i = int(ts / dt)
        if i < T:
            print(f"    t={ts:.1f}s: nexus z={base[i,2]:+.3f}  genesis z={gen_base[i,2]:+.3f}")

    out = dict(roll)
    out["base"] = gen_base.tolist()
    json.dump(out, open("/tmp/biped_rollout_genesis.json", "w"))
    print("\nSaved Genesis replay → /tmp/biped_rollout_genesis.json")


if __name__ == "__main__":
    main()

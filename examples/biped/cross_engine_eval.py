#!/usr/bin/env python3
"""Replay the joint-angle trajectory produced by the rapier-trained policy through
MuJoCo's physics, using identical PD gains, and compare the resulting base
trajectory to what rapier produced.

The point is to answer one question: does the rapier policy's stream of PD targets
also work as a controller in MuJoCo's physics? If MuJoCo can track the same joint
targets without the robot falling and the base trajectory matches, the policy is
sim-agnostic. If MuJoCo falls or wildly diverges, the policy is exploiting
rapier-specific contact behaviour.

Run: python3 examples/biped/cross_engine_eval.py
"""
import json
import math
import os
import sys
import tempfile

import mujoco
import numpy as np
import trimesh

HOME = os.path.expanduser("~")
XML = f"{HOME}/Documents/work/lerobot-humanoid-design/to_real_robot/RL_policy/robot.xml"
ASSETS = f"{HOME}/Documents/work/lerobot-humanoid-design/urdf/bipedal_plateform/urdf/assets"

# Rapier-side PD gains (per zealot-env/src/robots/lerobot_bipedal.rs JOINT_GAINS table).
GAINS = {
    "hipz": (30, 3),
    "hipx": (40, 3),
    "hipy": (60, 4),
    "knee": (60, 4),
    "ankley": (20, 1.5),
    "anklex": (20, 1.5),
}

def pick_gain(joint_name: str):
    for prefix, kd in GAINS.items():
        if joint_name.startswith(prefix):
            return kd
    raise ValueError(f"no gain for {joint_name}")


def build_model_with_actuators(joint_names):
    tmp = tempfile.mkdtemp()
    md = os.path.join(tmp, "m")
    os.makedirs(md)
    for fn in os.listdir(ASSETS):
        if fn.endswith(".stl"):
            os.symlink(os.path.join(ASSETS, fn), os.path.join(md, fn))
    trimesh.creation.box((0.02, 0.02, 0.02)).export(os.path.join(md, "torso_mesh.stl"))

    xml = open(XML).read().replace('meshdir="assets"', f'meshdir="{md}"')
    # Floor with friction 1.0 (same as rapier).
    xml = xml.replace(
        "<worldbody>",
        '<worldbody>\n  <geom name="floor" type="plane" size="5 5 .1" friction="1 .005 .0001"/>\n',
        1,
    )
    # Insert a position actuator per joint with the matching PD gains.
    actuator_xml = "<actuator>\n"
    for jn in joint_names:
        kp, kd = pick_gain(jn)
        actuator_xml += f'    <position name="act_{jn}" joint="{jn}" kp="{kp}" kv="{kd}"/>\n'
    actuator_xml += "  </actuator>\n"
    xml = xml.replace("</mujoco>", f"  {actuator_xml}</mujoco>")
    return mujoco.MjModel.from_xml_string(xml)


def main():
    roll = json.load(open("/tmp/biped_rollout.json"))
    base = np.array(roll["base"])         # (T, 7) pos + quat xyzw
    joints = np.array(roll["joints"])     # (T, 12)
    jnames = roll["joint_names"]
    dt_ctrl = roll["dt"]                  # control dt (1/50 s)
    T = len(base)
    print(f"Loaded rollout: {T} frames at {dt_ctrl:.4f}s control dt")

    model = build_model_with_actuators(jnames)
    data = mujoco.MjData(model)
    # Use the same physics dt as rapier (1/200) and decimate by 4 per control step.
    model.opt.timestep = 1.0 / 200.0
    decim = int(round(dt_ctrl / model.opt.timestep))
    print(f"MuJoCo physics dt: {model.opt.timestep:.4f}s, decimation: {decim}/control step")

    free_adr = model.joint("torso_subassembly_freejoint").qposadr[0]
    free_dofadr = model.joint("torso_subassembly_freejoint").dofadr[0]
    hinge_adr = {n: int(model.joint(n).qposadr[0]) for n in jnames}
    actuator_id = {n: int(model.actuator(f"act_{n}").id) for n in jnames}

    # Initialise MuJoCo to the rapier rollout's first frame.
    p0 = base[0, :3]
    qx, qy, qz, qw = base[0, 3:7]
    data.qpos[free_adr : free_adr + 3] = p0
    data.qpos[free_adr + 3 : free_adr + 7] = [qw, qx, qy, qz]  # MuJoCo wxyz
    for k, n in enumerate(jnames):
        data.qpos[hinge_adr[n]] = joints[0, k]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    # Drive MuJoCo with the recorded joint-angle trajectory as PD targets.
    mj_base = np.zeros((T, 7))  # (px, py, pz, qx, qy, qz, qw)
    mj_joints = np.zeros((T, 12))
    mj_base[0] = base[0]
    mj_joints[0] = joints[0]

    fell_at = None
    for t in range(1, T):
        for k, n in enumerate(jnames):
            data.ctrl[actuator_id[n]] = joints[t, k]
        # Step physics `decim` times per control step (matching rapier).
        for _ in range(decim):
            mujoco.mj_step(model, data)
        pos = data.qpos[free_adr : free_adr + 3].copy()
        quat_wxyz = data.qpos[free_adr + 3 : free_adr + 7].copy()
        mj_base[t] = [pos[0], pos[1], pos[2], quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
        for k, n in enumerate(jnames):
            mj_joints[t, k] = data.qpos[hinge_adr[n]]
        # Track when MuJoCo's torso drops below 0.40 m (= "fell" in our rapier env).
        if fell_at is None and pos[2] < 0.40:
            fell_at = t * dt_ctrl

    # Summary
    print()
    print("=== COMPARISON: rapier-trained joint targets, replayed under MuJoCo physics ===")
    print()
    print(f"  Rollout duration: {T * dt_ctrl:.2f} s")
    if fell_at is None:
        print(f"  MuJoCo torso z stayed above 0.40 m for the full rollout.")
    else:
        print(f"  *** MuJoCo torso fell below 0.40 m at t = {fell_at:.2f} s ***")
    print()
    print(f"  Final base position (m):")
    print(f"    rapier:  ({base[-1, 0]:+.3f}, {base[-1, 1]:+.3f}, {base[-1, 2]:+.3f})")
    print(f"    mujoco:  ({mj_base[-1, 0]:+.3f}, {mj_base[-1, 1]:+.3f}, {mj_base[-1, 2]:+.3f})")
    print()
    # Per-step xy divergence of the base.
    rapier_xy = base[:, :2]
    mj_xy = mj_base[:, :2]
    xy_diverge = np.linalg.norm(rapier_xy - mj_xy, axis=1)
    print(f"  Base XY divergence:")
    print(f"    at t=0.5s: {xy_diverge[int(0.5 / dt_ctrl)] * 100:.1f} cm")
    print(f"    at t=1.0s: {xy_diverge[int(1.0 / dt_ctrl)] * 100:.1f} cm")
    print(f"    at t=2.0s: {xy_diverge[int(2.0 / dt_ctrl)] * 100:.1f} cm")
    print(f"    at t=end:  {xy_diverge[-1] * 100:.1f} cm")
    print()
    # Joint-angle tracking error (MuJoCo's PD couldn't follow the commands precisely).
    joint_err = np.abs(mj_joints - joints).mean(axis=1)
    print(f"  Mean joint-tracking error (rad):")
    print(f"    at t=0.5s: {joint_err[int(0.5 / dt_ctrl)]:.3f}")
    print(f"    at t=1.0s: {joint_err[int(1.0 / dt_ctrl)]:.3f}")
    print(f"    at t=2.0s: {joint_err[int(2.0 / dt_ctrl)]:.3f}")
    print(f"    at t=end:  {joint_err[-1]:.3f}")
    print()
    # MuJoCo torso z over time
    print(f"  MuJoCo torso z (height) over time:")
    for i, ti in enumerate([0, int(0.5/dt_ctrl), int(1/dt_ctrl), int(2/dt_ctrl), int(3/dt_ctrl), T-1]):
        if ti < T:
            print(f"    t={ti*dt_ctrl:4.2f}s:  rapier z={base[ti,2]:+.3f}  mujoco z={mj_base[ti,2]:+.3f}")
    # Save a JSON of the MuJoCo replay for rendering. After the integrator
    # explodes the joint angles wrap around through thousands of radians, which
    # renders as visual gibberish. Detect the first frame whose inter-step joint
    # change is super-physical (>π rad / 20 ms) and freeze the trajectory there
    # so the side-by-side video clearly shows the collapse and then stops.
    dj = np.abs(np.diff(mj_joints, axis=0))
    bad_step = np.where(np.max(dj, axis=1) > np.pi)[0]
    last_valid = int(bad_step[0]) if len(bad_step) > 0 else T - 1
    for t in range(last_valid + 1, T):
        mj_base[t] = mj_base[last_valid]
        mj_joints[t] = mj_joints[last_valid]
    print(f"  MuJoCo trajectory frozen after t = {last_valid * dt_ctrl:.2f} s (integrator exploded)")

    out = dict(roll)
    out["base"] = mj_base.tolist()
    out["joints"] = mj_joints.tolist()
    with open("/tmp/biped_rollout_mujoco.json", "w") as f:
        json.dump(out, f)
    print()
    print("Saved MuJoCo-replayed trajectory → /tmp/biped_rollout_mujoco.json")
    print("Render with: python3 examples/biped/render_biped_mujoco.py /tmp/biped_rollout_mujoco.json /tmp/biped_mesh_mujoco.mp4 90")


if __name__ == "__main__":
    main()

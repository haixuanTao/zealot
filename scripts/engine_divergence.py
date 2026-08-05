#!/usr/bin/env python3
"""Short-horizon cross-engine divergence probe.

Initializes MuJoCo to a state sampled from a nexus rollout (base pose, joint
angles, and finite-difference velocities), applies the SAME recorded action
sequence open-loop, and reports the growth of the state difference vs the
nexus trajectory step by step.

The point is to separate two very different explanations for foreign-engine
stand failures:
  * a real per-step dynamics discrepancy  -> divergence is large and
    systematic in the FIRST control step, before instability can amplify;
  * amplification of numerical noise      -> divergence starts at ~0 and grows
    exponentially at the inverted-pendulum rate (e-folding ~0.25 s here),
    which is inherent to an unstable system and not a sim bug.

Usage: engine_divergence.py <nexus_rollout.json> [start_step] [horizon_steps]
"""
import json
import math
import os
import sys

import numpy as np
import mujoco

ROLL = sys.argv[1]
START = int(sys.argv[2]) if len(sys.argv) > 2 else 100
HORIZON = int(sys.argv[3]) if len(sys.argv) > 3 else 25

XML = ("/home/champagne/rt_build/bench-venv/lib/python3.12/site-packages/"
       "mujoco_playground/_src/locomotion/g1/xmls/scene_mjx_feetonly_flat_terrain.xml")
POLICY_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]
DEFAULT_POS = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0] * 2)
ACTION_SCALE = 0.5
CONTROL_DT = 0.02
PHYS_DT = 0.005
DECIMATION = 4
HELD = [("waist_yaw", 300., 5., 88.), ("waist", 300., 5., 50.),
        ("shoulder_pitch", 90., 2., 25.), ("shoulder_roll", 60., 1., 25.),
        ("shoulder", 20., 0.4, 25.), ("elbow", 60., 1., 25.),
        ("wrist", 4., 0.2, 25.)]
HELD_HOME = {"left_shoulder_pitch_joint": 0.2, "left_shoulder_roll_joint": 0.2,
             "left_elbow_joint": 1.28, "right_shoulder_pitch_joint": 0.2,
             "right_shoulder_roll_joint": -0.2, "right_elbow_joint": 1.28}


def leg_gains(n):
    if "knee" in n:
        return 200., 5., 139.
    if "hip" in n:
        return 100., 2.5, 88.
    if "ankle_roll" in n:
        return 20., 0.1, 50.
    return 20., 0.2, 50.


def quat_to_pitch_wxyz(w, x, y, z):
    return math.degrees(math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))))


def ang_vel_from_quats(q0_wxyz, q1_wxyz, dt):
    """Body-frame angular velocity from two orientations (small-angle)."""
    w0, x0, y0, z0 = q0_wxyz
    inv = np.array([w0, -x0, -y0, -z0])
    w1, x1, y1, z1 = q1_wxyz
    dq = np.array([
        inv[0] * w1 - inv[1] * x1 - inv[2] * y1 - inv[3] * z1,
        inv[0] * x1 + inv[1] * w1 + inv[2] * z1 - inv[3] * y1,
        inv[0] * y1 - inv[1] * z1 + inv[2] * w1 + inv[3] * x1,
        inv[0] * z1 + inv[1] * y1 - inv[2] * x1 + inv[3] * w1,
    ])
    if dq[0] < 0:
        dq = -dq
    ang = 2.0 * np.arccos(np.clip(dq[0], -1.0, 1.0))
    s = np.linalg.norm(dq[1:])
    axis = dq[1:] / s if s > 1e-9 else np.zeros(3)
    return axis * (ang / dt)


def run(perturb=None):
    """Simulate the recorded actions from the nexus state at START.
    `perturb` (mm on x) offsets the initial position. Returns the trajectory."""
    d = json.load(open(ROLL))
    base, joints, actions = d["base"], d["joints"], d["actions"]
    jnames = d["joint_names"]
    order = [jnames.index(n) for n in POLICY_JOINTS]

    model = mujoco.MjModel.from_xml_path(XML)
    model.opt.timestep = PHYS_DT
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.iterations = 100
    model.opt.ls_iterations = 50
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ACTUATION
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key)

    pol_q = np.array([model.joint(n).qposadr[0] for n in POLICY_JOINTS])
    pol_d = np.array([model.joint(n).dofadr[0] for n in POLICY_JOINTS])
    kp = np.zeros(12)
    kd = np.zeros(12)
    eff = np.zeros(12)
    for i, n in enumerate(POLICY_JOINTS):
        kp[i], kd[i], eff[i] = leg_gains(n)
        da = model.joint(n).dofadr[0]
        model.dof_damping[da] = 0.001
        model.dof_frictionloss[da] = 0.1
        model.dof_armature[da] = 0.02
    held = []
    for j in range(model.njnt):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if nm is None or model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        if nm in POLICY_JOINTS:
            continue
        for frag, hkp, hkd, heff in HELD:
            if frag in nm:
                held.append((model.jnt_qposadr[j], model.jnt_dofadr[j],
                             hkp, hkd, heff, HELD_HOME.get(nm, 0.0)))
                break
    free_q = model.jnt_qposadr[0]
    free_d = model.jnt_dofadr[0]

    # --- initialize to the nexus state at START (pose + FD velocities) ---
    b0, b1 = base[START], base[START + 1]
    q0_wxyz = np.array([b0[6], b0[3], b0[4], b0[5]])
    q1_wxyz = np.array([b1[6], b1[3], b1[4], b1[5]])
    data.qpos[free_q:free_q + 3] = b0[:3]
    if perturb:
        data.qpos[free_q] += perturb * 1e-3
    data.qpos[free_q + 3:free_q + 7] = q0_wxyz
    data.qpos[pol_q] = [joints[START][i] for i in order]
    for qa, da, _, _, _, qh in held:
        data.qpos[qa] = qh
    data.qvel[:] = 0.0
    # Prefer TRUE root velocities from the nexus dof_state dump
    # (BIPED_DUMP_VEL=1); fall back to finite differences. FD at the 50 Hz
    # control rate carries ~0.05-0.1 m/s of error, which was this probe's
    # dominant noise floor.
    dv = d.get("dof_vel")
    if dv is not None:
        w = np.array(dv[START][3:6])
        v_com = np.array(dv[START][:3])
        # nexus root DOFs are referenced to the root COM; MuJoCo's free joint
        # to the body ORIGIN. v_origin = v_com + w x (r_origin - r_com).
        pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        r_local = model.body_ipos[pelvis]
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, np.asarray(q0_wxyz))
        r_world = rot.reshape(3, 3) @ r_local
        sign = float(os.environ.get("PROBE_COM_SIGN", "-1"))
        data.qvel[free_d:free_d + 3] = v_com + sign * np.cross(w, r_world)
        data.qvel[free_d + 3:free_d + 6] = w
    else:
        data.qvel[free_d:free_d + 3] = (np.array(b1[:3]) - np.array(b0[:3])) / CONTROL_DT
        data.qvel[free_d + 3:free_d + 6] = ang_vel_from_quats(q0_wxyz, q1_wxyz, CONTROL_DT)
    jdof = d.get("joint_dof_idx")
    if dv is not None and jdof is not None:
        # TRUE joint velocities: FD at 50 Hz aliases the stand tremor by
        # >1 rad/s, which dominated this probe's noise floor.
        jv = [dv[START][jdof[jnames.index(n)]] for n in POLICY_JOINTS]
        data.qvel[pol_d] = jv
    else:
        data.qvel[pol_d] = [(joints[START + 1][i] - joints[START][i]) / CONTROL_DT
                            for i in order]
    mujoco.mj_forward(model, data)

    traj = []
    for k in range(HORIZON):
        a = np.array(actions[START + k])
        target = DEFAULT_POS + ACTION_SCALE * a
        for _ in range(DECIMATION):
            tau = np.clip(kp * (target - data.qpos[pol_q]) - kd * data.qvel[pol_d],
                          -eff, eff)
            data.qfrc_applied[:] = 0.0
            data.qfrc_applied[pol_d] = tau
            for qa, da, hkp, hkd, heff, qh in held:
                data.qfrc_applied[da] = np.clip(
                    hkp * (qh - data.qpos[qa]) - hkd * data.qvel[da], -heff, heff)
            mujoco.mj_step(model, data)
        traj.append((data.qpos[free_q:free_q + 3].copy(),
                     quat_to_pitch_wxyz(*data.qpos[free_q + 3:free_q + 7])))
    return traj


def main():
    d = json.load(open(ROLL))
    base = d["base"]
    nominal = run()
    perturbed = run(perturb=1.0)  # 1 mm initial offset — the noise floor
    print(f"init from step {START} of {ROLL}")
    print(f"{'t (s)':>6} | {'vs nexus':>18} | {'vs self +1mm':>18}")
    print(f"{'':6} | {'Δpos mm':>9} {'Δpitch°':>8} | {'Δpos mm':>9} {'Δpitch°':>8}")
    for k, (p, pitch) in enumerate(nominal):
        nb = base[START + k + 1]
        dnex = 1000.0 * float(np.linalg.norm(p - np.array(nb[:3])))
        npitch = quat_to_pitch_wxyz(nb[6], nb[3], nb[4], nb[5])
        pp, ppitch = perturbed[k]
        dself = 1000.0 * float(np.linalg.norm(p - pp))
        print(f"{(k+1)*CONTROL_DT:6.2f} | {dnex:9.2f} {pitch-npitch:+8.3f} | "
              f"{dself:9.2f} {pitch-ppitch:+8.3f}")


if __name__ == "__main__":
    main()

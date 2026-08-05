#!/usr/bin/env python3
"""UNRELIABLE — DO NOT TRUST. This tool compares nexus's obs joint_vel (a CONTROL-
STEP FINITE-DIFF / averaged velocity, biped_env_nexus.rs:915) against MuJoCo's
INSTANTANEOUS qvel, so its one-step Δq̇ "divergence" (the reported hipz 6× / ankley
4×) is a velocity-CONVENTION artifact, NOT a real dynamics gap. A direct mass-matrix
readback (2026-06-23) confirmed nexus's M matches MuJoCo within ~5% on every joint.
Kept only as a cautionary record. See memory: sim2real-gap-hipz-dynamics.

One-step motor/dynamics fidelity: nexus vs MuJoCo, from IDENTICAL state+action.

For each step t of a nexus rollout we have the nexus state (joints[t], base[t],
joint velocities from obs[t][28:40]) and the action[t]. We set MuJoCo to that
exact state, apply the same PD target, step ONE control step, and compare the
resulting joint velocity change Δq̇ to nexus's own Δq̇ (obs[t+1]−obs[t]).

Gravity is exact between the engines and free-fall was verified identical, so a
divergence here — especially in flight (no contact) — isolates the MOTOR/PD
torque response + articulated-body dynamics (M). Per-joint breakdown shows WHICH
joints (ankles? knees?) and whether it scales with the applied torque.

Usage: python3 motor_step_compare.py [rollout.json]
"""
import os, sys, json
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, mujoco
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim2sim_xval as X

ROLLOUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/v56_final_rollout.json"
gt = json.load(open(ROLLOUT))
jn = gt["joint_names"]
base = np.array(gt["base"], np.float64)     # (T,7) pos + quat(xyzw)
joints = np.array(gt["joints"], np.float64) # (T,12)
obs = np.array(gt["obs"], np.float64)       # (T,45); [28:40]=joint_vel
acts = np.array(gt["actions"], np.float64)  # (T,12)
dt = gt["dt"]
T = len(joints)

model = X.build_mujoco_model(jn); model.opt.timestep = X.PHYS_DT
data = mujoco.MjData(model)
fq = model.joint(X.FREEJOINT).qposadr[0]
fv = model.joint(X.FREEJOINT).dofadr[0]
hq = {n: int(model.joint(n).qposadr[0]) for n in jn}
hv = {n: int(model.joint(n).dofadr[0]) for n in jn}
aid = {n: int(model.actuator(f"act_{n}").id) for n in jn}
scale = np.array([X.pick_by_prefix(X.ACTION_SCALE, n) for n in jn])

per_joint_err = np.zeros(12)
per_joint_tau = np.zeros(12)
nexus_dq = np.zeros(12)
mj_dq = np.zeros(12)
cnt = 0
for t in range(1, T - 1):
    jv_t = obs[t, 28:40]            # nexus joint vel entering step t
    jv_next_nexus = obs[t + 1, 28:40]
    # base lin + ANG velocity by finite diff (world frame) so the free joint
    # carries the right momentum (hipz couples to base yaw — must not omit ang).
    vlin = (base[t, :3] - base[t - 1, :3]) / dt
    q0 = base[t - 1, 3:7]; q1 = base[t, 3:7]  # xyzw
    dq = (q1 - q0) / dt
    # ω_world = 2 * (q̇ ⊗ q⁻¹)_xyz
    x0, y0, z0, w0 = q1
    cx, cy, cz, cw = -x0, -y0, -z0, w0  # conj(q1)
    dx, dy, dz, dw = dq
    wx = 2 * (dw * cx + dx * cw + dy * cz - dz * cy)
    wy = 2 * (dw * cy - dx * cz + dy * cw + dz * cx)
    wz = 2 * (dw * cz + dx * cy - dy * cx + dz * cw)
    vang = np.array([wx, wy, wz])
    # set MuJoCo to nexus state t
    data.qpos[fq:fq + 3] = base[t, :3]
    data.qpos[fq + 3:fq + 7] = [base[t, 6], base[t, 3], base[t, 4], base[t, 5]]  # wxyz
    for k, n in enumerate(jn):
        data.qpos[hq[n]] = joints[t, k]
    data.qvel[:] = 0.0
    data.qvel[fv:fv + 3] = vlin
    data.qvel[fv + 3:fv + 6] = vang
    for k, n in enumerate(jn):
        data.qvel[hv[n]] = jv_t[k]
    mujoco.mj_forward(model, data)
    tgt = scale * acts[t]
    for k, n in enumerate(jn):
        data.ctrl[aid[n]] = tgt[k]
    tau = np.zeros(12)
    for _ in range(X.DECIMATION):
        mujoco.mj_step(model, data)
        for k, n in enumerate(jn):
            tau[k] += data.actuator_force[aid[n]]
    tau /= X.DECIMATION
    jv_next_mj = np.array([data.qvel[hv[n]] for n in jn])
    dq_nexus = jv_next_nexus - jv_t
    dq_mj = jv_next_mj - jv_t
    per_joint_err += np.abs(dq_mj - dq_nexus)
    per_joint_tau += np.abs(tau)
    nexus_dq += np.abs(dq_nexus)
    mj_dq += np.abs(dq_mj)
    cnt += 1

per_joint_err /= cnt; per_joint_tau /= cnt; nexus_dq /= cnt; mj_dq /= cnt
print(f"one-step joint Δq̇ divergence (nexus vs MuJoCo), {cnt} steps, |rad/s|:")
print(f"{'joint':<14}{'|Δq̇ err|':>10}{'nexus|Δq̇|':>11}{'mj|Δq̇|':>9}{'mj|τ|':>8}")
order = np.argsort(-per_joint_err)
for k in order:
    print(f"{jn[k]:<14}{per_joint_err[k]:>10.3f}{nexus_dq[k]:>11.3f}{mj_dq[k]:>9.3f}{per_joint_tau[k]:>8.2f}")
print(f"\nmean |Δq̇ err| over all joints: {per_joint_err.mean():.3f} rad/s/step")
print(f"(nexus mean |Δq̇| {nexus_dq.mean():.3f}, mujoco {mj_dq.mean():.3f} — relative err {per_joint_err.mean()/ (0.5*(nexus_dq.mean()+mj_dq.mean())+1e-9)*100:.0f}%)")

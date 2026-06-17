#!/usr/bin/env python3
"""Closed-loop sim-to-sim policy cross-validation harness for the zealot biped.

Runs the SAME trained policy network inside MuJoCo's physics (reactive: the 43-dim
observation is rebuilt from MuJoCo state every control step) and compares its
behaviour to the nexus (training-engine) rollout dump. This is a sim-to-real
fidelity gate: if the policy keeps the robot up for a similar amount of time in
both engines, nexus fidelity is good; a large survival gap means a sim-to-sim
mismatch the policy is exploiting.

Three phases, gated:
  PHASE 1  net parity     - obs[t] -> normalize + MLP must reproduce actions[t]
  PHASE 2  obs parity     - reconstruct obs[t] from base/joints/last_action
  PHASE 3  MuJoCo closed loop - run the policy reactively in MuJoCo physics

Phases 1 & 2 must PASS before phase 3 is trusted.

Usage:
    python3 examples/biped/sim2sim_xval.py [rollout.json] [policy.safetensors]

defaults: /tmp/biped_xval.json  /tmp/biped_policy_v7.safetensors
"""
import json
import os
import re
import sys
import tempfile

import numpy as np
from safetensors.numpy import load_file

# ----------------------------------------------------------------------------
# Config shared with the nexus env / cross_engine_eval.py
# ----------------------------------------------------------------------------
HOME = os.path.expanduser("~")
XML = f"{HOME}/Documents/work/lerobot-humanoid-design/to_real_robot/RL_policy/robot.xml"
FREEJOINT = "torso_subassembly_freejoint"

CONTROL_DT = 0.02
PHYS_DT = 1.0 / 200.0
DECIMATION = 4
# The velocity command the rollout was generated under. Default 0.4 m/s
# forward, overridable with BIPED_XVAL_CMD="vx,vy,yaw" — e.g. "0,0,0" to
# cross-validate a standing-only policy at its trained (zero-command) point.
COMMAND = np.array(
    [*(float(x) for x in os.environ.get("BIPED_XVAL_CMD", "0.4,0,0").split(",")), 0.0][:4],
    dtype=np.float64,
)
FALL_Z = 0.40

# Per-joint-family PD gains + effort caps (kp, kd, effort N.m) and action scale.
GAINS = {
    "hipz":   (30, 3,   88),
    "hipx":   (40, 3,   88),
    "hipy":   (60, 4,   88),
    "knee":   (60, 4,   88),
    "ankley": (20, 1.5, 44),
    "anklex": (20, 1.5, 44),
}
ACTION_SCALE = {
    "hipz": 0.733, "hipx": 0.55, "hipy": 0.367,
    "knee": 0.367, "anklex": 0.55, "ankley": 0.55,
}


def pick_by_prefix(table, joint_name):
    # longest matching prefix wins (so "anklex" beats nothing; all distinct here)
    best = None
    for prefix, val in table.items():
        if joint_name.startswith(prefix):
            if best is None or len(prefix) > len(best[0]):
                best = (prefix, val)
    if best is None:
        raise ValueError(f"no entry for joint {joint_name}")
    return best[1]


# ----------------------------------------------------------------------------
# Policy: obs normalizer (Welford) + ELU MLP, deterministic mean action.
# ----------------------------------------------------------------------------
class Policy:
    def __init__(self, path):
        sd = load_file(path)
        # contiguous actor layers from 0 until missing
        self.W, self.b = [], []
        l = 0
        while f"actor.w_{l}" in sd:
            self.W.append(sd[f"actor.w_{l}"].astype(np.float64))
            self.b.append(sd[f"actor.b_{l}"].astype(np.float64))
            l += 1
        self.n_layers = l
        self.mean = sd["obs_norm.mean"].astype(np.float64)
        self.m2 = sd["obs_norm.m2"].astype(np.float64)
        self.count = float(sd["obs_norm.count"].reshape(-1)[0])
        self.obs_dim = self.W[0].shape[1]
        self.act_dim = self.W[-1].shape[0]

    def normalize(self, x):
        var = np.maximum(self.m2 / self.count, 1e-8)
        xn = (x - self.mean) / np.sqrt(var)
        return np.clip(xn, -5.0, 5.0)

    def act(self, obs):
        a = self.normalize(np.asarray(obs, dtype=np.float64))
        for i in range(self.n_layers):
            z = self.W[i] @ a + self.b[i]
            if i < self.n_layers - 1:
                a = np.where(z > 0, z, np.exp(z) - 1.0)  # ELU
            else:
                a = z  # output layer linear
        return a


# ----------------------------------------------------------------------------
# Quaternion helpers. Quaternions here are XYZW.
# ----------------------------------------------------------------------------
def quat_rotate(q_xyzw, v):
    x, y, z, w = q_xyzw
    u = np.array([x, y, z])
    v = np.asarray(v, dtype=np.float64)
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def quat_rotate_inv(q_xyzw, v):
    x, y, z, w = q_xyzw
    return quat_rotate((-x, -y, -z, w), v)


def projected_gravity(q_xyzw):
    # body-frame direction of world-down [0,0,-1]
    return quat_rotate_inv(q_xyzw, np.array([0.0, 0.0, -1.0]))


# ----------------------------------------------------------------------------
# Phase 1 - net parity
# ----------------------------------------------------------------------------
def phase1(policy, gt):
    obs = np.array(gt["obs"], dtype=np.float64)
    actions = np.array(gt["actions"], dtype=np.float64)
    pred = np.array([policy.act(o) for o in obs])
    err = np.abs(pred - actions)
    mx, mn = err.max(), err.mean()
    ok = mx < 1e-4
    print("=" * 70)
    print("PHASE 1 - NET PARITY (obs -> normalize+MLP vs dumped actions)")
    print(f"  layers={policy.n_layers}  obs_dim={policy.obs_dim}  act_dim={policy.act_dim}")
    print(f"  max abs err = {mx:.3e}   mean abs err = {mn:.3e}")
    print(f"  {'PASS' if ok else 'FAIL'} (threshold 1e-4)")
    print()
    return ok


# ----------------------------------------------------------------------------
# Phase 2 - obs reconstruction parity
# ----------------------------------------------------------------------------
def phase2(gt):
    obs = np.array(gt["obs"], dtype=np.float64)
    actions = np.array(gt["actions"], dtype=np.float64)
    base = np.array(gt["base"], dtype=np.float64)      # px,py,pz, qx,qy,qz,qw
    joints = np.array(gt["joints"], dtype=np.float64)
    resets = set(gt.get("resets", []))
    T = len(obs)

    # Reset semantics recovered empirically from the dump (the brief's
    # "last_action[t]=actions[t-1]" was wrong - it is actually a 2-step lag with
    # a 2-step warmup, and command is zeroed for one step post-reset):
    #   * a reset logged at step s makes step s+1 the FRESH post-reset step.
    #   * last_action[t] = actions[t-2] (LAG 2), but = 0 during a 2-step warmup
    #     (the fresh step and the one after it: {fresh, fresh+1}).
    #   * joint_vel uses the same 2-step warmup; it is 0 on {fresh, fresh+1},
    #     and (joints[t]-joints[t-1])/dt otherwise.
    #   * command[t] = 0 ONLY on the fresh post-reset step (reset+1); it is
    #     [0.4,0,0,0] everywhere else INCLUDING step 0.
    fresh = set([0] + [s + 1 for s in resets if s + 1 < T])
    warmup = set()
    for f in fresh:
        warmup.add(f)
        if f + 1 < T:
            warmup.add(f + 1)
    # command is zeroed only on the post-reset fresh step, not on step 0
    cmd_zero = set(s + 1 for s in resets if s + 1 < T)

    recon = np.zeros_like(obs)
    for t in range(T):
        last_action = np.zeros(12) if (t in warmup) else actions[t - 2]
        jvel = np.zeros(12) if (t in warmup) else (joints[t] - joints[t - 1]) / CONTROL_DT
        cmd = np.zeros(4) if (t in cmd_zero) else COMMAND
        recon[t, 0:12] = last_action
        recon[t, 12:16] = cmd
        recon[t, 16:28] = joints[t]          # joint_pos_rel (default 0)
        recon[t, 28:40] = jvel
        recon[t, 40:43] = projected_gravity(base[t, 3:7])

    blocks = {
        "last_action": (0, 12),
        "command":     (12, 16),
        "joint_pos_rel": (16, 28),
        "joint_vel":   (28, 40),
        "proj_grav":   (40, 43),
    }
    # joint_vel: the 2-step warmup is reconstructed exactly (=0), so nothing to
    # exclude; any residual is pure finite-diff vs nexus-stored vel.
    exclude = np.zeros(T, dtype=bool)

    print("=" * 70)
    print("PHASE 2 - OBS RECONSTRUCTION PARITY (per-block max abs err)")
    results = {}
    for name, (a, b) in blocks.items():
        diff = np.abs(recon[:, a:b] - obs[:, a:b])
        if name == "joint_vel":
            mx = diff[~exclude].max()
            note = " (nexus finite-diff vs reconstruction)"
        else:
            mx = diff.max()
            note = ""
        results[name] = mx
        print(f"  {name:14s} max err = {mx:.3e}{note}")

    ok = (results["proj_grav"] < 1e-3 and results["joint_pos_rel"] < 1e-3
          and results["last_action"] < 1e-3 and results["command"] < 1e-3)
    # joint_vel sanity (loose)
    jv_ok = results["joint_vel"] < 1e-1
    print(f"  {'PASS' if ok and jv_ok else 'FAIL'} "
          f"(proj_grav & joint_pos_rel < 1e-3; joint_vel finite-diff < 1e-1)")
    print()
    return ok and jv_ok


# ----------------------------------------------------------------------------
# Phase 3 - MuJoCo closed-loop rollout
# ----------------------------------------------------------------------------
# The full robot model + its assets live here (robot.xml + assets/ with the 54
# STL meshes + the canonical sim_scene_safe.xml). The model ships a canonical
# MuJoCo eval scene (sim_scene_safe.xml: implicitfast integrator, Newton solver,
# dt=0.005, PD gains that match zealot) which we build on directly so the cross-
# val runs the INTENDED MuJoCo physics — real mesh foot colliders with their
# tuned contact params (condim/solref/solimp/friction/priority), real joint
# damping+frictionloss+armature, explicit per-link inertials.
MJCF_DIR = os.environ.get("BIPED_MJCF_DIR", f"{HOME}/tmp_eval/mjcf")
ACT_JOINT_ORDER = ["hipz", "hipx", "hipy", "knee", "ankley", "anklex"]


def build_mujoco_model(joint_names):
    """Compile the canonical MuJoCo scene faithfully.

    Uses the shipped sim_scene_safe.xml options (implicitfast / Newton / dt 0.005)
    and the REAL robot.xml (mesh foot colliders + contact params + joint
    damping/frictionloss/armature + visual meshes), with two corrections for a
    faithful match to the nexus-trained controller:
      * inertiafromgeom="false" -> use the explicit per-link <inertial> tags
        (the same masses/inertias nexus uses), NOT geom-recomputed inertia
        (the shipped scene's inertiafromgeom="true" inflates mass ~12.7->20.3 kg).
      * actuator forcerange = the per-joint effort caps (88/88/88/88/44/44 N.m)
        nexus applies via set_motor_max_force, which sim_scene_safe omits.
    PD gains (kp/kv) already match zealot; gear=1; ctrl = joint target (rad).
    Returns a model that has BOTH collisions and visual meshes, so the same model
    serves physics and rendering.
    """
    import mujoco
    scene = ['<mujoco model="xval">',
             '  <compiler inertiafromgeom="false"/>',
             '  <option timestep="0.005" gravity="0 0 -9.81" integrator="implicitfast"'
             ' solver="Newton" iterations="10" ls_iterations="20" cone="pyramidal"'
             ' impratio="1"/>',
             '  <include file="robot.xml"/>',
             '  <worldbody>',
             '    <light pos="0 0 5"/>',
             '    <geom name="floor" type="plane" size="10 10 0.1"'
             ' rgba="0.3 0.34 0.42 1" contype="1" conaffinity="1"/>',
             '  </worldbody>',
             '  <actuator>']
    for side in ("left", "right"):
        for j in ACT_JOINT_ORDER:
            kp, kv, eff = GAINS[j]
            scene.append(f'    <position name="act_{j}_{side}" joint="{j}_{side}"'
                         f' kp="{kp}" kv="{kv}" gear="1" forcerange="-{eff} {eff}"/>')
    scene.append("  </actuator>")
    scene.append("</mujoco>")
    path = os.path.join(MJCF_DIR, "_xval_scene.xml")
    with open(path, "w") as f:
        f.write("\n".join(scene))
    return mujoco.MjModel.from_xml_path(path)


def phase3(policy, gt):
    import mujoco
    jnames = gt["joint_names"]
    base = np.array(gt["base"], dtype=np.float64)
    joints = np.array(gt["joints"], dtype=np.float64)
    obs_gt = np.array(gt["obs"], dtype=np.float64)
    n_steps = len(base)

    model = build_mujoco_model(jnames)
    model.opt.timestep = PHYS_DT
    data = mujoco.MjData(model)

    free_qadr = model.joint(FREEJOINT).qposadr[0]
    free_dofadr = model.joint(FREEJOINT).dofadr[0]
    hinge_qadr = {n: int(model.joint(n).qposadr[0]) for n in jnames}
    hinge_dofadr = {n: int(model.joint(n).dofadr[0]) for n in jnames}
    act_id = {n: int(model.actuator(f"act_{n}").id) for n in jnames}
    scale = np.array([pick_by_prefix(ACTION_SCALE, n) for n in jnames])

    # init to nexus step-0 pose
    data.qpos[free_qadr:free_qadr + 3] = base[0, :3]
    qx, qy, qz, qw = base[0, 3:7]
    data.qpos[free_qadr + 3:free_qadr + 7] = [qw, qx, qy, qz]  # XYZW -> WXYZ
    for k, n in enumerate(jnames):
        data.qpos[hinge_qadr[n]] = joints[0, k]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    def read_obs(last_action, command, joint_vel):
        qjoint = np.array([data.qpos[hinge_qadr[n]] for n in jnames])
        qw_, qx_, qy_, qz_ = data.qpos[free_qadr + 3:free_qadr + 7]  # WXYZ
        pg = projected_gravity((qx_, qy_, qz_, qw_))               # XYZW
        o = np.zeros(43)
        o[0:12] = last_action
        o[12:16] = command
        o[16:28] = qjoint
        o[28:40] = joint_vel
        o[40:43] = pg
        return o

    print("=" * 70)
    print("PHASE 3 - MuJoCo CLOSED-LOOP ROLLOUT (policy reactive on MuJoCo state)")

    # sanity: projected_gravity at step 0 should match the dumped obs[0][40:43]
    obs0 = read_obs(np.zeros(12), COMMAND, np.zeros(12))
    print(f"  proj_grav sanity @step0  mujoco={np.array2string(obs0[40:43], precision=4)}"
          f"  dump={np.array2string(obs_gt[0, 40:43], precision=4)}")
    print(f"    abs diff = {np.abs(obs0[40:43] - obs_gt[0, 40:43]).max():.3e}")

    # Replicate the nexus episode-start conventions recovered in phase 2 for the
    # FIRST control step (this is a fresh start, like a post-reset):
    #   step 0  : command=[0.4,..], last_action=0, joint_vel=0  (warmup)
    #   step 1  : command=[0.4,..], last_action=0, joint_vel=0  (warmup)
    #   step>=2 : last_action = action from 2 steps ago (LAG 2), joint_vel live.
    # joint_vel is read as MuJoCo data.qvel per the brief.
    act_hist = [np.zeros(12), np.zeros(12)]  # [t-2, t-1]
    z_trace, xy_trace = [], []
    fell_step = None
    for t in range(n_steps):
        last_action = act_hist[0] if t >= 2 else np.zeros(12)
        if t < 2:
            joint_vel = np.zeros(12)
        else:
            joint_vel = np.array([data.qvel[hinge_dofadr[n]] for n in jnames])
        command = COMMAND  # never reset mid-run (we stop at fall)
        obs = read_obs(last_action, command, joint_vel)
        action = policy.act(obs)
        act_hist = [act_hist[1], action.copy()]
        target = ACTION_SCALE_DEFAULT_POS + scale * action
        for k, n in enumerate(jnames):
            data.ctrl[act_id[n]] = target[k]
        for _ in range(DECIMATION):
            mujoco.mj_step(model, data)
        z = float(data.qpos[free_qadr + 2])
        xy = data.qpos[free_qadr:free_qadr + 2].copy()
        z_trace.append(z)
        xy_trace.append(xy)
        if fell_step is None and z < FALL_Z:
            fell_step = t
            # stop once fallen: integrating a collapsed robot for hundreds more
            # steps just produces meaningless (often exploding) trajectory data.
            break

    z_trace = np.array(z_trace)
    xy_trace = np.array(xy_trace)
    survived = fell_step if fell_step is not None else n_steps
    print()
    print(f"  MuJoCo survived {survived} control steps "
          f"({survived * CONTROL_DT:.2f}s) before torso z < {FALL_Z}")
    if fell_step is None:
        print(f"    (stayed up for the full {n_steps}-step rollout)")
    print()
    print("  torso z trace:")
    for ts in (0.0, 0.2, 0.5, 1.0, 2.0):
        idx = int(round(ts / CONTROL_DT))
        if idx < len(z_trace):
            print(f"    t={ts:4.2f}s (step {idx:3d})  z = {z_trace[idx]:+.4f}")
    disp = float(np.linalg.norm(xy_trace[min(survived, n_steps) - 1] - base[0, :2]))
    print(f"\n  final base xy displacement = {disp * 100:.1f} cm "
          f"(at step {min(survived, n_steps) - 1})")
    print()

    # nexus comparison
    resets = sorted(gt.get("resets", []))
    if resets:
        gaps = np.diff([0] + resets)
        nexus_mean_gap = float(np.mean(gaps))
    else:
        nexus_mean_gap = float(n_steps)
    print("  --- NEXUS vs MuJoCo verdict ---")
    print(f"  nexus v7: reset (fell) every ~{nexus_mean_gap:.0f} control steps "
          f"(~{nexus_mean_gap * CONTROL_DT:.2f}s)")
    print(f"  mujoco  : survived {survived} control steps "
          f"(~{survived * CONTROL_DT:.2f}s)")
    ratio = survived / nexus_mean_gap if nexus_mean_gap > 0 else float("inf")
    if 0.5 <= ratio <= 2.0:
        verdict = ("SIMILAR -- policy fails on a comparable timescale in both "
                   "engines; nexus sim-to-sim fidelity looks GOOD for this policy.")
    elif ratio > 2.0:
        verdict = ("MuJoCo survives MUCH LONGER -- nexus is harder/more unstable "
                   "than MuJoCo (sim-to-sim GAP: policy over-fits nexus contact).")
    else:
        verdict = ("MuJoCo falls MUCH FASTER -- policy exploits nexus-specific "
                   "dynamics that MuJoCo does not reproduce (sim-to-sim GAP).")
    print(f"  ratio (mujoco/nexus) = {ratio:.2f}")
    print(f"  VERDICT: {verdict}")
    print()
    return survived


# default joint target position (0 for all joints)
ACTION_SCALE_DEFAULT_POS = 0.0


def main():
    rollout_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/biped_xval.json"
    policy_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/biped_policy_v7.safetensors"
    print(f"rollout : {rollout_path}")
    print(f"policy  : {policy_path}\n")

    gt = json.load(open(rollout_path))
    policy = Policy(policy_path)

    p1 = phase1(policy, gt)
    if not p1:
        print("PHASE 1 FAILED - net/normalizer math is wrong. Aborting.")
        sys.exit(1)
    p2 = phase2(gt)
    if not p2:
        print("PHASE 2 FAILED - obs reconstruction wrong. Aborting before MuJoCo.")
        sys.exit(1)
    phase3(policy, gt)


if __name__ == "__main__":
    main()

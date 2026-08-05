#!/usr/bin/env python3
"""Closed-loop MuJoCo sim2sim rollout of a zealot G1 policy → mp4.

Runs the CURRENT zealot G1 training config (`BIPED_ROBOT=g1_29dof_agile`,
`BIPED_OBS_HISTORY=5`) against MuJoCo as the cross-engine validator: the policy
network is loaded from the trainer's safetensors, the 45-dim observation frame
is rebuilt from MuJoCo state each control step with the trainer's exact
conventions, stacked 5-deep (oldest→newest, reset-replicated), normalized with
the checkpoint's Welford stats, and fed through the ELU MLP. Actions drive an
explicit torque-level PD at 200 Hz with WBC-AGILE's actuator parametrization —
the same one zealot bakes into the nexus solver (`unitree_g1_29dof_agile`).

Model: mujoco_playground's official G1 29-DOF (same joint names / ranges /
home pose as zealot's spec; flat-terrain scene, feet-only collision). Model
actuators are DISABLED (mjDSBL_ACTUATION) — all torques come from the explicit
PD below, policy legs + AGILE holding gains for waist/arms.

Trainer conventions replicated (source: biped_env_nexus.rs / velocity_flat.rs):
  - obs45 = [last_action(12), cmd(4), q−default(12), qdot_fd(12), proj_grav(3),
    sin 2πφ, cos 2πφ];  φ(t) = max(0, t−1)·control_dt / 0.7 (gait period)
  - last_action is LAG-2 (obs at decision t carries the action from t−2;
    zeros for the first two steps of each episode)
  - joint_vel is the FINITE-DIFF (q_t − q_{t−1})/control_dt (zeros at step 0)
  - PD target = clamp(default + 0.5·action, joint range)
  - control 50 Hz (decimation 4 × 1/200 s physics)

On a fall (pelvis z < 0.45 m or tilt > 70°) the episode re-initialises at the
home keyframe with a small yaw jitter, so the clip strings several attempts
together.

Usage:
  python3 sim2sim_g1_mujoco.py [policy.safetensors] [out.mp4] [seconds]
defaults: /tmp/biped_policy_gpu.safetensors.best  /tmp/g1_sim2sim_mujoco.mp4  30
Env: BIPED_CMD="vx,vy,yaw" (default 0.4,0,0)
"""
import os
import subprocess
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from safetensors.numpy import load_file

POLICY = sys.argv[1] if len(sys.argv) > 1 else "/tmp/biped_policy_gpu.safetensors.best"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/g1_sim2sim_mujoco.mp4"
SECONDS = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0

MODEL_XML = os.path.expanduser(
    "~/miniforge3/envs/mjx/lib/python3.12/site-packages/mujoco_playground"
    "/_src/locomotion/g1/xmls/scene_mjx_feetonly_flat_terrain.xml"
)

PHYS_DT = 1.0 / 200.0
DECIMATION = 4
CONTROL_DT = PHYS_DT * DECIMATION
GAIT_PERIOD = 0.7
HIST = 5
FALL_Z = 0.45
TILT_LIMIT = np.deg2rad(70.0)
W, H = 960, 540

_cmd = os.environ.get("BIPED_CMD", "0.4,0,0").split(",")
CMD = np.array([float(_cmd[0]), float(_cmd[1]), float(_cmd[2]), 0.0])

# Canonical policy joint order + zealot g1_29dof_agile actuator table
# (zealot-env/src/robots/unitree_g1.rs — unitree_g1_agile leg deltas applied).
POLICY_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]
DEFAULT_POS = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0] * 2)
ACTION_SCALE = 0.5


def leg_gains(name):
    if "knee" in name:
        return 200.0, 5.0, 139.0
    if "hip" in name:
        return 100.0, 2.5, 88.0
    if "ankle_roll" in name:
        return 20.0, 0.1, 50.0
    return 20.0, 0.2, 50.0  # ankle_pitch


# Upper-body holding gains: zealot's held_joints table (first fragment wins).
HELD = [
    ("waist_yaw", 300.0, 5.0, 88.0),
    ("waist", 300.0, 5.0, 50.0),
    ("shoulder_pitch", 90.0, 2.0, 25.0),
    ("shoulder_roll", 60.0, 1.0, 25.0),
    ("shoulder", 20.0, 0.4, 25.0),  # shoulder_yaw
    ("elbow", 60.0, 1.0, 25.0),
    ("wrist", 4.0, 0.2, 25.0),
]


class Policy:
    """Welford obs normalizer + ELU MLP (deterministic mean action)."""

    def __init__(self, path):
        sd = load_file(path)
        self.W, self.b = [], []
        l = 0
        while f"actor.w_{l}" in sd:
            self.W.append(sd[f"actor.w_{l}"].astype(np.float64))
            self.b.append(sd[f"actor.b_{l}"].astype(np.float64))
            l += 1
        self.mean = sd["obs_norm.mean"].astype(np.float64)
        self.m2 = sd["obs_norm.m2"].astype(np.float64)
        self.count = float(sd["obs_norm.count"].reshape(-1)[0])
        self.obs_dim = self.W[0].shape[1]
        self.act_dim = self.W[-1].shape[0]

    def act(self, obs):
        var = np.maximum(self.m2 / self.count, 1e-8)
        a = np.clip((obs - self.mean) / np.sqrt(var), -5.0, 5.0)
        for i, (w, bb) in enumerate(zip(self.W, self.b)):
            z = w @ a + bb
            a = z if i == len(self.W) - 1 else np.where(z > 0, z, np.expm1(z))
        return a


def projected_gravity(q_wxyz):
    """World-down [0,0,-1] in the base frame. MuJoCo quats are WXYZ."""
    w, x, y, z = q_wxyz
    u = np.array([-x, -y, -z])  # conjugate → world-to-body rotation
    v = np.array([0.0, 0.0, -1.0])
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def main():
    policy = Policy(POLICY)
    assert policy.act_dim == 12, policy.act_dim
    assert policy.obs_dim == 45 * HIST, policy.obs_dim

    model = mujoco.MjModel.from_xml_path(MODEL_XML)
    model.opt.timestep = PHYS_DT
    # The scene ships MJX-tuned solver options (Euler, 3 Newton iterations) —
    # far too loose for classic MuJoCo with a stiff explicit PD at 5 ms
    # (QACC blows up in ~0.2 s). Restore classic-strength settings.
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.iterations = 100
    model.opt.ls_iterations = 50
    # All torques come from the explicit PD below.
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ACTUATION
    model.vis.global_.offwidth, model.vis.global_.offheight = W, H
    data = mujoco.MjData(model)

    key_home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")

    # Joint bookkeeping: policy legs + PD-held upper body.
    pol_q = np.array([model.joint(n).qposadr[0] for n in POLICY_JOINTS])
    pol_d = np.array([model.joint(n).dofadr[0] for n in POLICY_JOINTS])
    pol_kp = np.zeros(12)
    pol_kd = np.zeros(12)
    pol_eff = np.zeros(12)
    pol_rng = np.zeros((12, 2))
    for i, n in enumerate(POLICY_JOINTS):
        pol_kp[i], pol_kd[i], pol_eff[i] = leg_gains(n)
        pol_rng[i] = model.joint(n).range
    # Align the policy joints' PASSIVE dynamics with the trained spec. The
    # playground MJCF bakes actuator-level damping into the joints (hip/knee
    # damping 2.0 ≈ AGILE's kd — their pipeline uses kp-only position
    # actuators) plus frictionloss 0.1 and per-CAD armature. zealot trains
    # with passive damping 0.001 / frictionloss 0 / armature 0.02 and applies
    # kd in the PD — leaving the model values in place DOUBLE-damps every
    # joint, which spares quasi-static balance but mistimes the swing leg
    # (measured: the walking policy face-planted at 0.8 s every attempt;
    # the standing-era policy was unaffected).
    for i, n in enumerate(POLICY_JOINTS):
        da = model.joint(n).dofadr[0]
        model.dof_damping[da] = 0.001
        model.dof_frictionloss[da] = 0.0
        model.dof_armature[da] = 0.02

    held = []  # (qposadr, dofadr, kp, kd, eff, q_home)
    home_qpos = model.key_qpos[key_home]
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name is None or model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        if name in POLICY_JOINTS:
            continue
        for frag, kp, kd, eff in HELD:
            if frag in name:
                qa, da = model.jnt_qposadr[j], model.jnt_dofadr[j]
                held.append((qa, da, kp, kd, eff, home_qpos[qa]))
                break
    print(f"policy joints: 12, held joints: {len(held)}")

    free_q = model.jnt_qposadr[0]  # floating base is joint 0

    renderer = mujoco.Renderer(model, height=H, width=W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = model.body("pelvis").id if mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "pelvis") >= 0 else 1
    cam.distance, cam.elevation, cam.azimuth = 2.6, -15.0, 135.0

    rng = np.random.default_rng(7)

    def reset():
        mujoco.mj_resetDataKeyframe(model, data, key_home)
        data.qpos[pol_q] = DEFAULT_POS
        # small yaw jitter so attempts differ
        yaw = rng.uniform(-0.3, 0.3)
        data.qpos[free_q + 3:free_q + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

    n_ctrl = int(SECONDS / CONTROL_DT)
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(int(1 / CONTROL_DT)), "-i", "-",
         "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", OUT],
        stdin=subprocess.PIPE,
    )

    reset()
    ep_t = 0          # control steps since episode start
    act_hist = [np.zeros(12), np.zeros(12)]  # [t-2, t-1]
    prev_q = data.qpos[pol_q].copy()
    frames_hist = None  # 5-frame obs history (list, oldest→newest)
    attempts, survived = 1, []
    dist0 = data.qpos[free_q:free_q + 2].copy()

    for t in range(n_ctrl):
        q = data.qpos[pol_q].copy()
        quat = data.qpos[free_q + 3:free_q + 7].copy()

        # --- 45-dim obs frame, trainer conventions ---
        o = np.zeros(45)
        o[0:12] = act_hist[0] if ep_t >= 2 else 0.0     # LAG-2 last_action
        o[12:16] = CMD
        o[16:28] = q - DEFAULT_POS
        o[28:40] = 0.0 if ep_t == 0 else (q - prev_q) / CONTROL_DT
        o[40:43] = projected_gravity(quat)
        ph = (max(0, ep_t - 1) * CONTROL_DT / GAIT_PERIOD) % 1.0
        o[43], o[44] = np.sin(2 * np.pi * ph), np.cos(2 * np.pi * ph)

        if frames_hist is None:
            frames_hist = [o.copy() for _ in range(HIST)]  # reset-replicate
        else:
            frames_hist = frames_hist[1:] + [o.copy()]
        action = policy.act(np.concatenate(frames_hist))

        target = np.clip(DEFAULT_POS + ACTION_SCALE * action, pol_rng[:, 0], pol_rng[:, 1])

        # --- 4 physics substeps with explicit torque PD @200 Hz ---
        for _ in range(DECIMATION):
            tau_leg = pol_kp * (target - data.qpos[pol_q]) - pol_kd * data.qvel[pol_d]
            data.qfrc_applied[:] = 0.0
            data.qfrc_applied[pol_d] = np.clip(tau_leg, -pol_eff, pol_eff)
            for qa, da, kp, kd, eff, qh in held:
                data.qfrc_applied[da] = np.clip(
                    kp * (qh - data.qpos[qa]) - kd * data.qvel[da], -eff, eff)
            mujoco.mj_step(model, data)

        prev_q = q
        act_hist = [act_hist[1], action.copy()]
        ep_t += 1

        renderer.update_scene(data, camera=cam)
        ff.stdin.write(renderer.render().tobytes())

        # --- fall / timeout check ---
        z = data.qpos[free_q + 2]
        up = projected_gravity(data.qpos[free_q + 3:free_q + 7])
        tilt = np.arccos(np.clip(-up[2], -1.0, 1.0))
        if z < FALL_Z or tilt > TILT_LIMIT or ep_t >= int(20.0 / CONTROL_DT):
            d = np.linalg.norm(data.qpos[free_q:free_q + 2] - dist0)
            why = "timeout" if ep_t >= int(20.0 / CONTROL_DT) else "fell"
            survived.append((ep_t * CONTROL_DT, d, why))
            print(f"attempt {attempts}: {why} after {ep_t * CONTROL_DT:.1f}s, "
                  f"traveled {d:.2f} m")
            reset()
            ep_t = 0
            act_hist = [np.zeros(12), np.zeros(12)]
            prev_q = data.qpos[pol_q].copy()
            frames_hist = None
            dist0 = data.qpos[free_q:free_q + 2].copy()
            attempts += 1

    ff.stdin.close()
    ff.wait()
    if survived:
        ts = [s for s, _, _ in survived]
        print(f"\n{len(survived)} completed attempts; mean survival "
              f"{np.mean(ts):.1f}s, best {max(ts):.1f}s")
    print(f"video → {OUT}")


if __name__ == "__main__":
    main()

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
import json
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

def _model_xml() -> str:
    override = os.environ.get("S2S_MODEL_XML")
    if override:
        return override
    import mujoco_playground
    return os.path.join(
        os.path.dirname(mujoco_playground.__file__),
        "_src/locomotion/g1/xmls/scene_mjx_feetonly_flat_terrain.xml",
    )

MODEL_XML = _model_xml()

# Step-scene height: the box geometry must MATCH the cue, or the eval measures
# a policy responding to a lie. (It did: S2S_STEP_H initially only fed the cue
# while the XML kept a hardcoded 0.10 m riser, so a 5/10/15 cm "sweep" was one
# 10 cm step three times.) Rewrite the geom to the requested height in a
# tempfile whenever S2S_STEP_H is set on a step scene.
if "S2S_STEP_H" in os.environ and "step_scene" in MODEL_XML:
    import re as _re, tempfile as _tf
    _h = float(os.environ["S2S_STEP_H"])
    _src = open(MODEL_XML).read()
    _src = _re.sub(
        r'(<geom name="step"[^>]*?pos="[-0-9.]+ [-0-9.]+ )[-0-9.]+(" size="[-0-9.]+ [-0-9.]+ )[-0-9.]+(")',
        lambda m: f"{m.group(1)}{_h/2:.4f}{m.group(2)}{_h/2:.4f}{m.group(3)}",
        _src, count=1)
    _fd, _tmp = _tf.mkstemp(suffix=".xml", dir=os.path.dirname(MODEL_XML))
    with os.fdopen(_fd, "w") as _f:
        _f.write(_src)
    MODEL_XML = _tmp

PHYS_DT = 1.0 / 200.0
DECIMATION = 4
CONTROL_DT = PHYS_DT * DECIMATION
# Gait clock — derived from the command, exact mirror of the trainer env
# (biped_env_nexus.rs): phase FROZEN below 0.1 m/s commanded; else the period
# lerps 0.8 s (at 0.1 m/s) -> 0.55 s (at 0.5 m/s). Deterministic from command
# history; no estimator. Policies trained before 2026-07-28 used a fixed
# free-running 0.7 s clock instead (S2S_LEGACY_CLOCK=1 restores it).
GAIT_PERIOD_SLOW = 0.8
GAIT_PERIOD_FAST = 0.55
LEGACY_CLOCK = os.environ.get("S2S_LEGACY_CLOCK") == "1"

def gait_period_for(cmd_speed: float) -> float:
    if LEGACY_CLOCK:
        return 0.7
    # Cap must match the trainer's BIPED_GAIT_SPEED_CAP: the gait clock is part
    # of the OBSERVATION, so a mismatch feeds the policy a phase it never saw.
    _cap = float(os.environ.get("S2S_GAIT_SPEED_CAP", "0.5"))
    t = (min(abs(cmd_speed), _cap) - 0.1) / 0.4
    # Floor matches the trainer's GAIT_PERIOD_MIN: above a 0.5 cap the linear
    # lerp would extrapolate to a sprint cadence the robot has never walked.
    return max(0.40, GAIT_PERIOD_SLOW + (GAIT_PERIOD_FAST - GAIT_PERIOD_SLOW) * max(t, 0.0))
HIST = 5
# Frame width: 45 for v21-and-earlier, 48 once the gyro was added (v22+).
# Sniffed from the checkpoint so one harness runs both.
FALL_Z = 0.45
TILT_LIMIT = np.deg2rad(70.0)
W, H = 960, 540

_cmd = os.environ.get("BIPED_CMD", "0.4,0,0").split(",")
CMD = np.array([float(_cmd[0]), float(_cmd[1]), float(_cmd[2]), 0.0])

# S2S_CMD_SWITCH="t:vx,vy,wz" — swap the command mid-episode at t seconds.
# Built for the walk->stand transient study: the gait clock follows the
# training contract (freezes at its CURRENT phase when speed drops below
# 0.1), so the policy sees exactly what a real command drop produces.
_sw = os.environ.get("S2S_CMD_SWITCH")
CMD_SWITCH = None
if _sw:
    _t, _v = _sw.split(":")
    _vv = [float(x) for x in _v.split(",")]
    CMD_SWITCH = (float(_t), np.array([_vv[0], _vv[1], _vv[2], 0.0]))

# S2S_TRANS_JSON=<path>: per-control-step transient/impact log —
# [t, |base angvel|, base z, F_left, F_right] with per-foot normal contact
# force in N (summed over that foot's active contacts). Slam metric: peak
# touchdown force in units of body weight; settle metric: |angvel| decay
# after a command drop.
TRANS_JSON = os.environ.get("S2S_TRANS_JSON")

# Canonical policy joint order + zealot g1_29dof_agile actuator table
# (zealot-env/src/robots/unitree_g1.rs — unitree_g1_agile leg deltas applied).
POLICY_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]
DEFAULT_POS = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0] * 2)

# Open-loop action replay (S2S_ACTION_REPLAY=<nexus rollout json>): ignore the
# policy and apply a recorded action sequence blind. Cross-engine physics
# comparison with the controller removed from the loop.
# Matched-state init (S2S_INIT_FROM="<nexus rollout json>:<step>"): start from
# a state sampled from a nexus rollout — pose, joint angles, and TRUE
# velocities (dof_vel/joint_dof_idx, written with BIPED_DUMP_VEL=1). Combined
# with S2S_ACTION_REPLAY this makes the cross-engine divergence probe.
_INIT = None
if os.environ.get("S2S_INIT_FROM"):
    _p, _s = os.environ["S2S_INIT_FROM"].rsplit(":", 1)
    with open(_p) as _f:
        _d = json.load(_f)
    _k = int(_s)
    _jdof = _d.get("joint_dof_idx")
    _dv = _d.get("dof_vel")
    _jn = _d["joint_names"]
    _INIT = {
        "pos": _d["base"][_k][:3],
        "quat_wxyz": [_d["base"][_k][6]] + _d["base"][_k][3:6],
        "joints": [_d["joints"][_k][_jn.index(n)] for n in POLICY_JOINTS],
        "lin": _dv[_k][:3] if _dv else [0.0] * 3,
        "ang": _dv[_k][3:6] if _dv else [0.0] * 3,
        "jvel": ([_dv[_k][_jdof[_jn.index(n)]] for n in POLICY_JOINTS]
                 if _dv and _jdof else [0.0] * 12),
        "step": _k,
    }

_REPLAY = None
if os.environ.get("S2S_ACTION_REPLAY"):
    with open(os.environ["S2S_ACTION_REPLAY"]) as _f:
        _REPLAY = np.array(json.load(_f)["actions"], dtype=np.float64)

# PD target = clamp(default_pos + ACTION_SCALE * action, joint range).
# 0.5 is the AGILE value every G1 generation through v26 trained at -- note the
# BASE unitree_g1() spec says 0.25, but BIPED_ROBOT=g1_29dof_agile chains
# through unitree_g1_agile(), which overwrites every joint with 0.5. v27+ can
# train at 0.25 (BIPED_ACTION_SCALE); running a checkpoint at the wrong scale
# halves or doubles every commanded joint excursion, so this MUST match the
# scale the checkpoint was trained with.
ACTION_SCALE = float(os.environ.get("S2S_ACTION_SCALE", "0.5"))

# Upper-body pose the harness HOLDS. Two valid choices, and they are NOT the
# same robot:
#   "home" (default) = the playground keyframe / deploy pose, shoulder 0.2,
#       elbow 1.28 -- which CANCELS the 73.2 deg elbow bend baked into the MJCF
#       body hierarchy, leaving the arms straight DOWN along the body. This is
#       what the LeRobot controller and the real robot use.
#   "rest" = every held joint at 0, i.e. the model's rest pose, which for this
#       robot is the forearms held OUT IN FRONT (73.2 deg of bend). This is what
#       zealot training held for v19-v27, before held_home was added.
# Set S2S_HELD_POSE=rest to evaluate a checkpoint under the geometry it was
# actually trained with.
HELD_POSE = os.environ.get("S2S_HELD_POSE", "home")

# Step-crossing eval (g1_step_scene.xml): the scene has ONE box step whose
# front edge crosses the path at S2S_STEP_EDGE_X with height S2S_STEP_H.
# When set, the harness computes the 5-slot step cue from this KNOWN geometry
# -- the same numbers the RealSense detector will produce -- so the eval
# isolates step EXECUTION from perception. Metrics gain crossed/cross_time.
STEP_EDGE_X = float(os.environ["S2S_STEP_EDGE_X"]) if "S2S_STEP_EDGE_X" in os.environ else None
STEP_H = float(os.environ.get("S2S_STEP_H", "0.10"))



KP_SCALE = float(os.environ.get("S2S_KP_SCALE", "1"))
KD_SCALE = float(os.environ.get("S2S_KD_SCALE", "1"))


def leg_gains(name):
    kp, kd, eff = _leg_gains_raw(name)
    return kp * KP_SCALE, kd * KD_SCALE, eff


# Ankle gains are checkpoint-dependent: AGILE-era policies (pre-v19) trained
# at 20/0.2 (roll 0.1); the v19+ "ankle package" trains at the unitree_rl_gym
# deploy pair 40/2.0. Default to the deploy pair (current checkpoints);
# S2S_ANKLE_KP / S2S_ANKLE_KD restore the old actuator for old checkpoints.
ANKLE_KP = float(os.environ.get("S2S_ANKLE_KP", "40"))
ANKLE_KD = float(os.environ.get("S2S_ANKLE_KD", "2.0"))


def _leg_gains_raw(name):
    if "knee" in name:
        return 200.0, 5.0, 139.0
    if "hip" in name:
        return 100.0, 2.5, 88.0
    if "ankle" in name:
        return ANKLE_KP, ANKLE_KD, 50.0


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
    assert policy.obs_dim % HIST == 0, policy.obs_dim
    frame = policy.obs_dim // HIST
    assert frame in (45, 48, 53), f"unexpected obs frame width {frame}"
    print(f"obs frame {frame} ({'no' if frame == 45 else 'with'} gyro{', step cue' if frame >= 53 else ''})")

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
        # frictionloss 0.1 = the trained spec (adopted from this very model's
        # baked-in value once training gained a frictionloss path).
        model.dof_frictionloss[da] = 0.1
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
                q_hold = 0.0 if HELD_POSE == "rest" else home_qpos[qa]
                held.append((qa, da, kp, kd, eff, q_hold))
                break
    print(f"policy joints: 12, held joints: {len(held)}")

    free_q = model.jnt_qposadr[0]  # floating base is joint 0
    free_d = model.jnt_dofadr[0]   # its first velocity DOF (linear x)

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
        # Start the held joints AT the pose we hold them at, or every reset
        # begins with the arms being dragged from the keyframe to the target.
        for _qa, _da, _kp, _kd, _eff, _qh in held:
            data.qpos[_qa] = _qh
        # small yaw jitter so attempts differ
        yaw = rng.uniform(-0.3, 0.3)
        data.qpos[free_q + 3:free_q + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        data.qvel[:] = 0.0
        if _INIT is not None:
            data.qpos[free_q:free_q + 3] = _INIT["pos"]
            data.qpos[free_q + 3:free_q + 7] = _INIT["quat_wxyz"]
            data.qpos[pol_q] = _INIT["joints"]
            data.qvel[free_d:free_d + 3] = _INIT["lin"]
            data.qvel[free_d + 3:free_d + 6] = _INIT["ang"]
            data.qvel[pol_d] = _INIT["jvel"]
        mujoco.mj_forward(model, data)

    n_ctrl = int(SECONDS / CONTROL_DT)
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(int(1 / CONTROL_DT)), "-i", "-",
         "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", OUT],
        stdin=subprocess.PIPE,
    )

    reset()
    frozen_phase = 0.0
    ep_t = 0          # control steps since episode start
    act_hist = [np.zeros(12), np.zeros(12)]  # [t-2, t-1]
    vel_track = []
    pitch_track = []
    traj = []  # per-step (body_vx, body_vy, yaw_rate) for direction-aware metrics
    foot_track = [] if os.environ.get("S2S_FOOT_JSON") else None
    _foot_gids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
                  for n in ("left_foot", "right_foot")]
    prev_q = data.qpos[pol_q].copy()
    frames_hist = None  # 5-frame obs history (list, oldest→newest)
    attempts, survived = 1, []
    dist0 = data.qpos[free_q:free_q + 2].copy()

    trans_track = [] if TRANS_JSON else None

    for t in range(n_ctrl):
        if CMD_SWITCH is not None and t * CONTROL_DT >= CMD_SWITCH[0]:
            CMD[:] = CMD_SWITCH[1]
        q = data.qpos[pol_q].copy()
        quat = data.qpos[free_q + 3:free_q + 7].copy()

        # --- 45-dim obs frame, trainer conventions ---
        o = np.zeros(frame)
        o[0:12] = act_hist[0] if ep_t >= 2 else 0.0     # LAG-2 last_action
        o[12:16] = CMD
        o[16:28] = q - DEFAULT_POS
        o[28:40] = 0.0 if ep_t == 0 else (q - prev_q) / CONTROL_DT
        o[40:43] = projected_gravity(quat)
        cmd_speed = (CMD[0]**2 + CMD[1]**2 + CMD[2]**2) ** 0.5  # incl. yaw, like the env
        if cmd_speed < 0.1 and not LEGACY_CLOCK:
            ph = frozen_phase
        else:
            ph = (max(0, ep_t - 1) * CONTROL_DT / gait_period_for(cmd_speed)) % 1.0
            frozen_phase = ph
        o[43], o[44] = np.sin(2 * np.pi * ph), np.cos(2 * np.pi * ph)
        if frame >= 48:
            # Base angular velocity, BODY frame -- which is exactly what
            # MuJoCo's free joint already stores in qvel[3:6]. (Verified: set
            # qvel[3:6]=[1,0,0] on a body yawed 90 deg and mj_objectVelocity
            # reports [0,1,0] in world.) An earlier version rotated this by
            # -yaw as if it were world-frame, corrupting roll/pitch and making
            # every gyro-era policy spin in MuJoCo while nexus showed ~0.
            o[45:48] = data.qvel[free_d + 3:free_d + 6]

        if frame >= 53:
            if STEP_EDGE_X is None:
                # Flat scene, no step: cue invalid, whole block zero -- the
                # same pattern the trainer emits when nothing is detected.
                o[48:53] = 0.0
            else:
                # Known step geometry: edge plane x = STEP_EDGE_X, normal +x
                # (world). Distance along the HEADING to the plane; height
                # signed by which side we are on; edge normal rotated into
                # the body frame. Matches the trainer oracle's conventions.
                px = data.qpos[free_q]
                on_top = px > STEP_EDGE_X
                dist_plane = STEP_EDGE_X - px
                _w, _x, _y, _z = quat  # MuJoCo order: w x y z
                hy = 2.0 * (_w * _z + _x * _y)
                hx = 1.0 - 2.0 * (_y * _y + _z * _z)
                byaw = np.arctan2(hy, hx)
                c, sn = np.cos(byaw), np.sin(byaw)
                # distance along heading (ray to plane); if walking away or
                # parallel, report invalid like the trainer's ranged probe.
                denom = c if abs(c) > 1e-3 else None
                d_head = dist_plane / denom if denom else None
                if d_head is not None and 0.0 < d_head < 1.5 and not on_top:
                    o[48] = min(d_head, 1.5)
                    o[49] = STEP_H
                    # edge normal +x world -> body frame
                    o[50] = -sn      # edge_sin
                    o[51] = c        # edge_cos
                    o[52] = 1.0
                elif d_head is not None and on_top and 0.0 < -d_head < 1.5 and c < 0.0:
                    # facing back off the platform: step DOWN
                    o[48] = min(-d_head, 1.5)
                    o[49] = -STEP_H
                    o[50] = sn
                    o[51] = -c
                    o[52] = 1.0
                else:
                    o[48:53] = 0.0
        if frames_hist is None:
            frames_hist = [o.copy() for _ in range(HIST)]  # reset-replicate
        else:
            frames_hist = frames_hist[1:] + [o.copy()]
        action = policy.act(np.concatenate(frames_hist))
        if _REPLAY is not None:
            _off = _INIT["step"] if _INIT is not None else 0
            action = _REPLAY[min(_off + t, len(_REPLAY) - 1)]

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

        # --- heading-frame velocity tracking (direction-aware, unlike traveled_m) ---
        qw, qx, qy, qz = data.qpos[free_q + 3:free_q + 7]
        base_yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        free_d = model.jnt_dofadr[0]
        wvx, wvy = data.qvel[free_d], data.qvel[free_d + 1]
        vel_track.append((np.cos(base_yaw) * wvx + np.sin(base_yaw) * wvy,
                          -np.sin(base_yaw) * wvx + np.cos(base_yaw) * wvy,
                          data.qvel[free_d + 5]))
        pitch_track.append(np.degrees(np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1))))

        base_p_t = data.qpos[free_q:free_q + 3].copy()
        traj.append([float(v) for v in base_p_t] + [float(v) for v in data.qpos[free_q + 3:free_q + 7]])
        # Optional foot-posture log (S2S_FOOT_JSON): world height of the toe and
        # heel corners of each foot box, so "toe-walking vs flat" is measured
        # from geometry rather than inferred from the ankle joint angle (which
        # is relative to the shin and says nothing on its own).
        if foot_track is not None:
            row = []
            for gid in _foot_gids:
                R = data.geom_xmat[gid].reshape(3, 3)
                c = data.geom_xpos[gid]
                hx, _, hz = model.geom_size[gid]
                toe = c + R @ np.array([hx, 0.0, -hz])
                heel = c + R @ np.array([-hx, 0.0, -hz])
                row += [float(toe[2]), float(heel[2])]
            foot_track.append(row)
        if trans_track is not None:
            # Per-foot normal contact force, N: sum |f_normal| over the
            # active contacts touching each foot geom (mj_contactForce's
            # first component is the contact-frame normal force).
            f6 = np.zeros(6)
            f_lr = [0.0, 0.0]
            for ci in range(data.ncon):
                con = data.contact[ci]
                for side, gid in enumerate(_foot_gids):
                    if con.geom1 == gid or con.geom2 == gid:
                        mujoco.mj_contactForce(model, data, ci, f6)
                        f_lr[side] += abs(float(f6[0]))
            trans_track.append([
                round(t * CONTROL_DT, 4),
                float(np.linalg.norm(data.qvel[free_d + 3:free_d + 6])),
                float(data.qpos[free_q + 2]),
                f_lr[0], f_lr[1],
            ])
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
    # Structured metrics for benchmark harnesses (S2S_METRICS_JSON=<path>).
    tpath = os.environ.get("S2S_TRAJ_JSON")
    if tpath:
        with open(tpath, "w") as f:
            json.dump(traj, f)
    fpath = os.environ.get("S2S_FOOT_JSON")
    if fpath and foot_track is not None:
        with open(fpath, "w") as f:
            json.dump({"cols": ["L_toe_z", "L_heel_z", "R_toe_z", "R_heel_z"],
                       "half_len": float(model.geom_size[_foot_gids[0]][0]),
                       "rows": foot_track}, f)
    if TRANS_JSON and trans_track is not None:
        with open(TRANS_JSON, "w") as f:
            json.dump({"cols": ["t", "angvel", "base_z", "F_left", "F_right"],
                       "rows": trans_track}, f)
    mpath = os.environ.get("S2S_METRICS_JSON")
    if mpath:
        import json as _json
        final_d = float(np.linalg.norm(data.qpos[free_q:free_q + 2] - dist0))
        final_t = ep_t * CONTROL_DT
        eps = [{"seconds": float(t), "traveled_m": float(d), "end": why}
               for (t, d, why) in survived]
        eps.append({"seconds": final_t, "traveled_m": final_d, "end": "clip_end"})
        vt = np.array(vel_track) if vel_track else np.zeros((1, 3))
        with open(mpath, "w") as f:
            _json.dump({
                "engine": "mujoco",
                "model_xml": MODEL_XML,
                "policy": POLICY,
                "command": [float(c) for c in CMD],
                "clip_seconds": SECONDS,
                "falls": sum(1 for e in eps if e["end"] == "fell"),
                "mean_body_vel": [float(v) for v in vt.mean(axis=0)],
                "settled_pitch_deg": float(np.mean(pitch_track[len(pitch_track)//2:])) if pitch_track else 0.0,
                "episodes": eps,
            "crossed": (float(data.qpos[free_q]) > (STEP_EDGE_X + 0.3)) if STEP_EDGE_X is not None else None,
            "final_x": float(data.qpos[free_q]),
            }, f, indent=1)
    if survived:
        ts = [s for s, _, _ in survived]
        print(f"\n{len(survived)} completed attempts; mean survival "
              f"{np.mean(ts):.1f}s, best {max(ts):.1f}s")
    print(f"video → {OUT}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Genesis sim2sim harness for the zealot G1 policy — third-engine transfer
check mirroring examples/biped/sim2sim_g1_mujoco.py: same 45-dim obs frames
(trainer conventions, stacked 5-deep, checkpoint Welford normalization), same
explicit torque PD at 200 Hz with decimation 4, same fall/timeout episode
logic and direction-aware metrics.

Usage: sim2sim_g1_genesis.py <policy.safetensors> <out.mp4> [seconds]
Env: BIPED_CMD="vx,vy,yaw" (default 0.4,0,0), S2S_METRICS_JSON=<path>.

Model: the mujoco_playground G1 feetonly MJCF, preprocessed for genesis —
the <contact> pair list is stripped (its floor pair references the scene
file) and the two foot boxes get explicit contype/conaffinity + friction 0.6
so genesis reproduces the feet-only contact set against a plane. Passive
dynamics are aligned to the trained spec at the XML level (damping 0.001,
armature 0.02, frictionloss 0.1 on the 12 policy joints), matching the
overrides the MuJoCo harness applies through the model API.
"""
import json
import os
import struct
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

POLICY = sys.argv[1]
OUT = sys.argv[2]
SECONDS = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
CMD = np.zeros(4)
CMD[:3] = [float(x) for x in os.environ.get("BIPED_CMD", "0.4,0,0").split(",")]

PHYS_DT = 0.005
DECIMATION = 4
CONTROL_DT = PHYS_DT * DECIMATION
GAIT_PERIOD_SLOW = 0.8
GAIT_PERIOD_FAST = 0.55
LEGACY_CLOCK = os.environ.get("S2S_LEGACY_CLOCK") == "1"
HIST = 5
FALL_Z = 0.45
TILT_LIMIT = np.deg2rad(70.0)
W, H = 960, 540

PLAYGROUND_XML = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "")
MODEL_XML = os.environ.get(
    "S2S_MODEL_XML",
    "/home/champagne/rt_build/bench-venv/lib/python3.12/site-packages/"
    "mujoco_playground/_src/locomotion/g1/xmls/g1_mjx_feetonly.xml")

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

ACTION_SCALE = 0.5


KP_SCALE = float(os.environ.get("S2S_KP_SCALE", "1"))
KD_SCALE = float(os.environ.get("S2S_KD_SCALE", "1"))


def leg_gains(name):
    kp, kd, eff = _leg_gains_raw(name)
    return kp * KP_SCALE, kd * KD_SCALE, eff


def _leg_gains_raw(name):
    if "knee" in name:
        return 200.0, 5.0, 139.0
    if "hip" in name:
        return 100.0, 2.5, 88.0
    if "ankle_roll" in name:
        return 20.0, 0.1, 50.0
    return 20.0, 0.2, 50.0  # ankle_pitch


HELD = [
    ("waist_yaw", 300.0, 5.0, 88.0),
    ("waist", 300.0, 5.0, 50.0),
    ("shoulder_pitch", 90.0, 2.0, 25.0),
    ("shoulder_roll", 60.0, 1.0, 25.0),
    ("shoulder", 20.0, 0.4, 25.0),
    ("elbow", 60.0, 1.0, 25.0),
    ("wrist", 4.0, 0.2, 25.0),
]

# Playground "home" keyframe pose for the held joints (arms bent, not
# hanging) — the MuJoCo harness holds these via the keyframe; zeros put the
# arms in a visibly wrong straight-down pose and shift the CoM.
HELD_HOME = {
    "left_shoulder_pitch_joint": 0.2,
    "left_shoulder_roll_joint": 0.2,
    "left_elbow_joint": 1.28,
    "right_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_elbow_joint": 1.28,
}


def gait_period_for(cmd_speed):
    if LEGACY_CLOCK:
        return 0.7
    t = (min(abs(cmd_speed), 0.5) - 0.1) / 0.4
    return GAIT_PERIOD_SLOW + (GAIT_PERIOD_FAST - GAIT_PERIOD_SLOW) * max(t, 0.0)


def projected_gravity(q_wxyz):
    w, x, y, z = q_wxyz
    u = np.array([-x, -y, -z])
    v = np.array([0.0, 0.0, -1.0])
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def load_safetensors(path):
    """Minimal pure-numpy safetensors reader (nyx-venv has no pip)."""
    out = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        blob = f.read()
    dt = {"F32": np.float32, "F64": np.float64, "I64": np.int64, "U32": np.uint32}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        b0, b1 = info["data_offsets"]
        arr = np.frombuffer(blob[b0:b1], dtype=dt[info["dtype"]])
        out[name] = arr.reshape(info["shape"])
    return out


class Policy:
    def __init__(self, path):
        sd = load_safetensors(path)
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


def preprocess_xml(path):
    """Genesis-ready copy: no <contact> (floor pair is scene-relative), feet
    get real contype/conaffinity, policy joints get the trained passive
    dynamics."""
    tree = ET.parse(path)
    root = tree.getroot()
    for parent in list(root.iter()):
        for c in list(parent):
            if c.tag in ("contact", "sensor", "keyframe"):
                parent.remove(c)
    for g in root.iter("geom"):
        if g.get("name") in ("left_foot", "right_foot"):
            g.set("contype", "1")
            g.set("conaffinity", "1")
            g.set("friction", "0.6")
    for j in root.iter("joint"):
        if j.get("name") in POLICY_JOINTS:
            j.set("damping", "0.001")
            j.set("armature", "0.02")
            j.set("frictionloss", "0.1")
    fd, tmp = tempfile.mkstemp(suffix=".xml", dir=os.path.dirname(path))
    with os.fdopen(fd, "w") as f:
        f.write(ET.tostring(root, encoding="unicode"))
    return tmp


def npy(x):
    """Genesis tensors → numpy."""
    try:
        return x.detach().cpu().numpy().astype(np.float64).reshape(-1)
    except AttributeError:
        return np.asarray(x, dtype=np.float64).reshape(-1)


def main():
    import genesis as gs

    policy = Policy(POLICY)
    assert policy.act_dim == 12 and policy.obs_dim == 45 * HIST

    gs.init(backend=gs.cpu, logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=PHYS_DT, substeps=1),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    tmp_xml = preprocess_xml(MODEL_XML)
    try:
        robot = scene.add_entity(gs.morphs.MJCF(file=tmp_xml))
        cam = scene.add_camera(res=(W, H), pos=(2.0, -1.8, 1.4),
                               lookat=(0.0, 0.0, 0.8), fov=40, GUI=False)
        scene.build()
    finally:
        os.unlink(tmp_xml)

    pol_idx = []
    for n in POLICY_JOINTS:
        j = robot.get_joint(n)
        idx = j.dofs_idx_local if hasattr(j, "dofs_idx_local") else [j.dof_idx_local]
        pol_idx.append(int(np.atleast_1d(np.asarray(idx))[0]))
    pol_kp = np.zeros(12)
    pol_kd = np.zeros(12)
    pol_eff = np.zeros(12)
    for i, n in enumerate(POLICY_JOINTS):
        pol_kp[i], pol_kd[i], pol_eff[i] = leg_gains(n)

    held = []  # (dof_idx, kp, kd, eff, q_home from the playground keyframe)
    for j in robot.joints:
        n = j.name
        if n in POLICY_JOINTS or j.n_dofs != 1:
            continue
        for frag, kp, kd, eff in HELD:
            if frag in n:
                idx = j.dofs_idx_local if hasattr(j, "dofs_idx_local") else [j.dof_idx_local]
                held.append((int(np.atleast_1d(np.asarray(idx))[0]), kp, kd, eff,
                             HELD_HOME.get(n, 0.0)))
                break
    print(f"policy joints: 12, held joints: {len(held)}")

    pelvis = robot.get_link("pelvis")
    all_idx = pol_idx + [h[0] for h in held]

    rng = np.random.default_rng(7)

    def reset():
        yaw = rng.uniform(-0.3, 0.3)
        robot.set_pos(np.array([0.0, 0.0, 0.79]))
        robot.set_quat(np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]))
        if _INIT is not None:
            robot.set_pos(np.array(_INIT["pos"]))
            robot.set_quat(np.array(_INIT["quat_wxyz"]))
            robot.set_dofs_position(np.array(_INIT["joints"]), pol_idx)
            robot.set_dofs_velocity(np.array(_INIT["lin"] + _INIT["ang"]),
                                    list(range(6)))
            robot.set_dofs_velocity(np.array(_INIT["jvel"]), pol_idx)
            return
        robot.set_dofs_position(DEFAULT_POS, pol_idx)
        if held:
            robot.set_dofs_position(np.array([h[4] for h in held]),
                                    [h[0] for h in held])
        robot.zero_all_dofs_velocity()

    n_ctrl = int(SECONDS / CONTROL_DT)
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt",
         "rgb24", "-s", f"{W}x{H}", "-r", str(int(1 / CONTROL_DT)), "-i", "-",
         "-c:v", "libx264", "-preset", "medium", "-crf", "20",
         "-pix_fmt", "yuv420p", OUT],
        stdin=subprocess.PIPE,
    )

    reset()
    frozen_phase = 0.0
    ep_t = 0
    act_hist = [np.zeros(12), np.zeros(12)]
    vel_track = []
    pitch_track = []
    traj = []
    prev_q = npy(robot.get_dofs_position(pol_idx))
    frames_hist = None
    attempts, survived = 1, []
    p0 = npy(pelvis.get_pos())[:2].copy()

    for t in range(n_ctrl):
        q = npy(robot.get_dofs_position(pol_idx))
        quat = npy(pelvis.get_quat())  # genesis: wxyz

        o = np.zeros(45)
        o[0:12] = act_hist[0] if ep_t >= 2 else 0.0
        o[12:16] = CMD
        o[16:28] = q - DEFAULT_POS
        o[28:40] = 0.0 if ep_t == 0 else (q - prev_q) / CONTROL_DT
        o[40:43] = projected_gravity(quat)
        cmd_speed = (CMD[0] ** 2 + CMD[1] ** 2 + CMD[2] ** 2) ** 0.5
        if cmd_speed < 0.1 and not LEGACY_CLOCK:
            ph = frozen_phase
        else:
            ph = (max(0, ep_t - 1) * CONTROL_DT / gait_period_for(cmd_speed)) % 1.0
            frozen_phase = ph
        o[43], o[44] = np.sin(2 * np.pi * ph), np.cos(2 * np.pi * ph)

        if frames_hist is None:
            frames_hist = [o.copy() for _ in range(HIST)]
        else:
            frames_hist = frames_hist[1:] + [o.copy()]
        action = policy.act(np.concatenate(frames_hist))
        if _REPLAY is not None:
            _off = _INIT["step"] if _INIT is not None else 0
            action = _REPLAY[min(_off + t, len(_REPLAY) - 1)]

        target = DEFAULT_POS + ACTION_SCALE * action
        for _ in range(DECIMATION):
            qq = npy(robot.get_dofs_position(all_idx))
            qd = npy(robot.get_dofs_velocity(all_idx))
            tau_leg = np.clip(pol_kp * (target - qq[:12]) - pol_kd * qd[:12],
                              -pol_eff, pol_eff)
            tau_held = np.array([
                np.clip(kp * (qh - qq[12 + k]) - kd * qd[12 + k], -eff, eff)
                for k, (_, kp, kd, eff, qh) in enumerate(held)])
            tau = np.concatenate([tau_leg, tau_held]) if held else tau_leg
            robot.control_dofs_force(tau, all_idx)
            scene.step()

        prev_q = q
        act_hist = [act_hist[1], action.copy()]
        ep_t += 1

        base_p = npy(pelvis.get_pos())
        quat = npy(pelvis.get_quat())
        w, x, y, z = quat
        base_yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        lv = npy(pelvis.get_vel())
        av = npy(pelvis.get_ang())
        vel_track.append((np.cos(base_yaw) * lv[0] + np.sin(base_yaw) * lv[1],
                          -np.sin(base_yaw) * lv[0] + np.cos(base_yaw) * lv[1],
                          av[2]))
        pitch_track.append(np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))))

        cam.set_pose(pos=(base_p[0] + 2.0, base_p[1] - 1.8, 1.4),
                     lookat=(base_p[0], base_p[1], 0.8))
        rgb = cam.render()[0]
        ff.stdin.write(np.ascontiguousarray(rgb[..., :3]).astype(np.uint8).tobytes())

        traj.append([float(v) for v in base_p[:3]] + [float(v) for v in quat[:4]])
        up = projected_gravity(quat)
        tilt = np.arccos(np.clip(-up[2], -1.0, 1.0))
        if base_p[2] < FALL_Z or tilt > TILT_LIMIT or ep_t >= int(20.0 / CONTROL_DT):
            d = float(np.linalg.norm(base_p[:2] - p0))
            why = "timeout" if ep_t >= int(20.0 / CONTROL_DT) else "fell"
            survived.append((ep_t * CONTROL_DT, d, why))
            print(f"attempt {attempts}: {why} after {ep_t * CONTROL_DT:.1f}s, "
                  f"traveled {d:.2f} m")
            reset()
            ep_t = 0
            act_hist = [np.zeros(12), np.zeros(12)]
            prev_q = npy(robot.get_dofs_position(pol_idx))
            frames_hist = None
            p0 = npy(pelvis.get_pos())[:2].copy()
            attempts += 1

    ff.stdin.close()
    ff.wait()
    print(f"video → {OUT}")
    tpath = os.environ.get("S2S_TRAJ_JSON")
    if tpath:
        with open(tpath, "w") as f:
            json.dump(traj, f)
    mpath = os.environ.get("S2S_METRICS_JSON")
    if mpath:
        base_p = npy(pelvis.get_pos())
        final_d = float(np.linalg.norm(base_p[:2] - p0))
        eps = [{"seconds": float(t), "traveled_m": float(d), "end": why}
               for (t, d, why) in survived]
        eps.append({"seconds": ep_t * CONTROL_DT, "traveled_m": final_d,
                    "end": "clip_end"})
        vt = np.array(vel_track) if vel_track else np.zeros((1, 3))
        with open(mpath, "w") as f:
            json.dump({
                "engine": "genesis",
                "model_xml": MODEL_XML,
                "policy": POLICY,
                "command": [float(c) for c in CMD],
                "clip_seconds": SECONDS,
                "falls": sum(1 for e in eps if e["end"] == "fell"),
                "mean_body_vel": [float(v) for v in vt.mean(axis=0)],
                "settled_pitch_deg": float(np.mean(pitch_track[len(pitch_track)//2:])) if pitch_track else 0.0,
                "episodes": eps,
            }, f, indent=1)


if __name__ == "__main__":
    main()

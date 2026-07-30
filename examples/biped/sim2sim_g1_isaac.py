#!/usr/bin/env python3
"""Sim2sim: run a zealot G1 policy (45x5 obs, safetensors MLP) in Isaac Sim.

Third-engine cross-validation next to sim2sim_g1_mujoco.py — identical policy
loading, obs convention, PD gains and fall rules; only the physics backend
differs (PhysX via isaacsim.core, external torque-PD like the MuJoCo harness).
Isaac scaffold (app boot, URDF import, zero-drive + manual efforts, camera)
follows sim2sim's examples/lerobot_legs/isaac_zealot.py.

Usage (needs the Isaac venv):
  OMNI_KIT_ACCEPT_EULA=YES ~/rt_build/isaac-venv/bin/python \
      examples/biped/sim2sim_g1_isaac.py [policy.safetensors] [out.mp4] [seconds]
Env: BIPED_CMD="vx,vy,yaw" (default 0.4,0,0)
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

POLICY = sys.argv[1] if len(sys.argv) > 1 else "/tmp/biped_policy_gpu.safetensors.best"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/g1_sim2sim_isaac.mp4"
SECONDS = float(sys.argv[3]) if len(sys.argv) > 3 else 20.0

# MuJoCo-flattened playground feetonly MJCF (defaults resolved, contact/
# sensor/keyframe stripped, trained passive dynamics baked into the leg
# joints) → USD via isaacsim.asset.importer.mjcf, cached under /tmp.
# The old URDF route is dead on the Isaac Sim 6.0.1 install (no
# URDFParseAndImportFile command, unitree_ros checkout gone).
FLAT_XML = ("/home/champagne/rt_build/bench-venv/lib/python3.12/site-packages/"
            "mujoco_playground/_src/locomotion/g1/xmls/_g1_isaac_flat.xml")
USD_DIR = "/tmp/g1_isaac_usd"
USD_FILE = os.path.join(USD_DIR, "_g1_isaac_flat", "_g1_isaac_flat.usda")

PHYS_DT = 1.0 / 200.0
DECIMATION = 4
CONTROL_DT = PHYS_DT * DECIMATION
GAIT_PERIOD_SLOW = 0.8
GAIT_PERIOD_FAST = 0.55
LEGACY_CLOCK = os.environ.get("S2S_LEGACY_CLOCK") == "1"


def gait_period_for(cmd_speed: float) -> float:
    if LEGACY_CLOCK:
        return 0.7
    t = (min(abs(cmd_speed), 0.5) - 0.1) / 0.4
    return GAIT_PERIOD_SLOW + (GAIT_PERIOD_FAST - GAIT_PERIOD_SLOW) * max(t, 0.0)


HIST = 5
FALL_Z = 0.45
TILT_LIMIT = np.deg2rad(70.0)
W, H = 960, 540
RENDER_EVERY = 2  # capture at 25 fps

_cmd = os.environ.get("BIPED_CMD", "0.4,0,0").split(",")
CMD = np.array([float(_cmd[0]), float(_cmd[1]), float(_cmd[2]), 0.0])

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
# hanging straight) — matches the MuJoCo harness's keyframe hold.
HELD_HOME = {
    "left_shoulder_pitch_joint": 0.2,
    "left_shoulder_roll_joint": 0.2,
    "left_elbow_joint": 1.28,
    "right_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_elbow_joint": 1.28,
}


def load_safetensors(path):
    """Pure-numpy safetensors reader (the Isaac venv has no pip)."""
    dt = {"F32": np.float32, "F64": np.float64, "I64": np.int64, "U8": np.uint8,
          "F16": np.float16, "I32": np.int32, "U32": np.uint32}
    raw = Path(path).read_bytes()
    (hlen,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8:8 + hlen])
    base = 8 + hlen
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        a, b = meta["data_offsets"]
        arr = np.frombuffer(raw[base + a:base + b], dtype=dt[meta["dtype"]])
        out[name] = arr.reshape(meta["shape"])
    return out


class Policy:
    """Welford obs normalizer + ELU MLP (deterministic mean action)."""

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


def projected_gravity(q_wxyz):
    w, x, y, z = q_wxyz
    u = np.array([-x, -y, -z])
    v = np.array([0.0, 0.0, -1.0])
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def main():
    policy = Policy(POLICY)
    assert policy.act_dim == 12, policy.act_dim
    assert policy.obs_dim % HIST == 0, policy.obs_dim
    frame = policy.obs_dim // HIST
    assert frame in (45, 48), f"unexpected obs frame width {frame}"
    print(f"obs frame {frame} ({'with' if frame == 48 else 'no'} gyro)", flush=True)

    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from isaacsim import SimulationApp

    # BIPED_ISAAC_NOVIDEO=1: physics-only via Isaac Lab's headless kit
    # experience — skips the RTX renderer entirely (librtx.scenedb crashes at
    # plugin startup under driver 595.71.05 on this box; AGILE trains fine
    # headless for the same reason). Numbers only, no camera/mp4.
    NOVIDEO = os.environ.get("BIPED_ISAAC_NOVIDEO") == "1"
    if NOVIDEO:
        exp = os.path.expanduser(
            "~/isaaclab/IsaacLab/apps/isaaclab.python.headless.kit")
        app = SimulationApp({"headless": True}, experience=exp)
    else:
        app = SimulationApp({"headless": True, "width": W, "height": H})

    import omni.kit.commands
    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.core.api.objects.ground_plane import GroundPlane
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.extensions import enable_extension
    from isaacsim.core.utils.types import ArticulationAction
    if not NOVIDEO:
        from isaacsim.core.utils.viewports import set_camera_view
        from isaacsim.sensors.camera import Camera
    from pxr import PhysxSchema, UsdLux, UsdPhysics

    enable_extension("isaacsim.asset.importer.mjcf")
    app.update()

    if not os.path.exists(USD_FILE):
        from isaacsim.asset.importer.mjcf import MJCFImporter, MJCFImporterConfig
        mcfg = MJCFImporterConfig(mjcf_path=FLAT_XML, fix_base=False,
                                  allow_self_collision=False)
        mcfg.usd_path = USD_DIR
        MJCFImporter(mcfg).import_mjcf()
    assert os.path.exists(USD_FILE), USD_FILE

    world = World(physics_dt=PHYS_DT, rendering_dt=CONTROL_DT * RENDER_EVERY,
                  stage_units_in_meters=1.0)
    GroundPlane(prim_path="/World/ground", z_position=0.0,
                color=np.array([0.25, 0.25, 0.28]))

    from isaacsim.core.utils.stage import add_reference_to_stage
    add_reference_to_stage(USD_FILE, "/World/g1")
    stage = omni.usd.get_context().get_stage()
    prim_path = None
    for p in stage.Traverse():
        if str(p.GetPath()).startswith("/World/g1") and p.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim_path = str(p.GetPath())
            break
    if not prim_path:
        raise RuntimeError("no articulation root under /World/g1")
    # Passive-dynamics parity with the trained spec (same fix as the MuJoCo
    # harness): PhysX joint friction 0, armature 0.02 on the policy joints —
    # the URDF import otherwise leaves PhysX defaults / URDF damping that
    # mistime the swing leg exactly like MuJoCo's double-damping did.
    for p in stage.Traverse():
        if p.IsA(UsdPhysics.RevoluteJoint) and p.GetName() in POLICY_JOINTS:
            api = PhysxSchema.PhysxJointAPI.Apply(p)
            # frictionloss 0.1 = the trained spec value.
            api.CreateJointFrictionAttr(0.1).Set(0.1)
            api.CreateArmatureAttr(0.02).Set(0.02)
    dome = UsdLux.DomeLight.Define(stage, "/World/dome")
    dome.CreateIntensityAttr(500.0)
    sun = UsdLux.DistantLight.Define(stage, "/World/sun")
    sun.CreateIntensityAttr(1500.0)

    robot = SingleArticulation(prim_path)
    world.scene.add(robot)
    world.reset()

    names = list(robot.dof_names)
    n = len(names)
    robot.get_articulation_controller().set_gains(kps=np.zeros(n), kds=np.zeros(n))
    pol_d = np.array([names.index(j) for j in POLICY_JOINTS])
    pol_kp = np.zeros(12); pol_kd = np.zeros(12); pol_eff = np.zeros(12)
    for i, jn in enumerate(POLICY_JOINTS):
        pol_kp[i], pol_kd[i], pol_eff[i] = leg_gains(jn)
    held = []  # (dof, kp, kd, eff, q_home from the playground keyframe)
    for d, jn in enumerate(names):
        if jn in POLICY_JOINTS:
            continue
        for frag, kp, kd, eff in HELD:
            if frag in jn:
                held.append((d, kp, kd, eff, HELD_HOME.get(jn, 0.0)))
                break
    print(f"policy joints: 12, held joints: {len(held)}, total dofs: {n}")

    cam = None
    if not NOVIDEO:
        cam = Camera(prim_path="/World/cam", resolution=(W, H))
        cam.initialize()

    rng = np.random.default_rng(7)

    def reset():
        jp = np.zeros(n)
        jp[pol_d] = DEFAULT_POS
        for d, _, _, _, qh in held:
            jp[d] = qh
        yaw = rng.uniform(-0.3, 0.3)
        robot.set_world_pose(position=np.array([0.0, 0.0, 0.79]),
                             orientation=np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]))
        robot.set_linear_velocity(np.zeros(3))
        robot.set_angular_velocity(np.zeros(3))
        if _INIT is not None:
            jp[pol_d] = _INIT["joints"]
            jv = np.zeros(n)
            jv[pol_d] = _INIT["jvel"]
            robot.set_world_pose(position=np.array(_INIT["pos"]),
                                 orientation=np.array(_INIT["quat_wxyz"]))
            robot.set_joint_positions(jp)
            robot.set_joint_velocities(jv)
            robot.set_linear_velocity(np.array(_INIT["lin"]))
            robot.set_angular_velocity(np.array(_INIT["ang"]))
            world.step(render=False)
            return
        robot.set_joint_positions(jp)
        robot.set_joint_velocities(np.zeros(n))
        world.step(render=False)

    ff = None
    if not NOVIDEO:
        ff = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{W}x{H}", "-r", str(int(1 / (CONTROL_DT * RENDER_EVERY))), "-i", "-",
             "-c:v", "libx264", "-preset", "medium", "-crf", "20",
             "-pix_fmt", "yuv420p", OUT], stdin=subprocess.PIPE)

    reset()
    frozen_phase = 0.0
    ep_t = 0
    act_hist = [np.zeros(12), np.zeros(12)]
    vel_track = []
    pitch_track = []
    traj = []  # per-step (body_vx, body_vy, yaw_rate)
    prev_q = np.asarray(robot.get_joint_positions())[pol_d].copy()
    frames_hist = None
    attempts, survived = 1, []
    pos0, _ = robot.get_world_pose()
    dist0 = np.asarray(pos0)[:2].copy()

    n_ctrl = int(SECONDS / CONTROL_DT)
    for t in range(n_ctrl):
        q = np.asarray(robot.get_joint_positions())[pol_d].copy()
        _, quat = robot.get_world_pose()
        quat = np.asarray(quat)

        o = np.zeros(frame)
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
        if frame >= 48:
            # Base angular velocity. Isaac's get_angular_velocity() reports in
            # the WORLD frame -- verified by yawing a body 90 deg, commanding
            # [1,0,0], and finite-differencing the orientation: it rotated about
            # world x, not body x. So the -yaw rotation below is CORRECT here
            # (unlike MuJoCo, whose qvel[3:6] is already body-frame).
            _w = np.asarray(robot.get_angular_velocity())
            _q = np.asarray(quat)
            _yaw = np.arctan2(2 * (_q[0] * _q[3] + _q[1] * _q[2]),
                              1 - 2 * (_q[2] * _q[2] + _q[3] * _q[3]))
            _c, _s = np.cos(-_yaw), np.sin(-_yaw)
            o[45] = _c * _w[0] - _s * _w[1]
            o[46] = _s * _w[0] + _c * _w[1]
            o[47] = _w[2]

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
            jq = np.asarray(robot.get_joint_positions())
            jv = np.asarray(robot.get_joint_velocities())
            tau = np.zeros(n)
            tl = pol_kp * (target - jq[pol_d]) - pol_kd * jv[pol_d]
            tau[pol_d] = np.clip(tl, -pol_eff, pol_eff)
            for d, kp, kd, eff, qh in held:
                tau[d] = np.clip(kp * (qh - jq[d]) - kd * jv[d], -eff, eff)
            robot.apply_action(ArticulationAction(joint_efforts=tau))
            world.step(render=False)

        prev_q = q
        act_hist = [act_hist[1], action.copy()]
        ep_t += 1

        if not NOVIDEO and t % RENDER_EVERY == 0:
            p, _ = robot.get_world_pose()
            p = np.asarray(p)
            set_camera_view(eye=[p[0] + 2.0, p[1] - 2.0, 1.1],
                            target=[p[0], p[1], 0.6],
                            camera_prim_path="/World/cam")
            world.render()
            rgba = cam.get_rgba()
            if rgba is not None and rgba.size:
                ff.stdin.write(np.ascontiguousarray(rgba[:, :, :3]).tobytes())

        p, quat2 = robot.get_world_pose()
        qw, qx, qy, qz = np.asarray(quat2)
        base_yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        lv = np.asarray(robot.get_linear_velocity())
        av = np.asarray(robot.get_angular_velocity())
        vel_track.append((np.cos(base_yaw) * lv[0] + np.sin(base_yaw) * lv[1],
                          -np.sin(base_yaw) * lv[0] + np.cos(base_yaw) * lv[1],
                          av[2]))
        pitch_track.append(np.degrees(np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1))))
        traj.append([float(v) for v in np.asarray(p)[:3]] + [float(v) for v in np.asarray(quat2)[:4]])
        z = np.asarray(p)[2]
        up = projected_gravity(np.asarray(quat2))
        tilt = np.arccos(np.clip(-up[2], -1.0, 1.0))
        if z < FALL_Z or tilt > TILT_LIMIT or ep_t >= int(20.0 / CONTROL_DT):
            d = np.linalg.norm(np.asarray(p)[:2] - dist0)
            why = "timeout" if ep_t >= int(20.0 / CONTROL_DT) else "fell"
            survived.append((ep_t * CONTROL_DT, d, why))
            print(f"attempt {attempts}: {why} after {ep_t * CONTROL_DT:.1f}s, "
                  f"traveled {d:.2f} m", flush=True)
            reset()
            ep_t = 0
            act_hist = [np.zeros(12), np.zeros(12)]
            prev_q = np.asarray(robot.get_joint_positions())[pol_d].copy()
            frames_hist = None
            pos0, _ = robot.get_world_pose()
            dist0 = np.asarray(pos0)[:2].copy()
            attempts += 1

    if ff is not None:
        ff.stdin.close()
        ff.wait()
    if survived:
        ts = [s for s, _, _ in survived]
        print(f"\n{len(survived)} completed attempts; mean survival "
              f"{np.mean(ts):.1f}s, best {max(ts):.1f}s")
    else:
        print(f"\nno falls in {SECONDS:.0f}s")
    print(f"video → {OUT}")
    tpath = os.environ.get("S2S_TRAJ_JSON")
    if tpath:
        with open(tpath, "w") as f:
            json.dump(traj, f)
    mpath = os.environ.get("S2S_METRICS_JSON")
    if mpath:
        p, _ = robot.get_world_pose()
        final_d = float(np.linalg.norm(np.asarray(p)[:2] - dist0))
        eps = [{"seconds": float(t2), "traveled_m": float(d2), "end": why}
               for (t2, d2, why) in survived]
        eps.append({"seconds": ep_t * CONTROL_DT, "traveled_m": final_d,
                    "end": "clip_end"})
        vt = np.array(vel_track) if vel_track else np.zeros((1, 3))
        with open(mpath, "w") as f:
            json.dump({
                "engine": "isaacsim-physx",
                "model_xml": FLAT_XML,
                "policy": POLICY,
                "command": [float(c) for c in CMD],
                "clip_seconds": SECONDS,
                "falls": sum(1 for e in eps if e["end"] == "fell"),
                "mean_body_vel": [float(v) for v in vt.mean(axis=0)],
                "settled_pitch_deg": float(np.mean(pitch_track[len(pitch_track)//2:])) if pitch_track else 0.0,
                "episodes": eps,
            }, f, indent=1)
    app.close()


if __name__ == "__main__":
    main()

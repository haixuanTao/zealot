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

URDF = os.path.expanduser(
    "~/Documents/work/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.urdf")

PHYS_DT = 1.0 / 200.0
DECIMATION = 4
CONTROL_DT = PHYS_DT * DECIMATION
GAIT_PERIOD = 0.7
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
ACTION_SCALE = 0.5


def leg_gains(name):
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
    assert policy.obs_dim == 45 * HIST, policy.obs_dim

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

    enable_extension("isaacsim.asset.importer.urdf")

    world = World(physics_dt=PHYS_DT, rendering_dt=CONTROL_DT * RENDER_EVERY,
                  stage_units_in_meters=1.0)
    GroundPlane(prim_path="/World/ground", z_position=0.0,
                color=np.array([0.25, 0.25, 0.28]))

    _, cfg = omni.kit.commands.execute("URDFCreateImportConfig")
    cfg.fix_base = False
    cfg.import_inertia_tensor = True
    ok, prim_path = omni.kit.commands.execute(
        "URDFParseAndImportFile", urdf_path=URDF, import_config=cfg,
        get_articulation_root=True)
    if not ok or not prim_path:
        raise RuntimeError(f"URDF import failed (ok={ok}, prim={prim_path})")

    stage = omni.usd.get_context().get_stage()
    # Passive-dynamics parity with the trained spec (same fix as the MuJoCo
    # harness): PhysX joint friction 0, armature 0.02 on the policy joints —
    # the URDF import otherwise leaves PhysX defaults / URDF damping that
    # mistime the swing leg exactly like MuJoCo's double-damping did.
    for p in stage.Traverse():
        if p.IsA(UsdPhysics.RevoluteJoint) and p.GetName() in POLICY_JOINTS:
            api = PhysxSchema.PhysxJointAPI.Apply(p)
            api.CreateJointFrictionAttr(0.0).Set(0.0)
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
    held = []  # (dof, kp, kd, eff) — target 0 (URDF zero pose)
    for d, jn in enumerate(names):
        if jn in POLICY_JOINTS:
            continue
        for frag, kp, kd, eff in HELD:
            if frag in jn:
                held.append((d, kp, kd, eff))
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
        yaw = rng.uniform(-0.3, 0.3)
        robot.set_world_pose(position=np.array([0.0, 0.0, 0.76]),
                             orientation=np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]))
        robot.set_linear_velocity(np.zeros(3))
        robot.set_angular_velocity(np.zeros(3))
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
    ep_t = 0
    act_hist = [np.zeros(12), np.zeros(12)]
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

        o = np.zeros(45)
        o[0:12] = act_hist[0] if ep_t >= 2 else 0.0
        o[12:16] = CMD
        o[16:28] = q - DEFAULT_POS
        o[28:40] = 0.0 if ep_t == 0 else (q - prev_q) / CONTROL_DT
        o[40:43] = projected_gravity(quat)
        ph = (max(0, ep_t - 1) * CONTROL_DT / GAIT_PERIOD) % 1.0
        o[43], o[44] = np.sin(2 * np.pi * ph), np.cos(2 * np.pi * ph)

        if frames_hist is None:
            frames_hist = [o.copy() for _ in range(HIST)]
        else:
            frames_hist = frames_hist[1:] + [o.copy()]
        action = policy.act(np.concatenate(frames_hist))
        target = DEFAULT_POS + ACTION_SCALE * action

        for _ in range(DECIMATION):
            jq = np.asarray(robot.get_joint_positions())
            jv = np.asarray(robot.get_joint_velocities())
            tau = np.zeros(n)
            tl = pol_kp * (target - jq[pol_d]) - pol_kd * jv[pol_d]
            tau[pol_d] = np.clip(tl, -pol_eff, pol_eff)
            for d, kp, kd, eff in held:
                tau[d] = np.clip(kp * (0.0 - jq[d]) - kd * jv[d], -eff, eff)
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
    app.close()


if __name__ == "__main__":
    main()

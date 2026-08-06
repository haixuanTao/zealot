#!/usr/bin/env python3
"""Live VR upper-body on a standing v28 velocity policy — full MuJoCo physics.

Legs: zealot v28 policy (12-DOF, 50 Hz, explicit torque PD @200 Hz) with a
zero velocity command (stand). Upper body: PD-held, targets streamed live
from the PICO via the same retargeting as the kinematic mirror. Link poses
stream to robot.html (same page as g1_web.py).

Obs/PD contract copied from examples/biped/sim2sim_g1_mujoco.py (zealot).

    python3 g1_vr_stand.py --host 172.18.130.111
"""

import argparse
import asyncio
import threading
import time
from pathlib import Path

import mujoco
import numpy as np
import websockets
import zmq
from safetensors.numpy import load_file

from g1_mirror import ClutchIK, parse_pose
from g1_web import Shared, build_manifest, serve_http

PUSH_QUEUE = []  # [vx, vy] velocity kicks from the web page
RESET_FLAG = []  # non-empty -> physics loop resets robot + box

HERE = Path(__file__).parent
SCENE = str(HERE / "playground_g1/xmls/scene_mjx_feetonly_flat_terrain.xml")
POLICY_PATH = str(Path(__file__).parent / "checkpoints/velocity_v28/policy.safetensors")

CONTROL_DT = 0.02
DECIMATION = 4
PHYS_DT = CONTROL_DT / DECIMATION
HIST = 5
FALL_Z = 0.45
TILT_LIMIT = np.deg2rad(70.0)
GAIT_PERIOD_SLOW = 0.8
GAIT_PERIOD_FAST = 0.55
GAIT_SPEED_CAP = 0.8   # v28 env.yaml gait_speed_cap
VX_MAX, WZ_MAX = 0.6, 0.8
STICK_DEADZONE = 0.15

# Pick-and-place: two tables + a light ball to carry between them
TABLE_HALF = [0.25, 0.2, 0.44]          # solid slab, top at 0.88 m
TABLE_POS = {"table1": [0.68, 0.45, 0.44], "table2": [0.68, -0.45, 0.44]}
BALL_R = 0.07
BALL_HOME = [0.435, 0.45, 0.96]         # overhanging table1's front edge
BALL_DENSITY = 200.0                    # ~0.29 kg
GRAB_RADIUS = 0.55


def gait_period_for(cmd_speed):
    t = (min(abs(cmd_speed), GAIT_SPEED_CAP) - 0.1) / 0.4
    return max(0.40, GAIT_PERIOD_SLOW + (GAIT_PERIOD_FAST - GAIT_PERIOD_SLOW) * max(t, 0.0))


def build_model():
    """Scene + two tables + a light ball with grab welds and contact pairs."""
    spec = mujoco.MjSpec.from_file(SCENE)

    for name, pos in TABLE_POS.items():
        spec.worldbody.add_geom(
            name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=TABLE_HALF, pos=pos,
            rgba=[0.45, 0.32, 0.22, 1.0], contype=0, conaffinity=0,
        )

    body = spec.worldbody.add_body(name="ball", pos=BALL_HOME)
    body.add_freejoint(name="ball_free")
    body.add_geom(
        name="ball_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[BALL_R, 0, 0],
        density=BALL_DENSITY, rgba=[0.85, 0.25, 0.25, 1.0],
        contype=0, conaffinity=0,
    )
    # Forearm capsules so cradling the ball has contact area
    for side in ("left", "right"):
        spec.body(f"{side}_wrist_roll_link").add_geom(
            name=f"{side}_forearm_collision", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=[0.035, 0, 0], fromto=[0, 0, 0, 0.12, 0, 0],
            contype=0, conaffinity=0, rgba=[1, 1, 1, 0],
        )

    GRIP = [3.0, 3.0, 0.05, 0.01, 0.01]
    ROLL = [0.8, 0.8, 0.005, 0.02, 0.02]  # rolling friction so it settles
    pairs = [("ball_geom", "floor", ROLL),
             ("ball_geom", "table1", ROLL),
             ("ball_geom", "table2", ROLL),
             ("ball_geom", "left_hand_collision", GRIP),
             ("ball_geom", "right_hand_collision", GRIP),
             ("ball_geom", "left_forearm_collision", GRIP),
             ("ball_geom", "right_forearm_collision", GRIP),
             ("ball_geom", "left_foot", None),
             ("ball_geom", "right_foot", None),
             ("left_foot", "table1", None),
             ("left_foot", "table2", None),
             ("right_foot", "table1", None),
             ("right_foot", "table2", None)]
    for g1, g2, fric in pairs:
        p = spec.add_pair()
        p.geomname1 = g1
        p.geomname2 = g2
        if fric is not None:
            p.friction = fric
    # Grab welds: ball <-> each hand, toggled at runtime by the Pico triggers
    for side in ("left", "right"):
        eq = spec.add_equality()
        eq.type = mujoco.mjtEq.mjEQ_WELD
        eq.objtype = mujoco.mjtObj.mjOBJ_BODY
        eq.name = f"grab_{side}"
        eq.name1 = f"{side}_wrist_yaw_link"
        eq.name2 = "ball"
        eq.active = False
        eq.data = [0.0] * 3 + [0.0] * 7 + [0.1]  # anchor, relpose (set at grab), torquescale

    for key in spec.keys:
        if len(key.qpos):
            key.qpos = list(key.qpos) + BALL_HOME + [1, 0, 0, 0]
    model = spec.compile()
    return model

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
    if "ankle" in name:
        return 40.0, 2.0, 50.0


# (prefix, kp, kd, effort) — first fragment wins, mirrors the zealot table
HELD = [
    ("waist_yaw", 300.0, 5.0, 88.0),
    ("waist", 300.0, 5.0, 50.0),
    ("shoulder_pitch", 90.0, 2.0, 25.0),
    ("shoulder_roll", 60.0, 1.0, 25.0),
    ("shoulder", 20.0, 0.4, 25.0),
    ("elbow", 60.0, 1.0, 25.0),
    ("wrist_roll", 12.0, 0.5, 25.0),  # raised from deploy 4/0.2 so twist tracks
    ("wrist", 4.0, 0.2, 25.0),
]

ARM_JOINT_NAMES = {
    "left": ["left_shoulder_pitch_joint", "left_shoulder_roll_joint",
             "left_shoulder_yaw_joint", "left_elbow_joint"],
    "right": ["right_shoulder_pitch_joint", "right_shoulder_roll_joint",
              "right_shoulder_yaw_joint", "right_elbow_joint"],
}
# retarget() output at arms-straight-down, in its own (GR00T-model) convention
RETARGET_ARMS_DOWN = np.array([0.0, 0.0, 0.0, np.pi / 2])


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
    w, x, y, z = q_wxyz
    u = np.array([-x, -y, -z])
    v = np.array([0.0, 0.0, -1.0])
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


class ArmStream:
    """Latest retargeted VR arm targets + thumbstick, stale-aware."""

    STALE_S = 1.0

    def __init__(self, host, port, topic, arm_lengths=(0.19, 0.26)):
        self.lock = threading.Lock()
        self.arm_lengths = arm_lengths
        self.clutch = ClutchIK(*arm_lengths)
        self.estop = False
        self.prev_x = False
        self.latest = None  # (left4, right4) in retarget convention
        self.twists = (0.0, 0.0)
        self.t_last = 0.0
        self.stick = np.zeros(4)
        self.t_stick = 0.0
        self.trig = np.zeros(2)
        self.rx = []
        threading.Thread(target=self._run, args=(host, port, topic), daemon=True).start()

    def _run(self, host, port, topic):
        ctx = zmq.Context()
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt_string(zmq.SUBSCRIBE, topic)
        sock.setsockopt(zmq.CONFLATE, 1)
        sock.connect(f"tcp://{host}:{port}")
        print(f"[zmq] subscribed tcp://{host}:{port}")
        while True:
            msg = parse_pose(sock.recv(), topic.encode())
            now = time.time()
            st = msg.get("stick")
            tr = msg.get("trig")
            with self.lock:
                if st is not None:
                    self.stick = np.asarray(st, dtype=float).flatten()[:4]
                    self.t_stick = now
                if tr is not None:
                    self.trig = np.asarray(tr, dtype=float).flatten()[:2]
            btn = msg.get("btn")
            a_pressed = x_pressed = False
            if btn is not None:
                b = np.asarray(btn).flatten()
                a_pressed = bool(b[0] > 0.5)
                x_pressed = bool(len(b) > 2 and b[2] > 0.5)
            # X = latching e-stop; A = resume (and clutch re-anchor, so the
            # comeback is jump-free).
            if x_pressed and not self.prev_x:
                with self.lock:
                    if not self.estop:
                        self.estop = True
                        print("[E-STOP] X pressed — arms home, command zeroed, grabs released. Press A to resume.")
            self.prev_x = x_pressed
            if a_pressed and self.estop:
                with self.lock:
                    self.estop = False
                print("[E-STOP] cleared by A — resuming (clutch re-anchored)")
            sj = msg.get("smpl_joints")
            if sj is None:
                continue
            frame_j = np.asarray(sj)[-1]
            was_rel = self.clutch.relative
            left, right, twists = self.clutch.update(frame_j, msg.get("wrist_quat"), a_pressed)
            if self.clutch.relative and not was_rel:
                print("[clutch] ANCHORED — relative mode (press A to re-anchor)")
            with self.lock:
                self.twists = tuple(twists)
                self.latest = (left, right)
                self.t_last = now
                self.rx.append(now)
                self.rx = [t for t in self.rx if now - t < 2.0]

    def get(self):
        with self.lock:
            fresh = (time.time() - self.t_last) < self.STALE_S and not self.estop
            return (self.latest if fresh else None), len(self.rx) / 2.0

    def get_twists(self):
        with self.lock:
            return self.twists

    def get_stick(self):
        with self.lock:
            if self.estop or (time.time() - self.t_stick) > self.STALE_S:
                return np.zeros(4)
            return self.stick.copy()

    def get_trig(self):
        with self.lock:
            if self.estop or (time.time() - self.t_stick) > self.STALE_S:
                return np.zeros(2)
            return self.trig.copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="172.18.130.111")
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--topic", default="pose")
    ap.add_argument("--http-port", type=int, default=8001)
    ap.add_argument("--ws-port", type=int, default=8766)
    args = ap.parse_args()

    policy = Policy(POLICY_PATH)
    assert policy.act_dim == 12
    frame = policy.obs_dim // HIST
    assert frame == 53, f"expected 53-dim frames, got {frame}"

    model = build_model()
    model.opt.timestep = PHYS_DT
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.iterations = 100
    model.opt.ls_iterations = 50
    model.opt.impratio = 10.0  # stiffer friction cone so grips don't slip
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ACTUATION
    data = mujoco.MjData(model)

    key_home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    robot_free = next(j for j in range(model.njnt)
                      if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
                      and model.joint(j).name != "ball_free")
    free_q = model.jnt_qposadr[robot_free]
    free_d = model.jnt_dofadr[robot_free]

    pol_q = np.array([model.joint(n).qposadr[0] for n in POLICY_JOINTS])
    pol_d = np.array([model.joint(n).dofadr[0] for n in POLICY_JOINTS])
    pol_rng = np.array([model.jnt_range[model.joint(n).id] for n in POLICY_JOINTS])
    pol_kp = np.array([leg_gains(n)[0] for n in POLICY_JOINTS])
    pol_kd = np.array([leg_gains(n)[1] for n in POLICY_JOINTS])
    pol_eff = np.array([leg_gains(n)[2] for n in POLICY_JOINTS])

    held = []  # (qa, da, kp, kd, eff, q_home, name, lo, hi)
    for jid in range(model.njnt):
        name = model.joint(jid).name
        if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE or name in POLICY_JOINTS:
            continue
        for frag, kp, kd, eff in HELD:
            if frag in name:
                qa, da = model.jnt_qposadr[jid], model.jnt_dofadr[jid]
                qh = float(model.key_qpos[key_home][qa])
                lo, hi = model.jnt_range[jid]
                held.append((qa, da, kp, kd, eff, qh, name, lo, hi))
                break
    print(f"policy joints: 12, held joints: {len(held)}")

    # Per-side conversion offsets: retarget-convention -> this model, derived
    # from the home keyframe (exact at arms-down, approximate elsewhere).
    arm_adr, arm_off, arm_rng = {}, {}, {}
    for side, names in ARM_JOINT_NAMES.items():
        adr = np.array([model.joint(n).qposadr[0] for n in names])
        home = model.key_qpos[key_home][adr]
        arm_adr[side] = adr
        arm_off[side] = home - RETARGET_ARMS_DOWN
        arm_rng[side] = np.array([model.jnt_range[model.joint(n).id] for n in names])
        print(f"[{side}] home={np.round(home,2)} offset={np.round(arm_off[side],2)}")

    box_body = model.body("ball").id
    wrist_adr = {s: model.joint(f"{s}_wrist_roll_joint").qposadr[0] for s in ("left", "right")}
    wrist_rng = {s: model.jnt_range[model.joint(f"{s}_wrist_roll_joint").id] for s in ("left", "right")}
    grab_eq = {s: model.equality(f"grab_{s}").id for s in ("left", "right")}
    hand_body = {s: model.body(f"{s}_wrist_yaw_link").id for s in ("left", "right")}

    # measure the robot's arm segment lengths from the home pose
    _d0 = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, _d0, key_home)
    mujoco.mj_forward(model, _d0)
    L1r = float(np.linalg.norm(_d0.xpos[model.body("left_elbow_link").id]
                               - _d0.xpos[model.body("left_shoulder_roll_link").id]))
    L2r = float(np.linalg.norm(_d0.xpos[model.body("left_wrist_yaw_link").id]
                               - _d0.xpos[model.body("left_elbow_link").id])) + 0.05  # to palm
    print(f"robot arm lengths: upper {L1r:.3f} m, fore+palm {L2r:.3f} m")

    arms = ArmStream(args.host, args.port, args.topic, arm_lengths=(L1r, L2r))
    shared = Shared(model.nbody)
    manifest = build_manifest(model)
    print(f"[manifest] {len(manifest)/1e6:.1f} MB")
    threading.Thread(target=serve_http, args=(str(HERE), manifest, args.http_port), daemon=True).start()

    state = {"falls": 0}

    def reset():
        mujoco.mj_resetDataKeyframe(model, data, key_home)
        # spawn directly in front of the ball's table, facing it
        data.qpos[free_q] = 0.15
        data.qpos[free_q + 1] = BALL_HOME[1]
        data.qpos[pol_q] = DEFAULT_POS
        for qa, _da, _kp, _kd, _eff, qh, _nm, _lo, _hi in held:
            data.qpos[qa] = qh
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

    def physics_loop():
        reset()
        CMD = np.array([0.0, 0.0, 0.0, 0.0])
        cmd_smooth = np.zeros(3)
        ep_t = 0
        act_hist = [np.zeros(12), np.zeros(12)]
        frames_hist = None
        prev_q = data.qpos[pol_q].copy()
        vr_fade = 0.0     # 0 = hold home, 1 = follow VR
        vr_tgt_smooth = None
        wrist_sm = np.zeros(2)
        frozen_phase = 0.0
        ball_floor_t = 0.0
        next_t = time.monotonic()
        while True:
            if RESET_FLAG:
                RESET_FLAG.clear()
                print("MANUAL RESET")
                reset()
                ep_t = 0
                act_hist = [np.zeros(12), np.zeros(12)]
                frames_hist = None
                prev_q = data.qpos[pol_q].copy()
                vr_fade = 0.0
                frozen_phase = 0.0
                cmd_smooth[:] = 0.0

            while PUSH_QUEUE:
                kick = PUSH_QUEUE.pop()
                data.qvel[free_d] += kick[0]
                data.qvel[free_d + 1] += kick[1]
                print(f"PUSH applied: {kick}")

            # --- triggers -> grab/release welds ---
            trig = arms.get_trig()
            for i, side in enumerate(("left", "right")):
                eid = grab_eq[side]
                active = bool(data.eq_active[eid])
                if trig[i] > 0.7 and not active:
                    hp = data.xpos[hand_body[side]]
                    bp = data.xpos[box_body]
                    dist = np.linalg.norm(bp - hp)
                    if dist >= GRAB_RADIUS and ep_t % 25 == 0:
                        print(f"grab {side} MISS: hand {dist:.2f} m from ball (need < {GRAB_RADIUS})")
                    if dist < GRAB_RADIUS:
                        # magnet grab: snap the ball into the palm
                        model.eq_data[eid][0:3] = 0.0
                        model.eq_data[eid][3:6] = [0.09, 0.0, 0.0]  # palm center, hand frame
                        model.eq_data[eid][6:10] = [1.0, 0.0, 0.0, 0.0]
                        data.eq_active[eid] = 1
                        print(f"GRAB {side} (from {dist:.2f} m)")
                elif trig[i] < 0.3 and active:
                    data.eq_active[eid] = 0
                    print(f"RELEASE {side}")

            # --- thumbstick -> velocity command (left stick: fwd/back + steer) ---
            st = arms.get_stick()
            lx, ly = st[0], st[1]
            vx = ly * VX_MAX if abs(ly) > STICK_DEADZONE else 0.0
            wz = -lx * WZ_MAX if abs(lx) > STICK_DEADZONE else 0.0
            cmd_smooth = 0.2 * np.array([vx, 0.0, wz]) + 0.8 * cmd_smooth
            if np.abs(cmd_smooth).max() < 0.02:
                cmd_smooth[:] = 0.0
            CMD[0:3] = cmd_smooth

            q = data.qpos[pol_q].copy()
            quat = data.qpos[free_q + 3:free_q + 7].copy()

            o = np.zeros(frame)
            o[0:12] = act_hist[0] if ep_t >= 2 else 0.0   # LAG-2 last_action
            o[12:16] = CMD
            o[16:28] = q - DEFAULT_POS
            o[28:40] = 0.0 if ep_t == 0 else (q - prev_q) / CONTROL_DT
            o[40:43] = projected_gravity(quat)
            cmd_speed = float(np.linalg.norm(CMD[0:3]))
            if cmd_speed < 0.1:
                ph = frozen_phase                        # stand: clock freezes
            else:
                ph = (max(0, ep_t - 1) * CONTROL_DT / gait_period_for(cmd_speed)) % 1.0
                frozen_phase = ph
            o[43], o[44] = np.sin(2 * np.pi * ph), np.cos(2 * np.pi * ph)
            o[45:48] = data.qvel[free_d + 3:free_d + 6]  # body-frame gyro
            o[48:53] = 0.0                               # no step cue
            if frames_hist is None:
                frames_hist = [o.copy() for _ in range(HIST)]
            else:
                frames_hist = frames_hist[1:] + [o.copy()]
            action = policy.act(np.concatenate(frames_hist))
            target = np.clip(DEFAULT_POS + ACTION_SCALE * action, pol_rng[:, 0], pol_rng[:, 1])

            # --- live VR arm targets with fade in/out ---
            vr, rx_hz = arms.get()
            vr_fade = min(1.0, vr_fade + CONTROL_DT / 0.5) if vr is not None else max(0.0, vr_fade - CONTROL_DT / 0.5)
            held_tgt = {}
            if vr is not None or vr_fade > 0.0:
                if vr is not None:
                    raw = {s: np.clip(vr[i] + arm_off[s], arm_rng[s][:, 0], arm_rng[s][:, 1])
                           for i, s in enumerate(("left", "right"))}
                    if vr_tgt_smooth is None:
                        vr_tgt_smooth = raw
                    else:
                        vr_tgt_smooth = {s: 0.6 * raw[s] + 0.4 * vr_tgt_smooth[s] for s in raw}
                    tw = np.array(arms.get_twists())
                    wrist_sm = 0.5 * tw + 0.5 * wrist_sm
                if vr_tgt_smooth is not None:
                    f = vr_fade * vr_fade * (3 - 2 * vr_fade)
                    for i, side in enumerate(("left", "right")):
                        wa = wrist_adr[side]
                        wt = float(np.clip(wrist_sm[i], wrist_rng[side][0], wrist_rng[side][1]))
                        held_tgt[wa] = f * wt
                    for side in ("left", "right"):
                        for k, qa in enumerate(arm_adr[side]):
                            qh = model.key_qpos[key_home][qa]
                            held_tgt[qa] = float(qh + f * (vr_tgt_smooth[side][k] - qh))

            for _ in range(DECIMATION):
                tau = pol_kp * (target - data.qpos[pol_q]) - pol_kd * data.qvel[pol_d]
                data.qfrc_applied[:] = 0.0
                data.qfrc_applied[pol_d] = np.clip(tau, -pol_eff, pol_eff)
                for qa, da, kp, kd, eff, qh, _nm, _lo, _hi in held:
                    tgt = held_tgt.get(qa, qh)
                    data.qfrc_applied[da] = np.clip(
                        kp * (tgt - data.qpos[qa]) - kd * data.qvel[da], -eff, eff)
                mujoco.mj_step(model, data)

            prev_q = q
            act_hist = [act_hist[1], action.copy()]
            ep_t += 1

            # ball on the floor and not held -> respawn it on the table after 2 s
            ball_held = bool(data.eq_active[grab_eq["left"]]) or bool(data.eq_active[grab_eq["right"]])
            if data.xpos[box_body][2] < 0.2 and not ball_held:
                ball_floor_t += CONTROL_DT
                if ball_floor_t > 2.0:
                    bq_adr = model.jnt_qposadr[model.joint("ball_free").id]
                    bd_adr = model.jnt_dofadr[model.joint("ball_free").id]
                    data.qpos[bq_adr:bq_adr + 7] = list(BALL_HOME) + [1, 0, 0, 0]
                    data.qvel[bd_adr:bd_adr + 6] = 0.0
                    ball_floor_t = 0.0
                    print("ball respawned on table")
            else:
                ball_floor_t = 0.0

            # fall check
            z = data.qpos[free_q + 2]
            g = projected_gravity(data.qpos[free_q + 3:free_q + 7])
            if z < FALL_Z or np.arccos(np.clip(-g[2], -1, 1)) > TILT_LIMIT:
                state["falls"] += 1
                print(f"FALL #{state['falls']} at ep_t={ep_t} — resetting")
                reset()
                ep_t = 0
                act_hist = [np.zeros(12), np.zeros(12)]
                frames_hist = None
                prev_q = data.qpos[pol_q].copy()
                vr_fade = 0.0
                frozen_phase = 0.0
                cmd_smooth[:] = 0.0

            shared.update(data.xpos, data.xquat, rx_hz > 1)
            with shared.lock:
                shared.rx_hz = rx_hz

            next_t += CONTROL_DT
            dt = next_t - time.monotonic()
            if dt > 0:
                time.sleep(dt)
            else:
                next_t = time.monotonic()

    threading.Thread(target=physics_loop, daemon=True).start()

    async def handler(ws):
        import json as _json

        async def reader():
            async for raw in ws:
                try:
                    m = _json.loads(raw)
                except ValueError:
                    continue
                p = m.get("push")
                if isinstance(p, list) and len(p) == 2:
                    PUSH_QUEUE.append(np.clip(np.asarray(p, dtype=float), -3.0, 3.0))
                if m.get("reset"):
                    RESET_FLAG.append(True)

        rtask = asyncio.create_task(reader())
        try:
            while True:
                await ws.send(shared.frame_json())
                await asyncio.sleep(1 / 30)
        except websockets.ConnectionClosed:
            pass
        finally:
            rtask.cancel()

    async def amain():
        async with websockets.serve(handler, "0.0.0.0", args.ws_port):
            print(f"[ws] listening on :{args.ws_port} — viewer http://localhost:{args.http_port}")
            await asyncio.Future()

    asyncio.run(amain())


if __name__ == "__main__":
    main()

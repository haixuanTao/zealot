#!/usr/bin/env python3
"""G1 upper-body mirror: PICO SMPL pose stream -> G1 arm joints in MuJoCo.

The robot stands kinematically (base + legs pinned); only the arms track.

Live mode (macOS needs mjpython for the viewer):
    mjpython g1_mirror.py --host 172.18.130.111
Offline sign-check (renders synthetic poses to PNGs, no viewer):
    python3 g1_mirror.py --test
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

G1_XML = str(Path(__file__).parent / "g1_model/g1_gear_wbc.xml")
HEADER_SIZE = 1024

# SMPL joint indices
PELVIS, NECK = 0, 12
L_SHOULDER, R_SHOULDER = 16, 17
L_ELBOW, R_ELBOW = 18, 19
L_WRIST, R_WRIST = 20, 21

ARM_JOINTS = {
    "left": ["left_shoulder_pitch_joint", "left_shoulder_roll_joint",
             "left_shoulder_yaw_joint", "left_elbow_joint"],
    "right": ["right_shoulder_pitch_joint", "right_shoulder_roll_joint",
              "right_shoulder_yaw_joint", "right_elbow_joint"],
}
STAND_POSE = {  # gentle standing crouch
    "left_hip_pitch_joint": -0.2, "left_knee_joint": 0.4, "left_ankle_pitch_joint": -0.2,
    "right_hip_pitch_joint": -0.2, "right_knee_joint": 0.4, "right_ankle_pitch_joint": -0.2,
}


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def torso_frame(j):
    """Rows = torso axes (x fwd, y left, z up) in world; transforms world->torso."""
    y = j[L_SHOULDER] - j[R_SHOULDER]
    y /= np.linalg.norm(y) + 1e-9
    z = j[NECK] - j[PELVIS]
    z -= y * (z @ y)
    z /= np.linalg.norm(z) + 1e-9
    x = np.cross(y, z)
    return np.stack([x, y, z])


def arm_angles(j, side):
    """SMPL positions -> (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow_qpos).

    G1 arm (from g1_gear_wbc.xml): pitch about +y, roll about +x, yaw about +z,
    elbow about +y. Model ZERO pose = upper arm hanging down, FOREARM FORWARD
    (elbow bent 90°); straight arm = elbow qpos +pi/2.
    """
    R = torso_frame(j)
    sh, el, wr = (
        (j[L_SHOULDER], j[L_ELBOW], j[L_WRIST])
        if side == "left"
        else (j[R_SHOULDER], j[R_ELBOW], j[R_WRIST])
    )
    u = R @ (el - sh)
    u /= np.linalg.norm(u) + 1e-9
    f = R @ (wr - el)
    f /= np.linalg.norm(f) + 1e-9

    # u = Ry(p) Rx(r) . (0,0,-1)  =>  r = asin(u_y), p = atan2(-u_x, -u_z)
    r = np.arcsin(np.clip(u[1], -1, 1))
    p = np.arctan2(-u[0], -u[2]) if abs(np.cos(r)) > 0.08 else 0.0

    flex = np.arccos(np.clip(u @ f, -1, 1))  # 0 = straight arm
    elbow_q = np.pi / 2 - flex               # model zero = 90° bend

    # forearm in the post-(pitch,roll) frame: w = Rz(yaw) Ry(elbow_q) . (1,0,0)
    w = rot_x(-r) @ rot_y(-p) @ f
    yaw = np.arctan2(w[1], w[0]) if np.sin(flex) > 0.15 else 0.0
    return np.array([p, r, yaw, elbow_q])


def retarget(j):
    return arm_angles(j, "left"), arm_angles(j, "right")


def scaled_wrist_target(j, side, Lr):
    """Torso-frame wrist target at ROBOT scale + upper-arm swivel hint.

    Fractional reach (|wrist-shoulder| / arm length) maps 1:1 between human
    and robot, so depth control is linear.
    """
    R = torso_frame(j)
    sh, el, wr = (
        (j[L_SHOULDER], j[L_ELBOW], j[L_WRIST])
        if side == "left"
        else (j[R_SHOULDER], j[R_ELBOW], j[R_WRIST])
    )
    u = R @ (el - sh)
    L1h = np.linalg.norm(u) + 1e-9
    L2h = np.linalg.norm(wr - el) + 1e-9
    t = R @ (wr - sh) * (Lr / (L1h + L2h))
    return t, u


def clamp_reach(t, L1r, L2r):
    d = np.linalg.norm(t) + 1e-9
    d_cl = np.clip(d, abs(L1r - L2r) + 1e-3, 0.995 * (L1r + L2r))
    return t * (d_cl / d)


# Self-collision fail-safe: a vertical keep-out capsule around the robot's
# torso/head/waist (torso axes). Wrist targets that would enter it are
# projected OUT to the surface — the hand slides along the body instead of
# being commanded through the chest. Coordinates are shoulder-relative
# (the IK's frame); the torso axis sits half a shoulder-width inboard.
KEEPOUT_RADIUS = 0.12          # slim enough that hands hanging at the thighs clear it
KEEPOUT_Z = (-0.35, 0.30)      # waist to head, relative to shoulder height
KEEPOUT_AXIS_X = -0.02         # axis slightly behind the shoulder line
SHOULDER_HALF_WIDTH = 0.147


def apply_keepout(t, side):
    """Project a shoulder-relative wrist target out of the torso capsule."""
    sign = 1.0 if side == "left" else -1.0
    ax, ay = KEEPOUT_AXIS_X, -sign * SHOULDER_HALF_WIDTH
    if not (KEEPOUT_Z[0] < t[2] < KEEPOUT_Z[1]):
        return t, False
    dx, dy = t[0] - ax, t[1] - ay
    dist = np.hypot(dx, dy)
    if dist >= KEEPOUT_RADIUS:
        return t, False
    if dist < 1e-6:
        dx, dy = 0.0, sign  # degenerate: push out to the arm's own side
        dist = 1.0
    s = KEEPOUT_RADIUS / dist
    out = t.copy()
    out[0] = ax + dx * s
    out[1] = ay + dy * s
    return out, True


def solve_arm_ik(t, u_hint, L1r, L2r):
    """Wrist target (torso frame, within reach) -> (pitch, roll, yaw,
    elbow_qpos with pi/2 = straight). `u_hint` = human upper-arm direction,
    picks the elbow-swivel solution closest to the operator's."""
    t = clamp_reach(t, L1r, L2r)
    d = np.linalg.norm(t)

    # elbow flexion from the triangle (L1r, L2r, d); 0 = straight
    cos_int = (L1r**2 + L2r**2 - d**2) / (2 * L1r * L2r)
    flex = np.pi - np.arccos(np.clip(cos_int, -1, 1))
    elbow_q = np.pi / 2 - flex

    # robot elbow position: on the circle around the shoulder-wrist axis,
    # at the swivel closest to the human's elbow direction
    a = t / d
    cos_sh = (L1r**2 + d**2 - L2r**2) / (2 * L1r * d)
    cos_sh = np.clip(cos_sh, -1, 1)
    perp = u_hint - (u_hint @ a) * a
    if np.linalg.norm(perp) < 1e-6:
        perp = np.array([0.0, 0.0, -1.0]) - a * (-a[2])
    perp /= np.linalg.norm(perp) + 1e-9
    e_pos = L1r * (cos_sh * a + np.sqrt(max(0.0, 1 - cos_sh**2)) * perp)

    ur = e_pos / (np.linalg.norm(e_pos) + 1e-9)
    r = np.arcsin(np.clip(ur[1], -1, 1))
    p = np.arctan2(-ur[0], -ur[2]) if abs(np.cos(r)) > 0.08 else 0.0

    f_r = t - e_pos
    w = rot_x(-r) @ rot_y(-p) @ f_r
    # Near-straight arms: the swivel is ill-defined and atan2 flips wildly —
    # hold yaw neutral until there's a real elbow bend.
    yaw = np.arctan2(w[1], w[0]) if np.sin(flex) > 0.3 else 0.0
    return np.array([p, r, yaw, elbow_q])


def arm_angles_ik(j, side, L1r, L2r):
    t, u = scaled_wrist_target(j, side, L1r + L2r)
    t, _ = apply_keepout(t, side)
    return solve_arm_ik(t, u, L1r, L2r)


def retarget_ik(j, L1r, L2r):
    return arm_angles_ik(j, "left", L1r, L2r), arm_angles_ik(j, "right", L1r, L2r)


class ClutchIK:
    """SONIC-style anchored (relative) retargeting.

    Absolute until the first A press. Pressing A captures BOTH the operator's
    current wrist positions and the robot's current wrist targets; from then
    on the operator's motion is applied as a scaled cartesian DELTA on top of
    the robot anchor — no jump at the anchor moment, and systematic
    calibration offset cancels. Press A again any time to re-anchor (lift the
    mouse). Wrist twist gets the same treatment.
    """

    def __init__(self, L1r, L2r):
        self.L1r, self.L2r = L1r, L2r
        self.Lr = L1r + L2r
        self.prev_a = False
        self.anchor = None
        self.last_t = {}
        self.last_tw = np.zeros(2)
        self._last_keepout_log = 0.0

    @property
    def relative(self):
        return self.anchor is not None

    def update(self, j, wrist_quats, a_pressed):
        """One frame: SMPL joints + wrist quats + A state -> (left4, right4, twists2)."""
        rising = a_pressed and not self.prev_a
        self.prev_a = a_pressed

        th, hint = {}, {}
        for side in ("left", "right"):
            th[side], hint[side] = scaled_wrist_target(j, side, self.Lr)
        tw = np.zeros(2)
        if wrist_quats is not None:
            tw = np.array([wrist_twist(j, wrist_quats, "left"),
                           wrist_twist(j, wrist_quats, "right")])

        if rising:
            self.anchor = {
                "h": {s: th[s].copy() for s in th},
                "r": {s: self.last_t.get(s, th[s]).copy() for s in th},
                "tw_h": tw.copy(),
                "tw_r": self.last_tw.copy(),
            }

        out = {}
        for side in ("left", "right"):
            if self.anchor is None:
                t = th[side]
            else:
                t = self.anchor["r"][side] + (th[side] - self.anchor["h"][side])
            t, blocked = apply_keepout(t, side)   # self-collision fail-safe
            t = clamp_reach(t, self.L1r, self.L2r)  # clamp BEFORE storing: no windup
            if blocked and time.monotonic() - self._last_keepout_log > 2.0:
                self._last_keepout_log = time.monotonic()
                print(f"[keepout] {side} wrist target grazing the torso — sliding on the surface")
            self.last_t[side] = t.copy()
            out[side] = solve_arm_ik(t, hint[side], self.L1r, self.L2r)

        if self.anchor is None:
            tw_out = tw
        else:
            tw_out = self.anchor["tw_r"] + (tw - self.anchor["tw_h"])
        tw_out = np.clip(tw_out, -1.9, 1.9)
        self.last_tw = tw_out.copy()
        return out["left"], out["right"], tw_out


# Wrist-twist extraction tunables (Pico wrist-joint local-frame conventions)
TWIST_AXIS_COL = 0     # which column of the hand rotation matrix to project
TWIST_SIGN = {"left": 1.0, "right": -1.0}
TWIST_OFFSET = {"left": 0.0, "right": 0.0}


def _quat_xyzw_to_mat(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def wrist_twist(j_zup, wrist_quats, side):
    """Signed hand rotation about the forearm axis, elbow plane as zero.

    j_zup: [24,3] SMPL joints (z-up, as streamed). wrist_quats: [4,4] xyzw in
    the ORIGINAL y-up tracking frame for joints [l_wrist, r_wrist, l_hand, r_hand].
    """
    si, ei, wi, qi = (L_SHOULDER, L_ELBOW, L_WRIST, 0) if side == "left" else (R_SHOULDER, R_ELBOW, R_WRIST, 1)
    # back to the y-up frame the quats live in: (x, y, z)_yup = (x, z, -y)_zup
    def yup(p):
        return np.array([p[0], p[2], -p[1]])
    S, E, W = yup(j_zup[si]), yup(j_zup[ei]), yup(j_zup[wi])
    a = W - E
    a /= np.linalg.norm(a) + 1e-9
    n0 = np.cross(E - S, W - E)
    if np.linalg.norm(n0) < 1e-6:
        return 0.0
    n0 /= np.linalg.norm(n0)
    h = _quat_xyzw_to_mat(wrist_quats[qi])[:, TWIST_AXIS_COL]
    v = h - (h @ a) * a
    if np.linalg.norm(v) < 1e-6:
        return 0.0
    v /= np.linalg.norm(v)
    ang = np.arctan2(np.cross(n0, v) @ a, n0 @ v)
    return TWIST_SIGN[side] * ang + TWIST_OFFSET[side]


def parse_pose(data, topic=b"pose"):
    body = data[len(topic):]
    hdr = json.loads(body[:HEADER_SIZE].rstrip(b"\x00").decode())
    payload, off, out = body[HEADER_SIZE:], 0, {}
    dt = {"f32": "<f4", "f64": "<f8", "i32": "<i4", "i64": "<i8"}
    for f in hdr["fields"]:
        d = np.dtype(dt[f["dtype"]])
        n = int(np.prod(f["shape"]))
        out[f["name"]] = np.frombuffer(payload, d, n, off).reshape(f["shape"])
        off += d.itemsize * n
    return out


class G1Rig:
    def __init__(self):
        import mujoco

        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(G1_XML)
        self.data = mujoco.MjData(self.model)
        self.arm_adr = {
            side: [self.model.joint(n).qposadr[0] for n in names]
            for side, names in ARM_JOINTS.items()
        }
        self.reset()

    def reset(self):
        self.data.qpos[:] = 0
        self.data.qpos[:7] = [0, 0, 0.78, 1, 0, 0, 0]
        for name, val in STAND_POSE.items():
            self.data.qpos[self.model.joint(name).qposadr[0]] = val

    def set_arms(self, left, right):
        for adr, v in zip(self.arm_adr["left"], left):
            self.data.qpos[adr] = v
        for adr, v in zip(self.arm_adr["right"], right):
            self.data.qpos[adr] = v
        self.mj.mj_forward(self.model, self.data)


def synthetic(pose):
    """Synthetic SMPL joints (z-up, standing at origin) for sign tests."""
    j = np.zeros((24, 3))
    j[PELVIS] = [0, 0, 0.95]
    j[NECK] = [0, 0, 1.45]
    j[L_SHOULDER] = [0, 0.20, 1.40]
    j[R_SHOULDER] = [0, -0.20, 1.40]
    UP, FORE = 0.28, 0.25
    if pose == "hang":  # arms straight down
        j[L_ELBOW] = j[L_SHOULDER] + [0, 0, -UP]
        j[R_ELBOW] = j[R_SHOULDER] + [0, 0, -UP]
        j[L_WRIST] = j[L_ELBOW] + [0, 0, -FORE]
        j[R_WRIST] = j[R_ELBOW] + [0, 0, -FORE]
    elif pose == "tpose":  # arms straight out sideways
        j[L_ELBOW] = j[L_SHOULDER] + [0, UP, 0]
        j[R_ELBOW] = j[R_SHOULDER] + [0, -UP, 0]
        j[L_WRIST] = j[L_ELBOW] + [0, FORE, 0]
        j[R_WRIST] = j[R_ELBOW] + [0, -FORE, 0]
    elif pose == "forward":  # arms straight forward
        j[L_ELBOW] = j[L_SHOULDER] + [UP, 0, 0]
        j[R_ELBOW] = j[R_SHOULDER] + [UP, 0, 0]
        j[L_WRIST] = j[L_ELBOW] + [FORE, 0, 0]
        j[R_WRIST] = j[R_ELBOW] + [FORE, 0, 0]
    elif pose == "lpose":  # upper arms down, forearms forward (calibration L)
        j[L_ELBOW] = j[L_SHOULDER] + [0, 0, -UP]
        j[R_ELBOW] = j[R_SHOULDER] + [0, 0, -UP]
        j[L_WRIST] = j[L_ELBOW] + [FORE, 0, 0]
        j[R_WRIST] = j[R_ELBOW] + [FORE, 0, 0]
    return j


def run_test():
    import PIL.Image

    rig = G1Rig()
    renderer = rig.mj.Renderer(rig.model, height=480, width=480)
    out = Path(__file__).parent / "test_renders"
    out.mkdir(exist_ok=True)
    for pose in ["hang", "tpose", "forward", "lpose"]:
        left, right = retarget(synthetic(pose))
        rig.reset()
        rig.set_arms(left, right)
        cam = rig.mj.MjvCamera()
        cam.lookat[:] = [0, 0, 0.8]
        cam.distance, cam.azimuth, cam.elevation = 2.2, 160, -10
        renderer.update_scene(rig.data, camera=cam)
        PIL.Image.fromarray(renderer.render()).save(out / f"{pose}.png")
        print(f"{pose:8s} L(p,r,y,e)={np.round(left, 2)}  R={np.round(right, 2)}")
    print(f"renders in {out}")


def run_live(host, port, topic):
    import mujoco.viewer
    import zmq

    rig = G1Rig()
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt_string(zmq.SUBSCRIBE, topic)
    sock.setsockopt(zmq.CONFLATE, 1)
    sock.connect(f"tcp://{host}:{port}")
    print(f"subscribed tcp://{host}:{port}; launching viewer...")
    smooth_l = np.zeros(4)
    smooth_r = np.zeros(4)
    alpha = 0.35
    with mujoco.viewer.launch_passive(rig.model, rig.data) as viewer:
        while viewer.is_running():
            if sock.poll(timeout=0):
                msg = parse_pose(sock.recv(zmq.NOBLOCK), topic.encode())
                sj = msg.get("smpl_joints")
                if sj is not None:
                    left, right = retarget(np.asarray(sj)[-1])
                    smooth_l = alpha * left + (1 - alpha) * smooth_l
                    smooth_r = alpha * right + (1 - alpha) * smooth_r
            rig.set_arms(smooth_l, smooth_r)
            viewer.sync()
            time.sleep(1 / 60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="172.18.130.111")
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--topic", default="pose")
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    if args.test:
        run_test()
    else:
        run_live(args.host, args.port, args.topic)

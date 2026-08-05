#!/usr/bin/env python3
"""Fake PICO pose publisher for testing the viewer without hardware.

Emits protocol-v3 messages ([topic][1024B JSON header][binary fields]) on
tcp://*:5556 topic 'pose', matching pack_pose_message() in
gear_sonic/utils/teleop/zmq/zmq_planner_sender.py: an animated SMPL
skeleton waving its arms plus sinusoidal wrist joints.
"""

import json
import time

import numpy as np
import zmq

HEADER_SIZE = 1024
TOPIC = b"pose"

# T-pose-ish SMPL joint offsets relative to parent (z-up, meters)
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
OFFSETS = np.array([
    [0, 0, 0.95],        # 0 pelvis (world z)
    [0.09, 0, -0.06],    # 1 l_hip
    [-0.09, 0, -0.06],   # 2 r_hip
    [0, 0, 0.12],        # 3 spine1
    [0, 0, -0.40],       # 4 l_knee
    [0, 0, -0.40],       # 5 r_knee
    [0, 0, 0.13],        # 6 spine2
    [0, 0, -0.42],       # 7 l_ankle
    [0, 0, -0.42],       # 8 r_ankle
    [0, 0, 0.05],        # 9 spine3
    [0, 0.12, -0.05],    # 10 l_foot
    [0, 0.12, -0.05],    # 11 r_foot
    [0, 0, 0.10],        # 12 neck
    [0.08, 0, 0.03],     # 13 l_collar
    [-0.08, 0, 0.03],    # 14 r_collar
    [0, 0, 0.12],        # 15 head
    [0.10, 0, 0],        # 16 l_shoulder
    [-0.10, 0, 0],       # 17 r_shoulder
    [0.26, 0, 0],        # 18 l_elbow
    [-0.26, 0, 0],       # 19 r_elbow
    [0.25, 0, 0],        # 20 l_wrist
    [-0.25, 0, 0],       # 21 r_wrist
    [0.08, 0, 0],        # 22 l_hand
    [-0.08, 0, 0],       # 23 r_hand
])


def skeleton_at(t: float) -> np.ndarray:
    joints = np.zeros((24, 3))
    off = OFFSETS.copy()
    wave = 0.5 * np.sin(2 * np.pi * 0.5 * t)
    # wave both forearms up and down (rotate elbow->wrist and wrist->hand about y)
    for i, sign in ((20, 1), (21, -1), (22, 1), (23, -1)):
        c, s = np.cos(wave), np.sin(wave)
        x, _, z = off[i]
        off[i] = [c * x - s * z * sign, 0, s * x * sign + c * z]
    off[0][2] += 0.03 * np.sin(2 * np.pi * 1.0 * t)  # bob
    for i in range(24):
        p = PARENTS[i]
        joints[i] = off[i] if p < 0 else joints[p] + off[i]
    return joints


def pack(pose_data: dict, version: int = 3) -> bytes:
    fields, blobs = [], []
    for key, value in pose_data.items():
        value = np.ascontiguousarray(value)
        dtype_str = {"float32": "f32", "float64": "f64", "int32": "i32", "int64": "i64"}[value.dtype.name]
        fields.append({"name": key, "dtype": dtype_str, "shape": list(value.shape)})
        blobs.append(value.tobytes())
    header = {"v": version, "endian": "le", "count": 1, "fields": fields}
    hb = json.dumps(header, separators=(",", ":")).encode().ljust(HEADER_SIZE, b"\x00")
    return TOPIC + hb + b"".join(blobs)


def main():
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.bind("tcp://*:5556")
    print("fake publisher on tcp://*:5556 topic 'pose' (protocol v3), ctrl-c to stop")
    t0 = time.time()
    frame = 0
    while True:
        t = time.time() - t0
        joint_pos = np.zeros((1, 29), dtype=np.float32)
        joint_pos[0, 23:29] = 0.6 * np.sin(2 * np.pi * 0.5 * t + np.arange(6))  # wrists
        msg = pack({
            "joint_pos": joint_pos,
            "joint_vel": np.zeros((1, 29), dtype=np.float32),
            "smpl_joints": skeleton_at(t)[None].astype(np.float32),
            "smpl_pose": np.zeros((1, 21, 3), dtype=np.float32),
            "frame_index": np.array([frame], dtype=np.int64),
        })
        sock.send(msg)
        frame += 1
        time.sleep(1 / 50)


if __name__ == "__main__":
    main()

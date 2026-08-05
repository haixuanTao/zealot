#!/usr/bin/env python3
"""Retarget bridge: PICO pose stream -> named G1 joint targets over WebSocket.

The reusable middle piece between a headset server (headset_server/pose_pub.py
on the robot, or fake_publisher.py locally) and any consumer that PD-holds the
G1 upper body — the zealot website demo's "Connect VR" setting, the MuJoCo
sim2sim harness, or anything else that speaks JSON.

Emits at ~50 Hz on ws://0.0.0.0:8765 (one JSON object per message):

    {
      "arm_targets": {"left_shoulder_pitch_joint": -0.31, ...},   # radians
      "stick": [lx, ly, rx, ry],
      "trig": [l, r],
      "fresh": true          # false -> body tracking stale, targets = last
    }

Joint names follow the Unitree G1 MJCF convention (shoulder pitch/roll/yaw,
elbow, wrist_roll per side). Consumers pick the joints they know and ignore
the rest.

    python3 retarget_bridge.py --host 172.18.130.111       # robot stream
    python3 retarget_bridge.py --host localhost            # fake_publisher
"""

import argparse
import asyncio
import json
import threading
import time

import numpy as np
import websockets
import zmq

from g1_mirror import parse_pose, retarget_ik, wrist_twist

# retarget-convention -> Unitree G1 MJCF zero conventions (playground model):
# home (arms hanging) = pitch 0.2, roll ±0.2, elbow 1.28; retarget's hanging
# pose = (0, 0, 0, pi/2). Offsets bridge the two.
ARM_OFFSET = {
    "left": np.array([0.2, 0.2, 0.0, 1.28 - np.pi / 2]),
    "right": np.array([0.2, -0.2, 0.0, 1.28 - np.pi / 2]),
}
ARM_JOINTS = {
    "left": ["left_shoulder_pitch_joint", "left_shoulder_roll_joint",
             "left_shoulder_yaw_joint", "left_elbow_joint"],
    "right": ["right_shoulder_pitch_joint", "right_shoulder_roll_joint",
              "right_shoulder_yaw_joint", "right_elbow_joint"],
}
# G1 arm segment lengths (upper arm, forearm+palm) for fractional-reach IK
L1R, L2R = 0.185, 0.234
SMOOTH = 0.6


class Bridge:
    def __init__(self, host, port, topic):
        self.lock = threading.Lock()
        self.targets = {}
        self.stick = [0.0] * 4
        self.trig = [0.0, 0.0]
        self.t_body = 0.0
        self._smooth = None
        self._wrist_sm = np.zeros(2)
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
                    self.stick = [float(v) for v in np.asarray(st).flatten()[:4]]
                if tr is not None:
                    self.trig = [float(v) for v in np.asarray(tr).flatten()[:2]]
            sj = msg.get("smpl_joints")
            if sj is None:
                continue
            frame = np.asarray(sj)[-1]
            left, right = retarget_ik(frame, L1R, L2R)
            raw = {"left": left + ARM_OFFSET["left"], "right": right + ARM_OFFSET["right"]}
            wq = msg.get("wrist_quat")
            tw = np.zeros(2)
            if wq is not None:
                tw = np.array([wrist_twist(frame, wq, "left"), wrist_twist(frame, wq, "right")])
            with self.lock:
                if self._smooth is None:
                    self._smooth = raw
                else:
                    self._smooth = {s: SMOOTH * raw[s] + (1 - SMOOTH) * self._smooth[s] for s in raw}
                self._wrist_sm = 0.5 * tw + 0.5 * self._wrist_sm
                t = {}
                for side in ("left", "right"):
                    for name, val in zip(ARM_JOINTS[side], self._smooth[side]):
                        t[name] = round(float(val), 4)
                t["left_wrist_roll_joint"] = round(float(self._wrist_sm[0]), 4)
                t["right_wrist_roll_joint"] = round(float(self._wrist_sm[1]), 4)
                self.targets = t
                self.t_body = now

    def snapshot(self):
        with self.lock:
            return json.dumps({
                "arm_targets": self.targets,
                "stick": self.stick,
                "trig": self.trig,
                "fresh": (time.time() - self.t_body) < 1.0,
            })


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="172.18.130.111")
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--topic", default="pose")
    ap.add_argument("--ws-port", type=int, default=8765)
    ap.add_argument("--fps", type=float, default=50.0)
    args = ap.parse_args()

    bridge = Bridge(args.host, args.port, args.topic)

    async def handler(ws):
        print(f"[ws] client connected: {ws.remote_address}")
        try:
            while True:
                await ws.send(bridge.snapshot())
                await asyncio.sleep(1.0 / args.fps)
        except websockets.ConnectionClosed:
            print("[ws] client disconnected")

    async with websockets.serve(handler, "0.0.0.0", args.ws_port):
        print(f"[ws] serving retargeted joint targets on :{args.ws_port}")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())

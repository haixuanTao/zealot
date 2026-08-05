#!/usr/bin/env python3
"""ZMQ → WebSocket bridge for the GR00T-WholeBodyControl PICO pose stream.

Subscribes to the pico_manager_thread_server PUB socket (topic 'pose'),
parses the [topic][1024-byte JSON header][binary fields] wire format, and
rebroadcasts the latest frame as JSON to browser WebSocket clients.
Also serves the static viewer page over HTTP.

Usage:
    python bridge.py --host 172.18.130.111            # robot stream
    python bridge.py --host localhost                 # fake_publisher test
Then open http://localhost:8000
"""

import argparse
import asyncio
import http.server
import json
import struct
import threading
import time
from functools import partial
from pathlib import Path

import numpy as np
import websockets
import zmq

HEADER_SIZE = 1024
DTYPE_MAP = {
    "f32": np.dtype("<f4"),
    "f64": np.dtype("<f8"),
    "i32": np.dtype("<i4"),
    "i64": np.dtype("<i8"),
    "u8": np.dtype("u1"),
    "bool": np.dtype("u1"),
}


def parse_pose_message(data: bytes, topic: str) -> dict | None:
    body = data[len(topic):]
    if len(body) < HEADER_SIZE:
        return None
    header = json.loads(body[:HEADER_SIZE].rstrip(b"\x00").decode("utf-8"))
    payload = body[HEADER_SIZE:]
    out = {"_version": header.get("v")}
    offset = 0
    for field in header["fields"]:
        dt = DTYPE_MAP.get(field["dtype"])
        if dt is None:
            return None
        shape = field["shape"]
        nbytes = dt.itemsize * int(np.prod(shape))
        if offset + nbytes > len(payload):
            return None
        arr = np.frombuffer(payload, dtype=dt, count=int(np.prod(shape)), offset=offset)
        out[field["name"]] = arr.reshape(shape)
        offset += nbytes
    return out


class LatestFrame:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None
        self.rx_count = 0
        self.rx_window = []

    def put(self, frame):
        now = time.time()
        with self.lock:
            self.frame = frame
            self.rx_count += 1
            self.rx_window.append(now)
            self.rx_window = [t for t in self.rx_window if now - t < 2.0]

    def get(self):
        with self.lock:
            hz = len(self.rx_window) / 2.0
            return self.frame, self.rx_count, hz


def zmq_reader(host: str, port: int, topic: str, latest: LatestFrame):
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt_string(zmq.SUBSCRIBE, topic)
    sock.setsockopt(zmq.CONFLATE, 1)
    sock.connect(f"tcp://{host}:{port}")
    print(f"[zmq] subscribed to tcp://{host}:{port} topic={topic!r}")
    while True:
        data = sock.recv()
        try:
            msg = parse_pose_message(data, topic)
        except Exception as e:
            print(f"[zmq] parse error: {e}")
            continue
        if msg is not None:
            latest.put(msg)


def frame_to_json(msg: dict, rx_count: int, hz: float) -> str:
    out = {
        "version": msg.get("_version"),
        "rx_count": rx_count,
        "rx_hz": round(hz, 1),
        "fields": [k for k in msg if not k.startswith("_")],
    }
    sj = msg.get("smpl_joints")
    if sj is not None:
        out["smpl_joints"] = np.asarray(sj)[-1].round(4).tolist()  # last frame, [24,3]
    jp = msg.get("joint_pos")
    if jp is not None:
        out["joint_pos"] = np.asarray(jp)[-1].round(4).tolist()  # [29]
    fi = msg.get("frame_index")
    if fi is not None:
        out["frame_index"] = int(np.asarray(fi).flat[-1])
    for hand in ("left_hand_joints", "right_hand_joints"):
        hj = msg.get(hand)
        if hj is not None:
            out[hand] = np.asarray(hj).flatten().round(3).tolist()
    return json.dumps(out)


async def ws_handler(ws, latest: LatestFrame, fps: float):
    print(f"[ws] client connected: {ws.remote_address}")
    try:
        while True:
            frame, rx_count, hz = latest.get()
            if frame is not None:
                await ws.send(frame_to_json(frame, rx_count, hz))
            else:
                await ws.send(json.dumps({"waiting": True}))
            await asyncio.sleep(1.0 / fps)
    except websockets.ConnectionClosed:
        print("[ws] client disconnected")


def serve_http(directory: str, port: int):
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=directory)
    httpd = http.server.ThreadingHTTPServer(("0.0.0.0", port), handler)
    print(f"[http] viewer at http://localhost:{port}")
    httpd.serve_forever()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="172.18.130.111", help="pico streamer host")
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--topic", default="pose")
    ap.add_argument("--ws-port", type=int, default=8765)
    ap.add_argument("--http-port", type=int, default=8000)
    ap.add_argument("--fps", type=float, default=30.0, help="websocket push rate")
    args = ap.parse_args()

    latest = LatestFrame()
    threading.Thread(target=zmq_reader, args=(args.host, args.port, args.topic, latest), daemon=True).start()
    threading.Thread(target=serve_http, args=(str(Path(__file__).parent), args.http_port), daemon=True).start()

    async with websockets.serve(lambda ws: ws_handler(ws, latest, args.fps), "0.0.0.0", args.ws_port):
        print(f"[ws] listening on :{args.ws_port}")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""Web G1 upper-body mirror: PICO pose stream -> retarget -> FK -> browser.

Serves http://localhost:8001 (robot.html + mesh manifest) and streams link
poses over WebSocket :8766. The robot stands kinematically; arms mirror the
operator. No physics, no native window — plain Python + three.js.

    python3 g1_web.py --host 172.18.130.111
"""

import argparse
import asyncio
import base64
import http.server
import json
import threading
import time
from functools import partial
from pathlib import Path

import numpy as np
import websockets
import zmq

from g1_mirror import G1Rig, parse_pose, retarget

import mujoco


def build_manifest(model) -> bytes:
    meshes = []
    for i in range(model.nmesh):
        va, vn = model.mesh_vertadr[i], model.mesh_vertnum[i]
        fa, fn = model.mesh_faceadr[i], model.mesh_facenum[i]
        verts = model.mesh_vert[va : va + vn].astype(np.float32)
        faces = model.mesh_face[fa : fa + fn].astype(np.uint32)
        meshes.append({
            "verts": base64.b64encode(verts.tobytes()).decode(),
            "faces": base64.b64encode(faces.tobytes()).decode(),
        })
    geoms = []
    for g in range(model.ngeom):
        base = {
            "body": int(model.geom_bodyid[g]),
            "pos": model.geom_pos[g].tolist(),
            "quat": model.geom_quat[g].tolist(),  # wxyz
        }
        gname = model.geom(g).name or ""
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            geoms.append({**base, "type": "mesh", "mesh": int(model.geom_dataid[g])})
        elif model.geom_rgba[g][3] > 0.5 and "foot" not in gname and "collision" not in gname:
            # visible primitives (tables, ball, props) — skip collision helpers
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
                geoms.append({**base, "type": "box", "size": model.geom_size[g].tolist(),
                              "rgba": model.geom_rgba[g].tolist()})
            elif model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
                geoms.append({**base, "type": "sphere", "size": model.geom_size[g].tolist(),
                              "rgba": model.geom_rgba[g].tolist()})
    return json.dumps({"nbody": model.nbody, "meshes": meshes, "geoms": geoms}).encode()


class Shared:
    def __init__(self, nbody):
        self.lock = threading.Lock()
        self.xpos = np.zeros((nbody, 3))
        self.xquat = np.zeros((nbody, 4))
        self.rx_hz = 0.0
        self.rx_times = []

    def update(self, xpos, xquat, got_msg):
        now = time.time()
        with self.lock:
            self.xpos = xpos.copy()
            self.xquat = xquat.copy()
            if got_msg:
                self.rx_times.append(now)
            self.rx_times = [t for t in self.rx_times if now - t < 2.0]
            self.rx_hz = len(self.rx_times) / 2.0

    def frame_json(self):
        with self.lock:
            return json.dumps({
                "xpos": np.round(self.xpos, 4).tolist(),
                "xquat": np.round(self.xquat, 4).tolist(),
                "rx_hz": round(self.rx_hz, 1),
            })


def sim_thread(host, port, topic, rig, shared):
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt_string(zmq.SUBSCRIBE, topic)
    sock.setsockopt(zmq.CONFLATE, 1)
    sock.connect(f"tcp://{host}:{port}")
    print(f"[zmq] subscribed tcp://{host}:{port} topic={topic!r}")
    smooth_l = np.zeros(4)
    smooth_r = np.zeros(4)
    alpha = 0.35
    while True:
        got = False
        if sock.poll(timeout=15):
            msg = parse_pose(sock.recv(zmq.NOBLOCK), topic.encode())
            sj = msg.get("smpl_joints")
            if sj is not None:
                left, right = retarget(np.asarray(sj)[-1])
                smooth_l = alpha * left + (1 - alpha) * smooth_l
                smooth_r = alpha * right + (1 - alpha) * smooth_r
                got = True
        rig.set_arms(smooth_l, smooth_r)  # runs mj_forward
        shared.update(rig.data.xpos, rig.data.xquat, got)
        time.sleep(1 / 60)


def serve_http(directory, manifest, port):
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=directory, **kw)

        def do_GET(self):
            if self.path == "/manifest.json":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(manifest)))
                self.end_headers()
                self.wfile.write(manifest)
            else:
                if self.path in ("/", ""):
                    self.path = "/robot.html"
                super().do_GET()

        def log_message(self, *a):
            pass

    httpd = http.server.ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"[http] robot viewer at http://localhost:{port}")
    httpd.serve_forever()


async def ws_handler(ws, shared, fps=30.0):
    try:
        while True:
            await ws.send(shared.frame_json())
            await asyncio.sleep(1.0 / fps)
    except websockets.ConnectionClosed:
        pass


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="172.18.130.111")
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--topic", default="pose")
    ap.add_argument("--http-port", type=int, default=8001)
    ap.add_argument("--ws-port", type=int, default=8766)
    args = ap.parse_args()

    rig = G1Rig()
    manifest = build_manifest(rig.model)
    print(f"[manifest] {len(manifest) / 1e6:.1f} MB, {rig.model.nmesh} meshes")
    shared = Shared(rig.model.nbody)
    threading.Thread(target=sim_thread, args=(args.host, args.port, args.topic, rig, shared), daemon=True).start()
    threading.Thread(target=serve_http, args=(str(Path(__file__).parent), manifest, args.http_port), daemon=True).start()
    async with websockets.serve(lambda ws: ws_handler(ws, shared), "0.0.0.0", args.ws_port):
        print(f"[ws] listening on :{args.ws_port}")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())

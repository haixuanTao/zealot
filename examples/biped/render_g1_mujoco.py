#!/usr/bin/env python3
"""Render a zealot G1 rollout JSON through MuJoCo's offscreen mesh renderer.

Plays base pose + 12 joint angles back as qpos on mujoco_playground's G1
(feet-only) mesh model with a flat-terrain scene, tracking camera with a slow
orbit, piped to ffmpeg.

Usage: render_g1_mujoco.py rollout.json out.mp4 [probe.png]
"""
import json
import os
import subprocess
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

SRC, OUT = sys.argv[1], sys.argv[2]
PROBE = sys.argv[3] if len(sys.argv) > 3 else None

P = os.path.expanduser(
    "~/miniforge3/envs/mjx/lib/python3.12/site-packages/mujoco_playground/_src/locomotion/g1/xmls"
)
model = mujoco.MjModel.from_xml_path(f"{P}/scene_mjx_feetonly_flat_terrain.xml")
model.vis.global_.offwidth, model.vis.global_.offheight = 1280, 720
data = mujoco.MjData(model)
# Start from the model's home keyframe so non-driven joints (waist/arms) hold
# a natural pose instead of zeros.
if model.nkey > 0:
    mujoco.mj_resetDataKeyframe(model, data, 0)

d = json.load(open(SRC))
base = np.array(d["base"])      # (T, 7) pos + quat XYZW (zealot convention)
joints = np.array(d["joints"])  # (T, 12) canonical order == joint_names
names = d["joint_names"]
resets = set(d.get("resets", []))
dt = d["dt"]
T = len(base)

fa = model.joint("floating_base_joint").qposadr[0]
jadr = [model.joint(n).qposadr[0] for n in names]

W, H = 1280, 720
renderer = mujoco.Renderer(model, height=H, width=W)
cam = mujoco.MjvCamera()
cam.distance, cam.elevation = 2.4, -14.0

fps = round(1.0 / dt)
ff = None
if PROBE is None:
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", OUT],
        stdin=subprocess.PIPE,
    )

flash = 0
for t in range(T):
    data.qpos[fa:fa + 3] = base[t, :3]
    x, y, z, w = base[t, 3:7]
    data.qpos[fa + 3:fa + 7] = [w, x, y, z]  # MuJoCo quats are WXYZ
    for k, a in enumerate(jadr):
        data.qpos[a] = joints[t, k]
    mujoco.mj_forward(model, data)
    cam.lookat[:] = [base[t, 0], base[t, 1], 0.72]
    cam.azimuth = 135.0 + 12.0 * np.sin(2 * np.pi * t / T)
    if t in resets:
        flash = 5  # brief red tint marks an episode reset (fall)
    renderer.update_scene(data, camera=cam)
    px = renderer.render()
    if flash > 0:
        px = px.copy()
        px[:, :, 0] = np.minimum(255, px[:, :, 0].astype(int) + 70).astype(np.uint8)
        flash -= 1
    if PROBE is not None:
        import PIL.Image
        PIL.Image.fromarray(px).save(PROBE)
        print(f"probe frame {t} → {PROBE}")
        sys.exit(0)
    ff.stdin.write(px.tobytes())

ff.stdin.close()
ff.wait()
print(f"wrote {OUT} ({T / fps:.1f}s @ {fps}fps, {len(resets)} resets)")

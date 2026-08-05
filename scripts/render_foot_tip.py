#!/usr/bin/env python3
"""Render the foot-tip-stability JSON dump from `foot_tip_stability` into an mp4.

The Rust example simulates a single foot-shaped box on a ground plane at several
initial tilts. We replay those per-step poses through MuJoCo's offscreen renderer
with an inline MJCF (single box + plane + light), so the user can SEE whether the
box rests on an edge or on a face.

Usage: python3 render_foot_tip.py /tmp/foot_tip_poses.json /tmp/foot_tip.mp4
"""
import json
import os
import subprocess
import sys
import tempfile

import imageio.v2 as imageio
import mujoco
import numpy as np

src = sys.argv[1] if len(sys.argv) > 1 else "/tmp/foot_tip_poses.json"
out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/foot_tip.mp4"
W, H = 960, 540
RENDER_FPS = 50  # subsample the 200 Hz sim → 50 fps mp4 (every 4th step)

with open(src) as f:
    d = json.load(f)
dt = d["dt"]
he = d["half_extents"]  # [hx, hy, hz]
scenarios = d["scenarios"]  # name -> list of [x,y,z,qx,qy,qz,qw]
print(f"dt={dt:.5f}s   foot half-extents={tuple(he)}   {len(scenarios)} scenarios")

# Inline MJCF: gridded ground plane, single box driven via freejoint qpos. Z-up
# world matching the Rust sim. The checker-pattern floor anchors the eye to a
# visible "this is the ground" surface so the box doesn't look like it's floating.
XML = f"""
<mujoco>
  <visual><global offwidth="{W}" offheight="{H}"/></visual>
  <asset>
    <texture type="2d" name="grid" builtin="checker" rgb1="0.45 0.50 0.58" rgb2="0.22 0.25 0.30" width="512" height="512"/>
    <material name="grid_mat" texture="grid" texrepeat="40 40" specular="0" shininess="0"/>
  </asset>
  <worldbody>
    <light name="top" pos="0 0 3" dir="0 0 -1" diffuse="0.85 0.85 0.85"/>
    <light name="side" pos="0.5 0.5 0.6" dir="-0.4 -0.4 -0.6" diffuse="0.4 0.4 0.4"/>
    <geom name="floor" type="plane" size="2 2 0.05" material="grid_mat"/>
    <body name="foot" pos="0 0 0">
      <freejoint name="foot_free"/>
      <geom name="foot_geom" type="box" size="{he[0]} {he[1]} {he[2]}" rgba="0.95 0.55 0.20 1"/>
    </body>
  </worldbody>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)
free_adr = model.joint("foot_free").qposadr[0]

# Static camera, low elevation so the floor is read as a horizon: the box's
# resting-on-ground state is unambiguous, not floating against a flat slab.
cam = mujoco.MjvCamera()
cam.distance = 0.65
cam.elevation = -8.0
cam.azimuth = 35.0
cam.lookat[:] = [0.0, 0.0, 0.05]

renderer = mujoco.Renderer(model, height=H, width=W)
tmp = tempfile.mkdtemp()

step_step = max(1, int(round(1.0 / (dt * RENDER_FPS))))  # every Nth sim step
print(f"rendering: 1 frame every {step_step} sim steps ({RENDER_FPS} fps)")

# Big text overlays via opencv-style PIL fallback — we use mujoco's nothing-fancy
# renderer + just write the scenario+timestamp into the file name and rely on a
# burned-in caption via ffmpeg drawtext (post-render).
frame_paths = []  # (path, caption) per frame
frame_idx = 0
for name, poses in scenarios.items():
    print(f"  [{name}] {len(poses)} sim steps")
    for k in range(0, len(poses), step_step):
        p = poses[k]
        # Rust quaternion is xyzw; MuJoCo wants wxyz.
        data.qpos[free_adr : free_adr + 3] = [p[0], p[1], p[2]]
        data.qpos[free_adr + 3 : free_adr + 7] = [p[6], p[3], p[4], p[5]]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam)
        pix = renderer.render()
        path = os.path.join(tmp, f"f{frame_idx:06d}.png")
        imageio.imwrite(path, pix)
        t = k * dt
        # Caption = scenario name + sim time. We pass via a sidecar text file
        # mapping frame index → caption, drawn by ffmpeg in a single pass.
        frame_paths.append((frame_idx, f"{name}    t={t:5.2f}s"))
        frame_idx += 1

# Write the ffmpeg drawtext filter — one drawtext per frame range. Simpler: use
# `subtitles=` style is heavy; instead use `drawtext` with frame-indexed text via
# `metadata` or a single text file with timing. Easiest: pre-burn the caption into
# each PNG with PIL.
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

try:
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 28)
except OSError:
    font = ImageFont.load_default()
for idx, caption in frame_paths:
    p = os.path.join(tmp, f"f{idx:06d}.png")
    img = Image.open(p)
    draw = ImageDraw.Draw(img)
    # Black drop-shadow for legibility on the slate floor.
    for dx, dy in [(-2, -2), (2, -2), (-2, 2), (2, 2)]:
        draw.text((24 + dx, 16 + dy), caption, fill=(0, 0, 0), font=font)
    draw.text((24, 16), caption, fill=(255, 255, 255), font=font)
    img.save(p)

subprocess.run(
    [
        "ffmpeg", "-y", "-framerate", str(RENDER_FPS), "-i", f"{tmp}/f%06d.png",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out,
    ],
    check=True,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
print(f"wrote {out}  ({frame_idx} frames @ {RENDER_FPS}fps = {frame_idx/RENDER_FPS:.1f}s)")

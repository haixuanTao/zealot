#!/usr/bin/env python3
"""Render a LeRobot-bipedal rollout (JSON from `biped_render`) WITH its meshes by
playing the qpos trajectory back through MuJoCo's offscreen renderer.

The rollout's base pose + joint angles become MuJoCo qpos; we point the model's
meshdir at the URDF asset dir (where the 53 referenced STLs live), stub the one
missing visual mesh, add a ground plane + light, render each frame, and pipe to
ffmpeg.

Usage: python3 render_biped_mujoco.py /tmp/biped_rollout.json /tmp/biped_mesh.mp4
"""
import json
import os
import subprocess
import sys
import tempfile

import mujoco
import numpy as np
import trimesh

src = sys.argv[1] if len(sys.argv) > 1 else "/tmp/biped_rollout.json"
out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/biped_mesh.mp4"
HOME = os.path.expanduser("~")
XML = f"{HOME}/Documents/work/lerobot-humanoid-design/to_real_robot/RL_policy/robot.xml"
ASSETS = f"{HOME}/Documents/work/lerobot-humanoid-design/urdf/bipedal_plateform/urdf/assets"
W, H = 720, 720

with open(src) as f:
    d = json.load(f)
base = np.array(d["base"])          # (T, 7) pos + quat xyzw
joints = np.array(d["joints"])      # (T, 12)
jnames = d["joint_names"]
resets = set(d["resets"])
dt = d["dt"]
T = len(base)
print(f"{T} frames, {joints.shape[1]} joints")

# --- assemble a renderable model: meshdir → URDF assets, stub the missing mesh,
#     add a ground plane + light + a tracking-friendly setup. ---
tmp = tempfile.mkdtemp()
mesh_tmp = os.path.join(tmp, "meshes")
os.makedirs(mesh_tmp)
for fn in os.listdir(ASSETS):
    if fn.endswith(".stl"):
        os.symlink(os.path.join(ASSETS, fn), os.path.join(mesh_tmp, fn))
# The one visual mesh absent from the URDF assets — stub with a tiny box.
trimesh.creation.box((0.02, 0.02, 0.02)).export(os.path.join(mesh_tmp, "torso_mesh.stl"))

xml = open(XML).read()
xml = xml.replace('meshdir="assets"', f'meshdir="{mesh_tmp}"')
# Enlarge the offscreen framebuffer (default 640) and inject a ground plane +
# light right after <worldbody>.
visual = f'  <visual><global offwidth="{W}" offheight="{H}"/></visual>\n  '
inject = """
    <light name="top" pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.30 0.34 0.42 1"/>
"""
xml = xml.replace("<worldbody>", visual + "<worldbody>" + inject, 1)

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# qpos addresses: free joint (base) then each hinge by name.
free_adr = model.joint("torso_subassembly_freejoint").qposadr[0]
hinge_adr = {n: int(model.joint(n).qposadr[0]) for n in jnames}

# STATIC, FLAT tripod camera (optional azimuth override in deg). It does NOT move
# and does NOT tilt — fixed lookat framing the whole walk path, elevation 0
# (horizontal / eye-level), so the robot simply walks across a still frame.
AZIMUTH = float(sys.argv[3]) if len(sys.argv) > 3 else 90.0
cx = float((base[:, 0].min() + base[:, 0].max()) / 2)
cy = float((base[:, 1].min() + base[:, 1].max()) / 2)
span = float(max(np.ptp(base[:, 0]), np.ptp(base[:, 1])))

renderer = mujoco.Renderer(model, height=H, width=W)
cam = mujoco.MjvCamera()
cam.distance = max(3.0, span * 1.4 + 2.0)  # far enough to frame the whole walk
cam.elevation = 0.0  # absolutely flat (horizontal)
cam.azimuth = AZIMUTH
cam.lookat[:] = [cx, cy, 0.45]  # fixed — does not follow the robot

frames_dir = os.path.join(tmp, "frames")
os.makedirs(frames_dir)
import imageio.v2 as imageio  # noqa: E402

for i in range(T):
    p = base[i, :3]
    qx, qy, qz, qw = base[i, 3:7]
    data.qpos[free_adr : free_adr + 3] = p
    data.qpos[free_adr + 3 : free_adr + 7] = [qw, qx, qy, qz]  # MuJoCo wxyz
    for k, n in enumerate(jnames):
        data.qpos[hinge_adr[n]] = joints[i, k]
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=cam)  # camera fixed; only the robot moves
    pix = renderer.render()
    imageio.imwrite(os.path.join(frames_dir, f"f{i:05d}.png"), pix)
    if i % 100 == 0:
        print(f"  rendered {i}/{T}")

fps = int(round(1.0 / dt))
subprocess.run(
    [
        "ffmpeg", "-y", "-framerate", str(fps), "-i", f"{frames_dir}/f%05d.png",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out,
    ],
    check=True,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
print(f"wrote {out}  ({T/fps:.1f}s @ {fps}fps)")

#!/usr/bin/env python3
"""Render a LeRobot-bipedal rollout (JSON from the `biped_render` example) as a
3D mp4: the kinematic skeleton (links as segments), feet as markers, a ground
grid, and an orbiting camera. Z-up (matches the MuJoCo model / the real robot).

Usage: python3 render_biped.py /tmp/biped_rollout.json /tmp/biped.mp4
"""
import json
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation

src = sys.argv[1] if len(sys.argv) > 1 else "/tmp/biped_rollout.json"
out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/biped.mp4"

with open(src) as f:
    d = json.load(f)

frames = np.array(d["frames"])          # (T, n_bodies, 3)  world (x, y, z), Z-up
edges = d["edges"]                        # [[parent, child], ...]
feet = d["feet"]
resets = set(d["resets"])
dt = d["dt"]
T = len(frames)
print(f"{T} frames, {frames.shape[1]} bodies, {len(edges)} links")

# Follow the torso (body 0) in X/Y so the robot stays centered as it moves.
torso_xy = frames[:, 0, :2]

fig = plt.figure(figsize=(7, 7), facecolor="#0e1320")
ax = fig.add_subplot(111, projection="3d")
ax.set_facecolor("#0e1320")
GRID = "#2b3a5e"
SPAN = 0.9  # half-width of the view box around the torso


def draw_ground(cx, cy):
    g = np.linspace(-SPAN, SPAN, 9)
    for x in g:
        ax.plot([cx + x, cx + x], [cy - SPAN, cy + SPAN], [0, 0], color=GRID, lw=0.6)
    for y in g:
        ax.plot([cx - SPAN, cx + SPAN], [cy + y, cy + y], [0, 0], color=GRID, lw=0.6)


def update(i):
    ax.cla()
    ax.set_facecolor("#0e1320")
    pts = frames[i]
    cx, cy = torso_xy[i]
    draw_ground(cx, cy)
    fallen = i in resets
    link_col = "#ff6b6b" if fallen else "#5ad1ff"
    # Links.
    for a, b in edges:
        xs = [pts[a, 0], pts[b, 0]]
        ys = [pts[a, 1], pts[b, 1]]
        zs = [pts[a, 2], pts[b, 2]]
        ax.plot(xs, ys, zs, color=link_col, lw=3.0, solid_capstyle="round")
    # Joints + feet.
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color="#cfe8ff", s=14, depthshade=True)
    for fi in feet:
        ax.scatter(pts[fi, 0], pts[fi, 1], pts[fi, 2], color="#ffd166", s=55)
    # Torso marker.
    ax.scatter(pts[0, 0], pts[0, 1], pts[0, 2], color="#ffffff", s=40)

    ax.set_xlim(cx - SPAN, cx + SPAN)
    ax.set_ylim(cy - SPAN, cy + SPAN)
    ax.set_zlim(0, 1.0)
    ax.set_box_aspect((1, 1, 0.55))
    ax.view_init(elev=12, azim=(i * 0.4) % 360 - 60)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_axis_off()
    ax.text2D(0.02, 0.95, f"LeRobot bipedal — t={i*dt:5.2f}s", transform=ax.transAxes,
              color="#cfe8ff", fontsize=11, family="monospace")
    if fallen:
        ax.text2D(0.02, 0.90, "reset (fell)", transform=ax.transAxes, color="#ff6b6b",
                  fontsize=10, family="monospace")
    return []


fps = int(round(1.0 / dt))
anim = FuncAnimation(fig, update, frames=T, interval=1000 * dt, blit=False)
writer = FFMpegWriter(fps=fps, bitrate=3500)
anim.save(out, writer=writer, dpi=110)
print(f"wrote {out}  ({T/fps:.1f}s @ {fps}fps)")

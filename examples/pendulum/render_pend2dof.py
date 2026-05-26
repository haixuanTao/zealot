#!/usr/bin/env python3
"""Render the 2-DOF (ball-joint) inverted-pendulum CSV (from `pendulum2dof`) as a
3D mp4: shaded rod bar, a floor grid + reference axes for depth, the balanced
cone around vertical, and a fading tip trail showing the 3D wobble. Orbiting cam.

Physics is Y-up; map physics (x,y,z) -> plot (x,z,y).
Usage: python3 render_pend2dof.py /tmp/pend2dof.csv /tmp/pend2dof.mp4
"""
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation, FFMpegWriter

csv = sys.argv[1] if len(sys.argv) > 1 else "/tmp/pend2dof.csv"
out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/pend2dof.mp4"
DT = 1.0 / 60.0
HX, HY, HZ = 1.0, 0.12, 0.12
TOL = np.radians(25.0)
LIGHT = np.array([0.35, 0.9, 0.25]); LIGHT /= np.linalg.norm(LIGHT)
AMB, DIF = 0.45, 0.55
GRID = "#4a5d86"

d = np.genfromtxt(csv, delimiter=",", names=True)
n = len(d)

C = np.array([[sx * HX, sy * HY, sz * HZ] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
FACES = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]
NORMALS = np.array([[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1.0]])
P = [0, 2, 1]  # physics (x,y,z) -> plot (x,z,y)


def quat_R(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


# precompute tip path (rod far end) in plot coords for the trail
tips = []
for i in range(n):
    R = quat_R(d["qx"][i], d["qy"][i], d["qz"][i], d["qw"][i])
    com = np.array([d["x"][i], d["y"][i], d["z"][i]])
    tip = com + R @ np.array([HX, 0, 0])
    tips.append(tip[P])
tips = np.array(tips)

fig = plt.figure(figsize=(6.6, 6.0))
fig.patch.set_facecolor("#0e1117")
ax = fig.add_subplot(projection="3d")
fig.suptitle("2-DOF inverted pendulum on nexus GPU physics (ball joint, balanced)",
             color="white", fontsize=11)
L = 2.6


def setup(step):
    ax.cla()
    ax.set_facecolor("#0e1117")
    ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_zlim(0, 3.0)
    ax.set_box_aspect((1, 1, 0.85))
    ax.set_axis_off()
    ax.view_init(elev=18, azim=-55 + step * 0.18)
    # floor grid
    g, ticks = 2.4, np.linspace(-2.4, 2.4, 9)
    # wireframe floor grid (no fill, so the lines are never painted over)
    for t in ticks:
        ax.plot([-g, g], [t, t], [0, 0], color=GRID, lw=1.0)
        ax.plot([t, t], [-g, g], [0, 0], color=GRID, lw=1.0)
    # reference axes at the pivot: X red, Y(up) green, Z blue
    ax.plot([0, 1.0], [0, 0], [0, 0], color="#d9534f", lw=2)
    ax.plot([0, 0], [0, 1.0], [0, 0], color="#4f8fd9", lw=2)   # physics Z -> plot Y
    ax.plot([0, 0], [0, 0], [0, 1.0], color="#5cb85c", lw=2)   # physics Y(up) -> plot Z
    # balanced cone around +up (within TOL)
    aa = np.linspace(0, 2 * np.pi, 28)
    r = 2.3 * np.sin(TOL)
    h = 2.3 * np.cos(TOL)
    ring = [(r * np.cos(a), r * np.sin(a), h) for a in aa]
    cone = [[(0, 0, 0), ring[k], ring[k + 1]] for k in range(len(ring) - 1)]
    ax.add_collection3d(Poly3DCollection(cone, facecolor="green", alpha=0.06, edgecolor="none"))
    ax.scatter([0], [0], [0], c="white", s=20)


def draw_bar(i, rgb):
    R = quat_R(d["qx"][i], d["qy"][i], d["qz"][i], d["qw"][i])
    com = np.array([d["x"][i], d["y"][i], d["z"][i]])
    w = (R @ C.T).T + com
    wn = (R @ NORMALS.T).T
    verts, cols = [], []
    for fi, f in enumerate(FACES):
        shade = AMB + DIF * max(0.0, float(wn[fi] @ LIGHT))
        verts.append([w[k][P] for k in f])
        cols.append(np.clip(np.array(rgb) * shade, 0, 1))
    ax.add_collection3d(Poly3DCollection(verts, facecolors=cols, edgecolor=(0, 0, 0, 0.4), linewidths=0.3))
    near = (com - R @ np.array([HX, 0, 0]))[P]
    ax.plot([0, near[0]], [0, near[1]], [0, near[2]], color=np.array(rgb) * 0.7, lw=2)


def update(i):
    setup(i)
    tilt = np.degrees(np.arccos(np.clip(d["y"][i] / max(1e-6, np.hypot(d["x"][i], np.hypot(d["y"][i], d["z"][i]))), -1, 1)))
    up = np.radians(tilt) < TOL
    # tip trail (last 50 frames), fading
    k0 = max(0, i - 50)
    tr = tips[k0:i + 1]
    if len(tr) > 1:
        ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], color="#ffd24a", lw=1.4, alpha=0.9)
    draw_bar(i, (0.2, 0.85, 0.35) if up else (0.95, 0.55, 0.2))
    ax.text2D(0.02, 0.95, f"t={i * DT:4.2f}s   tilt={tilt:4.1f}°", transform=ax.transAxes,
              color="white", family="monospace", fontsize=9)


anim = FuncAnimation(fig, update, frames=n, interval=1000 * DT, blit=False)
anim.save(out, writer=FFMpegWriter(fps=60, bitrate=3000), savefig_kwargs={"facecolor": "#0e1117"})
print(f"wrote {out} ({n} frames, {n * DT:.1f}s)")

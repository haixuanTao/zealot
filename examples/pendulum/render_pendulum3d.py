#!/usr/bin/env python3
"""Render the inverted-pendulum trajectory CSV (full 6-DOF, from
`inverted_pendulum record`) as a 3D mp4 in the same shaded style as the boxes:
the rod is drawn as an oriented bar, controlled vs baseline side by side, with a
pivot, an upright target, and the balanced wedge. Camera slowly orbits.

Physics is Y-up; map physics (x,y,z) -> plot (x,z,y). Usage:
  python3 render_pendulum3d.py /tmp/pendulum_traj.csv /tmp/pendulum3d.mp4
"""
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation, FFMpegWriter

csv = sys.argv[1] if len(sys.argv) > 1 else "/tmp/pendulum_traj.csv"
out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/pendulum3d.mp4"
DT = 1.0 / 60.0
TOL = 25.0                       # balanced tolerance (deg)
HX, HY, HZ = 1.0, 0.12, 0.12     # rod half-extents (matches the cuboid collider)
LIGHT = np.array([0.35, 0.9, 0.25]); LIGHT /= np.linalg.norm(LIGHT)
AMB, DIF = 0.45, 0.55

d = np.genfromtxt(csv, delimiter=",", names=True)
n = len(d)

C = np.array([[sx * HX, sy * HY, sz * HZ] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
FACES = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]
NORMALS = np.array([[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1.0]])
PERM = [0, 2, 1]


def quat_R(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def bar(ax, p, q, rgb):
    R = quat_R(*q)
    w = (R @ C.T).T + p
    wn = (R @ NORMALS.T).T
    verts, cols = [], []
    for fi, f in enumerate(FACES):
        shade = AMB + DIF * max(0.0, float(wn[fi] @ LIGHT))
        verts.append([w[i][PERM] for i in f])
        cols.append(np.clip(np.array(rgb) * shade, 0, 1))
    ax.add_collection3d(Poly3DCollection(verts, facecolors=cols,
                                         edgecolor=(0, 0, 0, 0.4), linewidths=0.3))
    # thin arm from pivot (origin) to the rod's near end
    near = (p - R @ np.array([HX, 0, 0]))[PERM]
    ax.plot([0, near[0]], [0, near[1]], [0, near[2]], color=np.array(rgb) * 0.7, lw=2)


def angle_deg(x, y):
    return np.degrees(np.arctan2(y, x))


fig = plt.figure(figsize=(9.5, 5.2))
fig.patch.set_facecolor("#0e1117")
fig.suptitle("Inverted pendulum on nexus GPU physics (3D) — start 45° off vertical",
             color="white", fontsize=11)
axc = fig.add_subplot(1, 2, 1, projection="3d")
axb = fig.add_subplot(1, 2, 2, projection="3d")
time_txt = fig.text(0.5, 0.04, "", ha="center", color="white", family="monospace", fontsize=10)


def setup(ax, title):
    ax.set_facecolor("#0e1117")
    ax.set_xlim(-2.6, 2.6); ax.set_ylim(-2.6, 2.6); ax.set_zlim(-2.6, 2.6)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.set_title(title, color="white", fontsize=10)
    # balanced wedge (in the swing plane, plot depth=0) + upright target
    th = np.radians(np.linspace(90 - TOL, 90 + TOL, 24))
    wedge = [[(0, 0, 0)] + [(2.3 * np.cos(a), 0, 2.3 * np.sin(a)) for a in th]]
    ax.add_collection3d(Poly3DCollection(wedge, facecolor="green", alpha=0.10, edgecolor="none"))
    ax.plot([0, 0], [0, 0], [0, 2.4], ls=":", c="0.6", lw=1)
    ax.scatter([0], [0], [0], c="white", s=18)


def update(i):
    for ax, px, py, pz, q, title, rgb in (
        (axc, "cx", "cy", "cz", ("cqx", "cqy", "cqz", "cqw"), "Controlled (velocity-PD)", (0.25, 0.5, 0.95)),
        (axb, "bx", "by", "bz", ("bqx", "bqy", "bqz", "bqw"), "Baseline (no motor)", (0.9, 0.3, 0.25)),
    ):
        ax.cla()
        setup(ax, title)
        ax.view_init(elev=16, azim=-60 + i * 0.12)
        p = np.array([d[px][i], d[py][i], d[pz][i]])
        qq = tuple(d[k][i] for k in q)
        ang = angle_deg(p[0], p[1])
        up = abs(ang - 90.0) < TOL
        bar(ax, p, qq, (0.2, 0.85, 0.35) if up else rgb)
        ax.text2D(0.02, 0.93, f"θ={ang:6.1f}°", transform=ax.transAxes,
                  family="monospace", color="white", fontsize=9)
    time_txt.set_text(f"t = {i * DT:4.2f} s")


anim = FuncAnimation(fig, update, frames=n, interval=1000 * DT, blit=False)
anim.save(out, writer=FFMpegWriter(fps=60, bitrate=3000),
          savefig_kwargs={"facecolor": "#0e1117"})
print(f"wrote {out} ({n} frames, {n * DT:.1f}s)")

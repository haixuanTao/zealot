#!/usr/bin/env python3
"""Render the inverted-pendulum trajectory CSV (from `inverted_pendulum record`)
to an mp4: controlled (balances) vs baseline (falls), side by side.

Usage: python3 render_video.py /tmp/pendulum_traj.csv /tmp/pendulum.mp4
"""
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

csv = sys.argv[1] if len(sys.argv) > 1 else "/tmp/pendulum_traj.csv"
out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/pendulum.mp4"
DT = 1.0 / 60.0
TOL_DEG = 25.0  # "balanced" tolerance used by the sim

d = np.genfromtxt(csv, delimiter=",", names=True)
cx, cy, bx, by = d["cx"], d["cy"], d["bx"], d["by"]
n = len(cx)


def angle_deg(x, y):
    return np.degrees(np.arctan2(y, x))


fig, (axc, axb) = plt.subplots(1, 2, figsize=(9, 4.8))
fig.suptitle("Inverted pendulum on nexus GPU physics — start 45° off vertical", fontsize=12)

panels = []
for ax, title, color in ((axc, "Controlled (velocity-PD)", "tab:blue"),
                         (axb, "Baseline (no motor)", "tab:red")):
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)
    # upright target + balanced wedge
    ax.plot([0, 0], [0, 2.4], ls=":", c="gray", lw=1)
    th = np.radians(np.linspace(90 - TOL_DEG, 90 + TOL_DEG, 30))
    ax.fill(np.r_[0, 2.4 * np.cos(th)], np.r_[0, 2.4 * np.sin(th)],
            color="green", alpha=0.08)
    ax.plot(0, 0, "ko", ms=6)  # pivot
    rod, = ax.plot([], [], "-", c=color, lw=4, solid_capstyle="round")
    bob, = ax.plot([], [], "o", c=color, ms=12)
    txt = ax.text(-2.8, 2.5, "", fontsize=9, family="monospace")
    panels.append((rod, bob, txt))

time_txt = fig.text(0.5, 0.02, "", ha="center", fontsize=10, family="monospace")


def upright(x, y):
    return abs(angle_deg(x, y) - 90.0) < TOL_DEG


def update(i):
    for (rod, bob, txt), (x, y) in zip(panels, ((cx, cy), (bx, by))):
        rod.set_data([0, x[i]], [0, y[i]])
        bob.set_data([x[i]], [y[i]])
        ok = upright(x[i], y[i])
        bob.set_color("tab:green" if ok else ("tab:blue" if x is cx else "tab:red"))
        txt.set_text(f"θ={angle_deg(x[i], y[i]):6.1f}°\n{'UPRIGHT' if ok else '       '}")
    time_txt.set_text(f"t = {i * DT:4.2f} s   (step {i}/{n - 1})")
    return [a for p in panels for a in p[:2]]


anim = FuncAnimation(fig, update, frames=n, interval=1000 * DT, blit=False)
anim.save(out, writer=FFMpegWriter(fps=60, bitrate=2400))
print(f"wrote {out} ({n} frames, {n * DT:.1f}s)")

#!/usr/bin/env python3
"""Replay a biped rollout JSON through the NATIVE nexus renderer -> mp4.

Replaces the matplotlib skeleton view. Differences that matter:

  * The ground is the TRUE terrain geometry. For the Step family the dumped
    height grid is binary, and this script re-emits it as box cells (flat tops
    + vertical walls) exactly like the trainer's collision mesh -- matplotlib's
    `plot_surface` interpolates between samples, which renders a vertical wall
    as a ramp and made the step edge look sloped when it is not.
  * The robot is drawn as capsules along the skeleton edges (the rollout
    records link POSITIONS only, so link meshes are not reconstructable), with
    balls at the joints.
  * Rendering is nexus's raytracer via the headless viewer, streamed through
    ffmpeg. Poses are driven per frame with `NexusState.set_body_pose` (added
    for this) + `viewer.sync_rapier` -- nothing is simulated.

Usage:
    render_biped_nexus_native.py <rollout.json> <out.mp4> [--spp N]
"""
import json
import os
import subprocess
import sys

import numpy as np
from nexus3d import (
    ColliderBuilder,
    NexusState,
    NexusViewer,
    Pose,
    RigidBodyBuilder,
    vec3,
    vec4,
)

IDENT = None  # Pose identity, set after import (needs vec3)


def visual(viewer, state, handle, col):
    # Headless inserts register no visuals; attach the collider's shape
    # explicitly so the raytracer actually draws it.
    viewer.insert_visual_shape(0, handle, col.shared_shape(), IDENT)

SRC = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/biped_native.mp4"
SPP = int(sys.argv[sys.argv.index("--spp") + 1]) if "--spp" in sys.argv else 2
W, H = 1280, 720
CAPS_R = 0.024
BALL_R = 0.032

with open(SRC) as f:
    d = json.load(f)
frames = np.array(d["frames"], dtype=np.float32)  # (T, n_bodies, 3)
edges = d["edges"]
dt = d["dt"]
T, NB = frames.shape[0], frames.shape[1]
print(f"{T} frames, {NB} bodies, {len(edges)} edges")

viewer = NexusViewer(W, H, True)
viewer.init_backend()
viewer.set_up_axis(vec3(0.0, 0.0, 1.0))
viewer.set_draw_ui(False)
viewer.set_raytracer_samples_per_frame(SPP)
viewer.add_directional_light(vec3(-0.4, 0.3, -1.0))

state = NexusState()
IDENT = Pose(vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0))

# --- terrain: true geometry, not an interpolated surface -------------------
terr = d.get("terrain")
if terr and any(h != 0.0 for h in terr["heights"]):
    n = int(round(2 * terr["half"] / terr["hs"])) + 1
    hs = terr["hs"]
    x0 = terr["cx"] - terr["half"]
    y0 = terr["cy"] - terr["half"]
    Hf = np.array(terr["heights"], dtype=np.float32).reshape(n, n)
    verts, tris = [], []

    def quad(a, b, c, dd):
        base = len(verts)
        verts.extend([a, b, c, dd])
        tris.append([base, base + 1, base + 2])
        tris.append([base, base + 2, base + 3])

    binaryish = len(set(np.round(Hf.flatten(), 3))) <= 6
    for j in range(n - 1):
        for i in range(n - 1):
            xa, xb = x0 + i * hs, x0 + (i + 1) * hs
            ya, yb = y0 + j * hs, y0 + (j + 1) * hs
            if binaryish:
                # Box-cell: flat top at the cell's own height + vertical walls
                # to differing neighbours. Matches the trainer's Step mesh.
                h = float(Hf[j, i])
                quad([xa, ya, h], [xb, ya, h], [xb, yb, h], [xa, yb, h])
                if i + 1 < n - 1 and abs(Hf[j, i + 1] - h) > 1e-6:
                    h2 = float(Hf[j, i + 1])
                    lo, hi = min(h, h2), max(h, h2)
                    quad([xb, ya, hi], [xb, yb, hi], [xb, yb, lo], [xb, ya, lo])
                if j + 1 < n - 1 and abs(Hf[j + 1, i] - h) > 1e-6:
                    h2 = float(Hf[j + 1, i])
                    lo, hi = min(h, h2), max(h, h2)
                    quad([xa, yb, hi], [xb, yb, hi], [xb, yb, lo], [xa, yb, lo])
            else:
                # Smooth families: the piecewise-linear surface IS the truth.
                quad(
                    [xa, ya, float(Hf[j, i])],
                    [xb, ya, float(Hf[j, i + 1])],
                    [xb, yb, float(Hf[j + 1, i + 1])],
                    [xa, yb, float(Hf[j + 1, i])],
                )
    body = state.insert_body(RigidBodyBuilder.fixed().build())
    col = ColliderBuilder.trimesh(verts, tris).build()
    state.insert_collider_in(0, col, body)
    visual(viewer, state, body, col)
else:
    ground = state.insert_body(RigidBodyBuilder.fixed().build())
    col = ColliderBuilder.cuboid(50.0, 50.0, 0.05).translation(vec3(0, 0, -0.05)).build()
    state.insert_collider_in(0, col, ground)
    visual(viewer, state, ground, col)

# --- robot: capsules along edges, balls at joints --------------------------
ball_handles = []
for _ in range(NB):
    h = state.insert_body(RigidBodyBuilder.kinematic_position_based().build())
    col = ColliderBuilder.ball(BALL_R).build()
    state.insert_collider_in(0, col, h)
    visual(viewer, state, h, col)
    ball_handles.append(h)
edge_handles = []
for _ in edges:
    h = state.insert_body(RigidBodyBuilder.kinematic_position_based().build())
    col = ColliderBuilder.capsule_z(0.1, CAPS_R).build()
    state.insert_collider_in(0, col, h)
    visual(viewer, state, h, col)
    edge_handles.append(h)

state.finalize(viewer)


def quat_between_z(a, b):
    """Quaternion (x,y,z,w) rotating +Z onto (b-a)."""
    v = b - a
    ln = np.linalg.norm(v)
    if ln < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0]), 1e-4
    v = v / ln
    z = np.array([0.0, 0.0, 1.0])
    c = float(np.dot(z, v))
    if c > 0.9999:
        return np.array([0.0, 0.0, 0.0, 1.0]), ln
    if c < -0.9999:
        return np.array([1.0, 0.0, 0.0, 0.0]), ln
    ax = np.cross(z, v)
    s = np.sqrt((1 + c) * 2)
    return np.array([ax[0] / s, ax[1] / s, ax[2] / s, s / 2]), ln


ff = subprocess.Popen(
    ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
     "-s", f"{W}x{H}", "-r", str(round(1.0 / dt)), "-i", "-",
     "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", OUT],
    stdin=subprocess.PIPE,
)

for t in range(T):
    P = frames[t]
    for k in range(NB):
        state.set_body_pose(ball_handles[k], vec3(*P[k]), vec4(0, 0, 0, 1), 0)
    for m, (pa, pb) in enumerate(edges):
        a, b = P[pa], P[pb]
        q, ln = quat_between_z(a, b)
        mid = (a + b) / 2
        state.set_body_pose(edge_handles[m], vec3(*mid), vec4(*q), 0)
        # capsule_z half-height is fixed at build; scale visually via z... not
        # supported per-frame, so length mismatch is accepted: capsules are a
        # skeleton indication, joints carry the accuracy.
    base = P[0]
    viewer.set_camera(vec3(base[0] + 2.6, base[1] - 2.6, base[2] + 1.2),
                      vec3(base[0], base[1], base[2] - 0.2))
    viewer.sync_rapier(state, 0)
    viewer.render_frame()
    rgb = np.asarray(viewer.snap_rgb())
    ff.stdin.write(np.ascontiguousarray(rgb[:, :, :3]).tobytes())

ff.stdin.close()
ff.wait()
print(f"wrote {OUT} ({T} frames @ {round(1.0/dt)} fps, native nexus renderer)")

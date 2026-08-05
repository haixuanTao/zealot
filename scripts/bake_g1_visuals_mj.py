#!/usr/bin/env python3
"""Bake G1 visual meshes in RAW menagerie/MuJoCo body frames for the browser
MuJoCo comparison demo (`website/static/bench/three_mujoco_bench.html`).

Unlike `bake_g1_visuals.py` (which rotates meshes into zealot's converted,
axis-normalized link frames for the nexus demos), the MuJoCo page drives
bodies straight from `data.xpos/xquat` of the playground MJCF — so the meshes
must stay in the ORIGINAL body-local frames. Each visual mesh geom of the
menagerie 29-DOF G1 is transformed by its geom-local pose (already
body-local), decimated, and grouped per body name; body names match the
playground model (same lineage).

Output format = bake_g1_visuals.py's (u32 count; per entry: u8 name_len,
name, f32*4 rgba, u32 nv, u32 nf, f32 verts, u32 tris), written to
website/static/bench/g1_visuals_mj29.bin.

Run: python3 scripts/bake_g1_visuals_mj.py [decimate_ratio=0.15]
"""

import struct
import sys
from pathlib import Path

import fast_simplification
import mujoco
import numpy as np

WORK = Path.home() / "Documents/work"
XML = WORK / "mujoco_menagerie/unitree_g1/g1.xml"
OUT = WORK / "zealot/website/static/bench/g1_visuals_mj29.bin"
DECIMATE = float(sys.argv[1]) if len(sys.argv) > 1 else 0.15


def quat_to_mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, dtype=float))
    return m.reshape(3, 3)


def main():
    model = mujoco.MjModel.from_xml_path(str(XML))
    per_body: dict[str, list] = {}
    n_geoms = 0
    for g in range(model.ngeom):
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        # Visual-only geoms (contype 0 conaffinity 0 group 2 in menagerie);
        # take every mesh geom — the collision set is primitive anyway.
        mid = model.geom_dataid[g]
        va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
        verts = model.mesh_vert[va:va + vn].astype(np.float64)
        faces = model.mesh_face[fa:fa + fn].astype(np.int64)
        # Compiled mesh_vert are in the mesh's own frame with mesh_pos/quat
        # already folded into geom_pos/quat by the compiler.
        R = quat_to_mat(model.geom_quat[g])
        verts = verts @ R.T + model.geom_pos[g]
        body = model.body(model.geom_bodyid[g]).name
        rgba = model.geom_rgba[g].copy()
        per_body.setdefault(body, []).append((rgba, verts, faces))
        n_geoms += 1

    entries = []
    for body, geoms in per_body.items():
        # Merge the body's mesh geoms into one buffer, then decimate.
        vs, fs, off = [], [], 0
        rgba = geoms[0][0]
        for _, v, f in geoms:
            vs.append(v)
            fs.append(f + off)
            off += len(v)
        v = np.concatenate(vs).astype(np.float32)
        f = np.concatenate(fs).astype(np.uint32)
        v2, f2 = fast_simplification.simplify(v, f, 1.0 - DECIMATE)
        entries.append((body, rgba, v2.astype(np.float32), f2.astype(np.uint32)))

    buf = bytearray(struct.pack("<I", len(entries)))
    total_v = total_f = 0
    for name, rgba, v, f in entries:
        nb = name.encode()
        buf += struct.pack("<B", len(nb)) + nb
        buf += struct.pack("<4f", *rgba)
        buf += struct.pack("<II", len(v), len(f))
        buf += v.tobytes()
        buf += f.astype("<u4").tobytes()
        total_v += len(v)
        total_f += len(f)

    OUT.write_bytes(buf)
    print(
        f"baked {n_geoms} mesh geoms -> {len(entries)} bodies, "
        f"{total_v} verts / {total_f} tris, {len(buf) / 1e6:.2f} MB -> {OUT}"
    )
    for name, _, v, f in entries[:5]:
        print(f"  {name}: {len(v)} v / {len(f)} t")


if __name__ == "__main__":
    sys.exit(main())

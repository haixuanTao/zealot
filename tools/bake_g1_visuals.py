#!/usr/bin/env python3
"""Bake the Unitree G1 visual meshes into zealot's converted 12-DOF link
frames for the website demo (`examples/biped/g1_web.rs`).

Source: mujoco_menagerie/unitree_g1/g1.xml (same g1_description lineage as the
unitree_rl_gym model tools/convert_unitree_biped.py converted). Every visual
mesh geom is mapped to its nearest KEPT ancestor body (the 13 bodies of
assets/robots/unitree_g1_12dof.xml — upper body fuses into the pelvis at
qpos = 0), transformed into that body's frame, then rotated into zealot's
axis-normalized link frame (R = quat taking ẑ to the body's hinge axis;
converted-local = R⁻¹ · original-local — same math as convert_unitree_biped).

Self-check: recomputes each kept body's converted pos/quat and compares with
the checked-in XML before writing anything.

Output (little-endian, examples/biped/assets/g1_visuals.bin):
  u32 n_entries, then per entry:
    u8 name_len, name bytes (kept-body name),
    f32*4 rgba, u32 n_verts, u32 n_tris,
    f32*3*n_verts positions, u32*3*n_tris indices.

Run: python3 tools/bake_g1_visuals.py [decimate_ratio=0.15]
"""

import struct
import sys
from pathlib import Path

import fast_simplification
import mujoco
import numpy as np

WORK = Path.home() / "Documents/work"
MENAGERIE = WORK / "mujoco_menagerie/unitree_g1"
# Which converted model to bake for: `12dof` (upper body fused into the
# pelvis) or `29dof` (torso/arms articulated; wrist pitch/yaw welded).
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "12dof"
OUT = WORK / f"zealot/examples/biped/assets/g1_visuals_{VARIANT}.bin"
CONVERTED_XML = WORK / f"zealot/assets/robots/unitree_g1_{VARIANT}.xml"

import xml.etree.ElementTree as ET

# Kept bodies = exactly the converted model's body set, in document order.
KEPT = [e.get("name") for e in ET.parse(CONVERTED_XML).getroot().iter("body")]
assert KEPT[0] == "pelvis", KEPT[:3]

DECIMATE = float(sys.argv[2]) if len(sys.argv) > 2 else 0.15


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_rotate(q, v):
    qv = np.array([0.0, *v])
    return quat_mul(quat_mul(q, qv), quat_conj(q))[1:]


def quat_to_mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, dtype=float))
    return m.reshape(3, 3)


def axis_to_z_quat(axis):
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    q = np.zeros(4)
    mujoco.mju_quatZ2Vec(q, a)
    return q


def main():
    m = mujoco.MjModel.from_xml_path(str(MENAGERIE / "g1.xml"))
    d = mujoco.MjData(m)
    # Fused bodies bake at the "stand" keyframe (natural arms — Unitree's
    # joint ZERO is the elbows-bent-90° CAD pose), with the (unsimulated)
    # fingers lightly curled. Kept bodies' own geoms are qpos-independent.
    if m.nkey:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    for j in range(m.njnt):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        if n and "hand" in n:
            lo, hi = m.jnt_range[j]
            d.qpos[m.jnt_qposadr[j]] = lo + 0.6 * (hi - lo)
    mujoco.mj_forward(m, d)

    name_to_body = {
        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b): b for b in range(m.nbody)
    }
    for k in KEPT:
        assert k in name_to_body, f"menagerie g1.xml missing body {k}"

    # Axis-normalizing rotation per kept body (identity for the pelvis) — from
    # the kept body's own hinge axis, exactly like convert_unitree_biped.
    # Menagerie bodies with multiple joints (e.g. a floating base variant)
    # would need care; here every kept non-root body carries one hinge.
    R = {"pelvis": np.array([1.0, 0.0, 0.0, 0.0])}
    for k in KEPT[1:]:
        b = name_to_body[k]
        assert m.body_jntnum[b] == 1, k
        j = m.body_jntadr[b]
        R[k] = axis_to_z_quat(m.jnt_axis[j])

    # ---- Self-check: recompute the converted body pos/quat, compare to XML.
    xml_bodies = {
        e.get("name"): e for e in ET.parse(CONVERTED_XML).getroot().iter("body")
    }
    kept_parent = {}
    for k in KEPT[1:]:
        p = m.body_parentid[name_to_body[k]]
        pname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, p)
        assert pname in KEPT, f"{k} parent {pname} not kept"
        kept_parent[k] = pname
        rp_inv = quat_conj(R[pname])
        pos = quat_rotate(rp_inv, m.body_pos[name_to_body[k]])
        quat = quat_mul(quat_mul(rp_inv, m.body_quat[name_to_body[k]]), R[k])
        want_pos = np.array([float(x) for x in xml_bodies[k].get("pos").split()])
        want_quat = np.array(
            [float(x) for x in (xml_bodies[k].get("quat") or "1 0 0 0").split()]
        )
        assert np.allclose(pos, want_pos, atol=1e-4), (k, pos, want_pos)
        if np.dot(quat, want_quat) < 0:
            quat = -quat
        assert np.allclose(quat, want_quat, atol=1e-4), (k, quat, want_quat)
    print(f"frame self-check OK for {len(KEPT)} bodies")

    # Nearest kept ancestor for every body (world → None).
    def kept_ancestor(b):
        while b != 0:
            n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
            if n in KEPT:
                return n
            b = m.body_parentid[b]
        return None

    # ---- Collect mesh geoms grouped by (kept body, rgba).
    groups = {}  # (kept, rgba tuple) -> [(verts, faces)]
    total_in = 0
    for g in range(m.ngeom):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        # Visual-only geoms (menagerie: group 2 visual / group 3 collision).
        if m.geom_contype[g] != 0 or m.geom_conaffinity[g] != 0:
            continue
        body = m.geom_bodyid[g]
        kept = kept_ancestor(body)
        if kept is None:
            continue
        ka = name_to_body[kept]
        # Geom world pose (qpos=0) → kept-body local → converted frame.
        gq = np.zeros(4)
        mujoco.mju_mat2Quat(gq, d.geom_xmat[g].reshape(9))
        a_inv = quat_conj(d.xquat[ka])
        p_local = quat_rotate(a_inv, d.geom_xpos[g] - d.xpos[ka])
        q_local = quat_mul(a_inv, gq)
        r_inv = quat_conj(R[kept])
        p_conv = quat_rotate(r_inv, p_local)
        q_conv = quat_mul(r_inv, q_local)

        mid = m.geom_dataid[g]
        va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
        verts = m.mesh_vert[va : va + vn].astype(np.float64)
        faces = m.mesh_face[fa : fa + fn].astype(np.int64)
        total_in += fn
        rot = quat_to_mat(q_conv)
        verts = verts @ rot.T + p_conv
        rgba = tuple(np.round(m.geom_rgba[g], 4))
        groups.setdefault((kept, rgba), []).append((verts, faces))

    # ---- Merge per group, decimate, write.
    out = bytearray()
    entries = []
    total_out = 0
    for (kept, rgba), parts in sorted(groups.items()):
        vs, fs, off = [], [], 0
        for v, f in parts:
            vs.append(v)
            fs.append(f + off)
            off += len(v)
        v = np.vstack(vs)
        f = np.vstack(fs)
        if DECIMATE < 1.0 and len(f) > 500:
            v2, f2 = fast_simplification.simplify(
                v.astype(np.float32), f.astype(np.int64), 1.0 - DECIMATE
            )
            v, f = np.asarray(v2, dtype=np.float64), np.asarray(f2, dtype=np.int64)
        total_out += len(f)
        entries.append((kept, rgba, v.astype("<f4"), f.astype("<u4")))

    out += struct.pack("<I", len(entries))
    for kept, rgba, v, f in entries:
        nb = kept.encode()
        out += struct.pack("<B", len(nb)) + nb
        out += struct.pack("<4f", *rgba)
        out += struct.pack("<II", len(v), len(f))
        out += v.tobytes()
        out += f.tobytes()
    OUT.write_bytes(bytes(out))
    print(
        f"wrote {OUT} ({len(out) / 1e6:.2f} MB, {len(entries)} groups, "
        f"{total_in} → {total_out} tris @ ratio {DECIMATE})"
    )


if __name__ == "__main__":
    main()

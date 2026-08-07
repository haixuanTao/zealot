#!/usr/bin/env python3
"""Bake the converted 12-DOF G1 (assets/robots/unitree_g1_12dof.xml) into a
JSON the three.js + rapier.js benchmark can consume: per body, the WORLD rest
pose of the body origin (accumulated down the tree), the parent index, the
mass, the world hinge axis (converted models hinge about local +Z), the home
angle, and the sole capsules (local frame). The visual meshes come from
examples/biped/assets/g1_visuals_12dof.bin (same body names, converted local
frames — a mesh node rotated by the body's rest quat renders correctly on an
identity-rotation rapier body).

Output: website/static/bench/g1_model.json
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


def quat_to_mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, dtype=float))
    return m.reshape(3, 3)


def mat_to_quat(m):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(m, dtype=float).reshape(9))
    return q

ZEALOT = Path(__file__).resolve().parent.parent
XML = ZEALOT / "assets/robots/unitree_g1_12dof.xml"
OUT = ZEALOT / "website/static/bench/g1_model.json"

HOME = {
    "left_hip_pitch_joint": -0.1, "left_knee_joint": 0.3, "left_ankle_pitch_joint": -0.2,
    "right_hip_pitch_joint": -0.1, "right_knee_joint": 0.3, "right_ankle_pitch_joint": -0.2,
}
# g1_29dof_agile leg gains (kp, kd, effort) — what the v7 policy trained with
# (mirrors sim2sim_g1_mujoco.py leg_gains; first fragment wins).
GAINS = [
    ("knee", 200.0, 5.0, 139.0),
    ("hip", 100.0, 2.5, 88.0),
    ("ankle_roll", 20.0, 0.1, 50.0),
    ("ankle", 20.0, 0.2, 50.0),
]


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_rotate(q, v):
    qv = np.array([0.0, *v])
    qc = np.array([q[0], -q[1], -q[2], -q[3]])
    return quat_mul(quat_mul(q, qv), qc)[1:]


def parse_vec(s, n, default):
    return np.array([float(x) for x in s.split()]) if s else np.array(default[:n])


bodies = []


def walk(el, parent_idx, ppos, pquat):
    pos = parse_vec(el.get("pos"), 3, [0, 0, 0])
    quat = parse_vec(el.get("quat"), 4, [1, 0, 0, 0])
    wpos = ppos + quat_rotate(pquat, pos)
    wquat = quat_mul(pquat, quat)
    inertial = el.find("inertial")
    joint = el.find("joint")
    caps = []
    for g in el.findall("geom"):
        f = [float(x) for x in g.get("fromto").split()]
        caps.append({"a": f[0:3], "b": f[3:6], "r": float(g.get("size"))})
    name = el.get("name")
    kp, kd, eff = 0.0, 0.0, 0.0
    if joint is not None:
        for frag, p, d, e in GAINS:
            if frag in name:
                kp, kd, eff = p, d, e
                break
    # Inertial properties in the IDENTITY-rotation body frame the benchmark
    # uses (rest world rotation baked into attachments): com = R·ipos;
    # inertia M -> R M R^T, then principal decomposition.
    ipos = parse_vec(inertial.get("pos"), 3, [0, 0, 0])
    fi = [float(x) for x in inertial.get("fullinertia").split()]
    M = np.array([[fi[0], fi[3], fi[4]], [fi[3], fi[1], fi[5]], [fi[4], fi[5], fi[2]]])
    R = quat_to_mat(wquat)
    Mw = R @ M @ R.T
    evals, evecs = np.linalg.eigh(Mw)
    if np.linalg.det(evecs) < 0:
        evecs[:, 0] = -evecs[:, 0]
    bodies.append({
        "name": name,
        "parent": parent_idx,
        "pos": [round(float(x), 6) for x in wpos],
        # wxyz
        "quat": [round(float(x), 6) for x in wquat],
        "mass": float(inertial.get("mass")),
        "com": [round(float(x), 6) for x in quat_rotate(wquat, ipos)],
        "inertia": [round(float(x), 8) for x in evals],
        # wxyz — principal frame in the identity-rotation body frame
        "inertiaQuat": [round(float(x), 6) for x in mat_to_quat(evecs)],
        "joint": joint.get("name") if joint is not None else None,
        # Converted models hinge about the CHILD's local +Z → world axis at rest.
        "axis": [round(float(x), 6) for x in quat_rotate(wquat, [0, 0, 1])] if joint is not None else None,
        "home": HOME.get(joint.get("name"), 0.0) if joint is not None else 0.0,
        "kp": kp,
        "kd": kd,
        "effort": eff,
        "range": [float(x) for x in joint.get("range").split()] if joint is not None else None,
        "capsules": caps,
    })
    me = len(bodies) - 1
    for child in el.findall("body"):
        walk(child, me, wpos, wquat)


root = ET.parse(XML).getroot()
pelvis = root.find("worldbody/body")
walk(pelvis, -1, np.zeros(3), np.array([1.0, 0, 0, 0]))
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(bodies))
print(f"wrote {OUT} ({len(bodies)} bodies)")
for b in bodies[:3]:
    print(" ", b["name"], b["pos"], b["joint"])

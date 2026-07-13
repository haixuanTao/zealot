#!/usr/bin/env python3
"""Convert Unitree humanoid descriptions to zealot's MJCF dialect.

zealot's env (`examples/biped/biped_env_nexus.rs::parse_mjcf`) consumes a narrow
MJCF dialect, shaped by the LeRobot bipedal export it was built against:

- every joint is a hinge about the CHILD body's **local +Z** axis
  (`GenericJointBuilder` frees `AngZ` with `local_frame2 = IDENTITY`);
- `<inertial pos=... mass=... fullinertia="Ixx Iyy Izz Ixy Ixz Iyz"/>` in the
  LINK frame (no inertial-frame quat);
- foot soles are `<geom class="collision" fromto=... size=.../>` capsules, and
  ONLY feet carry them (that is how the env finds the foot links);
- ranges in radians.

Unitree models use per-joint `axis` attributes (X for roll, Y for pitch, Z for
yaw) and principal-frame `diaginertia + quat` inertials, so this script:

1. loads the source model with MuJoCo (MJCF or URDF);
2. for URDF sources, first strips visual/collision geometry and converts every
   non-leg revolute joint to `fixed`, then lets MuJoCo `fusestatic` weld the
   waist/arms/head into the pelvis (mass + inertia composed exactly);
3. rotates every jointed body's frame so the hinge axis becomes local +Z
   (children and inertials are counter-rotated, so world-frame physics is
   IDENTICAL — verified by FK below);
4. emits the 1-root + 12-leg-body tree in the zealot dialect, synthesizing the
   sole capsules from the source's foot collision spheres (G1) or the foot
   STL's bounding box (H2 Plus);
5. VERIFIES the emitted file: python FK over the emitted XML (the same
   parent-compose zealot does) must reproduce MuJoCo's `xpos`/`xquat` for every
   kept body at qpos0, the world-frame inertia tensor of every body must match,
   and total mass must match.

Outputs land in `assets/robots/` and are what `RobotSpec::{unitree_g1,
unitree_h2_plus}` point at. Run from the zealot repo root:

    python3 tools/convert_unitree_biped.py

Requires `mujoco` (tested with 3.7.0) and the unitree_ros sparse checkout at
~/Documents/work/unitree_ros (see docs in the RobotSpec constructors).
"""

import os
import re
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

WORK = Path(os.environ["HOME"]) / "Documents/work"
UNITREE = WORK / "unitree_ros/robots"
OUT_DIR = Path(__file__).resolve().parent.parent / "assets/robots"

LEG_JOINT_RE = re.compile(r"^(left|right)_(hip_(pitch|roll|yaw)|knee|ankle_(pitch|roll))_joint$")


# ---------------------------------------------------------------- quaternions
# wxyz convention throughout (matches MuJoCo).
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
    """Quaternion R (wxyz) with R·ẑ = axis (minimal arc)."""
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    z = np.array([0.0, 0.0, 1.0])
    q = np.zeros(4)
    mujoco.mju_quatZ2Vec(q, a)  # rotation taking ẑ to `a`
    # sanity
    assert np.allclose(quat_rotate(q, z), a, atol=1e-9), (a, quat_rotate(q, z))
    return q


def fmt(x):
    return f"{x:.9g}"


def fmt_vec(v):
    return " ".join(fmt(x) for x in v)


# ------------------------------------------------------------------- sources
def load_g1():
    """G1: official legs-only 12-DOF MJCF (unitree_rl_gym), meshes stripped."""
    src = UNITREE / "g1_description/g1_12dof.xml"
    root = ET.parse(src).getroot()
    for asset in root.findall("asset"):
        root.remove(asset)
    for parent in root.iter():
        for geom in list(parent.findall("geom")):
            if geom.get("type") == "mesh":
                parent.remove(geom)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    return model, src


def load_h2_plus(tmp_dir):
    """H2 Plus: URDF with non-leg joints fixed, geometry stripped, fusestatic."""
    src = UNITREE / "h2_plus/H2_Plus.urdf"
    root = ET.parse(src).getroot()
    # The upstream URDF ships the world link but its floating base joint is
    # commented out, leaving `world` an orphan (two roots) AND the pelvis
    # jointless — `fusestatic` would weld the whole robot into the world. Drop
    # the orphan link and re-add the floating base explicitly.
    for link in list(root.findall("link")):
        if link.get("name") == "world":
            root.remove(link)
    ET.SubElement(root, "link", {"name": "world"})
    fb = ET.SubElement(root, "joint", {"name": "floating_base_joint", "type": "floating"})
    ET.SubElement(fb, "parent", {"link": "world"})
    ET.SubElement(fb, "child", {"link": "pelvis"})
    for link in root.findall("link"):
        for tag in ("visual", "collision"):
            for el in list(link.findall(tag)):
                link.remove(el)
    for joint in root.findall("joint"):
        name = joint.get("name")
        if joint.get("type") in ("revolute", "continuous") and not LEG_JOINT_RE.match(name):
            joint.set("type", "fixed")
    # MuJoCo URDF extension: weld the (now jointless) upper body into the pelvis.
    # The upstream URDF already carries a <mujoco> element (meshdir etc.) —
    # replace it, ours is the only compiler config the stripped model needs.
    for el in list(root.findall("mujoco")):
        root.remove(el)
    mj = ET.SubElement(root, "mujoco")
    ET.SubElement(mj, "compiler", {"fusestatic": "true", "balanceinertia": "true"})
    tmp = Path(tmp_dir) / "h2_plus_stripped.urdf"
    tmp.write_text(ET.tostring(root, encoding="unicode"))
    model = mujoco.MjModel.from_xml_path(str(tmp))
    return model, src


# ------------------------------------------------------------- foot geometry
def capsules_from_extents(lo, hi, r):
    """Two sole-edge capsules from a (near-planar) footprint box, axis-generic.

    The axis normalization rotates foot frames (e.g. G1's sole plane ends up at
    x=const, not z=const), so the sole's thickness axis is detected as the box's
    thinnest extent — the same rule zealot's Rust foot-collider builder uses to
    reconstruct the foot from these capsules' centerlines.
    """
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    ext = hi - lo
    t = int(np.argmin(ext))
    f = [a for a in range(3) if a != t]
    long_ax, wide_ax = (f[0], f[1]) if ext[f[0]] >= ext[f[1]] else (f[1], f[0])
    tc = (lo[t] + hi[t]) / 2.0
    caps = []
    for w in (lo[wide_ax], hi[wide_ax]):
        a, b = np.zeros(3), np.zeros(3)
        a[t] = b[t] = tc
        a[wide_ax] = b[wide_ax] = w
        a[long_ax], b[long_ax] = lo[long_ax], hi[long_ax]
        caps.append((a, b, r))
    return caps


def g1_sole_capsules(model, body_id, r_inv_quat):
    """Sole capsules from the 4 collision corner spheres on the foot link."""
    pts, radius = [], None
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != body_id:
            continue
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_SPHERE:
            continue
        if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
            continue
        pts.append(model.geom_pos[g].copy())
        radius = float(model.geom_size[g][0])
    assert len(pts) == 4, f"expected 4 sole spheres, got {len(pts)}"
    pts = np.array([quat_rotate(r_inv_quat, p) for p in pts])
    return capsules_from_extents(pts.min(axis=0), pts.max(axis=0), radius)


def stl_vertices(path):
    data = path.read_bytes()
    n = struct.unpack_from("<I", data, 80)[0]
    assert len(data) == 84 + n * 50, f"{path} is not binary STL"
    rows = np.frombuffer(data[84:], dtype=np.uint8).reshape(n, 50)
    return np.frombuffer(rows[:, 12:48].tobytes(), dtype="<f4").reshape(n * 3, 3)


def h2_sole_capsules(body_name, r_inv_quat):
    """Sole capsules from the foot STL: slice the bottom 2r of the mesh (the
    sole surface), take its footprint box, inset by r so the capsule SURFACE
    ends at the mesh edge."""
    side = "left" if body_name.startswith("left") else "right"
    verts = stl_vertices(UNITREE / f"h2_plus/meshes/{side}_ankle_pitch_link.stl")
    r = 0.005  # same sole capsule radius the G1 model uses
    zmin = float(verts[:, 2].min())
    sole = verts[verts[:, 2] < zmin + 2 * r]
    lo, hi = sole.min(axis=0).astype(float), sole.max(axis=0).astype(float)
    # Inset footprint by r; keep the vertical slice as-is (capsule bottom ≈ zmin).
    lo[:2] += r
    hi[:2] -= r
    corners = np.array(
        [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    )
    corners = np.array([quat_rotate(r_inv_quat, c) for c in corners])
    return capsules_from_extents(corners.min(axis=0), corners.max(axis=0), r)


# ------------------------------------------------------------------- emitter
def convert(model, sole_capsules_fn, out_path, robot_name, foot_suffix, default_pose):
    nb = model.nbody
    # Kept bodies: everything except world (fusestatic already removed the rest).
    kept = list(range(1, nb))
    root_body = 1
    # Every non-root kept body must carry exactly one hinge.
    for b in kept:
        njnt = model.body_jntnum[b]
        if b == root_body:
            continue
        assert njnt == 1, f"body {b} has {njnt} joints"
        j = model.body_jntadr[b]
        assert model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
        assert np.allclose(model.jnt_pos[j], 0.0, atol=1e-9), "joint pos offset unsupported"

    def body_name(b):
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)

    # Axis-normalizing rotation per body (identity for the root).
    R = {root_body: np.array([1.0, 0.0, 0.0, 0.0])}
    for b in kept:
        if b == root_body:
            continue
        j = model.body_jntadr[b]
        R[b] = axis_to_z_quat(model.jnt_axis[j])

    children = {b: [c for c in kept if model.body_parentid[c] == b] for b in kept}

    lines = []

    def emit_body(b, indent):
        pad = "  " * indent
        pq, name = model.body_parentid[b], body_name(b)
        r_inv = quat_conj(R[b])
        if b == root_body:
            pos, quat = model.body_pos[b], model.body_quat[b]
        else:
            rp_inv = quat_conj(R[pq])
            pos = quat_rotate(rp_inv, model.body_pos[b])
            quat = quat_mul(quat_mul(rp_inv, model.body_quat[b]), R[b])
        lines.append(f'{pad}<body name="{name}" pos="{fmt_vec(pos)}" quat="{fmt_vec(quat)}">')
        # Inertial, rotated into the normalized link frame, as fullinertia.
        ipos = quat_rotate(r_inv, model.body_ipos[b])
        iquat = quat_mul(r_inv, model.body_iquat[b])
        rm = quat_to_mat(iquat)
        M = rm @ np.diag(model.body_inertia[b]) @ rm.T
        full = [M[0, 0], M[1, 1], M[2, 2], M[0, 1], M[0, 2], M[1, 2]]
        lines.append(
            f'{pad}  <inertial pos="{fmt_vec(ipos)}" mass="{fmt(model.body_mass[b])}" '
            f'fullinertia="{fmt_vec(full)}"/>'
        )
        if b == root_body:
            lines.append(f"{pad}  <freejoint/>")
        else:
            j = model.body_jntadr[b]
            jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            rng = model.jnt_range[j]
            dof = model.jnt_dofadr[j]
            damping = model.dof_damping[dof]
            armature = model.dof_armature[dof]
            # Zero damping/armature (URDF sources don't carry them) is OMITTED:
            # zealot's env prefers a present MJCF `damping` attr over the
            # RobotSpec value, and a literal 0 would silence the spec's.
            extra = ""
            if damping > 0.0:
                extra += f' damping="{fmt(damping)}"'
            if armature > 0.0:
                extra += f' armature="{fmt(armature)}"'
            lines.append(
                f'{pad}  <joint name="{jname}" axis="0 0 1" range="{fmt_vec(rng)}"{extra}/>'
            )
            if name.endswith(foot_suffix):
                for a, c, r in sole_capsules_fn(model, b, r_inv):
                    lines.append(
                        f'{pad}  <geom class="collision" fromto="{fmt_vec(a)} {fmt_vec(c)}" '
                        f'size="{fmt(r)}"/>'
                    )
        for c in children[b]:
            emit_body(c, indent + 1)
        lines.append(f"{pad}</body>")

    emit_body(root_body, 2)
    body_xml = "\n".join(lines)
    xml = f"""<!-- GENERATED by tools/convert_unitree_biped.py — do not hand-edit.
     Legs-only (12-DOF) {robot_name} in zealot's MJCF dialect: upper body fused
     into the pelvis, every hinge axis normalized to local +Z, link-frame
     fullinertia, sole contact capsules on the ankle_roll links. -->
<mujoco model="{robot_name}">
  <compiler angle="radian"/>
  <default>
    <default class="collision"/>
  </default>
  <worldbody>
{body_xml}
  </worldbody>
</mujoco>
"""
    out_path.write_text(xml)

    # ------------------------------------------------------------ verification
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)

    # Python FK over the EMITTED file, exactly like zealot's build_env_scene:
    # world = parent_world ∘ (pos, quat), joints at rest.
    troot = ET.parse(out_path).getroot()
    world = {}

    def fk(el, parent_pose):
        pos = np.array([float(x) for x in el.get("pos").split()])
        quat = np.array([float(x) for x in el.get("quat").split()])
        ppos, pquat = parent_pose
        wpos = ppos + quat_rotate(pquat, pos)
        wquat = quat_mul(pquat, quat)
        world[el.get("name")] = (wpos, wquat, el)
        for c in el.findall("body"):
            fk(c, (wpos, wquat))

    top = troot.find("worldbody/body")
    fk(top, (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])))

    assert len(world) == len(kept)
    mass_total = 0.0
    for b in kept:
        name = body_name(b)
        wpos, wquat, el = world[name]
        assert np.allclose(wpos, data.xpos[b], atol=1e-6), (name, wpos, data.xpos[b])
        # Orientation: must equal MuJoCo's frame composed with the normalizer.
        expect = quat_mul(data.xquat[b], R[b])
        dot = abs(np.dot(wquat, expect))
        assert dot > 1 - 1e-9, (name, wquat, expect)
        inertial = el.find("inertial")
        m = float(inertial.get("mass"))
        mass_total += m
        assert abs(m - model.body_mass[b]) < 1e-9
        # World-frame inertia + COM must match MuJoCo's.
        com_local = np.array([float(x) for x in inertial.get("pos").split()])
        com_w = wpos + quat_rotate(wquat, com_local)
        com_mj = data.xpos[b] + quat_to_mat(data.xquat[b]) @ model.body_ipos[b]
        assert np.allclose(com_w, com_mj, atol=1e-6), (name, com_w, com_mj)
        f = [float(x) for x in inertial.get("fullinertia").split()]
        M = np.array([[f[0], f[3], f[4]], [f[3], f[1], f[5]], [f[4], f[5], f[2]]])
        rw = quat_to_mat(wquat)
        Mw = rw @ M @ rw.T
        ri = quat_to_mat(quat_mul(data.xquat[b], model.body_iquat[b]))
        Mw_mj = ri @ np.diag(model.body_inertia[b]) @ ri.T
        assert np.allclose(Mw, Mw_mj, atol=1e-8), (name, Mw, Mw_mj)
    assert abs(mass_total - model.body_mass[1:].sum()) < 1e-6

    # Sole bottom in the world at qpos0 (straight legs, root at model default):
    # spawn height = root_z − sole_bottom_z puts the sole exactly on the ground.
    sole_bottom = np.inf
    for name, (wpos, wquat, el) in world.items():
        for g in el.findall("geom"):
            f = [float(x) for x in g.get("fromto").split()]
            r = float(g.get("size"))
            for p in (f[:3], f[3:]):
                z = (wpos + quat_rotate(wquat, np.array(p)))[2] - r
                sole_bottom = min(sole_bottom, z)
    root_z = float(model.body_pos[root_body][2])
    print(f"{robot_name}: {len(kept)} bodies, {mass_total:.2f} kg, FK/inertia verified")
    print(f"  straight-leg spawn height (sole on ground) = {root_z - sole_bottom:.4f}")

    # Foot-forward axis in the normalized foot frame (the env's foot-yaw obs
    # needs it): the source model's foot +X, expressed post-normalization.
    for b in kept:
        if body_name(b).endswith(foot_suffix) and body_name(b).startswith("left"):
            fwd = quat_rotate(quat_conj(R[b]), np.array([1.0, 0.0, 0.0]))
            wpos = world[body_name(b)][0]
            print(f"  foot link '{body_name(b)}': forward (local) = {fmt_vec(fwd)}, "
                  f"origin above sole at qpos0 = {wpos[2] - sole_bottom:.4f}")

    # Base height at the DEFAULT (bent-knee) pose — the reward target.
    data2 = mujoco.MjData(model)
    for jname, angle in default_pose.items():
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        assert j >= 0, jname
        data2.qpos[model.jnt_qposadr[j]] = angle
    mujoco.mj_kinematics(model, data2)
    lowest = np.inf
    for b in kept:
        if not body_name(b).endswith(foot_suffix):
            continue
        _, _, el = world[body_name(b)]
        rot = quat_to_mat(quat_mul(data2.xquat[b], R[b]))
        for g in el.findall("geom"):
            f = [float(x) for x in g.get("fromto").split()]
            r = float(g.get("size"))
            for p in (f[:3], f[3:]):
                z = (data2.xpos[b] + rot @ np.array(p))[2] - r
                lowest = min(lowest, z)
    print(f"  default-pose base height (reward target) = {root_z - lowest:.4f}")
    print(f"  wrote {out_path}")
    return root_z - sole_bottom


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    import tempfile

    # Bent-knee home pose, per unitree_rl_gym's G1 config (reused for H2 Plus —
    # no official RL config exists for it as of 2026-07).
    default_pose = {
        f"{side}_{j}_joint": a
        for side in ("left", "right")
        for j, a in [("hip_pitch", -0.1), ("knee", 0.3), ("ankle_pitch", -0.2)]
    }

    model, _ = load_g1()
    convert(
        model,
        g1_sole_capsules,
        OUT_DIR / "unitree_g1_12dof.xml",
        "unitree_g1_12dof",
        foot_suffix="ankle_roll_link",  # G1 chain: knee → ankle_pitch → ankle_roll
        default_pose=default_pose,
    )

    with tempfile.TemporaryDirectory() as tmp:
        model, _ = load_h2_plus(tmp)
        convert(
            model,
            lambda m, b, rq: h2_sole_capsules(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b), rq),
            OUT_DIR / "unitree_h2_plus_12dof.xml",
            "unitree_h2_plus_12dof",
            foot_suffix="ankle_pitch_link",  # H2 chain: knee → ankle_roll → ankle_pitch
            default_pose=default_pose,
        )


if __name__ == "__main__":
    main()

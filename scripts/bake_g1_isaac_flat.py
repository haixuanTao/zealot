#!/usr/bin/env python3
"""Bake the flattened G1 MJCF that sim2sim_g1_isaac.py imports to USD.

Needs an env with mujoco + mujoco_playground (the `mjx` conda env on this
box). Produces a self-contained MJCF from the playground feetonly flat scene:
  - defaults resolved (mujoco compile + save),
  - contact/sensor/keyframe/tendon/equality stripped (the MJCF->USD importer
    chokes on them; the harness sets pose/gains itself),
  - trained passive dynamics baked into the leg joints (damping 0.001,
    armature 0.02, frictionloss 0.1 — the zealot spec),
  - absolute meshdir (menagerie assets),
  - contype/conaffinity re-enabled on the floor + foot boxes: the MJX model
    is contact-PAIR-driven with contype 0 everywhere, so a naive import has
    ZERO collidable geoms and the robot free-falls through the floor.

Usage: python3 scripts/bake_g1_isaac_flat.py [out.xml]
       then S2S_MODEL_XML=<out.xml> scripts/sim2sim_g1_isaac.py ...
"""
import os
import re
import sys

import mujoco
import mujoco_playground

out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/_g1_isaac_flat.xml"
xdir = os.path.join(os.path.dirname(mujoco_playground.__file__), "_src/locomotion/g1/xmls")
mesh = os.path.abspath(os.path.join(xdir, "../../../../../mujoco_menagerie/unitree_g1/assets"))
m = mujoco.MjModel.from_xml_path(os.path.join(xdir, "scene_mjx_feetonly_flat_terrain.xml"))
mujoco.mj_saveLastXML(out, m)
s = open(out).read()

for tag in ("contact", "sensor", "keyframe", "tendon", "equality"):
    s = re.sub(rf"<{tag}>.*?</{tag}>", "", s, flags=re.S)

def fix_joint(mm):
    tag = mm.group(0)
    if "_hip_" in tag or "_knee_" in tag or "_ankle_" in tag:
        tag = re.sub(r'damping="[^"]*"', 'damping="0.001"', tag)
        tag = re.sub(r'armature="[^"]*"', 'armature="0.02"', tag)
        tag = re.sub(r'frictionloss="[^"]*"', 'frictionloss="0.1"', tag)
    return tag
s = re.sub(r"<joint [^>]*/>", fix_joint, s)

s = s.replace("<compiler ", f'<compiler meshdir="{mesh}" ', 1)
s = re.sub(r'file="[^"]*/([^"/]+\.(?:STL|stl|obj))"', r'file="\1"', s)

def enable(mm):
    tag = mm.group(0)
    tag = re.sub(r'\s*contype="[^"]*"', "", tag)
    tag = re.sub(r'\s*conaffinity="[^"]*"', "", tag)
    return tag.replace("<geom ", '<geom contype="1" conaffinity="1" ', 1)
s = re.sub(r'<geom[^>]*(?:name="floor"|class="foot")[^>]*/>', enable, s)

open(out, "w").write(s)
mujoco.MjModel.from_xml_path(out)  # must still compile
print(f"baked + validated: {out}")

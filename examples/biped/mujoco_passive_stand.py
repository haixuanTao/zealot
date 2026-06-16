#!/usr/bin/env python3
"""MuJoCo reference for zealot's `passive_stand`: the SAME LeRobot biped, zero
action (PD holding the all-zero nominal pose), spawned at z=0.72, watched for
fall. MuJoCo's solver is the known-good baseline — if the robot holds a stand
here but collapses in nexus, nexus's multibody/contact solver is the bug.

Matches zealot's config: per-joint kp/kd/armature (WBC-AGILE values), default
pose all-zero, 200 Hz physics / 50 Hz control (decimation 4), 10deg spawn pitch.

Run: python3 examples/biped/mujoco_passive_stand.py [seconds]
"""
import os
import sys
import tempfile
import numpy as np
import mujoco
import trimesh

HOME = os.path.expanduser("~")
XML = os.environ.get("BIPED_ROBOT_XML", f"{HOME}/tmp_eval/mjcf/robot.xml")
ASSETS = os.environ.get("BIPED_ASSETS", f"{HOME}/tmp_eval/mjcf/assets")
SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0

# Per-joint-family (kp, kd, armature) — identical to zealot LeRobotBipedal::family().
FAM = {
    "hipz": (30.0, 3.0, 0.0227),
    "hipx": (40.0, 3.0, 0.1333),
    "hipy": (60.0, 4.0, 0.1408),
    "knee": (60.0, 4.0, 0.1233),
    "ankley": (20.0, 1.5, 0.0299),
    "anklex": (20.0, 1.5, 0.0299),
}
JOINTS = [
    "hipz_right", "hipx_right", "hipy_right", "knee_right", "ankley_right", "anklex_right",
    "hipz_left", "hipx_left", "hipy_left", "knee_left", "ankley_left", "anklex_left",
]

def fam_of(name):
    for k in ("ankley", "anklex", "hipz", "hipx", "hipy", "knee"):
        if name.startswith(k):
            return FAM[k]
    raise KeyError(name)

# Build model: point meshdir at the STL assets (stub the one missing mesh),
# inject floor + light, force a 200 Hz timestep.
tmp = tempfile.mkdtemp()
mesh_tmp = os.path.join(tmp, "meshes")
os.makedirs(mesh_tmp)
for fn in os.listdir(ASSETS):
    if fn.endswith(".stl"):
        os.symlink(os.path.join(ASSETS, fn), os.path.join(mesh_tmp, fn))
# Stub torso_mesh only if the asset dir doesn't already provide it.
if not os.path.exists(os.path.join(mesh_tmp, "torso_mesh.stl")):
    trimesh.creation.box((0.02, 0.02, 0.02)).export(os.path.join(mesh_tmp, "torso_mesh.stl"))

xml = open(XML).read()
xml = xml.replace('meshdir="assets"', f'meshdir="{mesh_tmp}"')
inject = ('<option timestep="0.005"/>\n'
          '<worldbody>\n'
          '<light name="top" pos="0 0 3" dir="0 0 -1"/>\n'
          '<geom name="floor" type="plane" size="5 5 0.1"/>')
xml = xml.replace("<worldbody>", inject, 1)
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# Base freejoint: pos (0,0,0.72), quat wxyz = (0.9962,0,-0.0872,0) ≈ 10° pitch.
free_adr = model.joint("torso_subassembly_freejoint").qposadr[0]
data.qpos[free_adr:free_adr + 3] = [0.0, 0.0, 0.72]
data.qpos[free_adr + 3:free_adr + 7] = [0.9962, 0.0, -0.0872, 0.0]

# Per-joint qpos/dof addresses + gains; override armature to WBC values.
jinfo = []
for n in JOINTS:
    j = model.joint(n)
    kp, kd, arm = fam_of(n)
    dofadr = int(j.dofadr[0])
    model.dof_armature[dofadr] = arm
    jinfo.append((int(j.qposadr[0]), dofadr, kp, kd))

mujoco.mj_forward(model, data)
torso_bid = model.body("torso_subassembly").id

decim = 4
steps = int(round(SECONDS / 0.005))
print(f"MuJoCo passive stand: {XML.split('/')[-1]}, {SECONDS}s @200Hz, zero action (PD->nominal)")
print(f"{'t(s)':>6} {'torso_z':>9} {'min_foot_z':>10}")
tau = np.zeros(model.nv)
for s in range(steps):
    if s % decim == 0:  # 50 Hz control: PD to q_target = 0
        tau[:] = 0.0
        for (qadr, dadr, kp, kd) in jinfo:
            tau[dadr] = kp * (0.0 - data.qpos[qadr]) - kd * data.qvel[dadr]
    data.qfrc_applied[:] = 0.0
    data.qfrc_applied[[d for (_, d, _, _) in jinfo]] = tau[[d for (_, d, _, _) in jinfo]]
    mujoco.mj_step(model, data)
    if s % (decim * 10) == 0:
        tz = data.xpos[torso_bid][2]
        print(f"{s*0.005:6.2f} {tz:9.3f}")
tz = data.xpos[torso_bid][2]
print(f"FINAL torso_z={tz:.3f}  ({'STOOD' if tz > 0.5 else 'FELL'})")

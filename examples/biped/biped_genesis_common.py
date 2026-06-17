#!/usr/bin/env python3
"""Shared, framework-independent constants for the Genesis biped benchmark.

These mirror the LeRobot bipedal spec + sim config used by the Rust env so the
Genesis comparison is apples-to-apples. Sources (keep in sync):
  - robot spec / PD gains / armature: zealot-env/src/robots/lerobot_bipedal.rs
  - sim dt / decimation / spawn / solver iters: examples/biped/biped_env_nexus.rs
  - obs layout / command ranges: zealot-env/src/tasks/velocity_flat.rs

Imported by genesis_biped_env.py, bench_genesis_rollout.py,
cross_engine_eval_genesis.py.
"""
import os

HOME = os.path.expanduser("~")

# --- assets ---------------------------------------------------------------
# MJCF carries the model's damping / frictionloss / limits / masses / meshes;
# the Rust sims use it. URDF is the kinematic source. Genesis can load either.
ROBOT_XML = f"{HOME}/Documents/work/lerobot-humanoid-design/to_real_robot/RL_policy/robot.xml"
ROBOT_URDF = f"{HOME}/Documents/work/lerobot-humanoid-design/urdf/bipedal_plateform/urdf/robot.urdf"
ASSETS = f"{HOME}/Documents/work/lerobot-humanoid-design/urdf/bipedal_plateform/urdf/assets"

# --- canonical joint order (lerobot_bipedal.rs JOINT_NAMES, alphabetical) ---
JOINT_NAMES = [
    "anklex_left", "anklex_right", "ankley_left", "ankley_right",
    "hipx_left", "hipx_right", "hipy_left", "hipy_right",
    "hipz_left", "hipz_right", "knee_left", "knee_right",
]
NUM_JOINTS = 12

# Per-family (kp, kd, effort N·m, action_scale, armature kg·m²) — lerobot_bipedal.rs family().
_FAMILY = {
    "hipz": (30.0, 3.0, 88.0, 0.733, 0.0227),
    "hipx": (40.0, 3.0, 88.0, 0.55, 0.1333),
    "hipy": (60.0, 4.0, 88.0, 0.367, 0.1408),
    "knee": (60.0, 4.0, 88.0, 0.367, 0.1233),
    # ankley / anklex
    "ankle": (20.0, 1.5, 44.0, 0.55, 0.0299),
}


def _fam(name: str):
    for pref in ("hipz", "hipx", "hipy", "knee"):
        if name.startswith(pref):
            return _FAMILY[pref]
    return _FAMILY["ankle"]  # anklex / ankley


KP = [_fam(n)[0] for n in JOINT_NAMES]
KD = [_fam(n)[1] for n in JOINT_NAMES]
EFFORT = [_fam(n)[2] for n in JOINT_NAMES]
ACTION_SCALE = [_fam(n)[3] for n in JOINT_NAMES]
ARMATURE = [_fam(n)[4] for n in JOINT_NAMES]
DEFAULT_POS = [0.0] * NUM_JOINTS  # neutral home pose

# --- sim config (biped_env_nexus.rs) --------------------------------------
PHYS_DT = 1.0 / 200.0   # 0.005 s, 200 Hz physics
DECIMATION = 4          # physics steps per control step
CONTROL_DT = PHYS_DT * DECIMATION  # 0.02 s, 50 Hz control
SOLVER_ITERS = 8
GRAVITY = (0.0, 0.0, -9.81)  # Z-up
SPAWN_Z = 0.72
BASE_LINK = "torso_subassembly"
FOOT_LINKS = ["foot_left", "foot_right"]

# --- task (velocity_flat.rs) ----------------------------------------------
OBS_DIM = 43            # [last_action(12), command(4), joint_pos_rel(12), joint_vel(12), proj_gravity(3)]
ACT_DIM = 12
HIDDEN = [256, 256, 128]
# command [vx, vy, yaw_rate, aux=0]
CMD_VX = (-0.5, 0.5)
CMD_VY = (-0.3, 0.3)
CMD_YAW = (-0.2, 0.2)

# env-step sweep for the throughput bench
N_SWEEP = [512, 1024, 2048, 4096, 8192]

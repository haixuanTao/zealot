"""Unitree G1 (29-DOF EDU) low-level constants.

Motor index map for the `hg` message set (`rt/lowcmd` / `rt/lowstate`).
Joint names follow the official MJCF/URDF (mujoco_menagerie `unitree_g1`);
ranges are the MJCF joint limits in radians.
"""

# Hardware motor order: 12 legs, 3 waist, 7 left arm, 7 right arm.
G1_MOTOR_NAMES = [
    "left_hip_pitch_joint",      # 0
    "left_hip_roll_joint",       # 1
    "left_hip_yaw_joint",        # 2
    "left_knee_joint",           # 3
    "left_ankle_pitch_joint",    # 4
    "left_ankle_roll_joint",     # 5
    "right_hip_pitch_joint",     # 6
    "right_hip_roll_joint",      # 7
    "right_hip_yaw_joint",       # 8
    "right_knee_joint",          # 9
    "right_ankle_pitch_joint",   # 10
    "right_ankle_roll_joint",    # 11
    "waist_yaw_joint",           # 12
    "waist_roll_joint",          # 13
    "waist_pitch_joint",         # 14
    "left_shoulder_pitch_joint",  # 15
    "left_shoulder_roll_joint",   # 16
    "left_shoulder_yaw_joint",    # 17
    "left_elbow_joint",           # 18
    "left_wrist_roll_joint",      # 19
    "left_wrist_pitch_joint",     # 20
    "left_wrist_yaw_joint",       # 21
    "right_shoulder_pitch_joint",  # 22
    "right_shoulder_roll_joint",   # 23
    "right_shoulder_yaw_joint",    # 24
    "right_elbow_joint",           # 25
    "right_wrist_roll_joint",      # 26
    "right_wrist_pitch_joint",     # 27
    "right_wrist_yaw_joint",       # 28
]

G1_NUM_MOTORS = len(G1_MOTOR_NAMES)
MOTOR_INDEX = {name: i for i, name in enumerate(G1_MOTOR_NAMES)}

# MJCF joint limits (rad), motor order.
G1_JOINT_LIMITS = {
    "left_hip_pitch_joint": (-2.5307, 2.8798),
    "left_hip_roll_joint": (-0.5236, 2.9671),
    "left_hip_yaw_joint": (-2.7576, 2.7576),
    "left_knee_joint": (-0.087267, 2.8798),
    "left_ankle_pitch_joint": (-0.87267, 0.5236),
    "left_ankle_roll_joint": (-0.2618, 0.2618),
    "right_hip_pitch_joint": (-2.5307, 2.8798),
    "right_hip_roll_joint": (-2.9671, 0.5236),
    "right_hip_yaw_joint": (-2.7576, 2.7576),
    "right_knee_joint": (-0.087267, 2.8798),
    "right_ankle_pitch_joint": (-0.87267, 0.5236),
    "right_ankle_roll_joint": (-0.2618, 0.2618),
    "waist_yaw_joint": (-2.618, 2.618),
    "waist_roll_joint": (-0.52, 0.52),
    "waist_pitch_joint": (-0.52, 0.52),
    "left_shoulder_pitch_joint": (-3.0892, 2.6704),
    "left_shoulder_roll_joint": (-1.5882, 2.2515),
    "left_shoulder_yaw_joint": (-2.618, 2.618),
    "left_elbow_joint": (-1.0472, 2.0944),
    "left_wrist_roll_joint": (-1.97222, 1.97222),
    "left_wrist_pitch_joint": (-1.61443, 1.61443),
    "left_wrist_yaw_joint": (-1.61443, 1.61443),
    "right_shoulder_pitch_joint": (-3.0892, 2.6704),
    "right_shoulder_roll_joint": (-2.2515, 1.5882),
    "right_shoulder_yaw_joint": (-2.618, 2.618),
    "right_elbow_joint": (-1.0472, 2.0944),
    "right_wrist_roll_joint": (-1.97222, 1.97222),
    "right_wrist_pitch_joint": (-1.61443, 1.61443),
    "right_wrist_yaw_joint": (-1.61443, 1.61443),
}

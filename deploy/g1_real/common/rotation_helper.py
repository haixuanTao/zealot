"""Quaternion helpers (numpy only). Quaternions are scalar-first (w, x, y, z)."""
import numpy as np


def projected_gravity(quat_wxyz):
    """World-down unit vector [0, 0, -1] expressed in the body frame."""
    w, x, y, z = quat_wxyz
    u = np.array([-x, -y, -z])  # conjugate -> world-to-body rotation
    v = np.array([0.0, 0.0, -1.0])
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def quat_to_matrix(quat_wxyz):
    w, x, y, z = quat_wxyz
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat(m):
    """Rotation matrix -> quaternion (w, x, y, z)."""
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        return np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s,
                         (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    i = int(np.argmax(np.diag(m)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(max(1.0 + m[i, i] - m[j, j] - m[k, k], 1e-12)) * 2
    q = np.zeros(4)
    q[0] = (m[k, j] - m[j, k]) / s
    q[1 + i] = 0.25 * s
    q[1 + j] = (m[j, i] + m[i, j]) / s
    q[1 + k] = (m[k, i] + m[i, k]) / s
    return q


def transform_imu_data(waist_yaw, waist_yaw_omega, imu_quat, imu_omega):
    """Torso-mounted IMU -> pelvis frame (undo the waist yaw joint).

    Only needed when `imu_type: torso`; the G1's IMU is in the pelvis.
    """
    cy, sy = np.cos(waist_yaw), np.sin(waist_yaw)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    r_torso = quat_to_matrix(imu_quat)
    r_pelvis = r_torso @ rz.T
    omega = rz @ np.asarray(imu_omega) - np.array([0.0, 0.0, waist_yaw_omega])
    return matrix_to_quat(r_pelvis), omega

"""Policy wrappers: map raw robot state -> per-motor (q_target, kp, kd).

Two backends:
  - `zealot`: zealot-trained safetensors checkpoint (obs45 x history MLP,
    conventions ported 1:1 from examples/biped/sim2sim_g1_mujoco.py).
  - `agile_e2e`: WBC-AGILE's exported end-to-end ONNX velocity policy
    (normalization/history/action-scaling baked into the graph).
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class RobotObs:
    """Raw hardware state, hardware motor order (29)."""
    q: np.ndarray      # joint positions (rad)
    dq: np.ndarray     # joint velocities (rad/s)
    quat: np.ndarray   # pelvis orientation, scalar-first (w, x, y, z)
    gyro: np.ndarray   # pelvis angular velocity, body frame (rad/s)


class BasePolicy:
    """A policy consumes RobotObs + a velocity command and returns PD targets.

    `step` returns {motor_index: (q_target, kp, kd)} for the motors the policy
    controls; the deploy loop holds every other motor at its startup target.
    """

    control_dt = 0.02

    def reset(self):
        raise NotImplementedError

    def step(self, obs: RobotObs, cmd: np.ndarray) -> dict:
        raise NotImplementedError


def load_policy(policy_cfg: dict, base_dir: str) -> BasePolicy:
    kind = policy_cfg["type"]
    if kind == "zealot":
        from .zealot_policy import ZealotPolicy
        return ZealotPolicy(policy_cfg, base_dir)
    if kind == "agile_e2e":
        from .agile_e2e import AgileE2EPolicy
        return AgileE2EPolicy(policy_cfg, base_dir)
    raise ValueError(f"unknown policy type '{kind}' (expected zealot | agile_e2e)")

"""zealot G1 walking policy on real hardware.

Loads the trainer's safetensors checkpoint (Welford obs normalizer + ELU MLP)
and rebuilds the 45-dim observation frame with the exact training conventions
(source: examples/biped/sim2sim_g1_mujoco.py / biped_env_nexus.rs):

  obs45 = [last_action(12), cmd(4), q - default(12), qdot(12),
           projected_gravity(3), sin 2*pi*phase, cos 2*pi*phase]

  - last_action is LAG-2: the frame at decision t carries the raw policy
    output from t-2 (zeros for the first two steps after reset).
  - joint velocity is FINITE-DIFFERENCED from consecutive 50 Hz joint
    positions (zeros at step 0), NOT the measured motor velocity — that is
    what the policy saw in training. `joint_vel_source: measured` switches
    to the hardware dq if you want to experiment.
  - gait phase(t) = (max(0, t - 1) * control_dt / gait_period) mod 1.
  - frames are stacked oldest -> newest, reset-replicated, then normalized
    by the checkpoint's Welford stats (clip +-5) inside the network.

Action -> PD target: clamp(default + action_scale * action, joint limits).
"""
import os

import numpy as np
from safetensors.numpy import load_file

from common.g1_constants import G1_JOINT_LIMITS, G1_MOTOR_NAMES
from common.rotation_helper import projected_gravity
from . import BasePolicy, RobotObs

OBS_FRAME = 45
NUM_ACTIONS = 12

# The 12 policy joints are hardware motors 0..11, in the same order the
# trainer uses (left leg hip pitch/roll/yaw, knee, ankle pitch/roll; right leg).
POLICY_MOTORS = list(range(12))


class ZealotPolicy(BasePolicy):
    def __init__(self, cfg: dict, base_dir: str):
        path = cfg["path"]
        if not os.path.isabs(path):
            path = os.path.join(base_dir, path)
        sd = load_file(path)
        self.weights, self.biases = [], []
        layer = 0
        while f"actor.w_{layer}" in sd:
            self.weights.append(sd[f"actor.w_{layer}"].astype(np.float64))
            self.biases.append(sd[f"actor.b_{layer}"].astype(np.float64))
            layer += 1
        mean = sd["obs_norm.mean"].astype(np.float64)
        m2 = sd["obs_norm.m2"].astype(np.float64)
        count = float(sd["obs_norm.count"].reshape(-1)[0])
        self.norm_mean = mean
        self.norm_std = np.sqrt(np.maximum(m2 / count, 1e-8))

        obs_dim = self.weights[0].shape[1]
        act_dim = self.weights[-1].shape[0]
        if act_dim != NUM_ACTIONS or obs_dim % OBS_FRAME != 0:
            raise ValueError(
                f"{path}: expected a G1 checkpoint (obs = k*{OBS_FRAME}, "
                f"act = {NUM_ACTIONS}); got obs {obs_dim}, act {act_dim}")
        self.hist = obs_dim // OBS_FRAME

        self.control_dt = float(cfg.get("control_dt", 0.02))
        self.gait_period = float(cfg.get("gait_period", 0.7))
        self.action_scale = float(cfg.get("action_scale", 0.5))
        self.vel_source = cfg.get("joint_vel_source", "fd")
        if self.vel_source not in ("fd", "measured"):
            raise ValueError("joint_vel_source must be 'fd' or 'measured'")

        self.default_pos = np.asarray(cfg["default_angles"], dtype=np.float64)
        assert self.default_pos.shape == (NUM_ACTIONS,)
        self.kps = np.asarray(cfg["kps"], dtype=np.float64)
        self.kds = np.asarray(cfg["kds"], dtype=np.float64)
        limits = np.array([G1_JOINT_LIMITS[G1_MOTOR_NAMES[m]] for m in POLICY_MOTORS])
        self.lo, self.hi = limits[:, 0], limits[:, 1]

        print(f"[zealot] {os.path.basename(path)}: obs {obs_dim} "
              f"({self.hist}x{OBS_FRAME}), {len(self.weights)} layers, "
              f"gait {self.gait_period}s, action_scale {self.action_scale}")
        self.reset()

    def reset(self):
        self.t = 0
        self.act_hist = [np.zeros(NUM_ACTIONS), np.zeros(NUM_ACTIONS)]  # [t-2, t-1]
        self.prev_q = None
        self.frames = None

    def _forward(self, obs):
        a = np.clip((obs - self.norm_mean) / self.norm_std, -5.0, 5.0)
        # errstate: Apple Accelerate BLAS raises spurious FP-flag warnings on
        # matmul even when every output is finite; results are checked below.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            for i, (w, b) in enumerate(zip(self.weights, self.biases)):
                z = w @ a + b
                a = z if i == len(self.weights) - 1 else np.where(z > 0, z, np.expm1(z))
        if not np.isfinite(a).all():
            raise FloatingPointError("policy produced non-finite action")
        return a

    def step(self, obs: RobotObs, cmd: np.ndarray) -> dict:
        q = obs.q[POLICY_MOTORS].astype(np.float64)

        frame = np.zeros(OBS_FRAME)
        frame[0:12] = self.act_hist[0] if self.t >= 2 else 0.0
        frame[12:15] = cmd[:3]
        frame[15] = 0.0  # reserved command slot
        frame[16:28] = q - self.default_pos
        if self.vel_source == "measured":
            frame[28:40] = obs.dq[POLICY_MOTORS]
        else:
            frame[28:40] = 0.0 if self.prev_q is None else (q - self.prev_q) / self.control_dt
        frame[40:43] = projected_gravity(obs.quat)
        phase = (max(0, self.t - 1) * self.control_dt / self.gait_period) % 1.0
        frame[43] = np.sin(2 * np.pi * phase)
        frame[44] = np.cos(2 * np.pi * phase)

        if self.frames is None:
            self.frames = [frame.copy() for _ in range(self.hist)]
        else:
            self.frames = self.frames[1:] + [frame.copy()]

        action = self._forward(np.concatenate(self.frames))
        target = np.clip(self.default_pos + self.action_scale * action, self.lo, self.hi)

        self.prev_q = q
        self.act_hist = [self.act_hist[1], action.copy()]
        self.t += 1

        return {m: (target[i], self.kps[i], self.kds[i])
                for i, m in enumerate(POLICY_MOTORS)}

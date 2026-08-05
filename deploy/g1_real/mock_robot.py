"""Simulated robot I/O for offline pipeline verification (no SDK, no robot).

Motors track commanded targets with a first-order lag; the IMU reports an
upright pelvis with small noise. The remote follows a script: START at 1 s,
A at 2 s, forward stick during the policy phase, select at the end. This
exercises the full deploy loop — config parsing, state machine, policy
inference, PD command assembly, safety checks — at real-time rate.
"""
import time

import numpy as np

from common.g1_constants import G1_NUM_MOTORS
from common.remote_controller import KeyMap, RemoteController


class _Motor:
    __slots__ = ("q", "dq")

    def __init__(self):
        self.q = 0.0
        self.dq = 0.0


class _Imu:
    def __init__(self):
        self.quaternion = [1.0, 0.0, 0.0, 0.0]
        self.gyroscope = [0.0, 0.0, 0.0]


class _LowState:
    def __init__(self):
        self.tick = 1
        self.motor_state = [_Motor() for _ in range(G1_NUM_MOTORS)]
        self.imu_state = _Imu()


class _MotorCmd:
    __slots__ = ("mode", "q", "qd", "kp", "kd", "tau")

    def __init__(self):
        self.mode = 1
        self.q = self.qd = self.kp = self.kd = self.tau = 0.0


class _LowCmd:
    def __init__(self):
        self.motor_cmd = [_MotorCmd() for _ in range(G1_NUM_MOTORS)]


class MockRobotIO:
    def __init__(self, config, run_seconds: float = 8.0):
        self.config = config
        self.low_state = _LowState()
        self.remote = RemoteController()
        self.t0 = time.perf_counter()
        self.run_seconds = run_seconds
        self.policy_started_at = None
        self.sends = 0
        self.rng = np.random.default_rng(0)
        print(f"[mock] simulated robot; policy will run for {run_seconds:.0f} s")

    def state_age(self) -> float:
        return 0.0

    def new_cmd(self):
        return _LowCmd()

    def send(self, cmd):
        self.sends += 1
        # First-order tracking of commanded targets (only motors with kp > 0).
        for i, mc in enumerate(cmd.motor_cmd):
            m = self.low_state.motor_state[i]
            if mc.kp > 0.0:
                new_q = m.q + 0.4 * (mc.q - m.q)
                m.dq = (new_q - m.q) / self.config.control_dt
                m.q = new_q
            else:
                m.dq = 0.0
        # Small IMU noise so obs aren't perfectly constant.
        wob = 0.01 * self.rng.standard_normal(2)
        self.low_state.imu_state.quaternion = [1.0, wob[0], wob[1], 0.0]
        self.low_state.imu_state.gyroscope = list(0.02 * self.rng.standard_normal(3))
        self._script()

    def _script(self):
        t = time.perf_counter() - self.t0
        for i in range(16):
            self.remote.button[i] = 0
        self.remote.lx = self.remote.ly = self.remote.rx = self.remote.ry = 0.0
        if 1.0 < t < 1.5:
            self.remote.button[KeyMap.start] = 1
        started = t > 5.0  # after ramp (3 s) + margin
        if started and self.policy_started_at is None:
            self.remote.button[KeyMap.A] = 1
            self.policy_started_at = t
        elif self.policy_started_at is not None:
            run_t = t - self.policy_started_at
            self.remote.ly = 0.8  # forward command
            if run_t > self.run_seconds:
                self.remote.button[KeyMap.select] = 1

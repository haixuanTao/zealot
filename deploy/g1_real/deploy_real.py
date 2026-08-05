#!/usr/bin/env python3
"""Run a trained walking policy on a physical Unitree G1 (29-DOF, hg DDS).

State machine (matching the standard Unitree deployment flow):

  connect -> ZERO TORQUE  --start-->  ramp to default pose (3 s)
          -> HOLD DEFAULT --A------>  POLICY RUNNING (sticks = velocity cmd)
          -> damping on: select or B button, pelvis tilt > limit,
             stale LowState, or Ctrl-C.

Usage:
  python3 deploy_real.py <net_interface> --config configs/g1_agile_e2e.yaml
  python3 deploy_real.py --mock --config configs/g1_zealot.yaml   # no robot

PREREQUISITES (real robot): robot hanging from a hoist for first trials,
suspend mode entered on the Unitree remote (L2+R2 -> debug mode, joints go
to damping), PC on the robot LAN (192.168.123.x).
"""
import argparse
import os
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common.g1_constants import G1_NUM_MOTORS
from common.remote_controller import KeyMap
from common.rotation_helper import projected_gravity, transform_imu_data
from policies import RobotObs, load_policy

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Config:
    def __init__(self, path):
        with open(path) as f:
            raw = yaml.safe_load(f)
        self.control_dt = float(raw["control_dt"])
        self.imu_type = raw.get("imu_type", "pelvis")
        self.lowcmd_topic = raw.get("lowcmd_topic", "rt/lowcmd")
        self.lowstate_topic = raw.get("lowstate_topic", "rt/lowstate")
        self.policy = raw["policy"]
        s = raw["startup"]
        self.leg_motors = list(s["leg_motors"])
        self.leg_angles = np.asarray(s["leg_angles"], dtype=np.float64)
        self.leg_kps = np.asarray(s["leg_kps"], dtype=np.float64)
        self.leg_kds = np.asarray(s["leg_kds"], dtype=np.float64)
        self.hold_motors = list(s["hold_motors"])
        self.hold_angles = np.asarray(s["hold_angles"], dtype=np.float64)
        self.hold_kps = np.asarray(s["hold_kps"], dtype=np.float64)
        self.hold_kds = np.asarray(s["hold_kds"], dtype=np.float64)
        assert len(self.leg_motors) == len(self.leg_angles)
        assert len(self.hold_motors) == len(self.hold_angles)
        covered = set(self.leg_motors) | set(self.hold_motors)
        assert covered == set(range(G1_NUM_MOTORS)), \
            f"startup config must cover all 29 motors, missing {set(range(G1_NUM_MOTORS)) - covered}"
        self.max_cmd = np.asarray(raw["max_cmd"], dtype=np.float64)
        sf = raw.get("safety", {})
        self.max_tilt_rad = np.deg2rad(float(sf.get("max_tilt_deg", 55.0)))
        self.lowstate_timeout = float(sf.get("lowstate_timeout", 0.2))
        self.max_target_step = float(sf.get("max_target_step", 0.6))


class Controller:
    def __init__(self, config: Config, io):
        self.config = config
        self.io = io  # RealRobotIO or MockRobotIO: .low_state, .remote, .new_cmd(), .send(), .state_age()
        self.policy = load_policy(config.policy, BASE_DIR)
        if abs(self.policy.control_dt - config.control_dt) > 1e-9:
            print(f"WARNING: policy dt {self.policy.control_dt} != config dt "
                  f"{config.control_dt}; using policy dt")
            self.config.control_dt = self.policy.control_dt
        self.cmd = io.new_cmd()
        self.overruns = 0

    # ---- state helpers -----------------------------------------------------

    def robot_obs(self) -> RobotObs:
        ls = self.io.low_state
        q = np.array([ls.motor_state[i].q for i in range(G1_NUM_MOTORS)])
        dq = np.array([ls.motor_state[i].dq for i in range(G1_NUM_MOTORS)])
        quat = np.array(ls.imu_state.quaternion, dtype=np.float64)  # w, x, y, z
        gyro = np.array(ls.imu_state.gyroscope, dtype=np.float64)
        if self.config.imu_type == "torso":
            waist_yaw = ls.motor_state[12].q
            waist_yaw_omega = ls.motor_state[12].dq
            quat, gyro = transform_imu_data(waist_yaw, waist_yaw_omega, quat, gyro)
        return RobotObs(q=q, dq=dq, quat=quat, gyro=gyro)

    def set_motor(self, idx, q, kp, kd):
        mc = self.cmd.motor_cmd[idx]
        mc.q = float(q)
        mc.qd = 0.0
        mc.kp = float(kp)
        mc.kd = float(kd)
        mc.tau = 0.0

    def joystick_cmd(self):
        r = self.io.remote
        def dz(v):
            return v if abs(v) > 0.1 else 0.0
        return np.array([
            dz(r.ly) * self.config.max_cmd[0],
            dz(-r.lx) * self.config.max_cmd[1],
            dz(-r.rx) * self.config.max_cmd[2],
            0.0,
        ])

    def safety_trip(self, obs: RobotObs):
        if self.io.remote.button[KeyMap.select] or self.io.remote.button[KeyMap.B]:
            return "stop button (select/B)"
        tilt = np.arccos(np.clip(-projected_gravity(obs.quat)[2], -1.0, 1.0))
        if tilt > self.config.max_tilt_rad:
            return f"pelvis tilt {np.degrees(tilt):.0f} deg"
        age = self.io.state_age()
        if age > self.config.lowstate_timeout:
            return f"LowState stale ({age * 1e3:.0f} ms)"
        return None

    def damping(self, reason):
        print(f"\nDAMPING: {reason}")
        for i in range(G1_NUM_MOTORS):
            self.set_motor(i, 0.0, 0.0, 8.0)
        for _ in range(5):
            self.io.send(self.cmd)
            time.sleep(0.002)

    # ---- state machine -----------------------------------------------------

    def zero_torque_until_start(self):
        print("Zero torque. Press START on the remote to ramp to the default pose.")
        while not self.io.remote.button[KeyMap.start]:
            for i in range(G1_NUM_MOTORS):
                self.set_motor(i, 0.0, 0.0, 0.0)
            self.io.send(self.cmd)
            time.sleep(self.config.control_dt)

    def ramp_to_default(self, seconds=3.0):
        print(f"Ramping to default pose over {seconds:.0f} s...")
        c = self.config
        motors = c.leg_motors + c.hold_motors
        targets = np.concatenate([c.leg_angles, c.hold_angles])
        kps = np.concatenate([c.leg_kps, c.hold_kps])
        kds = np.concatenate([c.leg_kds, c.hold_kds])
        q0 = np.array([self.io.low_state.motor_state[m].q for m in motors])
        steps = int(seconds / c.control_dt)
        for k in range(steps):
            alpha = (k + 1) / steps
            qk = (1 - alpha) * q0 + alpha * targets
            for j, m in enumerate(motors):
                self.set_motor(m, qk[j], kps[j], kds[j])
            self.io.send(self.cmd)
            time.sleep(c.control_dt)

    def hold_until_a(self):
        print("Holding default pose. Press A to START THE POLICY, select/B to abort.")
        while not self.io.remote.button[KeyMap.A]:
            if self.io.remote.button[KeyMap.select] or self.io.remote.button[KeyMap.B]:
                return False
            self.io.send(self.cmd)  # re-send the ramp's final targets
            time.sleep(self.config.control_dt)
        return True

    def run_policy(self):
        c = self.config
        print("POLICY RUNNING. Left stick: vx/vy, right stick: yaw. "
              "select/B: stop.  Ctrl-C: stop.")
        self.policy.reset()
        # Held (non-policy) motors keep their startup targets every tick.
        holds = list(zip(c.hold_motors, c.hold_angles, c.hold_kps, c.hold_kds))
        next_tick = time.perf_counter()
        ticks = 0
        while True:
            obs = self.robot_obs()
            trip = self.safety_trip(obs)
            if trip:
                self.damping(trip)
                return
            cmd = self.joystick_cmd()
            targets = self.policy.step(obs, cmd)
            for m, (q_t, kp, kd) in targets.items():
                # never ask a motor to jump further than max_target_step from
                # where it currently is (protects against policy transients)
                q_c = float(np.clip(q_t, obs.q[m] - c.max_target_step,
                                    obs.q[m] + c.max_target_step))
                self.set_motor(m, q_c, kp, kd)
            for m, q_t, kp, kd in holds:
                self.set_motor(m, q_t, kp, kd)
            self.io.send(self.cmd)

            ticks += 1
            if ticks % 250 == 0:  # every 5 s
                tilt = np.degrees(np.arccos(np.clip(
                    -projected_gravity(obs.quat)[2], -1.0, 1.0)))
                print(f"  t={ticks * c.control_dt:6.1f}s cmd=[{cmd[0]:+.2f} "
                      f"{cmd[1]:+.2f} {cmd[2]:+.2f}] tilt={tilt:4.1f}deg "
                      f"overruns={self.overruns}")

            next_tick += c.control_dt
            now = time.perf_counter()
            if now < next_tick:
                time.sleep(next_tick - now)
            else:
                self.overruns += 1
                next_tick = now  # don't try to catch up


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("net", nargs="?", default=None,
                        help="network interface bound to the robot LAN (e.g. en7, eth0)")
    parser.add_argument("--config", default="configs/g1_agile_e2e.yaml")
    parser.add_argument("--mock", action="store_true",
                        help="run against a simulated robot (no SDK, no hardware)")
    args = parser.parse_args()

    config = Config(os.path.join(BASE_DIR, args.config)
                    if not os.path.isabs(args.config) else args.config)

    if args.mock:
        from mock_robot import MockRobotIO
        io = MockRobotIO(config)
    else:
        if not args.net:
            parser.error("network interface required (or use --mock)")
        from real_robot import RealRobotIO
        io = RealRobotIO(config, args.net)

    controller = Controller(config, io)
    try:
        controller.zero_torque_until_start()
        controller.ramp_to_default()
        if controller.hold_until_a():
            controller.run_policy()
        else:
            controller.damping("aborted before policy start")
    except KeyboardInterrupt:
        controller.damping("Ctrl-C")
    print("Exit.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Read-only connectivity probe: subscribe to rt/lowstate for a few seconds.

Sends NOTHING to the robot. Prints tick, IMU, a few joint angles and the
remote-button word if the G1's DDS traffic is visible on the interface.

Usage: python3 probe_lowstate.py <net_interface> [seconds]
"""
import sys
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

iface = sys.argv[1]
seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0

count = 0
last = None


def on_msg(msg: LowState_):
    global count, last
    count += 1
    last = msg


ChannelFactoryInitialize(0, iface)
sub = ChannelSubscriber("rt/lowstate", LowState_)
sub.Init(on_msg, 10)

print(f"listening on {iface} for {seconds:.0f}s...")
t0 = time.time()
while time.time() - t0 < seconds:
    time.sleep(0.5)
    print(f"  {count} msgs", flush=True)

if last is None:
    print("NO LowState received — robot not visible on this interface/domain.")
    sys.exit(1)

q = last.imu_state.quaternion
print(f"\nLowState OK: {count} msgs in {seconds:.0f}s (~{count / seconds:.0f} Hz)")
print(f"  tick={last.tick} mode_machine={last.mode_machine}")
print(f"  imu quat wxyz=[{q[0]:+.3f} {q[1]:+.3f} {q[2]:+.3f} {q[3]:+.3f}]")
print(f"  gyro=[{last.imu_state.gyroscope[0]:+.3f} {last.imu_state.gyroscope[1]:+.3f} "
      f"{last.imu_state.gyroscope[2]:+.3f}]")
for i in (0, 3, 4, 12):
    m = last.motor_state[i]
    print(f"  motor[{i:2d}] q={m.q:+.3f} dq={m.dq:+.3f}")
print(f"  remote bytes[2:4]={bytes(last.wireless_remote[2:4]).hex()}")

#!/usr/bin/env python3
"""Read-only LowState recorder + health report. Publishes NOTHING.

Records q/dq/tau/temperature for all 29 motors, IMU quat/gyro/accel and the
remote word at the incoming rate for N seconds, saves an .npz, and prints a
sanity report (rate, quat norm, joint limits, velocity/temp ranges).

Usage: python3 record_lowstate.py <net_interface> [seconds] [out.npz]
"""
import sys
import time

import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common.g1_constants import G1_JOINT_LIMITS, G1_MOTOR_NAMES, G1_NUM_MOTORS
from common.rotation_helper import projected_gravity

iface = sys.argv[1]
seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 15.0
out = sys.argv[3] if len(sys.argv) > 3 else "/tmp/g1_lowstate.npz"

rows = []
t0 = None


def on_msg(msg: LowState_):
    global t0
    now = time.perf_counter()
    if t0 is None:
        t0 = now
    q = [msg.motor_state[i].q for i in range(G1_NUM_MOTORS)]
    dq = [msg.motor_state[i].dq for i in range(G1_NUM_MOTORS)]
    tau = [msg.motor_state[i].tau_est for i in range(G1_NUM_MOTORS)]
    temp = [msg.motor_state[i].temperature[0] for i in range(G1_NUM_MOTORS)]
    rows.append((now - t0, msg.tick, msg.mode_machine,
                 list(msg.imu_state.quaternion), list(msg.imu_state.gyroscope),
                 list(msg.imu_state.accelerometer), q, dq, tau, temp))


ChannelFactoryInitialize(0, iface)
sub = ChannelSubscriber("rt/lowstate", LowState_)
sub.Init(on_msg, 50)

print(f"recording on {iface} for {seconds:.0f}s (read-only)...")
t_end = time.time() + seconds
while time.time() < t_end:
    time.sleep(1.0)
    print(f"  {len(rows)} samples", flush=True)

if not rows:
    print("NO DATA — robot not visible on this interface.")
    sys.exit(1)

t = np.array([r[0] for r in rows])
tick = np.array([r[1] for r in rows])
quat = np.array([r[3] for r in rows])
gyro = np.array([r[4] for r in rows])
accel = np.array([r[5] for r in rows])
q = np.array([r[6] for r in rows])
dq = np.array([r[7] for r in rows])
tau = np.array([r[8] for r in rows])
temp = np.array([r[9] for r in rows])

np.savez_compressed(out, t=t, tick=tick, quat=quat, gyro=gyro, accel=accel,
                    q=q, dq=dq, tau=tau, temp=temp,
                    mode_machine=np.array([r[2] for r in rows]),
                    names=np.array(G1_MOTOR_NAMES))
print(f"saved {len(rows)} samples -> {out}\n")

# ---- health report ---------------------------------------------------------
dur = t[-1] - t[0]
rate = (len(rows) - 1) / dur if dur > 0 else 0
print(f"rate: {rate:.0f} Hz over {dur:.1f}s   mode_machine={rows[-1][2]}")

qn = np.linalg.norm(quat, axis=1)
print(f"imu quat norm: {qn.min():.4f}..{qn.max():.4f} (want ~1)")
pg = np.array([projected_gravity(qq) for qq in quat])
tilt = np.degrees(np.arccos(np.clip(-pg[:, 2], -1, 1)))
print(f"pelvis tilt: mean {tilt.mean():.1f} deg, max {tilt.max():.1f} deg")
print(f"gyro |max|: {np.abs(gyro).max():.2f} rad/s   "
      f"accel z mean: {accel[:, 2].mean():+.2f} (want ~+9.8 upright)")

print(f"\n{'joint':28s} {'q range':>16s} {'limit':>16s} {'|dq|max':>8s} "
      f"{'|tau|max':>8s} {'temp':>5s}")
warn = []
for i, name in enumerate(G1_MOTOR_NAMES):
    lo, hi = G1_JOINT_LIMITS[name]
    flag = ""
    if q[:, i].min() < lo - 0.02 or q[:, i].max() > hi + 0.02:
        flag = " OUT-OF-LIMIT"
        warn.append(f"{name}: q outside MJCF limits")
    if temp[:, i].max() > 70:
        flag += " HOT"
        warn.append(f"{name}: temperature {temp[:, i].max():.0f}C")
    print(f"{name:28s} [{q[:, i].min():+.2f},{q[:, i].max():+.2f}] "
          f"[{lo:+.2f},{hi:+.2f}] {np.abs(dq[:, i]).max():8.2f} "
          f"{np.abs(tau[:, i]).max():8.1f} {temp[:, i].max():5.0f}{flag}")

print()
if rate < 100:
    warn.append(f"low state rate {rate:.0f} Hz (expect ~500)")
if abs(qn.mean() - 1) > 0.01:
    warn.append("IMU quaternion not normalized")
if warn:
    print("WARNINGS:")
    for w in warn:
        print(f"  - {w}")
else:
    print("ALL CHECKS PASS.")

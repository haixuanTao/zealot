"""Always-on PICO publisher: body joints (content-gated) + thumbsticks.

Every message carries "stick" [lx,ly,rx,ry] and "trig" [l,r]; smpl fields ride
along only when body tracking is FRESH (timestamp advanced AND joints actually
changed). Runs on the G1 Orin inside GR00T-WholeBodyControl's .venv_teleop.

Deploy:  scp orin/pose_pub.py unitree@172.18.130.111:~/
Run:     source ~/GR00T-WholeBodyControl/.venv_teleop/bin/activate
         nohup python -u ~/pose_pub.py > /tmp/pose_pub.log 2>&1 &
(needs roboticsservice running: systemctl status roboticsservice)
"""
import json, time
import numpy as np
import zmq
import xrobotoolkit_sdk as xrt

HEADER_SIZE = 1024

def pack(data, version=2, topic=b"pose"):
    fields, blobs = [], []
    for k, v in data.items():
        v = np.ascontiguousarray(v)
        dt = {"float32": "f32", "float64": "f64", "int32": "i32", "int64": "i64"}[v.dtype.name]
        fields.append({"name": k, "dtype": dt, "shape": list(v.shape)})
        blobs.append(v.tobytes())
    hdr = json.dumps({"v": version, "endian": "le", "count": 1, "fields": fields},
                     separators=(",", ":")).encode().ljust(HEADER_SIZE, b"\x00")
    return topic + hdr + b"".join(blobs)

def get_trig():
    try:
        return np.array([float(xrt.get_left_trigger()), float(xrt.get_right_trigger())], dtype=np.float32)
    except Exception:
        return np.zeros(2, dtype=np.float32)

def get_btn():
    try:
        return np.array([float(xrt.get_A_button()), float(xrt.get_B_button()),
                         float(xrt.get_X_button()), float(xrt.get_Y_button())], dtype=np.float32)
    except Exception:
        return np.zeros(4, dtype=np.float32)

def get_stick():
    try:
        l = list(xrt.get_left_axis())[:2]
        r = list(xrt.get_right_axis())[:2]
        return np.array(l + r, dtype=np.float32)
    except Exception:
        return np.zeros(4, dtype=np.float32)

xrt.init()
ctx = zmq.Context()
sock = ctx.socket(zmq.PUB)
sock.bind("tcp://*:5556")
print("pose_pub: bound tcp://*:5556 (body + sticks)")
frame = 0
last_stamp = None
last_body = None
frozen_count = 0
while True:
    time.sleep(1 / 90)
    fields = {"stick": get_stick(), "trig": get_trig(), "btn": get_btn(),
              "frame_index": np.array([frame], dtype=np.int64)}
    fresh = False
    if xrt.is_body_data_available():
        stamp = int(xrt.get_time_stamp_ns())
        if stamp != last_stamp:
            last_stamp = stamp
            body = np.array(xrt.get_body_joints_pose())  # [24,7] xyz + quat(xyzw)
            if last_body is None or not np.array_equal(body, last_body):
                if frozen_count >= 270:
                    print("BODY RESUMED: joints moving again")
                frozen_count = 0
                last_body = body
                fresh = True
            else:
                frozen_count += 1
                if frozen_count == 270:
                    print("BODY FROZEN: joints identical for ~3s (trackers asleep?)")
    if fresh:
        pos = last_body[:, :3]
        zup = np.stack([pos[:, 0], -pos[:, 2], pos[:, 1]], axis=1)
        fields["smpl_joints"] = zup[None].astype(np.float32)
        fields["smpl_pose"] = np.zeros((1, 21, 3), dtype=np.float32)
        # wrist/hand orientations (xyzw quats, ORIGINAL y-up frame) for twist
        fields["wrist_quat"] = last_body[[20, 21, 22, 23], 3:7].astype(np.float32)
    sock.send(pack(fields))
    frame += 1
    if frame % 3000 == 0:
        print(f"sent {frame} msgs (sticks always, body when fresh)")

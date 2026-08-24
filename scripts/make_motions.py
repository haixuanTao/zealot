#!/usr/bin/env python3
"""Build ~/sonic-motions for zealot from the LAFAN1 G1 retargeted dataset.

Downloads the locomotion clips (walk/run/sprint — arm swing during gait) from
lvhaidong/LAFAN1_Retargeting_Dataset and converts them from the headerless
radians CSV to zealot's SONIC CSV format (header row, degrees, cm root).
"""
import math, os, sys, urllib.request, json

OUT = "/workspace/sonic-motions"
BASE = "https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/resolve/main/g1/"
API = "https://huggingface.co/api/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/tree/main/g1"

# G1 29-dof column order per the dataset README (after root xyz + quat xyzw).
JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

def quat_to_euler_xyz_deg(qx, qy, qz, qw):
    # XYZ intrinsic euler from quaternion, degrees.
    sinr = 2 * (qw * qx + qy * qz)
    cosr = 1 - 2 * (qx * qx + qy * qy)
    x = math.atan2(sinr, cosr)
    sinp = max(-1.0, min(1.0, 2 * (qw * qy - qz * qx)))
    y = math.asin(sinp)
    siny = 2 * (qw * qz + qx * qy)
    cosy = 1 - 2 * (qy * qy + qz * qz)
    z = math.atan2(siny, cosy)
    r2d = 180.0 / math.pi
    return x * r2d, y * r2d, z * r2d

def convert(name, text):
    r2d = 180.0 / math.pi
    hdr = ["Frame", "root_translateX", "root_translateY", "root_translateZ",
           "root_rotateX", "root_rotateY", "root_rotateZ"] + [j + "_dof" for j in JOINTS]
    out = [",".join(hdr)]
    for i, line in enumerate(t for t in text.splitlines() if t.strip()):
        v = [float(x) for x in line.split(",")]
        if len(v) != 7 + 29:
            raise SystemExit(f"{name}: row {i} has {len(v)} cols, want 36")
        ex, ey, ez = quat_to_euler_xyz_deg(v[3], v[4], v[5], v[6])
        row = [str(i), f"{v[0]*100:.4f}", f"{v[1]*100:.4f}", f"{v[2]*100:.4f}",
               f"{ex:.4f}", f"{ey:.4f}", f"{ez:.4f}"] + [f"{x*r2d:.4f}" for x in v[7:]]
        out.append(",".join(row))
    return "\n".join(out) + "\n"

def main():
    os.makedirs(OUT, exist_ok=True)
    with urllib.request.urlopen(API) as r:
        files = [f["path"].split("/")[-1] for f in json.load(r)]
    picks = [f for f in files if f.split("_")[0].rstrip("0123456789") in ("walk", "run", "sprint")]
    print(f"{len(files)} clips in dataset, converting {len(picks)} locomotion clips")
    for f in picks:
        with urllib.request.urlopen(BASE + f) as r:
            text = r.read().decode()
        csv = convert(f, text)
        with open(os.path.join(OUT, f), "w") as fh:
            fh.write(csv)
        print(f"  {f}: {csv.count(chr(10)) - 1} frames")
    print("done ->", OUT)

if __name__ == "__main__":
    main()

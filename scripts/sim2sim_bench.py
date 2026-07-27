#!/usr/bin/env python3
"""Cross-engine sim2sim benchmark for a zealot G1 checkpoint.

Runs the SAME policy closed-loop in (a) nexus (source engine, current
production physics: d4 + 4 substeps + NEXUS_SUBSTEP_REFRESH, NF240/DR1,
depenetration clamp) and (b) MuJoCo (mujoco_playground's official G1 29-DOF
feetonly scene), at a fixed command set, and emits per-run metrics JSON plus
a combined markdown/CSV table ready for release.

Usage:
  python3 scripts/sim2sim_bench.py <checkpoint.safetensors> <label> [outdir]

Requires: target/release/examples/biped_render_nexus (features gpu,biped_gpu),
a python with mujoco + mujoco_playground (BENCH_PY env, default
~/rt_build/bench-venv/bin/python), ffmpeg on PATH.
"""
import json, math, os, subprocess, sys

CKPT = os.path.abspath(sys.argv[1])
LABEL = sys.argv[2] if len(sys.argv) > 2 else "unlabeled"
OUT = os.path.abspath(sys.argv[3] if len(sys.argv) > 3 else f"bench/sim2sim/{LABEL}")
BENCH_PY = os.environ.get("BENCH_PY", os.path.expanduser("~/rt_build/bench-venv/bin/python"))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIP_S = 15
COMMANDS = [("stand", "0,0,0"), ("walk04", "0.4,0,0"), ("walk08", "0.8,0,0")]

# nexus eval env: production physics, nominal DR, flat ground, no actuation
# delay (the MuJoCo leg models none), fixed command.
NEXUS_ENV = {
    "KHAL_BACKEND": "webgpu", "BIPED_RENDER_ENVS": "1",
    "BIPED_ROBOT": "g1_29dof_agile", "BIPED_OBS_HISTORY": "5",
    "BIPED_CONTACT_SENSE": "1", "BIPED_CONTACT_CAP": "128",
    "BIPED_CONTACT_REDUCE": "1", "BIPED_MOTOR_DELAY": "0,0",
    "BIPED_DECIMATION": "4", "BIPED_SOLVER_ITERS": "4",
    "NEXUS_SUBSTEP_REFRESH": "1", "BIPED_CONTACT_NF": "240",
    "BIPED_CONTACT_DR": "1", "BIPED_MAX_CORR_VEL": "0.2",
}

def pitch(qx, qy, qz, qw):
    return math.degrees(math.asin(max(-1.0, min(1.0, 2 * (qw * qy - qz * qx)))))

def nexus_leg(name, cmd):
    steps = CLIP_S * 50
    rollout = f"{OUT}/nexus_{name}.json"
    env = dict(os.environ, **NEXUS_ENV, BIPED_EVAL_CMD=cmd, BIPED_RENDER_CMD=cmd)
    subprocess.run([f"{ROOT}/target/release/examples/biped_render_nexus",
                    "0", str(steps), rollout, CKPT],
                   env=env, check=True, capture_output=True)
    d = json.load(open(rollout))
    b, names, joints = d["base"], d["joint_names"], d["joints"]
    fall = next((t for t, f in enumerate(b) if f[2] < 0.45), None)
    upto = fall if fall is not None else len(b)
    x0, y0 = b[0][0], b[0][1]
    traveled = math.hypot(b[upto-1][0]-x0, b[upto-1][1]-y0)
    ankles = [i for i, n in enumerate(names) if "ankle_pitch" in n]
    apin = sum(1 for t in range(upto) for i in ankles if joints[t][i] < -0.8)
    return {
        "engine": "nexus", "command": cmd, "clip_seconds": CLIP_S,
        "survived_s": round(upto * 0.02, 2), "fell": fall is not None,
        "traveled_m": round(traveled, 3),
        "mean_speed": round(traveled / max(upto * 0.02, 1e-6), 3),
        "max_pitch_deg": round(max(abs(pitch(*f[3:7])) for f in b[:upto]), 2),
        "ankle_pinned_frac": round(apin / max(upto * len(ankles), 1), 3),
    }

def mujoco_leg(name, cmd):
    mjson = f"{OUT}/mujoco_{name}.metrics.json"
    env = dict(os.environ, BIPED_CMD=cmd, S2S_METRICS_JSON=mjson, MUJOCO_GL="egl")
    subprocess.run([BENCH_PY, f"{ROOT}/examples/biped/sim2sim_g1_mujoco.py",
                    CKPT, f"{OUT}/mujoco_{name}.mp4", str(CLIP_S)],
                   env=env, check=True, capture_output=True)
    m = json.load(open(mjson))
    eps = m["episodes"]
    total_s = sum(e["seconds"] for e in eps)
    total_d = sum(e["traveled_m"] for e in eps)
    longest = max(e["seconds"] for e in eps)
    return {
        "engine": "mujoco", "command": cmd, "clip_seconds": CLIP_S,
        "survived_s": round(longest, 2), "fell": m["falls"] > 0,
        "falls": m["falls"],
        "traveled_m": round(total_d, 3),
        "mean_speed": round(total_d / max(total_s, 1e-6), 3),
    }

def main():
    os.makedirs(OUT, exist_ok=True)
    results = {"label": LABEL, "checkpoint": os.path.basename(CKPT), "runs": []}
    for name, cmd in COMMANDS:
        for leg in (nexus_leg, mujoco_leg):
            r = leg(name, cmd)
            r["run"] = name
            results["runs"].append(r)
            print(f'{r["engine"]:>6} {name:>7}: survived {r["survived_s"]:>5}s'
                  f'  fell={r["fell"]}  speed {r["mean_speed"]} m/s')
    with open(f"{OUT}/results.json", "w") as f:
        json.dump(results, f, indent=1)
    # markdown table
    lines = [f"# sim2sim benchmark — {LABEL}", "",
             "| run | engine | survived (s) | fell | mean speed (m/s) | commanded vx |",
             "|---|---|---:|---|---:|---:|"]
    for r in results["runs"]:
        lines.append(f'| {r["run"]} | {r["engine"]} | {r["survived_s"]} | '
                     f'{"yes" if r["fell"] else "no"} | {r["mean_speed"]} | '
                     f'{r["command"].split(",")[0]} |')
    open(f"{OUT}/results.md", "w").write("\n".join(lines) + "\n")
    print(f"\nwrote {OUT}/results.json + results.md")

if __name__ == "__main__":
    main()

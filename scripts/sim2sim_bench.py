#!/usr/bin/env python3
"""Cross-engine sim2sim benchmark for a zealot G1 checkpoint.

Runs the SAME policy closed-loop in the source engine (nexus, current
production physics: d4 + 4 substeps + NEXUS_SUBSTEP_REFRESH, NF240/DR1,
depenetration clamp) and in every available foreign engine — MuJoCo
(mujoco_playground's official G1 29-DOF feetonly scene), Genesis, and
Isaac Sim (PhysX) — at a fixed command set, and emits per-run metrics JSON
plus a combined markdown/CSV table ready for release.

The signed body-frame forward velocity (`body_vx`) is the load-bearing
column: traveled distance alone cannot distinguish tracking from
anti-tracking (the v16 backward-walk exploit was invisible to it).

Usage:
  python3 scripts/sim2sim_bench.py <checkpoint.safetensors> <label> [outdir]

Engines are skipped with a warning when their python is missing:
  MuJoCo:  BENCH_PY   (default ~/rt_build/bench-venv/bin/python)
  Genesis: GENESIS_PY (default ~/rt_build/nyx-venv/bin/python)
  Isaac:   ISAAC_PY   (default ~/rt_build/isaac-venv/bin/python)
Requires target/release/examples/biped_render_nexus (features gpu,biped_gpu)
and ffmpeg on PATH.
"""
import json, math, os, subprocess, sys

CKPT = os.path.abspath(sys.argv[1])
LABEL = sys.argv[2] if len(sys.argv) > 2 else "unlabeled"
OUT = os.path.abspath(sys.argv[3] if len(sys.argv) > 3 else f"bench/sim2sim/{LABEL}")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIP_S = 15
COMMANDS = [("stand", "0,0,0"), ("slow02", "0.2,0,0"),
            ("walk04", "0.4,0,0"), ("walk08", "0.8,0,0")]

# (engine, harness script, python env var, default python, extra env)
FOREIGN = [
    ("mujoco", "examples/biped/sim2sim_g1_mujoco.py", "BENCH_PY",
     "~/rt_build/bench-venv/bin/python", {"MUJOCO_GL": "egl"}),
    ("genesis", "examples/biped/sim2sim_g1_genesis.py", "GENESIS_PY",
     "~/rt_build/nyx-venv/bin/python", {}),
    ("isaacsim-physx", "examples/biped/sim2sim_g1_isaac.py", "ISAAC_PY",
     "~/rt_build/isaac-venv/bin/python",
     {"ACCEPT_EULA": "Y", "OMNI_KIT_ACCEPT_EULA": "YES", "PRIVACY_CONSENT": "Y"}),
]

# nexus eval env: production physics, nominal DR, flat ground, no actuation
# delay (the foreign models carry none), fixed command.
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

def yaw_of(qx, qy, qz, qw):
    return math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))

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
    fwd = 0.0
    for t in range(max(upto - 1, 0)):
        yw = yaw_of(*b[t][3:7])
        fwd += math.cos(yw) * (b[t+1][0] - b[t][0]) + math.sin(yw) * (b[t+1][1] - b[t][1])
    ankles = [i for i, n in enumerate(names) if "ankle_pitch" in n]
    apin = sum(1 for t in range(upto) for i in ankles if joints[t][i] < -0.8)
    return {
        "engine": "nexus", "command": cmd, "clip_seconds": CLIP_S,
        "survived_s": round(upto * 0.02, 2), "fell": fall is not None,
        "traveled_m": round(traveled, 3),
        "mean_speed": round(traveled / max(upto * 0.02, 1e-6), 3),
        "body_vx": round(fwd / max((upto - 1) * 0.02, 1e-6), 3),
        "max_pitch_deg": round(max(abs(pitch(*f[3:7])) for f in b[:upto]), 2),
        "ankle_pinned_frac": round(apin / max(upto * len(ankles), 1), 3),
    }

def foreign_leg(engine, script, py, extra_env, name, cmd):
    mjson = f"{OUT}/{engine}_{name}.metrics.json"
    env = dict(os.environ, BIPED_CMD=cmd, S2S_METRICS_JSON=mjson, **extra_env)
    if os.path.exists(mjson):
        os.remove(mjson)          # never read a stale result from a prior run
    proc = subprocess.run([py, f"{ROOT}/{script}",
                           CKPT, f"{OUT}/{engine}_{name}.mp4", str(CLIP_S)],
                          env=env, capture_output=True, text=True)
    # Isaac Sim's kit runtime exits 0 even when the embedded python raises, so
    # the return code is NOT a reliable failure signal -- a missing metrics file
    # is. Surface the child's traceback instead of dying on FileNotFoundError
    # several frames away from the actual cause.
    if not os.path.exists(mjson):
        tail = "\n".join((proc.stderr or proc.stdout or "").strip().splitlines()[-15:])
        raise RuntimeError(
            f"{engine}/{name} wrote no metrics (exit {proc.returncode}):\n{tail}")
    m = json.load(open(mjson))
    eps = m["episodes"]
    total_s = sum(e["seconds"] for e in eps)
    total_d = sum(e["traveled_m"] for e in eps)
    longest = max(e["seconds"] for e in eps)
    vx = m.get("mean_body_vel", [0.0, 0.0, 0.0])[0]
    return {
        "engine": engine, "command": cmd, "clip_seconds": CLIP_S,
        "survived_s": round(longest, 2), "fell": m["falls"] > 0,
        "falls": m["falls"],
        "traveled_m": round(total_d, 3),
        "mean_speed": round(total_d / max(total_s, 1e-6), 3),
        "body_vx": round(vx, 3),
    }

def main():
    os.makedirs(OUT, exist_ok=True)
    engines = []
    for engine, script, var, default, extra in FOREIGN:
        py = os.environ.get(var, os.path.expanduser(default))
        if os.path.exists(py):
            engines.append((engine, script, py, extra))
        else:
            print(f"SKIP {engine}: no python at {py} (set {var})")
    results = {"label": LABEL, "checkpoint": os.path.basename(CKPT), "runs": []}
    for name, cmd in COMMANDS:
        legs = [lambda n=name, c=cmd: nexus_leg(n, c)]
        legs += [lambda e=e, s=s, p=p, x=x, n=name, c=cmd: foreign_leg(e, s, p, x, n, c)
                 for (e, s, p, x) in engines]
        for leg in legs:
            # One engine failing must not cost the other 15 runs: record the
            # failure as a run and carry on, so a partial battery is still
            # readable and missing engines are visible rather than silent.
            try:
                r = leg()
            except Exception as exc:
                print(f"FAIL {name}: {exc}", flush=True)
                results["runs"].append({"engine": "?", "run": name,
                                        "error": str(exc)})
                continue
            r["run"] = name
            results["runs"].append(r)
            print(f'{r["engine"]:>14} {name:>7}: survived {r["survived_s"]:>5}s'
                  f'  fell={r["fell"]}  body_vx {r["body_vx"]:+.2f} m/s')
    with open(f"{OUT}/results.json", "w") as f:
        json.dump(results, f, indent=1)
    lines = [f"# sim2sim benchmark — {LABEL}", "",
             "| run | engine | survived (s) | fell | body vx (m/s) | commanded vx |",
             "|---|---|---:|---|---:|---:|"]
    for r in results["runs"]:
        lines.append(f'| {r["run"]} | {r["engine"]} | {r["survived_s"]} | '
                     f'{"yes" if r["fell"] else "no"} | {r["body_vx"]:+.2f} | '
                     f'{r["command"].split(",")[0]} |')
    open(f"{OUT}/results.md", "w").write("\n".join(lines) + "\n")
    print(f"\nwrote {OUT}/results.json + results.md")

if __name__ == "__main__":
    main()

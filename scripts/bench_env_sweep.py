#!/usr/bin/env python3
"""Benchmark biped_train_gpu s/iter across env counts and plot the curve.

Runs the trainer once per env count, parses the per-iteration wall-clock
column from its log (printed every 10 iters), discards iter 0 (env build +
shader warmup), and averages the rest. Writes <out>.csv always and <out>.png
if matplotlib is available.

The trainer only prints an iter line when `it % 10 == 0` or on the last
iter, so --iters below 11 yields no usable (non-warmup) sample. Default 21
gives two samples per point (iters 10 and 20).

Usage (from a checkout with target/release/biped_train_gpu built):
    scripts/bench_env_sweep.py                      # 256..4096, 21 iters
    scripts/bench_env_sweep.py --envs 512,1024 --iters 11
    KHAL_BACKEND=webgpu scripts/bench_env_sweep.py  # backend forced

Heads-up on duration: on an M-series Mac, 4096 envs runs ~30 s/iter, so the
default sweep takes ~15-20 min. Iterations are deterministic (see
docs/determinism.md), so one sweep is as good as three.
"""

import argparse
import csv
import os
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

# Per-iter log line: "  10   1.00   -0.2022   8504   0.782   4.44e-4   0.0120   30.4"
ITER_RE = re.compile(
    r"^\s*(\d+)\s+[-\d.]+\s+[-\d.]+\s+\d+\s+[-\d.]+\s+[\d.eE+-]+\s+[-\d.]+\s+([\d.]+)\s*$"
)


def parse_iter_seconds(log_text: str) -> dict[int, float]:
    """iteration -> wall seconds, from the trainer's periodic stat lines."""
    out = {}
    for line in log_text.splitlines():
        m = ITER_RE.match(line)
        if m:
            out[int(m.group(1))] = float(m.group(2))
    return out


def run_one(binary: Path, iters: int, envs: int, keep_log: Path | None) -> list[float]:
    with tempfile.TemporaryDirectory() as td:
        ckpt = Path(td) / "bench.safetensors"
        proc = subprocess.run(
            [str(binary), str(iters), str(envs), str(ckpt)],
            capture_output=True,
            text=True,
        )
        log = proc.stdout + proc.stderr
    if keep_log:
        keep_log.write_text(log)
    if proc.returncode != 0:
        sys.exit(f"trainer failed at {envs} envs (exit {proc.returncode}):\n{log[-2000:]}")
    samples = [s for it, s in sorted(parse_iter_seconds(log).items()) if it != 0]
    if not samples:
        sys.exit(f"no non-warmup iter lines parsed at {envs} envs — is --iters >= 11?\n{log[-1000:]}")
    return samples


def plot(rows: list[tuple[int, float, float, float]], png: Path, backend: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink2, series = "#fcfcfb", "#0b0b0b", "#52514e", "#2a78d6"
    envs = [r[0] for r in rows]
    mean = [r[1] for r in rows]

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)

    ax.plot(envs, mean, color=series, linewidth=2, marker="o", markersize=6, zorder=3)
    if len(rows) and any(r[2] != r[3] for r in rows):
        ax.fill_between(envs, [r[2] for r in rows], [r[3] for r in rows],
                        color=series, alpha=0.15, linewidth=0, zorder=2)

    # Sparse benchmark curve: each point carries its value (ink, not series color).
    for x, y in zip(envs, mean):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=9, color=ink2)

    ax.set_xscale("log", base=2)
    ax.set_xticks(envs)
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.minorticks_off()
    ax.set_ylim(bottom=0)
    ax.set_xlabel("environments", color=ink2)
    ax.set_ylabel("seconds / training iteration", color=ink2)
    ax.set_title("G1 biped_train_gpu — iteration time vs env count", color=ink, loc="left")
    ax.text(0, 1.02, "", transform=ax.transAxes)
    ax.tick_params(colors=ink2)
    ax.grid(axis="y", color="#e6e5e2", linewidth=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d8d7d3")
    ax.annotate(f"backend={backend}  ·  min–max band over sampled iters",
                (0, -0.16), xycoords="axes fraction", fontsize=8, color=ink2)

    fig.tight_layout()
    fig.savefig(png, facecolor=surface, bbox_inches="tight")
    print(f"wrote {png}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--envs", default="256,512,1024,2048,4096",
                    help="comma-separated env counts (default 256,512,1024,2048,4096)")
    ap.add_argument("--iters", type=int, default=21,
                    help="iters per run; must be >= 11 to get a non-warmup sample (default 21)")
    ap.add_argument("--bin", default="target/release/biped_train_gpu",
                    help="trainer binary (default target/release/biped_train_gpu)")
    ap.add_argument("--out", default="bench_env_sweep",
                    help="output basename for .csv/.png (default bench_env_sweep)")
    ap.add_argument("--keep-logs", action="store_true",
                    help="save each run's log next to the outputs")
    args = ap.parse_args()

    binary = Path(args.bin)
    if not binary.exists():
        sys.exit(f"{binary} not found — build it first (see scripts/train.sh for features)")
    env_counts = [int(e) for e in args.envs.split(",")]
    backend = os.environ.get("KHAL_BACKEND", "auto")

    rows = []
    for envs in env_counts:
        print(f"[{envs} envs] running {args.iters} iters...", flush=True)
        keep = Path(f"{args.out}_{envs}.log") if args.keep_logs else None
        samples = run_one(binary, args.iters, envs, keep)
        row = (envs, statistics.mean(samples), min(samples), max(samples))
        rows.append(row)
        print(f"[{envs} envs] {row[1]:.2f} s/iter (n={len(samples)}, min {row[2]:.2f}, max {row[3]:.2f})",
              flush=True)

    csv_path = Path(f"{args.out}.csv")
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["envs", "s_per_iter_mean", "s_per_iter_min", "s_per_iter_max"])
        w.writerows(rows)
    print(f"wrote {csv_path}")

    try:
        plot(rows, Path(f"{args.out}.png"), backend)
    except ImportError:
        print("matplotlib not available — CSV written, skipping the PNG")


if __name__ == "__main__":
    main()

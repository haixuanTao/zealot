#!/usr/bin/env python3
"""Tail a `biped_train_gpu` log and stream its metrics to Weights & Biases.

The Rust trainer prints a fixed table (no W&B integration), so this sidecar
parses each data row and logs it — no changes to the trainer needed. It
backfills any rows already in the file, then follows the file live (like
`tail -f`). Finishes the run after the trainer stops writing.

Usage:
  python3 examples/biped/wandb_logger.py <logfile> <run_name> [project]
e.g.
  python3 examples/biped/wandb_logger.py /tmp/train_v7.log v7 zealot-biped
"""
import sys
import time
import wandb

logfile = sys.argv[1]
run_name = sys.argv[2] if len(sys.argv) > 2 else "run"
project = sys.argv[3] if len(sys.argv) > 3 else "zealot-biped"

# Columns printed by biped_train_gpu: iter curr step_rew falls torso_z lr kl sec
COLS = ["iter", "curr", "step_rew", "falls", "torso_z", "lr", "kl", "sec"]


def parse(line):
    f = line.split()
    if len(f) != len(COLS):
        return None
    try:
        return (
            int(f[0]),
            {
                "curriculum": float(f[1]),
                "reward": float(f[2]),
                "falls": int(f[3]),
                "torso_z": float(f[4]),
                "lr": float(f[5]),
                "kl": float(f[6]),
                "sec_per_iter": float(f[7]),
            },
        )
    except ValueError:
        return None  # header row ("iter curr ...") or non-numeric line


def parse_rb(line):
    """Parse the structured per-component reward line emitted by the trainer:
      [rb] iter <it> name=val ... term_illegal=N term_fell=N term_timeout=N samples=N
    Returns (iter, {metric: value}) or None. Reward terms are namespaced under
    `reward/`, termination counts under `term/`.
    """
    if not line.startswith("[rb]"):
        return None
    f = line.split()
    if len(f) < 3 or f[1] != "iter":
        return None
    try:
        step = int(f[2])
    except ValueError:
        return None
    metrics = {}
    for tok in f[3:]:
        if "=" not in tok:
            continue
        key, _, val = tok.partition("=")
        try:
            v = float(val)
        except ValueError:
            continue
        if key.startswith("term_"):
            metrics[f"term/{key[5:]}"] = int(v)
        elif key == "samples":
            metrics["term/samples"] = int(v)
        else:
            metrics[f"reward/{key}"] = v
    return (step, metrics) if metrics else None


run = wandb.init(project=project, name=run_name, config={"logfile": logfile})
print(f"[wandb] logging {logfile} → {project}/{run_name}  ({run.url})")

# The trainer prints the table row and the `[rb]` component line for the SAME
# iter, back-to-back. Buffer metrics per step and flush when the step advances
# so both land in one W&B history row (steps must be non-decreasing).
last_step = -1
pending_step = None
pending = {}


def flush():
    global pending_step, pending, last_step
    if pending_step is not None and pending and pending_step > last_step:
        wandb.log(pending, step=pending_step)
        last_step = pending_step
    pending_step, pending = None, {}


def stage(step, metrics):
    global pending_step, pending
    if step != pending_step:
        flush()
        pending_step = step
    pending.update(metrics)


idle = 0.0
with open(logfile) as fh:
    while True:
        line = fh.readline()
        if not line:
            # No new data. Give up after 3 min of silence (run finished).
            time.sleep(2.0)
            idle += 2.0
            if idle > 180.0:
                break
            continue
        idle = 0.0
        p = parse_rb(line) or parse(line)
        if p is None:
            continue
        stage(*p)

flush()
wandb.finish()
print(f"[wandb] done at iter {last_step}")

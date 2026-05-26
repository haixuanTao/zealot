# Pendulum examples — getting started

A self-contained tour of GPU rigid-body simulation and reinforcement learning on
an inverted pendulum, running on [nexus](https://github.com/dimforge/nexus)'s
batched WebGPU physics pipeline.

These were ported verbatim from nexus-rl's `pendulum_headless` crate. They drive
`nexus3d` **directly** — they do *not* go through `zealot-env` / `zealot-rl` yet
(those crates are still scaffolds). Think of this folder as the working prototype
that the `zealot-*` crates will eventually be factored out of.

All programs are **headless**: they print scores/metrics to stdout, and a few can
record a trajectory CSV that the Python scripts here turn into an mp4.

## Prerequisites

1. **The `cargo-gpu` toolchain.** `nexus3d` and `vortx` compile their shaders from
   Rust via [`cargo-gpu`](https://github.com/Rust-GPU/cargo-gpu):
   ```sh
   cargo install cargo-gpu
   ```
2. **A WebGPU-capable GPU** — Metal (macOS), Vulkan, or DX12. No display needed.
3. **The sibling dimforge checkouts.** The pendulum build pulls `nexus3d` by path
   and redirects `khal`/`vortx`/`parry3d`/`rapier3d` to local forks via the
   `[patch.crates-io]` table in the workspace `Cargo.toml`. Those paths are
   relative (`../khal`, `../vortx`, …), so all the repos must sit side by side:
   ```
   work/
   ├── zealot/        ← you are here
   ├── nexus-cuda/    (provides crates/nexus3d)
   ├── khal/
   ├── vortx/
   ├── parry/
   └── rapier/
   ```
   A standalone clone of `zealot` alone will **not** build these examples.
4. **For video rendering only:** Python 3 with `numpy` + `matplotlib`, and
   `ffmpeg` on `PATH`.

> The first build is slow — it compiles the physics engine and its shaders. Use
> `--release`; the simulation is far too slow in debug.

## Quick start

Train a pendulum to swing up and balance — from scratch, end to end. Each env
starts hanging (pole down) and PPO learns, from reward alone, to swing it up and
hold it vertical. The policy and value MLPs train on the CPU while many
randomized rollouts run in parallel on the GPU; you'll see the mean reward climb
over iterations as it discovers the swing-up.

```sh
cargo run --release --example pendulum_ppo --features pendulum
```

Training runs 45 iterations and takes a couple of minutes. **The CSV is written
only at the very end** — wait for the run to print `recorded … → /tmp/pendulum_ppo.csv`
before rendering (don't render in a parallel terminal). Then turn it into a video
with one more command — see [Visualizing it](#visualizing-it).

Pass a seed to vary the run, e.g. `-- 7`. The rest of this folder fills in the
steps around it: the physics and a hand-tuned controller it improves on, simpler
learners, and the same idea with the policy trained on the GPU — see below.

## The examples, in learning order

Everything below needs `--features pendulum`. `pendulum_gpu_policy` additionally
needs `gpu` (it trains with vortx). Arguments come after `--`.

### 1. Physics & control

| Example | What it does | Run |
| --- | --- | --- |
| `inverted_pendulum` | Single rod on a revolute joint; a PD velocity-motor balances it upright vs. an unactuated baseline that falls. | `cargo run --release --example inverted_pendulum --features pendulum` |
| `pendulum2dof` | Rod on a *ball* joint (tips any direction); two-axis PD holds it vertical. Optional CSV out. | `cargo run --release --example pendulum2dof --features pendulum -- /tmp/pend2dof.csv` |
| `pendulum_batch` | `N` randomized 2-DOF pendulums packed into one batched GPU state and stepped together — the vectorized-env substrate RL needs. Reports score distribution + throughput. | `cargo run --release --example pendulum_batch --features pendulum -- 4096 7` |
| `pendulum_reset` | Verifies nexus's per-env reset (`reset_env_from`) is bit-for-bit correct. | `cargo run --release --example pendulum_reset --features pendulum` |
| `pendulum_smoke` | *Optional GPU sanity check* — a passive 20-link chain swinging under gravity, prints the tip pose. Useful to confirm the `nexus3d` path runs after the first build. | `cargo run --release --example pendulum_smoke --features pendulum` |

### 2. Reinforcement learning (policy lives on CPU, physics on GPU)

| Example | Algorithm | Run |
| --- | --- | --- |
| `pendulum_learn` | CEM (gradient-free) learns a *linear* balance policy; whole population evaluated in one batched rollout. | `cargo run --release --example pendulum_learn --features pendulum -- 1` |
| `pendulum_pg` | REINFORCE: an MLP (4→16 tanh→2) Gaussian policy, hand-written backprop + Adam. | `cargo run --release --example pendulum_pg --features pendulum -- 1` |
| `pendulum_ppo` | PPO **swing-up**: starts hanging, must swing up and hold. Separate policy + value MLPs, GAE(λ), clipped surrogate. | `cargo run --release --example pendulum_ppo --features pendulum -- 1` |

The trailing number is the RNG seed (default `1`).

### 3. RL with the policy *on the GPU*

| Example | What it does | Run |
| --- | --- | --- |
| `pendulum_gpu_policy` | Trains an MLP policy entirely on the GPU with vortx (GEMM + tanh forward, hand-rolled backward, Adam), then reads the weights back and deploys them in the batched physics to confirm it balances. | `cargo run --release --example pendulum_gpu_policy --features "pendulum gpu"` |

## Visualizing it

The examples are headless — they print metrics. To *watch* a pendulum, an example
records a per-step pose CSV and a Python script (needs `numpy`/`matplotlib`/`ffmpeg`)
turns it into an mp4.

**The trained PPO policy (the quick start).** `pendulum_ppo` automatically records
a deterministic rollout of the learned policy after training, by default to
`/tmp/pendulum_ppo.csv`. Render that swing-up:

```sh
cargo run --release --example pendulum_ppo --features pendulum   # writes /tmp/pendulum_ppo.csv
python3 render_pend2dof.py /tmp/pendulum_ppo.csv /tmp/pendulum_ppo.mp4
```

(Pass a second arg to change the CSV path, e.g. `-- 1 /tmp/run.csv`.)

**Other recordable demos** — same renderer family, fed by the hand-tuned controllers:

```sh
# 2-DOF ball-joint (PD controller)
cargo run --release --example pendulum2dof --features pendulum -- /tmp/pend2dof.csv
python3 render_pend2dof.py /tmp/pend2dof.csv /tmp/pend2dof.mp4

# single-rod revolute: controlled vs. unactuated baseline, side by side
cargo run --release --example inverted_pendulum --features pendulum -- record 45 /tmp/pendulum_traj.csv
python3 render_video.py      /tmp/pendulum_traj.csv /tmp/pendulum.mp4     # 2D side-by-side
python3 render_pendulum3d.py /tmp/pendulum_traj.csv /tmp/pendulum3d.mp4   # 3D view
```

## How this maps to zealot

These programs each rebuild the scene, step `nexus3d`, and roll their own
training loop — so they duplicate a lot of boilerplate. The plan is to factor the
shared pieces out:

- the batched env (scene build, `step`, `reset`, observations, reward) → **`zealot-env`**
- the policy / PPO / PG / Adam machinery → **`zealot-rl`**

…leaving these examples as thin drivers. Until then they're the canonical
reference for what those crates need to provide.

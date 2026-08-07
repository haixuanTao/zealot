# Getting started

Zero to "my own policy walks in a browser", in three steps.

## 1. Watch it walk (nothing to install)

Open **[haixuantao.github.io/zealot](https://haixuantao.github.io/zealot/)**.
That page is not a video — it is the training environment compiled to
WebAssembly, stepping nexus GPU physics in your browser and driving a Unitree
G1 with a trained policy at 50 Hz.

Things to try:

- **Tap the ground** — the robot walks there (a small navigator steering the
  policy's velocity command). Arrows/WASD or a gamepad drive it manually.
- **Switch engines** — the rapier.js and MuJoCo tabs step the *same
  checkpoint* through two entirely different physics engines on the same
  terrain: sim2sim validation you can watch.
- **Load another policy** — the Policy picker lists the published
  checkpoints; pasting any Hugging Face handle (`owner/repo`) or
  `.safetensors` URL loads that instead.
- **Make it harder** — terrain difficulty / roughness / slope sliders. The
  URL tracks your configuration, so it's shareable.

The WebGPU tab needs Chrome; Safari/Firefox/iOS open on the CPU engines
automatically.

## 2. Train the humanoid

Prerequisites (one-time): Rust, plus the
[`cargo-gpu`](https://github.com/Rust-GPU/cargo-gpu) toolchain for the
Rust-GPU shader builds — see [development.md](development.md) for the exact
versions and gotchas. Then:

```sh
# Auto-detects the backend: native CUDA (+cuTile) on an NVIDIA box,
# WebGPU/Metal otherwise. ~71 k env-steps/s at N=4096 on a 5090.
scripts/train.sh 50000 4096 my_policy.safetensors
```

The **production config is the code default** — bare `train.sh` trains the
29-DOF G1 with terrain curriculum, AGILE domain randomization, pushes, motor
delay, 5-frame observation history, and the production reward weights. Every
knob stays overridable via its `BIPED_*` env var; knobs consumed by more than
one crate are declared exactly once in `zealot_env::knobs` (one source of
truth — a default cannot fork between the env and the trainer). The startup
log echoes the effective config, including
`actor obs: 265 = 5 frames x 53 dims`.

The checkpoint is a plain safetensors file: actor-critic weights plus the
observation-normalizer state. Sim2sim harnesses for MuJoCo / Genesis / Isaac
live in `scripts/`.

## 3. Watch *your* policy walk

Upload the checkpoint to any Hugging Face model repo and paste the repo
handle into the demo's Policy box — or hand someone a link:

```
https://haixuantao.github.io/zealot/?ckpt=your-name/your-repo
```

The demo detects the observation layout from the checkpoint itself.

To make a policy the one the site loads by default, tag it as the latest
release — the demo fetches the fixed name `g1_walk_latest.safetensors` from
the [policy repo](https://huggingface.co/haixuantao/zealot-g1-locomotion)
at boot, so this needs no site redeploy:

```sh
scripts/publish_policy.sh my_policy.safetensors g1_v27_iter50000
```

---

Next: [explanation.md](explanation.md) for how and why the stack is built
this way · [benchmarks.md](benchmarks.md) for the numbers ·
[development.md](development.md) for toolchain depth.

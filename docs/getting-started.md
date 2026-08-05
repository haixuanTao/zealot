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
# RTX-class GPU; ~71 k env-steps/s at N=4096 on a 5090
BIPED_ROBOT=g1_29dof_agile BIPED_CUTILE_GEMM=1 BIPED_TERRAIN=1 \
  cargo run --release --example biped_train_gpu \
  --features "gpu biped_gpu cutile" -- 50000 4096 my_policy.safetensors
```

The checkpoint is a plain safetensors file: actor-critic weights plus the
observation-normalizer state. Sim2sim harnesses for MuJoCo / Genesis / Isaac
live in `scripts/` and `examples/biped/`.

## 3. Watch *your* policy walk

Upload the checkpoint to any Hugging Face model repo and paste the repo
handle into the demo's Policy box — or hand someone a link:

```
https://haixuantao.github.io/zealot/?ckpt=your-name/your-repo
```

The demo detects the observation layout from the checkpoint itself.

---

Next: [explanation.md](explanation.md) for how and why the stack is built
this way · [benchmarks.md](benchmarks.md) for the numbers ·
[development.md](development.md) for toolchain depth.

# zealot website

The [live demo site](https://haixuantao.github.io/zealot/): a Vite + React
page embedding the zealot **training environment compiled to wasm** — nexus
GPU physics as WebGPU compute shaders, with the control loop (observation
assembly → policy GEMMs → PD targets) fully GPU-resident — plus two CPU
sim2sim tabs (rapier.js and the official MuJoCo wasm build) stepping the same
policy over the same terrain.

The wasm demos are the `g1_terrain_web` / `g1_web` cargo examples of this
workspace (shared implementation: `examples/biped/g1_web_demo.rs`); nothing is
fetched from sibling checkouts at runtime — MJCF, meshes and the default
checkpoint are embedded, and other checkpoints load from Hugging Face by
handle or URL.

## Build & run locally

Prerequisites: Node.js ≥ 20, Rust with the `wasm32-unknown-unknown` target
(shader compilation needs the [`cargo-gpu`](https://github.com/Rust-GPU/cargo-gpu)
toolchain), and the sibling `../nexus` / `../khal-unified` checkouts
the workspace `[patch]` table points at.

```bash
cd website
npm install
./scripts/build-demos.sh        # wasm demos → public/demos/  (SKIP_WASM_OPT=1 for fast iteration)
npm run build                   # site → dist/
npm run preview                 # serve dist/ locally
```

The WebGPU tab needs Chrome (or another Chromium). Safari, Firefox and iOS
are detected and open on the CPU tabs instead.

## Deploy

```bash
npm run deploy                  # build + push dist/ to the gh-pages branch
```

Run it **from `website/`**. Demos under `public/demos/` are gitignored build
output — rebuild them first if the Rust side changed. The page self-heals a
cached `index.html` after a deploy (content-hashed bundle comparison), but
the demo's `pkg/*.wasm` is not content-hashed: hard-reload when testing a
fresh deploy.

## Demo URL knobs

`?n=` robots · `?lvl=` terrain difficulty (0–19) · `?amp=` roughness % ·
`?slope=` degrees · `?ckpt=` policy (HF `owner/repo`, `owner/repo/file
.safetensors`, or any URL) · `?diag=1` policy-I/O readback diagnostics ·
`?prof=1` per-kernel GPU timings to the console · `?snap=0` pipelined pose
readback (default is the per-frame blocking snapshot — every rendered frame
is a fresh sim pose) · `?phys=`/`?fused=`/`?dpr=` perf A/B knobs.

The site page itself accepts the same scene knobs and writes them back to the
address bar on Apply, so a configured demo is a shareable link.

## Test harnesses (puppeteer, `scripts/`)

- `check_perf.mjs` — fps / real-time % / **walking speed** / falls matrix
  (`CASES='n=1&lvl=4|n=3&lvl=4' node scripts/check_perf.mjs`; a frozen sim
  fails loudly — never trust a run that doesn't assert `spd > 0`).
- `check-browser-fallback.mjs` — the per-browser default-tab matrix (spoofed
  UAs + a real-Firefox pass; `SITE_URL=` to point at production).
- `check-demo.mjs` — boot check with periodic screenshots (`DEMO_URL=`).
- `shader_check.mjs` — hooks `createShaderModule` before the wasm boots and
  reports which kernels the browser's WGSL validation rejects, dumping the
  generated WGSL so the `:299:21`-style line numbers in the error resolve to
  readable source (`DEMO_URL=`, `OUT_DIR=`; may run headless — module creation
  is not rAF-driven).

These run headed on purpose: headless Chrome throttles rAF-driven wasm, and a
long-lived interactive Chrome profile degrades after many WebGPU loads — the
harnesses' fresh instances are the reliable measurement environment.

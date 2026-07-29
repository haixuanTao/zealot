# zealot Website

This website is built using [Docusaurus](https://docusaurus.io/) and embeds a realtime
WebAssembly demo of the Unitree G1 humanoid simulated by [nexus](https://nexus.dimforge.com)
GPU physics. It is adapted from the nexus demos website (`../../nexus/website`).

The demo is the `g1_web3` binary from the **sibling nexus checkout**
(`../../nexus/crates/examples3d/`): nexus rigid-body physics as WebGPU compute
shaders, rendered with the nexus viewer, all compiled to wasm. The G1 scene is
pre-baked by `g1_bake3` (native) from the MuJoCo Menagerie MJCF into a serialized
rapier blob embedded in the wasm module — no filesystem access at runtime.

(The zealot repo also has a LeRobot-biped demo with a trained walking policy —
`cargo run --release --example lerobot_biped_web --features lerobot_biped_web` —
native-only for now: browsers still reject two nexus-cuda kernels it depends on.)

## Prerequisites

- [Node.js](https://nodejs.org/) (v20 or later)
- [Rust](https://rustup.rs/) with the `wasm32-unknown-unknown` target
- The sibling `../../nexus` checkout (branch with the G1 demo bins) and
  `../../../mujoco_menagerie` next to it (for the one-time scene bake)
- [wasm-bindgen-cli](https://rustwasm.github.io/wasm-bindgen/) (optional: the build script
  auto-installs the version matching nexus's `Cargo.lock` into `nexus/target/` if the
  global one doesn't match)
- `wasm-opt` (optional, from [binaryen](https://github.com/WebAssembly/binaryen) — shrinks the
  module; skipped if absent or if `SKIP_WASM_OPT=1`)

Install the required Rust tooling:

```bash
rustup target add wasm32-unknown-unknown
```

## Installation

```bash
npm install
```

## Building the Demo

```bash
npm run build:demos
```

The demo is built to `static/demos/g1_web/` and will be included in the website.
Note that it requires a WebGPU-enabled browser to run (Chrome/Edge today; Firefox
behind `dom.webgpu.enabled`; Safari currently mis-simulates the contacts).

To iterate quickly on the demo itself, run it natively instead:

```bash
cd ../../nexus && cargo run --release -p nexus_examples_3d --bin g1_web3
```

## Local Development

```bash
npm start
```

This starts a local development server at http://localhost:3000. Site changes are
reflected live; re-run `npm run build:demos` after changing the Rust demo.

## Build for Production

Build everything (demo + website):

```bash
npm run build:all
```

Or build just the website (assumes the demo is already built):

```bash
npm run build
```

The static site is generated in the `build` directory. `./publish.sh` builds
everything and copies the `.htaccess` in; rsync `build/` to any static host.

## Project Structure

```
website/
├── src/
│   ├── pages/          # React pages (index, demos)
│   └── css/            # Custom styles
├── static/
│   ├── demos/          # Compiled WASM demo (build output, not committed)
│   └── img/            # Images and logos
├── scripts/
│   └── build-demos.sh  # Demo build script
└── docusaurus.config.ts
```

# Building & development

Moved from the README — the full toolchain setup (cargo-gpu, the native-CUDA
cubin chain) and the repo's hook/test conventions.

## Building

`zealot-env` will depend on `nexus3d`, whose Rust-GPU shaders require
[`cargo-gpu`](https://github.com/Rust-GPU/cargo-gpu):

```sh
cargo install cargo-gpu
```

### Native CUDA (cuda-oxide)

The native-CUDA fast path embeds prebuilt sm_120 cubins of the nexus + vortx
shader crates. They are compiled by **upstream NVlabs
[cuda-oxide](https://github.com/NVlabs/cuda-oxide)** through its unified
host-target interception (no nvptx64 cross-target step, no compiler fork) and
lowered with full O3 (`opt` -> `llc` -> `ptxas`). The whole chain is one
script:

```sh
scripts/build_cubins.sh   # backend -> .ll -> O3 cubins -> trainer re-embed
```

The backend is cloned + built automatically (cached under
`~/.cache/zealot/cuda-oxide`); `BACKEND_REV` / `CUDA_OXIDE_BACKEND` override
it.

Cubins land in `./cubins/` (gitignored), which the repo's
`.cargo/config.toml [env]` feeds to the embed step — so
`cargo run --features "gpu biped_gpu cutile"` needs no exported env. The two
machine-specific paths go in `~/.cargo/config.toml` once per box:

```toml
[env]
CUDA_TOOLKIT_PATH = "/path/to/cuda-13-headers-shim"   # CUDA >= 13.2 headers (build)
CUTILE_TILEIRAS_PATH = "/path/to/cuda-13.3/bin/tileiras"  # cuTile runtime JIT
```

Prerequisites: sibling checkouts `../nexus`, `../vortx-unified`,
`../khal-unified` (see the `[patch.crates-io]` table), a CUDA >= 12.8 `ptxas`,
and libdevice. The backend builds from stock
[NVlabs/cuda-oxide](https://github.com/NVlabs/cuda-oxide) `main` (>= `6247276`;
our two blocking fixes, [#518](https://github.com/NVlabs/cuda-oxide/pull/518)
and [#520](https://github.com/NVlabs/cuda-oxide/pull/520), merged 2026-07-28). Cubins are embedded at trainer build time via
`CUDA_OXIDE_SHADERS_PTX_NEXUS_RBD_SHADERS3D` / `..._VORTX_SHADERS` — rebuild
the trainer after rebuilding cubins.


## Configuration: one source of truth

Runtime knobs are `BIPED_*` / `NEXUS_*` env vars, and **the default IS the
production training config** — a bare run needs zero of them. Any knob read
by more than one crate (env + trainer + web demo) is declared exactly once in
`zealot-env/src/knobs.rs` (name, default, doc); consumers share the static,
so defaults cannot diverge between sites. Single-site knobs stay at their
call site — move them to `knobs.rs` the moment a second reader appears. Wasm
demos configure via `Knob::set_override` (no process env in the browser).

## Development

Versioned git hooks in `.githooks/` enforce formatting, warnings, and tests.
Enable them once per clone:

```sh
git config core.hooksPath .githooks
```

- **pre-commit** runs `cargo fmt --check` (workspace members only) and
  `cargo check --workspace --all-targets` with `RUSTFLAGS="-D warnings"`, so any
  warning fails the commit.
- **pre-push** runs `cargo test --workspace` (also with `-D warnings`) so the
  test suite only gates pushes, keeping individual commits fast.

The `gpu` feature is intentionally left off everywhere — its checks need the
`cargo-gpu` toolchain. To run the same checks by hand:

```sh
cargo fmt -p zealot -p zealot-env -p zealot-rl
RUSTFLAGS="-D warnings" cargo check --workspace --all-targets
RUSTFLAGS="-D warnings" cargo test --workspace
```

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
scripts/full_unified_chain.sh   # backend -> .ll -> O3 cubins -> trainer
```

Prerequisites: sibling checkouts `../nexus`, `../vortx-unified`,
`../khal-unified` (see the `[patch.crates-io]` table), a CUDA >= 12.8 `ptxas`,
and libdevice. The backend builds from stock
[NVlabs/cuda-oxide](https://github.com/NVlabs/cuda-oxide) `main` (>= `6247276`;
our two blocking fixes, [#518](https://github.com/NVlabs/cuda-oxide/pull/518)
and [#520](https://github.com/NVlabs/cuda-oxide/pull/520), merged 2026-07-28). Cubins are embedded at trainer build time via
`CUDA_OXIDE_SHADERS_PTX_NEXUS_RBD_SHADERS3D` / `..._VORTX_SHADERS` — rebuild
the trainer after rebuilding cubins.

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

//! <!-- The crate-level docs below are the single source of truth for the
//! training guides. They are pulled verbatim from `docs/train-on-5090.md`
//! and `docs/train-on-macos.md` via `include_str!`, so `cargo doc` renders
//! exactly those files and they can never drift. Edit the markdown, not
//! this file. -->
#![doc = include_str!("../docs/train-on-5090.md")]
#![doc = "\n\n---\n\n"]
#![doc = include_str!("../docs/train-on-macos.md")]

// This crate (the `zealot` workspace-root package) has no library API — it hosts
// the runnable GPU-training examples under `examples/`. This `lib.rs` exists only
// to give rustdoc a crate root so the training guide is generated from source.

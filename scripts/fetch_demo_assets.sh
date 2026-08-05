#!/bin/bash
# Fetch the demo/bench assets from Hugging Face into their build locations.
# Assets are NOT in git (binary blobs); the wasm demos include_bytes! them at
# compile time and the bench pages serve them statically, so run this once
# before `cargo build --example g1_web ...` or `website/scripts/build-demos.sh`
# (build-demos calls it automatically). Skips files that already exist.
#
# Source of truth: https://huggingface.co/haixuantao/zealot-g1-locomotion (assets/)
set -eo pipefail
BASE=https://huggingface.co/haixuantao/zealot-g1-locomotion/resolve/main/assets
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

fetch () { # $1 remote name, $2 local path
  [ -s "$ROOT/$2" ] && return 0
  mkdir -p "$(dirname "$ROOT/$2")"
  echo "fetch $1 -> $2"
  curl -fsSL "$BASE/$1" -o "$ROOT/$2"
}

fetch g1_visuals_12dof.bin   examples/biped/assets/g1_visuals_12dof.bin
fetch g1_visuals_29dof.bin   examples/biped/assets/g1_visuals_29dof.bin
fetch g1_walk_v24.safetensors examples/biped/assets/g1_walk_v24.safetensors
fetch g1_walk_v26.safetensors examples/biped/assets/g1_walk_v26.safetensors
fetch robot.xml              examples/biped/assets/robot.xml
fetch g1_mjcf_web.xml        website/public/bench/g1_mjcf_web.xml
fetch g1_model.json          website/public/bench/g1_model.json
fetch g1_visuals_12dof.bin   website/public/bench/g1_visuals_12dof.bin
fetch g1_visuals_mj29.bin    website/public/bench/g1_visuals_mj29.bin
echo "assets ready"

#!/bin/bash
# Publish a trained checkpoint to the Hugging Face policy repo and re-tag it
# as the release the live site loads by default.
#
#   scripts/publish_policy.sh ckpt.safetensors [g1_v27_iter50000]
#
# Uploads the file twice: once under its versioned name (second arg, default
# the file's own basename), and once as `g1_walk_latest.safetensors` — the
# fixed name the website demo and the sim2sim bench pages fetch at boot, so
# a release needs NO site redeploy. Needs `hf auth login` once.
#
# Browser caveat: the web demos build 45- or 48-dim obs frames. A checkpoint
# with wider frames (e.g. the 53-dim step-cue policies) uploads fine but the
# site will refuse it and fall back to the embedded policy — extend the obs
# shader (zealot-obs-shaders) before tagging one of those as latest.
set -eo pipefail
REPO=haixuantao/zealot-g1-locomotion

ckpt="$1"
[ -s "$ckpt" ] || { echo "usage: $0 ckpt.safetensors [release_name]" >&2; exit 1; }
name="${2:-$(basename "$ckpt" .safetensors)}"

hf upload "$REPO" "$ckpt" "$name.safetensors" \
    --commit-message "release $name"
hf upload "$REPO" "$ckpt" g1_walk_latest.safetensors \
    --commit-message "tag $name as g1_walk_latest"
echo "released: https://huggingface.co/$REPO/blob/main/$name.safetensors (tagged latest)"

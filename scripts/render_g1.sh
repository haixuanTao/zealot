#!/bin/bash
export PATH="$HOME/.cargo/bin:$PATH"
cd ~/Documents/work/zealot
export CUDA_OXIDE_SHADERS_PTX_NEXUS_RBD_SHADERS3D=$HOME/nexus_ptx/nexus_rbd_shaders3d.cubin
export CUDA_OXIDE_SHADERS_PTX_VORTX_SHADERS=$HOME/nexus_ptx/vortx_shaders.cubin
export NEXUS_SMALL_SORT=1 BIPED_CUTILE_GEMM=1 BIPED_CUDA=1 BIPED_ROBOT=g1_29dof_agile
export BIPED_OBS_HISTORY=5 BIPED_CONTACT_SENSE=1 BIPED_CONTACT_CAP=128 BIPED_CONTACT_REDUCE=1 NEXUS_FIXED_GRID=1
echo "[rollout] loading g1_sense_v5, dumping 400-step rollout..."
cargo run --release --example biped_render_nexus --features "gpu biped_gpu cuda_backend" \
    -- 0 400 /tmp/g1_rollout.json ~/overnight/g1_sense_v5.safetensors
echo "[rollout] exit=$?"

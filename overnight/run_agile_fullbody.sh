#!/bin/bash
# TRUE-parity AGILE run (2026-07-16 pm): FULL-BODY G1 (31 nv, wrists welded)
# with AGILE gains, terrain curriculum, GPU-side actuator delay, obs history,
# AGILE pushes + 25% standing commands. 15000 iters @4096 (~20 h).
cd ~/Documents/work/zealot
D=~/Documents/work/zealot/overnight
echo "agile-fullbody start $(date)" >> $D/STATUS
env NEXUS_SMALL_SORT=1 BIPED_CUTILE_GEMM=1 BIPED_CUDA=1 \
    BIPED_TERRAIN=1 BIPED_ROBOT=g1_29dof_agile \
    BIPED_MOTOR_DELAY=0,4 BIPED_OBS_HISTORY=5 BIPED_MIRROR_AUG=1 \
    BIPED_PUSH_VEL=0.5 BIPED_PUSH_ANGVEL=0.25 BIPED_STAND_PROB=0.25 \
    BIPED_GRAPH=1 BIPED_CONTACT_CAP=128 BIPED_CONTACT_REDUCE=1 \
    ./target/release/examples/biped_train_gpu 15000 4096 $D/g1_agile_fullbody.safetensors \
    > $D/g1_agile_fullbody.log 2>&1
echo "agile-fullbody exit $? $(date)" >> $D/STATUS

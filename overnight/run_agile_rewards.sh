#!/usr/bin/env bash
# AGILE-exact reward-parity experiment: WBC-AGILE's G1 term set (no stepping
# rewards, no extra stand income, stds 0.2, height target 0.72), everything
# else at full parity (terrain curriculum, delay, history, pushes). Fresh
# start — the g1_agile_fullbody checkpoint is a converged stand-still optimum.
set -u
D=/home/baguette/Documents/work/zealot/overnight
cd /home/baguette/Documents/work/zealot
env NEXUS_SMALL_SORT=1 BIPED_CUTILE_GEMM=1 BIPED_CUDA=1 \
    BIPED_AGILE_REWARDS=1 BIPED_POWER_W=0 BIPED_LR_MIN=1e-5 BIPED_NORM_FREEZE=1 \
    BIPED_TERRAIN=1 BIPED_ROBOT=g1_29dof_agile \
    BIPED_MOTOR_DELAY=0,4 BIPED_OBS_HISTORY=5 BIPED_MIRROR_AUG=1 \
    BIPED_PUSH_VEL=0.5 BIPED_PUSH_ANGVEL=0.25 BIPED_STAND_PROB=0.25 \
    BIPED_GRAPH=1 BIPED_CONTACT_CAP=128 BIPED_CONTACT_REDUCE=1 \
    ./target/release/examples/biped_train_gpu 15000 4096 $D/g1_agile_rewards.safetensors \
    > $D/g1_agile_rewards.log 2>&1
echo "agile-rewards exit $? $(date)" >> $D/STATUS

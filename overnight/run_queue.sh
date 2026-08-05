#!/bin/bash
# Overnight sequential training queue — 2026-07-15. G1 12-DOF, AGILE-parity
# flags (delay 0..4 substeps + 5-frame obs history) + mirror aug, 4096 envs.
# Logs + checkpoints land in this directory. Survives the launching session.
cd ~/Documents/work/zealot
D=~/Documents/work/zealot/overnight
COMMON="NEXUS_SMALL_SORT=1 BIPED_CUTILE_GEMM=1 BIPED_CUDA=1 BIPED_ROBOT=g1 BIPED_MOTOR_DELAY=0,4 BIPED_OBS_HISTORY=5 BIPED_MIRROR_AUG=1"
BIN=./target/release/examples/biped_train_gpu

echo "queue start $(date)" > $D/STATUS

echo "run1 velocity: start $(date)" >> $D/STATUS
env $COMMON $BIN 4000 4096 $D/g1_velocity.safetensors > $D/g1_velocity.log 2>&1
echo "run1 velocity: exit $? $(date)" >> $D/STATUS

echo "run2 stand: start $(date)" >> $D/STATUS
env $COMMON BIPED_STAND_FRAC=1.0 BIPED_PUSH_VEL=0.5 $BIN 2000 4096 $D/g1_stand.safetensors > $D/g1_stand.log 2>&1
echo "run2 stand: exit $? $(date)" >> $D/STATUS

echo "run3 push-recovery: start $(date)" >> $D/STATUS
env $COMMON BIPED_PUSH_VEL=1.0 BIPED_PUSH_ANGVEL=0.25 $BIN 4000 4096 $D/g1_pushrecovery.safetensors > $D/g1_pushrecovery.log 2>&1
echo "run3 push-recovery: exit $? $(date)" >> $D/STATUS

echo "queue DONE $(date)" >> $D/STATUS

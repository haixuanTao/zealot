#!/bin/bash
# AGILE-matched G1 velocity training (2026-07-16): WBC-AGILE's perturbation
# recipe (pushes ±0.5 m/s + ±0.25 rad/s every ~2-5 s), AGILE-parity actuator
# delay (0..4 substeps) + 5-frame obs history, mirror aug, 5000 iters @4096
# (= AGILE's max_iterations; ~492M env-steps).
cd ~/Documents/work/zealot
D=~/Documents/work/zealot/overnight
echo "agile-matched start $(date)" >> $D/STATUS
env NEXUS_SMALL_SORT=1 BIPED_CUTILE_GEMM=1 BIPED_CUDA=1 \
    BIPED_ROBOT=g1 BIPED_MOTOR_DELAY=0,4 BIPED_OBS_HISTORY=5 BIPED_MIRROR_AUG=1 \
    BIPED_PUSH_VEL=0.5 BIPED_PUSH_ANGVEL=0.25 \
    ./target/release/examples/biped_train_gpu 5000 4096 $D/g1_agile_matched.safetensors \
    > $D/g1_agile_matched.log 2>&1
echo "agile-matched exit $? $(date)" >> $D/STATUS

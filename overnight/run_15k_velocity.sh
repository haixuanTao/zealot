#!/bin/bash
# 15k-iteration G1 velocity-tracking run (2026-07-16). AGILE-matched modeling:
# actuator delay 0..4 substeps, 5-frame obs history, pushes ±0.5 m/s + ±0.25
# rad/s every ~2-5 s, full ±0.5 m/s command range. Gait clock 6 @ 0.9 s period
# (G1 cadence). Long-schedule curriculum: stand 10%, full commands by 30%.
cd ~/Documents/work/zealot
D=~/Documents/work/zealot/overnight
echo "15k-velocity start $(date)" >> $D/STATUS
env NEXUS_SMALL_SORT=1 BIPED_CUTILE_GEMM=1 BIPED_CUDA=1 \
    BIPED_ROBOT=g1 BIPED_MOTOR_DELAY=0,4 BIPED_OBS_HISTORY=5 BIPED_MIRROR_AUG=1 \
    BIPED_PUSH_VEL=0.5 BIPED_PUSH_ANGVEL=0.25 \
    BIPED_MAX_CSCALE=1.0 BIPED_GAIT_CLOCK_W=6 BIPED_GAIT_PERIOD=0.9 \
    BIPED_STAND_FRAC=0.1 BIPED_RAMP_END=0.3 \
    ./target/release/examples/biped_train_gpu 15000 4096 $D/g1_velocity_15k.safetensors \
    > $D/g1_velocity_15k.log 2>&1
echo "15k-velocity exit $? $(date)" >> $D/STATUS

#!/bin/bash
# THE full AGILE-consistency run (2026-07-16): terrain curriculum + AGILE
# actuator gains + actuator delay + obs history + AGILE pushes + 25% standing
# commands. 15000 iters @4096. Replaces the flat-ground 15k run (stopped at
# ~5060 iters, not stepping — terrain is the stepping mechanism).
cd ~/Documents/work/zealot
D=~/Documents/work/zealot/overnight
echo "agile-full start $(date)" >> $D/STATUS
env NEXUS_SMALL_SORT=1 BIPED_CUTILE_GEMM=1 BIPED_CUDA=1 \
    BIPED_TERRAIN=1 BIPED_ROBOT=g1_agile \
    BIPED_MOTOR_DELAY=0,4 BIPED_OBS_HISTORY=5 BIPED_MIRROR_AUG=1 \
    BIPED_PUSH_VEL=0.5 BIPED_PUSH_ANGVEL=0.25 BIPED_STAND_PROB=0.25 \
    ./target/release/examples/biped_train_gpu 15000 4096 $D/g1_agile_full.safetensors \
    > $D/g1_agile_full.log 2>&1
echo "agile-full exit $? $(date)" >> $D/STATUS

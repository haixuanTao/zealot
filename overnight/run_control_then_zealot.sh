#!/usr/bin/env bash
# Night plan 2026-07-17→18:
# 1) CONTROL: WBC-AGILE itself, 3000 iters @4096 — ground truth for when
#    terrain levels start moving in the reference implementation (never
#    measured; without it our iter-N "not walking yet" verdicts have no
#    yardstick). Console log keeps rsl_rl's Curriculum/terrain_levels lines.
# 2) Then the zealot AGILE-rewards run (fresh, BIPED_LR_MIN=1e-5).
set -u
D=/home/baguette/Documents/work/zealot/overnight
cd /home/baguette/WBC-AGILE
WANDB_MODE=disabled OMNI_KIT_ACCEPT_EULA=YES \
    ~/isaaclab/.venv/bin/python scripts/train.py --task Velocity-G1-History-v0 \
    --headless --num_envs 4096 --max_iterations 3000 \
    > $D/agile_control_3k.log 2>&1
echo "agile-control exit $? $(date)" >> $D/STATUS
exec $D/run_agile_rewards.sh

#!/usr/bin/env bash
set -uo pipefail
cd /root/gripper-mujoco
export HF_HOME=/root/autodl-tmp/gripper-mujoco-hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MUJOCO_GL=egl PYTHONUNBUFFERED=1
base=/root/autodl-tmp/smolvla-official-reproduction-v2
OUT=$base/acquisition/q2_restart_20261005
git rev-parse HEAD > $OUT/code_commit.txt
for step in 010000 025000; do
  .venv-lerobot/bin/python scripts/demo_restart_rollouts.py \
    --checkpoint $base/full_25k_seed1000/checkpoints/$step/pretrained_model \
    --output $OUT/step_$step --task 0 --task 1 --task 2 --task 3 --demos 10 > $OUT/step_$step.log 2>&1 &
done
wait
echo "DONE $(date -Is)"; echo EXIT_STATUS=0

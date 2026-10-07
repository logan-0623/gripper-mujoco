#!/usr/bin/env bash
set -uo pipefail
cd /root/gripper-mujoco
export HF_HOME=/root/autodl-tmp/gripper-mujoco-hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MUJOCO_GL=egl PYTHONUNBUFFERED=1
P=.venv-lerobot/bin/python
base=/root/autodl-tmp/smolvla-official-reproduction-v2
OUT=$base/acquisition/e3_e3b_20261007
ck=$base/full_25k_seed1000/checkpoints
git rev-parse HEAD > $OUT/code_commit.txt
fail() { echo "FAILED: $1 $(date -Is)"; echo EXIT_STATUS=1; exit 1; }
arm() { $P scripts/window_entry_probe.py --checkpoint $ck/$1/pretrained_model --output $OUT/$2 --task 0 --task 1 --task 2 --task 3 --episodes 20 --push-steps $3 --push-action 1.0 > $OUT/$2.log 2>&1; }
echo "BATCH1 $(date -Is)"
arm 010000 k10_push0 0 & arm 025000 k25_push0 0 & arm 010000 k10_push4 4 & arm 025000 k25_push4 4 &
for job in $(jobs -p); do wait $job || fail "batch1 job $job"; done
echo "BATCH2 $(date -Is)"
arm 010000 k10_push2 2 & arm 025000 k25_push2 2 & arm 010000 k10_push6 6 & arm 025000 k25_push6 6 &
for job in $(jobs -p); do wait $job || fail "batch2 job $job"; done
echo "DONE $(date -Is)"; echo EXIT_STATUS=0

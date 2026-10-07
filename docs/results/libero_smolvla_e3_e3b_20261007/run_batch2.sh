#!/usr/bin/env bash
set -uo pipefail
cd /root/gripper-mujoco
export HF_HOME=/root/autodl-tmp/gripper-mujoco-hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MUJOCO_GL=egl PYTHONUNBUFFERED=1
P=.venv-lerobot/bin/python
OUT=/root/autodl-tmp/smolvla-official-reproduction-v2/acquisition/e3_e3b_20261007
ck=/root/autodl-tmp/smolvla-official-reproduction-v2/full_25k_seed1000/checkpoints
fail() { echo "FAILED: $1 $(date -Is)"; echo EXIT_STATUS=1; exit 1; }
arm() { $P scripts/window_entry_probe.py --checkpoint $ck/$1/pretrained_model --output $OUT/$2 --task 0 --task 1 --task 2 --task 3 --episodes 20 --push-steps $3 --push-action 1.0 > $OUT/$2.log 2>&1; }
while kill -0 6507 6509 6511 6512  2>/dev/null || true; do alive=0; for p in 6507 6509 6511 6512 ; do kill -0 $p 2>/dev/null && alive=1; done; [ $alive -eq 0 ] && break; sleep 60; done
for a in k10_push0 k25_push0 k10_push4 k25_push4; do test -f $OUT/$a/summary.json || fail "batch1 $a summary missing"; done
echo "BATCH1_DONE $(date -Is)"
echo "BATCH2 $(date -Is) (push 8 and 12 steps; 4 steps at action 1.0 moved only ~2.2 cm)"
arm 010000 k10_push8 8 & arm 025000 k25_push8 8 & arm 010000 k10_push12 12 & arm 025000 k25_push12 12 &
for job in $(jobs -p); do wait $job || fail "batch2 job $job"; done
echo "DONE $(date -Is)"; echo EXIT_STATUS=0

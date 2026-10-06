#!/usr/bin/env bash
set -uo pipefail
cd /root/gripper-mujoco
export HF_HOME=/root/autodl-tmp/gripper-mujoco-hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MUJOCO_GL=egl PYTHONUNBUFFERED=1
P=.venv-lerobot/bin/python
base=/root/autodl-tmp/smolvla-official-reproduction-v2
OUT=$base/acquisition/e1_e2_20261006
ck=$base/full_25k_seed1000/checkpoints
K10=$ck/010000/pretrained_model; K25=$ck/025000/pretrained_model
fail() { echo "FAILED: $1 $(date -Is)"; echo EXIT_STATUS=1; exit 1; }
while kill -0 10802 2>/dev/null || kill -0 10803 2>/dev/null; do sleep 60; done
for s in 010000 025000; do test -f $OUT/e1/step_$s/summary.json || fail "e1 step_$s summary missing"; done
echo "E1_DONE $(date -Is)"
echo "E2 $(date -Is)"
run_e2() { $P scripts/grasp_window_switch.py --base $1 --donor $2 --window $3 --output $OUT/e2/$4 --task 0 --task 1 --task 2 --task 3 --episodes 20 > $OUT/e2_$4.log 2>&1; }
run_e2 $K10 $K25 never base10k &
run_e2 $K10 $K25 grasp base10k_grasp25k &
run_e2 $K25 $K10 grasp base25k_grasp10k &
run_e2 $K10 $K25 always donor25k &
for job in $(jobs -p); do wait $job || fail "e2 job $job"; done
echo "DONE $(date -Is)"; echo EXIT_STATUS=0

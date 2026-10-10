#!/usr/bin/env bash
# E4 eval completion (5k missing tasks + 2.5k checkpoints), E5 (15k/20k), instruction-swap trace.
set -uo pipefail
cd /root/gripper-mujoco
export HF_HOME=/root/autodl-tmp/gripper-mujoco-hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MUJOCO_GL=egl PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
P=.venv-lerobot/bin/python
base=/root/autodl-tmp/smolvla-official-reproduction-v2; OUT=$base/acquisition/e4_20261008; R2=$OUT/eval_r2
LIN=$base/full_25k_seed1000/checkpoints
mkdir -p $R2; git rev-parse HEAD > $R2/code_commit.txt
echo "START $(date -Is)"
for arm in LE LA; do
  test -d $OUT/$arm/merged_002500 || $P scripts/merge_lora.py --adapter $OUT/$arm/train/checkpoints/002500/pretrained_model --output $OUT/$arm/merged_002500 > $OUT/$arm.merge2500.log 2>&1 || { echo "FAILED merge $arm"; exit 1; }
done
ck() { case $1 in C|W) echo $OUT/$1/train/checkpoints/00$2/pretrained_model;; *) echo $OUT/$1/merged_00$2;; esac; }
jobs=$R2/jobs.txt; : > $jobs
add() { echo "$P scripts/window_entry_probe.py --checkpoint $1 --output $R2/$2_push$3_t$4 --task $4 --episodes 20 --push-steps $3 --push-action 1.0 > $R2/$2_push$3_t$4.log 2>&1 && echo DONE $2_push$3_t$4 || echo FAIL $2_push$3_t$4" >> $jobs; }
for arm in C W LE LA; do
  for t in 1 2 3; do add $(ck $arm 5000) ${arm}_5000 0 $t; done; add $(ck $arm 5000) ${arm}_5000 8 1
  for t in 0 1 2 3; do add $(ck $arm 2500) ${arm}_2500 0 $t; done; for t in 0 1; do add $(ck $arm 2500) ${arm}_2500 8 $t; done
done
for s in 015000 020000; do
  for t in 0 1 2 3; do add $LIN/$s/pretrained_model k${s:0:2}_ 0 $t; done; for t in 0 1; do add $LIN/$s/pretrained_model k${s:0:2}_ 8 $t; done
done
TR=/root/autodl-tmp/instruction_swap_trace_20261011
( $P scripts/instruction_swap_eval.py --output $TR --trace --condition other --step 5000 --step 15000 --step 25000 --workers 4 \
  && $P scripts/instruction_swap_eval.py --output $TR --trace --condition correct --step 25000 --workers 4 ) > $TR.log 2>&1 &
trace=$!
xargs -P 9 -I{} bash -c {} < $jobs
wait $trace && echo "TRACE OK" || echo "TRACE FAILED"
echo "END $(date -Is)"

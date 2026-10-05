#!/usr/bin/env bash
set -euo pipefail
trap "code=\$?; echo EXIT_STATUS=\$code" EXIT
cd /root/gm-main
export HF_HOME=/root/autodl-tmp/gripper-mujoco-hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
P=/root/gripper-mujoco/.venv-lerobot/bin/python
base=/root/autodl-tmp/smolvla-official-reproduction-v2
OUT=$base/acquisition/q0b_q6a_20261005
bank=/root/gripper-mujoco/outputs/representation_study/libero_smolvla/state_bank
dataset=$HF_HOME/lerobot/hub/datasets--lerobot--libero/snapshots/a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4
metadata=$HF_HOME/hub/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/snapshots/7b375e1b73b11138ff12fe22c8f2822d8fe03467
ckpts=$base/full_25k_seed1000/checkpoints
traces=$base/acquisition/flow_e1_v2_train512_r3
git rev-parse HEAD > $OUT/code_commit.txt
for step in 025000 005000 010000 015000 020000; do
  echo "Q0B $step $(date -Is)"
  $P -m interaction_vla.representation_study.libero.stage_patching \
    --bank $bank --dataset-root $dataset --metadata $metadata \
    --checkpoint $ckpts/$step/pretrained_model --contract-checkpoint $ckpts/025000/pretrained_model \
    --trace $traces/natural_$step --output $OUT/stage_patching/$step \
    --device cuda --batch-size 4 --max-pairs 64 --repeat 0 --repeat 1 --repeat 2 > $OUT/stage_patching_$step.log 2>&1
  tail -c 600 $OUT/stage_patching/$step/report.json >/dev/null
done
echo "Q6A $(date -Is)"
$P -m interaction_vla.representation_study.libero.representation_structure \
  --bank $bank \
  --checkpoint 5k $traces/natural_005000 --checkpoint 10k $traces/natural_010000 \
  --checkpoint 15k $traces/natural_015000 --checkpoint 20k $traces/natural_020000 \
  --checkpoint 25k $traces/natural_025000 \
  --success $base/acquisition/timeline_v2_states0_39_summary/report.json \
  --output $OUT/representation_structure > $OUT/representation_structure.log 2>&1
echo "DONE $(date -Is)"

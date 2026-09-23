#!/usr/bin/env bash
# Frozen-model exploratory run: smoke -> paired episode caches -> CPU readouts.
set -euo pipefail
trap 'code=$?; printf "EXIT_STATUS=%s\n" "$code"' EXIT
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
run_root="${1:?Usage: bash scripts/run_checkpoint_readouts.sh NEW_ABSOLUTE_OUTPUT_DIR}"
[[ "$run_root" = /* ]] || { echo 'Output must be an absolute path'; exit 2; }
[[ ! -e "$run_root" ]] || { echo "Refusing existing output: $run_root"; exit 2; }
export HF_HOME=/root/autodl-tmp/gripper-mujoco-hf-cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
base=/root/autodl-tmp/smolvla-official-reproduction-v2
bank="$project_root/outputs/representation_study/libero_smolvla/state_bank"
dataset="$HF_HOME/lerobot/hub/datasets--lerobot--libero/snapshots/a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4"
metadata="$HF_HOME/hub/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/snapshots/7b375e1b73b11138ff12fe22c8f2822d8fe03467"
checkpoints="$base/full_25k_seed1000/checkpoints"
contract="$checkpoints/025000/pretrained_model"
for input in "$bank/manifest.json" "$dataset/meta/info.json" "$metadata/config.json" "$base/acquisition/lineage_v2.json"; do
  [[ -f "$input" ]] || { echo "Missing input: $input"; exit 2; }
done
for step in 005000 010000 015000 020000 025000; do
  [[ -f "$checkpoints/$step/pretrained_model/model.safetensors" ]] || exit 2
done
mkdir "$run_root"
git rev-parse HEAD > "$run_root/code_commit.txt"
py=(bash scripts/python.sh -W error::RuntimeWarning)
common=(--bank "$bank" --dataset-root "$dataset" --metadata "$metadata"
        --contract-checkpoint "$contract" --device cuda --batch-size 4
        --suite libero_spatial --task-id 0 --task-id 1 --task-id 2 --task-id 3
        --split-group episode --noise-repeats 3)
echo 'CHECK: migrated checkpoint hashes and unchanged lineage'
"${py[@]}" -m interaction_vla.representation_study.libero.acquisition lineage \
  --checkpoint-root "$checkpoints" \
  --base-checkpoint /root/autodl-tmp/gripper-mujoco-outputs/representation_study/libero_smolvla/stages/pretrained/checkpoint \
  --step 5000 --step 10000 --step 15000 --step 20000 --step 25000 \
  --output "$run_root/lineage.json"
"${py[@]}" -c 'import json,sys; a,b=(json.load(open(p)) for p in sys.argv[1:]); assert a==b, "Migrated lineage differs"' \
  "$base/acquisition/lineage_v2.json" "$run_root/lineage.json"
for partition in train validation; do
  "${py[@]}" -m interaction_vla.representation_study.libero.flow_trace plan \
    "${common[@]}" --partition "$partition" --max-states 128 --minimum-episodes 4 \
    > "$run_root/${partition}_plan.json"
done
echo 'SMOKE: 8 states, 3 noise repeats, frozen 5k checkpoint'
"${py[@]}" -m interaction_vla.representation_study.libero.flow_trace run \
  "${common[@]}" --partition train --max-states 8 --minimum-episodes 4 \
  --checkpoint "$checkpoints/005000/pretrained_model" --output "$run_root/smoke"
args=()
for step in 005000 010000 015000 020000 025000; do
  for partition in train validation; do
    echo "EXTRACT: checkpoint=$step partition=$partition states=128 repeats=3"
    "${py[@]}" -m interaction_vla.representation_study.libero.flow_trace run \
      "${common[@]}" --partition "$partition" --max-states 128 --minimum-episodes 4 \
      --checkpoint "$checkpoints/$step/pretrained_model" --output "$run_root/${step}_${partition}"
  done
  args+=(--checkpoint "$step" "$run_root/${step}_train" "$run_root/${step}_validation")
done
echo 'READOUTS: CKA + cross-checkpoint Ridge; validation is exploratory, not confirmation'
"${py[@]}" -m interaction_vla.representation_study.libero.checkpoint_readouts \
  --bank "$bank" "${args[@]}" --output "$run_root/readouts"
echo ALL_DONE

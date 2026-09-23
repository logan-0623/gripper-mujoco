#!/usr/bin/env bash
# Bounded exploration. Semantic masks are optional explicit annotations, never guessed.
set -euo pipefail
trap 'code=$?; printf "EXIT_STATUS=%s\n" "$code"' EXIT
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
run_root="${1:?Usage: run_candidate_controls.sh NEW_OUTPUT_DIR [MASK_SPEC_DIRECTORY]}"
[[ "$run_root" = /* && ! -e "$run_root" ]] || { echo 'Use a new absolute output path'; exit 2; }
mask_root="${2:-}"
export HF_HOME=/root/autodl-tmp/gripper-mujoco-hf-cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
base=/root/autodl-tmp/smolvla-official-reproduction-v2
cache="$base/acquisition/spatial_checkpoint_exploration_20260923"
bank="$project_root/outputs/representation_study/libero_smolvla/state_bank"
dataset="$HF_HOME/lerobot/hub/datasets--lerobot--libero/snapshots/a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4"
metadata="$HF_HOME/hub/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/snapshots/7b375e1b73b11138ff12fe22c8f2822d8fe03467"
checkpoints="$base/full_25k_seed1000/checkpoints"
py=(bash scripts/python.sh -W error::RuntimeWarning)
test -f "$cache/readouts/report.json"
mkdir "$run_root"
git rev-parse HEAD > "$run_root/code_commit.txt"
args=()
for step in 005000 010000 015000 020000 025000; do
  args+=(--checkpoint "$step" "$cache/${step}_train" "$cache/${step}_validation")
done
echo 'CONTROL: train-only moment alignment; unchanged raw CKA'
"${py[@]}" -m interaction_vla.representation_study.libero.checkpoint_readouts \
  --bank "$bank" "${args[@]}" --transfer-moment-alignment --output "$run_root/moment_alignment"
for tap in expert_middle expert_late; do
  echo "DISCOVER: $tap shared 5k/25k train-only directions"
  "${py[@]}" -m interaction_vla.representation_study.libero.acquisition discover \
    --trace "5k=$cache/005000_train" --trace "25k=$cache/025000_train" \
    --before 5k --after 25k --tap "$tap" --rank 16 --output "$run_root/candidates/$tap"
done
common=(--bank "$bank" --dataset-root "$dataset" --metadata "$metadata"
        --contract "$checkpoints/025000/pretrained_model"
        --candidates "$run_root/candidates/expert_middle/candidates.json"
        --candidates "$run_root/candidates/expert_late/candidates.json" --dose 0.5)
echo 'SMOKE: all suppression/fixed-query arms on four validation states'
"${py[@]}" -m interaction_vla.representation_study.libero.candidate_controls \
  "${common[@]}" --checkpoint "$checkpoints/005000/pretrained_model" \
  --train-trace "$cache/005000_train" --reference-trace "$cache/005000_validation" \
  --max-states 4 --repeats 1 --output "$run_root/smoke"
for step in 005000 015000 025000; do
  masks=()
  if [[ -n "$mask_root" ]]; then masks=(--mask-spec "$mask_root/$step.json"); fi
  echo "OFFLINE: $step, 32 states x 3 repeats, no-op + 8 suppression arms"
  "${py[@]}" -m interaction_vla.representation_study.libero.candidate_controls \
    "${common[@]}" "${masks[@]}" --checkpoint "$checkpoints/$step/pretrained_model" \
    --train-trace "$cache/${step}_train" --reference-trace "$cache/${step}_validation" \
    --max-states 32 --repeats 3 --output "$run_root/$step"
done
echo 'OFFLINE_DONE; starting six-rollout contact-triggered pilot, not confirmation'
"${py[@]}" -m interaction_vla.representation_study.libero.phase_pilot \
  --offline "$run_root" --checkpoint-root "$checkpoints" --output "$run_root/contact_pilot"
echo 'Semantic masking only ran if explicit mask specifications were supplied'
echo ALL_DONE

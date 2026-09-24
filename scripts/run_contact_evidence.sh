#!/usr/bin/env bash
# Contact readout subspace + paired candidate/no-op local effects.
set -euo pipefail
trap 'code=$?; printf "EXIT_STATUS=%s\n" "$code"' EXIT
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
out="${1:?Usage: run_contact_evidence.sh NEW_ABSOLUTE_OUTPUT_DIR}"
[[ "$out" = /* && ! -e "$out" ]] || { echo 'Use a new absolute output path'; exit 2; }
export HF_HOME=/root/autodl-tmp/gripper-mujoco-hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
base=/root/autodl-tmp/smolvla-official-reproduction-v2
# Use a cache regenerated with the current flow_trace implementation.  The
# old cache remains preserved, but stale source hashes must never be mixed into
# candidate/no-op effects.
cache="${FLOW_CACHE:-$base/acquisition/spatial_checkpoint_exploration_20260923}"
bank="$root/outputs/representation_study/libero_smolvla/state_bank"
dataset="$HF_HOME/lerobot/hub/datasets--lerobot--libero/snapshots/a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4"
metadata="$HF_HOME/hub/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/snapshots/7b375e1b73b11138ff12fe22c8f2822d8fe03467"
checkpoints="$base/full_25k_seed1000/checkpoints"
py=(bash scripts/python.sh -W error::RuntimeWarning)
mkdir "$out"
git rev-parse HEAD > "$out/code_commit.txt"
reports=()
for step in 005000 015000 025000; do
  echo "CONTACT_SUBSPACE $step"
  "${py[@]}" -m interaction_vla.representation_study.libero.contact_subspace \
    --bank "$bank" --train-trace "$cache/${step}_train" --output "$out/contact/$step"
  echo "PAIRED_CONTROLS $step"
  "${py[@]}" -m interaction_vla.representation_study.libero.candidate_controls \
    --bank "$bank" --dataset-root "$dataset" --metadata "$metadata" \
    --contract "$checkpoints/025000/pretrained_model" \
    --candidates "$out/contact/$step/expert_middle/candidates.json" \
    --candidates "$out/contact/$step/expert_late/candidates.json" \
    --target-id contact_0 --control-id low_change_0 --control-id matched_random_0 --control-id matched_random_1 \
    --checkpoint "$checkpoints/$step/pretrained_model" \
    --train-trace "$cache/${step}_train" --reference-trace "$cache/${step}_validation" \
    --max-states 32 --repeats 3 --dose 0.5 --output "$out/effects/$step"
  reports+=(--latent "$out/effects/$step/report.json")
done
echo "JOINT_EVIDENCE; input-side SR remains TBD until independent mask episodes exist"
"${py[@]}" -m interaction_vla.representation_study.libero.joint_evidence \
  "${reports[@]}" --output "$out/joint_evidence.json"
echo ALL_DONE

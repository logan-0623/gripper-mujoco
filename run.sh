#!/usr/bin/env bash
set -euo pipefail

# Reproduce the self-trained SmolVLA capability-acquisition experiment.
# Required immutable inputs are supplied as paths so this works on any server.
# Example:
# BASE_CHECKPOINT=/data/smolvla_base METADATA=/data/smolvlm \
# DATASET_ROOT=/data/libero STATE_BANK=/data/state_bank OUTPUT_ROOT=/data/run \
# nohup ./run.sh all > /data/run.log 2>&1 &
: "${BASE_CHECKPOINT:?path to the immutable lerobot/smolvla_base snapshot}"
: "${METADATA:?path to the immutable SmolVLM2-500M-Video-Instruct snapshot}"
: "${DATASET_ROOT:?path to the immutable lerobot/libero snapshot}"
: "${STATE_BANK:?path to the audited LIBERO StateBank}"
: "${OUTPUT_ROOT:?new or resumable output root with at least 40 GiB free}"

PYTHON=${PYTHON:-.venv-lerobot/bin/python}
STAGE=${1:-all}
ATLAS=${ACTION_ATLAS_ROOT:-research/action-atlas}
ATLAS_COMMIT=b8b0db331df18fc30a3fd92c45ec721d35d3ee52
TRAIN="$OUTPUT_ROOT/full_25k_seed1000"
CHECKPOINTS="$TRAIN/checkpoints"
CONTRACT="$CHECKPOINTS/025000/pretrained_model"
ACQ="$OUTPUT_ROOT/acquisition"
TASKS=(--suite libero_spatial --task-id 0 --task-id 1 --task-id 3)

export HF_HOME=${HF_HOME:-"$OUTPUT_ROOT/hf-cache"}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

require_file() { test -f "$1" || { echo "missing file: $1" >&2; exit 1; }; }

prepare() {
  require_file "$PYTHON"
  require_file "$BASE_CHECKPOINT/config.json"
  require_file "$METADATA/config.json"
  require_file "$DATASET_ROOT/meta/info.json"
  require_file "$STATE_BANK/manifest.json"
  "$PYTHON" - <<PY
import json
from pathlib import Path
p = Path(${STATE_BANK@Q}) / "manifest.json"
d = json.loads(p.read_text())
assert d.get("audit_passed") is True, f"StateBank audit has not passed: {p}"
PY
  if ! test -d "$ATLAS/.git"; then
    git clone https://github.com/CWRU-AISM/action-atlas.git "$ATLAS"
  fi
  git -C "$ATLAS" checkout --detach "$ATLAS_COMMIT"
  test "$(git -C "$ATLAS" rev-parse HEAD)" = "$ATLAS_COMMIT"
  test -z "$(git -C "$ATLAS" status --porcelain)" || {
    echo "Action Atlas checkout is dirty: $ATLAS" >&2; exit 1;
  }
  mkdir -p "$OUTPUT_ROOT"
  echo "inputs verified; output=$OUTPUT_ROOT"
}

train() {
  if test -f "$CONTRACT/config.json"; then
    echo "reuse completed 25k lineage: $TRAIN"
    return
  fi
  BASE_CHECKPOINT="$BASE_CHECKPOINT" DATASET_ROOT="$DATASET_ROOT" \
    OUTPUT_ROOT="$OUTPUT_ROOT" bash scripts/run_smolvla_official_reproduction.sh full
}

timeline() {
  local lineage="$ACQ/lineage.json" root="$ACQ/timeline"
  mkdir -p "$ACQ"
  test -f "$lineage" || "$PYTHON" -m interaction_vla.representation_study.libero.acquisition lineage \
    --checkpoint-root "$CHECKPOINTS" --base-checkpoint "$BASE_CHECKPOINT" \
    --step 5000 --step 10000 --step 15000 --step 20000 --step 25000 --output "$lineage"
  "$PYTHON" -W error::RuntimeWarning -m interaction_vla.representation_study.libero.acquisition evaluate \
    --lineage "$lineage" --output "$root" --task 0 --task 1 --task 2 --task 3 \
    --initial-state-offset 10 --episodes 10
  test -f "$ACQ/timeline-summary/report.json" || \
    "$PYTHON" -m interaction_vla.representation_study.libero.acquisition summarize \
      --lineage "$lineage" --root "$root" --output "$ACQ/timeline-summary"
}

trace_one() {
  local checkpoint=$1 partition=$2 output=$3; shift 3
  "$PYTHON" -W error::RuntimeWarning -m interaction_vla.representation_study.libero.flow_trace run \
    --bank "$STATE_BANK" --dataset-root "$DATASET_ROOT" --checkpoint "$checkpoint" \
    --contract-checkpoint "$CONTRACT" --metadata "$METADATA" --output "$output" \
    --device cuda --batch-size 4 --partition "$partition" --split-group episode \
    "${TASKS[@]}" "$@"
}

traces() {
  local partition root reference step
  for partition in train validation; do
    root="$ACQ/flow_${partition}_episode_n128_r3"
    reference="$root/natural_25k"
    trace_one "$CONTRACT" "$partition" "$reference" --max-states 128 --noise-repeats 3
    for step in 005000 010000 015000 020000 025000; do
      trace_one "$CHECKPOINTS/$step/pretrained_model" "$partition" "$root/fixed_$step" \
        --max-states 128 --noise-repeats 3 --reference-trace "$reference"
    done
  done
}

discover() {
  local train="$ACQ/flow_train_episode_n128_r3"
  local validation="$ACQ/flow_validation_episode_n128_r3"
  local candidates="$ACQ/candidates_episode_train_n128_r3"
  local tap partition root
  for tap in expert_middle expert_late; do
    test -f "$candidates/$tap/candidates.json" || \
      "$PYTHON" -m interaction_vla.representation_study.libero.acquisition discover \
        --trace 5k="$train/fixed_005000" --trace 10k="$train/fixed_010000" \
        --trace 25k="$train/fixed_025000" --before 5k --after 10k \
        --tap "$tap" --rank 32 --output "$candidates/$tap"
    for partition in train validation; do
      root="$train"; test "$partition" = validation && root="$validation"
      test -f "$candidates/$tap/trajectory_$partition/report.json" || \
        "$PYTHON" -m interaction_vla.representation_study.libero.acquisition candidate-trajectory \
          --trace 5k="$root/fixed_005000" --trace 10k="$root/fixed_010000" \
          --trace 15k="$root/fixed_015000" --trace 20k="$root/fixed_020000" \
          --trace 25k="$root/fixed_025000" --candidates "$candidates/$tap/candidates.json" \
          --output "$candidates/$tap/trajectory_$partition"
    done
    test -f "$candidates/$tap/interpret_validation_25k.json" || \
      "$PYTHON" -m interaction_vla.representation_study.libero.acquisition candidate-interpret \
        --trajectory "$candidates/$tap/trajectory_validation" --state-bank "$STATE_BANK" \
        --checkpoint 25k --output "$candidates/$tap/interpret_validation_25k.json"
  done
}

action_effects() {
  local out="$ACQ/action_effects_episode_controls_n128_r2"
  local candidates="$ACQ/candidates_episode_train_n128_r3"
  local tap group sign role name stages=() stage edited=()
  trace_one "$CONTRACT" holdout "$out/baseline" --task-id 2 --max-states 128 \
    --minimum-episodes 8 --noise-repeats 2
  for tap in expert_middle expert_late; do
    edited=()
    for group in early middle late all; do
      case "$group" in
        early) stages=(0 1 2);; middle) stages=(3 4 5 6);;
        late) stages=(7 8 9);; all) stages=(0 1 2 3 4 5 6 7 8 9);;
      esac
      for role in formation_0 low_change_0 matched_random_0; do
        for sign in -1 1; do
          name="${tap}_${role}_${group}_${sign}"
          args=(--max-states 128 --minimum-episodes 8 --noise-repeats 2 \
                --candidates "$candidates/$tap/candidates.json" \
                --candidate-id "$role" --dose "$sign")
          for stage in "${stages[@]}"; do args+=(--edit-stage "$stage"); done
          trace_one "$CONTRACT" holdout "$out/$name" --task-id 2 "${args[@]}"
          edited+=(--edited "$name=$out/$name")
        done
      done
    done
    test -f "$out/gate_$tap.json" || \
      "$PYTHON" -m interaction_vla.representation_study.libero.acquisition gate \
        --baseline "$out/baseline" "${edited[@]}" \
        --candidates "$candidates/$tap/candidates.json" --state-bank "$STATE_BANK" \
        --executed-steps 10 --minimum-episodes 8 --output "$out/gate_$tap.json"
  done
}

closed_loop() {
  local candidates="$ACQ/candidates_episode_train_n128_r3" tap out
  for tap in expert_middle expert_late; do
    out="$ACQ/closed_loop_${tap}_development"
    "$PYTHON" - <<PY
import json
from pathlib import Path
p = Path(${ACQ@Q}) / "action_effects_episode_controls_n128_r2" / "gate_$tap.json"
assert json.loads(p.read_text())["passed_candidate_ids"], f"no formation candidate passed: {p}"
PY
    "$PYTHON" -m interaction_vla.representation_study.libero.acquisition closed-loop \
      --checkpoint "$CONTRACT" --candidates "$candidates/$tap/candidates.json" \
      --gate "$ACQ/action_effects_episode_controls_n128_r2/gate_$tap.json" --output "$out" \
      --task 0 --task 1 --task 2 --task 3 --dose 1 \
      --edit-stage 0 --edit-stage 1 --edit-stage 2 --edit-stage 3 --edit-stage 4 \
      --edit-stage 5 --edit-stage 6 --edit-stage 7 --edit-stage 8 --edit-stage 9 \
      --initial-state-offset 20 --episodes 10
    test -f "$out/summary.json" || \
      "$PYTHON" -m interaction_vla.representation_study.libero.acquisition closed-loop-summary \
        --root "$out" --output "$out/summary.json"
  done
}

# Run only after candidate choice and analysis rules are frozen from development results.
confirmation() {
  local candidates="$ACQ/candidates_episode_train_n128_r3" tap contract out
  for tap in expert_middle expert_late; do
    contract="$ACQ/confirmation_${tap}.json"
    out="$ACQ/closed_loop_${tap}_confirmation"
    test -f "$contract" || \
      "$PYTHON" -m interaction_vla.representation_study.libero.acquisition freeze-confirmation \
        --output "$contract" --task 0 --task 1 --task 2 --task 3 \
        --initial-state-offset 30 --episodes 10 \
        --used-plan "$ACQ/timeline/evaluation_plan.json" \
        --used-plan "$ACQ/closed_loop_${tap}_development/plan.json"
    "$PYTHON" -m interaction_vla.representation_study.libero.acquisition closed-loop \
      --checkpoint "$CONTRACT" --candidates "$candidates/$tap/candidates.json" \
      --gate "$ACQ/action_effects_episode_controls_n128_r2/gate_$tap.json" \
      --output "$out" --task 0 --task 1 --task 2 --task 3 --dose 1 \
      --edit-stage 0 --edit-stage 1 --edit-stage 2 --edit-stage 3 --edit-stage 4 \
      --edit-stage 5 --edit-stage 6 --edit-stage 7 --edit-stage 8 --edit-stage 9 \
      --initial-state-offset 30 --episodes 10 --confirmation-contract "$contract"
    test -f "$out/summary.json" || \
      "$PYTHON" -m interaction_vla.representation_study.libero.acquisition closed-loop-summary \
        --root "$out" --output "$out/summary.json"
  done
}

case "$STAGE" in
  prepare) prepare;; train) prepare; train;; timeline) prepare; timeline;;
  traces) prepare; traces;; discover) prepare; discover;;
  action-effects) prepare; action_effects;; closed-loop) prepare; closed_loop;;
  confirmation) prepare; confirmation;;
  all) prepare; train; timeline; traces; discover; action_effects; closed_loop;;
  *) echo "usage: $0 [prepare|train|timeline|traces|discover|action-effects|closed-loop|confirmation|all]" >&2; exit 2;;
esac

#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-smoke}
: "${BASE_CHECKPOINT:?set BASE_CHECKPOINT to the immutable lerobot/smolvla_base snapshot}"
: "${DATASET_ROOT:?set DATASET_ROOT to the immutable lerobot/libero snapshot}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT to an empty output root}"

test -f "$BASE_CHECKPOINT/config.json"
test -f "$DATASET_ROOT/meta/info.json"

export HF_HOME=${HF_HOME:-${DATASET_ROOT%%/lerobot/hub/*}}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

case "$MODE" in
  smoke)
    STEPS=${STEPS:-20}
    BATCH_SIZE=${BATCH_SIZE:-1}
    SAVE_FREQ=$STEPS
    LOG_FREQ=1
    OUTPUT_DIR="$OUTPUT_ROOT/smoke"
    ;;
  full)
    STEPS=25000
    BATCH_SIZE=32
    SAVE_FREQ=5000
    LOG_FREQ=200
    OUTPUT_DIR="$OUTPUT_ROOT/full_25k_seed1000"
    ;;
  *)
    echo "usage: $0 [smoke|full]" >&2
    exit 2
    ;;
esac

if test -d "$OUTPUT_DIR" && test -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)"; then
  echo "refusing to reuse non-empty output: $OUTPUT_DIR" >&2
  exit 1
fi

.venv-lerobot/bin/python -m lerobot.scripts.lerobot_train \
  --policy.path="$BASE_CHECKPOINT" \
  --policy.push_to_hub=false \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.scheduler_decay_steps=25000 \
  --dataset.repo_id=lerobot/libero \
  --dataset.root="$DATASET_ROOT" \
  --dataset.revision=a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4 \
  --dataset.use_imagenet_stats=false \
  --rename_map='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}' \
  --output_dir="$OUTPUT_DIR" \
  --job_name="smolvla-libero-official-$MODE" \
  --steps="$STEPS" \
  --batch_size="$BATCH_SIZE" \
  --num_workers="${NUM_WORKERS:-4}" \
  --seed=1000 \
  --cudnn_deterministic=true \
  --env_eval_freq=0 \
  --save_checkpoint=true \
  --save_freq="$SAVE_FREQ" \
  --log_freq="$LOG_FREQ" \
  --wandb.enable=false

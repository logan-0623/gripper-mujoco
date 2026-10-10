#!/usr/bin/env bash
set -uo pipefail
cd /root/gripper-mujoco
export HF_HOME=/root/autodl-tmp/gripper-mujoco-hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MUJOCO_GL=egl PYTHONUNBUFFERED=1
P=.venv-lerobot/bin/python
base=/root/autodl-tmp/smolvla-official-reproduction-v2; OUT=$base/acquisition/e4_20261008
DS=$HF_HOME/lerobot/hub/datasets--lerobot--libero/snapshots/a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4
CK10=$base/full_25k_seed1000/checkpoints/010000/pretrained_model
git rev-parse HEAD > $OUT/code_commit.txt
fail() { echo "FAILED: $1 $(date -Is)"; echo EXIT_STATUS=1; exit 1; }
common=(--policy.path=$CK10 --policy.push_to_hub=false --policy.freeze_vision_encoder=false --policy.train_expert_only=false
  --policy.scheduler_warmup_steps=200 --policy.scheduler_decay_steps=5000
  --dataset.repo_id=lerobot/libero --dataset.root=$DS --dataset.revision=a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4 --dataset.use_imagenet_stats=false
  "--rename_map={\"observation.images.image\":\"observation.images.camera1\",\"observation.images.image2\":\"observation.images.camera2\"}"
  --steps=5000 --batch_size=32 --num_workers=4 --seed=1000 --cudnn_deterministic=true --env_eval_freq=0
  --save_checkpoint=true --save_freq=2500 --log_freq=200 --wandb.enable=false)
FULL_LR=(--policy.optimizer_lr=6.63e-5 --policy.scheduler_decay_lr=3.62e-5)
LORA_LR=(--policy.optimizer_lr=3e-4 --policy.scheduler_decay_lr=1.64e-4)
LAYERS="(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
train() { local arm=$1 weight=$2; shift 2
  echo "TRAIN $arm $(date -Is)"
  $P scripts/e4_train.py --window-weight $weight --dataset-root $DS --table-report $OUT/$arm/window_table.json -- "${common[@]}" "$@" --output_dir=$OUT/$arm/train --job_name=e4-$arm > $OUT/$arm.train.log 2>&1 || fail "train $arm"
  test -d $OUT/$arm/train/checkpoints/005000/pretrained_model || fail "checkpoint $arm"; }
train C 1 "${FULL_LR[@]}"
train W 3 "${FULL_LR[@]}"
train LE 3 "${LORA_LR[@]}" --peft.method_type=LORA "--peft.target_modules=.*lm_expert\.layers\.[0-9]+\.$LAYERS" "--peft.full_training_modules=[]"
train LA 3 "${LORA_LR[@]}" --peft.method_type=LORA "--peft.target_modules=.*(lm_expert|text_model)\.layers\.[0-9]+\.$LAYERS" "--peft.full_training_modules=[]"
for arm in LE LA; do $P scripts/merge_lora.py --adapter $OUT/$arm/train/checkpoints/005000/pretrained_model --output $OUT/$arm/merged_005000 > $OUT/$arm.merge.log 2>&1 || fail "merge $arm"; done
ckpt() { case $1 in C|W) echo $OUT/$1/train/checkpoints/005000/pretrained_model;; *) echo $OUT/$1/merged_005000;; esac; }
echo "EVAL $(date -Is)"
for arm in C W LE LA; do
  $P scripts/window_entry_probe.py --checkpoint $(ckpt $arm) --output $OUT/eval/${arm}_push0 --task 0 --task 1 --task 2 --task 3 --episodes 20 --push-steps 0 --push-action 1.0 > $OUT/eval_${arm}_push0.log 2>&1 &
  $P scripts/window_entry_probe.py --checkpoint $(ckpt $arm) --output $OUT/eval/${arm}_push8 --task 0 --task 1 --episodes 20 --push-steps 8 --push-action 1.0 > $OUT/eval_${arm}_push8.log 2>&1 &
done
for job in $(jobs -p); do wait $job || fail "eval job $job"; done
echo "DONE $(date -Is)"; echo EXIT_STATUS=0

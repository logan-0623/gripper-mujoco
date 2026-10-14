#!/usr/bin/env bash
# Second-seed lineage: seed 2000, checkpoints every 2500 steps; optimizer state kept only at 10k and 25k.
set -u
cd /root/gripper-mujoco
root=/root/autodl-tmp/smolvla-official-reproduction-v2
ckpts=$root/full_25k_seed2000/checkpoints
export OMP_NUM_THREADS=8 SEED=2000 SAVE_FREQ=2500 OUTPUT_ROOT=$root \
  BASE_CHECKPOINT=/root/autodl-tmp/models/smolvla_base_lineage \
  DATASET_ROOT=/root/autodl-tmp/gripper-mujoco-hf-cache/lerobot/hub/datasets--lerobot--libero/snapshots/a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4
bash scripts/run_smolvla_official_reproduction.sh full > $root/full_seed2000.log 2>&1 &
train=$!
prune() {  # drop training_state of every checkpoint except 010000, 025000 and the newest (may still be writing)
  newest=$(ls -d $ckpts/0* 2>/dev/null | sort | tail -1)
  for c in $(ls -d $ckpts/0* 2>/dev/null); do
    case $(basename $c) in 010000|025000) continue;; esac
    [ "$c" != "$newest" ] && rm -rf "$c/training_state"
  done
}
while kill -0 $train 2>/dev/null; do prune; sleep 300; done
wait $train; echo "train exit $?" >> $root/full_seed2000.log
newest=; for c in $(ls -d $ckpts/0* | sort); do case $(basename $c) in 010000|025000) ;; *) rm -rf "$c/training_state";; esac; done
df -h /root/autodl-tmp >> $root/full_seed2000.log

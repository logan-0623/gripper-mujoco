cd /root/gripper-mujoco
R=/root/autodl-tmp/smolvla-official-reproduction-v2/acquisition/stage0_20261012
L=/root/autodl-tmp/smolvla-official-reproduction-v2/full_25k_seed1000/checkpoints
DS=/root/autodl-tmp/gripper-mujoco-hf-cache/lerobot/hub/datasets--lerobot--libero/snapshots/a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
git rev-parse HEAD > $R/c0c_v2_code_commit.txt
.venv-lerobot/bin/python scripts/vlm_share.py $(for s in 005000 010000 015000 020000 025000; do echo --checkpoint $((10#$s / 1000))k=$L/$s/pretrained_model; done) \
  --processor-from $L/025000/pretrained_model --dataset-root $DS --task 0 --task 1 --task 2 --task 3 --batch-size 4 \
  --output $R/c0c_vlm_share_v2.json > $R/c0c_v2.log 2>&1
echo "0c rerun exit $?" >> $R/c0c_v2.log

# Stage 0 first batch: 0c (VLM share, offline, alone on the GPU), then E6c cross arms and 0b (E6 with C@2500 as base).
# Rollouts: one job per line, 8 in parallel, 8 threads each.
cd /root/gripper-mujoco
R=/root/autodl-tmp/smolvla-official-reproduction-v2/acquisition/stage0_20261012
L=/root/autodl-tmp/smolvla-official-reproduction-v2/full_25k_seed1000/checkpoints
C=/root/autodl-tmp/smolvla-official-reproduction-v2/acquisition/e4_20261008/C/train/checkpoints/002500/pretrained_model
DS=/root/autodl-tmp/gripper-mujoco-hf-cache/lerobot/hub/datasets--lerobot--libero/snapshots/a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4
B="$L/010000/pretrained_model"; D="$L/025000/pretrained_model"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MUJOCO_GL=egl PYTHONUNBUFFERED=1
P=".venv-lerobot/bin/python"
mkdir -p $R; git rev-parse HEAD > $R/code_commit.txt
$P scripts/vlm_share.py $(for s in 005000 010000 015000 020000 025000; do echo --checkpoint ${s:0:2}k=$L/$s/pretrained_model; done) \
  --processor-from $D --dataset-root $DS --task 0 --task 1 --task 2 --task 3 --output $R/c0c_vlm_share.json > $R/c0c.log 2>&1
echo "0c exit $?" > $R/jobs.log
{
for arm in "XE 0-4" "XL 5-9"; do set -- $arm
  for t in 0 1 2 3; do echo "$P scripts/flow_stage_mix.py --base $B --donor $D --donor-stages $2 --kv base --task $t --episodes 40 --output $R/e6c_$1_push0_t$t"; done; done
for arm in "N none" "A all" "E 0-4" "L 5-9"; do set -- $arm
  for t in 0 1 2 3; do echo "$P scripts/flow_stage_mix.py --base $C --donor $D --donor-stages $2 --kv own --task $t --episodes 40 --output $R/b0b_$1_push0_t$t"; done; done
} | xargs -P 8 -I{} sh -c "{} > \$(echo \"{}\" | sed \"s/.*--output //\").log 2>&1; echo \$? {}" >> $R/jobs.log 2>&1
echo ALLDONE >> $R/jobs.log

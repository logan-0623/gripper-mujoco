# Seed 2000 replication of E6 (10k vs 25k) + E8 smoke (w = 1 must match N). 8 jobs in parallel, 8 threads each.
cd /root/gripper-mujoco
R=/root/autodl-tmp/smolvla-official-reproduction-v2/acquisition/s2000_e6_20261014
L=/root/autodl-tmp/smolvla-official-reproduction-v2/full_25k_seed2000/checkpoints
B="$L/010000/pretrained_model"; D="$L/025000/pretrained_model"; W="$L/005000/pretrained_model"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
P=".venv-lerobot/bin/python"
mkdir -p $R
{
echo "$P scripts/flow_stage_mix.py --base $B --donor $W --donor-stages 0-4 --kv own --guide-weight 1 --task 0 --episodes 10 --output $R/smoke_G_w1_t0"
for arm in "N none own" "A all own" "E 0-4 own" "L 5-9 own" "V none donor" "X all base"; do set -- $arm
  for t in 0 1 2 3; do echo "$P scripts/flow_stage_mix.py --base $B --donor $D --donor-stages $2 --kv $3 --task $t --episodes 40 --output $R/$1_push0_t$t"; done; done
} | xargs -P 8 -I{} sh -c "{} > \$(echo \"{}\" | sed \"s/.*--output //\").log 2>&1; echo \$? {}" > $R/jobs.log 2>&1
echo ALLDONE >> $R/jobs.log

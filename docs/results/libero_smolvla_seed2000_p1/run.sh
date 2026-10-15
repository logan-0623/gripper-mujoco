# Seed 2000 P1: each checkpoint alone, Spatial 0-3 x initial states 0-39, same eval path and seed base as E6's N arm
# (flow_stage_mix with --donor-stages none; donor = the checkpoint itself). 10k is the replication's N arm.
cd /root/gripper-mujoco
R=/root/autodl-tmp/smolvla-official-reproduction-v2/acquisition/s2000_p1_20261014
L=/root/autodl-tmp/smolvla-official-reproduction-v2/full_25k_seed2000/checkpoints
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
P=".venv-lerobot/bin/python"
mkdir -p $R
{
for m in 025000 002500 005000 007500 012500 015000 017500 020000 022500; do C="$L/$m/pretrained_model"
  for t in 0 1 2 3; do echo "$P scripts/flow_stage_mix.py --base $C --donor $C --donor-stages none --task $t --episodes 40 --output $R/k${m}_t$t"; done; done
} | xargs -P 8 -I{} sh -c "{} > \$(echo \"{}\" | sed \"s/.*--output //\").log 2>&1; echo \$? {}" > $R/jobs.log 2>&1
echo ALLDONE >> $R/jobs.log

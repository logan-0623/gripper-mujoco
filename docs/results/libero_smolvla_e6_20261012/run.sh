# E6 + E3b task 2-3 refill. One job per line, 8 in parallel, 8 threads each.
cd /root/gripper-mujoco
R=/root/autodl-tmp/smolvla-official-reproduction-v2/acquisition/e6_20261012
L=/root/autodl-tmp/smolvla-official-reproduction-v2/full_25k_seed1000/checkpoints
B="$L/010000/pretrained_model"; D="$L/025000/pretrained_model"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
P=".venv-lerobot/bin/python"
{
for arm in "N none own" "A all own" "E 0-4 own" "L 5-9 own" "V none donor" "X all base"; do set -- $arm
  for t in 0 1 2 3; do echo "$P scripts/flow_stage_mix.py --base $B --donor $D --donor-stages $2 --kv $3 --task $t --episodes 40 --output $R/$1_push0_t$t"; done; done
for arm in "N none own" "A all own" "E 0-4 own" "L 5-9 own"; do set -- $arm
  for t in 0 1 2 3; do echo "$P scripts/flow_stage_mix.py --base $B --donor $D --donor-stages $2 --kv $3 --task $t --episodes 20 --push-steps 8 --output $R/$1_push8_t$t"; done; done
for m in 010000 025000; do for t in 2 3; do echo "$P scripts/window_entry_probe.py --checkpoint $L/$m/pretrained_model --task $t --episodes 20 --push-steps 8 --push-action 1.0 --output $R/e3b_${m}_push8_t$t"; done; done
} | xargs -P 8 -I{} sh -c "{} > \$(echo \"{}\" | sed \"s/.*--output //\").log 2>&1; echo \$? {}" > $R/jobs.log 2>&1
echo ALLDONE >> $R/jobs.log

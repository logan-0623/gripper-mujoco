# Seed 2000 E7 (noise scaling) + E8 (autoguidance), started only if the E6 replication gate passes
# (A - N CI lower bound > 0; preregistered 2026-10-14). 8 jobs in parallel, 8 threads each.
cd /root/gripper-mujoco
A=/root/autodl-tmp/smolvla-official-reproduction-v2/acquisition
G=$A/s2000_e6_20261014; R=$A/s2000_e7_e8_20261014
L=/root/autodl-tmp/smolvla-official-reproduction-v2/full_25k_seed2000/checkpoints
K10="$L/010000/pretrained_model"; K25="$L/025000/pretrained_model"
K5="$L/005000/pretrained_model"; K15="$L/015000/pretrained_model"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
P=".venv-lerobot/bin/python"
until grep -q ALLDONE $G/jobs.log 2>/dev/null; do sleep 300; done
mkdir -p $R
$P scripts/stage0_analysis.py --replication $G --output $G/replication_summary.json > $R/gate.log 2>&1
$P -c "import json,sys; sys.exit(0 if json.load(open('$G/replication_summary.json'))['decision']['gap_measurable'] else 1)" \
  || { echo "GATE FAILED: E7/E8 not run" >> $R/gate.log; exit 0; }
{
# E7: scale the initial flow noise on every chunk (10k and 25k), and 10k sigma=0 inside the window only
for m in "k10 $K10 $K25" "k25 $K25 $K10"; do set -- $m
  for s in 0.5 0; do for t in 0 1 2 3; do
    echo "$P scripts/flow_stage_mix.py --base $2 --donor $3 --donor-stages none --noise-scale $s --task $t --episodes 40 --output $R/e7_$1_s${s}_t$t"; done; done; done
for t in 0 1 2 3; do echo "$P scripts/flow_stage_mix.py --base $K10 --donor $K25 --donor-stages none --noise-scale 0 --noise-scope window --task $t --episodes 40 --output $R/e7_k10_s0win_t$t"; done
# E8: G1 5k/0-4/w2, G2 5k/0-4/w1.5, G3 5k/all/w2, G4 15k/0-4/w2
for g in "G1 $K5 0-4 2" "G2 $K5 0-4 1.5" "G3 $K5 all 2" "G4 $K15 0-4 2"; do set -- $g
  for t in 0 1 2 3; do echo "$P scripts/flow_stage_mix.py --base $K10 --donor $2 --donor-stages $3 --kv own --guide-weight $4 --task $t --episodes 40 --output $R/e8_$1_t$t"; done; done
} | xargs -P 8 -I{} sh -c "{} > \$(echo \"{}\" | sed \"s/.*--output //\").log 2>&1; echo \$? {}" > $R/jobs.log 2>&1
echo ALLDONE >> $R/jobs.log

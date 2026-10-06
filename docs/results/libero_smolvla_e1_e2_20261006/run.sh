#!/usr/bin/env bash
set -uo pipefail
cd /root/gripper-mujoco
export HF_HOME=/root/autodl-tmp/gripper-mujoco-hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MUJOCO_GL=egl PYTHONUNBUFFERED=1
P=.venv-lerobot/bin/python
base=/root/autodl-tmp/smolvla-official-reproduction-v2
OUT=$base/acquisition/e1_e2_20261006
ck=$base/full_25k_seed1000/checkpoints
K10=$ck/010000/pretrained_model; K25=$ck/025000/pretrained_model
git rev-parse HEAD > $OUT/code_commit.txt
fail() { echo "FAILED: $1 $(date -Is)"; echo EXIT_STATUS=1; exit 1; }
echo "SMOKE $(date -Is)"
$P scripts/demo_restart_rollouts.py --checkpoint $K25 --output $OUT/smoke/e1_replay --task 1 --points events --mode demo_replay > $OUT/smoke_e1_replay.log 2>&1 &
$P scripts/demo_restart_rollouts.py --checkpoint $K25 --output $OUT/smoke/e1_policy --task 1 --points events --mode policy > $OUT/smoke_e1_policy.log 2>&1 &
$P scripts/grasp_window_switch.py --base $K10 --donor $K25 --window grasp --output $OUT/smoke/e2_grasp --task 1 --episodes 2 > $OUT/smoke_e2.log 2>&1 &
for job in $(jobs -p); do wait $job || fail "smoke job $job"; done
$P - <<PY || fail "smoke checks"
import json
o="$OUT/smoke"
r=json.load(open(o+"/e1_replay/summary.json"))["by_label"]
print("replay", {k:(v["success"],v["episodes"]) for k,v in r.items()})
assert sum(v["episodes"] for v in r.values())==30
w=json.load(open(o+"/e2_grasp/summary.json"))
print("e2", {k:w[k] for k in ("success","episodes","mean_donor_fraction")}, [(x["window_opened_step"],x["window_closed_step"]) for x in w["rows"]])
assert w["episodes"]==2
PY
echo "E1 $(date -Is)"
$P scripts/demo_restart_rollouts.py --checkpoint $K10 --output $OUT/e1/step_010000 --task 0 --task 1 --task 2 --task 3 --points events --mode policy > $OUT/e1_10k.log 2>&1 &
$P scripts/demo_restart_rollouts.py --checkpoint $K25 --output $OUT/e1/step_025000 --task 0 --task 1 --task 2 --task 3 --points events --mode policy > $OUT/e1_25k.log 2>&1 &
$P scripts/demo_restart_rollouts.py --checkpoint $K25 --output $OUT/e1/demo_replay --task 0 --task 1 --task 2 --task 3 --points events --mode demo_replay > $OUT/e1_replay.log 2>&1 &
for job in $(jobs -p); do wait $job || fail "e1 job $job"; done
echo "E2 $(date -Is)"
run_e2() { $P scripts/grasp_window_switch.py --base $1 --donor $2 --window $3 --output $OUT/e2/$4 --task 0 --task 1 --task 2 --task 3 --episodes 20 > $OUT/e2_$4.log 2>&1; }
run_e2 $K10 $K25 never base10k &
run_e2 $K10 $K25 grasp base10k_grasp25k &
for job in $(jobs -p); do wait $job || fail "e2 job $job"; done
run_e2 $K25 $K10 grasp base25k_grasp10k &
run_e2 $K10 $K25 always donor25k &
for job in $(jobs -p); do wait $job || fail "e2 job $job"; done
echo "DONE $(date -Is)"; echo EXIT_STATUS=0

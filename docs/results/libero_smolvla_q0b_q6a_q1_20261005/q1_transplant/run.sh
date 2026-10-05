#!/usr/bin/env bash
set -euo pipefail
trap "code=\$?; echo EXIT_STATUS=\$code" EXIT
cd /root/gm-main
export HF_HOME=/root/autodl-tmp/gripper-mujoco-hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MUJOCO_GL=egl PYTHONUNBUFFERED=1
P=/root/gripper-mujoco/.venv-lerobot/bin/python
base=/root/autodl-tmp/smolvla-official-reproduction-v2
OUT=$base/acquisition/q1_transplant_20261005
ck=$base/full_25k_seed1000/checkpoints
git rev-parse HEAD > $OUT/code_commit.txt
for arm in "vlm5k_expert25k 005000 025000" "vlm25k_expert5k 025000 005000"; do
  set -- $arm
  echo "BUILD $1 $(date -Is)"
  test -d $OUT/models/$1 || $P -m interaction_vla.representation_study.libero.module_transplant build \
    --prefix-checkpoint $ck/$2/pretrained_model --action-checkpoint $ck/$3/pretrained_model --output $OUT/models/$1
  echo "EVAL $1 $(date -Is)"
  $P -m interaction_vla.representation_study.libero.module_transplant evaluate \
    --checkpoint $OUT/models/$1 --output $OUT/eval/$1 > $OUT/eval_$1.log 2>&1
done
T=$base/acquisition/timeline_v2_states0_39
$P -m interaction_vla.representation_study.libero.module_transplant summarize \
  --arm 5k $T/step_005000 --arm vlm5k_expert25k $OUT/eval/vlm5k_expert25k \
  --arm vlm25k_expert5k $OUT/eval/vlm25k_expert5k --arm 25k $T/step_025000 --output $OUT/summary
echo "DONE $(date -Is)"

cd /root/gripper-mujoco
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
P=".venv-lerobot/bin/python scripts/instruction_swap_eval.py"
$P --output /root/autodl-tmp/instruction_swap_double_check_20261011/object_25k --suite libero_object --step 25000 --condition correct --condition empty --condition other --condition paraphrase --trace --workers 5 > /root/autodl-tmp/instruction_swap_double_check_20261011/object_25k.log 2>&1 &
$P --output /root/autodl-tmp/instruction_swap_double_check_20261011/official_spatial --lineage /root/autodl-tmp/instruction_swap_double_check_20261011/official_lineage --step 0 --condition correct --condition empty --condition other --condition paraphrase --trace --workers 5 > /root/autodl-tmp/instruction_swap_double_check_20261011/official_spatial.log 2>&1 &
$P --output /root/autodl-tmp/instruction_swap_double_check_20261011/spatial_paraphrase --step 15000 --step 25000 --condition correct --condition paraphrase --trace --workers 2 > /root/autodl-tmp/instruction_swap_double_check_20261011/spatial_paraphrase.log 2>&1 &
wait

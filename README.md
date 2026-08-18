# Control-Aligned Interaction Representations

本项目研究一个具体问题：**什么样的 structured interaction representation 真正能被连续机器人控制策略利用？**

策略输入是 agent RGB、wrist RGB、10D 末端状态（位置、6D 旋转、夹爪）和任务语言，输出是 7D 连续笛卡尔动作。当前控制器固定为 ACT。Interaction Graph 不复制完整 MuJoCo state，只回答：目标是谁、夹爪与目标的关系、目标与容器的关系、干扰物风险、交互阶段与下一步关系变化。

当前实现包括：

- Franka MuJoCo 接触物理、扰动恢复和闭环评估；
- 标准 `LeRobotDataset`、agent/wrist RGB、language metadata；
- ReflectVLM Graph pretraining 和 MuJoCo Graph fine-tuning；
- 89D causal Interaction Graph v2；
- ACT 的 Flat、Privileged Teacher、Predicted Random、Predicted Reflect 对比；
- representation diagnostics、policy sensitivity、step trace、failure analysis；
- Flat → Entity+Geometry → Interaction-State → Full → Shuffled 渐进消融。

π0、SmolVLA、多任务语言泛化和真实机器人尚未实现。当前数据只有一个任务指令，不能宣称语言泛化。

## 当前证据

已完成的四条件实验每个条件包含 3 个 policy seeds、共 60 个闭环 rollout：

| Representation | Success | Target drop | Action clipping |
|---|---:|---:|---:|
| Flat | 30.0% | 6.7% | 0.117 |
| Privileged Teacher Graph | 35.0% | 0.0% | 0.096 |
| Predicted Random | 40.0% | 10.0% | 0.114 |
| Predicted Reflect | 41.7% | 0.0% | 0.092 |

这些结果支持的稳健表述是：Interaction Graph 具有一定 inductive bias，但 aggregate Graph accuracy 不等于 control utility。Privileged Teacher Graph 不是 policy-performance upper bound；代码中的 `oracle_graph_v2` 是历史条件名。Reflect 初始化没有稳定提高成功率，只留下较少 drop 和 clipping 的次级信号，尚不足以宣称改善安全性。

当前论文问题是：

> When and why do structured interaction representations help continuous visuomotor control?

## 现在从哪里继续

先检查本机产物；`READY` 就跳过，命令默认拒绝覆盖已完成输出：

```bash
for artifact in \
  outputs/graph_control/graph_v2_pilot/diagnostics/test/report.json \
  outputs/graph_control/graph_v2_pilot/diagnostics/test/sensitivity/report.json \
  outputs/graph_control/graph_v2_pilot/traced_evaluation/report.json \
  outputs/graph_control/graph_v2_pilot/traced_evaluation/failure_analysis/report.json \
  outputs/graph_control/control_alignment_ablation/cache/report.json \
  outputs/graph_control/control_alignment_ablation/smoke/comparison.json \
  outputs/graph_control/control_alignment_ablation/runs/comparison.json \
  outputs/graph_control/control_alignment_ablation/runs/evaluation/report.json
do
  test -e "$artifact" && echo "READY   $artifact" || echo "MISSING $artifact"
done
```

当前 Mac workspace 已完成 token diagnostics 和 ablation cache。下一条应运行 sensitivity；若只关心最关键的新增训练，也可以先运行 ablation smoke。

### 1. Representation diagnostics

```bash
.venv-lerobot/bin/python -m interaction_vla.graph_control diagnose \
  --config configs/graph_v2_act_pilot_macos.yaml \
  --partition test
```

输出：`outputs/graph_control/graph_v2_pilot/diagnostics/test/report.json`。报告比较 correctness、temporal smoothness、second-order jitter、relation flips、entropy、effective range 和 Teacher–Predicted 距离。

### 2. Frozen-policy sensitivity

```bash
.venv-lerobot/bin/python -m interaction_vla.graph_control sensitivity \
  --config configs/graph_v2_act_pilot_macos.yaml \
  --partition test
```

输出：`outputs/graph_control/graph_v2_pilot/diagnostics/test/sensitivity/report.json`。该命令不训练，只对冻结 ACT 做 group masking 和有限扰动，回答 ACT 实际使用哪些 token。它需要逐个加载 12 个 checkpoint；Mac 可能运行数小时，4090 更合适，但当前命令不支持中断续跑。

### 3. Resumable step trace

```bash
.venv-lerobot/bin/python -m interaction_vla.graph_control trace \
  --config configs/graph_v2_act_pilot_macos.yaml
```

输出：`outputs/graph_control/graph_v2_pilot/traced_evaluation/`。它记录 Graph error → action → contact/grasp/release → outcome 的逐步链路。共 240 个 episode，带进度条和 episode-level resume；中断后直接重跑同一命令。

### 4. Failure-conditioned analysis

trace 完成后运行：

```bash
.venv-lerobot/bin/python -m interaction_vla.graph_control failure-analysis \
  --config configs/graph_v2_act_pilot_macos.yaml \
  --traces outputs/graph_control/graph_v2_pilot/traced_evaluation
```

输出：`outputs/graph_control/graph_v2_pilot/traced_evaluation/failure_analysis/report.json`。阈值只用 train split 拟合，结果称为 descriptive Failure Association Score，不作因果表述。

### 5. Progressive representation ablation

五个条件保持相同 89D 输入宽度、ACT 容量、初始化、row order、split 和 10 epochs；所有非 Flat 条件来自同一个 `predicted_random_v2` estimator。Shuffled Graph 保留序列和近似边缘分布，但破坏 observation–token correspondence。

```bash
.venv-lerobot/bin/python -m interaction_vla.graph_control ablation-inspect \
  --config configs/control_alignment_ablation_macos.yaml

.venv-lerobot/bin/python -m interaction_vla.graph_control ablation-cache \
  --config configs/control_alignment_ablation_macos.yaml

# 只跑 seed 0、每个条件一个 optimizer step；先验证完整链路。
.venv-lerobot/bin/python -m interaction_vla.graph_control ablation-smoke \
  --config configs/control_alignment_ablation_macos.yaml

# 正式训练：3 seeds × 5 conditions × 10 epochs，耗时很长且不能续训。
.venv-lerobot/bin/python -m interaction_vla.graph_control ablation-compare \
  --config configs/control_alignment_ablation_macos.yaml

# 正式闭环：3 seeds × 5 conditions × 20 paired cases。
.venv-lerobot/bin/python -m interaction_vla.graph_control ablation-evaluate \
  --config configs/control_alignment_ablation_macos.yaml
```

主要输出：

```text
outputs/graph_control/control_alignment_ablation/
├── cache/
├── smoke/
└── runs/
    ├── comparison.json
    └── evaluation/report.json
```

正式对比预注册为 `Entity+Geometry−Flat`、`Interaction−Entity+Geometry`、`Full−Interaction`、`Full−Flat` 和 `Full−Shuffled`。没有把单个 seed 当作结论，也没有自动 scientific pass/fail gate。

## 环境

### macOS / Apple Silicon

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -r requirements-macos.txt
.venv/bin/python -m interaction_vla.macos_mjpython

python3.12 -m venv .venv-lerobot
.venv-lerobot/bin/python -m pip install --upgrade pip
.venv-lerobot/bin/python -m pip install -r requirements-lerobot-macos.lock.txt

export HF_HOME=/tmp/gripper-mujoco-hf-cache
```

检查：

```bash
.venv-lerobot/bin/python -c 'import lerobot, torch, torchcodec; print(lerobot.__version__, torch.__version__, torch.backends.mps.is_available())'
```

`av`、`cv2`、SDL 和 Homebrew FFmpeg 可能打印重复 Objective-C class 警告。以退出码和最终 JSON 的 `passed` 为准。

### Linux / NVIDIA CUDA

固定环境为 Python 3.12、PyTorch 2.10 + CUDA 12.8、LeRobot 0.6.1：

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg libgl1 libegl1

python3.12 -m venv .venv-lerobot
.venv-lerobot/bin/python -m pip install --upgrade pip
.venv-lerobot/bin/python -m pip install -r requirements-lerobot-linux-cuda.txt

export MUJOCO_GL=egl
export HF_HOME=/tmp/gripper-mujoco-hf-cache

.venv-lerobot/bin/python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
```

CUDA 配置和输出完全隔离：

| macOS config | Linux/CUDA config |
|---|---|
| `graph_v2_act_pilot_macos.yaml` | `graph_v2_act_pilot_linux_cuda.yaml` |
| `control_alignment_ablation_macos.yaml` | `control_alignment_ablation_linux_cuda.yaml` |
| `mujoco_graph_v2_finetune_macos.yaml` | `mujoco_graph_v2_finetune_linux_cuda.yaml` |
| `reflectvlm_graph_pretrain_macos.yaml` | `reflectvlm_graph_pretrain_linux_cuda.yaml` |

在上述新增实验命令中把 config 替换成 CUDA 版本即可。CUDA 输出使用 `*_cuda` 根目录，不覆盖 Mac 结果。ACT/Graph 训练会明显加速；MuJoCo 数据采集和闭环 rollout 仍主要受 CPU 限制。

## Fresh clone 的前置实验

`outputs/` 是本机产物，不会随普通 `git push` 上传。新机器必须按顺序建立以下工件；已有机器从最早的 `MISSING` 项继续。

### 1. 数据与 ACT recovery gate

```bash
.venv/bin/python -m interaction_vla.validate_physics_expert \
  --config configs/physics_pilot_macos.yaml

.venv-lerobot/bin/python -m interaction_vla.lerobot_bridge collect \
  --config configs/lerobot_act_recovery_macos.yaml

.venv-lerobot/bin/python -m interaction_vla.lerobot_bridge validate \
  --config configs/lerobot_act_recovery_macos.yaml

.venv-lerobot/bin/python -m interaction_vla.lerobot_bridge act-check \
  --config configs/lerobot_act_recovery_macos.yaml

.venv-lerobot/bin/python -m interaction_vla.lerobot_bridge act-train \
  --config configs/lerobot_act_recovery_macos.yaml

.venv-lerobot/bin/python -m interaction_vla.lerobot_bridge act-diagnose \
  --config configs/lerobot_act_recovery_macos.yaml \
  --checkpoint outputs/graph_control/act_recovery/train/checkpoint

.venv-lerobot/bin/python -m interaction_vla.lerobot_bridge act-recovery \
  --config configs/lerobot_act_recovery_macos.yaml \
  --checkpoint outputs/graph_control/act_recovery/train/checkpoint
```

### 2. Reflect pretraining 与 MuJoCo split

```bash
.venv-lerobot/bin/python -m interaction_vla.graph_pretrain inspect \
  --config configs/reflectvlm_graph_pretrain_macos.yaml
.venv-lerobot/bin/python -m interaction_vla.graph_pretrain train \
  --config configs/reflectvlm_graph_pretrain_macos.yaml
.venv-lerobot/bin/python -m interaction_vla.graph_pretrain evaluate \
  --config configs/reflectvlm_graph_pretrain_macos.yaml \
  --checkpoint outputs/graph_pretrain/reflectvlm/checkpoint.pt \
  --partition test

.venv-lerobot/bin/python -m interaction_vla.graph_finetune inspect \
  --config configs/mujoco_graph_v2_finetune_macos.yaml
.venv-lerobot/bin/python -m interaction_vla.graph_finetune split \
  --config configs/mujoco_graph_v2_finetune_macos.yaml
```

### 3. Teacher prerequisite、Graph estimator 与四条件 ACT

```bash
.venv-lerobot/bin/python -m interaction_vla.graph_control inspect \
  --config configs/graph_v2_act_oracle_macos.yaml
.venv-lerobot/bin/python -m interaction_vla.graph_control cache \
  --config configs/graph_v2_act_oracle_macos.yaml
.venv-lerobot/bin/python -m interaction_vla.graph_control compare \
  --config configs/graph_v2_act_oracle_macos.yaml
.venv-lerobot/bin/python -m interaction_vla.graph_control evaluate \
  --config configs/graph_v2_act_oracle_macos.yaml

.venv-lerobot/bin/python -m interaction_vla.graph_finetune compare \
  --config configs/mujoco_graph_v2_finetune_macos.yaml

.venv-lerobot/bin/python -m interaction_vla.graph_control inspect \
  --config configs/graph_v2_act_pilot_macos.yaml
.venv-lerobot/bin/python -m interaction_vla.graph_control cache \
  --config configs/graph_v2_act_pilot_macos.yaml
.venv-lerobot/bin/python -m interaction_vla.graph_control compare \
  --config configs/graph_v2_act_pilot_macos.yaml
.venv-lerobot/bin/python -m interaction_vla.graph_control evaluate \
  --config configs/graph_v2_act_pilot_macos.yaml
```

Linux fresh clone 使用对应 `*_linux_cuda.yaml` 配置。若数据集或 Reflect checkpoint 已从 Mac 复制，可跳过其生成步骤，但路径和 provenance 必须与 CUDA config 一致。

## 长任务

正式 ACT compare 没有 epoch-level resume；中断后需从该条命令重新开始。trace 支持 episode-level resume。服务器建议使用 `tmux`；Mac 可使用 `caffeinate`。

```bash
mkdir -p outputs/logs
nohup env HF_HOME=/tmp/gripper-mujoco-hf-cache \
  .venv-lerobot/bin/python -m interaction_vla.graph_control ablation-compare \
  --config configs/control_alignment_ablation_linux_cuda.yaml \
  > outputs/logs/control_alignment_ablation_cuda.log 2>&1 &

tail -f outputs/logs/control_alignment_ablation_cuda.log
```

## 测试

```bash
PYTHONPYCACHEPREFIX=/tmp/gripper-mujoco-pycache \
  .venv/bin/python -m pytest tests/interaction_vla -q

HF_HOME=/tmp/gripper-mujoco-hf-cache \
PYTHONPYCACHEPREFIX=/tmp/gripper-mujoco-lerobot-pycache \
  .venv-lerobot/bin/python -m pytest tests/interaction_vla -q
```

项目目录：`interaction_vla/` 是实现，`configs/` 是固定实验配置，`tests/` 是验证，`outputs/` 是本机数据、checkpoint、cache、GIF 和报告。

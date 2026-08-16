# Interaction-Structured VLA

本项目研究：**什么样的 interaction graph 对机器人连续控制策略真正有价值？**

策略输入为 agent RGB、wrist RGB、10D 末端状态（位置、6D 旋转、夹爪）和任务语言，
输出为 7D 连续笛卡尔动作。Graph 不复制完整 MuJoCo state，只编码任务相关实体、当前
交互关系、关系的时间变化，以及下一步应改变的关系。

当前已实现：

- Franka MuJoCo 接触物理与 Graph vs Flat；
- 标准 `LeRobotDataset`、双 RGB、language metadata 和 ACT；
- ReflectVLM Graph pretraining；
- 89D causal Interaction Graph v2；
- `ReflectVLM pretraining -> MuJoCo Graph fine-tuning -> ACT control`。

π0、SmolVLA 和多语言泛化尚未实现。当前数据只有一个任务指令，不能宣称语言泛化。

## 先确认从哪里开始

`outputs/` 中的数据、checkpoint、cache 和报告都是本机产物，普通 `git push` 不会上传。
在当前机器上应保留已有产物并从最早缺失的一项继续；新机器必须从“从零运行”开始。

检查关键产物：

```bash
for path in \
  outputs/lerobot/franka_lerobot_act_pilot/meta/info.json \
  outputs/graph_control/act_recovery/evaluation/recovery_report.json \
  outputs/graph_pretrain/reflectvlm/checkpoint.pt \
  outputs/graph_finetune/mujoco_graph_v2/split_manifest.json \
  outputs/graph_control/graph_v2_oracle/cache/seed_0/flat.npz \
  outputs/graph_control/graph_v2_oracle/runs/comparison.json \
  outputs/graph_control/graph_v2_oracle/runs/evaluation/report.json
do
  test -e "$path" && echo "READY   $path" || echo "MISSING $path"
done
```

如果前五项为 `READY`、只有正式 Oracle 结果缺失，直接运行“Oracle Graph v2 vs Flat”。
如果是 fresh clone，则按下面顺序完整运行，不要跳过 gate。

## macOS 环境

已验证环境：Apple Silicon、Python 3.12、MuJoCo 3.3.4、LeRobot 0.6.1、
PyTorch 2.10、TorchCodec 0.10。项目使用两个独立虚拟环境。

```bash
# 物理实验环境
uv venv --python 3.12
uv pip install --python .venv/bin/python -r requirements-macos.txt
.venv/bin/python -m interaction_vla.macos_mjpython

# LeRobot、ACT 和 Graph 环境；lock 文件用于固定当前已验证版本
python3.12 -m venv .venv-lerobot
.venv-lerobot/bin/python -m pip install --upgrade pip
.venv-lerobot/bin/python -m pip install -r requirements-lerobot-macos.lock.txt

# Hugging Face 数据缓存。首次运行 ReflectVLM 需要联网。
export HF_HOME=/tmp/gripper-mujoco-hf-cache
```

快速检查：

```bash
.venv/bin/python -c 'import mujoco, torch; print(mujoco.__version__, torch.__version__)'
.venv-lerobot/bin/python -c 'import lerobot, torch, torchcodec; print(lerobot.__version__, torch.__version__, torch.backends.mps.is_available())'
```

最后一个值表示 MPS 是否可用；`False` 时项目会回退 CPU，并不表示环境安装失败。

macOS 中 `av`、`cv2` 和 Homebrew FFmpeg 可能打印重复 Objective-C class 警告。这不是
本项目 gate 的失败原因；以命令退出码和最终 JSON 的 `passed` 字段为准。

## Linux + NVIDIA CUDA（RTX 4090）

Linux CUDA 配置保持与 macOS 相同的 dataset、seed、split、epoch 和 batch size，只更换
训练设备并使用独立的 `*_cuda` 输出目录。当前固定环境是 Python 3.12、PyTorch 2.10、
CUDA 12.8 和 LeRobot 0.6.1。

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg libgl1 libegl1

python3.12 -m venv .venv-lerobot
.venv-lerobot/bin/python -m pip install --upgrade pip
.venv-lerobot/bin/python -m pip install \
  -r requirements-lerobot-linux-cuda.txt

nvidia-smi
.venv-lerobot/bin/python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'

# 无桌面的 Linux 服务器使用 EGL 渲染 MuJoCo 相机。
export MUJOCO_GL=egl
export HF_HOME=/tmp/gripper-mujoco-hf-cache
```

`device: auto` 的优先级是 CUDA、MPS、CPU；Linux 配置显式使用 `device: cuda`，CUDA
不可用时会立即报错，不会静默回退 CPU。

可以直接复制并复用以下设备无关产物，减少 4090 机器上的前置工作。复制已有的
Reflect checkpoint 后，将它放到 CUDA 配置使用的独立目录：

```bash
mkdir -p outputs/graph_pretrain/reflectvlm_cuda
cp /path/to/copied/reflectvlm/checkpoint.pt \
  outputs/graph_pretrain/reflectvlm_cuda/checkpoint.pt
```

LeRobotDataset 保持原路径 `outputs/lerobot/franka_lerobot_act_pilot/`。CUDA 训练结果使用
独立的 `*_cuda` 目录，不会覆盖 macOS checkpoint。

如果没有复制 LeRobotDataset，先在 Linux 生成 expert gate 和数据：

```bash
.venv-lerobot/bin/python -m interaction_vla.validate_physics_expert \
  --config configs/physics_pilot_macos.yaml

.venv-lerobot/bin/python -m interaction_vla.lerobot_bridge collect \
  --config configs/lerobot_act_recovery_linux_cuda.yaml

.venv-lerobot/bin/python -m interaction_vla.lerobot_bridge validate \
  --config configs/lerobot_act_recovery_linux_cuda.yaml
```

`physics_pilot_macos.yaml` 的文件名是历史遗留；其 MuJoCo 物理参数不依赖 macOS，CUDA
bridge 复用它是为了保持 expert gate 和数据合同一致。

先在 CUDA 上重新建立 ACT recovery gate：

```bash
.venv-lerobot/bin/python -m interaction_vla.lerobot_bridge act-check \
  --config configs/lerobot_act_recovery_linux_cuda.yaml

.venv-lerobot/bin/python -m interaction_vla.lerobot_bridge act-train \
  --config configs/lerobot_act_recovery_linux_cuda.yaml

.venv-lerobot/bin/python -m interaction_vla.lerobot_bridge act-diagnose \
  --config configs/lerobot_act_recovery_linux_cuda.yaml \
  --checkpoint outputs/graph_control/act_recovery_cuda/train/checkpoint

.venv-lerobot/bin/python -m interaction_vla.lerobot_bridge act-recovery \
  --config configs/lerobot_act_recovery_linux_cuda.yaml \
  --checkpoint outputs/graph_control/act_recovery_cuda/train/checkpoint
```

如果没有复制 Reflect checkpoint，使用 CUDA 预训练：

```bash
.venv-lerobot/bin/python -m interaction_vla.graph_pretrain inspect \
  --config configs/reflectvlm_graph_pretrain_linux_cuda.yaml

.venv-lerobot/bin/python -m interaction_vla.graph_pretrain train \
  --config configs/reflectvlm_graph_pretrain_linux_cuda.yaml

.venv-lerobot/bin/python -m interaction_vla.graph_pretrain evaluate \
  --config configs/reflectvlm_graph_pretrain_linux_cuda.yaml \
  --checkpoint outputs/graph_pretrain/reflectvlm_cuda/checkpoint.pt \
  --partition test
```

运行 CUDA Oracle Graph v2：

```bash
.venv-lerobot/bin/python -m interaction_vla.graph_finetune inspect \
  --config configs/mujoco_graph_v2_finetune_linux_cuda.yaml

.venv-lerobot/bin/python -m interaction_vla.graph_finetune split \
  --config configs/mujoco_graph_v2_finetune_linux_cuda.yaml

.venv-lerobot/bin/python -m interaction_vla.graph_control inspect \
  --config configs/graph_v2_act_oracle_linux_cuda.yaml

.venv-lerobot/bin/python -m interaction_vla.graph_control cache \
  --config configs/graph_v2_act_oracle_linux_cuda.yaml

.venv-lerobot/bin/python -m interaction_vla.graph_control compare \
  --config configs/graph_v2_act_oracle_linux_cuda.yaml

.venv-lerobot/bin/python -m interaction_vla.graph_control evaluate \
  --config configs/graph_v2_act_oracle_linux_cuda.yaml
```

Oracle gate 通过后运行 predicted Graph 和四条件主实验：

```bash
.venv-lerobot/bin/python -m interaction_vla.graph_finetune compare \
  --config configs/mujoco_graph_v2_finetune_linux_cuda.yaml

.venv-lerobot/bin/python -m interaction_vla.graph_control inspect \
  --config configs/graph_v2_act_pilot_linux_cuda.yaml

.venv-lerobot/bin/python -m interaction_vla.graph_control cache \
  --config configs/graph_v2_act_pilot_linux_cuda.yaml

.venv-lerobot/bin/python -m interaction_vla.graph_control compare \
  --config configs/graph_v2_act_pilot_linux_cuda.yaml

.venv-lerobot/bin/python -m interaction_vla.graph_control evaluate \
  --config configs/graph_v2_act_pilot_linux_cuda.yaml
```

Linux 服务器使用 `tmux` 或 `nohup` 保持长任务；`caffeinate` 只适用于 macOS。MuJoCo
采集和闭环 rollout 仍主要受 CPU 限制，4090 的主要收益来自 ACT、Graph pretraining 和
Graph fine-tuning。

## 从零运行

### 1. 生成 expert gate 和 LeRobotDataset

先验证 MuJoCo expert，再采集 50 个 episode。这里直接使用 recovery 配置采集，不依赖
另一个本地 smoke report。

```bash
.venv/bin/python -m interaction_vla.validate_physics_expert \
  --config configs/physics_pilot_macos.yaml

.venv-lerobot/bin/python -m interaction_vla.lerobot_bridge collect \
  --config configs/lerobot_act_recovery_macos.yaml

.venv-lerobot/bin/python -m interaction_vla.lerobot_bridge validate \
  --config configs/lerobot_act_recovery_macos.yaml
```

输出数据位于 `outputs/lerobot/franka_lerobot_act_pilot/`，包含标准 LeRobot 数据和独立
teacher sidecar。目标目录已存在时命令会拒绝覆盖；保留旧实验，或在新配置中使用新的
输出目录。

### 2. 建立可用的 ACT 闭环基线

Graph 实验只有在 ACT recovery gate 通过后才有意义。

```bash
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

必须生成且通过：

```text
outputs/graph_control/act_recovery/evaluation/recovery_report.json
```

固定 gate 是 train-seen success `>= 0.80` 且 held-out success `>= 0.30`。未通过时不要
继续 Graph 主实验。

### 3. ReflectVLM Graph pretraining

```bash
.venv-lerobot/bin/python -m interaction_vla.graph_pretrain inspect \
  --config configs/reflectvlm_graph_pretrain_macos.yaml

.venv-lerobot/bin/python -m interaction_vla.graph_pretrain train \
  --config configs/reflectvlm_graph_pretrain_macos.yaml

.venv-lerobot/bin/python -m interaction_vla.graph_pretrain evaluate \
  --config configs/reflectvlm_graph_pretrain_macos.yaml \
  --checkpoint outputs/graph_pretrain/reflectvlm/checkpoint.pt \
  --partition test
```

首次 `inspect/train` 会从 `yunhaif/ReflectVLM-data-expert` 下载数据。SSL 失败时重新运行
同一命令即可复用已下载的 Hugging Face cache；不要在 fresh clone 上设置
`HF_HUB_OFFLINE=1`。

### 4. 固定 Graph v2 split 并生成 Oracle cache

```bash
.venv-lerobot/bin/python -m interaction_vla.graph_finetune inspect \
  --config configs/mujoco_graph_v2_finetune_macos.yaml

.venv-lerobot/bin/python -m interaction_vla.graph_finetune split \
  --config configs/mujoco_graph_v2_finetune_macos.yaml

.venv-lerobot/bin/python -m interaction_vla.graph_control inspect \
  --config configs/graph_v2_act_oracle_macos.yaml

.venv-lerobot/bin/python -m interaction_vla.graph_control cache \
  --config configs/graph_v2_act_oracle_macos.yaml
```

这一步冻结 40/5/5 episode split，并生成相同 6722-row 的 Flat 与 Oracle Graph v2 token
cache。cache 已存在时不要重复执行。

## Oracle Graph v2 vs Flat

这是 predicted Graph fine-tuning 之前的必要实验：先证明“正确 Graph”本身能提高 ACT，
再研究视觉预测 Graph 能恢复多少 oracle gap。

```bash
.venv-lerobot/bin/python -m interaction_vla.graph_control compare \
  --config configs/graph_v2_act_oracle_macos.yaml

.venv-lerobot/bin/python -m interaction_vla.graph_control evaluate \
  --config configs/graph_v2_act_oracle_macos.yaml
```

正式报告位于：

```text
outputs/graph_control/graph_v2_oracle/runs/evaluation/report.json
```

只有同时满足以下条件才继续：

- `oracle_graph_v2 - flat` 成功率至少提高 `0.10`；
- wrong-object stable grasp 不增加；
- `oracle_gate.passed` 为 `true`。

查看 gate：

```bash
.venv-lerobot/bin/python -c 'import json; p=json.load(open("outputs/graph_control/graph_v2_oracle/runs/evaluation/report.json")); print(json.dumps(p["oracle_gate"], indent=2))'
```

## Oracle gate 通过之后

### 1. MuJoCo Graph v2 fine-tuning

训练 random-init 与 ReflectVLM-init 的配对视觉 Graph estimator：

```bash
.venv-lerobot/bin/python -m interaction_vla.graph_finetune compare \
  --config configs/mujoco_graph_v2_finetune_macos.yaml
```

输出位于 `outputs/graph_finetune/mujoco_graph_v2/`。该 checkpoint 是 Graph estimator，
不是可以直接 rollout 的控制策略。

### 2. 四条件 ACT 连续控制主实验

```bash
.venv-lerobot/bin/python -m interaction_vla.graph_control inspect \
  --config configs/graph_v2_act_pilot_macos.yaml

.venv-lerobot/bin/python -m interaction_vla.graph_control cache \
  --config configs/graph_v2_act_pilot_macos.yaml

.venv-lerobot/bin/python -m interaction_vla.graph_control compare \
  --config configs/graph_v2_act_pilot_macos.yaml

.venv-lerobot/bin/python -m interaction_vla.graph_control evaluate \
  --config configs/graph_v2_act_pilot_macos.yaml
```

最终比较 Flat、Oracle Graph v2、Predicted Random v2 和 Predicted Reflect v2，并报告
paired contrasts、wrong-object interaction、`oracle_gap_recovered` 以及 ID/OOD 指标。

## 长训练与中断

当前 ACT/Graph 正式训练没有 `tqdm`，通常只在完成后输出最终 JSON，也不支持从 epoch
checkpoint 续训。中断后需要从该条正式训练命令重新开始。M2 8 GB 上 Oracle 的 Flat +
Oracle、各 10 epoch 训练通常需要数小时。

建议长任务在后台运行。以 Oracle compare 为例：

```bash
mkdir -p outputs/logs

nohup caffeinate -dimsu \
  env HF_HOME=/tmp/gripper-mujoco-hf-cache \
  .venv-lerobot/bin/python -m interaction_vla.graph_control compare \
  --config configs/graph_v2_act_oracle_macos.yaml \
  > outputs/logs/graph_v2_oracle_compare.log 2>&1 &

oracle_train_pid=$!
echo "$oracle_train_pid"
```

查看进程和日志：

```bash
ps -p "$oracle_train_pid" -o pid,etime,%cpu,%mem,state,command
tail -f outputs/logs/graph_v2_oracle_compare.log
```

日志长时间没有新增不代表训练停止。Oracle compare 的第一个 checkpoint 出现时，Flat
已经完成，整体大约过半：

```bash
find outputs/graph_control/graph_v2_oracle \
  -path '*/flat/checkpoint/training_summary.json'
```

## 测试

不要在 M2 8 GB 正式训练期间同时运行完整测试。空闲时运行：

```bash
PYTHONPYCACHEPREFIX=/tmp/gripper-mujoco-pycache \
  .venv/bin/python -m pytest tests/interaction_vla -q

HF_HOME=/tmp/gripper-mujoco-hf-cache \
PYTHONPYCACHEPREFIX=/tmp/gripper-mujoco-lerobot-pycache \
  .venv-lerobot/bin/python -m pytest tests/interaction_vla -q
```

## 项目目录

```text
interaction_vla/    MuJoCo、Graph、ACT、训练、评估和 LeRobot bridge
configs/            固定的 smoke/pilot 实验配置
tests/              单元测试与端到端验证
outputs/            本机数据、cache、checkpoint、GIF 和报告
```

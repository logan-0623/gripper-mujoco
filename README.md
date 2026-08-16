# Interaction-Structured VLA

本项目研究：**什么样的 interaction graph 对机器人连续控制策略真正有价值？**

Graph 不复制完整 MuJoCo state，只编码任务实体、当前交互关系和下一步应改变的关系。
它满足 task-conditioned、object-centric、坐标不变、时序一致，并可由 agent RGB、
wrist RGB、10D 末端状态和任务语言估计。

当前已经接通：

- Franka MuJoCo 接触物理的 Graph vs Flat 主实验；
- 标准 `LeRobotDataset`、双 RGB、10D state、7D continuous action 和 ACT；
- ReflectVLM Graph pretraining；
- ReflectVLM → MuJoCo Graph fine-tuning；
- frozen predicted Graph → ACT continuous-control 四条件对照。

π0、SmolVLA 和多语言泛化尚未实现。当前数据只有一个任务指令，因此不能宣称语言泛化。

![Franka contact expert](docs/media/franka_contact_expert.gif)

## 安装与测试

物理环境：

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -r requirements-macos.txt
.venv/bin/python -m interaction_vla.macos_mjpython
```

LeRobot 环境：

```bash
python3.12 -m venv .venv-lerobot
.venv-lerobot/bin/python -m pip install -r requirements-lerobot-macos.txt
```

测试：

```bash
.venv/bin/python -m pytest tests/interaction_vla -q

HF_HOME=/tmp/gripper-mujoco-hf-cache \
  .venv-lerobot/bin/python -m pytest tests/interaction_vla -q
```

临时代理：

```bash
export https_proxy=http://127.0.0.1:7890
export http_proxy=http://127.0.0.1:7890
export all_proxy=socks5://127.0.0.1:7890
```

## 已完成实验

### 1. Graph vs Flat 接触物理

```bash
.venv/bin/python -m interaction_vla.validate_physics_expert \
  --config configs/physics_interaction_chunk_pilot_macos.yaml

.venv/bin/python -m interaction_vla.physics_data collect \
  --config configs/physics_interaction_chunk_pilot_macos.yaml

.venv/bin/python -m interaction_vla.train \
  --config configs/physics_interaction_chunk_pilot_macos.yaml \
  --representation flat --model-seed 0

.venv/bin/python -m interaction_vla.train \
  --config configs/physics_interaction_chunk_pilot_macos.yaml \
  --representation graph --model-seed 0
```

查看 learned Graph rollout：

```bash
.venv/bin/python -m interaction_vla.physics_visualize dashboard \
  --config configs/physics_interaction_chunk_pilot_macos.yaml \
  --controller graph \
  --checkpoint outputs/interaction_graph_physics/interaction_chunk_pilot/graph/seed_0/checkpoint.pt \
  --layout crowded --object-count 4 --seed 2140049
```

### 2. LeRobotDataset 与基础 ACT

```bash
.venv-lerobot/bin/python -m interaction_vla.lerobot_bridge collect \
  --config configs/lerobot_act_pilot_macos.yaml

.venv-lerobot/bin/python -m interaction_vla.lerobot_bridge validate \
  --config configs/lerobot_act_pilot_macos.yaml
```

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

### 4. MuJoCo Graph fine-tuning

```bash
.venv-lerobot/bin/python -m interaction_vla.graph_finetune inspect \
  --config configs/mujoco_graph_finetune_pilot_macos.yaml

.venv-lerobot/bin/python -m interaction_vla.graph_finetune compare \
  --config configs/mujoco_graph_finetune_pilot_macos.yaml
```

已有 pilot 结果支持：Reflect 初始化主要改善 next-relation goal、operator、predicate 和
residual；静态实体/关系重建并不是稳定收益来源。

## ACT 闭环基线（已完成）

已完成的 Graph-conditioned ACT v1 审计包含 240 次 rollout，但所有条件成功数均为 0；
Flat 在训练集已见场景也会 timeout。因此该结果不能支持 Graph 优于或差于 Flat，当前瓶颈
是基础 ACT 尚未学会闭环任务。

按顺序运行：

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

结果写入 `outputs/graph_control/act_recovery/`。当前 recovery gate 已通过：train-seen
成功率为 0.90，held-out 成功率为 0.70，分别高于 0.80 和 0.30 的门槛。下一项实验是
Interaction Graph v2：先验证 oracle Graph 是否提高连续控制，再训练视觉 predicted Graph。

## 项目结构

```text
interaction_vla/    环境、Graph、ACT、训练、评估与 LeRobot bridge
configs/            smoke/pilot 实验配置
tests/              单元测试与端到端验证
outputs/            本地数据、checkpoint、GIF 和报告
docs/               设计与实验记录
```

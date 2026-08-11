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

## 下一步：Graph-conditioned ACT 主实验

四个条件使用完全相同的双 RGB、10D proprioception、ACT 参数量、初始化、数据 split、
batch 顺序和固定 5 epoch 预算：

1. `flat`：75D Graph token 全零；
2. `predicted_random`：随机初始化后 MuJoCo fine-tune 的 Graph；
3. `predicted_reflect`：ReflectVLM 初始化后 MuJoCo fine-tune 的 Graph；
4. `oracle_current`：仅当前实体/关系使用 causal MuJoCo teacher，下一关系目标仍由视觉
   Graph 预测。

`oracle_current` 是 privileged perception upper bound，不是可部署模型；它不会读取用未来
轨迹计算的 `annotation.tc_tig.relation_goal`。

先跑工程 smoke：

```bash
.venv-lerobot/bin/python -m interaction_vla.graph_control inspect \
  --config configs/graph_control_act_smoke_macos.yaml

.venv-lerobot/bin/python -m interaction_vla.graph_control cache \
  --config configs/graph_control_act_smoke_macos.yaml

.venv-lerobot/bin/python -m interaction_vla.graph_control smoke \
  --config configs/graph_control_act_smoke_macos.yaml
```

smoke 只验证四个条件都能完成一次 optimizer update、保存并无误重载，不产生策略性能结论。

smoke 通过后运行正式三 seed 实验：

```bash
.venv-lerobot/bin/python -m interaction_vla.graph_control inspect \
  --config configs/graph_control_act_pilot_macos.yaml

.venv-lerobot/bin/python -m interaction_vla.graph_control cache \
  --config configs/graph_control_act_pilot_macos.yaml

.venv-lerobot/bin/python -m interaction_vla.graph_control compare \
  --config configs/graph_control_act_pilot_macos.yaml

.venv-lerobot/bin/python -m interaction_vla.graph_control evaluate \
  --config configs/graph_control_act_pilot_macos.yaml
```

主要比较：

- `predicted_reflect - flat`：视觉 Graph 是否提高连续控制；
- `predicted_reflect - predicted_random`：ReflectVLM 预训练是否有迁移价值；
- `oracle_current - predicted_reflect`：当前 Graph 感知误差是否是瓶颈。

正式报告使用 policy seed 作为独立重复单位，保留每个 paired case，并报告 success、
wrong-object interaction/stable grasp、drop、timeout、IK projection、action clipping 和
gripper switching。

输出：

```text
outputs/graph_control/act_smoke/cache/       seed-0 frozen token cache
outputs/graph_control/act_smoke/runs/        四条件 one-update checkpoint
outputs/graph_control/act_pilot/cache/       三 seed frozen token cache
outputs/graph_control/act_pilot/runs/        正式 ACT checkpoint 与 paired report
```

## 项目结构

```text
interaction_vla/    环境、Graph、ACT、训练、评估与 LeRobot bridge
configs/            smoke/pilot 实验配置
tests/              单元测试与端到端验证
outputs/            本地数据、checkpoint、GIF 和报告
docs/               设计与实验记录
```

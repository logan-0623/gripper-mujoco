# Interaction-Structured VLA

这个项目研究一个问题：**什么样的 interaction graph 对机器人策略真正有价值？**

当前 Graph 不复制完整 MuJoCo state，而只表达：

1. 哪些实体与任务相关；
2. 它们当前处于什么交互关系；
3. 下一步应该改变哪一种关系。

Graph 的设计约束是 task-conditioned、object-centric、坐标不变、时序一致，且最终
能够从 RGB、腕部 RGB、末端状态和语言中估计。它主要回答目标物体、夹爪—目标、
目标—容器、干扰物和交互阶段这五类问题。

## 当前项目包含什么

项目现在有两条可运行链路：

- **Franka 接触物理 Graph/Flat 基线**：完整 Panda、双指夹爪、MuJoCo 接触、7D
  Cartesian action、H=8 action chunk、recovery 数据、paired evaluation 和
  edge-shuffle 消融。Flat 与 Graph 共用数据、时序头、controller 和训练预算，主要
  变量只有 encoder。
- **LeRobot/VLA bridge**：标准 `LeRobotDataset`、agent RGB、wrist RGB、10D
  末端状态、7D action、language task metadata、本地 Hugging Face checkpoint/data
  目录、ACT smoke 训练、MuJoCo 闭环 rollout 和双视角 GIF。

TC-TIG interaction graph 标签保存在 teacher sidecar 中，不会混入标准 policy batch。
当前 ACT 不使用语言；π0、SmolVLA、视觉估计 Graph 和 Hugging Face Hub upload
尚未接入。不要把 500-step ACT smoke 当作任务性能结果。

![Franka contact expert](docs/media/franka_contact_expert.gif)

## Mac 安装

项目使用两个独立环境。

物理 Graph/Flat 环境：

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements-macos.txt
.venv/bin/python -m interaction_vla.macos_mjpython
```

LeRobot 环境：

```bash
python3.12 -m venv .venv-lerobot
.venv-lerobot/bin/python -m pip install -r requirements-lerobot-macos.txt
```

已验证的关键版本是 Python 3.12、MuJoCo 3.3.4、LeRobot 0.6.1、Torch 2.10 和
TorchCodec 0.10。Apple Silicon 上优先使用 MPS，否则回退 CPU。

## 常用命令

### 运行测试

```bash
PYTHONPYCACHEPREFIX=/tmp/gripper-mujoco-pycache \
  .venv/bin/python -m pytest tests/interaction_vla -q

HF_HOME=/tmp/gripper-mujoco-hf-cache \
PYTHONPYCACHEPREFIX=/tmp/gripper-mujoco-lerobot-pycache \
  .venv-lerobot/bin/python -m pytest \
  tests/interaction_vla/lerobot_bridge -q
```

### 查看 Franka 物理场景

四视角 expert dashboard：

```bash
.venv/bin/python -m interaction_vla.physics_visualize dashboard \
  --controller expert \
  --layout crowded \
  --object-count 4 \
  --seed 2140049
```

键盘遥操作并保存 RGB-D：

```bash
.venv/bin/python -m interaction_vla.physics_visualize teleop \
  --layout normal \
  --object-count 3 \
  --seed 2140049 \
  --record outputs/teleop_demo.npz
```

控制键：`WASD` 控制 xy，`R/F` 控制 z，方向键控制 rx/ry，`Q/E` 控制 rz，
空格切换夹爪，`Z` 重置，`Esc` 退出。

## 实验 1：Graph vs Flat 接触物理主实验

当前推荐配置是：

```text
configs/physics_interaction_chunk_pilot_macos.yaml
```

它采集 200 条 base demonstration，使用 source-level train/validation/test split，
只从训练 source 生成 recovery。先完成 seed 0，不要一开始就扩大实验。

### 1.1 Expert gate 与数据采集

源码、config 或 controller 改动后，旧 gate、数据和 checkpoint 会因 provenance
不匹配而失效。此时从第一条命令重新运行，不要绕过校验。

```bash
.venv/bin/python -m interaction_vla.validate_physics_expert \
  --config configs/physics_interaction_chunk_pilot_macos.yaml

.venv/bin/python -m interaction_vla.physics_data collect \
  --config configs/physics_interaction_chunk_pilot_macos.yaml
```

### 1.2 训练 seed 0 Flat 与 Graph

```bash
.venv/bin/python -m interaction_vla.train \
  --config configs/physics_interaction_chunk_pilot_macos.yaml \
  --representation flat \
  --model-seed 0

.venv/bin/python -m interaction_vla.train \
  --config configs/physics_interaction_chunk_pilot_macos.yaml \
  --representation graph \
  --model-seed 0
```

### 1.3 先跑 ID 与 held-out recovery sanity

```bash
.venv/bin/python -m interaction_vla.physics_evaluate \
  --config configs/physics_interaction_chunk_pilot_macos.yaml \
  --model-seeds 0 \
  --conditions id_normal heldout_recovery \
  --episodes-per-count 5 \
  --output outputs/interaction_graph_physics/interaction_chunk_pilot/evaluation/seed0_sanity.json
```

先检查 stable contact/grasp/lift、wrong-object interaction、drop、transport progress，
再看 strict placement 和 task success。只有 seed 0 出现真实控制、抓取和放置信号，
才继续 OOD。

### 1.4 OOD 与 Graph edge-shuffle

```bash
.venv/bin/python -m interaction_vla.physics_evaluate \
  --config configs/physics_interaction_chunk_pilot_macos.yaml \
  --model-seeds 0 \
  --conditions count_ood crowded_ood \
  --episodes-per-count 5 \
  --include-edge-shuffle \
  --output outputs/interaction_graph_physics/interaction_chunk_pilot/evaluation/seed0_ood_ablation.json
```

edge-shuffle 用于确认 Graph 是否真的使用关系结构。单个 GIF 或单个 seed 不能证明
Graph > Flat；主结论最终需要预先固定的三个模型 seed 和完整 paired evaluation。
当前 pilot config 只登记 seed 0，扩大到三个 seed 前应复制并冻结一份新配置，将
`train.model_seeds` 明确设置为 `[0, 1, 2]`。

### 1.5 查看 learned rollout

```bash
.venv/bin/python -m interaction_vla.physics_visualize dashboard \
  --config configs/physics_interaction_chunk_pilot_macos.yaml \
  --controller graph \
  --checkpoint outputs/interaction_graph_physics/interaction_chunk_pilot/graph/seed_0/checkpoint.pt \
  --layout crowded \
  --object-count 4 \
  --seed 2140049
```

导出同初始状态的 Flat/Graph 对比 GIF：

```bash
.venv/bin/mjpython -m interaction_vla.physics_visualize export-comparison-gif \
  --config configs/physics_interaction_chunk_pilot_macos.yaml \
  --flat-checkpoint outputs/interaction_graph_physics/interaction_chunk_pilot/flat/seed_0/checkpoint.pt \
  --graph-checkpoint outputs/interaction_graph_physics/interaction_chunk_pilot/graph/seed_0/checkpoint.pt \
  --layout crowded \
  --object-count 4 \
  --seed 2140049 \
  --output docs/media/interaction_chunk_flat_vs_graph.gif
```

## 实验 2：LeRobotDataset 与 ACT 闭环

这条链路验证数据、模型、checkpoint 和 MuJoCo 控制是否接通，不验证语言泛化。
源码或 requirements 改动后，从 `collect` 重新生成 provenance 一致的制品。

```bash
.venv-lerobot/bin/python -m interaction_vla.lerobot_bridge collect \
  --config configs/lerobot_act_smoke_macos.yaml

.venv-lerobot/bin/python -m interaction_vla.lerobot_bridge validate \
  --config configs/lerobot_act_smoke_macos.yaml

.venv-lerobot/bin/python -m interaction_vla.lerobot_bridge smoke \
  --config configs/lerobot_act_smoke_macos.yaml
```

运行 ACT checkpoint 并导出 agent/wrist RGB GIF：

```bash
.venv-lerobot/bin/python -m interaction_vla.lerobot_bridge rollout \
  --config configs/lerobot_act_smoke_macos.yaml \
  --checkpoint outputs/lerobot/act_smoke/checkpoint \
  --object-count 2 \
  --gif outputs/lerobot/act_smoke/rollout.gif
```

主要输出：

```text
outputs/lerobot/franka_lerobot_act_smoke/   标准 LeRobotDataset + teacher sidecar
outputs/lerobot/act_smoke/checkpoint/       本地 Hugging Face checkpoint
outputs/lerobot/act_smoke/smoke_report.json 工程 smoke 报告
outputs/lerobot/act_smoke/rollout.json      闭环诊断
outputs/lerobot/act_smoke/rollout.gif       双视角动画
```

`passed=true` 表示工程链路通过；`task_success=false` 或 `timeout` 表示 smoke 模型没有
学会任务，不能写成成功率证据。

## 之后应该跑哪些实验

按下面顺序推进，避免一次改变 perception、representation 和 policy backbone 三个变量：

1. **完成物理 seed-0 sanity**：先证明 Flat 与 Graph 都具备基本接触、抓取和 transport
   能力。
2. **运行 OOD + edge-shuffle**：检查 Graph 的收益是否来自正确关系，而不是参数量。
3. **冻结三随机种子主实验**：新建 `[0,1,2]` 配置，完整运行 ID、held-out recovery、
   count OOD 和 crowded OOD，报告 paired Graph-minus-Flat delta。
4. **Graph representation ablation**：分别验证 task entity selection、gripper-target、
   target-receptacle、distractor risk、interaction phase 和 next-relation goal。当前只有
   Graph/Flat 与 edge-shuffle 已实现，其余消融需要先实现独立配置和 encoder 开关。
5. **视觉 Graph estimator**：固定已经验证的 Graph schema，再从 agent RGB、wrist RGB、
   10D 末端状态和语言预测实体、关系、phase 与 next-relation goal；不要输入完整 MuJoCo
   state。
6. **VLA backbone 对比**：在同一 LeRobotDataset、同一 Graph estimator 输出和同一评估
   cases 上比较 ACT、π0、SmolVLA。当前仓库只实现 ACT，π0/SmolVLA 接入是后续工作。

## 结果边界

历史 kinematic/recovery 实验曾显示 Graph 降低 wrong-object interaction，并且打乱边会
破坏表现，但旧结果没有证明稳定的 Graph > Flat 闭环成功率优势。当前主结论只能来自
`physics_interaction_chunk_pilot_macos.yaml` 对应的新报告：

```text
outputs/interaction_graph_physics/interaction_chunk_pilot/evaluation/
```

## 常见问题

- `bridge source/config/gate hash mismatch`：源码或配置变了，重新运行 collect/smoke，
  不要手工修改 fingerprint。
- `libpython3.12.dylib`：运行
  `.venv/bin/python -m interaction_vla.macos_mjpython`。
- macOS 的 `Class ... is implemented in both ...`：这是 PyAV/OpenCV/Homebrew 原生库
  重复加载警告；用进程退出码和最终 JSON 判断命令是否失败。
- TorchCodec：本项目要求 Torch 2.10 搭配 TorchCodec 0.10，不要升级成 0.11。

## 项目结构

```text
interaction_vla/    环境、Graph/Flat 模型、训练、评估、可视化和 LeRobot bridge
configs/            当前与历史实验配置
tests/              单元测试和端到端验证
outputs/            本地数据、checkpoint、GIF 和实验报告
docs/               设计记录、实验记录和媒体文件
```

历史 recovery、terminal-recovery 和 kinematic 配置仍保留用于复现，但新的研究工作优先
使用 interaction-chunk 物理主线和 LeRobot bridge。

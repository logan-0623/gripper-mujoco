# Interaction-Structured VLA

本项目研究：**什么样的 interaction graph 对机器人连续控制策略真正有价值？**

策略输入是 agent RGB、wrist RGB、10D 末端状态（位置、6D 旋转、夹爪）和任务语言。
Graph 不复制完整 MuJoCo state，只编码：任务相关实体、当前交互关系、关系的时间变化，以及
下一步应改变的关系。

当前已实现：

- Franka MuJoCo 接触物理与 Graph vs Flat 实验；
- 标准 `LeRobotDataset`、双 RGB、language-conditioned data、7D continuous action；
- ACT 训练、checkpoint、闭环 rollout 与 Hugging Face/LeRobot 数据接口；
- ReflectVLM Graph pretraining；
- 因果、对象中心、坐标不变、89D Interaction Graph v2；
- `ReflectVLM pretraining → MuJoCo Graph fine-tuning → ACT continuous control`。

π0、SmolVLA 和多语言泛化尚未实现。当前数据只有一个任务指令，不能宣称语言泛化。

![Franka contact expert](docs/media/franka_contact_expert.gif)

## 当前结论

- Graph vs Flat 接触物理主实验已完成，是 Graph 有价值的第一项证据。
- ACT recovery gate 已通过：train-seen `0.90`，held-out `0.70`，门槛为 `0.80/0.30`。
- ReflectVLM Graph pretraining 已完成。
- Graph v2 的 split、6722-row Flat/Oracle cache 和一步 paired ACT smoke 已通过。
- **尚未得到 Oracle Graph v2 的正式闭环结论。当前最早未通过的门槛就是 Oracle vs Flat。**

旧 `graph_control_act_*` 配置和 75D Graph cache 属于 v1，不可用于 Graph v2。

## 现在运行：Oracle Graph v2 vs Flat

准备工作已经完成。下一步只运行正式训练和闭环评估：

```bash
HF_HOME=/tmp/gripper-mujoco-hf-cache \
  .venv-lerobot/bin/python -m interaction_vla.graph_control compare \
  --config configs/graph_v2_act_oracle_macos.yaml

HF_HOME=/tmp/gripper-mujoco-hf-cache \
  .venv-lerobot/bin/python -m interaction_vla.graph_control evaluate \
  --config configs/graph_v2_act_oracle_macos.yaml
```

正式报告位于：

```text
outputs/graph_control/graph_v2_oracle/runs/evaluation/report.json
```

只有同时满足以下条件才继续 predicted Graph：

- `oracle_graph_v2 - flat` 成功率至少提高 `0.10`；
- wrong-object stable grasp 不增加。

一步 smoke checkpoint 已保留在
`outputs/graph_control/graph_v2_oracle/smoke_runs/`，不参与正式结论。

## Oracle gate 通过之后

先训练 MuJoCo Graph v2 的 random/Reflect 配对模型：

```bash
.venv-lerobot/bin/python -m interaction_vla.graph_finetune inspect \
  --config configs/mujoco_graph_v2_finetune_macos.yaml

.venv-lerobot/bin/python -m interaction_vla.graph_finetune compare \
  --config configs/mujoco_graph_v2_finetune_macos.yaml
```

再运行三 seed、四条件的连续控制主实验：

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

最终比较：Flat、Oracle Graph v2、Predicted Random v2、Predicted Reflect v2；报告保留
policy seed 独立重复、五个 paired contrasts、`oracle_gap_recovered` 和 ID/OOD 标签。

## 安装与测试

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -r requirements-macos.txt
.venv/bin/python -m interaction_vla.macos_mjpython

python3.12 -m venv .venv-lerobot
.venv-lerobot/bin/python -m pip install -r requirements-lerobot-macos.txt

HF_HOME=/tmp/gripper-mujoco-hf-cache \
  .venv-lerobot/bin/python -m pytest tests/interaction_vla -q
```

macOS 中 `av`、`cv2` 和 Homebrew FFmpeg 打印的重复 Objective-C class 信息是已知警告；
以命令退出码和最终 JSON 的 `passed` 字段为准。

## 文档

- [Graph v2 设计](docs/superpowers/specs/2026-08-14-act-control-recovery-graph-v2-design.md)
- [Graph v2 实施计划](docs/superpowers/plans/2026-08-14-interaction-graph-v2.md)
- [ACT recovery 实施计划](docs/superpowers/plans/2026-08-14-act-control-recovery.md)

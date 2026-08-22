# From Imitation to Improvement

本项目面向 ICRA，研究：

> **Does reinforcement learning induce action-relevant interaction structure that supervised imitation fails to capture?**

Interaction Graph 不再是所有策略必须使用的输入，而是统一的 measurement ontology、privileged label 和 intervention vocabulary。项目分别测量：

- **Accessible / C**：冻结表征能否被 lightweight probe 读出；
- **Used**：改变 latent 是否改变策略动作；
- **Useful / U**：该策略在固定闭环 case 上是否成功；
- **Plasticity / P**：固定 online budget 下的 RL learning-curve AUC。

ACT 保留为 controlled mechanism study；SmolVLA 是主投稿的 modern VLA validation；π0 是可选 external validation。旧 Graph-vs-Flat 结果和 pipeline 均保留，不会被新实验覆盖。

研究设计见 [ICRA experiment design](docs/superpowers/specs/2026-08-20-icra-interaction-representation-study-design.md)，项目状态见 [ccfa.yaml](ccfa.yaml)。

## 已有结论

旧 ACT 主实验为 3 policy seeds、每个条件共 60 个 paired rollout：

| Representation | Success | Target drop | Action clipping |
|---|---:|---:|---:|
| Flat | 30.0% | 6.7% | 0.117 |
| Privileged Teacher Graph | 35.0% | 0.0% | 0.096 |
| Predicted Random | 40.0% | 10.0% | 0.114 |
| Predicted Reflect | 41.7% | 0.0% | 0.092 |

目前只能稳健地说明：Graph prediction correctness 不等于 control utility；Teacher Graph 不是 policy-performance upper bound；Reflect 初始化没有稳定改善成功率。新的 stage-wise 实验用于检验 RL 是否选择性强化 contact、phase、next-relation 和 recovery 等 interaction-critical factors。

## 环境

### macOS / Apple Silicon

```bash
python3.12 -m venv .venv-lerobot
.venv-lerobot/bin/python -m pip install --upgrade pip
.venv-lerobot/bin/python -m pip install -r requirements-lerobot-macos.lock.txt

export HF_HOME=/tmp/gripper-mujoco-hf-cache
```

### Linux / RTX 4090

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg libgl1 libegl1

python3.12 -m venv .venv-lerobot
.venv-lerobot/bin/python -m pip install --upgrade pip
.venv-lerobot/bin/python -m pip install -r requirements-lerobot-linux-cuda.txt

export MUJOCO_GL=egl
export HF_HOME=/tmp/gripper-mujoco-hf-cache

.venv-lerobot/bin/python -c \
  'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
```

macOS 的 `av`、`cv2`、SDL、FFmpeg duplicate-class 信息通常只是动态库警告；以退出码与最终 JSON 的 `passed` 为准。

## 前置工件

新实验复用现有 LeRobot 数据集、split 和 ACT checkpoint：

```bash
for artifact in \
  outputs/lerobot/franka_lerobot_act_pilot/meta/info.json \
  outputs/graph_finetune/mujoco_graph_v2/split_manifest.json \
  outputs/graph_control/graph_v2_pilot/runs/seed_0/flat/checkpoint/config.json
do
  test -e "$artifact" && echo "READY   $artifact" || echo "MISSING $artifact"
done
```

若缺失，请先按 `configs/lerobot_act_recovery_{macos,linux_cuda}.yaml` 完成数据采集/验证，再按旧 `graph_v2_act_pilot_*` 配置完成 ACT 训练。`outputs/` 默认不随 Git 上传。

## 1. Fixed State Bank

所有 checkpoint 必须在同一批 held-out states 上测量。当前 State Bank 包含 expert-support 与 policy-shift 两个 domain，split 以 source episode 为单位，禁止 episode/group leakage。

先检查已有工件：

```bash
.venv-lerobot/bin/python -m interaction_vla.representation_study state-bank inspect \
  --config configs/representation_study/icra_act_macos.yaml
```

仅在 State Bank 不存在时采集：

```bash
.venv-lerobot/bin/python -m interaction_vla.representation_study state-bank collect \
  --config configs/representation_study/icra_act_macos.yaml
```

Linux 使用 `configs/representation_study/icra_act_linux_cuda.yaml`。两个配置共享同一 State Bank contract；不要分别采集两套科学样本。

## 2. Recovery RL v2 foundation

旧 `outputs/representation_study/icra/` 与已完成的 nominal-reset PPO 结果已冻结，不再追加 steps。新 protocol 单独写入 `icra_rl_v2*`，顺序固定为：分布校准 → PPO/SAC 屏选 → privileged Oracle gate → anchoring 屏选。

macOS：

```bash
CONFIG=configs/representation_study/recovery_rl_v2_act_macos.yaml

.venv-lerobot/bin/python -m interaction_vla.representation_study recovery-rl calibrate \
  --config "$CONFIG"

.venv-lerobot/bin/python -m interaction_vla.representation_study recovery-rl screen \
  --config "$CONFIG" --resume

.venv-lerobot/bin/python -m interaction_vla.representation_study recovery-rl oracle-gate \
  --config "$CONFIG"

.venv-lerobot/bin/python -m interaction_vla.representation_study recovery-rl anchor-screen \
  --config "$CONFIG" --resume
```

RTX 4090：

```bash
export MUJOCO_GL=egl
export HF_HOME=/tmp/gripper-mujoco-hf-cache
CONFIG=configs/representation_study/recovery_rl_v2_act_linux_cuda.yaml

.venv-lerobot/bin/python -m interaction_vla.representation_study recovery-rl calibrate --config "$CONFIG"
.venv-lerobot/bin/python -m interaction_vla.representation_study recovery-rl screen --config "$CONFIG" --resume
.venv-lerobot/bin/python -m interaction_vla.representation_study recovery-rl oracle-gate --config "$CONFIG"
.venv-lerobot/bin/python -m interaction_vla.representation_study recovery-rl anchor-screen --config "$CONFIG" --resume
```

`--resume` 在新目录与中断目录都安全；若已有 state 与当前 config、checkpoint 或 case manifest 不一致，程序会拒绝续训。后一个命令缺少前一个 passing gate 时会立即停止。

主要输出：

```text
outputs/representation_study/icra_rl_v2*/
├── calibration/
├── manifests/
├── anchors/
├── screen/{ppo,sac}/seed_*/
├── anchor_screen/{no_anchor,residual_only,full_anchoring}/
└── gates/{distribution,backend,oracle,anchoring}.json
```

## 3. SmolVLA modern VLA validation

建议在 4090 上运行。输入绑定到当前标准 LeRobotDataset：agent RGB、wrist RGB、10D end-effector state、language 和 7D continuous action。官方 base checkpoint 的 feature schema 会在加载时重绑定到本数据集，同时保留 foundation weights。

```bash
export MUJOCO_GL=egl
export HF_HOME=/tmp/gripper-mujoco-hf-cache
CONFIG=configs/representation_study/icra_smolvla_linux_cuda.yaml

# Foundation representation before robot SFT.
.venv-lerobot/bin/python -m interaction_vla.representation_study measure run \
  --config "$CONFIG" --backend smolvla --stage pretrained \
  --secondary-probe --closed-loop-intervention

# Robot SFT and its matched continued-SFT control.
for stage in sft continued_sft
do
  .venv-lerobot/bin/python -m interaction_vla.representation_study sft train \
    --config "$CONFIG" --backend smolvla --stage "$stage"
  .venv-lerobot/bin/python -m interaction_vla.representation_study measure run \
    --config "$CONFIG" --backend smolvla --stage "$stage" \
    --secondary-probe --closed-loop-intervention
done

# Fixed-budget residual PPO branches, both initialized from SmolVLA SFT.
for stage in rl_head rl_representation
do
  .venv-lerobot/bin/python -m interaction_vla.representation_study rl train \
    --config "$CONFIG" --backend smolvla --stage "$stage"
  .venv-lerobot/bin/python -m interaction_vla.representation_study measure run \
    --config "$CONFIG" --backend smolvla --stage "$stage" \
    --secondary-probe --closed-loop-intervention
done
```

论文主结论必须由 ACT controlled study 与 SmolVLA validation 共同支持；当前单一 language instruction 不能用于语言泛化 claim。

## 4. π0 optional validation

π0 不阻塞主实验，也不参与当前 residual-RL comparison。资源允许时运行：

```bash
CONFIG=configs/representation_study/icra_pi0_linux_cuda.yaml

.venv-lerobot/bin/python -m interaction_vla.representation_study measure run \
  --config "$CONFIG" --backend pi0 --stage pretrained --secondary-probe

for stage in sft continued_sft
do
  .venv-lerobot/bin/python -m interaction_vla.representation_study sft train \
    --config "$CONFIG" --backend pi0 --stage "$stage"
  .venv-lerobot/bin/python -m interaction_vla.representation_study measure run \
    --config "$CONFIG" --backend pi0 --stage "$stage" --secondary-probe
done
```

## 5. 汇总报告

每个 backend 的配置只汇总该 backend 已声明的阶段，并保留旧 Graph-vs-Flat 结果。报告严格区分 accessible、used、useful、plasticity 和 closed-loop intervention：

```bash
.venv-lerobot/bin/python -m interaction_vla.representation_study report build \
  --config configs/representation_study/icra_act_macos.yaml

.venv-lerobot/bin/python -m interaction_vla.representation_study report build \
  --config configs/representation_study/icra_smolvla_linux_cuda.yaml
```

主要输出：

```text
outputs/representation_study/icra/
├── state_bank/
├── latents/<backend>/<stage>/
├── probes/<backend>/<stage>/
├── interventions/<backend>/<stage>/
├── sft/<backend>/<stage>/
├── rl/<backend>/<stage>/
└── analysis/
    ├── policy_evaluation/
    └── reports/<backend>/
        ├── result_rows.json
        ├── relationship_rows.json
        ├── study_report.json
        └── study_report.md
```

`study_report.json` 即使实验未齐也会生成，`passed: true` 只表示报告构建成功；只有 `complete: true` 才表示该配置声明的 required artifacts 全部存在。

## 单项命令

调试时可拆开运行：

```bash
# Latent extraction is shard-resumable.
.venv-lerobot/bin/python -m interaction_vla.representation_study latents extract \
  --config configs/representation_study/icra_act_macos.yaml \
  --backend act --stage sft --partition all

# Frozen probes.
.venv-lerobot/bin/python -m interaction_vla.representation_study probes train \
  --config configs/representation_study/icra_act_macos.yaml \
  --backend act --stage sft --model linear

# Offline functional-use interventions.
.venv-lerobot/bin/python -m interaction_vla.representation_study interventions run \
  --config configs/representation_study/icra_act_macos.yaml \
  --backend act --stage sft

# Fixed paired closed-loop utility.
.venv-lerobot/bin/python -m interaction_vla.representation_study policy evaluate \
  --config configs/representation_study/icra_act_macos.yaml \
  --backend act --stage sft
```

## 测试

```bash
HF_HOME=/tmp/gripper-mujoco-hf-cache \
PYTHONPYCACHEPREFIX=/tmp/gripper-mujoco-lerobot-pycache \
  .venv-lerobot/bin/python -m pytest tests/interaction_vla -q
```

代码位于 `interaction_vla/`，固定实验配置位于 `configs/`，测试位于 `tests/`，本机数据/checkpoint/report 位于 `outputs/`。

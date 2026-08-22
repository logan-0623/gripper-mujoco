# From Imitation to Improvement

本项目面向 ICRA，研究：

> **Does reinforcement learning induce action-relevant interaction structure that supervised imitation fails to capture?**

Interaction Graph 在新主线中是 measurement language，不是强制 policy input。实验严格区分：

- `Accessible`：冻结 latent 能否被 lightweight probe 读出；
- `Useful`：固定闭环 case 上的 nominal/recovery 表现；
- `Plasticity`：固定 online budget 下的 learning-curve AUC；
- `Used`：latent intervention 是否真正改变动作或闭环结果。

ACT 是 controlled mechanism study；SmolVLA 是 modern VLA validation；π0 是可选 external validation。旧 Graph-vs-Flat pipeline 与结果保留且只读。

## 当前结论

旧 ACT 主实验使用 3 个 policy seeds，每个条件共 60 个 paired rollouts：

| Representation | Success | Target drop | Action clipping |
|---|---:|---:|---:|
| Flat | 30.0% | 6.7% | 0.117 |
| Privileged Teacher Graph | 35.0% | 0.0% | 0.096 |
| Predicted Random | 40.0% | 10.0% | 0.114 |
| Predicted Reflect | 41.7% | 0.0% | 0.092 |

这些结果支持的问题是 `representation correctness ≠ control utility`，但尚不能说明 RL 会改善 representation。Recovery RL v2 正式实验用于检验 RL-head 与 RL-representation 的 recovery learning、nominal retention 和 probe trajectory。

## 环境

macOS / Apple Silicon：

```bash
python3.12 -m venv .venv-lerobot
.venv-lerobot/bin/python -m pip install --upgrade pip
.venv-lerobot/bin/python -m pip install -r requirements-lerobot-macos.lock.txt
export HF_HOME=/tmp/gripper-mujoco-hf-cache
```

Linux / RTX 4090：

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

macOS 的 `av`、`cv2`、SDL、FFmpeg duplicate-class 信息通常只是动态库警告；以退出码和最终 JSON 的 `passed` 为准。

## 前置工件

```bash
for artifact in \
  outputs/lerobot/franka_lerobot_act_pilot/meta/info.json \
  outputs/graph_control/graph_v2_pilot/runs/seed_0/flat/checkpoint/config.json \
  outputs/representation_study/icra/sft/act/continued_sft/checkpoint/config.json
do
  test -e "$artifact" && echo "READY   $artifact" || echo "MISSING $artifact"
done
```

若缺少 LeRobotDataset 或 ACT SFT checkpoint，先按对应 `lerobot_act_recovery_*` 和 `graph_v2_act_pilot_*` 配置生成。`outputs/` 不随 Git 上传。

## ACT 正式实验

macOS：

```bash
CONFIG=configs/representation_study/recovery_rl_v2_act_macos.yaml
```

RTX 4090：

```bash
export MUJOCO_GL=egl
export HF_HOME=/tmp/gripper-mujoco-hf-cache
CONFIG=configs/representation_study/recovery_rl_v2_act_linux_cuda.yaml
```

### 1. Foundation gates

下面四步只需成功完成一次。它们校准 recovery 难度，在 PPO/SAC 中选择稳定 backend，验证 privileged Oracle-State residual interface，再选择 anchoring protocol。

```bash
.venv-lerobot/bin/python -m interaction_vla.representation_study recovery-rl calibrate \
  --config "$CONFIG"

.venv-lerobot/bin/python -m interaction_vla.representation_study recovery-rl screen \
  --config "$CONFIG" --resume

.venv-lerobot/bin/python -m interaction_vla.representation_study recovery-rl oracle-gate \
  --config "$CONFIG"

.venv-lerobot/bin/python -m interaction_vla.representation_study recovery-rl anchor-screen \
  --config "$CONFIG" --resume
```

### 2. State Bank v2

采集固定 1,200 states：nominal、perturbation、recovery 各 400；train/validation/test 按 source seed 隔离。

```bash
.venv-lerobot/bin/python -m interaction_vla.representation_study recovery-rl formal state-bank \
  --config "$CONFIG"
```

### 3. 正式 RL 训练

三个条件、三个 seeds；backend 和 anchoring 由 foundation gate 决定，不能在命令行更改。每个 run 固定 20,480 environment steps，在 `0/4096/8192/12288/16384/20480` 保存不可变 snapshot。

```bash
for condition in oracle_state rl_head rl_representation
do
  for seed_index in 0 1 2
  do
    .venv-lerobot/bin/python -m interaction_vla.representation_study recovery-rl formal train \
      --config "$CONFIG" \
      --condition "$condition" \
      --seed-index "$seed_index" \
      --resume
  done
done
```

`oracle_state` 验证 reward 与 residual action interface；`rl_head` 冻结全部 ACT 参数；`rl_representation` 只更新 ACT late-fusion。Policy encoder 与 privileged critic 分离。

### 4. Latent/probe timeline

RL 条件每个 seed 都测。SFT 与 continued-SFT 是固定 control，latent 只测一次，不伪装成三个独立 representation runs。

```bash
for condition in oracle_state rl_head rl_representation
do
  for seed_index in 0 1 2
  do
    .venv-lerobot/bin/python -m interaction_vla.representation_study recovery-rl formal measure \
      --config "$CONFIG" --condition "$condition" --seed-index "$seed_index"
  done
done

for condition in sft continued_sft
do
  .venv-lerobot/bin/python -m interaction_vla.representation_study recovery-rl formal measure \
    --config "$CONFIG" --condition "$condition" --seed-index 0
done
```

六个 checkpoint 都运行 frozen linear probes；step 0 与 20,480 额外运行 shallow MLP。Primary factors 是 geometry、phase、recovery state/type 和 next relation；contact/stable grasp 作为 final secondary metrics。

### 5. Paired closed-loop evaluation

每条 curve 固定使用同一组 30 nominal + 30 recovery development cases。最终结果使用独立 held-out 50 + 50 cases。两个 control 也使用三个 paired evaluation seeds。

```bash
for condition in sft continued_sft oracle_state rl_head rl_representation
do
  for seed_index in 0 1 2
  do
    .venv-lerobot/bin/python -m interaction_vla.representation_study recovery-rl formal evaluate \
      --config "$CONFIG" --condition "$condition" --seed-index "$seed_index"
  done
done
```

### 6. 汇总与 go/no-go

```bash
.venv-lerobot/bin/python -m interaction_vla.representation_study recovery-rl formal report \
  --config "$CONFIG"
```

主要输出：

```text
outputs/representation_study/icra_rl_v2*/
├── gates/{distribution,backend,oracle,anchoring}.json
├── state_bank_v2/
└── formal/
    ├── runs/{oracle_state,rl_head,rl_representation}/seed_*/snapshots/
    ├── controls/{sft,continued_sft}/
    ├── measurements/
    ├── evaluations/
    ├── result_rows.json
    ├── curve_rows.json
    ├── probe_trajectory_rows.json
    ├── pairwise_effects.json
    ├── study_report.json
    └── study_report.md
```

只有 `study_report.json` 的 `complete: true` 表示 required artifacts 全部存在，且 gate、case manifest、normalization、snapshot、probe ledger 与 evaluation 的绑定和内容哈希全部通过。迁移到 SmolVLA 还要求：

- RL-representation 相对 RL-head 的 recovery AUC 至少 2/3 seeds 同方向；
- RL-representation 每个 seed 的 nominal forgetting 不超过 10 个百分点；
- `modern_vla_ready: true`。

## SmolVLA modern VLA validation

ACT go/no-go 通过后再在 4090 上运行 SmolVLA。输入保持标准 LeRobotDataset：agent RGB、wrist RGB、10D end-effector state、language、7D continuous action。

```bash
export MUJOCO_GL=egl
export HF_HOME=/tmp/gripper-mujoco-hf-cache
CONFIG=configs/representation_study/icra_smolvla_linux_cuda.yaml

.venv-lerobot/bin/python -m interaction_vla.representation_study measure run \
  --config "$CONFIG" --backend smolvla --stage pretrained \
  --secondary-probe --closed-loop-intervention

for stage in sft continued_sft
do
  .venv-lerobot/bin/python -m interaction_vla.representation_study sft train \
    --config "$CONFIG" --backend smolvla --stage "$stage" --resume
  .venv-lerobot/bin/python -m interaction_vla.representation_study measure run \
    --config "$CONFIG" --backend smolvla --stage "$stage" \
    --secondary-probe --closed-loop-intervention
done

for stage in rl_head rl_representation
do
  .venv-lerobot/bin/python -m interaction_vla.representation_study rl train \
    --config "$CONFIG" --backend smolvla --stage "$stage" --resume
  .venv-lerobot/bin/python -m interaction_vla.representation_study measure run \
    --config "$CONFIG" --backend smolvla --stage "$stage" \
    --secondary-probe --closed-loop-intervention
done
```

π0 adapter 已保留，但不是 ICRA 主实验的阻塞项。当前单一 language instruction 不能用于语言泛化 claim。

## 测试

```bash
HF_HOME=/tmp/gripper-mujoco-pytest-hf-cache \
PYTHONPYCACHEPREFIX=/tmp/gripper-mujoco-lerobot-pycache \
  .venv-lerobot/bin/python -m pytest -q
```

研究设计见 [ICRA experiment design](docs/superpowers/specs/2026-08-20-icra-interaction-representation-study-design.md)，正式实现计划见 [Recovery RL v2 formal ACT plan](docs/superpowers/plans/2026-08-21-recovery-rl-v2-formal-act.md)，项目状态见 [ccfa.yaml](ccfa.yaml)。

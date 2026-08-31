# Interaction-Centric VLA Representation Study

本项目面向 ICRA，研究同一套物理交互因素在 VLA 训练过程中如何出现、迁移与重组：

```text
Training Stage × Representation Tap × Interaction Factor

Accessible → Functionally Used → Closed-loop Useful
```

Interaction Graph 是 privileged annotation / measurement vocabulary，不是必须输入 policy 的架构。正式因素为 `Entity`、`Geometry`、`Contact`、`StableGrasp`、`Phase`、`NextRelation`。当前主模型是 SmolVLA，标准数据与环境是 LIBERO；ACT/Graph-v2 结果保留为 controlled mechanism evidence。RL 暂停，不属于当前执行阶段。

## 当前证据状态

| 实验 | 状态 | 角色 |
|---|---|---|
| ACT Graph-v2，3 seeds、每条件 60 rollouts | `formal_evidence` | 受控机制证据 |
| Graph diagnostics / Reflect transfer / ACT stagewise | `pilot_complete` | 研究动机与诊断 |
| Recovery RL v2 calibration | `failed_gate` | SFT recovery success 未进入 30–50% 目标区间 |
| LIBERO State Bank | `formal_evidence` | 20 tasks、100 episodes、13,603 states；自动审计与 12 张 timeline 人工审批通过 |
| SmolVLA protocol v2 stages / latent / probes | `pilot_complete` | 80/96 cells 完成；存在跨服务器 latent confound，只作 pilot |
| SmolVLA longitudinal protocol v3 | `formal_evidence` | 8 conditions、4 taps、6 factors；288/384 cells complete，96 cells 因预注册 support gate 不可估计 |
| StableGrasp longitudinal recruitment | `not_run` | specificity/action-sensitivity 代码已实现；尚无实验结果 |

旧 ACT 成功率为 Flat 30.0%、Teacher Graph 35.0%、Predicted Random 40.0%、Predicted Reflect 41.7%。它说明 graph correctness 不能直接当作 control utility，但不回答新的 SmolVLA longitudinal question。

完整科学审计见 [LIBERO–VLA audit](docs/research/2026-08-23-libero-vla-representation-audit.md)，正式 State Bank 的轻量审计证据见 [State Bank package](docs/results/libero_state_bank_formal/README.md)，正式 Protocol-v3 报告见 [Protocol-v3 package](docs/results/libero_smolvla_protocol_v3/README.md)，机器可读状态见 [ccfa.yaml](ccfa.yaml)。

## Linux / RTX 4090 环境

LIBERO simulator 运行在 Linux；macOS 只用于单元测试和不依赖 simulator 的分析。

准备上传并在 4090 服务器从零执行时，优先按 [SERVER_RUNBOOK.md](SERVER_RUNBOOK.md) 操作；其中分开说明了代码发布、两套 LIBERO 数据、smoke gate、正式训练和断点续跑。本 README 保留科学背景与命令速查。

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg libgl1 libegl1

python3.12 -m venv .venv-lerobot
.venv-lerobot/bin/python -m pip install --upgrade pip
.venv-lerobot/bin/python -m pip install -r requirements-lerobot-linux-cuda.txt

export MUJOCO_GL=egl
export HF_HOME=/tmp/gripper-mujoco-hf-cache
```

正式 State Bank 同时需要：

- 官方 `lerobot/libero` LeRobotDataset；
- 原始 LIBERO HDF5 demonstrations（包含 simulator states、actions、`model_file`）。

默认原始数据目录为 `data/libero/raw/`。仅凭 RGB/state/action 不能生成 privileged contact 和 object-pose ground truth，代码会明确拒绝这种降级。

原始 HDF5 使用 LIBERO 官方下载器分别下载 `libero_spatial` 与 `libero_object`，然后让 `data/libero/raw` 指向 LIBERO 注册的 datasets 目录，或在 YAML 中改成对应的仓库内相对路径：

```bash
git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO
.venv-lerobot/bin/python third_party/LIBERO/benchmark_scripts/download_libero_datasets.py \
  --datasets libero_spatial --use-huggingface
.venv-lerobot/bin/python third_party/LIBERO/benchmark_scripts/download_libero_datasets.py \
  --datasets libero_object --use-huggingface
```

正式配置选择官方 `lerobot/libero`。所有训练阶段共享同一数据源和 State Bank，因此视频编码保持固定；配置直接固定 immutable Hub commit，避免服务器运行时依赖 `main` 解析。旧的 `HuggingFaceVLA/libero` 镜像体积约 35GB，并存在 episode 文件索引问题，不再用于本实验。

## Smoke 流程

Smoke 配置固定为每个 suite 3 个 task、每个 task 3 个 held-out episode、每个 episode 16 个 State Bank state；因此 task-group 与 episode-group 都具有非空 train/validation/test，仅用于端到端门禁，不作为正式结果。

```bash
CONFIG=configs/representation_study/libero_smolvla_smoke_linux_cuda.yaml

.venv-lerobot/bin/python -m interaction_vla.representation_study libero audit \
  --config "$CONFIG"

.venv-lerobot/bin/python -m interaction_vla.representation_study libero state-bank collect \
  --config "$CONFIG"

.venv-lerobot/bin/python -m interaction_vla.representation_study libero state-bank inspect \
  --config "$CONFIG"

.venv-lerobot/bin/python -m interaction_vla.representation_study libero state-bank visualize \
  --config "$CONFIG"

# 人工查看 outputs/.../timelines/*.png 后，显式通过语义门禁
.venv-lerobot/bin/python -m interaction_vla.representation_study libero state-bank approve-timelines \
  --config "$CONFIG"

.venv-lerobot/bin/python -m interaction_vla.representation_study libero stages plan \
  --config "$CONFIG"

.venv-lerobot/bin/python -m interaction_vla.representation_study libero stages snapshot \
  --config "$CONFIG"
```

`state-bank collect` 支持 episode shard 缓存和中断续跑；已有相同 scientific binding 的正式 State Bank 不会被覆盖。`visualize` 生成带 global/wrist RGB、Phase、Contact 与 StableGrasp 的 timeline；必须人工检查并执行 `approve-timelines`，probe 才会解锁。

## SmolVLA 分阶段训练

`stages plan` 固定 immutable SmolVLA base revision 和 task-balanced nested subsets：`D25 ⊂ D50 ⊂ D100`，并排除 State Bank episodes。`stages snapshot` 下载并哈希唯一 base checkpoint。三个 SFT stage 各自从同一 base snapshot 独立训练、使用相同 epoch budget，从而把 data fraction 与 sequential continuation 分开。随后先检查 dry run，再启动训练：

```bash
for stage in sft_25 sft_50 sft_100
do
  .venv-lerobot/bin/python -m interaction_vla.representation_study libero stages train \
    --config "$CONFIG" --stage "$stage" --dry-run

  .venv-lerobot/bin/python -m interaction_vla.representation_study libero stages train \
    --config "$CONFIG" --stage "$stage"
done
```

训练被终端中断后，用同一 stage 加 `--resume`；命令只接受 LeRobot 已保存的 `checkpoints/last`，不会从不完整目录猜测状态。

不存在的 checkpoint 始终记录为 `not_run`，不会生成虚假 checkpoint。训练元数据记录 base model、immutable dataset revision、episode subset、seed、data fraction、epochs、steps、config/code hash。

## Latent 与 probe

固定语义 taps：`vision_output`、`multimodal_fusion`、`action_expert_input`、`pre_action`；primary pooling 预注册为 `valid_token_mean`。

```bash
for stage in pretrained sft_25 sft_50 sft_100
do
  .venv-lerobot/bin/python -m interaction_vla.representation_study libero latents extract \
    --config "$CONFIG" --stage "$stage"
done

.venv-lerobot/bin/python -m interaction_vla.representation_study libero probes run \
  --config "$CONFIG"

.venv-lerobot/bin/python -m interaction_vla.representation_study libero probes report \
  --config "$CONFIG"
```

Primary probe 是 linear，shallow MLP 只做 capacity check。主 split 是 task-group，secondary 是 episode-group；严禁 frame-level random split。Contact/StableGrasp 报 AUPRC 和 balanced accuracy，Geometry 报 normalized MAE 与 R²，其余分类因素报 Macro-F1。置信区间按 task/episode cluster bootstrap，不把 frame 当重复实验。

Probe v2 同时报告两类 paired CI：`stage_deltas` 保持
Pretrained→各 SFT stage 的共同参照，`adjacent_stage_deltas` 报告
Pretrained→SFT-25、SFT-25→SFT-50、SFT-50→SFT-100 的相邻变化。
`probes report` 直接复用现有 `.cells`，不会重新拟合 probe；因此已有
Pretrained/SFT-25/SFT-50 时，它可以在 SFT-100 占用 GPU 训练期间使用 CPU
并行汇总。不要在 SFT 训练期间并行执行 latent extraction 或另一个
`probes run`。

现有 protocol v2 只保留为 pilot。正式下一步不重训 SmolVLA，而是复用中间
checkpoint，在同一服务器、同一 runtime 上重提取 8 个条件：

```bash
CONFIG=configs/representation_study/libero_smolvla_linux_cuda.yaml

.venv-lerobot/bin/python -m interaction_vla.representation_study \
  libero longitudinal plan --config "$CONFIG"

for condition in \
  pretrained d25_u16070 d50_u16324 d100_u16617 \
  d50_u32650 d100_u33234 d100_u49851 d100_u66470
do
  .venv-lerobot/bin/python -m interaction_vla.representation_study \
    libero longitudinal extract --config "$CONFIG" \
    --condition "$condition" --batch-size 8
done

.venv-lerobot/bin/python -m interaction_vla.representation_study \
  libero longitudinal inspect --config "$CONFIG"

.venv-lerobot/bin/python -m interaction_vla.representation_study \
  libero longitudinal probes --config "$CONFIG"

.venv-lerobot/bin/python -m interaction_vla.representation_study \
  libero longitudinal probe-report --config "$CONFIG"
```

`inspect` 必须报告 `passed: true` 且 `runtime_fingerprints: 1`。它同时形成约
16k/32k updates 的 matched-update 比较和 D100 optimization trajectory；旧
`latents/` 与 `probes/protocol_v2/` 不会被覆盖。`longitudinal probes` 只读取
State Bank 与本地 latent cache，不加载 SmolVLA、不访问 Hugging Face；正式配置使用
5-fold task/episode group cross-fit，输出到
`protocol_v3/probes/crossfit_v1/`，中断后会复用 scientific binding 相同的压缩 cell。
`probe-report` 只校验并汇总现有结果，不重新拟合。Classification probe 在 CUDA
可用时使用 GPU，Geometry ridge 使用 CPU；probe runtime fingerprint 属于 cell
binding，因此不要把同一输出目录在 CPU 与 CUDA 机器之间混合续跑。

Protocol v3 primary 仅使用 linear probe。`accessible` 表示 cross-fit probe 的
cluster-macro utility 置信区间高于最强的 task/instruction/time shortcut；它不表示
policy functionally used 该因素，更不表示 closed-loop useful。

## Intervention 与 RL 边界

当前只执行 `StableGrasp × action_expert_input × 4 checkpoints`。先运行只读审计和 64-state specificity smoke：

```bash
CONFIG=configs/representation_study/libero_smolvla_linux_cuda.yaml

.venv-lerobot/bin/python -m interaction_vla.representation_study \
  libero interventions audit --config "$CONFIG"

.venv-lerobot/bin/python -m interaction_vla.representation_study \
  libero interventions run --config "$CONFIG" \
  --max-states 64 --batch-size 16 --specificity-only
```

只有 `specificity.json` 的四个 checkpoint 全部 `passed: true`，才运行 formal offline action sensitivity：

```bash
.venv-lerobot/bin/python -m interaction_vla.representation_study \
  libero interventions run --config "$CONFIG" \
  --max-states 1600 --batch-size 16
```

64 与 1600 states 写入不同 profile 目录；相同 profile 可直接重跑续用。动作变化只叫 action-sensitive，只有未来 paired rollout 的任务结果变化才叫 closed-loop useful。Phase 是强制 specificity 对照；若它与 StableGrasp 同时被同等破坏，代码会停止在 specificity gate。

当前不要运行或调优 PPO/SAC，不要从 nominal demonstration 制造 Recovery label。RL 只有在离线 probe、closed-loop intervention、非饱和 perturbation distribution、Oracle-State residual recovery 四个前置 gate 都通过后才恢复。

## 测试

```bash
HF_HOME=/tmp/gripper-mujoco-pytest-hf-cache \
PYTHONPYCACHEPREFIX=/tmp/gripper-mujoco-lerobot-pycache \
  .venv-lerobot/bin/python -m pytest -q \
  tests/interaction_vla/representation_study/libero
```

真实 episode integration test 通过 `LIBERO_INTEGRATION_DEMO` 指向一个原始 HDF5；没有 Linux LIBERO runtime 时会明确 skip，不会伪装通过。旧实验命令与历史说明仍保存在 `docs/` 和现有 configs 中，旧 `outputs/` 不被新流程改写。

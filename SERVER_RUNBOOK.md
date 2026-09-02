# Linux / RTX 4090 Server Runbook

本手册只用于当前正式主线：

```text
LIBERO shared State Bank
→ SmolVLA pretrained / SFT-25 / SFT-50 / SFT-100
→ fixed latent taps
→ linear/MLP probes
→ Stage × Tap × Factor report
```

ACT/Graph-v2 是已经保留的受控机制证据；Recovery RL v2 是 `failed_gate`。原 Protocol-v3 StableGrasp functional-recruitment gate 已失败。当前只运行官方成功 SmolVLA checkpoint 的 positive-control kill test；不要运行 closed-loop intervention、PPO 或 SAC。

## 0. 先理解“上传”包含什么

`git push` 只会上传已经提交的代码和配置，不会上传：

- 未提交或未跟踪的文件；
- `data/` 下的 LIBERO 数据；
- `outputs/` 下的 checkpoint、State Bank 和报告；
- `.venv*`、Hugging Face cache；
- `ICRA_Interaction_Representation_Project_Plan.pdf`，除非你明确决定发布它。

因此，服务器能够开始实验需要同时满足：

1. 本次实现已经提交并推送到 `main`；
2. 服务器已经安装 Linux/CUDA 环境；
3. 服务器具有标准 LeRobotDataset 和匹配的原始 LIBERO HDF5。

## 1. 在本地发布当前实现

先确认当前分支和变更：

```bash
git branch --show-current
git status --short
```

分支应为 `main`。使用下面的白名单提交本次实现，避免把数据、历史输出或用户 PDF 加入 Git：

```bash
git add \
  README.md \
  SERVER_RUNBOOK.md \
  ccfa.yaml \
  requirements-lerobot-linux-cuda.txt \
  configs/representation_study/libero_smolvla_linux_cuda.yaml \
  configs/representation_study/libero_smolvla_smoke_linux_cuda.yaml \
  docs/research/2026-08-23-libero-vla-representation-audit.md \
  docs/superpowers/specs/2026-08-23-libero-interaction-representation-design.md \
  docs/superpowers/plans/2026-08-23-libero-interaction-representation.md \
  docs/superpowers/plans/2026-08-23-libero-server-runbook.md \
  interaction_vla/representation_study/cli.py \
  interaction_vla/representation_study/libero \
  tests/interaction_vla/test_cuda_profiles.py \
  tests/interaction_vla/representation_study/libero

git diff --cached --stat
git commit -m "feat: add LIBERO SmolVLA representation study"
git push origin main
```

提交前检查 `git diff --cached --stat`：其中不应出现 `data/`、`outputs/`、`.venv*` 或 PDF。

## 2. 更新服务器代码

新服务器：

```bash
git clone https://github.com/logan-0623/gripper-mujoco.git
cd gripper-mujoco
git switch main
```

已有 checkout：

```bash
cd /root/gripper-mujoco
git status --short
git pull --ff-only origin main
```

如果 `git pull` 提示本地 tracked 文件会被覆盖，先保留它们，再更新代码：

```bash
git stash push -m "server-local-before-libero"
git pull --ff-only origin main
git stash list
```

不要立刻执行 `git stash pop`。旧服务器上的 generated reports 可能会重新覆盖仓库文件；先用 `git stash show --stat stash@{0}` 检查内容。

确认服务器确实拿到了新入口：

```bash
test -f SERVER_RUNBOOK.md
test -f configs/representation_study/libero_smolvla_smoke_linux_cuda.yaml
test -d interaction_vla/representation_study/libero
git log -1 --oneline
```

## 3. 创建 Linux/CUDA 环境

以下配置面向 Linux x86_64、Python 3.12、CUDA 12.8 和 RTX 4090：

```bash
apt-get update
apt-get install -y ffmpeg libgl1 libegl1 git tmux

python3.12 -m venv .venv-lerobot
.venv-lerobot/bin/python -m pip install --upgrade pip
.venv-lerobot/bin/python -m pip install -r requirements-lerobot-linux-cuda.txt
```

设置当前 shell。AutoDL 建议把 Hugging Face cache 放在数据盘，避免 `/tmp` 随实例释放：

```bash
mkdir -p /root/autodl-tmp/gripper-mujoco-hf-cache
export HF_HOME=/root/autodl-tmp/gripper-mujoco-hf-cache
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export MUJOCO_GL=egl
```

AutoDL 当前无法直连 `huggingface.co`，但已验证 `hf-mirror.com` 可解析同一个固定 commit，因此这里设置 `HF_ENDPOINT`。镜像的 Xet CAS 路径会返回 401，所以同时设置 `HF_HUB_DISABLE_XET=1`，强制使用普通 HTTP 下载。如果服务器可以稳定直连官方 Hub，可省略这两个变量。如果不是 AutoDL，把 `HF_HOME` 改成该服务器的持久数据盘目录。

验证环境：

```bash
nvidia-smi

.venv-lerobot/bin/python -c \
  'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'

.venv-lerobot/bin/python -c \
  'import h5py, lerobot, libero, mujoco; print("LIBERO/LeRobot runtime READY")'
```

Hugging Face 登录不是强制条件，但可以减少限流和断连：

```bash
.venv-lerobot/bin/hf auth login
```

不要把 token 写入仓库、YAML 或终端日志。

## 4. 准备两套匹配数据

正式 State Bank 同时使用：

- 官方 `lerobot/libero`：RGB、robot state、action、language；
- 原始 LIBERO HDF5：simulator state、`model_file`、contact 和 object pose。

仅有 Hugging Face LeRobotDataset 不够生成 privileged annotations。

### 4.1 下载原始 LIBERO HDF5

```bash
git clone --depth 1 \
  https://github.com/Lifelong-Robot-Learning/LIBERO.git \
  third_party/LIBERO

.venv-lerobot/bin/python \
  third_party/LIBERO/benchmark_scripts/download_libero_datasets.py \
  --datasets libero_spatial --use-huggingface

.venv-lerobot/bin/python \
  third_party/LIBERO/benchmark_scripts/download_libero_datasets.py \
  --datasets libero_object --use-huggingface
```

查询当前 LIBERO 注册的数据目录：

```bash
.venv-lerobot/bin/python -c \
  'from libero.libero import get_libero_path; print(get_libero_path("datasets"))'
```

服务器上的官方原始数据盘固定为 `/root/autodl-tmp/libero/datasets`。将它链接到配置所要求的仓库相对路径：

```bash
LIBERO_DATASETS=/root/autodl-tmp/libero/datasets
mkdir -p data/libero
if [ -e data/libero/raw ]
then
  ls -ld data/libero/raw
else
  ln -s "$LIBERO_DATASETS" data/libero/raw
fi
```

检查目录形状：

```bash
test -d data/libero/raw/libero_spatial
test -d data/libero/raw/libero_object
find data/libero/raw -name '*.hdf5' -print -quit
```

如果 `data/libero/raw` 已存在，不要覆盖。先确认它是否已经指向正确的 datasets 根目录。

### 4.2 标准 LeRobotDataset

代码会按照配置从官方 `lerobot/libero` 下载。配置已经固定不可变的 40 位 Hub commit，因此服务器无需先调用 Hub API 解析 mutable `main`。第一次下载仍需要可访问 Hugging Face；后续会复用 `HF_HOME` cache。

旧版配置曾使用约 35GB 的 `HuggingFaceVLA/libero` 图像镜像。在当前服务器上，它会占满数据盘并导致 `DatasetGenerationError`，而且其 episode 文件索引不适合作为分阶段 SFT 子集来源。升级代码后，先确认旧缓存的精确删除范围：

```bash
HF_HOME=/root/autodl-tmp/gripper-mujoco-hf-cache \
  .venv-lerobot/bin/hf cache rm dataset/HuggingFaceVLA/libero \
  --cache-dir /root/autodl-tmp/gripper-mujoco-hf-cache/lerobot/hub \
  --dry-run
```

确认输出只包含 `dataset/HuggingFaceVLA/libero` 后再删除；这不会删除 `/root/autodl-tmp/libero/datasets` 中的原始 HDF5：

```bash
HF_HOME=/root/autodl-tmp/gripper-mujoco-hf-cache \
  .venv-lerobot/bin/hf cache rm dataset/HuggingFaceVLA/libero \
  --cache-dir /root/autodl-tmp/gripper-mujoco-hf-cache/lerobot/hub \
  --yes

df -h /root/autodl-tmp
```

### 4.3 单线程预取 LIBERO simulator assets

AutoDL 镜像对默认 8 路并发下载容易返回 HTTP 429。State Bank replay 前，先把 assets 单线程下载到 LIBERO 默认目录；中断后重复同一命令会续传：

```bash
mkdir -p /root/.cache/libero/assets
.venv-lerobot/bin/hf download lerobot/libero-assets \
  --repo-type dataset \
  --revision 0b3ea86be5fe169d0fd036ae63d1070ec09e90f6 \
  --local-dir /root/.cache/libero/assets \
  --max-workers 1
```

若出现 429 和 `Waiting ... before retry`，让进程等待并自动续传，不要同时启动第二个下载进程。命令正常返回后再运行 State Bank collect；否则不完整的顶层资产目录可能被 LIBERO 误判为已经下载完成。

Raw HDF5 的 `model_file` 保存了数据作者机器上的 `/Users/.../robosuite` 与 `chiliocosm/assets` 绝对路径。Collector 会把这两类路径严格重定位到当前环境并验证每个文件存在；不要在服务器上伪造 `/Users/yifengz/...` 目录或软链接。若仍看到该旧路径，先 `git pull --ff-only origin main`，确认服务器代码包含路径重定位修复。

State Bank annotation 会逐帧恢复官方 recorded state，再执行对应 action 做单步 replay 校验。预注册协议是 `teacher_forced_one_step_qpos`，报告中应同时显示 `"replay_mode": "teacher_forced_one_step"` 和 `"validation_vector": "qpos"`；它不是会累积速度误差的整段 open-loop rollout，也不声称验证 qvel 等价性。

## 5. 必须先跑 smoke gate

建议在 `tmux` 内运行，以免 SSH 断开终止任务：

```bash
tmux new -s libero-smoke
```

进入 tmux 后设置环境和配置：

```bash
cd /root/gripper-mujoco
export HF_HOME=/root/autodl-tmp/gripper-mujoco-hf-cache
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export MUJOCO_GL=egl
CONFIG=configs/representation_study/libero_smolvla_smoke_linux_cuda.yaml
```

### 5.1 State Bank

```bash
.venv-lerobot/bin/python -m interaction_vla.representation_study libero audit \
  --config "$CONFIG"

.venv-lerobot/bin/python -m interaction_vla.representation_study libero state-bank collect \
  --config "$CONFIG"
```

只有 `collect` 返回 `"passed": true` 并生成 `state_bank/manifest.json` 后，才继续运行：

```bash
.venv-lerobot/bin/python -m interaction_vla.representation_study libero state-bank inspect \
  --config "$CONFIG"

.venv-lerobot/bin/python -m interaction_vla.representation_study libero state-bank visualize \
  --config "$CONFIG"
```

`audit` 必须显示 `ready_for_collection: true`；`collect` 和 `inspect` 必须显示 `passed: true`。随后人工查看：

```text
outputs/representation_study/libero_smolvla_smoke/timelines/*.png
```

确认 RGB、Phase、Contact、StableGrasp 与运动过程一致后，才执行：

```bash
.venv-lerobot/bin/python -m interaction_vla.representation_study libero state-bank approve-timelines \
  --config "$CONFIG"
```

如果 timeline 不正确，停止。不要为了继续训练而批准错误标签。

### 5.2 SmolVLA stages

```bash
.venv-lerobot/bin/python -m interaction_vla.representation_study libero stages plan \
  --config "$CONFIG"

.venv-lerobot/bin/python -m interaction_vla.representation_study libero stages snapshot \
  --config "$CONFIG"
```

先检查三个训练命令，不消耗正式训练时间：

```bash
for stage in sft_25 sft_50 sft_100
do
  .venv-lerobot/bin/python -m interaction_vla.representation_study libero stages train \
    --config "$CONFIG" --stage "$stage" --dry-run
done
```

然后完成 smoke 的三个独立 SFT stage：

```bash
for stage in sft_25 sft_50 sft_100
do
  .venv-lerobot/bin/python -m interaction_vla.representation_study libero stages train \
    --config "$CONFIG" --stage "$stage"
done
```

### 5.3 Latents 与 probes

```bash
for stage in pretrained sft_25 sft_50 sft_100
do
  .venv-lerobot/bin/python -m interaction_vla.representation_study libero latents extract \
    --config "$CONFIG" --stage "$stage"

  .venv-lerobot/bin/python -m interaction_vla.representation_study libero latents inspect \
    --config "$CONFIG" --stage "$stage"
done

.venv-lerobot/bin/python -m interaction_vla.representation_study libero probes run \
  --config "$CONFIG"

.venv-lerobot/bin/python -m interaction_vla.representation_study libero probes report \
  --config "$CONFIG"
```

Probe protocol v2 会复用现有 State Bank、checkpoint 和 latent cache，不需要重新训练或重新提取。Smoke 使用一个确定性 seed 验证流程；正式配置使用三个按 `tap/factor/split` 匹配、且跨 training stage 完全相同的 probe seeds。旧的 v1 `probes/report.json` 和 `probes/.cells/` 保留不动；v2 写入 `probes/protocol_v2/`。`stage_deltas` 使用 Pretrained 作为共同参照；`adjacent_stage_deltas` 固定报告 Pretrained→SFT-25、SFT-25→SFT-50、SFT-50→SFT-100。`probes report` 直接读取现有 cell 的 paired payload 并原子补齐相邻 CI，不重新拟合 probe。

### 5.4 正式 longitudinal protocol v3（已完成时只做校验）

这一段只用于正式配置。它复用已训练 checkpoint，不重新训练模型；所有条件
必须在同一服务器顺序提取，不能并行或混用旧 `latents/`：

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

`longitudinal probes` 不需要 Hugging Face 网络，也不会再次加载 SmolVLA；
classification 自动使用 CUDA，Geometry ridge 使用 CPU。它对
8 conditions × 4 taps × 6 factors 分别执行 5-fold task-group 与
task-blocked episode-group cross-fit，并保存 OOF cell；命令中断后直接重跑即可续用
binding 相同的 `protocol_v3/probes/crossfit_v1/cells/*.json.gz`。runtime fingerprint
进入 binding，所以不要在 CPU/CUDA 或不同 Torch/CUDA runtime 间混合续跑；不要
并行启动两个相同 probe runner。

最终 gate 必须满足：`passed: true`、`missing_conditions: []`、
`runtime_gate.runtime_fingerprints: 1`、`state_banks: 1`、`implementations: 1`。
中断后重复同一个 condition 会复用 row cache。不要删除 protocol-v2 结果。

Smoke 的 5.1–5.3 通过后才进入正式配置；5.4 是已有正式 checkpoint 时的
当前续跑入口。

### 5.5 StableGrasp longitudinal recruitment（已完成并冻结）

服务器重启或打开新 shell 后，先恢复数据盘缓存环境。下面的 symbolic link 只需成功创建一次；环境变量需要在每个新 shell 中重新设置：

```bash
export HF_HOME=/root/autodl-tmp/gripper-mujoco-hf-cache
export HF_LEROBOT_HOME="$HF_HOME/lerobot"

DATASET_REV=a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4
DATASET_SNAPSHOT="$HF_LEROBOT_HOME/hub/datasets--lerobot--libero/snapshots/$DATASET_REV"
DATASET_ROOT="$HF_LEROBOT_HOME/lerobot/libero"

mkdir -p "$HF_LEROBOT_HOME/lerobot"
test -e "$DATASET_ROOT" || ln -s "$DATASET_SNAPSHOT" "$DATASET_ROOT"
test -e "$DATASET_ROOT/meta/info.json" && echo "LOCAL DATASET READY"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

必须看到 `LOCAL DATASET READY`。该映射让 `LeRobotDataset` 直接读取固定 revision snapshot；否则即使 34.9 GB 文件已下载，`root=None` 仍可能访问 Hugging Face API，并在断网时抛出 `ConnectError`。不要修改 config 来绕过此问题，否则会改变已有 specificity artifact 的 scientific binding。

先确认 frozen Protocol-v3 与四个 checkpoint 的只读绑定：

```bash
CONFIG=configs/representation_study/libero_smolvla_linux_cuda.yaml

.venv-lerobot/bin/python -m interaction_vla.representation_study \
  libero interventions audit --config "$CONFIG"
```

先做 64-state specificity smoke，不加载四个 policy 做动作推理：

```bash
.venv-lerobot/bin/python -m interaction_vla.representation_study \
  libero interventions run --config "$CONFIG" \
  --max-states 64 --batch-size 32 --specificity-only
```

检查：

```bash
.venv-lerobot/bin/python - <<'PY'
import json
from pathlib import Path
p = Path("outputs/representation_study/libero_smolvla/protocol_v3/recruitment/stable_grasp/n_0064/specificity.json")
j = json.loads(p.read_text())
print({"passed": j["passed"], "conditions": {k: v["passed"] for k, v in j["conditions"].items()}})
PY
```

任一 condition 失败就停止并保留报告。全部通过后才运行 1,600-state formal offline experiment：

```bash
.venv-lerobot/bin/python -m interaction_vla.representation_study \
  libero interventions run --config "$CONFIG" \
  --max-states 1600 --batch-size 32
```

`batch-size 32` 不是吞吐调优项，而是 frozen Protocol-v3 latent runtime binding。程序按原始连续 State Bank batches 恢复相同的语言 padding 和 BF16 kernel context，包括最后不足 32 states 的 batch；因此 formal 进度约为 408 contexts/checkpoint，当前 4080 SUPER 预计总计约 2 小时。首次升级会验证并保留已有 specificity/intervention artifacts，在报告中记录 `action_batch_context_fix_only` binding migration。输出位于 `protocol_v3/recruitment/stable_grasp/n_1600/`。重跑同一 profile 会复用 specificity 和已完成 action report；64-state smoke 不会覆盖 formal profile。该命令仍不运行 LIBERO closed-loop rollout。

完成后直接分析已有 gzip 行缓存；该命令仅使用 CPU，输出 `n_1600/cached_analysis.json`：

```bash
.venv-lerobot/bin/python -m interaction_vla.representation_study \
  libero interventions analyze --config "$CONFIG" --max-states 1600
```

### 5.6 官方 SmolVLA positive-control kill test（当前入口）

Protocol-v3 已冻结，不重跑、不覆盖。这里使用已经通过官方 LIBERO rollout
成功率 floor 的 `lerobot/smolvla_libero` 权重，只回答：成功 policy 中
StableGrasp 是否 accessible、factor-specific、并比 same-norm random 更影响动作。

先执行 5.5 开头的本地 dataset link 与三个 offline 环境变量，然后确认权重和
官方 eval 报告都存在：

```bash
CONFIG=configs/representation_study/libero_smolvla_linux_cuda.yaml
MODEL_DIR=/root/autodl-tmp/models/smolvla_libero_31d453f
EVAL_DIR=/root/autodl-tmp/gripper-mujoco-rollouts/official_smolvla_libero_spatial_task0

test -e "$MODEL_DIR/config.json" && echo "MODEL READY"
test -e "$EVAL_DIR/eval_info.json" && echo "EVAL READY"
```

绑定成功率报告与 checkpoint，然后只提取 `action_expert_input`：

```bash
.venv-lerobot/bin/python -m interaction_vla.representation_study \
  libero positive-control plan --config "$CONFIG" \
  --checkpoint "$MODEL_DIR" --eval-dir "$EVAL_DIR"

.venv-lerobot/bin/python -m interaction_vla.representation_study \
  libero positive-control extract --config "$CONFIG" --batch-size 32

.venv-lerobot/bin/python -m interaction_vla.representation_study \
  libero positive-control probe --config "$CONFIG"
```

若 probe 完整，运行 StableGrasp 的 specificity 和 offline action sensitivity：

```bash
.venv-lerobot/bin/python -m interaction_vla.representation_study \
  libero positive-control intervene --config "$CONFIG" \
  --factor stable_grasp --max-states 1600 --batch-size 32

.venv-lerobot/bin/python -m interaction_vla.representation_study \
  libero positive-control report --config "$CONFIG" \
  --factor stable_grasp
```

读取最终 `decision`：

- `continue_official_longitudinal`：idea 通过 kill test，才训练一条 official-style longitudinal trajectory；
- `replicate_contact_once`：只额外运行下面一次 Contact；
- `failed_specificity`：停止并诊断 factor basis；
- `pivot_interaction_supervised_sft`：不再扩 probe/intervention，转 interaction-supervised SFT。

```bash
.venv-lerobot/bin/python -m interaction_vla.representation_study \
  libero positive-control intervene --config "$CONFIG" \
  --factor contact --max-states 1600 --batch-size 32
```

Contact 命令只能在 StableGrasp 报告要求 `replicate_contact_once` 时运行。
`batch-size 32` 必须与 positive-control latent extraction 一致。输出只写到
`protocol_v4/positive_control/`，不会改动冻结的 Protocol-v3 报告。

## 6. 正式实验

正式实验使用隔离的输出目录，不覆盖 smoke：

```bash
tmux new -s libero-formal
```

进入 tmux 后：

```bash
cd /root/gripper-mujoco
export HF_HOME=/root/autodl-tmp/gripper-mujoco-hf-cache
export MUJOCO_GL=egl
CONFIG=configs/representation_study/libero_smolvla_linux_cuda.yaml
```

按与 smoke 相同的顺序执行正式 State Bank gate：

```bash
.venv-lerobot/bin/python -m interaction_vla.representation_study libero audit \
  --config "$CONFIG"
.venv-lerobot/bin/python -m interaction_vla.representation_study libero state-bank collect \
  --config "$CONFIG"
.venv-lerobot/bin/python -m interaction_vla.representation_study libero state-bank inspect \
  --config "$CONFIG"
.venv-lerobot/bin/python -m interaction_vla.representation_study libero state-bank visualize \
  --config "$CONFIG"
```

人工检查 12 条正式 timeline 后：

```bash
.venv-lerobot/bin/python -m interaction_vla.representation_study libero state-bank approve-timelines \
  --config "$CONFIG"
.venv-lerobot/bin/python -m interaction_vla.representation_study libero stages plan \
  --config "$CONFIG"
.venv-lerobot/bin/python -m interaction_vla.representation_study libero stages snapshot \
  --config "$CONFIG"
```

先逐个 dry run，再分别启动正式训练：

```bash
for stage in sft_25 sft_50 sft_100
do
  .venv-lerobot/bin/python -m interaction_vla.representation_study libero stages train \
    --config "$CONFIG" --stage "$stage" --dry-run
done

.venv-lerobot/bin/python -m interaction_vla.representation_study libero stages train \
  --config "$CONFIG" --stage sft_25
.venv-lerobot/bin/python -m interaction_vla.representation_study libero stages train \
  --config "$CONFIG" --stage sft_50
.venv-lerobot/bin/python -m interaction_vla.representation_study libero stages train \
  --config "$CONFIG" --stage sft_100
```

训练完成后：

```bash
for stage in pretrained sft_25 sft_50 sft_100
do
  .venv-lerobot/bin/python -m interaction_vla.representation_study libero latents extract \
    --config "$CONFIG" --stage "$stage"
  .venv-lerobot/bin/python -m interaction_vla.representation_study libero latents inspect \
    --config "$CONFIG" --stage "$stage"
done

.venv-lerobot/bin/python -m interaction_vla.representation_study libero probes run \
  --config "$CONFIG"
.venv-lerobot/bin/python -m interaction_vla.representation_study libero probes report \
  --config "$CONFIG"
```

若 Pretrained、SFT-25、SFT-50 的 latent 与 Probe v2 cell 已完成，可以在
SFT-100 使用 GPU 训练期间另开一个终端运行上面的 `probes report`。该汇总只
使用 CPU，并会先生成 SFT-25→SFT-50；缺失的 SFT-50→SFT-100 保持显式
`not_available`。当前正式服务器有 16 个 CPU 核，SFT dataloader 保留 4 个
workers 即可。不要同时启动 `latents extract` 或第二个 `probes run`，因为它们
会分别争用 GPU 或重复 probe 计算。

## 7. 中断与续跑

### State Bank

`state-bank collect` 每个 episode 保存一个绑定过的 shard。中断后直接重跑同一条命令；已经完成的 episode 会从 cache 恢复。完整 State Bank 已存在且 scientific binding 相同，也会直接返回现有 audit。

正式采集按固定 hash 顺序为每个 task 寻找 5 个 replay-valid episode。若某个 candidate 超出预注册的 `l2_p95` 或 `max_abs` 容差，它会保留在 `replay/report.json` 的 rejection audit 中，并由同一 task 的下一个 candidate 确定性补位；容差不会被自动放宽。报告中的 `episodes/accepted/acceptance_rate` 描述最终纳入 State Bank 的 episode，`candidate_attempts/candidate_rejected/candidate_acceptance_rate` 描述完整筛选过程。

如果旧版本已完成 100 个 replay、随后以 `94/100` 失败，不要删除 `.episode_shards`。更新代码后直接重跑同一正式命令；兼容迁移只接受该已知旧 pipeline binding，通常会读取原有 100 个 shard，并只运行每个失败 task 所需的补位 candidate：

```bash
CONFIG=configs/representation_study/libero_smolvla_linux_cuda.yaml
.venv-lerobot/bin/python -m interaction_vla.representation_study libero state-bank collect \
  --config "$CONFIG"
```

如果报告 scientific binding 不同，不要删除旧结果。修改配置的 `output_dir`，生成新的隔离实验目录。

### SmolVLA SFT

只有下面目录存在时才能续跑：

```text
outputs/representation_study/<run>/stages/<stage>/run/checkpoints/last/
```

续跑示例：

```bash
.venv-lerobot/bin/python -m interaction_vla.representation_study libero stages train \
  --config "$CONFIG" --stage sft_50 --resume
```

没有 `checkpoints/last` 时，代码会拒绝猜测训练状态。保留失败目录用于诊断，并为重跑使用新的 `output_dir`。

### Latent 与 probe

Latent 以 `state_id × checkpoint × tap` 分片缓存；同一 binding 下可直接重跑 `latents extract`。Probe cell 同样有严格 binding cache。若 checkpoint、State Bank、配置或实现发生改变，旧 cache 会被拒绝，不应手工混用。

如果曾在 `1f9a7cc` 之前运行正式 SmolVLA latent extraction，并看到 policy 期待 `image/image2`、preprocessor 却输出 `camera1/camera2` 的错误，那么旧 latent rows 使用了错误 camera binding。它们不能继续复用，但 SFT checkpoint 本身不受影响，无需重训。

更新代码后，先把旧 cache 移到可恢复备份；不要删除：

```bash
cd /root/gripper-mujoco
git pull --ff-only origin main

LATENT_ROOT=outputs/representation_study/libero_smolvla/latents
LATENT_BACKUP=outputs/representation_study/libero_smolvla/latents_invalid_camera_binding_20260825
test -e "$LATENT_ROOT"
test ! -e "$LATENT_BACKUP"
mv "$LATENT_ROOT" "$LATENT_BACKUP"
```

然后只重新抽取当前已经存在的正式 stages。当前至少运行 `pretrained` 和 `sft_25`：

```bash
CONFIG=configs/representation_study/libero_smolvla_linux_cuda.yaml
for stage in pretrained sft_25
do
  .venv-lerobot/bin/python -m interaction_vla.representation_study \
    libero latents extract --config "$CONFIG" --stage "$stage"
  .venv-lerobot/bin/python -m interaction_vla.representation_study \
    libero latents inspect --config "$CONFIG" --stage "$stage"
done
```

新 loader 会保留 checkpoint 的 `camera1/camera2/camera3` 输入契约，并用训练时相同的 rename map 处理 LIBERO 的 `image/image2`。新的 latent implementation binding 同时覆盖 loader 和 rename contract，因此旧目录即使移回原位也会被明确拒绝，而不会静默混入新结果。

如果 extraction 已经运行到 `1700/1701`，随后报告 `SmolVLA semantic tap metadata changed across batches`，这是 `13,603` 个 states 的最后一个 3-state batch 暴露出的旧 metadata bug。`ec4eb66` 已让 shape metadata 排除可变 batch 维。更新代码后，把这次未完成目录单独保留，再重新运行上面的 extraction；checkpoint 和 State Bank 都不需要重建：

```bash
git pull --ff-only origin main

LATENT_ROOT=outputs/representation_study/libero_smolvla/latents
LATENT_BACKUP=outputs/representation_study/libero_smolvla/latents_incomplete_batch_metadata_20260825
test -e "$LATENT_ROOT"
test ! -e "$LATENT_BACKUP"
mv "$LATENT_ROOT" "$LATENT_BACKUP"
```

## 8. 关键产物

Smoke 根目录：

```text
outputs/representation_study/libero_smolvla_smoke/
```

正式根目录：

```text
outputs/representation_study/libero_smolvla/
```

重点检查：

```text
state_bank/manifest.json
state_bank/audit/report.json
timelines/report.json
stages/pretrained/manifest.json
stages/sft_25/manifest.json
stages/sft_50/manifest.json
stages/sft_100/manifest.json
latents/<stage>/report.json
probes/protocol_v2/report.json
protocol_v3/conditions/manifest.json
protocol_v3/latent_gate/report.json
protocol_v3/probes/crossfit_v1/folds.json
protocol_v3/probes/crossfit_v1/report.json
```

快速检查：

```bash
find outputs/representation_study/libero_smolvla \
  \( -name 'report.json' -o -name 'manifest.json' \) -print | sort
```

这些实验只有实际报告通过后才能从 `implementation_only` / `not_started` 升级为结果。代码存在、dry run 成功或 checkpoint 文件存在，都不等于科学实验完成。

## 9. 当前停止线

Protocol v2 的 `probes report` 只保留为 pilot。正式结论读取
`protocol_v3/probes/crossfit_v1/report.json`，先分析 `Condition × Tap × Factor`：

- 哪些信息在 pretrained 已可解码；
- 哪些因素随 SFT 增强、减弱或向 action-proximal tap 移动；
- Contact、StableGrasp、NextRelation 是否仍然薄弱；
- task-group 与 task-blocked episode-group 是否一致；
- `paired_deltas` 的 `destination_minus_reference`、`improvement` 与 paired CI；区间跨零只能记为 inconclusive；
- `identical_latent_sanity.passed` 是否为 `true`。相同 latent 若产生不同 probe 结果会直接失败，不能解释为训练阶段变化。

Protocol v3 的 primary 只使用 linear probe；MLP capacity check 仍属于 protocol v2 pilot。
`primary_metric` 是 matched probe seeds 的 cluster-macro 均值，`probe_metric_std` 只描述 probe 优化敏感性，不是机器人任务的独立重复。条件变化应读取 paired delta，不能通过两个独立 accessibility 区间是否重叠来推断。

分类指标始终使用训练分区的完整类别全集。二元事件的 bootstrap 若有效重采样比例低于 `minimum_bootstrap_valid_rate`，该区间和对应 accessibility/stage-delta gate 会失败，不能从剩余条件样本推断。阶段报告分别给出 `delta_low/high` 和按指标方向转换后的 `improvement_low/high`；Geometry 不要把 raw delta 区间直接画成 improvement 区间。

在 StableGrasp specificity 与 offline action-sensitivity gate 得到可解释结果前，不运行：

```text
libero evaluate paired
recovery RL / PPO / SAC
```

这条停止线用于避免把“可解码”“被 policy 使用”和“对闭环成功有用”混成同一个结论。

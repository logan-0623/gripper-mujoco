# Linux / RTX 4090 Server Runbook

本手册只用于当前正式主线：

```text
LIBERO shared State Bank
→ SmolVLA pretrained / SFT-25 / SFT-50 / SFT-100
→ fixed latent taps
→ linear/MLP probes
→ Stage × Tap × Factor report
```

ACT/Graph-v2 是已经保留的受控机制证据；Recovery RL v2 是 `failed_gate`。当前不要运行 intervention、PPO 或 SAC。

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

只有以上流程全部通过，才进入正式配置。

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
probes/report.json
```

快速检查：

```bash
find outputs/representation_study/libero_smolvla \
  \( -name 'report.json' -o -name 'manifest.json' \) -print | sort
```

这些实验只有实际报告通过后才能从 `implementation_only` / `not_started` 升级为结果。代码存在、dry run 成功或 checkpoint 文件存在，都不等于科学实验完成。

## 9. 当前停止线

完成 `probes report` 后先分析 `Stage × Tap × Factor`：

- 哪些信息在 pretrained 已可解码；
- 哪些因素随 SFT 增强、减弱或向 action-proximal tap 移动；
- Contact、StableGrasp、NextRelation 是否仍然薄弱；
- 线性 probe 与 MLP capacity check 是否一致。

在 probe gate 得到可解释结果前，不运行：

```text
libero interventions run
libero evaluate paired
recovery RL / PPO / SAC
```

这条停止线用于避免把“可解码”“被 policy 使用”和“对闭环成功有用”混成同一个结论。

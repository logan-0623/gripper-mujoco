# Linux / CUDA 服务器手册

在 AutoDL 一类 Linux + RTX 4090 服务器上，按顺序完成：更新代码、建环境、准备数据、建 StateBank，再跑实验。研究问题与已有结论见 [docs/research-questions.md](docs/research-questions.md) 和 [docs/findings.md](docs/findings.md)。

`git push` 只上传已提交的代码与配置；`data/`、`outputs/`、`.venv*`、Hugging Face cache 和 checkpoint 都不会上传，必须在服务器上单独准备。

## 1. 更新服务器代码

新服务器：

```bash
git clone https://github.com/logan-0623/gripper-mujoco.git
cd gripper-mujoco
git switch main   # 或当前研究分支
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

## 2. 创建 Linux/CUDA 环境

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

## 3. 准备两套匹配数据

正式 State Bank 同时使用：

- 官方 `lerobot/libero`：RGB、robot state、action、language；
- 原始 LIBERO HDF5：simulator state、`model_file`、contact 和 object pose。

仅有 Hugging Face LeRobotDataset 不够生成 privileged annotations。

### 3.1 下载原始 LIBERO HDF5

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

### 3.2 标准 LeRobotDataset

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

### 3.3 单线程预取 LIBERO simulator assets

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

## 4. State Bank（所有离线分析共用的状态集合）

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


## 5. 训练谱系与能力时间轴：`run.sh`

`run.sh` 复现自训练 SmolVLA 谱系（seed=1000，5k–25k checkpoint）及其上的能力时间轴、flow trace 和离线动作作用。输入全部以只读路径传入：

```bash
BASE_CHECKPOINT=/data/smolvla_base METADATA=/data/smolvlm \
DATASET_ROOT=/data/libero STATE_BANK=/data/state_bank OUTPUT_ROOT=/data/run \
nohup ./run.sh all > /data/run.log 2>&1 &
```

可单独运行某个阶段：`prepare`、`train`、`timeline`、`traces`、`discover`、`action_effects`、`closed_loop`、`confirmation`。已有谱系位于 `/root/autodl-tmp/smolvla-official-reproduction-v2/`，`train` 会检测并复用。

## 6. 抓取结果分叉实验

在同一初态、同一动作前缀下，一支照常执行，另一支在稳定抓起后强制张开夹爪 8 步。每个 checkpoint 输出 `paired_summary.json` 和逐步物理事件：

```bash
.venv-lerobot/bin/python scripts/feedback_branch_pilot.py \
  --checkpoint "$CKPT/025000/pretrained_model" --output "$OUT/step_025000" \
  --task 0 --task 1 --episodes 10
```

## 7. 中断与续跑

`run.sh` 的各阶段在开始前检查已有输出（例如已完成的 25k 谱系、lineage 文件）并复用；中断后重跑同一阶段即可。续跑前不要手工删改输出目录里的 manifest 或哈希文件。

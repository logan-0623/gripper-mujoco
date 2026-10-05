# VLA 在训练中学会了什么

本项目研究 Vision-Language-Action 策略（SmolVLA，LIBERO 仿真）在训练过程中如何获得操作能力：能力存在于哪个模块，它是"知道更多"还是"做得更稳"，以及这些诊断能否转化为更快、更好的训练。

- **研究问题：** [docs/research-questions.md](docs/research-questions.md)。这是当前主线，包含五个可以给出 yes/no 结论的问题和执行顺序。
- **已确立的事实：** [docs/findings.md](docs/findings.md)。证据台账，每条都链接到原始结果。
- **服务器操作：** [SERVER_RUNBOOK.md](SERVER_RUNBOOK.md)。

一句话现状：**同一训练谱系在 5k→15k 步内成功率从 12% 涨到 84%。** 此前"内部信息不变、语义干预无效"的结论经审计不成立：测的是输入变量，干预点也不对，见 [审计](docs/notes/probe-intervention-audit-20261005.md)。下一步是用正确的测量重新检验"知道先于会做"，并定位能力形成的位置。

## 代码结构

```text
interaction_vla/
  representation_study/
    libero/            SmolVLA × LIBERO 主线
      runtime.py         LIBERO 离屏仿真、回放
      capability_events.py, annotation.py
                         逐步物理事件（接触、稳定抓取、抬起、掉落、恢复）
      acquisition.py     训练谱系、能力时间轴、闭环评测、确认实验（run.sh 的入口）
      flow_trace.py      各 checkpoint 在相同状态与噪声下的 flow 采样轨迹和激活
      flow_diff.py, checkpoint_readouts.py
                         跨 checkpoint 的差异、CKA 与读出
      candidate_controls.py, contact_subspace.py, flow_intervention_eval.py, phase_edit.py
                         方向干预与匹配随机对照
      context_dependence.py, stage_context_dependence.py
                         接触与非接触状态下的情境差
      paired_object_eval.py, joint_evidence.py, longitudinal_contact_summary.py
                         配对评测与证据汇总
      collector.py, sources.py, alignment.py, replay.py, audit.py, state_bank.py, ...
                         StateBank 构建与审计
    backends/lerobot.py  SmolVLA / pi0 加载、激活捕获、前向干预
    taps/, schemas/, state_bank/
                         tap 注册表、阶段 manifest、StateBank 数据结构
  smolvla_mujoco.py      在自定义 Franka MuJoCo 场景中运行 SmolVLA（运行接口，不是评测器）
  physics_env.py, franka_controller.py, contact_physics.py, ...
                         Franka MuJoCo 场景和控制器
scripts/
  feedback_branch_pilot.py     强制掉落的配对分叉实验
  run_*.sh / run_*.py          服务器上的阶段性批处理
  smolvla_gui.py, run_smolvla_mujoco.py
                               本地 MuJoCo 交互运行
run.sh                   训练谱系与能力时间轴的一键复现
configs/representation_study/  LIBERO StateBank 配置
docs/results/            各实验的轻量证据归档（JSON、CSV、README），不含权重和视频
docs/notes/              仍有参考价值的分析笔记
```

StateBank 相关命令（详见 runbook 第 4 节）：

```bash
.venv-lerobot/bin/python -m interaction_vla.representation_study libero audit --config CONFIG
.venv-lerobot/bin/python -m interaction_vla.representation_study libero state-bank {collect|inspect|visualize|approve-timelines} --config CONFIG
```

## 本地环境（macOS）

只使用项目内的 `.venv-lerobot`，项目根目录为当前目录：

```bash
bash scripts/python.sh scripts/check_environment.py   # 检查入口与依赖
.venv-lerobot/bin/python -m pytest -q tests            # 单元测试
```

当前锁定环境：Python 3.12.14、PyTorch 2.10.0、LeRobot 0.6.1、Transformers 5.5.4、MuJoCo 3.3.4。完整版本见 [requirements-action-atlas-macos.lock.txt](requirements-action-atlas-macos.lock.txt)。需要读取视频时使用 `scripts/python.sh`，它会为当前进程选择 Homebrew `ffmpeg@8`。

从零重建（仅在没有可用环境时执行）：

```bash
brew install uv ffmpeg@8
uv venv --python 3.12.14 .venv-lerobot
uv pip sync --python .venv-lerobot/bin/python requirements-action-atlas-macos.lock.txt
```

合并 lock 包含 `research/action-atlas/` 的 editable 安装，需要先取得固定版本：`git clone https://github.com/CWRU-AISM/action-atlas.git research/action-atlas && git -C research/action-atlas checkout --detach b8b0db331df18fc30a3fd92c45ec721d35d3ee52`（`run.sh prepare` 会自动执行同样的步骤）。Linux/CUDA 环境见 runbook。

### 模型与数据（固定版本）

| 本机路径 | 内容 | 来源 |
| --- | --- | --- |
| `outputs/pretrained/smolvla_libero/` | 官方 SmolVLA LIBERO 策略 | `HuggingFaceVLA/smolvla_libero` @ `6721902b` |
| `outputs/pretrained/SmolVLM2-500M-Instruct/` | tokenizer 与配置 | `HuggingFaceTB/SmolVLM2-500M-Instruct` @ `7b375e1b` |
| `outputs/datasets/libero/` | LeRobot 格式的 LIBERO 数据 | `lerobot/libero` @ `a1aaacb7` |
| `outputs/representation_study/libero_smolvla/state_bank/` | 本项目的状态、标签与划分 | 由 StateBank 流程生成，不随 Hub 下载 |

自训练的 5k–25k 谱系和大型原始结果只在服务器 `/root/autodl-tmp/smolvla-official-reproduction-v2/` 上。`docs/results/` 只保存可以进 Git 的摘要。

## 历史

2026-10-05 进行了一次重构：删除了 ACT 与交互图（graph）路线、残差 RL、早期 SFT 分阶段协议、交叉拟合探针、SAE 发现、官方 positive control 等已经结束的实验代码，以及对应的配置、测试和过程文档（约 7.6 万行）。这些实验的结论保留在 [docs/findings.md](docs/findings.md) 和 `docs/results/` 中。完整的旧代码可以随时取回：

```bash
git show pre-restructure:path/to/file      # 查看单个旧文件
git checkout pre-restructure -- path/to/dir  # 恢复整个目录
```

# Interaction-Centric VLA Representation Study

研究冻结的 Vision-Language-Action（VLA）策略内部，物理交互信息如何被表示、预测，以及是否参与控制。

> **当前问题：在一个有基本任务能力的冻结 VLA 中，可解释特征能否组成跨任务的预测性交互状态？这种状态与策略实际使用的信息有多大重合？**

当前主线是 **LIBERO + 官方 SmolVLA + Action Atlas**：复用既有物理标签与任务划分，保存逐 token 激活和策略动作计划，先比较 Hidden/PCA 与动作、时间对照，再考虑 SAE/RET 和机制干预。不要求 RET 胜过 SAE，也不以成功率提升作为唯一有价值的结果。

**状态更新：2026-09-09。代码交付后由用户运行验证；当前没有启动全量 token 提取或真实 Hidden/PCA 比较。SmolVLA 参数冻结，不启动新 SFT、RL、world model 或多模型比较。**

## 导航

- [当前完成到哪里](#当前完成到哪里)
- [研究问题与测量边界](#研究问题与测量边界)
- [代码分工与项目结构](#代码分工与项目结构)
- [环境与准备条件](#环境与准备条件)
- [运行当前实验](#运行当前实验)
- [数据协议与公平比较](#数据协议与公平比较)
- [查看结果与判断完成](#查看结果与判断完成)
- [历史实验与已有发现](#历史实验与已有发现)
- [已知限制与常见问题](#已知限制与常见问题)
- [后续阶段与复现边界](#后续阶段与复现边界)

详细协议、运行记录和参数解释见 [research/predictive-states.md](research/predictive-states.md)。上游源码与依赖接入记录见 [research/README.md](research/README.md)；其中早期状态按其记录日期理解，不覆盖本页的当前状态。

## 当前完成到哪里

必须区分“代码实现”“小样例验证”“真实数据结果”和“闭环证据”。

| 工作 | 已有证据 | 尚不能说明什么 |
| --- | --- | --- |
| 唯一本地环境 | `.venv-lerobot`；模型导入、基础物理、视频读写有历史通过记录 | 不等于 Linux/CUDA 或完整 LIBERO 仿真已验证 |
| 官方模型与数据 | SmolVLA 权重约 1.22 GB；固定 revision 的 LeRobot LIBERO 数据约 1.94 GB 已下载 | 数据下载不等于实验完成 |
| Shared StateBank | 20 个任务、100 个 episode、13,603 个状态；既有回放与标注审计归档 | 是测量基础，不是模型能力证明 |
| 真实观察输入与零改动重放 | CPU 上 4 帧真实输入通过；激活 `[4,480]`；零改动重放动作差为 0 | 不是仿真轨迹重放或任务成功率 |
| Token/action 噪声 pilot | 中间版本在 MPS 上完成 4 个训练任务、64 状态、3 个噪声 seed 的测量 | 后续源码有补充，不能称最终版本已通过验证 |
| 完整 token/action 缓存 | 实现了分片保存、绑定校验与续跑入口 | **全量 13,603 状态提取尚未运行** |
| Hidden/PCA 与控制读出 | 实现了比较入口及测试代码 | **真实全量比较尚未运行** |
| 官方 SAE/RET | 已有合成缓存训练与集成检查 | 不是机器人真实数据比较或原论文 benchmark 复现 |
| 特征干预、闭环、RL | 保留旧协议和条件性入口 | 当前新路线未执行；RL 仍冻结 |

最近一次完整回归记录是 **1,035 passed、1 skipped**，对应 token 缓存最终改动**之前**的代码。随后新增逻辑曾有 3 项测试通过；最终补充后的源码只做了语法与差异检查，需按下方命令重新验证。真实 LIBERO 原始 HDF5 集成仍未通过本机测试覆盖。

已有轻量科学证据在 [docs/results/](docs/results/)；本机生成结果在 `outputs/`，后者被 Git 忽略。不要将“仓库中存在历史报告”理解成“当前版本已经重新执行这些实验”。

## 研究问题与测量边界

### 四个问题分别回答

| 测量 | 输入与证据 | 合理的解释范围 |
| --- | --- | --- |
| Encoding：能读出什么？ | 当前 hidden/feature → 当前物理标签 | 指定读出器能够访问这些信息 |
| Future prediction：能预测什么？ | 截止当前的表示历史 → 未来物理标签 | 在该数据分布和信息条件下具有预测性 |
| Action sensitivity：是否影响策略输出？ | 匹配条件下的内部干预 → 动作变化 | 被干预计算对动作有影响；语义特异性还需控制 |
| Closed-loop dependence / utility：是否影响交互与任务？ | 配对初始状态、噪声与环境中的干预 rollout | 是否改变接触、夹持、释放、掉落及成功率 |

**能读出 ≠ 能预测变化 ≠ 策略使用了该语义 ≠ 对任务成功有益。** 读出失败也不能证明信息不存在；它可能超出当前读出器的能力。接触和稳定抓取只是有限的物理因素，不能据此声称发现完整 world model。

当前最需要区分的是：预测信号来自物理交互状态、任务进度，还是当前动作计划。因此，时间、robot state、策略动作计划和未来真实演示动作是不同用途的对照，不能混成一个排行榜。

### Interaction Graph 的角色

早期 ACT 实验将 Graph 作为策略输入；当前 VLA 主线将其作为 **privileged measurement vocabulary（特权测量语言）**。模拟器信息用于生成标签，不作为当前 SmolVLA 的额外输入。

```text
模拟器记录 ──→ StateBank 物理标签与固定任务划分 ───────────┐
                                                       │ 对齐评估
双相机 RGB + robot state + language                     │
                ↓                                      │
           冻结 SmolVLA                                 │
                ├─→ 每帧 token 激活 ─→ Hidden/PCA ───────┤
                └─→ 当前预测 action chunk ─→ 动作对照 ────┘

后续独立阶段：SAE/RET → 匹配特征干预 → 配对闭环
```

既有标签体系包括 Entity、Geometry、Contact、StableGrasp、Phase、NextRelation。当前预测实验只读出 **Contact 与 StableGrasp**：前者来自物理接触，后者结合双侧手指接触、过去短窗口内的相对位姿稳定及共同运动/离开支撑等条件，不以“夹爪闭合”直接代替稳定抓取。缺失标签保持缺失，不填零。普通 demonstration 不被伪造为 recovery 数据。

## 代码分工与项目结构

优先复用官方实现。本仓库的新增代码负责数据绑定、可恢复缓存、共同读出协议与完整性检查，不重写 SmolVLA、SAE 或 RET 的核心算法。

| 组件 | 用途 | 版本与差异 |
| --- | --- | --- |
| [LeRobot](https://github.com/huggingface/lerobot) | SmolVLA、数据读取、视频解码入口及模型前后处理 | 当前合并环境锁定 `lerobot==0.6.1` |
| [Action Atlas](https://github.com/CWRU-AISM/action-atlas) | SmolVLA 层访问、capture/injection hooks、官方 TopK SAE | `b8b0db331df18fc30a3fd92c45ec721d35d3ee52`；Apache-2.0 |
| [RET](https://github.com/ustaomeroglu/RET) | 预测性表示 encoder、predictor、EMA、training loop | `1b0d9b2ee0281d50f875a30a4d066cbb9df883b1`；MIT；[cached 模式补丁](research/ret-cached-robot.patch) |
| 本项目 StateBank | 物理标签、原始来源、episode/task 分组和回放参考 | 冻结既有 13,603 状态与划分 |
| 本项目诊断入口 | token/action 分片缓存、噪声检查、Hidden/PCA 与控制读出 | 当前需要用户验证最终版本 |

Action Atlas 的 LeRobot submodule 与已安装的 LeRobot 0.6.1 并非同一版本；不要直接初始化全部 submodule 后继续声称是相同环境。RET 的机器人时间序列输入属于适配，不是原论文语言序列 benchmark 的原样复现。Event-SAE 尚未接入，不能把普通 TopK SAE 称为 Event-SAE baseline。

```text
interaction_vla/
  representation_study/libero/
    state_bank.py / alignment.py / annotation.py / splits.py
    smolvla_smoke.py     # 官方离线冻结加载、真实输入与零改动检查
    token_cache.py       # pilot / extract / evaluate 命令入口
    token_readouts.py    # Hidden/PCA 与策略动作、本体状态对照
    predictive_states.py # 时序审计、共同读出、官方 SAE/RET 适配
    latents.py           # 既有缓存、观察绑定、逐状态噪声工具
  ...                   # 保留 ACT、Graph、旧探针与干预实现
research/
  predictive-states.md  # 当前协议、逐步命令、实测记录
  README.md             # 上游接入与环境记录
  action-atlas/         # 外部固定版本 checkout，不进入父仓库
  ret/                  # 外部固定版本 checkout，不进入父仓库
scripts/
  python.sh             # 唯一环境与进程内 FFmpeg 路径
  check_environment.py  # 导入、基础物理、合成视频检查
tests/                  # 工程与协议测试
docs/results/           # 轻量历史证据包
outputs/                # 本机数据、模型、缓存与结果，不进入 Git
ccfa.yaml               # 历史科学注册表与执行限制
SERVER_RUNBOOK.md       # 历史 Linux/CUDA 主线手册，不是当前一键启动脚本
```

## 环境与准备条件

### 已有工作环境：直接复用

本机只使用项目内 **`.venv-lerobot`**，不需要另外建立 SAE、RET 或测试环境。以下命令均假定当前目录为项目根目录。

```bash
# 检查环境入口与包依赖，不训练模型。
bash scripts/python.sh scripts/check_environment.py
uv pip check --python .venv-lerobot/bin/python
```

现有锁定环境包含 Python 3.12.14、PyTorch 2.10.0、LeRobot 0.6.1、Transformers 5.5.4、NumPy 2.2.6、MuJoCo 3.3.4 和 TorchCodec 0.10.0；完整包版本以 [requirements-action-atlas-macos.lock.txt](requirements-action-atlas-macos.lock.txt) 为准。

`scripts/python.sh` 为当前进程选择 Homebrew `ffmpeg@8`，不修改系统默认 FFmpeg 或 shell 配置。现有 TorchCodec 组合不能直接使用本机 FFmpeg 9；需要读取视频时使用这个入口。

### 从零恢复环境：仅在没有可用环境时执行

先按照 [上游接入说明](research/README.md) 获取固定版本的 `research/action-atlas/`；合并 lock 包含该目录的 editable 安装项，目录缺失时不能完成安装。

```bash
brew install uv ffmpeg@8
# 仅首次创建；已有可用 .venv-lerobot 时跳过。
uv venv --python 3.12.14 .venv-lerobot
uv pip sync --python .venv-lerobot/bin/python requirements-action-atlas-macos.lock.txt
```

`uv pip sync` 会删除该环境中未列入 lock 的包。不要日常重复 sync，更不要安装 Action Atlas 后单独 sync 旧 `requirements-lerobot-macos.lock.txt`，否则会移除新增依赖。macOS lock 不用于 Linux/CUDA，虚拟环境也不能直接跨机器复制。

### 模型、数据与 StateBank 不是一回事

| 本机路径 | 内容 | 固定来源 |
| --- | --- | --- |
| `outputs/pretrained/smolvla_libero/` | 完整策略、配置、归一化权重，约 1.22 GB | [HuggingFaceVLA/smolvla_libero](https://huggingface.co/HuggingFaceVLA/smolvla_libero)，revision `6721902bc4d61e50a3bfdb11dfb4cb626f05d102` |
| `outputs/pretrained/SmolVLM2-500M-Instruct/` | 11 个 tokenizer/配置文件，不重复下载 backbone 权重 | checkpoint 指定的 `HuggingFaceTB/SmolVLM2-500M-Instruct`，revision `7b375e1b73b11138ff12fe22c8f2822d8fe03467` |
| `outputs/datasets/libero/` | LeRobot 格式观察、动作、元数据和视频，约 1.94 GB | [lerobot/libero](https://huggingface.co/datasets/lerobot/libero)，revision `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4` |
| `outputs/representation_study/libero_smolvla/state_bank/` | 本项目标签、状态身份、划分与审计绑定 | 既有正式 StateBank；不是 Hub 数据集自带目录 |

**仅 `git clone` 或下载 LIBERO 不会获得完整 StateBank。** 新机器需要可信来源的 StateBank 原始文件，或按其原始回放/标注协议重建；`docs/results/` 里的轻量 manifest 不能代替 `records.jsonl` 与 split 文件。原始 LIBERO HDF5、仿真 assets 也不等于 LeRobot 视频数据：当前离线读出不重新运行模拟器，但真实回放与闭环需要另外准备。

如需恢复缺失的公开资产，以下命令需要联网；已有文件不要为了日常运行反复下载。在设置离线变量之前执行，或使用一个未设置这些变量的终端。

<details>
<summary>恢复固定版本的公开模型、配置和观察数据</summary>

```bash
.venv-lerobot/bin/hf download HuggingFaceVLA/smolvla_libero \
  --revision 6721902bc4d61e50a3bfdb11dfb4cb626f05d102 \
  --local-dir outputs/pretrained/smolvla_libero

.venv-lerobot/bin/hf download HuggingFaceTB/SmolVLM2-500M-Instruct \
  --revision 7b375e1b73b11138ff12fe22c8f2822d8fe03467 \
  added_tokens.json chat_template.json config.json generation_config.json \
  preprocessor_config.json processor_config.json special_tokens_map.json \
  tokenizer.json tokenizer_config.json vocab.json merges.txt \
  --local-dir outputs/pretrained/SmolVLM2-500M-Instruct

.venv-lerobot/bin/hf download lerobot/libero --repo-type dataset \
  --revision a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4 \
  --local-dir outputs/datasets/libero
```

策略主权重已记录的官方 SHA-256 为 `71d9563c8295284acba8fc2d5c19de000d6fe9ba58a406832af7ef3d221ed52f`。当前本机验证记录在 `outputs/pretrained/smolvla_libero-verification.json`；这个记录文件本身不会随 Git clone 或 Hub 下载自动出现。

</details>

## 运行当前实验

### 第一步：验证最终代码，再做小样例

以下命令由用户手动执行，不会因阅读 README 自动启动。每一步成功后再进入下一步，不要一次粘贴整段长流程后忽略中途失败。

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_CACHE="$PWD/outputs/datasets/.hf-cache"

bash scripts/python.sh -m pytest \
  tests/interaction_vla/representation_study/libero/test_token_cache.py \
  tests/interaction_vla/representation_study/libero/test_latents.py \
  -q -W error::RuntimeWarning
```

然后运行真实输入 pilot：

```bash
bash scripts/python.sh -m interaction_vla.representation_study.libero.token_cache pilot \
  --device mps --batch-size 4 \
  --output outputs/predictive_states/token_noise_check
```

Pilot 固定选择 4 个训练任务，每个任务的首个 episode 中均匀取 16 个状态，合计 64 状态；分别使用噪声 seed 0/1/2。噪声由 checkpoint 内容、state ID 和 noise seed 确定，在 CPU 上生成后传入策略；所有运行使用同样的观察，不根据结果挑选 seed。

每帧保存最后一次去噪调用的 `[50,480]` token 激活，以及当前观察产生的 `[50,7]` 策略动作块。每次运行还检查首批的重复 token 与官方 `select_action` 一致性。**token 位置是动作块位置，不是 50 个已经观测到的未来状态。**

### 第二步：完整 StateBank 提取

```bash
bash scripts/python.sh -m interaction_vla.representation_study.libero.token_cache extract \
  --device mps --batch-size 4 \
  --output outputs/predictive_states/smolvla_tokens_mps
```

目标是完整覆盖 **13,603 个 state ID**，使用固定 noise seed 0。仅 float32 token 数据约 1.31 GB，另外保存动作、身份与分片元数据。原始逐 token 缓存可供后续分析，不需要重跑模型才能重新选择池化方式。

旧版 batch=4 MPS pilot 的中位耗时约 0.19–0.21 秒/帧；线性外推全量约 44–49 分钟，**只是预算参考，不是全量实测**，也不含后续读出计算。最终版和具体机器的耗时以新 pilot 为准，不默认使用未经测量的 batch=16。

### 第三步：Hidden/PCA 与动作、时间对照

```bash
bash scripts/python.sh -W error::RuntimeWarning \
  -m interaction_vla.representation_study.libero.token_cache evaluate \
  --cache outputs/predictive_states/smolvla_tokens_mps \
  --output outputs/predictive_states/smolvla_hidden_pca
```

这个步骤在 CPU 上训练小型读出器和 PCA，**不更新 SmolVLA，也不训练 SAE/RET**。输入必须是完整 token/action cache；4 帧 smoke 输出或 64 状态 pilot 都不能代替完整缓存。

当前提供 8 个表示分支：`Hidden/PCA × token均值/首token × 当前/4帧历史`。控制包括时间、robot state 当前/历史、策略首动作、当前预测 action chunk，以及表示加 action chunk；另行报告真实演示未来动作条件。

如需审计时间窗口、查看官方 SAE/RET 的既有入口，见 [完整协议](research/predictive-states.md)。这些是后续或独立用途，不需要为了当前两步任务全部运行。

### 第四步：检查 Z+A 是否比 A 提供更好的预测

复用第三步已保存的预测，不重新提取或训练。`--run-dir` 指向完整评估目录；下面使用截距计算修复后运行的 `smolvla_hidden_pca_fixed`，若你的目录名不同请替换。输出目录必须尚不存在。

```bash
bash scripts/python.sh -W error::RuntimeWarning \
  -m interaction_vla.representation_study.libero.action_increment \
  --run-dir outputs/predictive_states/smolvla_hidden_pca_fixed \
  --output outputs/predictive_states/smolvla_action_increment
```

- **A**：当前观察下策略预测的 50 步动作计划；不是演示中实际执行的未来动作。
- **Z**：已有的全部 8 个 Hidden/PCA 表示，不按测试结果挑选最佳分支。
- **判据**：`gain_a_minus_za = Brier(A) − Brier(Z+A)`，正值表示该读出协议下加入 Z 后预测误差更低。主指标先逐任务求均值，再等权平均；另报窗口加权结果。
- **输出**：`summary.csv` 给出目标 × horizon × 表示 × 全部/变化/未变化窗口；`report.json` 还包含逐任务、逐 episode 的配对差值、输入维度、alpha 和来源哈希。state ID、任务、episode、标签与 StateBank 对齐，保存的预测须能重算原报告 Brier。
- **解释范围**：`+0` 是当前标签解码，`+1/5/10` 才是未来预测；“变化”仅指当前与未来端点标签不同。只有两个测试任务，当前仅描述配对增益，不给出逐帧独立性假设下的显著性结论。Z+A 维度更高，尚缺同维度无关特征等容量对照；正增益不证明因果使用，无增益也不证明 Z 没有额外信息。

### 第五步：同维度随机控制

在 Z+A 之后运行容量对照。它为每个 Z 分支生成同宽度的确定性随机特征，比较 A、Z+A 与 R+A；`R+A` 的增益代表仅增加输入维度和 Ridge 估计自由度可能带来的改善，`Z+A` 超过 `R+A` 才是更有意义的表征增量信号。

```bash
bash scripts/python.sh -W error::RuntimeWarning \
  -m interaction_vla.representation_study.libero.capacity_control \
  --cache outputs/predictive_states/smolvla_tokens_mps \
  --output outputs/predictive_states/smolvla_capacity_control
```

结果保存到 `report.json`；`gain_over_A` 定义为 `Brier(A) - Brier(condition)`，正值更好。该控制仍是有限 Ridge 读出器下的容量诊断，不是语义负对照、互信息估计或因果检验。

## 数据协议与公平比较

| 项目 | 当前定义 |
| --- | --- |
| StateBank | Spatial/Object 共 20 任务、100 episode、13,603 状态；经过回放兼容性筛选 |
| 时间单位 | 10 Hz；历史 4 帧；预测当前与未来 1/5/10 帧，即 0/0.1/0.5/1 秒 |
| 完整窗口 | 12,303；train 8,477 / validation 2,621 / test 1,205 |
| 划分 | 沿用既有 task-group split；不随机拆帧，不跨 episode 拼接，不插值补帧 |
| 标签 | Contact、StableGrasp；缺失保持缺失；各目标报告实际可评估数量 |
| PCA | 32 维；只在训练状态拟合归一化与 PCA；随后应用到 validation/test |
| 读出 | float32，StandardScaler + Ridge/LSQR；alpha 为 0.1/1/10/100，仅按 validation Brier 选择 |
| 指标 | 任务宏平均 Brier 为主；补充整体 Brier、AUPRC、标签变化子集及任务聚类差值区间 |
| 比较性质 | 当前是探索性诊断，尚未全面匹配表示维度、模型容量与训练预算 |

三类容易混淆的信息条件：

- **当前策略动作计划**：在当前观察下生成的预测 action chunk，当前时刻可获得，但不保证后续闭环真的执行整个 chunk。
- **未来实际演示动作**：从数据中读取未来动作，属于额外信息条件；不能称为反事实或自主预测。
- **保持当前真实标签**：使用特权当前状态的 persistence 对照，不是视觉策略免费拥有的信息。

当前时间对照只使用已观察到的 elapsed time，不使用完整 episode 时长归一化。动作距离位于 checkpoint 后处理后的策略坐标中，尚未经过环境裁剪、控制器缩放或实际执行；不能直接解释为米或弧度。

Ridge 的原始预测先检查有限值，再截断到 [0,1] 用于评分；这些分数不是已经校准的概率。读出程序报告的标签变化子集只表示当前与未来端点标签不同，不统计两端之间发生的全部事件。

测试集只有 **2 个留出任务**，且无表征对照结果已经查看。大量重叠窗口不等于大量独立任务，增加训练 seed 也不会增加任务样本量。当前“跨任务”指表示学习/读出器的任务留出；官方策略在原始训练中是否接触过相应任务或演示，尚未由本流程确认。

## 查看结果与判断完成

| 阶段 | 主要文件 | 完成条件 |
| --- | --- | --- |
| 噪声 pilot | `token_noise_check/noise_report.json`；各 `noise_*/manifest.json` | 三个 seed 完整；首批重复检查通过；报告真实噪声敏感性，不设任意低比值门槛 |
| 全量提取 | `smolvla_tokens_mps/binding.json`、`manifest.json`、`shards/` | `complete=true`、`full_state_bank=true`、`states=13603`；分片身份、形状、有限值与哈希匹配 |
| 进度 | 提取目录中的 `progress.json` | 运行中约每 32 帧更新；不是完整 manifest 的替代物 |
| 离线比较 | `smolvla_hidden_pca/report.json`、`sequence_audit.json` | 结果报告生成；逐项检查有效样本数、方法和信息条件 |
| 结果复核 | `test_predictions.npz`、`readouts.joblib`、`preprocessing.joblib` | 可核对逐状态预测和拟合器；仅加载自己可信来源的 joblib 文件 |

上表路径均相对 `outputs/predictive_states/`。查看运行进度可以在另一个终端执行：

```bash
cat outputs/predictive_states/smolvla_tokens_mps/progress.json
```

### 中断与续跑

- 提取可用**完全相同的命令**续跑：先验证已有分片，再跳过已完成部分。
- 同一输出目录只允许一个写入进程。不要并行执行两份相同命令。
- 修改源码、设备、batch size、checkpoint、数据或噪声条件时，使用新的输出目录；不手改 binding/manifest 强行拼接缓存。
- 完整缓存重复运行会校验并返回已有结果，不代表重新提取了一遍。
- 离线比较目前不支持中途续跑，也拒绝覆盖已有输出目录；失败后保留现场，排查后使用新目录重新执行。
- 不删除旧结果来让新命令“顺利通过”。`token_noise_pilot_mps/` 是中间版本历史 pilot，不是最终源码的可续跑目录。

### 已经存在的当前路线结果

真实标签/时间/动作对照保存在 `outputs/predictive_states/real_controls_seed42/`，**未使用 VLA 激活**。未来 10 帧的 Brier 如下，越低越好：

| 对照 | Contact | StableGrasp |
| --- | ---: | ---: |
| 训练集阳性率 | 0.2473 | 0.2488 |
| 已观察到的时间 | 0.1805 | 0.1882 |
| 保持当前真实标签 | 0.1618 | 0.1724 |
| 未来实际演示动作序列 | 0.0343 | 0.0599 |

Contact 有效测试窗口 1,205，StableGrasp 1,195，均来自 2 个任务。最后一行拥有未来真实动作，不能将其优势归功于 VLA 表征。真正需要回答的是：当前可获得的表示能否提供超过时间、本体状态和策略动作计划的信息。

## 历史实验与已有发现

以下来自旧协议与已归档证据，**不是当前 Action Atlas token 路线的复现结果**。保存它们是为了解释研究问题如何形成，而不是让新流程自动继承旧结论。

| 历史实验 | 已有观察 | 边界与证据 |
| --- | --- | --- |
| ACT Graph-v2 | 3 个 policy seed、每条件共 60 rollouts；Flat 30.0%、Teacher Graph 35.0%、random-init predicted Graph 40.0%、Reflect-pretrained predicted Graph 41.7% | 受控任务结果，不能推广为 Graph 普遍优于 Flat；[注册表](ccfa.yaml) |
| ReflectVLM Graph pretraining | 更接近 teacher 的 Graph 没有自然转化为稳定更高的控制表现 | Graph 预测质量与策略可用性不等价；[注册表](ccfa.yaml) |
| StateBank | 20 任务、100 episode 的回放、标签和分组审计归档 | 候选采纳 100/120，存在任务相关筛选偏差；[证据](docs/results/libero_state_bank_formal/README.md) |
| Protocol-v3 | expert-only SFT 下研究 8 个 checkpoint、4 taps、6 类因素；288 个可估计单元完成，96 个不可估计 | 不可估计单元不填零；上游视觉冻结；[报告](docs/results/libero_smolvla_protocol_v3/README.md) |
| 线性因素干预 | StableGrasp 定向干预没有在四个 checkpoint 上超过匹配随机干预 | 失败的是特定干预门槛，不是证明模型从不使用抓取信息；[证据](docs/results/libero_smolvla_protocol_v3/README.md) |
| 官方策略 positive control | 单个 Spatial task 上成功 9/10；Contact/StableGrasp 可读出，但 rank-one 定向干预仍弱于匹配随机 | 单任务门检，不是 LIBERO 整体成功率；[证据](docs/results/libero_smolvla_positive_control/README.md) |
| Protocol-v5 SAE | 8 个冻结候选中，4 个在所用参考字典上超过匹配随机动作效应门槛 | 主要涉及 gripper 连续输出；尚缺其他字典的独立干预复验和闭环验证；[归档](docs/results/libero_smolvla_sparse_features/README.md) |
| Recovery RL calibration | 未通过原预定分布门槛 | 保留负结果，不因此启动新的 RL；[注册表](ccfa.yaml) |

### 不能混用的 checkpoint 和 tap

| 条件 | 旧 positive control / SAE 路线 | 当前 token/action 路线 |
| --- | --- | --- |
| Checkpoint 来源 | `lerobot/smolvla_libero` | `HuggingFaceVLA/smolvla_libero` |
| Revision | `31d453f7edd78c839a8bbc39744a292686daf0de` | `6721902bc4d61e50a3bfdb11dfb4cb626f05d102` |
| 主 tap | `action_time_mlp_out`，历史名 `action_expert_input` | Action Atlas `expert/31/mlp/output` |
| 保存/分析单位 | 旧 720D pooled 表示 | `[50,480]` tokens，并派生均值/首 token |
| 已有能力记录 | 特定旧配置的单任务 9/10 | 当前配置尚无闭环成功率验证 |

不假定两者等价，也不把不同字典的 feature ID 当作同一个神经机制。

<details>
<summary>旧 SAE pilot 的具体发现与解释限制</summary>

Protocol-v5 在旧 720D pooled 激活上训练 3 个 TopK 字典：width 1440、k=32、5000 updates。归档报告中，test 集 explained variance 为 0.9974–0.9979，但只有 187–217 个 atom 在 validation 中存活；高重建质量并没有解决大量 inactive features 的问题。

候选先按跨 seed decoder/activation 一致性与任务/episode 覆盖选择，再检查语义和动作。参考字典中 517、694、977、981 四个候选超过匹配随机动作效应门槛；主要改变连续 gripper 幅度，没有据此证明夹爪符号翻转或夹持成功率改善。

这些结果提示“语义关联排序”可能不同于“动作敏感性排序”。但跨字典匹配到类似特征，不等于已经在每个字典上独立复现干预效应；激活范数接近原值，也不能证明干预没有离开训练分布。原始轻量报告和逐状态效应在 [归档目录](docs/results/libero_smolvla_sparse_features/README.md)，字典权重未随归档上传。

</details>

### 历史执行入口仅供定位

旧入口为 `libero features discover`、`libero features intervene`、`libero features report`，属于原 `interaction_vla.representation_study` CLI。参数与服务器配置见 [SERVER_RUNBOOK.md](SERVER_RUNBOOK.md)。

**该手册包含历史启动、提交和训练命令，不是本 README 当前任务的自动执行清单。** `ccfa.yaml` 仍保留旧科学问题、注册状态及关闭的执行门槛；本轮文档不修改它，不把 `execution_allowed=false` 当作可忽略提示。重启旧干预、闭环、SFT 或 RL 需要单独确认。

## 已知限制与常见问题

| 问题 | 应如何处理 |
| --- | --- |
| 本机能做实验吗？ | 能做已验证过的离线推理与数据处理；最终 token/readout 代码仍需用户验证。不能由此断言 LIBERO 闭环或 CUDA 已通过 |
| 找不到 StateBank | 先恢复原始记录与 split；下载 Hub 数据不能自动生成项目物理标签 |
| 离线加载报缺文件 | 核对 checkpoint、11 个 tokenizer/配置文件及数据；恢复资产时在联网终端执行固定 revision 下载 |
| FFmpeg / TorchCodec 导入失败 | 检查 `ffmpeg@8` 并使用 `bash scripts/python.sh`；不要随意替换系统动态库 |
| PyAV/OpenCV 重复 Objective-C 类提示 | 已记录为未解决风险；不要为了安静而删除第三方 wheel 的动态库 |
| `torch_shm_manager: Operation not permitted` | 历史官方 SAE 测试曾被沙箱共享内存权限拦截；这是运行权限问题，不应通过改算法掩盖 |
| `Resume binding differs` | 检查源码、batch、设备、数据与权重是否改变；采用新目录，不绕过绑定 |
| NaN / Inf 或 RuntimeWarning | 保存完整报错、命令和输出目录后停止该步骤；不要自动替换为零或全局屏蔽告警 |
| `mps` 不可用 | 先区分权限限制与硬件支持；可另用 CPU 和独立输出目录测量，不混合设备缓存 |

已有两类独立数值风险：NumPy dense matmul 在本机出现过输出有限但浮点告警异常的现象，读出/PCA 投影使用现成 SciPy 路径；既有 MuJoCo 接触力测试曾出现一次 NaN，隔离和完整重跑未稳定复现，根因仍未确定。前者的处理不能算作后者已修复。

基础诊断记录见 [工程检查报告](docs/research/engineering-audit.md)；其中旧环境版本、缺失权重和安装建议按记录日期理解，日常操作以本页的唯一环境和当前入口为准。

需要完整工程回归时，手动执行：

```bash
bash scripts/python.sh -m pytest tests research/action-atlas/tests -q -rs
```

这会包含小型合成训练与物理测试，不是只检查语法，也不是完整论文实验。不要把测试计数当作科学结果。

## 后续阶段与复现边界

1. **当前交付**：用户验证 token-preserving 缓存与噪声 pilot，完成全 StateBank 提取，运行 Hidden/PCA 与动作、时间对照。
2. **预测性表示**：在同一绑定缓存上比较官方 SAE/RET，检查历史顺序、维度/预算和 seed；不以某个方法必须获胜为前提。
3. **机制验证**：只在训练/验证数据上选择稳定候选，加入同 token、同去噪步、同范数/秩的随机及非候选特征控制。
4. **闭环验证**：先核实当前策略能力，再从配对初始状态测接触、夹持、释放、掉落及成功率，不只看动作大小。
5. **扩展轴**：world model、第二个 VLA 或 Init→SFT→RL，等待当前发现与资源条件支持后另行设计。

源码、模型、数据和实验记录各自版本化：保存 checkpoint revision/hash、上游 commit/patch、StateBank 身份、取层与 token/去噪规则、运行时、噪声及读出参数。相同方法名称不代表相同配置，代码完成不代表实验完成。

`outputs/`、虚拟环境和上游独立 checkout 被父仓库忽略；Git clone 不会恢复这些内容。准备发布时先用 `git status --short` 核查新增源码是否被版本控制记录，不自动提交、推送或上传私有研究材料。仓库使用 [MIT License](LICENSE)；Action Atlas、RET、模型、数据及其依赖分别保留自身许可证与归属，项目许可证不替代它们。

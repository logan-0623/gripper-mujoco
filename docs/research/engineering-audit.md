# 项目工程检查与论文代码接入判断

## 2026-09-07 接入更新

已按用户确认将 Action Atlas 官方仓库接入 `research/action-atlas/` 作为后续 SAE/VLA 干预的代码 base，保留旧实现及证据。仍只有 `.venv-lerobot` 一个环境；原锁定版本不变，加入上游依赖后为 117 个包。当前安装入口改用 `requirements-action-atlas-macos.lock.txt`，下方旧环境命令和 99 包描述为历史记录。具体版本、恢复命令和限制见 [研究 base 接入说明](../../research/README.md)。

上游源码未修改：官方 9 项测试通过，原版 SAE 在 CPU 合成数据上完成训练、保存及官方 hook 加载。完整原项目回归在沙箱外运行，修正既有 SAC 测试对 `mps` 与 `mps:0` 的比较误判后为 **1019 passed、1 skipped**；唯一跳过为真实 LIBERO 集成。未修改 SAC 算法。原生导入、物理步进与视频读写检查仍通过；PyAV/OpenCV 的重复类警告仍保留。

尚未加载真实策略权重、执行 LIBERO rollout 或复现论文指标。当前 LeRobot 0.6.1 与上游固定 submodule 并非同一版本；正式 baseline 前还需核实初始状态/seed 处理及激活位置。不能将“成功接入官方代码”表述为“已完整复现论文”。

---

检查日期：2026-09-06。范围：本地环境、全仓回归测试、LIBERO 主线关键实现，以及已归档 SAE 动作效应的统计重算。

## 结论

**项目有可以继续使用的实验基础，不需要推倒重写；但目前不能说整条服务器实验链路已经确认无误。** 本次修复了回放验收的边界漏洞和测试入口问题，重建唯一的 `.venv-lerobot`，补齐 SmolVLA/HDF5 依赖，并修复 TorchCodec 的 FFmpeg 版本不兼容。下一步应先在 Linux/CUDA 上补足真实数据与真实策略集成检查，再接入论文代码做对照。

这里区分三件事：测试通过说明被测试的实现行为符合预期；统计重算一致说明缓存到报告的计算可重复；它们都不能单独证明物理标签正确、策略干预有效或论文结论成立。

## 1. 实际完成了哪些检查

| 检查 | 本次结果 | 覆盖边界 |
| --- | --- | --- |
| 原 `.venv-lerobot` | 无法直接使用：缺少 Python 解释器，配置仍引用原机器路径 | 移到项目外的临时备份目录，未覆盖其内容 |
| 重建 `.venv-lerobot` | 项目唯一环境，按更新后的 lock 安装 99 个包 | macOS 开发/测试环境，不是 Linux/CUDA 训练环境 |
| 依赖一致性 | `uv pip check` 通过 | 包元数据兼容不等于所有后端都已执行 |
| 原生运行时检查 | SmolVLA/LeRobotDataset/TorchCodec 导入、MuJoCo 步进、3 帧合成 RGB 视频编码/解码全部通过 | 没有加载策略权重；视频检查不涉及真实 LIBERO 数据 |
| 全仓测试 | 修复后 **1013 passed, 7 skipped** | 包含基础物理、数据处理、ACT 小规模训练与加载、探针、干预与 SAE 单元测试 |
| SAE 归档完整性 | `SHA256SUMS` 中 9 个文件全部匹配 | 验证文件完整性，不是模型复现 |
| SAE 统计重算 | 8 个候选 × 6 个指标，共 48 组；均值、CI、p 值及主指标 BH q 值均与归档一致，最大数值差为 0 | 使用归档逐状态效应和现有统计函数；未重新执行策略推理 |
| Linux LIBERO 真回放 | **未执行** | 本机没有该集成测试所需的原始 HDF5/BDDL 与 Linux LIBERO 运行条件 |
| 官方 SmolVLA 在线推理及闭环干预 | **未执行** | 缺少本地完整权重、latent cache 和运行环境；mock 测试不能替代 |

最终环境：Python 3.12.14、PyTorch 2.10.0、MuJoCo 3.3.4、LeRobot 0.6.1、datasets 4.8.5、NumPy 2.2.6、Transformers 5.5.4、TorchCodec 0.10.0。`requirements-lerobot-macos.txt` 增加 h5py 与 SmolVLA extra，`requirements-lerobot-macos.lock.txt` 重新解析并固定全部间接依赖。没有保留诊断阶段临时安装但项目不需要的 SciPy，也没有下载模型权重。

`scripts/python.sh` 统一使用 `.venv-lerobot`，为当前进程设置 FFmpeg 8 的动态库和可执行文件路径，不写入 shell 配置。`scripts/check_environment.py` 独立检查模型类导入、数据集导入、MuJoCo 步进和合成视频的编码/解码，不需要联网或训练。

剩余原生库警告：SmolVLA 导入时，PyAV 与 OpenCV 的 wheel 会报告重复的 `AVFFrameReceiver` / `AVFAudioReceiver` Objective-C 类。此次导入、全仓测试和视频检查均成功，但不能保证相机采集路径不受影响。本次没有删除第三方 wheel 中的动态库或屏蔽该警告；macOS 相机采集仍需在沙箱外验证。

7 个跳过项：4 个 macOS CoreGraphics/画面采集测试、1 个 MPS 测试、1 个 SAC accelerator 测试、1 个真实 LIBERO 回放测试。沙箱内检测不到 CUDA/MPS；这不是对机器硬件是否支持 MPS 的最终判断。

## 2. 找到并修复的问题

### TorchCodec 无法加载系统 FFmpeg 9

严重性：高，视频数据读取路径不可用，但原回归测试没有直接覆盖这个导入。

实际导入 TorchCodec 0.10 时缺少 `libavutil.60.dylib` 等兼容动态库。机器的默认 FFmpeg 为 9，而该 TorchCodec 版本支持 4–8。已并行安装 keg-only 的 FFmpeg 8.1.2，并通过项目启动脚本选择它，没有切换系统默认 FFmpeg。兼容范围见 [TorchCodec v0.10.0](https://github.com/meta-pytorch/torchcodec/tree/v0.10.0)，并行安装方式见 [Homebrew ffmpeg@8](https://formulae.brew.sh/formula/ffmpeg@8)。

### 回放没有验证任何转移，也能被标记为通过

位置：[replay.py](../../interaction_vla/representation_study/libero/replay.py)。严重性：中，属于数据入口的错误验收风险。

修复前，空 `states/actions`，或只有一个 state 和一个 action、没有下一状态可供比较的轨迹，都可能得到 `passed=True`：误差列表为空，汇总误差默认为 0。已通过小例子复现。

现在要求轨迹非空且至少包含一条可验证状态转移；拒绝非有限/非正容差，并在执行动作前拒绝非有限的恢复状态。新增 16 个参数化回归案例。正常的官方等长 states/actions 轨迹仍受支持：最后一帧没有 next state，但前面的转移必须可验证。

**没有证据表明正式数据触发过这个漏洞。** 不应据此宣布已有结果无效。需要在原始数据上检查轨迹长度，才能判定历史影响。

代码更新会改变 collector 的 pipeline hash。如果使用新代码重新执行 `state-bank collect`，旧目录的 binding 可能被拒绝；这是已有版本保护的预期行为。继续读取冻结证据，不修改旧 manifest，也不绕过 hash 检查；若重新采集，使用新的输出目录。

### README 改版后，测试仍检查旧入口

位置：[test_project_contract.py](../../tests/interaction_vla/representation_study/libero/test_project_contract.py)。严重性：低，影响回归测试可信度。

README 已将旧的完整命令移到 `SERVER_RUNBOOK.md`，测试却仍要求 README 包含它们及一句过时文字。现在检查 README 指向 runbook、暴露当前 features 入口，并在 runbook 检查历史正式命令与禁止新 SFT/闭环/PPO/SAC 的边界。没有删除项目科学状态的断言。

### 基础依赖测试没有正确跳过 LeRobot 集成项

位置：两个 `tiny_lerobot_dataset` fixture，以及 dataset-bound backend 测试。严重性：低。

只安装基础依赖时，一处 fixture 抛出包元数据缺失错误，另一处直接导入 LeRobot 导致失败。已在这些真正依赖 LeRobot 的入口增加 `pytest.importorskip`。安装完整依赖后仍执行这些测试，不会把真实集成错误当作通过。

## 3. 主线实现中已经存在的可靠做法

| 环节 | 已检查到的保护 | 仍需要什么证据 |
| --- | --- | --- |
| 数据对齐 | raw HDF5 与 LeRobot 通过 action 序列、episode 身份及 immutable revision 绑定；不是仅凭文件顺序配对 | 在真实数据上抽查 RGB、robot state、接触和 action 的时间对齐 |
| 物理回放 | 每帧恢复 recorded state；采集动作前 observation/contact，再用下一状态检查一步动作误差 | 当前主门限检查 qpos，不等于自由滚动轨迹、qvel 和控制器内部状态全一致 |
| 标签 | StableGrasp 使用过去的短时间窗、双侧手指接触、相对位姿稳定和运动/离开支撑条件；不是仅以夹爪闭合代替抓取 | 标注阈值和物体映射仍需要可视化人工抽查 |
| 数据分组 | episode/task 分组；验证同一 episode 的帧不跨分区；SAE 归一化仅由训练集拟合 | 官方已训练策略是否见过这些演示，与本项目自身 SFT 排除 StateBank 是两回事 |
| 干预 | 同 checkpoint 下固定逐 state 噪声；重建原 extraction batch 上下文；核对 live pooled activation 与缓存；退出后清除 hook | 用真实模型验证 zero-delta 与无 hook 一致、干预确实触发在预期去噪步 |
| 统计 | episode-cluster bootstrap、候选主指标 BH 校正；已从归档效应重算一致 | 统计一致不补足其他 SAE seeds 的动作复验或闭环效果 |

相关实现位于 `interaction_vla/representation_study/libero/` 下的 `alignment.py`、`runtime.py`、`annotation.py`、`splits.py`、`latents.py`、`recruitment.py`、`feature_discovery.py`。

## 4. 不是单元测试能解决的实验风险

### 跨 checkpoint 的噪声不相同

`latents.py:deterministic_inference_noise` 使用 `checkpoint_id:state_id` 生成种子。同 checkpoint 的 original/target/random 可以配对，但同一个 StateBank state 换 checkpoint 时会换噪声。

因此现有 longitudinal 比较没有完全控制推理噪声；特别是接近输出的 action-expert tap，阶段差异可能混入噪声差异。这是混杂风险，不是已证明的结果错误。以后需要单独版本化的 shared-noise/multi-noise 对照；本次没有修改噪声函数或旧缓存。

### “action_expert_input”离动作生成本身很近

当前 hook 位于 `action_time_mlp_out`，捕获最后一个去噪步骤并对 action tokens 求均值。检查现有 LeRobot 源码可见，它直接处理 noisy action 与 timestep 的组合；最后一步的 noisy action 又已经受此前去噪和场景信息影响。

所以这些特征可能表达动作计划或动作状态，不能仅凭 probe/SAE 关联就称为独立的“物理世界表示”。需要 action-only 等匹配对照才能判断是否存在额外语义信息。更换到另一层是在改变实验问题，不能作为工程修复悄悄进行。

### SAE 动作阳性仍是一个受限结论

归档可重算出 517、694、977、981 四个阳性候选，但动作干预目前使用参考字典；“三个字典里匹配得到类似特征”不等于“三个字典都独立产生同样的因果动作效应”。每个候选当前只有一个 matched-random direction，且干预集中于高激活 held-out states，效应不能直接解释为全 StateBank 平均效应。

另外，范数接近原值不等于处于训练分布内；高重建解释方差也不等于闭环动作保真。现有代码没有证明 success 提升，报告不应越过这个边界。

## 5. 可以借鉴哪些论文代码

这里按 CCFA 文献检索规则，只核实公开论文/作者仓库；没有把私有实验数字送入搜索，也没有下载执行第三方仓库。核验日期为 2026-09-06；以下版本均是本次访问的 main 页面，正式接入必须另行固定 commit。

| 候选 | 已核实内容 | 对本项目的具体用途与限制 |
| --- | --- | --- |
| [Action Atlas / Not All Features Are Created Equal](https://github.com/CWRU-AISM/action-atlas) | 2026 论文配套 system/tool；仓库声明 Apache-2.0，已有 SmolVLA adapter、采集/消融/injection hooks | **优先作为接入对象。** 借鉴层访问、闭环消融和特征可视化；不是整库替换。SmolVLA adapter hook 的是 expert/VLM 层内 MLP，与你的 `action_time_mlp_out` 不同；默认 checkpoint 和 LeRobot submodule 也不同 |
| [Event-SAE](https://github.com/xc-j/Event-SAE) | 2026 预印本配套 method/tool；仓库声明 MIT，提供事件关键帧、特征排名、residual-preserving 干预；后端是 OpenVLA 与 π0.5 | 适合参考事件到 feature 的分析，以及闭环保真对照。它是 BatchTopK SAE，不能冒充当前 per-sample TopK 的等价替换；SmolVLA hook 必须适配。VLM 自动标注环节涉及外部服务，不能默认上传项目数据 |
| [DR.VLA](https://drvla.github.io/) | 2026 方法工作项目页，介绍通用/episode-specific features 与 steering | 可借鉴覆盖度诊断；本次访问的页面未提供可核验代码仓库和代码许可证，因此不列为可直接导入依赖 |

Action Atlas 的 [model_adapters.py](https://raw.githubusercontent.com/CWRU-AISM/action-atlas/main/experiments/model_adapters.py) 已检查到 SmolVLA 的 MLP 层选择。其 [hooks.py](https://raw.githubusercontent.com/CWRU-AISM/action-atlas/main/experiments/hooks.py) 中 injection 形状不匹配时会累计计数并返回原输出。因此即使复用论文代码，也要在本项目外层要求实际注入次数正确、shape mismatch 为 0，不能出现“命令跑完，但没有真正干预”。

仓库声明的许可证不自动涵盖所有依赖、数据或模型；实际导入时应保留相应 LICENSE/NOTICE，并检查导入文件及其子模块的许可。本次没有 vendor 任何外部实现。

## 6. 下一步顺序：先验收，再做科学对照

1. **本地基础层：本次已完成。** 只使用 `.venv-lerobot`，保留修复及测试。不要将虚拟环境复制到服务器；服务器按 Linux requirements 重建。
2. **服务器集成层：尚待执行。** 在独立输出目录检查一条真实 LIBERO 演示的回放与标签；加载固定官方 checkpoint，检查图像通道、预处理与动作尺度；验证 no-hook/zero-delta 等价、缓存/live activation 一致、hook 次数与去噪步一致。这些检查先于正式大规模实验。
3. **论文代码适配层：通过前一层后接入。** 固定 Action Atlas commit，在独立环境先复现它的最小原生例子，然后通过薄 adapter 连接本项目。显式记录 checkpoint、取层、token pooling、去噪步、action normalization 与原论文的差异。
4. **科学证据层：另立新协议。** 保留旧 TopK/旧噪声/旧结果作为基线；再开展跨字典独立干预、多随机对照和必要的动作基线。不要一边换数据、模型、SAE 和 hook，一边将变化统称为“优化”。

不要求先证明整个仓库“绝无 bug”才开始新实验；要求的是下一步将依赖的那条链路有清楚、可复跑的验收标准。当前建议是**可以继续准备下一步，暂不放行长时间训练或正式闭环结论**。

## 本地复跑

在项目根目录执行：

```bash
HF_HOME=/tmp/gripper-audit-hf \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
PYTHONPYCACHEPREFIX=/tmp/gripper-audit-pycache \
  bash scripts/python.sh -m pytest -q tests -rs

uv pip check --python .venv-lerobot/bin/python
bash scripts/python.sh scripts/check_environment.py
```

环境由 `uv venv --python 3.12.14 .venv-lerobot` 和 `uv pip sync --python .venv-lerobot/bin/python requirements-lerobot-macos.lock.txt` 建立，系统依赖为 `brew install ffmpeg@8`。日常安装使用 lock；有意升级时才重新解析 requirements 并重跑检查。

本次没有更新服务器、提交/推送 Git、启动新 SFT/RL/闭环实验，也没有改写原始结果、SAE 权重、科学 gate 或已生成项目介绍 HTML。

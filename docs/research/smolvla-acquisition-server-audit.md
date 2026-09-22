# SmolVLA 能力形成与 flow 干预：服务器结果审计

审计日期：2026-09-15。模式：结果与数值审计、实验设计对照。只读服务器检查与离线重算；未启动训练、策略推理或闭环 rollout，未修改服务器原始结果、代码或研究门禁。

## 1. 结论

新谱系已经具备限定任务上的强终点和可靠的能力变化记录；无标签变化方向能够在独立于候选拟合的 episodes 上复现，并能影响冻结策略的动作输出。但当前尚不能宣称发现了跨任务控制机制或解释了成功率增长。最重要的限制是验证只有 3 个 episodes、阶段匹配对照不足，以及全 50-token 动作效应不等于实际执行前 10-token 的效应。

当前阶段应描述为：**能力时间轴完成；固定点表征变化与离线动作敏感性 pilot 完成；机制特异性和闭环因果验证尚未完成。** “本批命令跑完”不等于整个研究设计完成。

## 2. 审计范围与来源

服务器仓库当前 commit：`9bf8de2`。结果根目录在该服务器的 `smolvla-official-reproduction-v2/acquisition/`；以下路径均相对该目录。当前 checkout 只标识检查时源码，不能自动当作全部历史运行的源码版本；缓存另有 source binding。

| 对象 | 本次检查结果 |
| --- | --- |
| 5k、10k、15k、20k、25k checkpoints | 五份 checkpoint tree hash、train_config hash 与 lineage 一致 |
| `timeline/` 与 `timeline-summary/` | 20 cells、200 条原始 episode，任务/状态顺序、success 逐条一致、8 项事件计数及来源哈希通过 |
| 能力配对变化 | 相邻 checkpoint 的 gains/losses 独立复算通过 |
| 全部 flow/干预 trace | 61 份 manifest、1,064 个 NPZ 分片，complete、binding/分片哈希、唯一且有序 state IDs、数值有限性通过 |
| 固定点输入 | 256-state discovery、128-state train、128-state validation 三组中，五阶段 checkpoint 的 epsilon、sigma、x_sigma 与各自 25k reference 数组完全相同 |
| episode 版候选 | 两个 tap 的 shared_basis hash、候选到 train/validation trajectory 的 hash、投影 NPZ hash 与数值有限性通过 |
| 48-state 动作干预 | 21 个条件均完整；状态、噪声配对及所检查 checkpoint/processor/data/runtime 合同一致；动作效应由原始数组重算 |
| 训练结束 | `full.log` 记录 25,000/25,000，5:54:46，正常 End of training |

这不是重新训练复现，也不包含全部视频人工复核、优化器更新轨迹逐步审计或每份旧实验的统计重算。原始图像/视频未全量读取。没有在最新 acquisition 结果树内找到闭环干预结果或正式 offline gate 汇总；21 个动作条件目前是独立缓存，不是已完成的机制结论报告。

可复核的来源哈希：

```text
lineage.json
498d1aeddc2fd381960afa15fd7b986c40162fd9ecb9ade422ada14953f22e4a
timeline-summary/report.json
0c319a337923a978008c4e40a35c0f7ab1caa84c128e03263c80ce5e70f77b49
candidates_episode_train_n128_r3/expert_middle/candidates.json
7128be890332ebc0059dceee6b8dbdc2f0da3a216cd8b2e86bcc4e0035777141
candidates_episode_train_n128_r3/expert_late/candidates.json
7e7a295cb5b55485f5b0c853e3f6b351204e2f755f934fa154ee24ea7f0ad61a
```

## 3. 能力与训练覆盖

| step | task 0 | task 1 | task 2 | task 3 | 总成功 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5k | 0/10 | 3/10 | 2/10 | 0/10 | 5/40 |
| 10k | 7/10 | 4/10 | 7/10 | 7/10 | 25/40 |
| 15k | 7/10 | 8/10 | 8/10 | 7/10 | 30/40 |
| 20k | 7/10 | 10/10 | 9/10 | 7/10 | 33/40 |
| 25k | 8/10 | 10/10 | 10/10 | 10/10 | 38/40 |

相邻阶段 gains/losses 依次为 22/2、9/4、5/2、6/1。它们来自同一组 Spatial 0–3、初始状态 10–19，而不是新增 200 个独立场景。单一训练 seed，checkpoint 选择参考了这批行为结果，结论是该谱系的探索性能力时间轴，不是训练规律的独立多 seed 确认。

新训练配置为 batch 32、25k steps、`episodes: null`，数据 revision 为 `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`。本次读取 episode metadata 与数据 Parquet 的 episode_index/task_index，确认 1,693 个唯一 episodes、40 个 task indices。按任务指令对齐 Spatial 0–3，数据集 task indices 分别为 34/37/38/35，含 45/45/46/43 个 episodes。

因此，这一新训练配置包含 Spatial task 3 的数据，不应继续称为旧 smoke 谱系的 task-3 未训练泛化测试。此处确认的是数据与无子集限制的训练配置，不是逐次 optimizer update 的 sampler 访问计数。也不能凭跨谱系结果把旧失败唯一归因于训练时长。

## 4. 候选发现与验证的实际样本

| 数据版本 | states | episodes | task 范围 |
| --- | ---: | ---: | --- |
| 初版 `flow_discovery_train_n256_r3` | 256 | 14 | Spatial 0、1、3 |
| 当前 `flow_train_episode_n128_r3` | 128 | 9 | Spatial 0、1、3 |
| 当前 `flow_validation_episode_n128_r3` | 128 | 3 | Spatial 0、1、3，每任务一个 episode |
| `action_effects_validation_n48_r2` | 48 | 3 | 同上，每任务 16 states |

当前 128-state train/validation 的 state 和 episode 交集均为零。验证 episodes 为 task 0/demo_41、task 1/demo_21、task 3/demo_6。它们曾出现在早期 256-state discovery 版本中，因此当前版本对拟合是留出，但不能将整段研究过程称为从未查看的全新确认数据。

task 2 不在当前表征/干预样本内，不能将这批机制结果直接推广到四任务能力表，更不能宣称跨未见任务泛化。episode 留出也不等于 policy 训练未见数据或任务。

## 5. 候选确实变化，但不能把变化方向直接命名为能力

每个 tap 冻结 4 个 formation、4 个 low-change、4 个 random 方向；共享 rank-32 基底使用 5k、10k、25k 的 train 表征，变化端点为 5k→10k。15k、20k 不参与基底拟合；25k 参与了拟合，不是完全留出的终点。

候选 `formation_0` 在 validation 上相对 5k→10k 变化方向的投影系数为：

| tap | 5k | 10k | 15k | 20k | 25k |
| --- | ---: | ---: | ---: | ---: | ---: |
| expert_middle | 0 | 1 | 1.127 | 1.176 | 1.178 |
| expert_late | 0 | 1 | 1.075 | 1.064 | 1.069 |

系数 0/1 是端点归一化定义，不是独立发现；5k 的方向 cosine 未定义，报告中的 NaN 对应零位移，不是激活 NaN。中间 checkpoints 的方向延续是有用证据，但 late 方向在 10k 后基本平台，而 success 仍从 62.5% 上升到 95%。它可能反映早期计算变化，而非完整能力刻度；单个主方向不能解释所有后续能力增长。

`interpret_validation_25k.json` 的增量 R²来自 `_incremental_r2`：在同批验证 states 上，以 frame_index/task identity 为 nuisance，对激活投影进行嵌套最小二乘拟合。它是样本内残差解释比例，不是独立预测分数，更不是未来预测或因果证据。标签在候选冻结后用于解释符合原则；不能据此再筛选候选并把同批数据作为最终确认。

## 6. 动作效应：全计划与实际执行段的结论不同

本节全部是本次从 `action_postprocessed` 重新计算的描述性结果，不是原始 gate 输出。条件固定为 25k、dose=+1、all stages；每个 state 先对 noise repeats、所选 action-token 前缀和 7 个动作维度计算 RMS，再对 48 states 取均值。

| tap / 方向 | 第一个动作 | 前 10 个动作 | 全 50 个动作 |
| --- | ---: | ---: | ---: |
| middle formation_0 | 0.002960 | 0.006556 | 0.040793 |
| middle low_change_0 | 0.002362 | 0.009392 | 0.022009 |
| middle matched_random_0 | 0.002232 | 0.002376 | 0.011983 |
| late formation_0 | 0.006377 | 0.010041 | 0.027982 |
| late low_change_0 | 0.002187 | 0.004223 | 0.019202 |
| late matched_random_0 | 0.001966 | 0.002205 | 0.007134 |

在全 50-token 指标下，两候选均高于本次两个控制；在前 10-token 指标下，middle 候选低于 low-change。当前评测合同每次执行 10 个动作后重新规划，因此不能只凭全 50-token 指标决定进入闭环。late 候选在三种汇总指标下都高于这两个特定控制，值得进一步验证，但仅 3 个 episodes 和一个随机方向不能确立广泛机制特异性。

按当前 postprocessed 坐标平方能量，两候选 all/+1 的效应分别约 98.8%（middle）、96.7%（late）来自 gripper 维度。该比例依赖各动作维度的尺度，不能称为物理能量或通用重要性。它提示应分开汇报平移、旋转、夹爪、夹爪符号/阈值变化，而不直接将总 RMS 解释成抓取机制。

实际完成 21 个条件：一个 baseline，两个 tap 各 8 个 formation_0（四个阶段范围 × 正负方向），另各一个 all/+1 的 low-change 与 random 控制。单个候选的正负干预不是多个剂量强度实验；early/middle/late 分别编辑 3/4/3 个 solver steps，比较时也要控制编辑次数。

## 7. 工程一致性与统计实现边界

- **输入固定通过，输出并非严格自一致。** 同一 25k 在自然/固定点路径上的输出不完全相同。在 128-state validation 上，各阶段 velocity 自差 RMS 为 0.002864–0.003526，对照 5k→10k 差异 RMS 0.145238–0.254464，比值约 1.4%–2.0%。middle/late 激活自差约为相应跨模型差异的 1.1%–1.3% / 1.8%–1.9%。自然/固定点 binding 除 query mode 和 reference 外未见其他差异；根因尚未定位，不直接归因浮点舍入，也不声称已通过零误差测试。跨模型主变化大于该底噪，但微小效应需单独校准。
- **bootstrap 单元与设计不符。** 当前 `acquisition.py:489` 的 `offline_action_gate` 按 state 重采样，而不是 episode cluster；48 states 只有 3 个 episodes，不能把这样的 CI 当作独立场景推断。此次没有调用该 gate 生成或批准结论。
- **gate 配对检查不足。** 该函数主要检查候选 ID 和 state IDs，未强制核对 baseline/edited 的 checkpoint、噪声、剂量、阶段和完整输入合同。本次对实际缓存进行了额外配对检查，不能把本次通过视为该函数对未来输入的保证。
- **控制被平均。** gate 将 low-change 与 random 的效应平均后比较，可能掩盖候选未超过某一强控制的问题。应保留分类型对照，不用平均值替代所有必要比较。
- **随机方向不足且非严格正交。** 当前仅运行每 tap 的 random_0。候选文件中 random_0 与 formation_0 点积约 −0.00374/−0.00432，范数约 1；不能宣称已经完成严格正交随机方向面板。若严格正交是协议要求，需验证构造结果。
- **存储约束。** 审计时数据盘 97% 使用、剩余约 3.5 GB；未删除文件。扩大实验前先安排存储或只保留所需读出产物，不能让空间不足造成缓存不完整。

## 8. 对当前实验设计的优化建议（未执行）

对照[当前设计 §12.16–12.17](../superpowers/specs/2026-09-10-predictive-interaction-state-experiment-design.md)：研究原则仍适用，执行顺序及若干评判规则需要更新。本报告不改写原设计或 `ccfa.yaml` 门禁。

| 优先级 | 必要调整 | 为什么 |
| --- | --- | --- |
| P0 | 登记新谱系的强终点、时间轴与覆盖；旧 smoke 保留独立历史记录 | 不能继续以旧 task-3 未训练或缺少强模型描述当前状态 |
| P0 | offline gate 主指标绑定实际执行前缀，同时报告 first action / full chunk 和各动作分量 | 已实测发现 middle 候选在前 10-token 上不超过 low-change |
| P0 | 先用现有数据输出逐 episode 效应；正式推断增加独立 episodes，并采用 task 内 episode 聚类 | 48 帧不能替代足够的独立场景；仅每 task 一个 episode 无法估计任务内跨 episode 变异 |
| P0 | 校准自然/固定点自差及零编辑路径，记录容差和来源版本 | 当前“固定输入完全相同”不保证输出零误差 |
| P1 | 对保留候选补齐相同 tap、stage、方向、剂量、token 范围的 low-change/random 控制 | all/+1 控制不能支持任意阶段/负方向结论；3/4/3 次编辑不能直接排名阶段重要性 |
| P1 | 区分正式候选拟合、观察过的验证与真正新确认集 | 当前 episode 划分无交集，但早期 n256 版本已查看过后续 validation episodes |
| P1 | 用已有 15k/20k 检查并报告不跟随能力变化的候选；物理标签仅作事后解释 | 避免将任何训练位移都称为能力获得机制 |
| P1 | 补有区分力的条件来源实验，再进入有预算的配对闭环 | 动作被改变不等于利用了目标图像、语言或物理状态；offline delta 不等于成功率效用 |

保留原设计的 no-op、匹配对照与独立确认要求，不因为这批缓存完整就降低科学门槛。已有 SAE/G2b 结果不能自动继承到新 checkpoint；RET、world model 和 RL 不自动启动。

下一步有两种合理投入方向，需用户选择而非由结果自动决定：

1. **优先验证限定功能机制（建议）**：固定当前候选，补独立 episodes、执行段统计和阶段匹配控制，再决定闭环。优势是直接利用已有缓存；成本较低。限制是结论主要关于当前模型/谱系的功能作用，不直接证明能力形成原因。
2. **优先解释能力形成**：先测同一冻结方向在 5k/10k/25k 中的配对功能效应，结合中间 checkpoints 与独立样本，区分通用扰动敏感性与随训练获得的作用。优势是更贴近主研究问题；需要更多推理和因果控制，跨 seed 训练重复仍是更后续的外部有效性证据。

无论选择哪条，都不要求候选提高成功率、不要求具有人类语义标签，也不要求未来预测通过才允许功能验证。当前最安全的总结是：**观察到随早期训练形成并在留出 episodes 延续的表征方向；干预可改变动作计划，其中一部分效应集中于夹爪及未立即执行的未来动作。是否构成任务相关控制机制，仍需更严格的功能与闭环检验。**

## 9. 2026-09-16 新服务器复核：目标是否偏移

本节是后续复核，不覆盖前述旧服务器实测结果。通过用户指定的新服务器只读检查：仓库 HEAD 仍为 `9bf8de2`，核心修改尚未提交。服务器与本地的 `acquisition.py`、`flow_trace.py`、`flow_diff.py`、`flow_intervention_eval.py`、`run.sh` 文件哈希逐一一致，故下列源码位置可由本地对应文件复查。

**新结果可用性：** 新服务器 `smolvla-official-reproduction-v2/` 下递归检查文件数为 0，虽有 action-effects、closed-loop 等目录，但没有 JSON、NPZ 或 checkpoint 文件。另查项目 outputs、gripper-mujoco-outputs、rollouts，只找到旧协议结果，未定位此次改进版 gate/confirmation 产物。不能由目录名确认新实验完成；可能是迁移未完整，原因未证实。本轮不重新连接旧服务器，不自动迁移或重训。

本地执行 `PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest tests/interaction_vla/representation_study/libero/test_acquisition.py -q -p no:cacheprovider`：10 passed。它是单元/合成检查，不替代真实数据复验，也未覆盖下述所有缺口。

### 9.1 应固定的科学目标

建议将主问题明确为：**有任务能力的冻结 VLA 中，是否存在承载任务相关物理交互信息的内部计算，这些计算是否被用于生成实际执行动作，并对闭环行为有可重复、情境相关的因果贡献？**

同谱系 5k→10k→25k 是发现候选与研究形成过程的一条比较轴，不应自动取代物理交互信息本身。当前变化方向的实验对象更准确地称为“候选子空间/计算方向”，而不是已经解释清楚的 SAE 特征、物理状态或预测状态。

| 问题 | 当前方法的作用 | 不能替代的证据 |
| --- | --- | --- |
| 物理交互信息是否可访问 | 冻结候选后的语义/几何关联与控制分析 | 不能单凭训练位移或动作 RMS 定义物理信息 |
| 信息是否参与动作生成 | 匹配条件下的候选干预、来源对照、执行段指标 | 不能将任意敏感方向都解释为自然策略使用的语义机制 |
| 是否对行为有因果贡献 | 匹配闭环干预，事先限定的行为后果与非目标损害 | 成功率下降本身也可能是非特异损伤 |
| 是否随训练形成 | 同一冻结候选在多个训练阶段的功能比较 | 只在 25k 干预不证明该方向造成 5k→10k 能力增长 |
| 是否为预测性交互状态 | 明确未来目标和信息预算的独立预测比较 | 前 10 步动作敏感性不等于状态预测或 PSR 充分性 |

从早期“跨任务预测性交互状态”转向“内部计算的功能/因果作用”，已经是实质性的范围调整，且符合此前用户对未来预测非必要条件的修订。需要在论文主张中显式选择：预测若不再是主检验，就不继续以“已恢复 predictive interaction state”总结结果。RET、SAE、flow、world model 是方法或比较轴，不是必须成功的研究目标。

### 9.2 用户列出的改动：实现状态与边界

| 改动 | 源码复核 | 必须保留的边界 |
| --- | --- | --- |
| 前 10 个动作主指标与分量报告 | 已实现 executed full/translation/rotation/gripper、planned full | 10 是当前部署合同，不是普适常数；7 维总 RMS 仍受分量尺度影响 |
| source_episode bootstrap | 按 suite/task/source_episode 聚合后重采样，已实现 | CI 对应 episode 等权，而部分展示均值仍按 state 等权；不同 episode 帧数不同时应统一 estimand |
| 最少 8 episodes | 默认 8，少于阈值不通过；参数可调 | 工程下限，不是功效保证；总数达标也不保证每任务有足够 episodes |
| stage/dose 匹配控制 | 离线 gate 已实现精确匹配；runner 规划 2 taps × 4 stage groups × 3 directions × 2 signs + baseline，即 49 条件 | gate 没有强制两种控制都存在，且仍将匹配控制平均；各只有一个 low-change/random 方向 |
| 状态/噪声/sigma 检查 | ID 对齐与 epsilon/sigma 一致性已实现 | 没有在 gate 内完整绑定 checkpoint、processor、runtime、候选版本及 tap |
| 单位范数、反对称性 | 已输出误差与残差诊断 | 未作为拒绝条件；`all_arrays_finite=True` 是报告常量，不能称为新增的实测检查 |
| 闭环 task-stratified paired bootstrap | 已实现 task 内重采样、task 等权汇总 | point estimate 仍为所有 pairs 的均值；cell 不等大时与 CI 目标不同；需另核对完整唯一配对 |
| 不覆盖 confirmation contract | 已拒绝覆盖同一路径，并检查提供的 used_plans 重叠 | 仅冻结 task/state cells，不包含完整模型/候选/剂量/阶段/分析/停止规则，也不能发现未提供的历史使用 |

对应源码：[离线 gate](../../interaction_vla/representation_study/libero/acquisition.py:489)、[确认合同与闭环](../../interaction_vla/representation_study/libero/acquisition.py:646)、[运行入口](../../run.sh:131)。

### 9.3 尚未解决的关键风险

1. **条件级 gate 被降为候选级许可。** `passed_candidate_ids` 合并所有通过行；闭环入口仅检查候选 ID，runner 固定执行 all/+1。这允许 early/−1 等条件的通过被用于未经对应验证的 all/+1。应绑定 checkpoint、候选文件 hash、tap、stage、dose、执行前缀与具体通过条件，而不是只传递方向名称。这是明确实现缺口，不是新增科学要求。
2. **“holdout”合并 validation 与 test。** 新 `flow_trace.select_records` 将二者并为 holdout，runner 用它作 gate 筛选。可以把这批数据正式登记为 development，但不能之后仍称其为最终 test。增加 episode 数不应隐含消耗最终确认集。
3. **确认集历史检查不完整。** runner 的 30–39 状态仅检查当前 timeline 和同 tap development 两份计划；旧 smoke 10–49 已评测过这组状态。它们可以是新模型的新测量，但不能称为整个研究从未查看的场景。确认合同需记录这一边界或使用真正保留的样本。
4. **工程诊断不能成为错误科学门槛。** 单位范数应有真实数值验证；正负加性隐藏干预的实现应正确，但非线性策略在有限 dose 下不必产生严格反对称动作。不能为了通过反对称性检查而排除真实非线性机制。应以小剂量局部响应、no-op 和匹配控制辅助解释。
5. **多重筛选与代理目标。** 两个 tap、多个 stage/sign 的 CI 中挑任一通过会引入选择效应。开发 gate 可以用于资源排序，但正式主张需预先指定对比或做适当多重比较处理/独立确认。超过平均随机动作效应只表示更敏感，不证明更重要、更物理或更有助于控制。

### 9.4 本轮建议

保留这些测量修正，但不要把研究目标改写为“让 gate passed”。先恢复/定位最新真实产物，修复条件级 gate 传递和确认合同绑定，再用固定候选回答一个具体的物理交互问题：在什么观测/任务情境下，该方向响应什么信息；干预后哪类实际动作与行为发生何种可重复变化；哪些匹配控制不能解释该变化。人类标签用作外部测量而非发现字典，负结果和无简洁语义的方向均保留。

当前最小主线应是 **物理信息关联与来源 → 实际动作功能 → 具体闭环后果**；训练阶段比较作为形成轴，未来预测作为单独可选问题。若用户坚持原始“预测状态”主张，则必须恢复预测证据，不能只凭动作干预完成该主张。本轮仅补审计记录，没有改代码、重设科学目标或启动实验。

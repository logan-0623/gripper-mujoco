# SmolVLA 交互能力获得中的内部计算变化与功能招募：实验设计

> 初稿：2026-09-10；最新目标与执行协议修订：2026-09-21（能力获得主轴；设计更新，非新增实测）
> 状态：实验设计；不授权启动训练、闭环评测或 RL  
> 当前有效协议：§0。顺序为具体能力变化 → 候选计算发现 → 跨训练阶段功能检验 → 物理解释与独立确认。训练形成是主轴，未来预测不是前置条件。
> 主要依据：[数学基础](../../research/vla-mathematical-foundations.md)、[当前研究协议](../../../research/predictive-states.md)、[项目状态](../../../ccfa.yaml)、[Observing and Controlling Features in VLA Models](../../../paper/2603.05487v1.pdf)

**主问题：当 SmolVLA 从不会某种交互行为到逐渐掌握该行为时，内部计算如何变化？相关信息是新形成的、被重新组织的，还是原本已经存在但后来才被动作生成系统功能性地利用？**

沿用 [Weekly Report](../../../Weekly%20Report.pdf) 的能力获得路线：先观察具体行为如何随训练变化，再解释其内部计算，不用单个成熟 checkpoint 的功能分析代替纵向研究。

阅读顺序：先读 §0；§1–11 是前期预测路线，§12 是形成该设计的历史方案、实现合同和结果记录。发生冲突时以 §0 为准；历史的“当前”“下一步”“待运行”均指当时，不作为新的启动指令。历史实测值不改写。最新证据边界见[服务器审计](../../research/smolvla-acquisition-server-audit.md)，预测路线见 [G2b 分析](../../research/g2b-results-analysis.md)。本次更新实验设计及本地审计代码；不更新 `ccfa.yaml` 或执行权限，也未启动新运行。

## 0. 当前有效研究协议（2026-09-21）

### 0.1 目标、范围与完成标准

主研究对象为同一训练谱系中能力不同的 SmolVLA checkpoints；各 checkpoint 在分析期间冻结，不等于训练期间 VLM 冻结。先在限定 LIBERO 任务/情境中研究一种具体交互能力及少量候选计算，不以方法排名、成功率超过官方、动作扰动最大或 gate passed 作为科学目标。所谓“物理信息”必须通过几何、接触关系、动力学相关条件或可操作的行为测量获得依据，不能仅由训练位移或动作 RMS 命名。

沿训练步 k 同时记录 **R_k（相关信息的可读性与组织方式）、U_k（对候选计算的功能依赖）、S_k（具体交互能力表现）**。R_k 与 S_k 同步不自动建立因果关系，U_k 需要干预及匹配控制。

| 竞争解释 | 需要的纵向证据 | 解释边界 |
| --- | --- | --- |
| 信息新形成或变得更可访问 | 相同输入、读出容量与数据预算下，可读性随能力增长；验证早期测量灵敏度 | 早期 probe 失败不证明信息不存在，优先表述为可访问性增长 |
| 已有信息重新组织 | 信息仍可读，但坐标、分布或下游访问方式改变；结合对齐与功能检验 | PCA 旋转、方差或相似度变化不单独证明形成新信息 |
| 已有信息被功能性招募 | 早期信息已可读，后期对相关计算的行为依赖增加，且一般敏感性不足以解释 | 只在 25k 改变动作不能解释能力增长；早期成功率触底时无下降不证明未使用 |

三种解释可以共存；压缩或混合编码变化也可作为结果，不要求归入唯一类别。第一阶段不要求证明信息严格地“从无到有”。

候选目前称为**方向/子空间**，不预称 SAE 语义特征、物理状态或预测状态。允许混合编码、无法简洁命名、可访问但不使用、作用只在部分情境成立。人工标签是外部测量工具，不是必须恢复的内部字典；无标签发现也不等于完全无先验。

第一阶段的主张限定为**任务相关的几何与交互信息如何参与动作生成**。目标相对位置属于几何关系，接触/持有/释放属于交互状态；摩擦、质量及其变化规律属于潜在动力学属性，不能由前两类结果直接推出。当前不主张发现一般物理模型或完整世界状态，也不新增动力学辨识实验。

无标签发现是本轮候选来源的选择，不是普适科学门槛。Contact、StableGrasp 等可用于外部测量和候选解释，不规定模型必须具有同名内部概念；若改用监督标签定位方向，须另记发现协议、选择数据与独立验证，不能用同一批样本同时挑选并确认。当前主问题是能力获得中的计算变化与功能作用，不要求未来预测通过；只有声明 predictive interaction state 时，才必须恢复独立未来预测与信息预算比较。

| 拟支持的主张 | 对应证据 | 单独不足的结果 |
| --- | --- | --- |
| 存在物理相关信息 | 冻结候选后的独立关联分析及时间、任务、机器人状态、动作等替代解释 | 训练差分大；样本内 R²高 |
| 信息来源与作用情境可定位 | 合理匹配的观察/条件对照，候选与局部计算响应 | 给任意输入加扰动后模型变化 |
| 候选参与实际动作生成 | 留出数据上，抑制或匹配替换自然候选分量的效应及必要控制 | 仅加性 steering 能改变动作 |
| 对限定行为有因果贡献 | 配对闭环的预先指定行为后果、success、非目标损害及匹配控制 | 动作 RMS 大；非特异性性能下降 |

上述是主张与证据的对应关系，不是要求每个候选全部通过的淘汰流水线。科学完成标准是：对预先限定的问题取得足以区分解释的证据，并完整报告正、负、混合或精度不足结果。无效或不确定候选不通过追加剂量、换任务或反复加样追逐正结果。

### 0.2 已核验证据与立即边界

以下五项为截至 2026-09-16 的历史审计快照，不代表最新服务器完整状态；本次文档修改未重新连接服务器。

- 旧服务器已核验新自训练谱系 5k–25k 的 200 条行为记录；四任务成功率为 12.5%、62.5%、75%、82.5%、95%。这是一个 seed、40 个配对初始条件上的探索性能力时间轴，不是全部 LIBERO 的成绩。
- 新谱系训练配置使用含 1,693 episodes 的数据，包含 Spatial 0–3；旧 smoke 的 task-3 未训练结论不适用于它。官方 checkpoint 是外部参照，不是该谱系终点。
- 旧服务器候选验证与干预只有 3 个独立 episodes。Middle 候选的全 50-step 效应优势在前 10-step 对 low-change 比较中未保持；不能沿用全计划指标的乐观解释。
- 固定点输入一致性已核验，但同一 25k 自然/固定点输出有非零差异；必须记录底噪及容差，不能宣称严格零误差。
- 2026-09-16 当时的新服务器相关实验树只有目录、没有文件；此项不代表后续服务器状态，不据此重新训练、迁移或覆盖结果。

此前对话中的后续只读审计还发现两项影响本轮设计的边界，原始产物位于服务器 `/root/autodl-tmp/smolvla-official-reproduction-v2/acquisition`：

- `closed_loop_middle_suppression_confirm_offset60_n8` 实际 `initial_state_id` 为 10–17，与已查看时间轴的 10–19 重叠。请求 offset 经环境取模后不代表新状态；该结果须按探索性复用解释，原始文件与运行记录保留。身份检查要求见 §0.6；当前代码已记录并校验实际状态身份，原始结果不改写。
- `full_25k_seed1000/checkpoints/025000/pretrained_model/config.json` 的 `train_expert_only=false`、`freeze_vision_encoder=false`，不能将该谱系预设为冻结 VLM 的 expert-only 训练。配置不代替逐模块权重变化核验；机制定位需区分上游条件变化与 expert 内部变化。

### 0.3 第一、二阶段：确定能力变化，再发现候选计算

先根据已有逐任务、逐 episode 的轨迹确定一种具体能力变化，目标行为暂记 TBD，不用总体成功率替代。保留失败、退步与任务异质性，不只选择后期成功的轨迹。优先复用 5k/10k/25k 三个比较点，但其“未掌握/过渡/掌握”角色由目标行为确定；15k/20k 用于核对轨迹。5k 不等于预训练初始化，官方权重是外部参照，不与自训练权重拼成同一训练序列。

候选发现并列保留两条通道：**表示本身发生变化**，以及**表示相对稳定但下游读取或动作敏感性发生变化**。5k→10k 最大差分不是唯一入口。优先复用现有 PCA/SVD 等组件；SAE、FFN 分析仅在能区分具体解释时加入，不要求同时运行所有方法。初步深查最多两个候选是资源建议，不是禁止探索其他线索；记录全部尝试、选择理由、文件 hash、tap、发现样本及历史使用。

**跨训练阶段的可比条件：**

- 使用相同 StateBank 样本、观测/语言、预处理语义、token 范围和对应 tap，记录 processor 差异。公共输入比较与各模型自然访问状态的比较分开，避免把状态分布变化误当计算变化。
- 固定 x_sigma/sigma 的局部比较，与相同初始噪声下各模型自然积分的比较分开：前者控制局部输入但可能偏离某阶段的自然分布，后者包含积分轨迹差异。两者不合并为同一效应。
- 跨阶段候选使用共享坐标或 discovery 数据拟合的对齐，报告误差、范数及激活尺度；同名 PCA/SAE 编号不自动代表同一特征。共享基底也不自动证明语义不变。
- R_k 分开报告“各阶段重新拟合、同容量同数据和调参预算的读出”与“冻结读出后的跨阶段迁移”。前者测可访问性，后者还受坐标变化影响；所有预处理与对齐只拟合于允许的训练/发现数据，按 episode 留出。

对冻结候选先做描述与替代解释比较：分别考虑任务身份、已观察时间、机器人状态、noisy action、语言/图像条件。最终计划动作可能是中介变量，不能将“控制最终动作后无增益”单独解释为信息没有被使用。样本内拟合结果与独立读出结果分开；解释关联不声称未来预测、互信息或因果作用。

需要定位信息来源时，选择**一个有区分力的来源对照**，不将完整来源归因设为所有候选的前置门禁：例如保持语言、机器人状态尽量匹配，并固定 x_sigma/sigma，比较不同目标相对位置的观测或对应观察路径条件；若首批证据指向 proprio，则研究该来源。donor 仅按已观察条件匹配，不能按未来事件、成功率或干预效应筛选。阈值由 discovery/validation 冻结，报告匹配覆盖和剩余不平衡；无有效 donor 时不强拼离分布输入。

在 development 阶段可据解释结果提出具体行为假设，但须在最终确认前冻结。例如“特定接近情境下影响夹爪启动时机”仅是可能的假设模板，不是当前发现。物理解释不成立而动作作用可重复时，保留为未解释功能方向，不强行命名。

**竞争解释与判别证据。** 以下是待检验解释，不是已发现的机制，也不要求分别启动三套实验；每个候选优先选择能区分其中两种解释的最小对照。

| 解释 | 应观察到的模式 | 判别对照与解释边界 |
| --- | --- | --- |
| 交互信息参与控制 | 候选随相关几何/交互条件变化，干预产生预定方向的行为变化 | 匹配可观察条件，比较相关与参照情境中的干预效应；仅有标签关联不够 |
| 主要是动作输出通道 | 候选强烈影响某动作分量，但缺少交互情境特异性 | 在可匹配的机器人状态和动作需求下比较不同交互情境；不以最终计划动作作为强制匹配变量阻断中介路径 |
| 一般敏感性或训练位移 | 影响广泛、损害不特异，匹配随机/低变化方向可产生相近结果 | 比较自然分量编辑、匹配控制及非目标损害；动作 RMS 大或训练差分大均不能单独区分 |

三种解释可以部分共存，不强行给混合编码分配唯一类别。每个候选在现有记录/confirmation contract 中补齐：`主要解释、最强替代解释、一个判别对照、主要行为量及预期方向、会削弱主要解释的结果、仍无法排除的混淆`。具体值由 development 证据确定，未确定时标 TBD，不能宣称可启动确认。若有效匹配不存在，报告覆盖不足，不以强拼输入制造区分力。

### 0.4 第三阶段：跨训练阶段检验自然计算功能

在选定的早期、过渡期和成熟期 checkpoints 上分别设置本模型 baseline、候选编辑与匹配控制。相同名义剂量不保证相同实际扰动，需记录自然分量大小、实际 hidden 扰动、相对激活尺度和编辑次数；剂量校准规则由 development 冻结，不按结果调到各阶段同样有害。随机/低变化控制在每个阶段核验，不能直接复用未经检查的控制向量。

U_k 比较候选相对本模型 baseline 的行为效应及其相对控制的差异如何随训练变化，不以动作 RMS 上升替代功能招募。早期 success 触底时，报告接近、接触、持有等可测进展和风险集；没有下降空间的指标不能证明早期未使用。R_k、U_k、S_k 的共同变化只能支持该谱系中的机制解释，不等于证明这种变化是唯一学习原因。

现有加性干预 h'=h+αd 保留为剂量和敏感性测量。对保留候选，再选一种与自然计算有关的检验：抑制自然存在的分量，或同模型内匹配 donor 替换。前者检验对移除的依赖，后者检验采用自然观测分量的定向替换；两者都需控制分布偏移和非特异损伤，不要求同时全量运行。donor 分量真实不保证与接收样本其余分量组合后的激活仍在分布内，抑制也可能同时移除混合编码的其他信息。因而单次敲除或替换不自动证明目标信息被自然策略使用；需结合信息关联、预定方向的作用、匹配控制和非目标损害共同解释。

沿用 no-op、同实际 hidden 扰动范数的随机方向与低变化方向；严格正交若被声明，须验证实际点积。控制必须匹配 tap、stage、dose/sign、token 范围及应用次数，分别展示控制类型，不只与平均控制比较。正式确认的随机面板沿用 §12.6 的至少八方向设计；探索可先用小面板，但不能声称已排除随机方向解释。阶段范围 3/4/3 次编辑不能直接排名“早中晚重要性”；记录即时 Δvelocity 与继续积分后的执行动作效应。

主要动作窗口为部署实际执行前缀：当前 n_action_steps=10，故用前 10 步；配置改变时由合同绑定，不静默复用常数。报告执行段 full、translation、rotation、gripper，first action 和完整 50-step plan 为辅助。跨量纲 full RMS 仅为汇总代理，不称物理能量；夹爪同时报告有意义的符号/阈值及时间变化，不用数值幅度直接等同抓取机制。

分开报告扰动强度、目标行为变化、情境特异性和非目标损害。候选 RMS 超过随机方向既非必要也非充分条件：随机方向可能造成更大的一般损伤，而较小候选效应可能改变关键时机。前 10 步是动作测量窗口，不是机制判据。Flow generation time 与环境时间仍按 §3.6 区分；阶段扫描只定位计算作用，不预设早期“规划”、晚期“修正”，也不据此声称环境动力学已被表示。

动作正负反对称性是诊断而非普适硬门槛。应验证加性编辑实现及小剂量局部响应，但有限剂量下的真实非线性不应被自动判错。范数、输入对齐、数组有限性必须实际计算；自然/固定点底噪与 hook/no-op 自一致性分别记录。

离线 gate 仅用于资源排序，不能取代科学结论。探索性闭环需要工程完整性、必要控制与有限预算，不要求先获得明确语义命名、显著 RMS 优势或理想行为方向；它也可以用于发现假设。正式确认才要求冻结行为假设、控制、精度与停止规则。当前代码中的保守门禁在另行修改前仍保留；本文调整不代表入口限制已修改或新运行已获授权。

### 0.5 第四阶段：物理解释与闭环确认

物理解释贯穿 discovery/development，可结合轨迹观察迭代，不必等因果 gate 通过后才开始。探索允许意外、负和混合结果；正式确认前再冻结主要情境、stage/dose 和预期行为变化，不直接运行大规模方向×剂量×任务网格。跨训练阶段使用相同实际初始条件，配对各模型基线、候选与必要控制，记录目标行为、success、非目标损害和终止/删失。候选导致适当情境下的选择性损害也可支持作用，不要求提高成功率。所有任务均退化且不具特异性时，优先解释为一般损伤。

若声明情境特异性，另预定一个有可比性的参照情境，比较“干预−基线”效应在两种情境间的差异，而不是以一组显著、另一组不显著作为差异证据。情境按干预前可观察条件定义，不按干预后的成功、接触或轨迹筛选；无法形成有效参照时，结论仅限被测情境。主要行为量的单位、事件定义、风险集/分母和观察窗口由候选假设确定并冻结，success 为共同报告项，不代替具体行为证据。

闭环动作比较包含轨迹分化后的反馈效应，不能称为固定 observation 下的纯局部效应。固定输入功能测量与闭环总行为后果分表报告。动作来源与环境 step 对齐；初始状态与策略噪声配对，重复噪声不增加独立场景数量。

恢复原始激活后的输出恢复属于工程自一致性检查，不单独称为机制 rescue。跨 checkpoint 激活替换还需要坐标、尺度和分布兼容性证据，不是第一阶段必选实验。

报告范围与证据一致：同任务新 episodes 是 episode 泛化；跨任务结论需预定任务级留出，研究分析未见不等于 policy 训练未见。单谱系可以研究该谱系的能力获得过程，一般性的能力形成主张需要独立谱系证据；更多评测 episodes 不等于更多训练 seeds。

### 0.6 数据、统计与完整确认合同

明确分开 discovery、development 与 confirmation。使用过的 validation/test 可以重新登记为 development，但不能重新命名后获得独立性。确认集记录完整历史使用；旧 smoke 已查看的状态不宣称全研究未见。此前无意消耗的留出数据如实披露，并另留确认样本。

样本独立性按**实际 suite/task/initial_state_id、状态库版本和可用的状态内容 hash**核验，不按请求 offset 区间判断。执行前解析映射，执行后核对记录与合同，检查取模回绕。已重叠的运行保留原文件，在分析中标注探索性复用；不能仅增大 offset 就称为新确认集。真正未查看的样本来源与数量暂记 TBD。

本轮建议统一目标量为 task 等权、task 内 episode 等权、episode 内 state 等权；先在每个 state 内汇总噪声重复及规定动作窗口。点估计与 CI 使用同一权重。离线在 task 内按 source_episode 重采样；闭环在 task 内按配对初始条件重采样并计算 task-macro 差值。报告各 task 和各 episode，不将帧当独立单位。与旧 state 加权估计量不同的结果另标协议，不拼接同一比较。

最少 8 episodes 是现有代码的工程下限，不是充分样本数，也不是科学上禁止小样本探索的标准。逐任务最低覆盖、目标效应/精度、最大资源与停止规则须在确认结果揭示前确定；用已有 development 估计方差，不凭本文件指定功效。多个 checkpoint、tap、阶段和符号属于多重选择：development 可筛选，confirmation 冻结主对比并按预定 family 处理多重比较。跨 checkpoint 效应差采用共享配对单位重采样，保留阶段间配对，不以两个独立 CI 是否重叠判断差异。

**负结果与精度规则。** 在确认前按主要行为量的实际含义冻结最小有意义效应 δ、区间水平及判定方法，不能依据确认结果反推阈值。八个随机方向是控制面板要求，也不是功效保证；面板与 episode 数服务于预定主对比及精度预算。

| 结果模式 | 允许的结论 |
| --- | --- |
| 方向和行为后果符合预定假设，效应区间支持主对比，匹配控制与一般损伤不足以解释 | 支持该模型、候选与被测情境下的功能作用；不自动推广为通用物理机制 |
| 干预落实且测量有足够灵敏度，预定区间完整落在 [−δ, δ] 内 | 支持该操作及条件下效应小于预定有意义范围；不等于信息不存在 |
| 区间覆盖有意义的正/负效应，或编辑效力、测量、匹配控制不可靠 | 精度或可识别性不足，不能以不显著宣称无作用 |
| 任务/情境效应相反，或行为变化与原解释不符 | 报告异质性或削弱原解释，不以总体均值或事后重命名挽救假设 |

解释空结果前先核对自然分量是否实际改变、行为量是否存在足够变异及已有正控制/测量验证是否覆盖该路径；无需为每个空结果新增一套训练。单一线性候选失效不能否定模型全部相关信息，但达到冻结预算后可结束为“当前候选未获支持”，不无限换坐标、方法或样本追逐正结果。

不可覆盖的 confirmation contract 应绑定：checkpoint/processor/候选 hash、tap、具体方向、stage、dose/sign、编辑 token、执行前缀、任务与样本清单、噪声规则、主要/次要指标、控制、估计量/CI、比较 family、预算与停止规则、历史使用清单。闭环入口必须消费**具体通过条件**，不能只消费 candidate ID；通过 early/−1 不授权 all/+1。文件不可覆盖只是可追溯性措施，不等于已经完成预注册或独立性审计。

本次新增的竞争解释、行为定义、参照情境（若声明特异性）、δ 和结果解释规则纳入同一合同，不另建流程系统。已冻结合同不覆盖；这些字段尚未填定或代码未支持时，记录为执行前缺项，不把本次文档优化当作实现完成。

纵向确认合同另绑定训练谱系、checkpoint 清单、实际训练冻结配置、候选跨阶段对应规则、读出重拟合/迁移协议、剂量校准规则、实际状态身份及跨阶段主对比。TBD 阻止正式确认，不阻止已获授权且有有限预算的探索。

### 0.7 当前交付、扩展与非目标

| 顺序 | 本轮交付 | 边界 |
| --- | --- | --- |
| 1 | 根据已有轨迹确定一种具体能力变化，核验谱系与实际初始状态 | 行为及阶段角色待证据确定，不自动把 5k→10k 命名为抓取涌现 |
| 2 | 从表示变化、下游读取变化两条通道发现少量候选，建立跨阶段对应 | 不只选最大训练差分，不用同批数据选方向并确认 |
| 3 | 在所选 checkpoints 上比较 R_k、U_k、S_k，配对自然分量编辑与控制 | 不只在 25k 干预；保证输入、预算、尺度与测量条件可比 |
| 4 | 结合轨迹解释物理意义，必要时加入来源/情境对照 | 不强制语义命名，不拼接不同 tap/候选为同一机制 |
| 5 | 补齐身份核验与合同后，在实际未查看样本上确认主对比 | 新执行仍需授权；正、负、混合及精度不足结果均交付 |

训练形成是本轮主轴，不再等成熟模型的功能问题全部解决后才追加。优先复用已有 5k–25k 权重；已有行为记录属于探索证据。更早 checkpoint 或更多训练谱系是否必要，由区分竞争解释的需要决定，不自动启动新训练。

未来预测轴：只有主张明确包含 predictive state 时，才恢复独立未来目标和信息预算比较；G2b 结果可保留，预测失败不自动否定控制作用。SAE/RET 对比、压缩、IB、OOD、world model、第二 VLA 和 RL 均为条件性扩展，不是第一阶段的必选项。已有官方参照足以支持限定研究对象的能力判断；是否扩大官方评测由精度需求决定，不再机械作为所有机制实验的前置工作。

完整 VLM→expert 路径归因与具身 Harness 为条件性扩展，仅在需要定位功能招募发生在哪条路径时加入；当前谱系不能预设上游冻结。

**本轮优化的核心：围绕能力变化，区分信息可访问性增长、重新组织与功能性招募；成熟模型的动作效应只是一段证据，不是终点。** 历史结果保持原样；本节中的新增比较、合同修正与采样安排是计划，不是已实现或已批准运行。

## 1. 前期预测路线的科学问题（历史记录）

在匹配历史、机器人状态、时间和策略动作计划后，哪一种 VLA 表示仍保留了对未来物理交互转变有用的信息；这些信息是否被策略实际用于生成动作，并能否在闭环中改变真实物理事件与任务成功率？

这项研究把四个结论层级分开：

1. **Accessible**：变量能否从表示中读出。
2. **Predictive**：表示是否在已知上下文后继续降低未来事件预测风险。
3. **Functionally used**：对表示做匹配干预是否改变策略动作。
4. **Closed-loop useful**：动作变化是否进一步改变物理轨迹、约束满足率或任务成功率。

任何较弱层级都不能替代较强层级。例如，高 probe accuracy 不能证明策略使用该信息；离线 action delta 也不能证明任务成功率提高。

## 2. 当前证据及其含义

### 2.1 已经成立的项目事实

| 证据 | 当前结果 | 对下一步的约束 |
| --- | --- | --- |
| StateBank | 20 tasks、100 episodes、13,603 states；selected replay acceptance = 1.0 | 继续使用固定 state/episode ID，不能为某一方法重采样更容易的状态 |
| SmolVLA longitudinal grid | 8 checkpoints × 4 taps × 6 factors；384 cells 中 288 complete、96 not estimable | 上游 vision 在 expert-only SFT 中冻结；训练演化结论应集中在 downstream/action pathway |
| 线性因子干预 | StableGrasp 可读，但所有 checkpoint 的 rank-one action effect 均未超过 matched random | 该 pooled rank-one intervention 未获支持；不能推广为所有线性子空间均未被使用，LEACE 擦除也不等价于该操作 |
| 官方 SmolVLA positive control | LIBERO Spatial task 0：9/10 success；Contact/StableGrasp rank-one erasure 的 action change 小于 random | 失败不能归因于完全失效的 policy；应转向非线性或稀疏特征，同时保留匹配控制 |
| SAE pilot | 3 个 SAE seed；seed 0 的 feature 694/981 产生显著 action delta；尚未跨 seed 独立复制 | 当前唯一允许推进的主门禁是跨 seed、orthogonal random、matched atom 复制 |
| 完整 token cache | 13,603 states；shape `[50, 480]`；固定 CPU noise seed；MPS 提取完成 | 可立即做便宜的离线条件预测，但该 tap 与 SAE pilot 的 720-D tap 不相同 |

### 2.2 SAE pilot 的保守解释

| Feature | 观察到的性质 | 当前允许的解释 |
| --- | --- | --- |
| 694 | Contact AUPRC 0.6491；StableGrasp AUPRC 0.5432；seed-0 target-minus-random first-action L2 = 0.2293，episode CI [0.1854, 0.2760] | 候选 contact/grasp/action feature；作用主要落在连续 gripper 分量，尚未证明是物理抽象 |
| 981 | Contact AUPRC 0.5836（负方向）；StableGrasp AUPRC 0.5012（负方向）；place/release-retreat 激活显著较高；target-minus-random first-action L2 = 0.3600，episode CI [0.3214, 0.3948] | 更像 phase 或 gripper motor feature；尚未证明具有跨 dictionary 稳定性 |

这些数值来自单一 reference dictionary 和单一随机方向，只能作为 pilot。进入闭环前，必须证明作用不是 dictionary 坐标偶然性、随机方向选择或单一 action component 尺度造成的。

### 2.3 当前未来预测结果暴露的核心问题

现有 480-D token-tap 探索中，StableGrasp 在全部样本上的未来预测有正增益：

| Target | Horizon | Action-only Brier | Hidden+Action Brier | 相对 Action-only 增益 | Changed subset 增益 |
| --- | ---: | ---: | ---: | ---: | ---: |
| StableGrasp | 5 | 0.0780 | 0.0636 | +0.0144 | -0.0356 |
| StableGrasp | 10 | 0.0703 | 0.0657 | +0.0046 | -0.0174 |
| Contact | 5 | 0.0572 | 0.0564 | +0.0008 | -0.0405 |
| Contact | 10 | 0.0636 | 0.0703 | -0.0067 | -0.0213 |

本表保留为 G2 之前的探索快照，不代替 §11 的完整 G2 结果。总体与 changed 分数的差异提示 persistence、读出几何及子集选择均需区分；changed 按未来结果筛选，不能作为 prospective event forecasting 的主要分数。后续主要风险必须在仅按当前状态定义、包含未来事件阳性与阴性的 risk set 上计算。

### 2.4 两个 tap 必须明确区分

- **720-D `action_expert_input`**：longitudinal grid 与 SAE pilot 的主 tap；适合连接表示、SAE feature 与 action causality。
- **480-D `action_atlas/expert/31/mlp/output` token tap**：已有完整 `[50,480]` cache；适合立刻做 Flow-Matching token 与条件预测诊断。

在相同 tap、相同 forward trace 和相同 noise realization 上比较前，不能把 480-D 的预测结果与 720-D 的 SAE 因果结果合并成“同一个表示同时 predictive 且 causal”。

## 3. 数学操作化

### 3.1 条件预测，而不是无条件 probe

令 $Z_t$ 为候选表示，$Y_{t+k}$ 为未来 Contact、StableGrasp 或 Phase 事件，$C_t$ 为允许的上下文。目标量是：

\[
I(Y_{t+k}; Z_t \mid C_t).
\]

实际实验使用有限 readout family，因此报告：

\[
\Delta_{\mathcal V}^{\mathrm{Brier}}
=R_{\mathcal V}(Y_{t+k}\mid C_t)
-R_{\mathcal V}(Y_{t+k}\mid C_t,Z_t).
\]

它是给定 probe family 下的条件预测增益，不直接称为 Shannon mutual information。Brier 风险的 Bayes 改善对应条件均值差的平方，因此适合检验表示是否在基线条件之后增加可预测信息。

上下文依次为：

- $A$：当前 policy 预测的 action chunk；
- $C$：elapsed time + 四帧 proprioceptive history；
- $AC$：二者联合；
- $ACY$：再加入当前目标标签，作为排除 persistence shortcut 的严格条件。

`ACY` 的未来增益回答：“已知当前是否 contact/grasp 后，表示还能否预测下一次转变？”

### 3.2 转变事件目标

除 endpoint label $Y_{t+k}$ 外，增加窗口内首次事件：

\[
E^+_{t,K}=\mathbf 1\{Y_t=0,\exists j\in[1,K]:Y_{t+j}=1\},
\]

\[
E^-_{t,K}=\mathbf 1\{Y_t=1,\exists j\in[1,K]:Y_{t+j}=0\}.
\]

分别对应 onset 与 offset。这样不会漏掉在 $t+1:t+K-1$ 中出现、但到 endpoint 已恢复的短暂事件。

### 3.3 动态表示的判据

一个 action-conditioned predictive state $S_t=\phi(H_t)$ 的理想目标是：

\[
p(F_t\mid H_t, A_{t:t+K-1})
\approx
p(F_t\mid S_t, A_{t:t+K-1}),
\]

其中 $H_t$ 是观测历史，$F_t$ 是未来 interaction events。实验不尝试直接估计连续变量互信息，而用三个可检验条件近似这个主张：

1. **Future sufficiency**：在匹配上下文下，$S_t$ 降低未来事件风险。
2. **Action sensitivity**：加入 action condition 后预测改善；否则表示可能只编码时间或被动状态。
3. **Microstate residual**：加入更完整的已观测历史后是否仍改善预测。无增益只表示当前读出家族未检出 residual，不能证明充分性；本项目激活历史不等于完整环境历史。

### 3.4 信息瓶颈只作为容量控制

不直接优化或报告未经证明的 $I(Z;H)$。对随机表示、PCA、AE、RET 使用固定维数、历史长度、训练数据和 tuning budget；对原生 TopK SAE 单独报告 dictionary width、active count、reconstruction explained variance。若使用 KL-to-prior，它只被描述为 compression surrogate，而不是互信息的精确值。

### 3.5 对比学习是条件消融，不是默认组件

RET 主实验先使用最小的 action-conditioned predictive loss。只有该路线通过基本门禁后，才比较 CPC/InfoNCE：正样本必须来自真实未来，负样本在 task、phase 或当前标签等条件内采样，避免模型仅靠 task identity 或时间识别正负对。随机 batch negatives 不足以支持“学习到了物理动态”的结论。

### 3.6 Flow Matching 的两个时间变量

SmolVLA action head 中的 generation time $\sigma$ 与环境时间 $t$ 不同。模型学习的是条件 action distribution 或 velocity field：

\[
v_\theta(a_\sigma,\sigma\mid c_t),
\]

而项目要研究的是：

\[
p(s_{t+1},Y_{t+1}\mid s_t,a_t).
\]

因此 50 个 denoising/flow tokens 的层内结构只能解释 action generation computation，不能直接称为 environment dynamics。物理预测必须跨环境 step 检验。

## 4. 假设与可证伪结果

> 历史预测路线的假设表；当前竞争解释和结果判读以 §0.3、§0.6 为准，不将本表作为本轮确认合同。

| ID | 假设 | 支持证据 | 反证或空结果 |
| --- | --- | --- | --- |
| H1 | 当前物理变量可读，但这种可读性不必对应策略实际使用 | probe 高、线性干预弱 | matched intervention 稳定改变动作与闭环事件 |
| H2 | Feature 694/981 的 action effect 是跨 SAE seed 的稳定 feature family | 两个独立 seed 的匹配 atom 均超过 random 与 non-candidate controls | 任一 family 仅在 seed 0 成立或只由 gripper 尺度解释 |
| H3 | Action-conditioned dynamic representation 比同容量静态表示更能预测 interaction transitions | onset/offset 条件风险下降，且不是 unchanged subset 驱动 | 只改善总体 Brier；changed/onset/offset 无改善 |
| H4 | 预测有用与控制有用是两个独立轴 | 同一 feature/representation 同时通过预测与干预测试 | 只通过其中一个轴；仍是有意义的分离结论 |
| H5 | 可干预 interaction variable 能在闭环改变真实物理事件 | 实际 Contact/StableGrasp、drop rate 或 success 的 paired 改善 | observer 输出改变但 simulator 事件不变，说明只控制了 readout |

不得把某一种方法必须胜出写入成功标准。Raw hidden、SAE、RET 任一者可能只在其中一个轴上占优；公平的 null result 仍能回答表示类型是否适合物理交互。

## 5. 实验执行顺序与门禁

> 历史 G0–G6 协议。是否启用以 §0 为准，SAE/RET、预测与旧闭环门禁不自动成为当前功能研究的必经步骤；代码权限仍须单独核验。

### G0：冻结证据与协议

**目标**：保证所有后续方法使用同一数据、split、forward trace 和统计单位。

执行项：

1. 固定 StateBank state/episode/task IDs、label 版本和 hash。
2. 分别记录 720-D 与 480-D tap 名称、checkpoint、inference noise seed、token aggregation。
3. 训练/validation/test 的 preprocessing 只在训练集拟合；超参数只在 validation 选。
4. 统计重采样单位为 episode；task-level macro 为主汇总，frame 不能作为独立样本计算显著性。
5. 已查看过的当前 test 结果全部标为 exploratory；真正 confirmatory 结果需要预先冻结的新 task/episode 集。

**通过条件**：manifest 与 hashes 可复现，且各方法不存在不同 state coverage。

### G1：完成当前 SAE 跨 seed 复制门禁

**这是当前唯一必须优先完成的因果实验。**

固定 feature family：

| Reference | Seed-matched atoms |
| --- | --- |
| seed 0 feature 694 | `[694, 1372, 795]` |
| seed 0 feature 981 | `[981, 828, 997]` |

协议：

1. 不再根据 action outcome 修改匹配 atom。
2. 对 seed 1、seed 2 的对应 atom 使用与 pilot 相同的 state IDs、forward/noise trace 和干预幅度定义。
3. 每个 target 使用至少 16 个与 decoder direction 正交、逐状态同范数的随机方向；报告随机分布，不只报告一个随机种子。
4. 为每个 target 预先从 validation 选择 matched non-candidate atoms，匹配 activation rate、feature contribution norm 与 decoder norm；选择过程不能查看 action effect。
5. 主指标保持与 pilot 一致：first-action L2 的 `target - control`；同时分解 xyz、rotation、gripper 分量。
6. episode-cluster bootstrap；两 families×两 independent seeds×两 control classes 共 8 个基本对比组成固定 BH family，不遗漏第二类 control。

每个 feature family 的通过条件：

- seed 1 与 seed 2 的 matched atom 均呈同方向效应；
- 相对 orthogonal-random distribution 与 matched non-candidate 的 episode-cluster CI 均排除 0；
- 结果不能只由少数 episode 或单一异常 action magnitude 产生。

若某个 family 未通过，标记为“未获得独立复制支持”，不进入 interaction-state 闭环；dictionary-specific 只是候选解释。缺数据或区间过宽与明确反向结果分开报告。若两个 family 均明确失败，停止这两个 family 的闭环路线并报告负结果。

### G2：运行已有 480-D 条件风险分析

现有 [`conditional_readouts.py`](../../../interaction_vla/representation_study/libero/conditional_readouts.py) 已实现：

- hidden/PCA × mean/first token × current/history；
- `A/C/AC/ACY` contexts；
- horizons `1/5/10`；
- `all/changed/unchanged/onset/offset` subsets；
- partition permutation、same-task episode swap nulls；
- 3 个 null seeds。

2026-09-11 已核对报告：complete=true，measured_fits=fits=1,368。保存原结果，不重复运行。该分析只使用已有 cache，不替代 G1。

**修正初稿的实现描述**：当前 onset/offset 只是当前与终点标签分别为 0→1 / 1→0 的子集，尚未实现窗口内首次事件。这些子集只含相应端点结果，不能据此报告事件 AUPRC 或宣称 prospective forecasting 已完成。真正的窗口事件采用 §11 G2b；复用数据和读出组件，但使用新的 protocol binding 与输出目录。

主报告：

- task-macro Brier 与 (Delta_{\mathcal V}^{\mathrm{Brier}})；
- changed/onset/offset 的 endpoint Brier，仅作为结果条件化诊断；
- aligned 表示相对 permutation/episode-swap null 的差异；
- per-task/per-episode 风险；当前报告没有 CI，追加不确定性分析不能将已查看的结果变为确认性证据。

此阶段结果仍属于 480-D secondary tap 且当前 test 已被查看，因此用于诊断和确定正式协议，不用于最终 confirmatory novelty claim。

### G3：在同一 720-D tap 上比较 Raw、SAE 与 dynamic representation

这是论文的核心离线实验。重新提取或补齐 720-D `action_expert_input` 的固定 cache，使 prediction 与 causal intervention 使用同一 activation source。

#### 表示组

| 表示 | 作用 | 容量公平性 |
| --- | --- | --- |
| Raw hidden | 未压缩的线性可读基线 | 报告原维度；有限 Ridge 性能不是其他读出器的上界 |
| PCA-32 | 固定线性压缩 | train-only fit |
| Gaussian random projection-32 | 随机几何控制 | 固定 seeds |
| Causal temporal average-32 | 最小历史平滑控制 | 只使用 (\le t) 信息 |
| AE-32 | 与 RET 共享 encoder 容量的静态重构基线 | 相同 architecture、data、steps、tuning budget |
| SAE-1440, TopK-32 | 已有稀疏解释性基线 | 单独报告 width、TopK、EV，不伪称与 32-D dense 等容量 |
| RET-32 | action-free predictive baseline | 与 AE 匹配训练预算 |
| action-conditioned RET-32 | 主要动态表示 | 与 RET 仅差 action condition |

所有使用历史的方法固定 (W=4)。每种学习方法至少 3 个 training seeds。超参数由 validation 选择，test 只运行一次。

#### Action 条件

- 一步预测使用实际执行的当前 action。
- 多步在线可用条件使用 policy 当前输出的 action chunk，并明确称为 policy intention。
- 真实未来 demonstration actions 只作为 privileged upper-information condition，不能与可部署方法并列宣称在线可用。

#### 指标

1. **Encoding**：当前 Contact、StableGrasp、Phase 的 task-macro Brier/AUPRC。
2. **Transition prediction**：(E^+_{t,K})、(E^-_{t,K}) 在 `AC` 与 `ACY` 条件下的风险增益。
3. **Latent closure**：未来 latent 的 normalized MSE、explained variance；只作为机制诊断。
4. **Action-conditioning gain**：action-conditioned RET 相对 action-free RET。
5. **Anti-collapse**：per-dimension variance、effective rank、constant-predictor gap。
6. **Microstate residual**：比较 $R(C,Z)$ 与 $R(C,Z,H)$。完整历史仍大幅改善则表示不充分。

#### Dynamic representation 通过条件

在预注册、包含两类结局的 prospective risk set 上，按 §11 冻结的多重比较规则检验：

- 相对 PCA-32 和 AE-32 的 task/episode-cluster CI 排除 0；
- 改善存在于 prospective onset/offset 风险；changed-only 改善不足以通过；
- anti-collapse 通过；
- action-conditioning gain 或 microstate residual 至少有一个支持“动态组织”解释。

若 RET 只改善 endpoint 总体 Brier，则仅报告 endpoint prediction 增益及 persistence 这一候选解释，不据此认定其机制。若未胜过静态基线，保留 null result，不自动追加复杂 loss。

#### 可选的对比学习消融

只有 action-conditioned RET 通过上述门禁后，才增加同容量 CPC/InfoNCE 版本。负样本必须在相同 task、相近 elapsed time 和相同当前标签内采样；训练预算与 RET 相同。它回答“contrastive geometry 是否进一步改善 transition separation”，不作为主方法成立的必要条件。

### G4：复现 2603.05487 的 observer-controller 作为直接基线

最新论文给出的线性 observer 为：

\[
\hat\zeta=f_l(x)=W_lx+b_l.
\]

当标量目标范围为 ([a,b]) 时，最小范数控制沿 (W_l) 方向投影：

\[
u^*=\begin{cases}
\frac{a-\hat\zeta}{\|W_l\|^2}W_l,&\hat\zeta<a,\\
0,&a\le\hat\zeta\le b,\\
\frac{b-\hat\zeta}{\|W_l\|^2}W_l,&\hat\zeta>b.
\end{cases}
\]

先在 SmolVLA 上适配论文的低层 feature：gripper open/closed、EE height、EE speed。这个实验验证 observer/controller 工程链路，并为 interaction feature intervention 提供直接基线。

固定比较：

1. no intervention；
2. language prompt，若目标可用语言表达；
3. paper-style observer controller；
4. same-norm matched random intervention。

layer/tap 选择只能用 validation；冻结后再评测。参照论文规模可使用 10 个 LIBERO Spatial tasks × 10 paired rollout seeds/condition 作为起点，但正式 episode 数应由 pilot variance/power analysis 决定，不能把 100 rollouts 自动视为充分。

必须分别报告：

- **observer compliance**：干预后 probe 输出是否落入目标范围；
- **physical compliance**：simulator 中实际 gripper/height/speed 是否满足约束；
- task success、paired success change；
- intervention norm、runtime 与 calibration/OOD error。

线性 observer 自动满足：

\[
\|f(x+\delta)-f(x)\|\le\|W\|\|\delta\|.
\]

因此局部 readout robustness 本身证据较弱；真正关键的是 held-out physical calibration。最小范数投影只保证 observer 输出改变，不保证真实机器人变量改变。

### G5：interaction-state 闭环干预

只有通过 G1 的 SAE family 或通过 G3 且具有有效 activation lift 的 dynamic variable 可以进入此阶段。

#### 条件

1. no intervention；
2. 2603.05487-style linear factor controller；
3. replicated SAE feature suppression/enhancement；
4. orthogonal random directions，报告完整分布；
5. matched non-candidate atoms；
6. sign-reversed intervention；
7. language prompt，仅在语义可表达时使用。

#### 干预时机

物理变量具有 phase dependence，不能在整条轨迹上无条件提高 contact。分两层：

- **Oracle phase gate**：使用真值 phase 窗口，只用于隔离机制；明确标为 privileged。
- **Predicted gate**：使用在线 phase/contact observer 触发，作为可部署的 end-to-end 条件。

优先检验三个具体目标：

- lift/transport 中维持 StableGrasp；
- approach-to-contact 中改变 Contact onset，而不提前闭合 gripper；
- place 后促进 release，同时不增加 pre-place drop。

#### 配对闭环协议

- 各 condition 使用相同 simulator initial state、task、environment seed 和 inference noise。
- task 是最高层汇总单位，episode 是 bootstrap cluster；禁止用 frame 数制造伪显著性。
- 先在 policy baseline 成功率足够的任务上评估机制，再在完整任务集报告外部有效性。
- 以 10 tasks × 10 paired seeds/condition 作为与最新论文可比的最低起点，并根据 pilot 方差扩展。

主结果：真实 Contact/StableGrasp constraint satisfaction、onset time、drop rate、task success、paired (Delta)success、action deviation、intervention norm。若 probe compliance 提高而 physical compliance 不变，结论是 observer hacking，不是 causal control。

#### RET 干预边界

RET latent 没有像 SAE decoder 那样天然的 activation-space inverse。除非另行训练并验证 lift (L:S_t\rightarrow h_t)，包括 round-trip error、off-manifold detection 与 matched-norm controls，否则 RET 只参与预测比较，不参与 activation steering。不能直接把 latent direction 加到 VLA hidden state。

### G6：独立确认与扩展

当前 2 个 held-out tasks 的结果已被查看，只能作为 exploratory。最终主要结论必须在分析前冻结的新 tasks/episodes 上确认。优先级如下：

1. 从尚未用于模型选择的 LIBERO tasks/episodes 构造 untouched confirmatory split；
2. 预注册主要 target、horizon、feature families、统计检验和停止规则；
3. 中心结果成立后，再考虑第二个 VLA architecture；
4. world-model 对照仅在 predictive state 通过 G3 后增加；
5. RL extension 保持 frozen，直到 SFT/representation 主问题得到明确结论。

## 6. 统计与公平性

### 6.1 共同数据访问

所有表示必须共享：

- task/episode/state split；
- history length 与允许的 action information；
- label construction 与 horizon；
- preprocessing fit 范围；
- probe family 与 hyperparameter search budget；
- training seeds 和 evaluation units。

Raw hidden 保留原维度，但有限 Ridge 风险不构成其他表示的性能上界；不能把性能差异直接归因为表示质量。PCA/AE/RET/RP-32 构成主要输出维度匹配组，训练容量另行报告。SAE 的稀疏容量单独呈现。

### 6.2 主要统计单位

- G2 现有 task macro 是每个 task 内按有效窗口计算 Brier，再等权平均 task；不是 episode 等权。G2b 保留这一主估计量，另报 episode 等权敏感性，不能混用两种数值。
- CI 使用 hierarchical bootstrap：先采样 task，再在 task 内采样 episode。
- 配对干预保持相同初始状态与随机性，报告 paired difference。
- 多个 feature/horizon 的正式检验使用预先定义 family 和 BH correction。
- 不用 frame-level iid standard error。

### 6.3 负控制

| 控制 | 排除的解释 |
| --- | --- |
| time/proprio/action/current-label contexts | task clock、机器人状态、策略计划、label persistence |
| partition permutation | 表示与目标的对应关系可以被打乱而结果不变 |
| same-task episode swap | task identity 或平均阶段足以解释结果 |
| random projection | 低维化本身带来的 regularization |
| matched AE | reconstruction/architecture capacity，而非 dynamics objective |
| action-free RET | history prediction，而非 controlled dynamics |
| orthogonal random intervention | 任意同范数扰动都改变动作 |
| matched non-candidate SAE atom | 稀疏 feature 的一般尺度/频率效应 |
| sign reversal | 方向性与剂量关系 |

## 7. 结果表模板

### 7.1 SAE 独立复制

| Family | Seed | Atom | Target effect | Orthogonal-random effect | Matched-atom effect | Episode CI | q | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| 694 | 1 | 1372 | TBD | TBD | TBD | TBD | TBD | TBD |
| 694 | 2 | 795 | TBD | TBD | TBD | TBD | TBD | TBD |
| 981 | 1 | 828 | TBD | TBD | TBD | TBD | TBD | TBD |
| 981 | 2 | 997 | TBD | TBD | TBD | TBD | TBD | TBD |

### 7.2 同 tap 表示比较

| Representation | History | Action condition | Dim/capacity | Current Brier | Onset Brier | Offset Brier | ΔBrier vs ACY | Latent EV | Effective rank | Status |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Raw hidden | 4 | no/yes | 720 | TBD | TBD | TBD | TBD | N/A | TBD | TBD |
| PCA | 4 | no/yes | 32 | TBD | TBD | TBD | TBD | N/A | TBD | TBD |
| Random projection | 4 | no/yes | 32 | TBD | TBD | TBD | TBD | N/A | TBD | TBD |
| AE | 4 | no | 32 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| SAE | 4 | no | 1440, TopK 32 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| RET | 4 | no | 32 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Action-conditioned RET | 4 | yes | 32 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### 7.3 闭环结果

| Condition | Observer compliance | Physical compliance | Contact onset | Drop rate | Success | Paired Δsuccess | Intervention norm | Runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No intervention | TBD | TBD | TBD | TBD | TBD | 0 | 0 | TBD |
| Prompt | TBD | TBD | TBD | TBD | TBD | TBD | 0 | TBD |
| Linear observer controller | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Replicated SAE feature | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Orthogonal random | TBD | TBD | TBD | TBD | TBD | TBD | matched | TBD |
| Matched non-candidate atom | TBD | TBD | TBD | TBD | TBD | TBD | matched | TBD |

## 8. 论文定位：与 2603.05487 的关系

| 维度 | 2603.05487 | 本项目应增加的证据 |
| --- | --- | --- |
| Feature | gripper、EE pose/speed 等低层连续或二值变量 | Contact、StableGrasp、Phase、onset/offset、failure/recovery |
| Representation | transformer activation + linear observer | Raw、PCA/AE、SAE、action-conditioned dynamic state |
| Prediction | 当前 robot state/action observability | 在 time/proprio/action/current-label 后的未来交互转变 |
| Control | 最小范数 observer controller | matched random/non-candidate、跨 seed SAE replication、phase-gated intervention |
| Closed loop | constraint following 与 task success | observer compliance 和 simulator physical compliance 分离；drop/success 的 paired effect |
| Flow head | 论文未分析 diffusion/flow action heads | 区分 generation time σ 与 environment time t，分析 SmolVLA flow-token computation |

这篇论文应作为直接 baseline，而不是本项目的新颖性来源。本项目的新增问题是：**可解释 feature 是否构成对未来物理交互充分、受 action 调制、被策略使用且能改变真实闭环事件的状态变量。**

## 9. 最小执行计划

按成本和证据依赖排序：

1. **立即、低成本**：G2 已完成，按 §11 补齐窗口内 onset/offset 的 G2b 设计与实现；不覆盖 G2。
2. **当前主门禁**：完成 feature 694/981 在 seed 1/2 上的 matched replication；未通过则停止对应 SAE family。
3. **条件性新增提取**：G2b 明确标签、读出器和历史效应后，再预算 720-D 同 tap cache 与 §5 G3；不一次性启动全部表示组。
4. **工程与基线验证**：适配 2603.05487 的低层 observer controller，先验证 observer compliance 与真实 physical compliance 的差异。
5. **闭环主实验**：只让通过前序门禁的 interaction variables 进入 paired rollout。
6. **独立确认**：在 untouched tasks/episodes 上一次性验证冻结的主要结论。

不要同时启动 CPC、world-model、第二 VLA 和 RL。它们只有在前一层证据明确后才有区分价值。当前最短路径是：**SAE 独立复制 + 条件转变预测 + 同 tap 公平比较**。

## 10. 结论解释矩阵

| Future predictive | Action/closed-loop causal | 允许的结论 |
| --- | --- | --- |
| 是 | 是 | 最强候选：策略用于控制的 predictive interaction state |
| 是 | 否 | 可用于 world-state monitoring 或 world-model prediction，但未证明策略使用 |
| 否 | 是 | action-generation/motor feature，而不是未来物理状态 |
| 否 | 否 | 当前方法下无证据；可能是静态相关、dictionary artifact 或容量不足 |

这张矩阵应成为最终论文组织结果的主结构，而不是按模型名称分别叙述。

## 11. 2026-09-11：基于完整 G2 的下一步设计

### 11.1 本次核对的最新证据

Mode：design。目标仍是冻结 SmolVLA 的预测性交互表示研究；本节是工作协议，不是投稿结论。依据本地报告、绑定文件和指标生成代码，不新增外部文献检索。

| 证据源 | 核对结果 | 证据边界 |
| --- | --- | --- |
| [G2 report](../../../outputs/predictive_states/smolvla_conditional_readouts/report.json) / [summary](../../../outputs/predictive_states/smolvla_conditional_readouts/summary.csv) | 1,368/1,368 fits，1,344 个表示/条件记录，6,720 个 subset 指标行；累计 fit 时间 11,605.68 秒，约 3.22 小时 | fit 时间不是完整端到端墙钟；所有结果为 exploratory |
| [G2 binding](../../../outputs/predictive_states/smolvla_conditional_readouts/binding.json) | 12,303 窗口，W=4，dt=0.1 秒，horizon=1/5/10；8 表示×4 contexts×2 targets×3 horizons，两个 null 各 3 seeds | StableGrasp 需要当前/未来标签有效；测试仅 object/4、spatial/8，共 10 episodes |
| [容量对照](../../../outputs/predictive_states/smolvla_capacity_control/report.json) | 320 指标行，包含相同输入宽度的 Gaussian R 与 R+A | 一组随机特征；匹配维度不匹配协方差，也不能与 G2 不同 eligibility 的数值直接相减 |
| [项目门禁](../../../ccfa.yaml) / [SAE 归档](../../results/libero_smolvla_sparse_features/report.json) | 尚无已核实的新 G1 独立复制通过记录；本地状态仍限制闭环 | 不把 seed-0 pilot 当作跨 seed 复制；不更改执行权限 |

G2 binding SHA-256 为 434f2a681780439427c93693bae6db8f4dfbb3f48279abc40efb6019592a6f90；StateBank SHA-256 为 a645164cea60d674b4edf72b1f65b1d168c0498944789c6e2910ac9813117d41。结果绑定的是现有 480-D expert/31/mlp/output token cache，不是 720-D SAE tap。

以下预先展示两种 current PCA 聚合的全部 K=5/10、ACY endpoint 结果，避免只摘最大值。数值为 task-macro Brier，正增益表示加入表示后风险降低；不是 MI 或显著性结论。

| Target | K | ACY baseline | +PCA mean current | ΔBrier | +PCA first-token current | ΔBrier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Contact | 5 | 0.047022 | 0.046591 | +0.000432 | 0.046805 | +0.000217 |
| Contact | 10 | 0.065345 | 0.065270 | +0.000075 | 0.065420 | −0.000075 |
| StableGrasp | 5 | 0.055032 | 0.053607 | +0.001425 | 0.054288 | +0.000744 |
| StableGrasp | 10 | 0.067074 | 0.065103 | +0.001971 | 0.065076 | +0.001998 |

完整 48 个 aligned cells/context 中，A、C、AC、ACY 的总体风险正增益数分别为 34、43、23、21。这只是描述性计数，不是相互独立的 192 次成功检验。

关键区分：

- StableGrasp K=10、PCA first-token current 的 ACY 增益在两个测试 task 都为正；相对 3 个 episode-swap 的总体 Brier 优势为 0.002393–0.004034。PCA mean current 同项相对 swap 为 −0.000602–0.001390，方向不稳定。提示 token 聚合值得保留作固定对照，不据此选出确认性 winner。
- hidden mean history 在 StableGrasp K=10、ACY 下总体增益为 −0.038550，changed 子集却为 +0.043970。该相反方向说明不能用 changed 改善替代总体风险，也不能认定“更多历史一定更好”。
- 容量对照中 StableGrasp K=5、hidden mean current 的 A/Z+A/R+A Brier 分别为 0.077987/0.071655/0.082557；简单增加随机输入维度没有复现该项改善。但它没有证明动态充分性或排除其他正则化效应。
- 当前 G2 alpha 由 validation 的窗口平均 Brier 选择，而报告主汇总为 task macro；G2b 将选择准则与主估计量一致。旧报告保持不变。

### 11.2 下一步先解决三个可区分的问题

| 问题 | 假设 | 最小证据 | 空结果的意义 |
| --- | --- | --- | --- |
| 是否预测事件发生，而非只识别 endpoint？ | 表示能预测未来窗口中 onset/offset | 仅按当前状态定义 risk set，保留未来阳性和阴性 | 只支持当前 endpoint 诊断，不支持事件预测 |
| 历史为什么常常变差？ | 拼接的维度/估计成本抵消了历史信息 | 当前 32-D 与整段历史压成 32-D，匹配 anchors | 当前读出器未检出历史增益；不等于环境无记忆 |
| 条件增益是否受线性可访问性限制？ | A 与 Z 的非线性交互比简单拼接更有用 | AC 与 AC+Z 同时使用相同浅层非线性读出 | MLP 若只提高 baseline，不应归功于表示 |

这里的 A 是当前冻结 policy 在演示观察上生成的计划动作 chunk，未来标签来自演示轨迹。因而所有结果首先是 demonstration-distribution forecasting；它们不等于执行该 chunk 后的预测，更不是 do(action) 效应。实际演示未来动作只留作 privileged 对照，不混入主模型。

### 11.3 G2b：prospective event readout 固定协议

> 历史预测实验协议；保留用于解释已有 G2b 结果，不作为当前功能干预的预测准入门槛，参见 §0.7。

**数据与标签**

1. 复用完整 480-D cache、固定 checkpoint/noise 和现有 train/validation/test task split。原测试集继续标 exploratory；不在这些任务上“重新划分”得到假装未查看的确认集。
2. 仅使用当前及过去信息构造输入。onset risk set 为当前 Y=0，目标为未来 1…K 任一步 Y=1；offset risk set 为当前 Y=1，目标为未来任一步 Y=0。两组都保留未发生事件的负例；不按未来结局先筛样本。
3. 两个物理标签×onset/offset×K={5,10}，共 8 个事件目标，对应 0.5/1.0 秒。主机制问题为 StableGrasp onset K=5 与 offset K=10；其余 6 项完整报告为次要诊断。此选择是本次探索性设计，不声称早于 G2 预注册。
4. 最小版本保留最长 horizon 的共同 anchors，要求当前到 K=10 的所需标签完整有效；所有 K 与方法在同一标签/risk set 内共享 eligibility。缺失和 episode 截断不填零，记录排除数量及 task/episode 分布。该版本估计的是完整可观察窗口上的风险，不能推广到被排除的终止窗口；hazard/删失模型暂不训练。
5. 每项先输出 train/validation/test 的正负窗口数、独立事件段数、episodes 和 tasks。train/validation 缺少任一类则标 not estimable；测试单类任务保留 Brier，AUPRC 标 NA 并报告覆盖，不能删掉该 task 美化宏平均。当前这些计数为 TBD，须由标签审计产生。
6. 当前 StableGrasp 标注代码使用截止 t 的后向窗口，并未用未来定义 Y_t；但相邻标签共享平滑历史。报告标注窗口长度，检查 onset 时间的标注滞后，不把短时预测直接解释为新动力学。依据 [annotation.py](../../../interaction_vla/representation_study/libero/annotation.py) 的 _stable_grasp。

**输入、读出器与固定计算量**

在分开的 onset/offset risk set 中，当前 Y 为常数，故 ACY 与 AC 不需要重复拟合。AC 保持 elapsed time、4 帧 proprioception 和当前 50×7 action chunk；risk-set gate 使用真值，明确是 privileged 诊断。

| Arm | 输入 | 区分目的 |
| --- | --- | --- |
| B0 | AC | 强上下文 baseline |
| B1 | AC + PCA-32(mean token at t) | 最简单静态表示 |
| B2 | AC + PCA-32(first token at t) | token 聚合效应 |
| B3 | AC + PCA-32(concat first-token history W=4) | 同输出维度的历史效应 |

B3 对整个 4×480 历史向量拟合一个 32-D PCA，不是旧 G2 的逐帧 PCA 后拼接成 128-D。scaler/PCA 只拟合训练 anchors，并在所有目标中固定；表示不使用标签训练。B1/B2/B3 都完整保留，不在看过 test 后删 arm。Raw 的高维诊断已有 G2，不重复纳入这一最小网格。

- 线性读出：复用 Ridge 与数值检查，alpha={0.1,1,10,100}，按 validation task-macro Brier 选择，输出仍明确为 clipped scores。
- 非线性读出：拟议单隐层 64 单元、ReLU、sigmoid 输出，BCE 训练；lr=0.001，batch=256，最多 200 epochs，patience=15；weight decay={0,0.0001,0.001,0.01}，seeds={0,1,2}。所有 arms 相同规则；按 3 seeds 的 validation task-macro Brier 平均值选 weight decay，逐 seed 报告测试风险，并对比同 seed B0，不挑最好 seed。输入宽度导致第一层参数数目不同，需报告参数量，不能宣称严格参数等容量。
- 负控制：Ridge 对 B1/B2/B3 各做 3 个 same-task episode-swap seeds，复用 G2 donor 原则；只交换 Z，AC 与标签保持 recipient。记录 elapsed mismatch、coverage；它仍是近似负控制，不是严格条件随机化检验。没有为 MLP 再展开完整 null 网格。
- 总计 8×4×(1+3)=128 个 aligned readout fits，加 8×3×3=72 个 swap fits，共 **200 个选参后模型 / 最多 800 个超参数候选训练**。另计算 8 个 train-prevalence 常数预测，不计优化 fit。MLP 有 early stopping，不能由 G2 的 3.22 小时线性外推本批耗时。
- 实施时先运行一个包含新事件标签、Ridge/MLP 与 swap 的完整关键路径小样例，测量时间及峰值内存，生成精确 plan 后再安排完整执行；本设计没有给未测得的运行时间作承诺。

**指标与统计**

主要指标是在上述完整 risk set 上的 task-macro Brier，以及同一读出家族内 B0−Bi 的 paired 增益。辅助报告 AUPRC、阳性率、校准分箱与事件覆盖；不以 class-balanced 重加权 Brier 代替真实发生率下的风险。

当前两个测试 task 分别报告结果；可在 task 内以 episode 重采样，再计算固定两 task 等权平均的 paired CI，2,000 次 bootstrap。该区间仅描述这两个任务内的 episode 不确定性，不代表未见任务总体。重叠窗口保持在同一 episode cluster；随机化/MLP seeds 不是新增实验样本。另报 episode 等权风险作敏感性分析。

正式确认前，在 validation 上固定一个表示/读出配置和上述两个 StableGrasp 主目标；确认性多重比较 family 为这两个 paired baseline 改善，采用 Holm 0.05，报告未校正效应与区间及校正结果。其余模型、horizon 和标签全部作为次要结果，不通过不断更换主目标获得通过。G2b 当前探索数据不使用显著性门槛决定论文成立。

**推进条件与停止分支**

- 数据门禁失败：先处理标签覆盖/终止定义，不训练 RET 来弥补不可估计的目标。
- 只有 MLP baseline 改善，AC+Z 没有额外增益：支持读出器容量解释，暂停“表示提供额外事件信息”的主张。
- 表示增益只存在于 endpoint 或旧 changed 子集：保留负结果，优先校准与事件定义，不追加 CPC、world model。
- 至少一个预定主事件在两个已有 test tasks 同方向改善，且 Ridge 支持的候选优于三种 swap seeds：作为 G3 小规模训练的资源投入信号，不作为统计确认。若增益只由 MLP 检出，则先补该候选同预算的 MLP swap 对照，不能借用 Ridge null 通过。
- 若任务间方向冲突或差异小于 episode 不确定性，归类 unresolved；可做预定的新数据确认，不将其记为“无信息”或“已验证动态表示”。

### 11.4 G3 如何缩小，以及理论如何进入实验

通过 G2b 的工程/资源门禁后，优先开展同 tap 的最小训练对照：PCA-32、与 RET encoder 匹配的 AE-32、action-free RET-32、action-conditioned RET-32，3 training seeds，固定 W=4 和训练数据。SAE 作为独立稀疏组补入，不能把 1440/TopK32 称为与 dense32 等容量。所有新增算法均为待实施/待验证状态。

| 文档依据 | 实际实验要求 | 不允许的推论 |
| --- | --- | --- |
| 数学基础 §11–12：条件信息、充分性 | 比较同家族 B0 与 B0+Z；动作条件分支须有 action-free 对照 | NLL/Brier 的有限模型差直接等于 MI |
| §13、§19.2：几何、有限记忆 | B3 对 B2 同为 32-D；后续测 history residual | W=4 已是充分物理状态 |
| §19：AIS | 分开测事件预测与后续实际决策/价值后果 | 把两标签的 Brier 差代入完整转移核误差得到 success 界 |
| §20：value equivalence、失真 | 先明确所支持的事件与策略分布，再扩跨策略查询 | 单一演示分布结果等于 generic world model |
| §21：生存分析、累计占用 | 本阶段先窗口事件；有明确删失/时长问题再扩 hazard 或 occupation | StableGrasp endpoint 等于首次抓住或抓稳时长 |

当前 RET 适配器是 action-free predictor；把 A 加入后续 readout 不等于训练了 action-conditioned RET。新增 predictor 的动作必须对应产生 latent transition 的真实已执行动作；当前 SmolVLA 计划 chunk 与演示下一状态不是配对动力学。若用演示 a_t 训练，明确其是行为数据条件预测适配，并报告部署动作分布差异，不把计划动作代入当等价实验。

训练预算先沿用既有 RET 2,000 steps 作有界 pilot，记录训练曲线和 anti-collapse；若预算下未收敛，结论是未完成方法比较，不能据此判定 RET 无效。正式训练预算根据训练/validation 收敛情况冻结，AE/RET 共享 encoder 容量、样本访问和调参额度，并报告实际 compute。不同自训练 latent 空间的 MSE/EV 不可直接横比为物理预测能力，事件风险仍是共同指标。

720-D 同 tap cache 需要先核对原 SAE dictionary 权重是否本地/服务器可用，并记录 checkpoint、tensor 语义、噪声和 forward trace。若继续先用 480-D 做训练 pilot，必须标 secondary-tap pilot，不能与 720-D SAE 的 action effect 合并主张。

### 11.5 因果路线与 2603.05487 的位置

本次复读本地 v1 PDF 的 observer/controller 定义、实验和 limitations：原文讨论 transformer 与 flow-matching hybrid 架构，但具体扩展到 diffusion/flow head 仍列为未来工作。项目应比较的是其线性 observer 最小范数控制思想；SmolVLA 上属于适配，不称原样复现。

G1 继续使用固定 feature families 和独立 dictionary seeds，配上逐状态同范数的随机方向与 matched atoms。单一 family 的 2 seeds×2 control classes 共 4 个基本对比，两个 families 共 8 个；正式统计需冻结这个 family，而不是把两个 control tests 漏算。通过与否按两个 seed、两类 controls 的完整证据解释；尚不能实施的原因需具体记录为缺权重、缺匹配信息或待执行，不能把缺失写作失败。

G4/G5 保留 observer compliance、实际物理 compliance、success 的分离。对动态 latent 的干预还需要有效 activation lift 和零编辑回放；预测性训练本身不给可执行编辑。oracle phase gate 只用于机制隔离；StableGrasp offset 不必然是掉落，正常放置/释放也会 offset，必须结合阶段/支撑真值才能命名 drop。

### 11.6 下一次工作的具体交付与确认数据

下一次实施只需复用已有 sequence_index、cache binding、PCA、Ridge、donor map，加入独立的窗口事件目标和上述有界 readout 网格。目标输出建议为 outputs/predictive_states/smolvla_event_readouts，使用不同于 conditional_readouts_v1 的 schema；当前尚未创建或实现，不提供假想可执行 CLI。

| 顺序 | 交付 | 当前状态 | 进入下一步的条件 |
| --- | --- | --- | --- |
| 1 | 8 事件目标 coverage/eligibility 审计、完整 plan | 已实现入口，待完整运行 | 双类训练/验证支持，终止与缺失已说明 |
| 2 | G2b 200-model 固定网格、paired 结果与 null 对照 | Ridge smoke 已通过，完整运行待执行 | 明确静态/历史/读出器三种解释 |
| 并行科学路线 | G1 独立 seed 与匹配干预复制 | 门禁未获新证据解除 | 原模型和权重可用，完整复制证据 |
| 3 | 同 tap 的最小 AE/RET 训练比较 | 条件性设计 | G2b 提供投入信号，预算与动作来源冻结 |
| 4 | 新留出数据上的冻结配置确认 | 待规划数据清单 | 配置、指标、假设冻结后才揭示结果 |
| 5 | 2603.05487 适配与 interaction 闭环 | 仍受门禁约束 | G1/同 tap 因果链通过，实际 rollout 资源明确 |

确认集优先使用此前未用于 readout/表示选择的 task IDs；先核对上游 checkpoint 的 LIBERO 训练覆盖，区分“对本研究分析未见”与“对 VLA 预训练未见”。若只能补同 task 新 episodes，结论限于 episode 泛化，不称跨任务确认。具体 task/episode 数不由现有 10 episodes 凭空确定：先用 train/validation 事件率与 episode 级方差估算在预先选定最小有用 ΔBrier 下的精度/功效，冻结数据预算后采集，不能根据新 test 的效果持续加样。

上述待执行状态为 G2b 运行前记录；最新实际结果见分析报告。当前实验主线如下。

## 12. 能力获得驱动的机制实验：历史设计与证据记录

本节保留 2026-09-10 至 2026-09-13 的设计演化、结果及技术细节。最新目标、优先级和统计合同以 §0 为准；旧模型覆盖、待运行状态及执行顺序不自动沿用到新谱系。

### 12.1 问题与基本边界

**先建立官方能力参照并检查自训练是否充分，再研究可追溯训练中的能力变化与内部计算变化，检验候选如何影响 conditional action flow、动作生成和闭环行为。** 不预设当前 smoke 终点已充分训练，不把已观察到的任务异质性直接归因于 flow matching 或通用训练干扰；E0 的补充合同见 §12.16。

首要贡献为实证机制发现；信息瓶颈是待检验假设，未来像素或完整 latent 重建不是成功条件。Contact/StableGrasp 用于候选发现后的物理解释，不要求每个候选对应预设概念。G2b 保留为测量工具，RET 不再是默认下一步。

沿用 SmolVLA、LIBERO 和现有环境。先研究同一次训练的真实谱系，再做候选级独立 training-seed 复制。SFT25/50/100 的不同数据覆盖点不能自动充当同次训练的时间轨迹；SAE dictionary seeds 也不等于 policy training seeds。

### 12.2 E0：确认能力变化与参数更新位置

先列出 checkpoint 父节点、optimizer step、样本覆盖、policy seed、权重 hash、预处理与可训练参数。本地已有训练报告和历史归档，但本轮未确认整套纵向权重本地可用；必须核对实际存放位置。

对 VLM、视觉连接器、state_proj、action 输入/输出投影、expert 分别核对参数变化。现有 train_expert_only 实现冻结 VLM；这一分支优先回答下游怎样读取既有信息。研究 VLM 本身学习需要另设解冻对照，当前不新增。

拟议能力 pilot：3 个同谱系 checkpoints×4 个按任务语义预先选择的抓取 tasks×10 个配对初始条件，共 120 baseline rollouts。噪声、初始状态和评测条件配对；不按待比较模型的成功率挑 task。

测 simulator 真值的抓取建立、提起、抓持持续时间、正常释放和任务成功，不能用待解释 observer 自评能力。持续时间明确终止/删失，offset 不直接称 drop。用 discovery 行为定位能力弱/过渡/强 checkpoint，再在留出初始条件验证；若只有前后两点，只称训练前后比较。若无能力增长，不使用“从不会到会”的叙事。

这些 rollout 是新路线的测量需求，不代表解除现有执行限制；可比归档结果可先复用。无法执行时明确缺口，不用 flow loss 下降替代能力证据。

### 12.3 E1：固定 observation 与 epsilon 的 flow 过程对照

拟议先从 discovery bank 固定抽取 512 states，按 task/episode 覆盖，不按 Contact/StableGrasp 标签筛选；留出独立 episodes 验证候选。

最多 3 checkpoints×512 states×3 个公共 epsilon，共 4,608 次 action-chunk generation。记录全部 10 个实际 solver 阶段，即 46,080 个阶段记录，不是环境 rollout 数。先测小批时间/内存，再冻结完整资源预算。

**噪声配对必须落实为同一张量**：现有 deterministic_inference_noise 哈希包含 checkpoint 身份，新噪声表须独立于 checkpoint，保存实际 epsilon 及 state/replicate IDs。

固定采集 VLM 观察路径输出、实际可训练的观察/状态连接位置、expert 中部/后部各一处、velocity 与 noisy action。按参数审计命名模块，不笼统称 projector。保留 action-token 位置，不先 pooling；当前 cache 只保留最后一次 expert 调用，50 tokens 是 chunk 位置，不是噪声阶段。

分别运行：

| 对照 | 固定项 | 解释 |
| --- | --- | --- |
| 自然生成 | 相同 observation 和初始 epsilon，各 checkpoint 自行积分 | 内部差异怎样累计成最终动作差异 |
| 固定输入点查询 | 在相同 observation、x_sigma、sigma 上查询各 checkpoint | 局部条件计算差异，排除 noisy action 本身不同 |

固定点来自预先指定 reference checkpoint 的自然轨迹；候选级再反向 reference 检查敏感性。这些点对其他模型可能离分布，不能替代自然生成结果。

### 12.4 E2：以能力变化筛候选，不以人工标签筛候选

首轮使用低成本共享坐标：仅在 discovery 数据拟合标准化和跨 checkpoint 共享 PCA，在相同输入的配对激活差异上做 SVD。得到的是变化子空间，不是已经证明的语义特征。

候选保留两条通道，各最多 4 个方向/小子空间：

1. 激活/子空间变化大的方向：形成、重组候选。
2. 小幅扰动引起的 velocity 响应随训练变化的方向：招募候选。

第二通道只在 discovery 数据筛选，正式作用在留出数据验证。仅筛激活变化会漏掉“表示不变、下游读取改变”。另冻结 4 个激活尺度接近的低变化方向作对照；最多 8 个主候选，不能根据正式闭环结果重新排名。

若使用 SAE，先核实已有权重，采用共享字典或验证过的跨字典匹配并报告残差；独立 atom 编号不表示同一特征或特征出生。Crosscoder 在共享坐标不足时再加入，不先训练多层大网格。

候选冻结后才用 Contact、StableGrasp、phase、末端几何及 action component 做解释。无法命名但作用可重复的候选，保留为功能子空间。最终 claim 不取决于能否给每个 atom 起人类概念名称。

### 12.5 E3：排除 noisy-action 回读，定位招募阶段

优先三个实际 solver 点：高噪声 sigma=1、中间约 0.5、低噪声约 0.1。不预设高噪声是粗规划、低噪声是细修正。

| 实验 | 方法 | 要区分的解释 |
| --- | --- | --- |
| 噪声重复 | observation 固定，改变 epsilon | 噪声/动作方案敏感性 |
| 观察条件交换 | x_sigma、sigma、任务及可匹配 robot state 固定，改变观察或指定观察路径激活 | observation-derived 信息作用；混合输入可能离分布，需匹配和报告 |
| action-input 基线 | 相同目标/读出预算下比较 x_sigma+sigma 与加 hidden | hidden 是否超出 noisy action 的可访问预测信息 |
| 单阶段干预 | 固定观察、噪声、layer，其余阶段不改，只干预一个阶段候选 | 作用发生在何处、何时 |

记录即时 Δvelocity、继续原 solver 后的 Δaction，随后才检验物理后果。FM velocity 不是末端真实速度；早期阶段剩余积分更长，需同时报告局部与累计效应，不能按最终 action delta 大小直接排序阶段重要性。

训练式混合 x_sigma 低噪声时含真实动作，可能产生 teacher-action 信息注入；自然推理从纯噪声开始的结果单列。旧 G2b 的 AC 包含最终计划动作，控制它可能阻断真实中介路径，不能据此单独判断信息是否被使用。

### 12.6 E4：匹配敲除、方向性干预与物理作用

> 历史 E4 完整方案。当前按 §0.4 选择抑制或匹配替换之一，不要求两者全量运行；行为判据、样本精度与确认规则以 §0.5–0.6 为准。

先在同一个训练后 checkpoint 内干预，避免首先引入跨模型坐标兼容性。最小条件：

- 原始 no-op 回放；
- 候选方向/子空间抑制；
- 至少 8 个逐样本同 hidden 扰动范数的随机方向，保留完整分布；
- 匹配尺度的低变化方向；
- validation 冻结的剂量与反方向；
- 同模型的匹配 donor activation patch，用于检验定向变化。

线性子空间可用 h'=h−UUᵀ(h−mu)，U 列正交、mu 仅由训练/发现数据拟合。SAE 编辑使用残差保留的 h'=h+D(z'−z)，不让重建误差冒充特征效应。同系数不等于同 hidden 范数。

擦除后放回原 activation 只验证回放正确性，不算机制 rescue。跨 checkpoint patch 需另验证表示对齐、尺度与接收模型兼容性；失败不能直接否定充分性。候选作用必须在留出 episodes、噪声重复和匹配控制上重现。

旧 SAE 候选继续受 G1 跨 dictionary seed 门禁约束；新子空间路线需独立验证，不能借用 SAE pilot 自动通过。G2b 预测结果不是 E4 的通用准入条件；准入依据是冻结候选在留出数据上的功能效应、匹配控制和回放完整性。新设计不改变 ccfa.yaml 权限。

闭环先最多 2 个候选、一个冻结噪声阶段。一个候选的拟议 pilot 为 4 tasks×10 paired 初始条件×4 条件（无干预、target、随机方向、低变化方向）=160 rollouts；用于效应/方差估计，不自动满足统计功效。随机方向从离线 validation 固定，不能在闭环选最弱对照。

报告抓取建立、非预期掉落、正常释放、任务成功，以及 xyz/rotation/gripper 动作变化和非目标损害。使用 simulator 状态独立测量；真值 phase 触发属于 privileged 机制隔离，在线触发另报。统计单位为 paired initial state/episode，不是 frame。

### 12.7 E5：最后检验 interaction-sufficient compression 与 OOD

压缩假设：在保持指定预测/控制效果的容差内，训练后所需表示预算更小。先冻结失真与容差，再测 r={1,2,4,8,16,32} 的任务风险/行为曲线；每个 rank 的子空间只在 discovery/validation 选择，不在 test 挑 rank。

以原策略行为保留、任务表现和非目标副作用为主；物理预测仅在主张包含预测充分性时作为必要指标，否则作为辅助解释。effective rank 或活跃 atom 数下降不等于压缩成立；只保留子空间、删除其余分量可能产生巨大分布偏移，必须报告偏离和匹配扰动控制。动作接近也不能替代闭环性能保持，失真和容差须对指定任务分布预先冻结。

首先称 representation-budget / task-distortion comparison。若需 Shannon IB 主张，另定义随机编码/量化和可估计的信息量；维数不是比特率，优化 FM loss 也不保证遵循 IB。

OOD 先固定一个不改变物理机制的视觉干扰（如受控背景/纹理）和一个改变控制要求的物理变化（如物体姿态）。检验对前者稳定、对后者适当响应。候选在 ID 冻结后直接测，不再选轴或剂量。压缩与 OOD 的相关性不是压缩导致泛化；因果结论另需控制能力、训练预算和其他差异。

### 12.8 最小近期交付与停止条件

当前先完成 §12.16 的 E0 官方参照、数据/训练审计和充分训练计划；E1 小批机制测量可作为实现准备。E2/E3 的正式候选发现须待研究比较对象和能力范围明确后开展；RET、CPC、第二 VLA、world model 和 RL 暂缓。

| 阶段 | 必存产物 | 停止/收窄条件 |
| --- | --- | --- |
| 能力变化 | 每次 rollout ID、物理事件、checkpoint 谱系 | 无能力增长则不使用能力获得叙事 |
| 模型差分 | observation ID、实际 epsilon、x_sigma、sigma、tap、activation、velocity、action | 噪声不匹配或只存最后一步，无法定位阶段机制 |
| 候选发现 | 发现/验证 split、共享坐标、冻结候选与规则 | 仅 atom 编号或 probe 最佳值不能证明新增信息 |
| 干预 | 每样本范数、匹配控制、Δvelocity、Δaction、物理结果 | 不超过控制则停止该候选作用主张 |
| 压缩/OOD | 预算曲线、容差、任务定义、留出扰动结果 | 只有维度下降不能证明交互充分压缩 |

一条训练谱系只支持该谱系结果；若宣称训练通常招募某种结构，需独立 policy training seeds 的候选级复制。具体预算待权重可用性和首批资源测量确定；上述数量均是拟议 pilot，不是已发生测量或执行授权。

### 12.9 实施前协议冻结：checkpoint、数据与实际可执行范围

本节将 E0–E5 中的工作原则细化为交接协议。核对日期 2026-09-11；不把协议完成写成实验完成。

**已核实的训练谱系**来自 [v3 conditions manifest](../../results/libero_smolvla_protocol_v3/protocol_v3/conditions/manifest.json)。D100 的四个训练中间点具有相同 training_binding_sha256；基础阶段来自同一记录的初始化。权重仍需核实，manifest 不能替代实际文件。

| 角色 | condition | optimizer step | 用途 |
| --- | --- | ---: | --- |
| 主比较起点 | pretrained | 0 | 同谱系基准，不预称不会抓取 |
| 主比较中间点 | d100_u16617 | 16617 | 固定早期训练点 |
| 主比较终点 | d100_u66470 | 66470 | 固定最终训练点 |
| 保留细化点 | d100_u33234 | 33234 | 主比较发现阶段差异后定位变化时间 |
| 保留细化点 | d100_u49851 | 49851 | 同上，不事后作为更有利终点替换 |

D25/D50 属于独立数据覆盖对照，不混入主时间轴。登记的 pretrained/SFT25/SFT50/SFT100 阶段目录本地检查均不存在，D100 中间点尚未取得权重实体验证。本地 outputs/pretrained/smolvla_libero 是另一份官方已训练模型，仅能做单模型 flow 采集与 hook 验证，不得伪造为这条 D100 序列的终点。

**参数与输入审计输出**必须包含：实际各 tensor hash、形状、相对权重差、requires_grad 配置、optimizer 参数组（若归档可用）、tokenizer/preprocessor/normalizer hash。当前模型配置显示 train_state_proj=true；视觉连接器、state_proj、expert 与动作投影必须分别登记。若 normalizer 发生改变，同一物理动作与 state 的归一化坐标也会改变；局部 velocity 对照需转换到一致坐标，或限定为同 normalizer 的 checkpoints，不能直接比较坐标不一致的范数。

**数据分区**：继续沿用 StateBank 的 train/validation/test task split。E2 的共享坐标、差分筛选与初始响应筛选只用 train；候选剂量、位置、noise-stage 选择只用 validation；test 只评价冻结选择。当前 test 已被 G2b 查看，仍称 exploratory evaluation。真正确认需要新的研究留出数据，不能靠重新命名 partition 获得独立性。

512 discovery states 从 train task 中按 task 尽量等额、task 内按 episode 尽量等额分配，按排序后的 state ID 用固定 seed=42 抽样；保留全部抽样 ID。另最多 256 validation states、256 test states按相同规则抽取，某组不足时全取并报告实际数，不跨组补齐。采样不使用物理标签、动作效果或候选分数。

E0 的四个任务只按公开任务描述中需要抬起/移动被抓物体这一语义规则，从可执行任务注册表按 suite/task ID 排序取前四个；若不足四个则停止冻结并登记缺项。冻结 task/initial-state 清单后才读取模型行为。120 rollouts 是 discovery 表型 pilot，后续留出初始条件的确认费用另计。

### 12.10 Flow 采样、筛选与干预的可复核定义

**时间/维度定义**：物理时间记 t，训练步记 k，flow 时间记 sigma，layer 记 l，action-token 位置记 j。当前配置 chunk_size=50、num_steps=10、max_action_dim=32、实际动作维数=7。自然采样记录 10 个输入点 sigma=1.0,0.9,…,0.1，以及积分后的最终 x_0；不得把 sigma=0 当作已执行过的额外网络调用。

保存完整 32-D noisy action/velocity，物理动作效果只在实际 7-D 输出经同一后处理后测量。填充维度的范数单列，不让无执行意义的维度主导主指标。保存每次条件 forward 的实际 mask、token 位置与 padding 规则。

**固定点计算量另计**：E1 的 4,608 次自然生成仅覆盖三个主 checkpoints 的 512 states×3 noise repeats。固定 reference 为最终 D100，在其同一批轨迹的 3 个预定 sigma 点查询另外两个 checkpoints，需要 2×512×3×3=9,216 次单阶段 velocity 查询；reference 的查询复用原记录。首轮不展开十阶段全量固定点网格，也不将单阶段查询与整段生成混为一个 fit。

**共享坐标**：每个 tap 独立拟合标准化参数，各 checkpoint 和 sigma 等权。首轮共享 PCA rank=32，拟合数据仅为 discovery；每个 state 的多个 tokens/noises 不当作独立统计样本。原始 action-token 位置仍保存，PCA 只投影 hidden 维度。差分在同一输入点、同一 token 位置配对后进行；全零变化或数值秩不足的 tap 标记，不强凑 32 个方向。

形成/重组筛选：每个预定 expert tap 的前后差分矩阵做 SVD，保留前两方向，两 taps 合计最多四个。符号按最大绝对坐标为正固定，近重根视为子空间而非可唯一识别的单轴；验证时冻结整个对应子空间。

招募筛选：从共享 PCA 的 32 个方向中，在预先固定的 32 discovery states、一个公共 noise realization、高/中/低三阶段上比较两个端点 checkpoint 的中心差分 velocity 响应。每方向使用 train 标度定义的 ±扰动；以训练前后响应变化排序，每个 tap 最多两方向，共最多四个。该筛选单独产生额外 forward 成本，按实际维数/有效候选生成 plan，不隐含在 E1 预算中。

将标准化空间的候选方向映回 raw hidden 后重新正交化，再执行 raw-space 编辑；映射、单位化及截断秩记录在 candidate artifact。候选作用先以全部 action tokens 上相同通道子空间操作为主；token-specific 作用只作后续消融，不在 test 搜索最佳 token。

**干预幅度**：主实验使用候选分量的 suppression，剂量系数预设 {0.25,0.5,1.0}，验证集选择最小可重复产生目标作用的剂量；若均无作用，保留 null，不扩大剂量追结果。每个样本的 random/低变化对照匹配 target 的实际 Frobenius hidden delta 范数。可比随机方向先投影到候选子空间的正交补，再归一化；正交补不足时标不可估计。零 target delta 的样本保留为零效应，不能只报告非零激活样本。

招募效应采用同一候选的差分对照：

\[
R=(D_{\mathrm{target}}-D_{\mathrm{control}})_{\mathrm{post}}
 -(D_{\mathrm{target}}-D_{\mathrm{control}})_{\mathrm{pre}}.
\]

D 分别指即时 velocity 或最终 action 的位移，不能混合单位。主 action 指标为实际执行的 first action，完整 chunk 指标次要；连续 gripper、translation、rotation 分开报告。R>0 只支持候选相对扰动敏感性增长，仍需观察条件实验与物理干预判断 interaction relevance。

**观察来源诊断**优先在固定 x_sigma 上交换 VLM 图像条件与其匹配的缓存路径，保持语言/proprio 不变；另做 proprio 路径对照，区分视觉观察与机器人自身状态。donor 在 discovery/validation 按同 task、机器人状态距离和已过去时间匹配，不使用未来标签或干预效果。匹配距离阈值由 discovery 距离分布固定，越界样本不强配并报告 coverage。prefix 改动须重建相应 KV cache，否则可能实际仍读取旧条件。

### 12.11 统计、资源与结果表合同

发现和验证的分工优先于 p 值门槛。离线主估计量为 task 等权、task 内 state 等权、每 state 的噪声重复先平均；所有方法用完全相同样本。CI 以 episode 整组重采样，保留其全部 states/noises/tokens。两个已有 test tasks 分别报告，固定 task 内 episode CI 不外推到任务总体。

八个候选在 validation 冻结各自一个 tap/sigma/剂量后，test 的主 family 是八个“target−八随机方向平均”的 first-action displacement 对比；采用双侧、episode-cluster 的推断，Holm 校正 family-wise 0.05。单独报告八随机方向的范围以及低变化控制的点估计/CI。招募 R 若作正式检验，另列预先冻结的八比较 family；不得从这两个 family 任选一个显著就称全部机制通过。当前小样本结果以效应和不确定性为主，未拒绝零不等于证明无使用。

闭环 pilot 固定 E0 的 4 tasks 和 10 paired initial conditions，每个 condition 使用相同环境随机性与按物理 step 编号的噪声序列。轨迹分化后观察不同是干预结果，不强行对齐后续物理状态。primary outcome 为冻结候选预期作用对应的物理事件，task success 与非目标损害作为并列必要报告；不能只报告 action L2 或 observer 达标。

| 结果表 | 必须包含的列 |
| --- | --- |
| 能力时间轴 | checkpoint/step、task、attempts、grasp/lift/hold/release/success、paired difference、CI |
| 参数路径 | module、frozen 配置、实际权重变化、normalizer hash、可比较性 |
| 候选出生/重组 | candidate/subspace ID、tap、shared basis、pre/post activation、差分/跨阶段读出、数据分区 |
| 功能招募 | candidate、sigma、剂量、target/random/低变化效应、R、action 分量、CI/校正项 |
| 观察/动作来源 | observation/proprio/noise 条件、x_sigma 基线与 residual、donor coverage、velocity/action 变化 |
| 闭环后果 | initial-state ID、condition、实际物理事件、success、副作用、paired CI |
| 预算与泛化 | rank、冻结容差、ID/OOD 条件、任务失真、实际干预范数、保持性能/不确定性 |

**持久化**：复用 outputs/predictive_states 下单一拟议目录 smolvla_acquisition_flow；本次不创建空结果目录。必须保存 binding、实际噪声、分区 ID、逐样本 trace/干预结果、共享 basis 和模型权重来源，分片原子写入并按 binding 校验恢复。不能再次只保存聚合 report，导致事后无法重算 CI。源码改动或协议改变使用独立 binding，不覆盖 G2b。

**存储预算**：按实际 tensor 元素数×dtype bytes×有效 forward 次数计算，分别列原始 activations、x/velocity、metadata、候选结果。VLM prefix 若同观察和 checkpoint 内不随噪声变化，只存一次并引用。完整 trace 是同模型物理分析的依据，不能为节省空间只留 mean pooling。

**最小验证范围**：仅验证新关键路径，包括共同 epsilon、sigma/调用次数、no-op 与原 policy chunk 一致、hook 仅触发指定阶段、KV cache 更新、rank-zero/zero-delta、匹配范数、分区隔离、保存重载。先少量真实观察做这些检查；它们是工程验证，不算能力获得或因果证据。正式运行前输出精确调用数、实测每批时间/峰值内存、存储估计与可恢复方案。

### 12.12 本次交付状态与真正阻塞项

已完成：研究问题分解、现有谱系登记核对、主 checkpoint 序列、数据分区、候选筛选、阶段干预、统计 family、资源计算与产物合同的设计。

尚未完成且不冒充实验结果：纵向权重获取/哈希复核、D100 行为能力曲线、flow trace 实现与测量、候选发现、闭环和压缩/OOD 实验。

明确阻塞：归档主序列权重不在登记的本地路径；尚未确认可访问的外部权重位置。单一官方模型可支持可逆的实现准备，但不能回答训练前后形成/招募。训练服务器、真实可用显存/时限和完整谱系权重未核实，因此不承诺总 GPU 小时，不新建训练来替代缺失归档。

下一实施入口应先做 E0 权重清单审计，再做一个官方模型上的 E1 trace 验证；只有对应数据/权重和执行范围明确后，才运行跨 checkpoint 实验。已有 ccfa.yaml 执行门禁保持原样。

### 12.13 SSH 只读核对：权重可用性与零成功率的解释更新

2026-09-11 已按用户授权登录训练服务器读取项目目录、权重与 rollout 记录。没有启动训练或新 rollout。以下更新 §12.9/12.12 的权重状态，不改变原始结果。

| 训练对象 | 服务器检查结果 | 对实验的意义 |
| --- | --- | --- |
| 正式 libero_smolvla 初始化 | checkpoint 存在 | 可核对基础模型，但不能替代训练后权重 |
| 正式 D25/D50/D100 | stage manifests 和 training reports 存在，三者 run 目录目前为空 | 正式 D100@16617/33234/49851/66470 尚不可执行；本次搜索未找到迁移副本，不断言权重已永久丢失 |
| libero_smolvla_smoke 的 SFT100 | 004108、008216、012324、016432、016435 checkpoints 保留，last 指向 016435 | 是另一条现存候选训练谱系，不能冒充正式 D100 |
| 服务器官方 SmolVLA | 独立模型目录存在 | 可作评测链路正控制，不是自训练谱系终点 |

smoke SFT100 manifest 记录 254 个训练 episodes、16,435 steps、seed=2057736129，最终 training_report 标记 resume=true、status=complete。该模型配置为 train_expert_only=true、freeze_vision_encoder=true、train_state_proj=true。其能力获得和评测结果仍未知；中间 checkpoint 的训练连续性及配置一致性尚需逐项审计。

已按项目原有目录哈希算法重算并与归档核对一致：正式初始化 bd8a2c55d285972df7cf3e87af2be93f503380e5088db90671f109135140e182；smoke SFT100@16435 f0a6154caabce3556aeafa61c316f708cbac9370ee1c70531373947f5048e66f；服务器官方模型 3154ece6ac5f6e78bea3617a99f38d4d6eeaaeb43c476f70670310d8d2c4707a。目录哈希不同于单一 model.safetensors 的文件哈希，不能混用。

**服务器已有 task-0 rollout 结果**：

| 归档目录名 | successes | 限定结论 |
| --- | --- | --- |
| d100_u66470_spatial_task0 | 0/5 | 正式 SFT100 在这次评测失败，不等于所有任务均不会抓取 |
| official_smolvla_libero_spatial_task0 | 0/5 | 官方模型早期也存在零成功率评测 |
| official_smolvla_positive_control_v1 | 9/10 | 后续官方正控制成功；有完整 evaluation contract |
| official_smolvla_protocol_v3_task0_retry | 10/10 | 另一次官方模型评测成功；本次未找到完整启动合同，不当成严格配对因果对照 |

9/10 正控制合同明确设置 n_action_steps=10、empty_cameras=1、num_steps=10、相机映射、init_states=true、单环境和 recording=false。服务器官方模型 config 默认 n_action_steps=50、empty_cameras=0；评测时存在显式覆盖。旧 [positive-control 设计](2026-09-01-smolvla-official-positive-control-design.md) 已记载旧 checkpoints 曾使用无效 rollout 协议。这支持优先核对评测链路，但本次未取得 D100 0/5 的完整启动配置，不能归因于某一个参数，更不能声称已修复。

**修订的下一步**：先用有成功记录的官方模型及其已归档协议检查当前评测链路，再对现存 smoke 谱系做有界能力测量。只有测到该谱系内部的能力差异，才将其作为 E1–E4 的替代研究对象，并明确更换数据规模和谱系；若无能力差异，先定位训练/适应失败，不把最终 checkpoint 预标为“会抓取”。正式 D100 只有恢复权重并完成同协议评测后才能回归主比较。

服务器现场快照：GPU 驱动报告 RTX 4080 SUPER、32,760 MiB 显存，检查时显存占用约 1 MiB、GPU utilization=0%；数据盘约 53 GiB 可用；进程表未显示 python/train/torchrun。硬件资源是现场观测，不作为后续运行保证。未开展新的复现 smoke，根因仍未确定。

### 12.14 2026-09-13 执行进度：E0 部分完成，E1 工程 smoke 完成

用户随后授权在现存 smoke SFT100 谱系上执行有界实验。结果归档见
[capability/flow pilot](../../results/libero_smolvla_acquisition_flow/README.md)；本节更新执行状态，原协议与停止条件保持不变。

**E0 capability pilot** 已完成设计中的数量结构：checkpoint `004108`、`008216`、`016435` × LIBERO Spatial task 0–3 × 10 个配对初始条件，共 120 rollouts。三点总成功分别为 11/40、7/40、14/40。`008216 -> 016435` 有 10 gains、3 losses，双侧 exact McNemar p=0.0923；逐任务方向不一致，task 3 在三个 checkpoint 均为 0/10。由此将能力变化判为 **exploratory partial / unresolved**，不通过“从不会抓取到会抓取”的叙事门禁。该批 rollout 只保存 task success，没有采集抓取建立、lift、hold、正常 release 与非预期 drop 的 simulator 真值，因此不等于完整 E0。

**E1 trace engineering smoke** 已在公共数据 binding 下运行。每个可用 checkpoint 记录 8 states × 3 个 checkpoint-independent epsilon × 10 个实际 solver 输入阶段，保存 x_sigma、velocity、expert middle/late token activations、final x0、normalized action 和 postprocessed action；相邻 `016432 -> 016435` 的差异接近零，支持配对实现没有制造大的伪差异。已完成的 flow diff 比较各模型的自然积分轨迹，不是 §12.3/§12.10 要求的相同 x_sigma 固定输入点查询。512-state 正式 trace、固定点查询和 VLM/观察路径 tap 尚未完成。

当前阶段表：

| 阶段 | 当前状态 | 已获得证据 | 下一 gate |
| --- | --- | --- | --- |
| E0 权重/能力 | `complete_gate_failed_task_heterogeneity` | 120 个 discovery success rollouts；320 个独立物理确认 rollouts | 停止全局能力获得叙事；决定是否冻结 task-conditioned reorganization 假设 |
| E1 flow trace | `engineering_smoke_complete` | 公共 epsilon、10 stages、完整 token/action arrays、自然轨迹 paired diff | 先实现固定输入点查询，再决定是否扩大至 512 states |
| E2 候选发现 | `not_started` | flow 差异只提供描述性前置证据 | 先冻结修订后的 task 0/1 方向性对照，或更换具有一致能力增长的谱系 |
| E3 来源/阶段定位 | `not_started` | noise repeats 已保存但尚未形成来源对照 | observation swap、action-input baseline、single-stage intervention |
| E4 因果/闭环 | `not_started` | 无 | E2/E3 冻结候选且超过 matched controls |
| E5 压缩/OOD | `not_started` | 无 | E4 后再评估 interaction-sufficient compression |

因此目前不能直接进入 SAE/RET 方法竞赛或闭环干预。最小下一步是补齐 E0 的独立物理表型确认与 E1 固定输入点实现；若能力差异在确认集仍小于 episode 不确定性或任务方向继续冲突，本谱系保留为训练重组的负/混合结果，不进入能力获得因果叙事。

### 12.15 2026-09-13 E0 独立确认结果

在未用于 §12.14 discovery 的 initial-state IDs 10–49 上，固定比较 smoke SFT100 checkpoint `008216` 与 `016435`。四个预定任务、每 cell 40 个配对初始条件全部完成，共 320 rollouts。8/8 cells 均具有完整且唯一的 ID 10–49，逐 episode 物理记录的 success 与 LeRobot `eval_info.json` 完全一致。原始数据与派生报告见 [E0 confirmation artifact](../../results/libero_smolvla_acquisition_flow/README.md)。

| outcome | 008216 | 016435 | task-macro paired delta；95% task-stratified bootstrap CI |
| --- | ---: | ---: | ---: |
| Contact | 89/160 | 94/160 | +3.1 pp；[-5.0, +10.6] |
| StableGrasp | 45/160 | 42/160 | -1.9 pp；[-10.0, +5.6] |
| Lift | 36/160 | 41/160 | +3.1 pp；[-5.6, +11.3] |
| Success | 28/160 | 32/160 | +2.5 pp；[-5.0, +10.0] |

最长稳定抓持时间的 task-macro 配对均值变化为 -0.049 s，95% CI [-0.193, +0.091]；最大抬升高度变化为 -0.00017 m，95% CI [-0.00955, +0.00915]。这些总体结果不支持全局抓取能力增长。

任务异质性很强。task 0（black bowl between plate and ramekin）StableGrasp/Lift 从 2/40 增至 15/40；task 1（black bowl next to ramekin）StableGrasp 从 30/40 降至 10/40、Lift 从 22/40 降至 9/40；task 2（black bowl from table center）StableGrasp 从 13/40 增至 17/40；task 3（black bowl on cookie box）两个 checkpoint 均从未接触目标。success 同样为 task 0 `4→12`、task 1 `18→6`、task 2 `6→14`、task 3 `0→0`。两例 `008216` task-0 success 没有满足严格五帧双侧接触 StableGrasp，但最大抬升超过 0.11 m；这表示严格 StableGrasp 规则未覆盖所有成功操纵方式，不应把它当成 success 标注错误。

**E0 决策：`complete_gate_failed_task_heterogeneity`。** 现有“从不会抓取到会抓取”的全局能力获得叙事停止；不能用 `008216 -> 016435` 筛选通用抓取 feature。确认数据支持一个更窄的新假设：训练在共享物体与目标、不同初始空间关系之间重新分配了 interaction competence，可能涉及 task-conditioned recruitment 或 interference。进入 E2 前必须先把这一假设、task 0/1 的方向性对照、候选冻结规则和独立验证单元写入协议；若不接受该范围修订，则返回寻找具有跨任务同方向能力增长的训练谱系。

### 12.16 2026-09-13 原则修订：可靠能力参照、充分训练与无标签机制发现

本节是用户确认后的当前设计。§12.15 的结果与原 gate 记录不变，但不再据此默认转向 task-conditioned reorganization：先排查训练不足、任务覆盖和评测错配，再决定是否把任务异质性作为主研究对象。本节是协议修订，不是训练启动、预算批准或实验完成记录。

#### A. 研究问题和证据边界

主问题：**有任务能力的 VLA 如何利用内部计算形成动作决策；这些计算在同一训练谱系中如何出现、改变或被下游招募？**

“有能力”由固定任务上的实际行为确定；“内部机制”由冻结候选的匹配干预及行为后果检验。允许某些机制没有简洁的人类语义名称，允许不同任务使用不同计算，也不要求每个训练 checkpoint 的能力单调上升。

| 主张 | 必需证据 | 不作为通用前提 |
| --- | --- | --- |
| 可访问的信息 | 留出数据上的读出与相关对照 | 闭环改善 |
| 预测性交互表示 | 匹配信息预算后的未来风险与时序对照 | 完整世界或像素重建 |
| 动作生成中的功能机制 | 留出数据上的阶段/方向干预超过匹配控制，信息来源有依据 | Contact/StableGrasp probe 或 G2b 必须通过 |
| 对特定控制行为的因果贡献 | 配对闭环中的定向后果、success 与非目标损害 | 干预一定提高成功率；必要机制的敲除可能使性能下降 |
| 任务相关充分性/压缩 | 预先限定任务和容差下的决策与闭环性能保持、残差信息检查 | 完整转移核恢复、所有物理变量可重建 |

G2b 保留为已有探索性测量，不作为所有 E2/E4 候选的淘汰器。控制 AC 中的最终动作计划可能阻断中介路径；条件预测无增益不等于策略不用该信息。预测性和功能性分别报告，不能将它们当作必然递进的阶梯。RET 仅在后续需要检验预测表示这一具体假设时进入同 tap 比较。

这一边界与 [Value Equivalence Principle](https://arxiv.org/abs/2011.03506) 中“针对指定策略/价值函数保留规划所需更新”的思想相容，但该理论不直接证明 SmolVLA 的隐状态充分，也不替代本项目的行为测量。

#### B. 模型参照：官方强模型与同谱系时间轴同时保留

| 比较 | 角色 | 可支持的解释 |
| --- | --- | --- |
| 固定官方 checkpoint vs 当前自训练 checkpoints | 强弱模型功能参照 | 同合同下的行为和内部计算差异；不是同次训练的前后因果归因 |
| 自训练同谱系早期/中期/充分训练终点 | 能力形成主比较 | 在已审计初始化、数据和训练配置下，该谱系的计算与能力变化 |
| 自训练终点 vs 官方 checkpoint | 复现质量与外部参照 | 是否达到预先定义的可比性能范围；不要求权重逐位一致 |

每份模型单独登记 repo/revision、权重与 normalizer hash、架构/tap、训练来源及评测合同。官方正控制的 9/10 只属于其 checkpoint 与 Spatial task 0；不能继承给 G2b 的另一份 checkpoint，也不能当作整个 LIBERO 的 90% 成功率。官方模型和我们中途模型允许比较，但不把官方模型伪装成自训练的最终节点。

共享架构不保证神经元坐标或 SAE atom 语义一致。先在同模型内做因果检验；跨模型比较使用验证过的共同坐标/特征匹配。跨模型 activation patch 必须额外核对基底、尺度、normalizer 和输入合同，不能把相同 atom 编号视作对齐。

#### C. E0a：先核对训练是否充分与 task 3 是否真正被训练

[SmolVLA v1 §4.3](https://arxiv.org/html/2506.01844v1#S4.SS3) 报告仿真训练 100,000 steps、batch size 64；这只是论文配方参照，不等于已确认每一个 Hub checkpoint 的完整训练合同。当前本地 [smoke 配置](../../../configs/representation_study/libero_smolvla_smoke_linux_cuda.yaml) 为 1 epoch、batch size 2、tasks_per_suite=3；归档谱系为 254 episodes、16,435 steps。实际启动覆盖、梯度累积、resume 与数据采样须查训练产物，不从配置文件直接推断实际执行。

`SFT100` 表示所定义数据子集的 100%，不等于官方预算完成度或模型已收敛。训练量按全局有效 batch、优化步、实际样本访问及任务覆盖共同核对，不能只比 step 数；state_bank holdout 与 policy 训练未见任务也分别标记。

| 检查 | 必须取得的证据 | 对失败的解释 |
| --- | --- | --- |
| task 3 覆盖 | 实际训练 episode IDs、suite/task IDs、sampler 访问计数或可恢复的采样合同 | 未覆盖则按泛化问题解释；不直接称训练集欠拟合 |
| 数据/输入 | 指令、相机映射、state/action 单位和归一化、时间对齐、动作 padding | 不匹配先修链路；旧输出保留为失配诊断 |
| 优化预算 | 初始化 hash、optimizer/scheduler、有效 batch、resume step/状态、模块更新、分任务训练/验证曲线 | 预算不足或恢复错误不能包装为成熟模型机制 |
| 评测链路 | 官方模型在相同任务/初始条件及已验证执行设置下的行为 | 双方失败优先排查共同链路；官方成功仍不能独自定位自训练失败原因 |

task 3 无接触作为观测保留，不自动归因于 flow matching。task 0/1 方向相反也不能独自证明灾难性遗忘或梯度干扰；这些机制需要覆盖/优化审计和后续可区分实验。

#### D. E0b：复现到可比能力并保留训练轨迹

优先复用官方训练代码和经过核对的公开数据/配方。先判断能否继续现存谱系：有兼容 optimizer、scheduler、normalizer 和采样合同才称连续恢复；若改换配方、数据或重新初始化优化器，登记为分支/新训练，不与旧轨迹无差别拼接。无需从零重训 VLM；起点采用哪份 VLM/VLA 权重必须对应拟复现的官方实验。

正式训练前必须形成一份可执行合同，包含以下字段；未测部分保持 TBD，不把论文预算直接当作已批准算力：

- 官方目标：固定模型 revision、任务集合、eval contract；论文报告分数与本地复测分开。
- 数据：版本、训练任务和 episodes、样本排除、训练/validation/最终确认分区；数据偏离官方时明确为适配。
- 优化：可训练模块、有效 batch、累积、学习率/调度、总步数、训练 policy seed；推理噪声 seed 另记。
- 资源：真实训练关键路径 smoke 的 seconds/update、峰值显存、预计 GPU 小时及存储；硬预算和终止规则为 TBD，完成测量后冻结。
- 保存：初始化、按预定 update 间隔的中间 checkpoints、最终/validation-selected checkpoint，连同 optimizer/scheduler/RNG 和 preprocessing。间隔在看行为结果前确定，不能只保存成功率突增附近。
- 评测：同一组 validation 初始条件定期看能力曲线；终点按预先冻结的 validation 规则选择。最终确认初始条件在选参期间不揭示；保留弱化任务，不能只汇报提升任务。

“接近官方”需在同合同下定义差距容差：令 Δ 为自训练减官方的 task-macro success，冻结非劣容差 δ 与置信区间规则；拟采用配对 CI 下界大于 −δ，同时报告每任务差距及未达能力下限的任务。δ、任务能力下限、rollout 数量和预算须在正式确认前确定，不能由已观察到的测试效果反推。CI 跨零不代表等效。样本数按 validation 的配对结果/方差与目标精度规划，不照搬 9/10 或现有 40 次为固定充分样本。

如果全任务达到官方附近的成本过高，可将结论限定到预先确定的任务子集；不得看 test 后删去 task 3 换取“复现成功”。即便尚未达到官方，单个有可靠能力的冻结模型仍可开展限定任务的机制研究；若要主张能力形成，则需同谱系内经留出行为确认的能力变化。训练充分后仍有任务异质性，再单独决定是否研究 task-conditioned reorganization；本修订不自动选择这一主线。

#### E. E1–E4：无标签发现，外部测量解释

Contact、StableGrasp、Phase、Recovery 不作为候选必须对应的内部字典，不用于 E2 候选排序、共享基底拟合或发现样本筛选。Recovery 尤其需要预定义失败/扰动和后续恢复的可操作标准，不能靠阶段名称自动生成真值。

沿用 §12.4 的激活变化与功能响应变化两条发现通道：仅 discovery 数据拟合共享 PCA/SVD 或已验证 SAE；冻结候选后，用留出轨迹、连续几何、动作方向及 simulator 事件做解释。模型可能使用混合特征或多维子空间；无法命名而功能可重复的候选仍保留。稀疏、低秩与重建目标本身也是人为先验，需报告，不称“完全无先验”。

动作响应筛选不能把同一批样本上的大 Δaction 再当作验证证据。候选、tap、剂量和预期行为方向在 discovery/validation 冻结，独立数据检验；未匹配或未通过的候选完整报告。

物理标签是外部测量工具：例如严格五帧双侧接触只是 StableGrasp 的操作定义，不是全部成功操纵方式。保留连续几何、接触/抬升/轨迹与 success；检查边界样本和合理阈值敏感性，不按是否支持候选来调阈值。发现依据与事后命名分开存档，禁止因高标签相关性重新选择主候选。

#### F. Flow 机制的评判标准

沿用 t（环境时间）、k（训练步）、σ（生成阶段）、l（层）、j（action-token 位置）的区分。不得由图像生成类比预设早期规划、后期接触；这是需要实测的阶段假设。[Flow Matching](https://arxiv.org/abs/2210.02747) 给出的是生成建模方法，不是本项目任务失败的因果解释。

| 等级 | 最小比较 | 支持什么；尚不支持什么 |
| --- | --- | --- |
| 局部计算 | 同 observation、x_sigma、sigma、normalizer 下的跨 checkpoint 查询 | 排除中间动作输入不同的混杂；不等于实际轨迹改变 |
| 条件来源 | 有效匹配的图像/指令/proprio 对照及公共噪声重复 | 候选是否响应决策相关条件；混合输入离分布需测量 |
| 生成阶段 | 高/中/低噪声的单阶段干预，同时记录 Δvelocity 和最终 Δaction | 定位作用时间；最终变化大不自动表示阶段更重要 |
| 功能特异性 | 候选、同范数随机方向、低变化方向、剂量/反方向、no-op | 排除一般损伤和尺度效应；不自动证明某个人工概念 |
| 物理后果 | 配对初始条件下验证预期轨迹/事件/成功与非目标损害 | 候选对限定行为的因果贡献；不等于所有任务通用机制 |

自然积分轨迹与固定输入点查询并列保留；reference 轨迹点对其他 checkpoint 可能离分布，候选级反向 reference 检验仍适用。零编辑回放是工程检查，敲除后恢复原激活也不是独立的机制 rescue。

主研究可检验“机制发生在 flow action expert 内”，无需先增加另一种模型。若升级为“任务差异由 flow matching 目标特有机制导致”，才增加匹配数据、backbone、预算和调参规则的非-flow 目标对照，或足以区分采样器因素的受控实验。只更换 solver 步数最多支持采样敏感性；不能单独证明训练目标因果作用。另列 action execution chunk length，避免将闭环反馈频率效应混作 flow 积分效应。

#### G. 当前执行顺序与交付

1. **已取得的审计证据**：归档训练覆盖确认为 Spatial 0–2 / Object 0–2，Spatial 3 是该 SFT 的泛化任务；产物完整与可恢复不等于新训练协议完整。剩余合同字段见 §12.17。
2. **扩展参照（下一步）**：官方 Spatial 0–3、初始状态 10–19 已完成 40 次；按 §12.17 扩展共同初始条件并分层比较，不重新把已有 pilot 当作待完成工作。
3. **训练准备**：据审计结果形成继续训练或官方配方复现的具体配置、checkpoint 保存计划、验证规则、成本与停止上限。目标是获得可靠能力与可追溯轨迹，不预承诺达到官方分数。
4. **机制发现**：在比较对象和行为能力确认后，补齐 E1 固定输入点和观察路径测量，按 E2 规则冻结候选。
5. **功能与闭环验证**：E3 来源/阶段定位 → E4 匹配干预；旧 SAE family 独立复制要求继续适用，新候选单独验证。
6. **按主张扩展**：G2b 用于预测解释，E5 用于任务相关充分性，RET/world model 为有明确问题时的对照。RL 保持冻结。

下一阶段完成的判据是取得有来源的能力参照、明确训练充分性和比较谱系，并产出可验证的候选机制；不是额外完成多少探针或某个方法必须胜出。训练预算、官方等效容差和正式确认规模尚待训练合同审计及资源测量；本文件不把这些未决值伪造为已批准参数。

#### H. 已实现接口（服务器审计与固定输入点 smoke 已完成，正式机制实验未完成）

训练合同审计复用现有 stage manifest 与 LeRobot episode catalog；训练任务、评测任务和待审计谱系必须在命令中分别声明。它独立报告 `training_coverage_complete`、`artifact_complete`、`resume_ready`、`execution_contract_complete` 与 `protocol_ready`，不再用单一 `reproduction_ready` 混合这些含义。缺失字段保持未知，不会把 `SFT100` 名称当成训练充分：

```bash
.venv-lerobot/bin/python -m interaction_vla.representation_study \
  libero stages audit-contract \
  --config configs/representation_study/libero_smolvla_smoke_linux_cuda.yaml \
  --stage sft_100 \
  --training-task libero_spatial:0 \
  --training-task libero_spatial:1 \
  --training-task libero_spatial:2 \
  --evaluation-task libero_spatial:0 \
  --evaluation-task libero_spatial:1 \
  --evaluation-task libero_spatial:2 \
  --evaluation-task libero_spatial:3
```

上述例子把 Spatial task 0–2 定义为当前 SFT 范围，把 task 3 定义为泛化评测；若服务器 manifest 证明了不同范围，应以新命令显式声明，不能覆盖旧 manifest。新 stage schema 将规划 batch 写入 manifest，训练命令读取该值；旧 schema 可审计和恢复，但不能无提示地作为新训练合同启动。

物理事件 recorder v3 分开记录 `geometric_lift`、`supported_lift`、`unintended_drop`、`recovered_after_drop` 与 `normal_release`。最终 success 不会抹去先前 drop；`lift` 字段暂作为旧结果兼容别名，含义等于 `supported_lift`。已有 recorder v1 结果保持原样，其 release/drop 字段不得用于恢复机制结论，需由 v3 重跑或从逐帧数据重算。
时间轴和闭环汇总现在会对每个 task 检查实际状态集合、状态数量、重复/缺失记录、配对 episode 数以及跨 checkpoint 的一致性；缺少 `initial_state_count` 或 requested/actual 身份时安全失败，不把旧产物默认当作未使用。`audit-longitudinal` 命令进一步核对同一 lineage 的表示轨迹（R_k）、逐 checkpoint 离线功能报告（U_k）和能力时间轴（S_k），缺少任一 checkpoint 的功能报告时 `protocol_ready=false`。这些是审计门禁，不代表现有结果已经通过纵向确认。
纵向确认合同现在额外绑定 lineage hash、checkpoint steps 与训练合同 hash、候选跨阶段 trajectory、读出重拟合/迁移协议、剂量校准、主比较 `[R_k,U_k,S_k]` 及 `longitudinal_audit_sha256`；缺项保持 `contract_complete=false`，闭环入口拒绝消费该合同。代码实现了合同结构和证据覆盖审计，但不会把缺少功能依赖报告的时间轴自动升级为机制结论。
审计入口示例：

```bash
bash scripts/python.sh -m interaction_vla.representation_study.libero.acquisition audit-longitudinal \
  --lineage LINEAGE.json --timeline TIMELINE/report.json \
  --candidates DISCOVERY/candidates.json --trajectory TRAJECTORY \
  --functional-report 5k=GATE_5K/report.json \
  --functional-report 10k=GATE_10K/report.json \
  --output LONGITUDINAL_AUDIT.json
```

固定输入 flow 查询沿用 `flow_trace run`，增加 `--reference-trace`。参考 trace 必须是完整自然积分 trace，并与目标运行具有相同 StateBank、dataset、processor contract、solver steps、chunk 和 action dimensions；输出 binding 标记 `query_mode=fixed_reference_points`。这里固定的是参考模型自然路径上的 `x_sigma, sigma`，目标 checkpoint 仍使用自身在同一 observation 下形成的 observation/language/proprio 条件：

```bash
.venv-lerobot/bin/python -m interaction_vla.representation_study.libero.flow_trace run \
  --checkpoint TARGET_CHECKPOINT \
  --contract-checkpoint CONTRACT_CHECKPOINT \
  --metadata METADATA \
  --dataset-root DATASET \
  --reference-trace NATURAL_REFERENCE_TRACE \
  --output NEW_FIXED_QUERY_OUTPUT \
  --device cuda --batch-size 4 --max-states 512 --noise-repeats 3
```

固定点输出只包含 `epsilon/sigma/x_sigma/velocity/expert_middle/expert_late`；不把强制替换每阶段输入后得到的 solver 终点称为自然生成动作。闭环和候选干预仍须在训练审计、能力比较与候选冻结后单独执行。

### 12.17 2026-09-13 官方参照后的当前执行规划

本节更新 §12.16 G 的执行队列，不修改历史结果。模式为实验规划；新增 rollout、训练和干预均尚未启动。依据为[官方参照与服务器检查](../../results/libero_smolvla_official_reference_v2/README.md)、[训练合同审计](../../results/libero_smolvla_official_reference_v2/training_contract_audit.json)及[原同谱系结果](../../results/libero_smolvla_acquisition_flow/confirmation_raw/)。

#### A. 已知事实与当前问题

- 官方 checkpoint tree `3154ece6ac5f6e78bea3617a99f38d4d6eeaaeb43c476f70670310d8d2c4707a` 在 Spatial 0–3、初始状态 10–19 分别成功 **8/10、7/10、7/10、8/10**。这是本地执行合同下的能力参照，不是完整 LIBERO benchmark 复现。每任务 10 次中相差一次既不能确立任务异质性，也不能证明任务等效。
- 自训练 254 episodes 覆盖 Spatial 0–2 和 Object 0–2。Spatial 3 未进入该 SFT 数据范围；其无接触是泛化失败观测，不是训练内任务欠拟合的直接证据。这里的覆盖结论不描述官方 checkpoint 的训练范围。
- 原自训练 008216→016435 的逐任务变化仍有效，但“官方模型各任务表现不同”“该谱系训练中各任务变化方向不同”“差异由 flow 机制导致”是三个独立问题，分别需要行为重复、同谱系对照和受控干预。
- 真实模型固定点查询已完成 8 states × 3 noise repeats，证明测量路径可用；它不是均衡覆盖 Spatial 0–3 的机制实验。不能用该 smoke 的阶段 RMS 直接解释四任务成功差异。

第一步回答：**扩大同合同评测后，官方模型的任务差异是否仍有可辨认的效应量？自训练模型与官方的差距发生在训练内任务、泛化任务，还是两者都有？** 有能力模型的机制研究可继续；“能力如何形成”的主张另需可靠的同谱系时间轴。

#### B. E0c：扩展能力参照，先复用已有结果

建议下一批只增加官方 **Spatial 0–3 × 初始状态 20–49，共 120 次 rollout**，与现有 10–19 合为每任务 40 次。120 是可核算的工作量，不是已证明充分的统计样本数或已批准的运行预算。先确认这些初始状态可用，沿用已有评测入口；不添加新评测框架。

**状态身份绑定修订**：`initial_state_offset` 只是请求的 simulator state 编号，必须同时记录 `requested_initial_state_id`、实际执行的 `initial_state_id` 和每个 task 的 `initial_state_count`。环境若按 `init_state_id % len(_init_states)` 取状态，合同重叠检查按实际 ID 进行；仅比较 offset 数字不能证明留出。

执行前冻结 checkpoint/processor hash、任务版本、图像与动作预处理、时间上限、reset/warmup、成功判定、环境版本及种子合同。保持官方 pilot 的 `n_action_steps=10`、solver steps=10、empty camera=1、单同步环境和 recorder v3。若必须改合同，另标运行，不直接混池。初始状态 seed 与动作生成噪声 seed 分开记录；只有噪声耦合也经核对，才称公共噪声对照。

已有自训练两个 checkpoint 的 10–49 success 可在合同兼容核对后复用，避免先重跑 320 次。recorder v1 的 release/drop/recovery 不与 v2 直接比较：优先核查是否有足够逐帧信息重算；没有则单列 v2 重评成本，不能默默更换标签。其他物理指标也须核对操作定义后才合并。

主要统计与交付：

1. 每模型、每任务报告 success `n/N`、区间及逐 episode 配对差距；模型比较按同任务同初始状态配对。不同任务中同编号状态不是同一物理场景，不能据编号做跨任务配对。
2. 以自训练覆盖为准，**Spatial 0–2 的训练内能力**与 **Spatial 3 的泛化能力**分开汇报；四任务 macro 仅为补充。报告模型差距是否随任务变化及其区间，不只比较两个 p 值或总平均。
3. 物理行为以 success 为能力主指标，连续几何、contact、supported lift 等解释失败位置；不把任一个人工事件设为全部成功策略必须经过的内部状态。
4. 多噪声重复若启用，按初始状态聚类，不能把同一状态多次运行算成独立场景。重复数、目标精度与停止规则在新结果揭示前冻结；需先测单次耗时才能确定预算。

10–19 已被查看；20–49 对官方虽未评测，对自训练已被查看。因此这一扩展是诊断性比较，不包装为整个研究的独立最终确认。候选验证与最终确认需另留真正未用于选参、选任务或选假设的 episodes/初始条件。

#### C. 结果决定研究分支，不要求任务差异存在

| 新证据 | 下一步 | 不允许的推论 |
| --- | --- | --- |
| 官方任务差距小，区间仍宽 | 按目标精度决定是否追加；也可限定为能力参照，转向模型内条件计算 | 未显著即等效；为了机制故事挑少数失败样本 |
| 官方在追加样本中仍显示稳定且有实际幅度的任务差距 | 比较任务条件、几何和生成阶段，冻结候选后验证 | 单凭任务排名归因 flow matching |
| 官方有能力，自训练差距仍大 | 优先区分训练内能力不足与未训练任务泛化；制定充分训练合同，同时允许官方模型内机制发现 | 官方与中途模型差异就是训练过程因果效应 |

这三种结果均可产生后续问题；不以 SAE、RET 或某个任务差异必须“成功”为继续条件。

#### D. E1–E4：最小 flow 机制证据链

复用已验证的自然 trace 与 `--reference-trace` 接口。下一批发现数据按 task/episode 预先抽样，覆盖研究范围，不按 Contact、StableGrasp 或成功/失败挑选候选。样本规模按已有提取成本测量确定；8-state smoke 不重复充当科学实验。

1. **局部计算**：同 observation、噪声与 `(x_sigma, sigma)` 下测 velocity/hidden 响应；自然路径结果并列保留。跨 checkpoint 固定点用于排除路径输入差异，反向 reference 检查离分布敏感性；官方与自训练先在各自坐标内分析，不直接互换同编号神经元。
2. **条件与阶段**：对图像、语言、proprio 使用有效匹配来源对照；对高/中/低噪声阶段分别干预。记录局部 Δvelocity、最终 Δaction 与执行动作，不预设“早期规划、后期精细控制”。最终动作效应需真正继续积分测量，不能从固定点查询伪造。
3. **功能特异性**：无标签 discovery 冻结候选、tap、剂量与作用方向；在留出数据比较 no-op、同范数随机方向、低变化方向和反向/剂量控制。随机方向数量沿用既定候选协议；旧 SAE family 的复制要求不因换主线取消。
4. **行为验证**：通过工程一致性和候选验证后，另行批准配对闭环；报告预期行为变化、成功及非目标损害。削弱必要计算造成失败也是机制证据，不要求干预改善成功率。

环境时间 t 与生成阶段 sigma 分开；solver 步数与 action execution chunk length 分开控制。上述证据可以支持“机制位于 flow action expert”，不足以支持“flow 训练目标特有地造成任务差异”；后者才需要匹配训练的非-flow 对照。图像生成控制目前仅提供实验类比，不作为已验证机制或新增模型的理由。

#### E. 训练准备与停止边界

当前产物完整、可恢复；旧 manifest 的 planned batch 未记录，新协议的预算上限、官方参照绑定、评测合同、非劣容差、任务能力下限、validation 与停止规则仍待冻结。技术上能 resume 不等于应该立即继续训练。

若投入能力形成研究，按 §12.16 D 先形成同谱系训练计划；继续旧谱系与采用官方配方的新分支必须区分。若先研究成熟冻结模型的内部计算，无需等待自己的训练达到官方水平，但结论不称“能力形成”。预算、终点容差和正式确认规模保留待决，不根据现有测试分数替用户确定。

**当前交付顺序：E0c 合同与成本检查 → 官方追加 120 次诊断评测 → 训练内/泛化分层比较 → 按证据选择冻结模型机制实验或同谱系训练投入 → 留出功能与闭环验证。** G2b 可解释预测信息，RET、world model 与 RL 均不自动启动。当前新增结果全部为待运行，不覆盖已有 raw JSON。

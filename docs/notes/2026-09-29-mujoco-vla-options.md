# MuJoCo 机械臂训练与 VLA 选项

日期：2026-09-29

## 结论

**可以在 MuJoCo 里训练机械臂策略，也可以在 MuJoCo 环境中运行/微调 VLA。**对当前项目，推荐先把现有 Franka 场景的 expert 示范做成可验证数据集，训练一个 ACT 行为克隆策略作为“本场景 baseline”；随后若目标是语言条件行为，再把相同示范对齐到 SmolVLA 的 LeRobot schema 做微调。若想最快获得一个公开、可复现的 MuJoCo VLA 对照，则优先试 LeRobot 的 VLABench + `lerobot/smolvla_vlabench`，但它不是当前自定义场景的策略。

## 选择

| 路线 | 可行性与适用目标 | 主要限制 |
|---|---|---|
| 当前自定义 MuJoCo + expert demos + ACT | 仓库已有 Franka LeRobot ACT 数据采集/训练配置；pilot 配置注明 50 episodes、2–3 个物体、10D state、7D action。适合先证明环境、示范、部署闭环完整。 | ACT 不使用语言指令；这只能作为控制/模仿学习 baseline，不是 VLA。当前配置列出的物体范围是 2–3，不能据此声称已有 5 个物体数据。 |
| 当前自定义 MuJoCo + SmolVLA 微调 | 可行：LeRobot 的 SmolVLA 接收多相机、机器人状态和自然语言指令，并支持对自有 LeRobot 数据集微调。需要收集/转换示范，并使相机键、状态维度/语义、动作空间、归一化与推理接口一致。 | 官方 SmolVLA 文档以约 50 个示范 episode 作为起步建议，并强调每种场景变化要有足够示范；30k steps 是训练预算，不会弥补输入格式错误或数据覆盖不足。当前 Mac 的 `libero_approx` 场景 rollout 曾以 `physics_failure` 结束，不能作为成功 baseline。 |
| 官方 LIBERO + SmolVLA/OpenVLA | 是 MuJoCo/robosuite 系的标准模拟 benchmark。当前仓库已经有 `lerobot/smolvla_libero` 权重及固定 revision 的 LIBERO 数据，并有官方 recipe；OpenVLA 上游也包含 LIBERO rollout evaluator。适合验证官方 benchmark 上的 VLA pipeline 和语言条件任务。 | 与当前自定义桌面、控制器和物体布局不是同一个任务分布。官方 benchmark 成绩不能直接代表自定义环境成功率。适配/运行依赖应按服务器 Linux 环境处理。 |
| VLABench + SmolVLA | LeRobot 官方集成文档将其列为 MuJoCo/dm_control + Franka Panda 的语言条件操作 benchmark，提供 `lerobot/smolvla_vlabench` checkpoint、LeRobot 格式训练集和 eval/train 命令。包含多对象、多任务，最接近“现成 MuJoCo VLA baseline”。 | VLABench 需要 Linux；要安装上游项目和场景资产。它是另一套 benchmark/environment，不是当前自定义 MJCF 场景。其模型 observation/action contract（7D state/action、相机键）不能不经适配直接当作本项目控制器。 |

## 推荐顺序

1. **先跑本项目的 ACT baseline。**复用现有 expert 和采集/训练流程，固定对象数、种子和成功判据；确认训练数据能被策略重放，并测闭环任务成功率。这回答“MuJoCo 机械臂能否训练”的问题，改动最少。
2. **再做本场景 SmolVLA。**把相同场景中的 expert 成功轨迹写成 SmolVLA 兼容的 LeRobot dataset，先做 schema 检查和训练集动作重放，再跑短 smoke，最后才跑 30k。3–5 个物体需在数据里都覆盖；以实际支持的 5 个为上限，不把 5–8 写进目标。
3. **并行可选：用官方 VLABench checkpoint 验证通用 MuJoCo VLA。**这是最快的公开 VLA 对照，但须独立报告 benchmark 名称，不能和自定义场景混报。

对文本 conditioning 的研究，需保持场景状态/示范和对象配置匹配，仅改变指令并比较同一 checkpoint 的动作/成功率；单凭 checkpoint 加载成功或屏幕上机械臂会动，不能算 VLA baseline 跑通。

## 一手来源

- Hugging Face LeRobot，SmolVLA 文档：多相机、state、语言输入；自有 LeRobot 数据微调；约 50 episode 起步建议及 20k 示例训练：[SmolVLA](https://huggingface.co/docs/lerobot/smolvla)。
- Hugging Face LeRobot，VLABench 文档：MuJoCo/dm_control、Franka Panda、预训练 `lerobot/smolvla_vlabench`、官方数据集与训练/评估命令、Linux 要求及 observation/action schema：[VLABench](https://huggingface.co/docs/lerobot/vlabench)。
- LIBERO 官方仓库：benchmark 提供 130 个任务与标准 task suites：[Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)。
- OpenVLA 官方仓库：LoRA 微调和自定义 RLDS 数据接入说明：[openvla/openvla](https://github.com/openvla/openvla)。其上游 LIBERO evaluator 明确在模拟环境中运行 checkpoint：[run_libero_eval.py](https://github.com/openvla/openvla/blob/main/experiments/robot/libero/run_libero_eval.py)。
- robosuite 官方文档：可通过键盘、SpaceMouse、DualSense 或 MuJoCo GUI 收集人工示范，并将演示用于学习：[Human Demonstrations](https://robosuite.ai/docs/algorithms/demonstrations.html)。
- 本地依据：ACT pilot 配置（已移除，见 `pre-restructure`）、[SmolVLA LIBERO recipe](../../scripts/run_smolvla_official_reproduction.sh)。这些配置说明仓库已有流程/资产，不证明其策略在当前自定义场景已经成功。

## 结论边界

这些官方资料证明可行路径与公开 benchmark 的存在；它们不证明当前自定义场景的 SmolVLA 已经成功，也不提供本项目 30k 微调后的预期成功率。训练步数、GPU 可行性和场景泛化需由目标数据集上的 smoke 与闭环评估决定。

## 研究 pivot：语言目标身份 × 物理交互证据

### 一手证据卡

1. **语言 referent 本身已有独立研究。**Scalise et al. 发布了桌面 clutter 情景里的 1,582 条自然语言目标描述，并用人类对象选择准确率标注清晰度。它研究“听懂指的是哪个物体”，不执行抓取，也不含接触/可抓取性结果。[论文](https://journals.sagepub.com/doi/10.1177/0278364918760992)
2. **语言到抓取几何已有端到端基线。**CGNet 从 RGB + 命令直接预测满足命令的抓取，在 VMRD 派生数据集评估，并报告三次实机实验；论文动机明确指出串行目标检索再抓取会在多物体重叠时累积歧义。它比纯检测更接近端到端，但评估中心是抓取位姿，不是可分离测量“目标选对”与物理交互证据利用。[论文](https://arxiv.org/abs/2104.00492)
3. **Referring grasp synthesis 把 grounding 与 grasping 放进同一任务。**CROG/OCID-VLG 在室内 clutter 图像上提供语言描述和 4-DoF 抓取；作者报告仿真和真实机器人实验。相关 Grasp-Anything++ 收集 1M 样本、3M+ 物体和 10M+ 抓取指令，以条件生成预测 grasp pose，并报告真实机器人 grasping。两类工作已覆盖语言 referent 到抓取姿态/执行，但主要终点是指定抓取生成或单次抓取，不是将 target identity 和接触可行证据作为正交因素干预。[CROG CoRL 2023](https://proceedings.mlr.press/v229/tziafas23a/tziafas23a.pdf)；[Grasp-Anything++ CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Vuong_Language-driven_Grasp_Detection_CVPR_2024_paper.html)
4. **语言会改变同一对象上的操作区域。**IGANet 指出既有 affordance 预测常不随语言指令改变，提出 instruction-guided manipulation affordance maps，并报告未见对象和指令上的真实场景实验。这覆盖“指令语义影响对象局部区域”，但公开摘要没有显示它系统地将目标身份、操作可行性证据、抓取成功分开评估。[论文](https://arxiv.org/abs/2408.10658)
5. **detect-to-execute 落差已有直接提示。**RT-Affordance 把语言先转成机器人关键阶段的 affordance pose，再让控制策略依据该 affordance 操作。论文报告一个新物体抓取评估中语言条件 RT-2 成功率为 28%，常能选中正确对象并接近，却抓错部位（例如抓锅底而非把手）；给策略 oracle affordance 后成功率升到 76%，预测 affordance 的分层模型也明显高于基线。该论文不同任务子集/汇总表的成功率不完全相同，因此这些数字只对应相应评估切片，不作通用结果。它足以说明“认出物体”与“依物理可供性抓取”并非同一能力，也意味着宽泛的“VLA 会认物体但不会抓”不足以构成新颖性主张。[OpenReview 论文](https://openreview.net/pdf?id=y4KugwU0qU)
6. **现有 VLA benchmark 已有语义与物理知识覆盖。**VLABench 提供多类别语言条件长程操作任务，覆盖语义指令、物理规律和操作执行，也提供自动生成训练数据。因此不能把“用 MuJoCo 测语言条件操作”本身作为空白；可能的切口是更严格拆解 target binding 与 interaction-evidence use。[ICCV 2025 论文](https://arxiv.org/abs/2412.18194)

### 相关工作簇与可能的窄缺口（推断）

- **Referential grounding / target selection：**Scalise 数据集及 CROG 评测指令是否找对物体；多以定位或 grasp grounding 为终点。
- **Language-conditioned grasp / affordance：**CGNet、CROG、IGANet、Grasp-Anything++ 把语言与抓取位姿、部位或 affordance 关联；其中部分已有真实执行。
- **VLA 长程 benchmark / 语言到动作分层：**VLABench 测多类语义和物理任务；RT-Affordance 显示显式 affordance 表征可补语言直接驱动动作的弱点。

据此，**可检验而非已证实的 gap**不是“语言目标变化会影响 VLA 成功率”：LangGap 已明确在固定桌面布局下改变指令语义，并显示模型存在 language-following 问题；VLA-Arena 也将语言与视觉 perturbation 作为独立诊断轴。宽泛地声称文本会改变对象选择/成功率，已经被这些工作覆盖。

仍可检验的更窄问题是：当 referent 身份已选对后，模型是否会利用**自己刚才的物理交互结果**来更新下一步接触/抓取动作，同时继续操作语言指定的同一个物体？RT-Affordance 和 CrayonRobo 已研究显式 affordance/接触姿态与接触后运动方向；FACT 已研究接触丰富任务的力觉/精细控制失败；VLA-Trace 已研究跨模态表示、attention knockout 和 rollout 级语义行为。因此可能的区分点只能是**闭环中由策略动作产生的接触结果反馈**如何调节动作与 referent persistence 的耦合，而非静态视觉 affordance、预先画好的接触提示、通用语言 perturbation 或一般性模态消融。是否真的没人做过，需要继续查这些工作全文、supplement、代码与相邻文献，不能作新颖性保证。

### 后续近邻工作与重叠风险（2026-10-02 primary-source scan）

- **LangGap (arXiv:2603.00592)：高度重叠于语言因果影响。**固定桌面 layout，按四个语义维度扰动指令，专门暴露 VLA 忽视语言的问题；因此不能再把“换目标名词/关系词后行为或成功率变化”作为本研究缺口。它主要测语义扰动及训练数据增强，没有在摘要中提出由机器人接触结果驱动的在线纠错机制。[arXiv v1](https://arxiv.org/abs/2603.00592)
- **FACT (arXiv:2608.01402)：高度重叠于物理执行失败，部分重叠于 interaction evidence。**分析 contact-rich 任务的精度失败和力信号处理，报告五个真实机器人任务、约 2,500 次 rollout；其重点是 flow-matching 训练和 force-signal 编码。新问题不能只说“精密接触失败没被研究”；可区别之处是接触后证据是否改变同一语言目标的下一段策略。[arXiv](https://arxiv.org/abs/2608.01402)
- **CrayonRobo (CVPR 2025)：高度重叠于接触点和接触后运动的显式条件。**用 object-centric 视觉/语言提示提供 contact point、夹爪方向和 post-contact movement direction，预测 SE(3) contact pose 并顺序完成长程任务；实验含模拟与真实机器人。静态给定的 interaction prompt 已被覆盖。候选区分必须明确是策略执行后观察到的成功/滑脱/未移动等 outcome evidence，而不是再画一个接触点提示。[CVF 正式论文](https://openaccess.thecvf.com/content/CVPR2025/html/Li_Object-Centric_Prompt-Driven_Vision-Language-Action_Model_for_Robotic_Manipulation_CVPR_2025_paper.html)；[补充材料](https://openaccess.thecvf.com/content/CVPR2025/supplemental/Li_Object-Centric_Prompt-Driven_Vision-Language-Action_CVPR_2025_supplemental.pdf)
- **VLA-Arena (ICML 2026)：高度重叠于语言和视觉因素正交化。**170 个任务覆盖 Safety、Distractor、Extrapolation、Long Horizon，并对任务施加独立 language W0–W4、visual V0–V4 perturbation。只做文本替换 × 图像扰动的 benchmark 消融不足以区分本研究；需要考察物理交互后出现的状态证据及其对后续动作的作用。[arXiv v4](https://arxiv.org/abs/2512.22539)
- **VLA-Trace (arXiv:2605.30117)：高度重叠于 VLA 因果诊断方法。**结合表征相似性、attention knockout 与 rollout 行为探针，研究 modality routing 和细粒度语义跟随。因此不能宣称“首次因果分析语言是否进入动作”。可区别的研究对象是由模型控制产生的接触结果，以及它对下一动作和 target persistence 的闭环行为效应。[arXiv v2](https://arxiv.org/abs/2605.30117)；[项目代码](https://github.com/VLA-Trace/vla-trace)

上述工作摘要和可读正式材料支持重叠判断；我没有完成逐项 supplement/代码审计，故“没有测某项”的表述均限于已检查范围。

### 一个可证伪的机制与最小协议（推断/建议）

**可证伪机制：**文本 referent 维持“正在操作哪个实例”的目标绑定；接触后由策略动作诱发的视觉/状态变化（对象滑脱、没有位移、稳定提升）决定“下一步怎么修正”。预测：当同一策略目标接触失败时，有效 outcome feedback 应改变下一 action chunk 的接触/撤退方向，同时保持目标实例不变；若只观察名词而忽略物理结果，目标会命中但重复同一失败动作。

**最小协议建议：**在本项目 MuJoCo Franka 的 3–5 物体 clutter 中固定一个 checkpoint、相机、初始场景 seed 和语言任务模板，选可清楚辨认的 target/decoy。让策略产生相同的 pre-contact prefix；在首次接触后对物理参数或目标初态做预注册的轻微干预，使其中一支出现可观察的滑脱/无位移，另一支稳定接触/提升。再从接触后的真实 observation 恢复闭环推理；不能成功时重复目标选择条件下的多次配对 episode。

1. **referent 因素：**目标 A/B 的指令只替换 referential phrase，句式和动作谓词保持不变；记录首次接触对象 ID。LangGap 已覆盖语言变化影响行为，所以此处只用它确认 target-lock 是否保持，不把语言扰动本身当新贡献。
2. **outcome-evidence 因素：**在目标 ID 和前缀相同的前提下，对比策略获得接触后真实状态变化 vs 诊断性 observation replay/遮蔽该次交互结果。后者仅作为小规模 action probe，不作为自然 rollout 成绩。记录下一 action chunk 相对接触点的方向变化、是否尝试新的接触/撤退策略、目标 ID 是否仍正确；再报告稳定抓取与放置完成率。
3. **统计和边界：**先报目标命中率，再报命中条件下的接触后恢复率及最终完成率；以配对 rollout 差值和 seed/episode bootstrap 区间为主。预先定义成功阈值、outcome 操作和 action-change metric。只用真实闭环结果支撑能力结论；遮蔽/replay 支撑对输入 outcome information 的因果敏感性，不足以单独证明模型内部机制。

若真实 outcome 与 replay 条件的下一动作没有可测差异，outcome-use 假设被否证；若动作改变但不保留目标实例，referent/outcome binding 假设被否证；若两者均变化但任务仍失败，则效应可能止于动作响应而不改善物理成功。必须先让当前 SmolVLA rollout 的 `physics_failure` 子因由 tracing 解释、并有成功策略及对应数据后，才适合做这个机制实验。

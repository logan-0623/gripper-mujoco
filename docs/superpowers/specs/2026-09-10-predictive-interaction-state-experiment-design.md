# VLA 抓取能力获得与条件动作流机制：实验设计

> 初稿：2026-09-10；最新证据修订：2026-09-11  
> 状态：实验设计；不授权启动训练、闭环评测或 RL  
> 当前项目门禁：先完成跨 SAE seed 与匹配对照复制，再决定是否进入闭环  
> 主要依据：[数学基础](../../research/vla-mathematical-foundations.md)、[当前研究协议](../../../research/predictive-states.md)、[项目状态](../../../ccfa.yaml)、[Observing and Controlling Features in VLA Models](../../../paper/2603.05487v1.pdf)

**当前主线已按用户研究定位更新为 §12：能力获得 → 内部变化 → 条件 action flow 中的功能招募 → 闭环物理后果。G2/G2b 已有结果，作为物理预测测量工具保留；RET 不再是下一步默认主实验。** §1–11 保留为前期预测路线与协议记录，涉及当前顺序、核心假设和方法优先级时以 §12 为准。最新 G2b 逐任务结果见 [分析报告](../../research/g2b-results-analysis.md)。本次仅更新设计，不启动训练或解除执行门禁。

## 1. 要回答的科学问题

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

| ID | 假设 | 支持证据 | 反证或空结果 |
| --- | --- | --- | --- |
| H1 | 当前物理变量可读，但这种可读性不必对应策略实际使用 | probe 高、线性干预弱 | matched intervention 稳定改变动作与闭环事件 |
| H2 | Feature 694/981 的 action effect 是跨 SAE seed 的稳定 feature family | 两个独立 seed 的匹配 atom 均超过 random 与 non-candidate controls | 任一 family 仅在 seed 0 成立或只由 gripper 尺度解释 |
| H3 | Action-conditioned dynamic representation 比同容量静态表示更能预测 interaction transitions | onset/offset 条件风险下降，且不是 unchanged subset 驱动 | 只改善总体 Brier；changed/onset/offset 无改善 |
| H4 | 预测有用与控制有用是两个独立轴 | 同一 feature/representation 同时通过预测与干预测试 | 只通过其中一个轴；仍是有意义的分离结论 |
| H5 | 可干预 interaction variable 能在闭环改变真实物理事件 | 实际 Contact/StableGrasp、drop rate 或 success 的 paired 改善 | observer 输出改变但 simulator 事件不变，说明只控制了 readout |

不得把某一种方法必须胜出写入成功标准。Raw hidden、SAE、RET 任一者可能只在其中一个轴上占优；公平的 null result 仍能回答表示类型是否适合物理交互。

## 5. 实验执行顺序与门禁

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

## 12. 能力获得驱动的机制实验：当前执行设计

### 12.1 问题与基本边界

**当 VLA 通过训练获得抓取能力时，发现哪些内部特征或子空间被形成、压缩、重组或招募，并检验它们如何影响 conditional action flow、动作生成和闭环行为。**

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

先在同一个训练后 checkpoint 内干预，避免首先引入跨模型坐标兼容性。最小条件：

- 原始 no-op 回放；
- 候选方向/子空间抑制；
- 至少 8 个逐样本同 hidden 扰动范数的随机方向，保留完整分布；
- 匹配尺度的低变化方向；
- validation 冻结的剂量与反方向；
- 同模型的匹配 donor activation patch，用于检验定向变化。

线性子空间可用 h'=h−UUᵀ(h−mu)，U 列正交、mu 仅由训练/发现数据拟合。SAE 编辑使用残差保留的 h'=h+D(z'−z)，不让重建误差冒充特征效应。同系数不等于同 hidden 范数。

擦除后放回原 activation 只验证回放正确性，不算机制 rescue。跨 checkpoint patch 需另验证表示对齐、尺度与接收模型兼容性；失败不能直接否定充分性。候选作用必须在留出 episodes、噪声重复和匹配控制上重现。

旧 SAE 候选继续受 G1 跨 dictionary seed 门禁约束；新子空间路线需独立验证，不能借用 SAE pilot 自动通过。新设计不改变 ccfa.yaml 权限。

闭环先最多 2 个候选、一个冻结噪声阶段。一个候选的拟议 pilot 为 4 tasks×10 paired 初始条件×4 条件（无干预、target、随机方向、低变化方向）=160 rollouts；用于效应/方差估计，不自动满足统计功效。随机方向从离线 validation 固定，不能在闭环选最弱对照。

报告抓取建立、非预期掉落、正常释放、任务成功，以及 xyz/rotation/gripper 动作变化和非目标损害。使用 simulator 状态独立测量；真值 phase 触发属于 privileged 机制隔离，在线触发另报。统计单位为 paired initial state/episode，不是 frame。

### 12.7 E5：最后检验 interaction-sufficient compression 与 OOD

压缩假设：在保持指定预测/控制效果的容差内，训练后所需表示预算更小。先冻结失真与容差，再测 r={1,2,4,8,16,32} 的任务风险/行为曲线；每个 rank 的子空间只在 discovery/validation 选择，不在 test 挑 rank。

同时测物理预测、原策略行为保留和非目标副作用。effective rank 或活跃 atom 数下降不等于压缩成立；只保留子空间、删除其余分量可能产生巨大分布偏移，必须报告偏离和匹配扰动控制。

首先称 representation-budget / task-distortion comparison。若需 Shannon IB 主张，另定义随机编码/量化和可估计的信息量；维数不是比特率，优化 FM loss 也不保证遵循 IB。

OOD 先固定一个不改变物理机制的视觉干扰（如受控背景/纹理）和一个改变控制要求的物理变化（如物体姿态）。检验对前者稳定、对后者适当响应。候选在 ID 冻结后直接测，不再选轴或剂量。压缩与 OOD 的相关性不是压缩导致泛化；因果结论另需控制能力、训练预算和其他差异。

### 12.8 最小近期交付与停止条件

近期只推进 E0 权重/谱系审计、E1 小批完整 flow trace、E2 共享坐标筛选、E3 观察/噪声区分；RET、CPC、第二 VLA、world model 和 RL 暂缓。

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

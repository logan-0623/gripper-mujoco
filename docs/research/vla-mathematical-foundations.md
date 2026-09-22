**VLA Predictive Interaction States：数学依据与文献阅读笔记**

检索日期：2026-09-10。范围：苏剑林「科学空间」中与信息、预测、表征几何及生成建模相关的文章，以及必要的原始论文。本文服务于 Raw hidden / SAE / RET-style representation 的研究定位；没有运行新实验、修改既有协议或更新 slides。

**核心判断：最直接的理论主线是“动作条件下的预测充分性 → 有限读出器可用的信息 → 策略中的因果作用”。** 信息瓶颈帮助定义应该保留什么，对比学习帮助塑造可读出的几何，Flow Matching 帮助理解动作生成中的条件分布。它们提供不同层面的依据，没有哪个定理保证 RET 优于 SAE。

阅读范围：首轮直接读取了博客 6024、6181、9368 的正文；本轮新增读取 4669 的正文及 9257 的条件引导主要推导。重新尝试 7818、8847、9379、9497、9509 的 spaces.ac.cn、www 和 kexue.fm 入口仍未取得全文，保留“未完整读取”标记；改读 TeaForN、Rectified Flow 等原始论文补足部分数学依据。关键定义、等式与适用条件以可读取的原始论文和明确标注的推导为准。这是围绕问题的定向检索，不是全站穷尽阅读。

**本轮新增导航（2026-09-10，第二轮）**：§10 博客补读与替代来源；§11 互信息估计及联合信息；§12 控制充分性；§13 动态坐标与可辨识性；§14 多步预测与事件时间；§15 条件得分及生成引导；§16 LEACE 与 CKA；§17 当前 G2 的解释；§18 新增文献与阅读范围。第一轮的来源表保留。

**续补导航**：§19 近似信息状态的价值误差界与有限记忆；§20 value equivalence 与 rate–distortion；§21 终点、首次事件和累计占用三种预测目标；§22 续补 6 篇文献。§18 与 §22 共登记新增 18 篇，逐项标注实际阅读范围。

**1. 博客阅读地图：每篇解决一个具体问题**

| 文章 | 主题与阅读价值 | 对本研究的连接 | 本次读取范围 |
| --- | --- | --- | --- |
| [从变分编码、信息瓶颈到正态分布：论遗忘的重要性](https://www.spaces.ac.cn/archives/6181)，2018-11-27 | 从 VAE 到信息瓶颈，理解压缩与保留目标信息的取舍 | SAE 重建的“保真”与未来交互预测的“保真”可以是不同目标 | 正文；重点为 IB / 变分推导 |
| [深度学习的互信息：无监督提取特征](https://www.spaces.ac.cn/archives/6024)，2018-10-02 | 从联合分布与边缘乘积分布的区分理解互信息学习 | 能重建激活不等于提取了最容易使用的物理变量 | 正文；重点为互信息、判别目标与负样本 |
| [TeaForN：让 Teacher Forcing 更有“远见”一些](https://www.spaces.ac.cn/archives/7818)，2020-10-27 | 条件序列分解、真实前缀与生成前缀的差别 | 一步 latent prediction 与多步自回归稳定性是不同问题 | 索引正文片段；全文未取到 |
| [CoSENT（一）：比 Sentence-BERT 更有效的句向量方案](https://www.spaces.ac.cn/archives/8847)，2022-01-06 | 训练目标与最终相似度使用方式的关系 | 稀疏重建、线性读出、未来预测和控制编辑并不优化同一种几何 | 索引正文片段；全文未取到 |
| [从局部到全局：语义相似度的测地线距离](https://www.spaces.ac.cn/archives/9368)，2022-12-07 | 局部邻近关系与全局距离；图最短路近似测地线 | 不能把 embedding 欧氏距离或二维可视化直接解释为物理差异 | 正文，包括“原理分析”“测地距离” |
| [生成扩散模型漫谈（十五）：构建 ODE 的一般步骤（中）](https://spaces.ac.cn/archives/9379)，2022-12-20 | 条件路径与边缘向量场 | 理解可采样的条件训练目标如何对应整体生成分布 | 索引公式与正文片段；全文未取到 |
| [生成扩散模型漫谈（十七）：构建 ODE 的一般步骤（下）](https://www.spaces.ac.cn/archives/9497)，2023-02-23 | ODE 构造与 Rectified Flow 相关讨论 | 区分生成路径、样本配对和真实环境轨迹 | 索引正文片段；全文未取到 |
| [生成扩散模型漫谈（十八）：得分匹配 = 条件得分匹配](https://spaces.ac.cn/archives/9509)，2023-02-28 | 条件目标与边缘目标的对应 | “用简单条件监督学习复杂分布”这一数学模式 | 索引引言片段；全文未取到，未据此复述证明 |

博客适合建立直觉，论文适合核对定理范围。一个具体例子：6024 使用标准半权重 JSD 定义时，上界应为 log 2；标准最优二分类目标是 2 JSD − 2 log 2，而不是数值上等于 KL 互信息。后两种判别量可作为学习目标，但不能把训练数值直接报告为 Shannon MI。此处是按定义核算的归一化说明。

**2. 先统一研究对象，避免不同“时间”和“状态”混在一起**

| 记号 | 含义 |
| --- | --- |
| \(\mathcal H_t\) | 截止物理时刻 t 的观察、已执行动作及任务条件的历史 |
| \(H_t\) | 从冻结策略提取的当前或最近 W 帧激活；不默认等于完整历史信息 |
| \(Z_t=f(H_t)\) | 固定编码器输出的 raw / PCA / SAE / RET-style 表示 |
| \(Y_t\) | 测量到的物理标签；现阶段实际只有 Contact 与 StableGrasp |
| \(C_t\) | 明确列出的对照信息，例如已过去的时间、proprioception、任务条件 |
| \(U_t^{(K)}\) | 从 t 开始的 K 步动作序列；必须区分计划动作、真实未来动作和实验指定动作 |
| \(\sigma\) | Flow Matching 的生成时间；不是环境时间 t |
| \(\ell\) | 网络层；也不是环境时间 t |

后文条件互信息使用同一总体分布、同一目标、同一对照信息。训练后固定编码器，在留出数据上分析；不允许目标标签或未来输入通过编码器拟合过程泄漏进测试样本。

**3. 互信息与信息瓶颈：给“更好的表示”一个准确含义**

**3.1 未来预测应先问增量信息**

直接看 \(I(Z_t;Y_{t+k})\)，可能主要测到任务身份、进度或当前状态的持续性。针对一个明确对照，问题可写为：

\[
I(Y_{t+k};Z_t\mid C_t).
\]

若想问“超过已知当前接触状态之后，还剩多少预测信息”，可再条件化 \(Y_t\)：

\[
I(Y_{t+k};Z_t\mid C_t,Y_t).
\]

这是不同的科学问题；真实 \(Y_t\) 可能是策略拿不到的特权标签。不能把所有变量机械地当成需要排除的混杂。特别是动作可能是表示影响未来的中介，给定动作之后测到的是剩余预测信息，不是总的决策相关性。

下面是适用于离散目标的直接推导。对任意输入 V，所有条件概率预测器中的最优负对数风险为：

\[
R^*_{\log}(V)=\inf_q\mathbb E[-\log q(Y\mid V)]
=H(Y\mid V),
\]

因为任意 q 的风险可分解为条件熵加上平均条件 KL。故：

\[
\boxed{R^*_{\log}(C)-R^*_{\log}(C,Z)=I(Y;Z\mid C).}
\]

它说明，理想概率预测中的风险改善与额外信息可以严格对应。实际训练只有有限模型与有限数据：若总体风险分别为条件熵加超额风险 \(\epsilon_C,\epsilon_{CZ}\)，则差值是

\[
I(Y;Z\mid C)+\epsilon_C-\epsilon_{CZ}.
\]

因此两个实际 probe 的 NLL 差既不自动等于 MI，也不自动是 MI 的下界。概率预测的统计基础见 [Gneiting & Raftery, 2007](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf)；上述条件风险等式是将标准定义应用到本问题的推导。

**3.2 对现有 Brier 协议，更准确的数学解释**

令 Y 为二元变量，\(m_C=\mathbb E[Y\mid C]\)，\(m_{CZ}=\mathbb E[Y\mid C,Z]\)。平方误差的条件期望投影给出：

\[
\boxed{R^*_{\rm Brier}(C)-R^*_{\rm Brier}(C,Z)
=\mathbb E[(m_{CZ}-m_C)^2].}
\]

证明只需展开 \(Y-m_C=(Y-m_{CZ})+(m_{CZ}-m_C)\)，交叉项的条件期望为零。这是额外条件信息改变最优事件概率的平方幅度，不是 Shannon MI。实际 Ridge、正则化、截断与有限样本仍引入超额风险；最优风险差非负不保证实际测试差非负。AUPRC 则主要衡量排序，不能替代这个概率风险解释。

这意味着现有代码完全可以先保留 Brier，准确报告“条件预测风险改善”。无须为了写互信息叙事，直接把既有指标改名或临时更换协议。

**3.3 Raw、SAE、RET 不能在相同输入上创造 Shannon 信息**

对固定确定性变换，条件数据处理不等式给出：

\[
I(Y;f(H)\mid C)\leq I(Y;H\mid C).
\]

若 f 可逆则相等。于是“RET 信息更多”不是公平比较后的准确默认说法；可能成立的是“相同预测器更容易读出”“小样本估计更稳定”“较小表示保留了更多任务相关信息”。若 RET 看 W 帧而 raw 只看一帧，则首先多了输入历史，不能把优势全归因于动态目标。

这正是 [Xu et al., A Theory of Usable Information Under Computational Constraints, ICLR 2020](https://arxiv.org/abs/2002.10689) 的价值：将观察者可用的预测函数族纳入信息定义。对线性 probe，本研究可以采用其思想，明确读出家族 \(\mathcal V\)，比较固定家族的预测风险；若用条件化扩展，应写明这是本研究的定义，不把任意 Ridge 差值称为论文原定义的精确估计。

**3.4 IB 支持“保留与未来有关的部分”，不支持无条件地越压缩越好**

经典 [Information Bottleneck](https://arxiv.org/abs/physics/0004057) 优化

\[
I(Z;H)-\beta I(Z;Y).
\]

对本问题，一种概念上的条件化改写是

\[
\min_{p(z\mid h)} I(Z;H\mid C)
-\beta I(Z;Y_{t+1:t+K}\mid C).
\]

这里“保真”由未来目标定义。充分性对应

\[
I(Y_{t+1:t+K};H\mid Z,C)=0.
\]

这仅表示 Z 保留了 H 中针对指定未来目标的全部条件信息；它没有证明 H 已包含完整物理状态，也没有证明策略使用 Z。

更直接地，对确定性 Z=f(H)，令 Y 代表指定未来目标，有

\[
I(Y;H\mid C)-I(Y;Z\mid C)
=I(Y;H\mid Z,C)
=\mathbb E_{H,C}D_{\rm KL}\!\left(p(Y\mid H,C)\|p(Y\mid f(H),C)\right).
\]

这给出一个与任务有关的“预测失真”：压缩后未来条件分布改变了多少。它区别于激活重建误差。该等式是链式法则的直接推导；实际只能通过有限预测器近似评估，不应把估计器误差省略。

若采用随机编码器 q，变分率项满足

\[
\mathbb E_H D_{\rm KL}(q(Z\mid H)\|r(Z))
=I(H;Z)+D_{\rm KL}(q(Z)\|r(Z)).
\]

所以 KL-to-prior 是 MI 上界，通常不等于 MI。[Deep Variational Information Bottleneck, ICLR 2017](https://www.alexalemi.com/publications/vib.pdf) 提供可训练变分形式。对连续确定性网络，直接写 \(I(H;f(H))\) 还可能得到无穷值；降低维数不等于有限信息码率。[Amjad & Geiger, TPAMI 2020](https://arxiv.org/abs/1802.09766) 专门讨论这些问题。因此 IB 目前更适合作为目标选择依据；若真优化码率，需要指定噪声、量化或随机编码等机制。

一个用于说明的构造例子：\(H=(N,\epsilon V)\)，其中 N 是大方差且与任务无关的扰动，V 决定未来接触，\(\epsilon\ll1\)。在一维线性重建约束下，平方误差/PCA 会偏好 N，而未来标签预测偏好 V。这证明两种目标可以冲突；它没有证明实际 VLA 具备此分解，更没有证明所有 SAE 都会丢掉 V。

**4. 条件预测与自回归：最强的补充依据是 Predictive State Representations**

**4.1 用未来实验的分布定义状态**

[Littman, Sutton & Singh, Predictive Representations of State, NIPS 2001](https://proceedings.neurips.cc/paper/2001/file/1e4d36177d71bbb3558e43af9577d70e-Paper.pdf) 把状态表为一组 action-observation tests 的预测概率，并要求它们足以确定所有测试的预测。原文还给出了有限 POMDP 与线性 PSR 的表示能力关系。它是“用未来定义状态”的直接先例。

以下是用于机器人问题的操作性改写，不是原论文已经证明的 VLA 结论：

\[
\mathcal H\sim\mathcal H'
\quad\Longleftrightarrow\quad
P(Y_{t+1:t+K}\mid\mathcal H,\operatorname{do}(U=u))
=P(Y_{t+1:t+K}\mid\mathcal H',\operatorname{do}(U=u))
\]

对选定范围内的所有 u 成立。相同预测等价类中的历史，可以压缩为同一个状态。比如，两次抓取的当前 Contact 都为真，但对同样的 lift 动作，一次会保持抓持，另一次会滑落；相同当前标签不足以将它们合并。

这也连接到 [Shalizi & Crutchfield, Computational Mechanics, 2001](https://arxiv.org/abs/cond-mat/9907176)：其 causal states 按未来条件分布对过去历史分组。这里的 “causal state” 是该理论的术语，不能仅凭名称获得机器人动作干预的因果识别。

实际只有两个标签、有限 horizon、有限动作覆盖时，最多逐步建立“针对这些交互目标的近似预测状态”。不能直接叫完整 Markov state 或完整 world model。有限 W 帧是否足够，也要有证据。

**4.2 必须分开三种预测分布**

| 分布 | 可以回答什么 | 关键限制 |
| --- | --- | --- |
| \(P^\pi(Y_{t+k}\mid H_t)\) | 现有行为策略下，未来通常怎样发展？ | 混合了环境规律和策略选择 |
| \(P(Y_{t+k}\mid H_t,U_{\rm realized})\) | 已知演示后续实际动作之后，还能预测什么？ | 未来信息条件化；不等于在线预测 |
| \(P(Y_{t+k}\mid H_t,\operatorname{do}(U=u))\) | 指定执行动作 u 后会怎样？ | 需要可识别的动作干预/相应假设和数据覆盖 |

闭环演示的未来动作可能已经响应了中间的接触或滑落，因而真实未来动作本身携带未来观测信息。仅把它拼到 probe 输入，不能获得反事实动力学模型。即使当前动作在充分历史条件下可以作无混杂处理，对整个自适应未来动作序列的朴素条件化仍需单独论证。

**4.3 自回归分解是分布恒等式，Markov 简化则需要假设**

对于事先指定的开环动作序列 u，链式法则给出：

\[
p(Y_{t+1:t+K}\mid\mathcal H_t,\operatorname{do}(U=u))
=\prod_{j=1}^K p(Y_{t+j}\mid\mathcal H_t,Y_{t+1:t+j-1},\operatorname{do}(U=u)).
\]

只有再假定某个 Z 是充分状态，才可以进一步用递推 \(p(Z_{t+1}\mid Z_t,a_t)\) 代替完整历史。在不预知未来干预的环境中，较远的未来动作不会影响较早的结果；这也不意味着可以对自适应的真实未来动作做同样删减。

一步 teacher-forced 预测输入真实历史，多步 rollout 输入自己的预测历史；其误差分布不同。[Ross et al., AISTATS 2011](https://proceedings.mlr.press/v15/ross11a.html) 对序列决策中策略改变输入分布的问题给出经典分析。它支持区分离线预测和闭环表现，不要求本项目采用 DAgger。

**4.4 预测 latent 也可能预测到错误的东西**

\(\|g(Z_t)-Z_{t+1}\|\) 很低，可能来自平滑任务进度、动作模板、episode identity，或表示塌缩。若编码器和预测器可自由优化，常量 Z 可使纯 latent 一步误差为零；缩小尺度也可降低未归一化 L1/L2 误差。EMA、stop-gradient 等是具体方法机制，不能单独当作所有设置下的防塌缩定理。

因此 RET-style 的物理意义需要由独立未来物理目标、条件增量预测与行为证据来检验。当前实现使用 L1 一步 latent 预测且没有动作输入，准确称呼是“历史激活上的预测性压缩候选”，尚不是已经建立的动作条件 PSR。

**5. 对比学习与表征几何：关键是定义哪些差别应该保留**

[Contrastive Predictive Coding](https://arxiv.org/html/1807.03748v2) 已经把历史表示、未来预测与对比目标结合起来。因此“预测未来来学习表示”本身有明确先例。

对于一个正样本和 N−1 个独立边缘负样本，InfoNCE 为

\[
\mathcal L_N=-\mathbb E\log
\frac{\exp s(Z,Y^+)}{\sum_{j=1}^{N}\exp s(Z,Y_j)},
\qquad I(Z;Y)\geq\log N-\mathcal L_N.
\]

最优分类分数对应密度比 \(p(y\mid z)/p(y)\)。对本问题作条件化扩展时，如果在每个 C 条件内按 \(p(y\mid C)\) 独立采负样本，对应的是 \(p(y\mid z,C)/p(y\mid C)\)。这解释了“在相似任务/时间背景下比较未来”的意义；随意的 hard negative、同轨迹相邻帧或近似匹配不自动满足这个 MI 下界的采样条件。

[Wang & Isola, ICML 2020](https://proceedings.mlr.press/v119/wang20k.html) 用 alignment 与 hypersphere uniformity 分析对比学习。对本研究的推论是：测几何前，应先声明要对齐什么、分开什么。球面均匀不是物理状态的公理。

若以“当前 Contact 一样”定义正样本，可能错误地拉近“稳固抓持”和“即将滑落”；若以同 episode 定义正样本，则可能鼓励记忆演示。因此更合适的几何参照可以是相同动作下未来结果分布的差异。一个候选定义是：

\[
d_{\rm pred}(i,j)=\mathbb E_{u\sim\mu}
D\!\left(P(Y_{\rm future}\mid\mathcal H_i,\operatorname{do}(u)),
P(Y_{\rm future}\mid\mathcal H_j,\operatorname{do}(u))\right).
\]

这里 \(\mu\) 指定动作测试范围，D 可取适合结果空间的分布距离。这是由预测等价思想导出的研究候选，尚未在本项目估计；其可信度依赖两侧分布是否可识别。若只比较单个二元事件，可直接比较 Bernoulli 概率，无须为了形式而构造复杂流形模型。

博客 9368 对局部距离和全局语义距离的讨论提供一个有用提醒：近邻图能近似某种测地线，仍要先有合理的局部距离。近邻图本身不会把静态 embedding 变成物理动力学。类似地，CKA、PCA、二维投影可以显示结构变化，但不能单独证明某个 contact 机制的产生。

**6. 生成模型与 Flow Matching：它最能帮助你正确解释 action expert**

**6.1 条件生成学习的是分布，而不是唯一正确的未来**

设 \(A_1\sim p_{\rm data}(A\mid c)\) 是条件 c 下的演示动作块，\(A_0\sim\mathcal N(0,I)\) 是独立噪声。考虑线性插值

\[
A_\sigma=(1-\sigma)A_0+\sigma A_1,\qquad B=A_1-A_0.
\]

以 \(\mathbb E\|v_\theta(A_\sigma,\sigma,c)-B\|^2\) 训练时，平方损失的总体最优解为

\[
\boxed{v^*(a,\sigma,c)=\mathbb E[B\mid A_\sigma=a,\sigma,c].}
\]

这是条件期望投影。展开平方可得训练风险等于 \(\mathbb E\|v_\theta-v^*\|^2\) 加一个与参数无关的条件方差项。[Lipman et al., Flow Matching for Generative Modeling, ICLR 2023](https://arxiv.org/html/2210.02747v2) 的 Theorem 1–2 给出了条件向量场聚合、连续性方程和 CFM/FM 梯度等价的正式结果。

上述动作条件写法是将该原理应用到 VLA。端点零噪声形式应理解为标准极限/在开区间内使用并满足相应正则条件；不能忽略密度与 ODE 存在性条件。向量场匹配确定的是分布路径，不保证每条生成样本轨迹沿原始训练配对的直线运动。

**6.2 生成时间与物理时间属于两个系统**

\[
\frac{dA_\sigma}{d\sigma}=v_\theta(A_\sigma,\sigma,c_t)
\quad\text{与}\quad
s_{t+1}\sim P_{\rm env}(s_{t+1}\mid s_t,a_t)
\]

前者生成动作，后者描述动作执行后的环境。一个 action-flow policy 学的是 \(p_\pi(A\mid c_t)\)；即使 world model 也采用 Flow Matching，其目标变量和条件分布仍不同。

所以 action expert 的激活更准确地记为

\[
h_t^{(\ell,\sigma)}=F(\text{observation},\text{task},A_\sigma,\sigma,\ldots).
\]

它可以同时携带世界观测、动作意图和生成过程信息。从中预测 future contact，可能利用了计划中的接近/闭合动作。这不是无价值信息，但必须与“表征了接触动力学”区分。

这直接解释了为何本地缓存需要记录 layer、denoising call、action token、noise seed 和 pooling。共同缓存只保证数据来源一致，不自动消除这些因素。Flow loss 也不是现成的 action log-likelihood；要报告分布 KL/NLL，必须有相应密度计算，而不能以 MSE 代替。

对两个二元物理标签，概率分类器已能表示所需条件分布。Flow Matching 目前最值得用于解释现有策略架构；只有扩展到连续、多模态的未来轨迹预测时，才需要判断是否值得另建生成模型。

**7. 从 Prediction 到 Control，还需要状态抽象和真正的内部干预**

**7.1 Bisimulation 给出“为什么某种压缩对控制可以够用”**

在 MDP 中，bisimulation 要求被合并的状态在每个动作下具有相同奖励，以及对下一步等价类的相同转移分布。[Zhang et al., Learning Invariant Representations for Reinforcement Learning without Reconstruction, ICLR 2021](https://arxiv.org/abs/2006.10742) 将这类结构用于无重建的控制表征学习。它说明“保留控制相关区别，而不重建全部观察”已有明确理论与方法先例。

对本项目的迁移有条件：Contact/StableGrasp 并不是完整 MDP 状态；有限数据不覆盖所有动作；稀疏奖励可能无法区分很多有物理意义的状态。奖励定义也决定了抽象适用于哪些任务。即便某个 Z 对一个任务的最优控制足够，也未必是通用 world state。

因此 world-model extension 更准确的问题是：**在同样的动作测试、物理目标和可用观察下，VLA 表示保留的是跨策略稳定的预测结构，还是当前策略下的行为规律？** 单纯看两个 latent space 的相似度不足以回答。

**7.2 后处理 probe 的 Z 不是天然的策略中介**

如果 Z 只是分析者从 H 后处理得到的，原策略并没有执行 \(Z\to a\) 这条边。单独修改保存的 Z，不会改变机器人。必须构造真实 activation intervention：

\[
Z=f(H),\qquad H'=\operatorname{Lift}(H,Z'),\qquad
a'=\pi_{\rm downstream}(H',\text{其余计算状态}).
\]

Lift 是研究方法的一部分，不能省略。[Geiger et al., Causal Abstractions of Neural Networks, NeurIPS 2021](https://arxiv.org/abs/2106.02997) 的关键是对齐高层变量与内部表示，并比较相应 interchange interventions 的结果。它支持检验明确的因果结构假设，而不只是测“扰动后输出变了”。

SAE 有线性 decoder 时，可保留重建残差：

\[
H=DZ+b+e,\qquad H'=H+D\,\delta Z.
\]

这避免了仅为了编辑特征而把所有残差 e 一起删除，但 D 的方向仍可能影响多个物理/行为变量，需要验证语义特异性。RET 的时序 encoder 没有自动附带这样的反向映射；任意 \(Z'\) 可能对应多种历史，甚至没有可信历史对应。不能直接把 RET 的 32 维向量加到策略激活上。

**7.3 干预幅度的公平比较由实际激活空间决定**

下式是直接线性代数结果：

\[
\|\delta H\|_2^2=\delta Z^\top D^\top D\,\delta Z.
\]

因此不同 decoder、不同 SAE feature 或 raw/RET 坐标中的同一个 \(\alpha\)，通常不是同强度干预。至少应报告实际 \(\delta H\) 的尺度；语义比较还需要匹配方向、位置、调用次数和同一初始条件等证据。

\(\Delta a\approx J_H\delta H\) 描述局部敏感性；有 \(\Delta a\) 不等于该方向是 contact state。闭环后出现 \(\Delta\text{success}\) 说明该干预影响结果，仍要区分目标机制与一般破坏。对应的证据顺序应是：目标状态读出按预期变化、动作响应符合具体假设、闭环物理事件随之变化，再评价任务成败。真实物理 state intervention 与“改变模型对该 state 的内部表示”也不是同一个操作。

**8. 理论对现有项目的直接解释**

以下依据 [当前实现记录](../../research/predictive-states.md)，不是本次重新运行实验所得。

| 当前事实 | 数学上能如何解释 | 尚不能据此得出 |
| --- | --- | --- |
| Contact / StableGrasp，horizon 0/1/5/10 | 指定标签及有限时间跨度上的 encoding / prediction | 完整交互事件过程、全部物理状态或 Markov 充分性 |
| StandardScaler + Ridge，按验证集 Brier 选 alpha，输出截断 | 固定读出协议下的事件预测风险 | 精确条件 MI、已校准概率 |
| 真实未来演示动作进入条件读出器 | 对真实后续动作条件化的离线诊断 | 在线动作条件预测、反事实动力学 |
| RET 取 W 帧历史，当前 predictor 无动作输入 | 历史激活上的自主 latent prediction | 已实现 action-conditioned PSR |
| PCA 历史 W×dim、RET dim、SAE 更大字典 | 同划分与共同读出规则下的初步比较 | 完全容量匹配；优势全部来自训练目标 |
| 实际 action expert 有去噪调用和 action token | 混合了感知、动作及生成过程的表示 | 单独的 physical state representation |
| 真实零改动 hook 回放通过 | 注入接口在已检查输入上保持输出 | 存在因果特征、闭环控制改善 |
| 完整真实表示比较和闭环还未完成 | 目前是理论定位与可执行接口准备 | RET/SAE 胜负或新的科学发现 |

这些理论无需转化成一套同时包含 IB、InfoNCE、FM、bisimulation 的大 loss。优先用它们澄清现有比较的 estimand：谁看到了什么、预测哪个分布、压缩保留了什么、干预究竟落在哪个计算变量上。

**9. 可支持的研究定位与阅读顺序**

| 理论簇 | 已有工作已经覆盖 | 对本项目仍需实证回答的问题 | 可以形成的贡献边界 |
| --- | --- | --- | --- |
| IB / usable information | 压缩与任务信息、有限观察者的可用信息 | 哪种冻结 VLA 表示让物理信息更易读出并跨任务稳定？ | 具体物理变量与公平读出下的结果，不是信息论新原理 |
| PSR / predictive representations | 用动作条件未来预测定义状态 | VLA hidden/SAE/RET 保留多少未来交互区别？ | 针对物理交互与策略内部表示的验证，不是首次提出预测状态 |
| CPC / contrastive geometry | 通过预测与对比学习表示 | 所学距离对应未来物理区别，还是任务/演示身份？ | 具有物理依据的几何诊断，不是首次未来对比学习 |
| Flow Matching | 用条件向量场训练生成分布 | action expert 的预测信息来自观测还是生成中的动作计划？ | 现有策略的机制分析，不是将生成 ODE 当作环境动力学 |
| Bisimulation / causal abstraction | 控制等价状态与内部因果对齐 | 可预测的物理信息是否被策略以预期方式使用？ | 有匹配对照的内部干预及闭环后果 |

这里的未回答问题是跨来源综合后形成的研究候选，不是穷尽检索后证明的文献空白。此前 VLA probe / SAE / intervention 工作仍是直接相关先例；本次数学检索不推翻已有 novelty 边界。

建议先读 PSR 和 V-information，再读 IB/VIB 与 CPC，之后读 Flow Matching，最后把 causal abstraction 与 bisimulation 作为 Control / world-model 部分的依据。这样每读一篇都能对应一个必须证明的问题。

| 原始文献 | 年份与来源状态 | 类型 | 本次重点读取 / 用途 |
| --- | --- | --- | --- |
| [The Information Bottleneck Method](https://arxiv.org/abs/physics/0004057) | 1999 方法；2000 arXiv 版本 | theory/proof | 目标 Eq.15、相关性与压缩；不引用其未核对实验 |
| [Deep Variational Information Bottleneck](https://www.alexalemi.com/publications/vib.pdf) | ICLR 2017 | method + benchmark | 变分界 Eq.13–17；KL 率项的含义 |
| [Learning Representations for Neural Network-Based Classification Using the Information Bottleneck Principle](https://arxiv.org/abs/1802.09766) | TPAMI 2020；arXiv v6 元数据 | theory/proof | 摘要与出版信息；确定性 IB 的病态性，未复核全部证明 |
| [A Theory of Usable Information Under Computational Constraints](https://arxiv.org/abs/2002.10689) | ICLR 2020 | theory/proof | §2 定义与 §3.2；固定读出器下的信息可用性 |
| [Strictly Proper Scoring Rules, Prediction, and Estimation](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf) | JASA 2007 | theory/proof | proper scoring 的定义及概率预测基础；本笔记另行推导条件风险差 |
| [Predictive Representations of State](https://proceedings.neurips.cc/paper/2001/file/1e4d36177d71bbb3558e43af9577d70e-Paper.pdf) | NIPS 2001，vol.14 | theory/proof | tests、充分统计量、Theorem 1 和 POMDP 构造 |
| [Computational Mechanics: Pattern and Prediction, Structure and Simplicity](https://arxiv.org/abs/cond-mat/9907176) | Journal of Statistical Physics 2001 | theory/proof | §IV.A Definition 5；按未来分布划分历史 |
| [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/html/1807.03748v2) | 2018 preprint；读取 2019 v2 | method + benchmark | §2.3 与附录；InfoNCE、密度比和采样条件 |
| [Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere](https://proceedings.mlr.press/v119/wang20k.html) | ICML 2020 | theory/proof | 官方摘要、引言；对比几何的两个性质 |
| [Flow Matching for Generative Modeling](https://arxiv.org/html/2210.02747v2) | ICLR 2023；读取 v2 | method + benchmark | §3 Theorem 1–2、条件路径；用于生成分布解释 |
| [Learning Invariant Representations for Reinforcement Learning without Reconstruction](https://arxiv.org/abs/2006.10742) | ICLR 2021 | method + benchmark | §3 bisimulation 定义与部分可观测假设边界；未照搬其因果最小性主张 |
| [Causal Abstractions of Neural Networks](https://arxiv.org/abs/2106.02997) | NeurIPS 2021；读取 v2 PDF | theory/proof | 方法框架、probe 反例、Figure 1 interchange interventions |
| [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://proceedings.mlr.press/v15/ross11a.html) | AISTATS 2011 | method + benchmark | 官方摘要与引言；序列决策中的输入分布变化 |

检索使用了公开方法名、博客标题及站点限定词，例如 information bottleneck / predictive state representations / usable information / contrastive predictive coding / flow matching / causal abstraction / bisimulation。优先作者原文、官方会议、作者主页与 arXiv；未向外部提交私有草稿或实验文件。未进行论文打分，也未将理论论文的结论当作本项目的实测结果。

**可检验的问题是：在相同历史和动作信息下，哪种 VLA 表示保留了对未来物理交互有用、容易读出、并能通过明确内部干预验证其决策作用的区别？** 预测充分性、可读性、因果作用可以相互支持，但必须分别取得证据。

---

**10. 第二轮博客阅读：补什么、取得了什么、还缺什么**

| 来源 | 本轮阅读状态 | 新增价值与替代路径 |
| --- | --- | --- |
| [更别致的词向量模型（二）：对语言进行建模，4669](https://www.spaces.ac.cn/archives/4669)，2017-11-19 | 读取正文，尤其 PMI 与可加性推导 | 区分共现概率与超过背景频率的关联；§11 讨论单 feature 和联合信息 |
| [生成扩散模型漫谈（九）：条件控制生成结果，9257](https://www.spaces.ac.cn/archives/9257)，2022-08-30 | 读取条件输入、近似分布、连续情形、无分类器部分 | §15 讨论为何 hidden probe 不能直接充当 action-space guidance |
| TeaForN，7818 | 重试未取到全文 | 改读 [TeaForN 原论文 §3](https://aclanthology.org/2020.emnlp-main.702.pdf)，不声称博客补读成功 |
| CoSENT，8847 | 重试未取到全文 | 几何问题改读 Zimmermann et al. 和 Kornblith et al.；两者不是 CoSENT 的复现 |
| 扩散 ODE 系列，9379 / 9497 | 重试未取到全文 | 补读 Rectified Flow 原论文 §3 的耦合与传输成本结果 |
| 得分匹配，9509 | 重试未取到全文 | §15 独立写出条件期望推导；Vincent 2011 出版页摘要可读，但作者 PDF 遇到访问验证 |

新检索到的 9262（统一扩散模型理论篇）和 11428（预测数据而非噪声）仅有索引片段，未纳入已读证据。它们分别与已有 FM 框架重叠、涉及尚未核对的参数化结论，暂不据此增加方法。

**11. 互信息：从相关特征走向可检验的信息增益**

**11.1 有意义的训练目标，也可能不是可靠的 MI 测量仪**

[Poole et al., ICML 2019](https://proceedings.mlr.press/v97/poole19a/poole19a.pdf) 分析变分 MI 界，指出 InfoNCE 的 \(\log N\) 上限及 bias/variance 取舍。[McAllester & Stratos, AISTATS 2020](https://proceedings.mlr.press/v108/mcallester20a/mcallester20a.pdf) 研究更一般的限制：从 N 个样本得到的、distribution-free 的高置信 MI 下界受 \(O(\log N)\) 量级限制。后者不是“所有 MI 估计都不可能”的定理；有分布假设或不要求这类保证时，问题不同。

对本项目的推论：先评价 held-out 条件概率风险，比新增一个高维 MINE 分数更容易解释。Contact 是二元目标，本身有

\[
I(Y;Z\mid C)\le H(Y\mid C)\le \log 2.
\]

所以不能拿“大 MI 难测”直接解释二元 Contact 的 probe 失败。这里更实际的困难是条件覆盖、稀有转变、读出器容量和轨迹内相关性。13,603 个 state 不等于 13,603 个独立重复实验，不能把 frame 数直接代入 iid 样本保证。

**11.2 单个 feature 没信号，联合表示仍可能有信息**

博客 [4669](https://www.spaces.ac.cn/archives/4669) 用 PMI 说明联合频率相对独立背景的偏离，并在朴素分解假设下讨论可加性。迁移到 SAE 时必须保留这些假设：稀疏激活不意味着各 feature 独立，更不意味着给定物理标签后仍条件独立。

下面是本笔记构造的精确反例，采用以 2 为底的对数：

\[
Z_1,Z_2\overset{\mathrm{iid}}{\sim}\mathrm{Bernoulli}(1/2),
\quad Y=Z_1\oplus Z_2;
\quad I(Y;Z_1)=I(Y;Z_2)=0,\quad I(Y;Z_1,Z_2)=1.
\]

逐 feature 排名没有找到 Contact feature，不能证明 SAE 不包含 Contact；找到一个相关 feature，也不能证明它独占该信息。比较单 atom、少量联合 atoms 和固定容量非线性 readout，可以区分单轴解释与组合编码。这里不要求穷举 1440 维组合，而是明确单特征审计的结论范围。

**11.3 条件化动作可能去掉中介，也可能揭示协同信息**

另一个本笔记构造：Z、A 为独立均匀二元变量，未来事件 \(Y=Z\oplus A\)，则

\[
I(Y;Z)=0,\qquad I(Y;Z\mid A)=1\ \mathrm{bit}.
\]

同一状态变量的意义取决于施加什么动作，action-conditioned probe 因而可能揭示无条件 probe 看不到的结构。反过来，若存在 \(Z\rightarrow A\rightarrow Y\) 的完全中介链，给定 A 后的剩余信息可以为零，尽管 Z 对未来有因果作用。

所以 G2 的 A/AC/ACY 比较衡量“给定动作计划后的剩余预测能力”；它既不是 representation 的总价值，也不是 mediation effect 的估计。零增益不能直接排除策略使用。

还需区分“加入 action 输入”与“能够表达 action-state interaction”：若 readout 只是对拼接后的 \([Z,A]\) 做线性回归，它无法表示上述 XOR。只有编码中已有交互结构，或预测器显式包含相应交互项/非线性时才可能恢复该关系。因此 action 条件下的线性读出负结果，仍受读出家族限制。

**12. 哪些预测目标可能保留控制所需信息？**

**12.1 直接先例：Rakelly et al.**

[Which Mutual-Information Representation Learning Objectives are Sufficient for Control?，NeurIPS 2021](https://arxiv.org/html/2106.07278) §3–5 区分最优策略充分性和最优 Q 函数充分性，并分析：

\[
\mathbb J_{\rm fwd}=I(Z_{t+1};Z_t,A_t),\qquad
\mathbb J_{\rm state}=I(Z_{t+1};Z_t),\qquad
\mathbb J_{\rm inv}=I(A_t;Z_{t+1}\mid Z_t).
\]

原文报告 forward information 的充分性结果，并给出 state-only 与 inverse information 可以不充分的反例。解释结论必须保留 Markov 输入、数据覆盖、分布可被精确建模、全局目标最优等理想化条件；它不是对冻结 VLA hidden、有限演示或当前 RET 的保证。

**项目推论**：action-free RET 与 action-conditioned RET 应视为两个假设。尤其

\[
\min\mathbb E\|g(Z_t,A_t)-Z_{t+1}\|^2
\]

不等于最大化 \(\mathbb J_{\rm fwd}\)：联合训练常量编码器会使这个 MSE 为零，而互信息目标还涉及表示的边缘不确定性。原论文不能用来保证 L1 latent prediction 的充分性。

可操作的用法是固定历史与容量，检验 action condition 是否改善未来事件概率，再检查跨策略/动作覆盖下是否仍成立。论文提供比较理由，实验负责判断 VLA 是否满足对应条件。

**12.2 Denoised MDP：可预测的东西仍可能与控制无关**

[Wang et al., Denoised MDPs，ICML 2022](https://proceedings.mlr.press/v162/wang22c/wang22c.pdf) 按 action controllability 与 reward relevance 区分信息，通过结构化 latent transitions 分离干扰。§2.3 指出具有时间结构的背景噪声仍可预测，所以未来预测未必排除它。

对 VLA 的推论：task clock、背景运动、演示风格都可能带来低 latent loss。即使动态表示胜过 SAE，也需要说明保留了哪种未来区别。反过来，“不可由机器人控制”不等于“可删除”：外部物体运动仍可能决定避障与抓取。能否删除依赖任务和因果结构，不能只看 action correlation。

这里借用其分类视角，不要求立刻重建完整 world model。将预测性、动作可影响性、目标相关性分别记录，有助于解释 feature 694/981 是 motor feature、phase marker 还是物理状态候选。

**13. 动态表示的数学优势：坐标选择与可辨识性**

**13.1 Koopman：同样的信息，可以在更简单的坐标中演化**

[Lusch, Kutz & Brunton, Nature Communications 2018](https://doi.org/10.1038/s41467-018-07210-0) 研究学习坐标使非线性动力学获得可用的线性表示。对自主系统 \(s_{t+1}=F(s_t)\)，Koopman 算子作用于 observable：

\[
(\mathcal K g)(s)=g(F(s)).
\]

它对函数 g 是线性的，但通常是无限维算子；不存在“任意物理系统都有精确 32-D 线性 latent”的一般保证。原文结合重建、latent linearity 和未来状态预测，而非只优化一步 latent MSE。

**候选诊断**：在每个既有表示上拟合同预算的

\[
\hat Z_{t+1}=AZ_t+Ba_t+b,
\]

再与非线性 predictor 比较。线性预测更好可能说明动态关系更容易读出，而非 Shannon 信息增加。该加性控制形式是本项目选定的基线，不是 Koopman 理论自动推出的通用受控模型。

需要检查多步递推与 direct multi-horizon prediction。接触、碰撞、滑落可能使单一平滑低维近似失效；若全程线性失败而 phase 内有效，这本身可能揭示交互阶段结构。

**13.2 Nonlinear ICA：重建成功为何不足以恢复物理因子**

[Hyvärinen, Sasaki & Turner, AISTATS 2019](https://proceedings.mlr.press/v89/hyvarinen19a/hyvarinen19a.pdf) 利用 auxiliary variable 和 generalized contrastive learning 给出 nonlinear ICA 的可辨识性条件，包括条件独立的源、可逆混合、辅助变量对源分布足够丰富的调制，以及相应模型/极限条件。

加入动作、历史或实验条件，可能有助于确定潜在因子。但对 VLA 只能作为候选依据：Contact、StableGrasp 和 phase 往往耦合，冻结网络也可能丢失信息，不满足可逆生成假设。简单拼接 action 不会自动触发该定理。

一个独立的代数说明：若 \(H=D(Z)\)，对可逆 T 有

\[
H=(D\circ T^{-1})(T(Z)).
\]

重建无法区分这些坐标。SAE 的稀疏约束会缩小允许的变换集合，却仍不能单靠 reconstruction EV 确认某轴就是 Contact。跨 seed matching 应同时看 activation、decoder direction、条件预测和干预响应，允许必要时以 subspace 为单位比较。

**13.3 对比学习恢复到什么等价类很重要**

[Zimmermann et al., ICML 2021](https://proceedings.mlr.press/v139/zimmermann21a/zimmermann21a.pdf) 的 Theorem 2 在 uniform hypersphere、指定 positive-pair 分布、可微单射生成映射及大样本极限等条件下，得到恢复至正交变换的结果；后续定理讨论其他几何。

项目推论：positive pairs 与距离函数隐含对潜在变化方式的假设。如果只恢复到旋转等价类，可能得到好的物理 subspace，却没有天然的“grasp 轴”。所以“更适合预测的几何”与“monosemantic features”应分开评价。

**14. 多步预测：还要关心事件何时发生**

**14.1 TeaForN 原文补读**

[Goodman, Ding & Soricut, TeaForN，EMNLP 2020](https://aclanthology.org/2020.emnlp-main.702.pdf) §3 在训练时堆叠 decoders，让后续 decoder 接收前一个 decoder 的输出表示并预测更远 token；推理不要求保留同样堆叠开销。它是语言生成方法，不提供机器人闭环稳定性保证。

迁移到 RET 首先区分：

| 评估 | 输入 | 问题 |
| --- | --- | --- |
| one-step teacher-forced | 真实 \(Z_t,a_t\) | 单步条件映射是否准确？ |
| direct K-step | 真实初始历史、允许的动作条件 | 初始表示是否包含远期信息？ |
| recursive K-step | 前一步预测的 \(\hat Z\) | 误差传播后是否仍能预测？ |

**14.2 可检查的误差递推**

以下是本笔记推导。若真实 latent 动态为 F，预测器为 g，固定相同动作；在所有涉及状态上模型一步误差至多 \(\epsilon\)，F 对状态为 L-Lipschitz，则：

\[
e_{k+1}\le Le_k+\epsilon,\qquad
e_K\le L^Ke_0+\epsilon\sum_{j=0}^{K-1}L^j.
\]

低平均 one-step loss 未给出一致误差界，也未保证闭环留在拟合区域。本项目尚未测量 L 或 \(\epsilon\)，不能代入想象中的常数声称稳定或安全。

**14.3 First-passage distribution：从“第 K 帧是什么”到“何时发生”**

对当前 \(Y_t=0\) 的状态定义首次 onset：

\[
T=\inf\{j\ge1:Y_{t+j}=1\}.
\]

X 为规定的初始可用信息，离散 hazard 为

\[
\lambda_j(X)=P(T=j\mid T\ge j,X).
\]

由链式分解：

\[
P(T>K\mid X)=\prod_{j=1}^{K}(1-\lambda_j(X)),\qquad
P(T=j\mid X)=\lambda_j(X)\prod_{i<j}(1-\lambda_i(X)).
\]

它不要求各时刻事件独立；生存条件已进入 hazard 定义。观测到 onset 时贡献“之前未发生、此刻发生”的似然；轨迹结束前未发生应按 right censoring 处理，不能当成永远不会发生。若截尾与未知结果相关，还需解释删失机制。

这些是本笔记直接展开的概率恒等式，不是 VLA 已有 hazard state 的证明。它们支持预测 Contact 建立时间，也能处理窗口中发生后又消失的短暂事件。

**15. 条件得分、Flow Matching 与生成引导**

**15.1 条件监督学习边缘场：后验平均**

博客 9509 未完整读取；[Vincent 2011 出版记录](https://doi.org/10.1162/NECO_a_00142) 仅用于确认 score matching/denoising 联系。以下独立写出推导，不冒充已核验的原文证明。

设腐化核为 \(q_\sigma(x\mid x_0,c)\)，可以交换微分与积分且密度为正：

\[
p_\sigma(x\mid c)=\int q_\sigma(x\mid x_0,c)p(x_0\mid c)\,dx_0.
\]

对 x 求导并除以边缘密度：

\[
\nabla_x\log p_\sigma(x\mid c)
=\mathbb E[\nabla_x\log q_\sigma(x\mid X_0,c)\mid X_\sigma=x,c].
\]

权重来自后验 \(p(x_0\mid x,c)\)，不是无条件平均样本。令条件得分为 B，\(\bar B=\mathbb E[B\mid X_\sigma,c]\)，平方投影得到：

\[
\mathbb E\|s_\theta-B\|^2
=\mathbb E\|s_\theta-\bar B\|^2+\mathbb E\|B-\bar B\|^2.
\]

分布固定时，最后一项不依赖参数。因此两种 loss 数值通常不同，但梯度和总体最优解一致。这与 §6 conditional velocity matching 共享条件期望结构；score 和 velocity 仍是不同场，其转换依赖具体概率路径。

**15.2 Rectified Flow 的传输路径不等于物理轨迹**

[Liu, Gong & Liu, Rectified Flow，ICLR 2023](https://arxiv.org/html/2209.03003) Theorem 3.5 在 rectifiability 条件下证明 rectification 不增加凸位移成本，相邻定理讨论路径拉直及 optimal coupling。

项目推论：生成空间中更直可以降低采样离散化难度，但不代表机器人轨迹更直、更安全，也不代表 Contact prediction error 更小。训练样本配对、生成耦合与环境执行轨迹是三个对象；不能据此声称 FM latent 符合牛顿动力学。

**15.3 从 classifier guidance 到内部干预，还缺坐标映射**

博客 [9257](https://www.spaces.ac.cn/archives/9257) 用 Bayes score 分解联系条件生成和引导。以本项目记号重写：

\[
\nabla_a\log p_\sigma(a\mid y,c)
=\nabla_a\log p_\sigma(a\mid c)
+\nabla_a\log p_\sigma(y\mid a,c).
\]

若 y 是未来 StableGrasp，需要对 noisy action、generation time 和 context 条件化的预测器；只在最后 denoising call 的 hidden 上训练 probe 尚不等于它。

固定其余条件，若 \(h=F(a_\sigma,\sigma,c)\)，链式法则给出：

\[
\nabla_{a_\sigma}\log q(y\mid h)
=J_F(a_\sigma)^\top\nabla_h\log q(y\mid h).
\]

直接加 hidden direction 与经 Jacobian 修改 action-space field 是不同操作；后者还需要噪声时刻校准及 score/velocity 推导。这是未来方法连接，而非当前可直接套用的 steering 公式。

**16. 既有负结果：擦除一个 probe，不等于擦除线性可用信息**

**16.1 LEACE 提供更明确的擦除目标**

[Belrose et al., LEACE（2023）](https://arxiv.org/html/2306.03819) §3–4 将指定凸损失下的 linear guardedness 与类条件均值、交叉协方差联系起来，推导最小改动的 affine erasure。以 H 为 activation、Y 为 one-hot concept：

\[
\min_{P,b}\mathbb E\|PH+b-H\|_M^2,\qquad P\Sigma_{HY}=0.
\]

保证依赖论文的预测器/损失家族和矩条件，不是“任何 accuracy 都不能超过 chance”，也不是非线性独立性或跨分布保证。

**项目推论**：让旧 probe 输出变小，不必让重训 probe 失去信息。rank-one 干预未超过 random，支持的是“当前方向、幅度、tap 和样本下未检出预期作用”；不足以否定所有线性表示机制。

更严格的后续依据是在 train split 拟合 eraser，在独立数据上重训/评估 readout，再检测 action effect。LEACE 可作为明确定义的线性基线，本笔记不改变先完成 SAE replication 的执行顺序。

**16.2 CKA 与 Ridge 衡量不同的不变性**

[Kornblith et al., ICML 2019](https://proceedings.mlr.press/v97/kornblith19a/kornblith19a.pdf) 比较表示相似度的不变性。对中心化矩阵 X、Y：

\[
\mathrm{CKA}_{\rm linear}(X,Y)
=\frac{\|Y^\top X\|_F^2}{\|X^\top X\|_F\|Y^\top Y\|_F}.
\]

Linear CKA 对正交变换与整体缩放不变，对任意可逆线性变换并不都不变。

下面是本笔记的代数推论：若 \(Z=TH\)，T 可逆，无约束线性 readout 可一一对应；但 Ridge 惩罚变为：

\[
\lambda\|w_Z\|^2=\lambda w_H^\top T^{-1}T^{-\top}w_H.
\]

因此信息不变，有限样本 Ridge 风险仍可能改变。StandardScaler 处理各坐标尺度，不等于消除所有相关性和 conditioning 差异。CKA drift、probe 改善、success 变化须分别报告；一个 CKA 值不能证明 feature birth。

**17. 对当前 G2 的直接帮助**

本节为文献推论，与 [下一阶段实验设计](../superpowers/specs/2026-09-10-predictive-interaction-state-experiment-design.md) 对接，没有新增实测结果。

| 可能的结果 | 解释 | 区分解释的证据 |
| --- | --- | --- |
| 总体 Brier 改善，changed 无改善 | 可能主要是 persistence，也可能有子集选择及小样本影响 | 预定义 risk set 的 onset/offset 目标与 CI |
| A → A+Z 有增益 | 动作计划外还有可读信息 | 加入 proprio/time/current-label 后是否仍有增益 |
| ACY → ACY+Z 无增益 | 当前 readout 家族下未检出剩余信息 | 不直接等于“未使用”；另需 activation intervention |
| PCA/RET 优于 raw Ridge | 几何、正则化或输入容量都可能贡献 | 相同历史、容量对照及 readout-family sensitivity |
| episode swap 后分数相近 | task/phase 背景可能足以解释部分结果 | donor 时间差、coverage、aligned-minus-null |
| gripper 对干预敏感但预测弱 | motor feature 与 future-state feature 可能分离 | 同 tap、同 feature family 的预测和闭环测量 |

两个统计边界：

1. 按最终是否 changed/onset/offset 筛选的 endpoint subset 是 outcome-conditioned 诊断，不自动是 prospective forecasting metric。新增 onset 模型应只按当前 \(Y_t=0\) 定义 risk set，同时保留后来发生与未发生者；offset 对当前 \(Y_t=1\) 同理。
2. same-task、nearest-time episode swap 是负控制，不等于从精确 \(p(Z\mid C)\) 抽样，不能仅凭它宣称通过严格 conditional-randomization test 或证明条件独立。

此前确认的 1,368-fit 命令运行的是**已有 endpoint 条件读出协议**，不自动包含 hazard/窗口内首次事件、LEACE 或 action-conditioned RET。保留完整固定 G2 结果；扩展另行预定义，避免把不同协议混入同一个 resumable run。

第 8 节是首轮实现快照，其“准备阶段”表述不能代替后续进展：项目后来已有 SAE pilot、完整 token cache 和 exploratory future readouts；最新结果与门禁以项目记录和实验设计为准。本轮只更新数学阅读笔记。

**18. 新增原始文献登记**

“已读”指指定章节，不把摘要当全文；优先级表示与项目问题的贴近程度，不是质量评分。

| ID | 文献与稳定入口 | 年份/来源状态 | 类型 | 本轮检查范围 | 用途 |
| --- | --- | --- | --- | --- | --- |
| N1 | [Which Mutual-Information Representation Learning Objectives are Sufficient for Control?](https://arxiv.org/html/2106.07278) | NeurIPS 2021；[正式 PDF](https://proceedings.neurips.cc/paper_files/paper/2021/file/dd45045f8c68db9f54e70c67048d32e8-Paper.pdf) | theory/proof | §3–5 定义、命题与反例，§6 设置；未逐行复核附录 | 最高：action-conditioned 动机 |
| N2 | [Denoised MDPs: Learning World Models Better Than the World Itself](https://proceedings.mlr.press/v162/wang22c.html) | ICML 2022 | pure method | PDF §2 分类与 predictive distractor；未核实验数字 | 高：预测性与控制相关性分离 |
| N3 | [On Variational Bounds of Mutual Information](https://proceedings.mlr.press/v97/poole19a/poole19a.pdf) | ICML 2019 | theory/proof | 变分界、§2.3–2.4 ceiling 与 bias/variance | 高：避免 MI 数值误读 |
| N4 | [Formal Limitations on the Measurement of Mutual Information](https://proceedings.mlr.press/v108/mcallester20a.html) | AISTATS 2020 | theory/proof | 引言、§4–5 保证范围；未完整复核证明 | 中：估计边界 |
| N5 | [Nonlinear ICA Using Auxiliary Variables and Generalized Contrastive Learning](https://proceedings.mlr.press/v89/hyvarinen19a.html) | AISTATS 2019 | theory/proof | PDF Theorem 1 假设、variability 讨论 | 高：因子可辨识性 |
| N6 | [Contrastive Learning Inverts the Data Generating Process](https://proceedings.mlr.press/v139/zimmermann21a.html) | ICML 2021 | theory/proof | PDF §3、Theorem 2/5/6 条件与等价类 | 高：geometry 与单轴解释 |
| N7 | [Deep learning for universal linear embeddings of nonlinear dynamics](https://doi.org/10.1038/s41467-018-07210-0) | Nature Communications 2018 | pure method | Results 的 Koopman 定义与三类 loss，Methods 概览 | 高：动态坐标 |
| N8 | [TeaForN: Teacher-Forcing with N-grams](https://aclanthology.org/2020.emnlp-main.702/) | EMNLP 2020 | pure method | §3、Figure 1；未核任务分数 | 中：多步训练 |
| N9 | [Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/html/2209.03003) | [ICLR 2023](https://iclr.cc/virtual/2023/oral/12626) | pure method | §3 Theorem 3.5 及 straightness/optimal-coupling 结果 | 中：生成与物理路径分离 |
| N10 | [LEACE: Perfect linear concept erasure in closed form](https://arxiv.org/html/2306.03819) | 2023 原始论文；本轮 OpenReview 验证受阻，未据此新增录用状态 | pure method | §2 guardedness、§3–4 协方差与最小扰动 | 高：擦除保证范围 |
| N11 | [Similarity of Neural Network Representations Revisited](https://proceedings.mlr.press/v97/kornblith19a/kornblith19a.pdf) | ICML 2019 | pure method | CKA 定义、Table 1 不变性；未核实验数字 | 高：drift 与 probe 区别 |
| N12 | [A Connection Between Score Matching and Denoising Autoencoders](https://doi.org/10.1162/NECO_a_00142) | Neural Computation 2011 | theory/proof | 出版信息与摘要；作者 PDF 访问验证失败 | 背景入口；§15 为独立推导 |

**检索记录与取舍**：先重试原来未读取的 5 篇博客，再以公开词组 mutual information sufficiency control、variational bounds mutual information、nonlinear ICA auxiliary variables、contrastive learning identifiability、Koopman embeddings、TeaForN、score matching denoising、LEACE、CKA、Rectified Flow 检索。按标题/arXiv ID 合并重复项；正文优先 PMLR、NeurIPS、ACL、作者原文和 arXiv。另筛到 Action-Sufficient State Representation Learning、Neural Predictive Belief Representations、Predictable MDP Abstraction 等候选，未深入核对，未列作已读依据。未采用仅有聚合站解读的技术结论，未上传项目私有材料。

这轮建议先读 **N1 → N10 → N5/N6 → N7**，分别回答“预测目标覆盖了什么”“干预擦掉了什么”“feature 轴是否可辨识”“动态目标是否让未来演化更简单”。N3/N4 界定指标解释，N8/N9/N12 支撑多步和生成部分。

**19. 预测误差在什么条件下能约束控制损失？**

**19.1 Approximate Information State：从历史压缩到价值误差界**

[Subramanian et al., JMLR 2022](https://jmlr.org/papers/volume23/20-1165/20-1165.pdf) 的 Definition 7 与 Theorem 9 提供一个比“预测未来所以有利于控制”更明确的论证。以下将原文符号改写为本项目记号，采用有限时域、未折扣版本。

令 \(Z_t=\sigma_t(\mathcal H_t)\)。对所有相关历史 \(h\) 和动作 \(a\)，要求奖励近似与下一表示的分布近似：

\[
\left|\mathbb E[R_t\mid h,a]-\hat r_t(\sigma_t(h),a)\right|\leq\epsilon_t,
\qquad
d_{\mathcal F}\!\left(
 P(Z_{t+1}\in\cdot\mid h,a),
 \hat P_t(\cdot\mid\sigma_t(h),a)
\right)\leq\delta_t.
\]

这里 \(d_{\mathcal F}\) 是 integral probability metric；相应函数尺度 \(\rho_{\mathcal F}\) 满足
\(\left|\mathbb E_\mu f-\mathbb E_\nu f\right|\leq\rho_{\mathcal F}(f)d_{\mathcal F}(\mu,\nu)\)。
例如采用适当的 Wasserstein-1 度量时，函数尺度可取 Lipschitz 常数。定义递推：

\[
\alpha_{T+1}=0,\qquad
\alpha_t=\epsilon_t+
\rho_{\mathcal F}(\hat V_{t+1})\delta_t+\alpha_{t+1}.
\]

则最优历史价值与近似动态规划价值之差至多为 \(\alpha_t\)；由该近似模型的贪心策略提升回历史空间得到的策略，其价值损失至多为 \(2\alpha_t\)。这些结论需要原文的一致误差条件、相应函数尺度有限以及规定的规划过程。

**项目推论**：RET 的潜在价值可以分成“减少未来分布误差”和“让后续价值函数更容易近似”两部分。但当前冻结 VLA 并不是这个近似模型上规划得到的策略；Contact/StableGrasp 的平均 Brier 改善也不是上述完整转移核的一致误差 \(\delta_t\)。因此不能把现有 probe 分数代入公式得到 success 保证。这个定理提供的是证据缺口清单：覆盖哪些未来变量、哪些动作和历史，以及这些变量怎样进入实际决策。

**19.2 有限历史窗口：需要过滤稳定性，不能直接认定 W=4 足够**

[Kara & Yüksel, JMLR 2022](https://jmlr.org/papers/v23/20-1152.html) 研究已知模型、有限动作与观测集合等设定，在非线性滤波稳定性条件下建立有限记忆策略的近似最优性。本轮读取摘要与引言，未复核误差界证明；不移植其收敛速率到连续动作 VLA。

**项目诊断建议**：固定预测目标、样本和调参预算，比较不同 W，并检查加入更早历史后是否还有 held-out 增益。若没有检出增益，只支持“在所测窗口和读出器下，较早历史未提供额外可读信息”。激活窗口不是观测—动作历史，平坦的曲线也可能来自读出器容量或统计功效不足，不能据此证明充分状态。

**20. World state 必须重建整个世界吗？Value equivalence 与任务失真**

**20.1 用 Bellman 更新定义必须保留的差异**

[Grimm et al., NeurIPS 2020](https://arxiv.org/html/2011.03506) 定义相对于策略集合 \(\Pi\) 和价值函数集合 \(\mathcal V\) 的 value equivalence：

\[
\mathcal T_m^\pi v=\mathcal T_{\tilde m}^\pi v,
\qquad \forall\pi\in\Pi,\ v\in\mathcal V.
\]

两个模型的转移分布可以不同，但在指定策略和价值函数上产生相同 Bellman 更新。扩大 \(\Pi\) 或 \(\mathcal V\) 会增加约束。这是相对函数集合的模型等价，不是任意两个相近 embedding 的等价。

**项目推论**：generic world state 与 policy-specific interaction state 可以转化为“表示支持多大的策略与任务函数集合”。只在单一策略下预测 contact，尚不足以区分它们；跨策略、动作扰动、不同任务回报或事件查询的迁移，才逐渐扩大已验证范围。当前两个物理标签不是完整价值函数集合，预测它们也不自动满足 Bellman 等价。

**20.2 Rate–distortion：压缩什么，取决于允许损失什么**

[Arumugam & Van Roy, 2022 preprint](https://arxiv.org/html/2206.02025) 将率失真与 value equivalence 联系起来。经典形式为
\(R(D)=\inf_{p(\tilde X\mid X):\,\mathbb E d(X,\tilde X)\leq D}I(X;\tilde X)\)。
论文使用模型之间的 Bellman 失真，形式为

\[
d_{\Pi,\mathcal V}(m,\tilde m)
=\sup_{\pi\in\Pi,\,v\in\mathcal V}
\left\|\mathcal T_m^\pi v-\mathcal T_{\tilde m}^\pi v\right\|_\infty^2.
\]

原文压缩对象是对环境模型的不确定性，不是 VLA activation；不能将其结论直接改称 hidden-state 信息瓶颈定理。

**项目适配，非原文结论**：Raw / SAE / RET 的差别可用失真对象解释：

| 失真对象 | 可以测什么 | 不能自动推出什么 |
| --- | --- | --- |
| 激活重建 | 原 hidden 在所选范数下恢复得多好 | 未来交互可预测、原策略使用该信息 |
| 未来事件分布 | 所选时间范围内 NLL/Brier、校准 | 完整动力学充分性 |
| Bellman/控制后果 | 指定策略与任务函数的更新误差或闭环结果 | 跨所有策略、任务的普适状态 |

固定维度与稀疏度是工程容量对照，不等于固定 Shannon bit rate。若后续使用“rate–distortion curve”一词，需要定义随机编码、量化码率或可计算的码长；否则称 dimension–performance 或 sparsity–performance curve 更准确。

**21. 三种“预测未来”目标：终点、首次事件、累计占用**

**21.1 首次事件需要处理删失**

[Gensheimer & Narasimhan, PeerJ 2019](https://arxiv.org/html/1805.00917) 的离散时间生存模型以条件 hazard 构造个体似然，并处理右删失。这为 §14.3 提供原始方法来源；机器人事件定义与时间离散方式仍需另行规定。

**项目适配**：令 \(T_e\) 为当前 risk set 中样本的首次事件步数，\(\lambda_j=P(T_e=j\mid T_e\geq j,Z_t,C_t)\)。若首次事件在 j 被观测到，其似然是
\(\lambda_j\prod_{i<j}(1-\lambda_i)\)；若完整观察完第 c 步仍无事件、随后记录结束，其生存似然是 \(\prod_{i\leq c}(1-\lambda_i)\)。该似然使用条件非信息性删失等假设；若记录结束由相关失败状态触发，需要显式建模终止/竞争事件，不能一律当普通右删失。

因此不能把缺失的未来标签填成“事件未发生”。先定义哪些 episode 终止是任务事件，哪些只是记录截断，再决定评价样本和似然。

**21.2 Successor features 描述的是累计未来，不等于首次发生概率**

[Carvalho et al., Predictive representations, 2024](https://arxiv.org/html/2402.06590) §2.3–2.5 回顾 successor representation/features。以下采用下一状态产生奖励的记号，给出项目适配：

\[
\psi^\pi(s,a)
=\mathbb E_\pi\!\left[
\sum_{j=0}^{\infty}\gamma^j\phi(S_{t+j+1})
\mid S_t=s,A_t=a\right],\qquad 0\leq\gamma<1.
\]

若奖励满足 \(r(S_{t+1})=w^\top\phi(S_{t+1})\)，则线性期望直接给出
\(Q^\pi(s,a)=w^\top\psi^\pi(s,a)\)。这里 \(\phi\) 是有界 cumulants；该等式依赖奖励表示和策略条件，不保证 hidden \(Z_t\) 已经是充分的 \(s\)。

**项目推论**：令某个 cumulant 为 StableGrasp 指示量，预测对象便是折扣后的稳定抓取占用时长；令其为状态转换事件指示量，则得到累计事件数。两者一般都不是“未来首次成功概率”。

| 问题 | 数学目标 | 可以区分的行为 |
| --- | --- | --- |
| K 步后是否接触？ | \(P(Y_{t+K}=1\mid Z_t,C_t)\) | 终点状态 |
| K 步内是否首次接触？ | \(P(T_e\leq K\mid Z_t,C_t)\) | 接触发生时间；须定义当前 risk set |
| 未来能稳定抓持多久？ | \(\mathbb E[\sum_{j=1}^{K}\gamma^{j-1}Y_{t+j}\mid Z_t,C_t]\) | 短暂抓住与持续抓稳 |

这三种量在“先接触、后脱落”的轨迹上可以明显不同。它们给出不同的预测目标，不应混用同一个“future-state accuracy”名称；任何目标的改善仍需独立的干预与闭环证据才能支持控制结论。

**22. 继续补读登记与阅读顺序**

| ID | 文献与入口 | 年份/来源状态 | 本轮读取范围 | 与项目的连接 |
| --- | --- | --- | --- | --- |
| N13 | [Approximate Information State for Approximate Planning and Reinforcement Learning in Partially Observed Systems](https://jmlr.org/papers/v23/20-1165.html) | JMLR 2022 | PDF §3.2 Definition 7、Theorem 9、相关证明与 Remark 11；未通读 83 页 | 预测近似如何进入价值损失界 |
| N14 | [The Value Equivalence Principle for Model-Based Reinforcement Learning](https://arxiv.org/html/2011.03506) | NeurIPS 2020；[正式 PDF](https://proceedings.neurips.cc/paper/2020/file/3bb585ea00014b0e3ebe4c6dd165a358-Paper.pdf) | 定义、策略/函数集合的等价关系 | generic 与 policy-specific 的数学边界 |
| N15 | [Between Rate-Distortion Theory & Value Equivalence in Model-Based Reinforcement Learning](https://arxiv.org/html/2206.02025) | 2022 arXiv；未另行确认发表版本 | §1–3 与 §4 开头的率失真、模型失真定义 | 比较重建失真、预测失真与控制失真 |
| N16 | [Near Optimality of Finite Memory Feedback Policies in Partially Observed Markov Decision Processes](https://jmlr.org/papers/v23/20-1152.html) | JMLR 2022 | 摘要、引言与设定；未复核定理证明 | 有限窗口的条件与诊断动机 |
| N17 | [A Scalable Discrete-Time Survival Model for Neural Networks](https://arxiv.org/html/1805.00917) | PeerJ 2019，7:e6257；[正式 PDF](https://peerj.com/articles/6257.pdf) | Methods 的 hazard、生存函数与似然部分 | onset 时间及右删失 |
| N18 | [Predictive representations: building blocks of intelligence](https://arxiv.org/html/2402.06590) | 2024 arXiv 综述；未另行确认发表版本 | §2.3–2.5 successor features 与 cumulants | 折扣未来占用，区别于首次事件 |

续检使用 approximate information state performance bound、finite memory POMDP filter stability、value equivalence rate distortion、discrete time survival likelihood、successor features cumulants 等公开词组。N15 与同作者的 Deciding What to Model（arXiv:2206.02072）是不同文献，未合并或借用后者结果；N18 是综述入口，不作为本项目 novelty 的证明。

若只精读三篇，优先 **N1 → N13 → N14**：先厘清预测目标的充分性，再看近似误差到价值损失的条件，最后确定模型需要保留哪些任务差异。N10 支撑因果擦除边界，N17 支撑事件时间目标。当前 G2 仍回答有限读出器下的条件预测增益；这些补充用于解释结果和形成后续可检验问题，不代表扩展实验已经实现或完成。

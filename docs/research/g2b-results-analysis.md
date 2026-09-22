# G2b 事件预测结果分析

分析日期：2026-09-11。仅分析已有报告、检查当前实现和重算标签覆盖；未训练或重新拟合模型。

**最新结论（补入 task/task_mlp 结果）：StableGrasp onset K=5 的 Ridge B1/B3 在两个任务均优于 AC，且各自均优于三组 swap，已满足此前约定的小规模 RET pilot 点估计投入条件。episode-cluster CI 仍缺，不能宣布统计确认或正式训练比较门禁全部通过。offset K=10 仍未通过 null 检查。**

## 最新逐任务复核

新增来源：[task Ridge](../../outputs/predictive_states/smolvla_event_readouts_task/report.json)、[task MLP](../../outputs/predictive_states/smolvla_event_readouts_task_mlp/report.json)。分别 complete=true，104/488 条记录；新 MLP 文件含 104 条 Ridge 和 384 条 MLP 候选。以下旧章节保留为首次汇总分析，关于“缺逐任务风险”的描述已由本节更新；缺逐样本、episode 风险和 CI 的限制仍成立。

逐条核对新报告 task macro 与两个 task Brier 的算术平均，488 条均一致。新旧测试 task-macro 差异最大为 7.45e−9，属于此精度下的舍入量级；本次主要补充了任务拆分，不是新的独立数据复制。仍按三个 seeds 的外部 validation task-macro 均值选择 MLP weight decay，不按测试集选参。

### 两个预定主目标：Ridge 逐任务结果

正增益为 AC Brier−aligned Brier；null 范围为三组 swap Brier−aligned Brier，不是 CI。

| 目标 | Arm | object/4 增益 | spatial/8 增益 | object/4 null 优势范围 | spatial/8 null 优势范围 |
| --- | --- | ---: | ---: | --- | --- |
| StableGrasp onset K=5 | B1 当前 mean PCA | +0.005685 | +0.005299 | [+0.003267, +0.004761] | [+0.000570, +0.009986] |
| StableGrasp onset K=5 | B2 当前 first PCA | +0.001315 | +0.003845 | [−0.000606, +0.000254] | [−0.003011, +0.004848] |
| StableGrasp onset K=5 | B3 历史 first PCA | +0.000985 | +0.005194 | [+0.000406, +0.001137] | [+0.008875, +0.015222] |
| StableGrasp offset K=10 | B1 当前 mean PCA | +0.002026 | +0.002410 | [−0.000175, +0.004874] | [−0.020966, −0.004419] |
| StableGrasp offset K=10 | B2 当前 first PCA | −0.002420 | +0.004978 | [−0.002482, −0.001784] | [−0.001945, +0.017717] |
| StableGrasp offset K=10 | B3 历史 first PCA | −0.005166 | +0.010673 | [−0.004367, −0.002220] | [−0.003621, +0.032621] |

onset K=5 的 B1 是最清楚的简单基线：两个 task 的增益幅度接近；B3 也满足点估计条件，但在 object/4 的改善较小。不能因 B3 使用历史就预设其更接近动态状态。offset K=10 的 B1 虽然两 task 都改善，却在 spatial/8 输给全部 swap；B2/B3 的总体正增益掩盖了 object/4 的退化。

### MLP：总体 seed 一致不等于跨任务一致

| 主目标 | Arm | object/4 三 seeds 平均增益 | spatial/8 三 seeds 平均增益 | 解释 |
| --- | --- | ---: | ---: | --- |
| StableGrasp onset K=5 | B1 | +0.001167 | −0.000095 | 接近零且跨 seed 不稳定 |
| StableGrasp onset K=5 | B2 | −0.002180 | −0.002325 | 两 task 均值均退化 |
| StableGrasp onset K=5 | B3 | −0.002229 | +0.007149 | 总体正增益由 spatial/8 带动 |
| StableGrasp offset K=10 | B1 | +0.015085 | +0.000441 | spatial/8 seed 间方向不稳 |
| StableGrasp offset K=10 | B2 | +0.005209 | +0.009193 | spatial/8 存在反向 seed，且缺 MLP null |
| StableGrasp offset K=10 | B3 | −0.003882 | −0.008709 | 两 task 平均均退化 |

因此撤回“MLP 历史表示具备跨任务稳定收益”这一可能的推断；旧分析只观察到每个 seed 的总体均值为正。现有结果支持读出器、任务与事件之间存在异质性，不支持“非线性读出总能释放历史信息”。

### 次要目标与资源决策

- Contact onset K=5 的 B1 两 task 都改善，但 object/4 未超过所有 swap；K=10 同项在 spatial/8 未超过所有 swap。
- Contact offset K=10 的 B3：object/4 为 −0.004285，spatial/8 为 +0.017190，总体优势明显由单任务驱动。
- StableGrasp onset K=10 的 B3 两 task 均改善且超过各自全部 swap；保留为次要一致性证据，不把它事后升级为主目标。
- 此前的中等投入条件只要求一个预定主事件跨两个 task 同方向且优于 null，Ridge onset K=5 的 B1/B3 现在满足。可据此准备冻结 tap、数据、预算的有界 AE/RET pilot，不等于确认性发现，也不意味着策略因果使用已成立。
- 最新建议额外要求查看 episode 不确定性与少数 episode 驱动问题，这一部分仍未完成。新目录依然仅有 report.json，没有逐样本预测/模型/episode 风险，不能计算有效 paired CI。下一次执行必须保存这些产物；不要将两个 task 或三个 seeds 当成大量独立样本。
- MLP 的全输入缩放、内部 early stopping、Ridge 选参口径等实现边界仍见下文 §6；本次增加 task 指标没有证明这些问题已修正。

本轮只复核结果并更新分析。没有启动 RET、修改训练代码或更改 ccfa.yaml 执行权限。

---

以下 §1–7 为首次总体分析记录，数值仍有效；阶段判断以以上最新逐任务复核为准。

## 1. 来源与口径

- [Ridge 报告](../../outputs/predictive_states/smolvla_event_readouts/report.json)：complete=true，104 条记录。
- [MLP 报告](../../outputs/predictive_states/smolvla_event_readouts_mlp/report.json)：complete=true，488 条，包含与前一报告逐条完全一致的 104 条 Ridge，以及 384 条 MLP 候选。这不是 Ridge 的独立复制。
- MLP 候选数为 8 目标×4 arms×4 weight decays×3 seeds，不是独立数据集数。
- B0=AC；B1=AC+mean-current PCA32；B2=AC+first-current PCA32；B3=AC+first-history PCA32。均使用冻结 480-D token cache。
- 下表 Brier 为 task macro；正 Δ=Brier(B0)−Brier(Bi)。null 优势=Brier(swap)−Brier(aligned)。
- MLP 在每个 arm/目标下，按三个 seeds 的 validation_task_macro_brier 均值选 weight decay，再报告三个 seeds 的测试风险均值。没有按 test 选参数或最好 seed；这里不是 ensemble prediction 风险。
- 两个已有输出目录都只有 report.json，没有逐样本 score、模型权重或逐 task/episode 风险。不能从总体均值恢复逐任务效应及 paired CI。`event_readouts.py` 已补充逐 task 指标输出；需要用新入口重跑，旧报告不被覆盖。

## 2. Ridge：完整八个目标

| 标签 | 事件 | K | 测试正/负窗口 | B0 Brier | B1 Δ | B2 Δ | B3 Δ |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| Contact | onset | 5 | 69/493 | 0.057599 | +0.004510 | −0.002121 | −0.001639 |
| Contact | onset | 10 | 124/438 | 0.065553 | +0.005398 | −0.002346 | +0.001496 |
| Contact | offset | 5 | 61/582 | 0.081467 | −0.001626 | −0.000463 | +0.000494 |
| Contact | offset | 10 | 124/519 | 0.120365 | −0.006230 | +0.003010 | +0.006452 |
| StableGrasp | onset | 5 | 71/600 | 0.047645 | +0.005492 | +0.002580 | +0.003089 |
| StableGrasp | onset | 10 | 130/541 | 0.057333 | +0.008847 | +0.001801 | +0.003389 |
| StableGrasp | offset | 5 | 56/468 | 0.100109 | +0.002496 | −0.001299 | −0.002739 |
| StableGrasp | offset | 10 | 109/415 | 0.138676 | +0.002218 | +0.001279 | +0.002753 |

mean-current 更有利于 Contact onset；first-history 在 Contact offset K=10 更好。没有一种聚合方式在所有事件上占优。StableGrasp offset 也不等于 drop，正常释放同样会产生该标签事件。

## 3. Ridge：相对三个 episode-swap 的优势范围

| 标签 | 事件 | K | B1 null 优势 | B2 null 优势 | B3 null 优势 |
| --- | --- | ---: | --- | --- | --- |
| Contact | onset | 5 | [0.000964, 0.003446] | [−0.002703, 0.000138] | [−0.004735, −0.001507] |
| Contact | onset | 10 | [0.000643, 0.004839] | [−0.006603, 0.003365] | [−0.005409, −0.000691] |
| Contact | offset | 5 | [−0.002842, −0.000022] | [−0.000961, 0.000473] | [0.000458, 0.001740] |
| Contact | offset | 10 | [−0.006994, −0.004964] | [0.000166, 0.004323] | [0.005484, 0.008758] |
| StableGrasp | onset | 5 | [0.001918, 0.007373] | [−0.001809, 0.002551] | [0.004641, 0.008179] |
| StableGrasp | onset | 10 | [−0.001882, 0.011061] | [−0.005602, 0.003331] | [0.001392, 0.009977] |
| StableGrasp | offset | 5 | [0.001940, 0.008398] | [−0.004071, 0.004761] | [−0.003653, 0.004743] |
| StableGrasp | offset | 10 | [−0.008046, −0.002297] | [−0.001993, 0.007967] | [−0.003603, 0.014127] |

这些是随机化 seeds 的范围，不是置信区间。StableGrasp onset K=5 的 B1/B3 均优于三个总体 swap；offset K=10 没有一个 arm 满足该条件。onset K=10 的 B1 虽然增益最大，却输给其中一个 swap，不能只按增益挑选。

## 4. MLP：validation 选参后的完整结果

| 标签 | 事件 | K | MLP B0 Brier | B1 Δ | B2 Δ | B3 Δ | WD：B0/B1/B2/B3 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Contact | onset | 5 | 0.064260 | +0.005830 | +0.002156 | +0.000732 | .001/0/.0001/0 |
| Contact | onset | 10 | 0.068568 | +0.007088 | −0.000729 | −0.002865 | 0/.01/.001/.0001 |
| Contact | offset | 5 | 0.081964 | −0.003284 | −0.003672 | −0.006862 | .01/.0001/.001/.01 |
| Contact | offset | 10 | 0.134841 | +0.009773 | +0.006092 | +0.006429 | .01/.001/.01/.0001 |
| StableGrasp | onset | 5 | 0.038984 | +0.000536 | −0.002253 | +0.002460 | .001/.01/.01/.0001 |
| StableGrasp | onset | 10 | 0.048529 | −0.013769 | −0.002666 | −0.014075 | .0001/0/.001/.01 |
| StableGrasp | offset | 5 | 0.097923 | −0.002085 | −0.002529 | −0.005342 | .01/.01/.001/.01 |
| StableGrasp | offset | 10 | 0.141799 | +0.007763 | +0.007201 | −0.006296 | .01/0/.0001/.01 |

两个主事件的逐 seed paired 增益如下；这些 seeds 不是独立任务重复。

| 目标 | Arm | seed 0 | seed 1 | seed 2 |
| --- | --- | ---: | ---: | ---: |
| StableGrasp onset K=5 | B1 | −0.001617 | +0.001724 | +0.001502 |
| StableGrasp onset K=5 | B2 | −0.002381 | −0.007026 | +0.002649 |
| StableGrasp onset K=5 | B3 | +0.002834 | +0.001767 | +0.002779 |
| StableGrasp offset K=10 | B1 | +0.009827 | +0.020484 | −0.007021 |
| StableGrasp offset K=10 | B2 | +0.013052 | +0.011476 | −0.002925 |
| StableGrasp offset K=10 | B3 | −0.007706 | +0.015523 | −0.026704 |

StableGrasp onset K=5 的 MLP B0=0.038984，已低于 Ridge B1=0.042153。Ridge 的部分表示优势可能与有限线性可访问性有关；但实现差异意味着这不是纯粹的容量因果消融。MLP B3 将均值进一步降至 0.036524，三个 seeds 相对 MLP B0 都为正，值得追踪。B1 的剩余平均增益很小，且一个 seed 为负。

StableGrasp onset K=10 的 MLP B1/B3 在三个 seeds 均劣于 B0；offset K=10 的 B1/B2 平均改善也不稳定。不能解释为 MLP 普遍优于 Ridge，或历史普遍有用。MLP 未运行自身的 null，不能借用 Ridge null 支持 MLP 独有结论。

## 5. 已重算的逐任务事件覆盖

本轮使用原 StateBank 与现有 event_labels 重算标签，不重新训练。全部八个目标在两个测试任务均有正负窗口，每个任务有 5 episodes。两个主目标：

| 目标 | task | 正窗口 | 负窗口 | 含正窗口 episodes |
| --- | --- | ---: | ---: | ---: |
| StableGrasp onset K=5 | libero_object/4 | 25 | 276 | 5 |
| StableGrasp onset K=5 | libero_spatial/8 | 46 | 324 | 5 |
| StableGrasp offset K=10 | libero_object/4 | 43 | 325 | 5 |
| StableGrasp offset K=10 | libero_spatial/8 | 66 | 90 | 5 |

offset K=10 的任务阳性率约为 11.7% 与 42.3%，异质性很大。71 个 onset 正窗口、109 个 offset 正窗口不等于独立事件数，重叠窗口可能指向同一事件。独立事件段数尚未重算。覆盖成立不等于两个任务都取得预测增益。

本批数据的每个标签/事件在 K=5 与 K=10 的 risk masks 差异均为 0，实际共享 eligibility。不过代码只检查所请求 horizon 的标签有效性，并未一般性地强制 K=10；报告声明对其他缺失分布不自动成立。

## 6. 当前实现与原设计的差异

检查对象：[event_readouts.py](../../interaction_vla/representation_study/libero/event_readouts.py)。报告没有保存入口源码 hash，因此本轮检查不等于验证运行时源码版本。

1. 没有保存逐样本预测与模型。报告 limitations 虽写了 episode clustering，实际没有 clustered CI，不能宣称聚类推断已完成。
2. 当前 MLP 使用 sklearn 内部随机训练行划分的 15% validation 做 early stopping，未按 episode/task 分组。外部 validation 用于本次 WD 选择，不控制训练停止。这不是 test 泄漏，但重叠窗口可能使内部停止准则偏乐观，且不同于原设计。
3. Ridge 对全输入执行 train-only StandardScaler；MLP 直接接收 AC+PCA scores，没有对应的全输入标准化。生成 PCA 前的缩放不等于对 MLP 所有输入缩放，因此两类读出差异还包含尺度/优化因素。
4. Ridge 仍按 validation 窗口平均 Brier 选 alpha，不是计划的 task-macro 准则。
5. MLP 用 np.errstate(all="ignore") 包住 fit/predict，再检查最终参数和概率有限。有限输出不能证明中间数值过程无异常，也不能宣称全过程 RuntimeWarning 严格检查通过。
6. null 在 risk set 内选择 donor；报告的 elapsed mismatch p95 为 1.0–1.7 秒，已达到或超过预测跨度。它是粗略 task/time 对照，不是精确 phase 匹配或条件随机化检验。aligned 胜出也可能包含更精细的时间信息。

这些差异不抹去本批测量值，但将其限定为探索性结果。本轮未修改实验代码、未覆盖原始结果、未自动重跑 MLP。

## 7. 结论与下一步

### 7.1 逐任务审计结果（新输出）

逐任务输出位于 `outputs/predictive_states/smolvla_event_readouts_task/`，MLP 扩展位于 `outputs/predictive_states/smolvla_event_readouts_task_mlp/`。新报告与旧报告的总体 Ridge 指标一致，说明逐任务输出没有改变拟合协议。

StableGrasp onset K=5 的 Ridge B1（AC + PCA-32 mean-current）满足 §11 的投入信号条件：

| task | B0 task Brier | B1 task Brier | B0−B1 | B1 相对 3 个 swap 的优势范围 |
| --- | ---: | ---: | ---: | ---: |
| `libero_object/4` | 0.026582 | 0.020897 | +0.005685 | +0.003267 … +0.004761 |
| `libero_spatial/8` | 0.068708 | 0.063409 | +0.005299 | +0.000570 … +0.009986 |

两个 task 均为正向，且 B1 的 aligned Brier 均低于三个 episode-swap null。因而该结果可以解除“是否值得做有界 G3/RET pilot”的资源门禁，但只代表一个预先指定事件和一个 Ridge 表示臂的探索性投入信号，不是统计确认，也不证明策略因果使用了该特征。

StableGrasp offset K=10 不满足同一条件：B1 在两个 task 方向为正，但均未稳定优于所有 null；B2/B3 在两个 task 上方向不一致或 null 优势不稳定。MLP validation 选参后也没有产生一个稳定复现该门禁的跨 task 表示臂，因此不能用 MLP 候选替代 Ridge null 证据。

| 科学问题 | 当前判断 |
| --- | --- |
| 是否有值得追踪的事件预测信号？ | 有探索性支持，优先 StableGrasp onset K=5 的 B1/B3 |
| 是否在两个任务均改善？ | StableGrasp onset K=5 的 Ridge B1 是；offset K=10 否定性/混合 |
| 是否通过 RET pilot 投入门槛？ | 通过有界 G3/RET pilot 的资源信号；不等于科学确认或完整 RET 训练许可 |
| 是否证明动态表示优于静态表示？ | 否；只有 PCA current/history，尚无 RET 对比 |
| 是否证明原策略使用这些信息？ | 否；无新干预或闭环证据 |

逐 task 指标已补齐并确认与旧总体指标一致；下一步可在冻结 B1/StableGrasp onset K=5 配置后，计算固定 task 内 episode-cluster CI，再决定 G3 pilot 的最小预算。由于当前报告仍没有逐样本预测或模型权重，尚未完成 cluster bootstrap，也不应把该投入信号写成显著性结论。

若同时修正标准化、early stopping 或选参准则，应另设新协议，不能把新旧结果拼为同一个 paired comparison。先解决 Ridge 门槛审计，再决定 RET pilot；MLP 特有信号需要其自身 null。独立确认仍需要配置冻结后的新留出数据。

当前 A 是演示观察上的策略计划 chunk，未来来自演示轨迹；本批结论限于该数据分布的条件预测，不是执行计划动作后的动力学预测或 do(action) 效应。结果解释参照[实验设计 §11](../superpowers/specs/2026-09-10-predictive-interaction-state-experiment-design.md)与[数学依据](vla-mathematical-foundations.md)。

# `formation_0` 情境依赖探索（2026-09-29）

## 问题与停止规则

在同一批 development StateBank 状态上，比较冻结 SmolVLA 的 `formation_0` 干预相对 16 条 matched-random 方向的动作作用，在 simulator `gripper_target` 接触状态与可比较的无接触参照状态之间是否不同。情境差定义为：

\[
\Delta_{ctx}=U_{contact}-U_{reference},
\]

其中每个状态的候选作用是 `formation_0 - median(16 matched-random)`，匹配变量仅使用 `frame_index`、robot state 和 `gripper_target_distance`，不使用动作效果或目标事件。统计单位是 task 内 source interaction episode；CI 为 task-stratified、episode-cluster bootstrap。该分析是离线动作作用，不是闭环成功率或模型内部“Contact 特征”的证明。

## 已执行

- checkpoints：`005000`、`025000`；同一 64 个状态、同一噪声/候选数组与 StateBank。
- 主匹配：同 episode 优先，否则同 task 的最近标准化参照；39 对，37 对同 episode，平均匹配距离 3.615。
- 敏感性 A：只允许同 episode；37 对，平均距离 3.635。
- 敏感性 B：匹配距离 ≤3.0；13 对，4 tasks、5 interaction episodes。
- 工程检查：两个 checkpoint 的输出均 `complete`，effect hash 与 StateBank 绑定，数组有限，no-op 约束未被改动。

## 结果（主匹配；task-macro gap，95% bootstrap CI）

| checkpoint | 前 10 步 full RMS | gripper RMS | 完整 50-step plan | translation | rotation |
|---|---:|---:|---:|---:|---:|
| 5k | +0.004084 [0.002057, 0.006111] | +0.011822 [0.006720, 0.016924] | +0.004292 [0.003082, 0.005501] | +0.000299 [-0.000234, 0.000833] | +0.000114 [-0.000043, 0.000271] |
| 25k | +0.012837 [0.011377, 0.014296] | +0.035544 [0.031196, 0.039892] | +0.005178 [0.004514, 0.005841] | +0.000279 [-0.000059, 0.000617] | −0.000135 [-0.000337, 0.000066] |

同 episode 敏感性仍给出正的 full/gripper/plan gap（5k：+0.004546/+0.013423/+0.005003；25k：+0.012770/+0.035798/+0.005116），因此总体趋势不是由两条跨 episode donor 造成的。距离 ≤3.0 时只剩 13 对，CI 很窄但每个 task 的 episode 数不足；它只能作为方向一致的探索性检查，不能作为独立确认。

## 当前结论

结果支持一个**待复核的情境依赖候选**：`formation_0` 的离线动作作用在接触状态相对无接触参照更大，主要由 gripper 分量驱动；translation/rotation 没有稳定证据。5k 和 25k 都出现该模式，25k 的 full/gripper gap 更大。

但本轮不能声称“物理交互特征已被发现”或“策略在闭环使用该特征”，原因是：

1. 这是 development 数据，且至少一个 task 只有一个 interaction episode；报告中的 bootstrap `status` 因此是 `inconclusive`，不能当作确认门禁通过。
2. 平均匹配距离约 3.6 个标准化单位，参照并非严格近邻；≤3.0 只保留 13 对。
3. 候选相对随机方向的动作 RMS 是功能响应指标，不是内部编码、因果闭环效用或成功率指标。

## 独立 episode 补采后的结果

保持 `formation_0`、late tap、dose=0.5、16 条 matched-random 控制和共同噪声不变，只新增 Spatial 9 的 `demo_9` 状态；与原有 `demo_15` 状态去重后，合并数据为 93 states、57 matched pairs、55 same-episode pairs、8 个 interaction episodes。4 个 task 的 bootstrap status 均为 `measured`。

| checkpoint | full RMS gap | gripper RMS gap | plan RMS gap | translation | rotation |
|---|---:|---:|---:|---:|---:|
| 5k | +0.004415 [0.002143, 0.006688] | +0.013442 [0.008305, 0.018544] | +0.003699 [0.001559, 0.005839] | −0.000017 [−0.000761, 0.000711] | +0.000028 [−0.000166, 0.000222] |
| 25k | +0.003871 [0.000414, 0.007329] | +0.011542 [0.002583, 0.020502] | +0.000955 [−0.001332, 0.003283] | −0.000020 [−0.000600, 0.000561] | +0.000067 [−0.000191, 0.000325] |

这一步使 full/gripper 的情境差从样本不足的探索性信号升级为 development-level、task-stratified 的可测结果；plan、translation、rotation 仍不稳定。平均匹配距离降至 3.204，但仍不是严格近邻，因此不能直接声称闭环功能使用。

## 决策

本结果足以继续一个**有限的验证步骤**，不支持立即投入 RET、单 stage 大规模扫描或闭环预算：

1. 冻结 `formation_0`、late tap、dose 和 16-control 面板；不要再根据结果调参。
2. 补足每个 task 至少 2 个独立 interaction episodes，并预先保留未查看 episodes。
3. 在相同配对合同下复跑 full/gripper/plan，优先确认 gripper 效应是否仍具有情境差，而不是追求整体 RMS 变大。
4. 只有验证集上情境差、匹配质量和 task-stratified CI 同时稳定，才进入单一 flow-stage 的离线定位；闭环 rollout 仍需另行通过 paired episode 与 task-stratified bootstrap 门禁。

独立 episode 补采已满足 development gate，下一步可以冻结参数进入单一 flow-stage 离线定位；闭环仍保持未启动。

产物（服务器）：

- 主分析：`.../acquisition/context_dependence_20260929/full_005000`、`full_025000`
- 同 episode：`.../acquisition/context_dependence_20260929/same_episode_005000`、`same_episode_025000`
- 距离敏感性：`.../acquisition/context_dependence_20260929/caliper3_005000`、`caliper3_025000`
- 补采合并：`.../acquisition/context_dependence_20260929/merged_005000`、`merged_025000`

## 单 flow-stage 定位结果（2026-09-30）

在相同 93-state merged effects 上，逐 stage 比较 fixed-point velocity response 的情境差。5k 的正向区间主要位于 stage 0–1（分别 +0.000557、+0.000673）；25k 的正向区间主要位于 stage 6–9（分别 +0.000419、+0.000770、+0.001160、+0.001949）。stage 1–3 在 25k 反而为负，stage 7 在 5k 也为负。这些结果不是一个跨 checkpoint 稳定的单 stage。

因此本 gate 的结论是：**flow-stage 作用位置随训练发生重组/迁移的迹象，但尚未得到可冻结的单一 stage。** 不能事后选择 5k 的 stage 1 或 25k 的 stage 9 直接进入闭环；否则会把 checkpoint-specific 结果误报为一般机制。下一步若继续，应预先定义跨 checkpoint 的 stage 集合或使用 checkpoint-specific 的验证合同，并在未查看样本上确认；当前不启动定向闭环。

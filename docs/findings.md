# 已确立的事实（证据台账）

更新：2026-10-05。只收录有原始结果支撑的观察；每条注明证据位置和不能推出什么。实验过程、协议细节和被放弃的路线保存在 git 标签 `pre-restructure` 中。

## 被研究的对象

- **模型：** 自训练 SmolVLA 谱系，seed=1000，LIBERO 官方配方，AdamW，lr 1e-4，batch 32，共 25k 步。**全量微调**：`train_expert_only=false`、`freeze_vision_encoder=false`。保存了 5k、10k、15k、20k、25k 五个 checkpoint。见 [lineage.json](results/libero_smolvla_acquisition_timeline/lineage.json)。
- **推理：** 只看当前帧（单帧观测），每次预测 50 步动作块、执行前 10 步，flow 采样 10 步。
- **状态集合：** StateBank 来自 20 个任务、100 条演示，共 13,603 个状态，带接触、稳定抓取、阶段等仿真标签，已通过审计。见 [libero_state_bank_formal](results/libero_state_bank_formal/README.md)。

## F1. 能力在 5k→15k 之间形成，并且不同任务的形成方式不同

每个 checkpoint 评测 160 个 episode（LIBERO Spatial 0–3，每个任务 40 个初态）。成功率依次为：

| 5k | 10k | 15k | 20k | 25k |
|---:|---:|---:|---:|---:|
| 12.5% | 59.4% | 83.8% | 82.5% | 86.3% |

稳定抓取次数从 31/160 升到 146/160，曲线形状与成功率相同。逐任务看并不单调，例如 task 0 在 15k 是 35/40，到 20k 和 25k 都是 30/40。
证据见 [acquisition_timeline](results/libero_smolvla_acquisition_timeline/README.md)。

**不能推出：** 只有一个训练 seed，checkpoint 也只有 5 个，无法确定"转变"发生在哪一步。

## F2.（已降级）"Contact 可读性不变"测的是一个输入里本来就有的变量

方法细节见 [探针与干预审计](notes/probe-intervention-audit-20261005.md)。

- 报告的 0.187–0.200 是 MSE gain。这个指标的上限等于标签方差，约 0.246，所以这些数字相当于 R² ≈ 0.76–0.81。只用机器人本体状态是 0.214，R² ≈ 0.87，比隐藏层还高。
- 特征来自 action expert，在 50 个动作 token 上取平均，取最后一个去噪 stage；状态是演示状态而非策略自己的状态；验证集只有 4 条独立 episode，没有 CI。
- 把本体状态和时间作为条件后，隐藏层在任何 checkpoint 都没有增量。

**可以说的：** Contact 由输入决定，所以从 5k 起就能读出，这不意外。
**不能说的：** "能力形成期间内部表示不变"。这个实验没有测与能力相关的变量，而且指标已经接近饱和。

## F3.（已降级）语义方向的干预实验：两次设计无效，一次结论不确定

- **Protocol v3 与官方 positive control：** 干预点在 `action_in_proj` / `action_time_mlp_out`。这里只编码带噪动作和时间步，没有观测通路。干预只在最后一个去噪步施加，对最终动作的影响要乘以 dt = 0.1。**这种设计测不出模型是否使用了该信息，结果作废。**
- **自训练谱系的 Contact 子空间：** 在 expert 中层或后层，用 Ridge 方向去掉 50% 投影，只改一层，32 个状态、4 条独立 episode，只测离线动作。力度、擦除方式和样本量都不足。**结论不确定，不能算负结果。**
- task 1 状态 22–25 的闭环 4/4 只有 4 个状态，没有统计意义。

## F4. 不用标签找到的方向有小而稳定的动作作用，但从 5k 起就存在

- **SAE（Protocol v5，官方模型）：** 8 个候选特征中有 4 个通过动作效应门槛。属于 pilot，未做闭环。见 [sparse_features](results/libero_smolvla_sparse_features/README.md)。
- **`formation_0`（late tap）：** 相对 16 条匹配随机方向，前 10 步动作 RMS 差在 5k 为 +0.0028，在 25k 为 +0.0055。效应主要来自夹爪维度。5k 时 8/8 个演示已经为正。
- **接触与非接触状态的情境差：** 补充独立 episode 后，5k 为 +0.0044，25k 为 +0.0039，CI 都大于 0。见 [context-dependence 笔记](notes/context-dependence-20260929.md)。

**不能推出：** 这些都是离线动作差异，量级为千分之几，和闭环成功率之间的关系还没有建立。

## F5. 自训练 25k 在同一批 40 个任务和初态配对上优于官方 checkpoint

自训练 25k 成功 38/40，官方 30/40；配对差 +20pp，95% CI 为 [+7.5, +35.0]。两者的接触率和抓取率接近，差别主要出现在抓取之后。
见 [official_recipe_25k](results/libero_smolvla_official_recipe_25k/README.md) 和 [official_reference_v2](results/libero_smolvla_official_reference_v2/README.md)。

## F6. 恢复能力取决于任务，而且训练数据里几乎没有恢复示范

- 自然掉落很少：25k 是 4/40，其中只有 2 次恢复。
- **强制掉落配对试跑（25k）：** 稳定抓起后强制张开夹爪 8 步。
  - Spatial 0：open 分支 0/10 完成，3/8 再次抓稳；对应 baseline 为 8/10。
  - Spatial 1：open 分支 9/10 完成，10/10 再次抓稳。
  - Spatial 1 的再次抓取发生在扰动后约 39 步以上，跨了多个动作块，所以不是旧动作块惯性执行的结果。
- 训练数据中，Spatial 0 和 1 各 45 条演示，分别只有 5 条和 4 条出现多次夹爪闭合。
- 见 [feedback_outcome_pilot](results/libero_feedback_outcome_pilot/README.md)。

**不能推出：** 5k 和 10k 的结果还在服务器上，没有拉回。Spatial 0 的失败是否属于物理上不可恢复，还没有逐条看视频确认。

## 汇总：哪些事实目前站得住

站得住的有四条：
- 能力在 5k→15k 之间形成，而且各任务的形成方式不同（F1）。
- 不用标签找到的方向在离线动作上有小而稳定的作用（F4）。
- 自训练的 25k 在同一批配对上优于官方模型（F5）。
- 被推离轨迹后能否恢复，取决于任务（F6）。

"模型早就知道、只是还不会做"目前**只是一个假设**，还不是事实。原因是：可读性的测量对象选错了（Contact 是输入变量），指标也已经饱和；干预实验要么设计上无效，要么力度不足。要检验这个假设，需要按 [审计](notes/probe-intervention-audit-20261005.md) 第 3 节重新测。具体安排见 [研究问题](research-questions.md) 的 Q0。

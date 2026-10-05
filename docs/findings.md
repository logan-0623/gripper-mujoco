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

## F2. 物理交互信息很早就能读出，之后几乎不变

- 在 expert-late 层，Contact 的线性可读性五个点依次为 0.187、0.201、0.198、0.198、0.200（验证集 MSE gain）。成功率同期从 12% 涨到 86%。
- 只用机器人本体状态读 Contact，gain 就有 0.214。把本体状态和时间作为条件后，隐藏层在任何 checkpoint 都不再提供额外信息。
- StableGrasp 在 20k、25k 有小幅条件增益，但逐演示留一检验只有 5/11 个为正，5k 时也已经有信号。

**这意味着：** 用线性探针能看到的"模型知道什么"，在能力形成期间没有明显变化。
**不能推出：** 这不说明模型不使用这些信息，也不说明非线性的读出不变。

## F3. 沿"有语义的方向"编辑内部表示，效果不超过随机方向

以下几次独立尝试结论一致：

- **Protocol v3**（早期 SFT 谱系）：StableGrasp 定向干预在 4 个 checkpoint 上都没有超过同范数随机方向。见 [protocol_v3](results/libero_smolvla_protocol_v3/README.md)。
- **官方 SmolVLA**（task 0 成功 9/10）：Contact 和 StableGrasp 在 `action_expert_input` 可读，但秩 1 擦除对动作的影响小于随机方向。见 [positive_control](results/libero_smolvla_positive_control/README.md)。
- **自训练谱系**：Contact 候选方向相对随机的动作效应约为 ±1e-4 RMS。在 task 1 状态 22–25 上做闭环，baseline、Contact 干预、随机干预都是 4/4 成功。

**不能推出：** 秩 1 线性编辑无效，不代表信息没有被非线性地使用。

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

## 汇总：这些事实指向什么

F1、F2、F3 放在一起是一个反差：**能力在 10k 步内从 12% 涨到 84%，而能用探针读出、用编辑验证的内部"知识"在这段时间里几乎没有变化。** 所以模型在这段时间学到的，更可能是**怎么做**（动作映射，以及在自己产生的状态分布上保持稳定），而不是**知道什么**。

F6 给出了同一个现象的行为版本：同一个模型，被推离演示轨迹后，能否回到可以成功的状态，取决于任务。

这两点是 [研究问题](research-questions.md) 的出发点。

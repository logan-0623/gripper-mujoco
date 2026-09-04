# Interaction-Centric VLA Representation Study

本项目研究一个比“Graph 输入是否优于 Flat 输入”更基本的问题：

> **机器人策略内部形成了什么物理交互表示，这些信息只是可以被读出，还是实际参与了动作生成，并最终帮助闭环控制？**

研究对象是语言条件机器人操作策略。主要实验使用 **LIBERO + SmolVLA**；ACT 与显式 Interaction Graph 保留为受控机制实验。Graph 在当前项目中不是必须输入策略的网络模块，而是一套用于描述物理交互的统一测量语言。

```text
RGB + wrist RGB + robot state + language
                    │
                    ▼
                  VLA latent
                    │
       ┌────────────┼────────────┐
       ▼            ▼            ▼
   Accessible   Functionally   Closed-loop
   可被解码       Used          Useful
                  影响动作        影响任务结果
```

这三个层次必须分开测量。probe 能读出某个因素，不表示策略一定使用它；改变 latent 后动作发生变化，也不表示该变化有利于任务成功。

## 1. 这里的 representation 是什么

机器人学习中的 “representation” 并不是单一概念。它可以指策略的输入、网络内部 latent，也可以指动作的输出形式。

| 层次 | 常见表示 | 代表路线 | 它解决的问题 |
|---|---|---|---|
| 原始观测 | RGB、多视角/腕部 RGB、语言、关节或末端状态 | [OpenVLA](https://arxiv.org/abs/2406.09246)、[SmolVLA](https://arxiv.org/abs/2506.01844) | 从通用视觉语言输入直接学习动作 |
| 3D/空间 | point cloud、depth、相对位姿、3D token | [FP3](https://arxiv.org/abs/2503.08950) | 显式保留机器人操作所需的三维几何 |
| 对象中心 | object slot、目标/容器特征、对象级 token | [Object-Centric Representations](https://arxiv.org/abs/2505.11563) | 将任务相关实体与背景、纹理和干扰物分离 |
| Affordance | 接触点、关键姿态、轨迹草图、可供性计划 | [RT-Affordance](https://arxiv.org/abs/2411.02704) | 用轻量中间目标连接感知、规划与执行 |
| 图与关系 | entity、relation、scene graph、状态转移 | [Compose by Focus](https://arxiv.org/abs/2509.16053) | 表达“谁与谁处于什么关系，下一步应改变什么关系” |
| 连续动作块 | 一次预测未来一段动作 | [ACT](https://arxiv.org/abs/2304.13705) | 减少逐步预测误差并表达多步运动 |
| 连续生成 | diffusion / flow-matching action expert | [π0](https://arxiv.org/abs/2410.24164)、[SmolVLA](https://huggingface.co/blog/smolvla) | 从噪声直接生成连续动作轨迹 |
| 离散动作 token | 每维离散化或频域压缩 token | [FAST](https://arxiv.org/abs/2501.09747) | 让语言模型式自回归训练适配高频连续动作 |

当前主流 VLA 通常让多模态 Transformer **隐式学习**交互结构，而不是显式输出 scene graph。近期工作也已经广泛使用 probe、激活干预和 sparse autoencoder 研究这些内部表示，例如 [Not All Features Are Created Equal](https://arxiv.org/abs/2603.19233)、[Sparse Autoencoders for VLA Models](https://arxiv.org/abs/2603.19183)、[Event-Grounded SAE](https://arxiv.org/abs/2605.17204) 和 [VLA-Trace](https://arxiv.org/abs/2605.30117)。因此，本项目的目标不是声称“首次分析 VLA latent”，而是用同一套物理交互变量，严格区分它们的可访问性、动作相关性与闭环作用。

## 2. Interaction Graph 在本项目中的角色

早期项目把 Graph 作为策略输入：

```text
视觉/状态 → Interaction Graph → ACT → continuous action
```

这能够检验结构化归纳偏置是否有用，却无法判断 VLA 自己学到了什么。现在 Graph 被改成 privileged measurement vocabulary：

```text
LIBERO simulator state ──► 物理交互标签
VLA hidden activation  ──► probe / intervention / rollout
                              │
                              └── 比较标签、动作与任务结果
```

统一 ontology 包含六类因素：

| 因素 | 含义 | 当前定义 |
|---|---|---|
| `Entity` | 当前任务涉及谁 | target、receptacle/support 与相关 distractor；结果必须与 task-ID/instruction shortcut 比较 |
| `Geometry` | 实体之间在哪里 | gripper→target、target→goal 的相对平移、rotation-6D 与距离，不使用绝对世界坐标作为主标签 |
| `Contact` | 是否发生物理接触 | 来自 MuJoCo/robosuite contact，而不是从 RGB 猜测 |
| `StableGrasp` | 是否形成稳定抓取 | 双指接触、有限相对位姿漂移，并且目标与末端共同运动或已经离开支撑面；contact 本身不够 |
| `Phase` | 当前处于哪个操作阶段 | 由接触、稳定抓取、支撑关系和目标谓词触发，例如 approach、contact、lift、transport、place、release |
| `NextRelation` | 下一步应建立或解除什么关系 | 例如 near→contact、contact→stable_grasp、stable_grasp→off_support、near_goal→inside/on；不是“下一帧 Phase” |

`Recovery` 没有从普通成功 demonstration 中伪造出来。它只允许在未来显式构造 perturbation/recovery 轨迹时加入。

## 3. 核心测量框架

### Accessible：信息能否被读出

冻结 VLA，以 latent 为输入训练 cross-fit linear probe。只有 probe 在 held-out task/episode groups 上超过最强的 majority、task ID、instruction 或 normalized-time shortcut，并且 clustered confidence interval 通过门限，才记为 `accessible=true`。

这测量的是：latent 是否包含一个简单读出器可以访问的信息？

### Functionally used：策略是否沿该信息方向生成动作

对 held-out state 使用只由训练 fold 学到的因素方向进行 latent intervention，并与相同 rank、token 位置、L2 norm 和近似激活尺度的随机方向比较：

$$
U_f = \Delta a_{factor} - \Delta a_{matched\ random}.
$$

主要动作指标是 first executed action，分别报告 translation、rotation 和 gripper；完整 action chunk 只作次要分析。

### Closed-loop useful：该信息是否影响任务结果

需要从相同 LIBERO initial state、环境 seed 和 policy noise 运行 paired rollouts，并比较 success、grasp loss、drop、premature release、transport failure 与 placement failure。当前项目还没有完成这一层，因此不能把离线 action displacement 写成 control utility。

## 4. 实验基础设施

### Shared LIBERO State Bank

所有 checkpoint 共用同一批状态和同一份 privileged labels，禁止为不同训练阶段独立抽样或重新标注。

当前正式 State Bank 包含：

- 20 个 LIBERO Spatial/Object tasks；
- 100 个 episodes；
- 13,603 个 states；
- global RGB、wrist RGB、language、robot state 与 action；
- simulator replay reference、相对几何、contact、stable grasp、phase 与 next relation；
- task-group 和 episode-group splits，所有帧始终跟随所属 episode；
- deterministic replay audit 与 12 条人工检查 timeline。

Replay 只接受误差满足门限的 episode。成功构建的 100 个 episode 接受率为 100%；候选 episode 的总体接受率为 83.3%，因此这是一套经过筛选的可重放状态库，不是任意 LIBERO demonstration 的无损镜像。

### SmolVLA 路径与固定 taps

SmolVLA 接收多视角 RGB、语言和 robot state。VLM 产生上下文表示，flow-matching action expert 在多次 denoising 中生成长度为 50 的连续动作块。

| Tap | 精确位置 | 代表什么 |
|---|---|---|
| `vision_output` | `model.vlm_with_expert.embed_image` 输出 | 每个相机视角的视觉 token |
| `multimodal_fusion` | 第一次 prefix/prefill 的 normalized hidden state | 图像、语言与状态融合后的上下文 |
| `action_expert_input` | 最终 denoising call 的 `model.action_time_mlp_out`，raw shape `[50, 720]` | 送入 action expert 计算路径的动作条件表示 |
| `pre_action` | 最终 denoising call 中 `model.action_out_proj` 的输入 | 最靠近连续动作输出的表示 |

正式 pooling 在看结果前固定。`action_expert_input` 和 `pre_action` 对 50 个 action positions 求均值；视觉与 prefix 表示只对有效 token 求均值。

### 训练阶段

Protocol-v3 比较 8 个同 runtime checkpoint：

```text
Pretrained
D25@16k
D50@16k ── D50@32k
D100@16k ── D100@33k ── D100@50k ── D100@66k
```

`D25 ⊂ D50 ⊂ D100`，三个 SFT run 都从同一个 base checkpoint 独立开始。当前 recipe 是 expert-only SFT，vision encoder 被冻结。因此结果描述的是**冻结上游视觉表示时，下游动作路径如何重组**，不能直接推广到任意 end-to-end VLA fine-tuning。

## 5. 已完成的实验

| 实验 | 状态 | 回答的问题 |
|---|---|---|
| ACT Graph-v2，3 policy seeds、每条件 60 rollouts | `formal_evidence` | 显式 interaction structure 是否可能改善小型连续控制策略？ |
| ReflectVLM Graph pretraining | `pilot_complete` | 更接近 teacher 的语义 Graph 是否自然转化为更高控制成功率？ |
| Recovery RL v2 calibration | `failed_gate` | 当前 recovery distribution 是否适合比较 SFT→RL plasticity？ |
| LIBERO State Bank | `formal_evidence` | 能否在所有 checkpoint 上使用完全相同、可重放的物理标签状态？ |
| SmolVLA Protocol-v3 cross-fit probes | `formal_evidence` | 哪些 interaction factors 在哪些 tap、哪些 SFT 阶段可被读出？ |
| StableGrasp longitudinal linear intervention | `failed_gate` | StableGrasp 的线性 probe 方向是否比 matched random 更影响动作？ |
| Official SmolVLA positive control | `failed_gate` | 上述失败是否只是因为自训练 checkpoint 根本不会做任务？ |
| Protocol-v5 label-blind sparse features | `pilot_complete` | 不强迫 latent 对齐人工标签时，能否找到稳定且真正影响动作的内部特征？ |

状态含义：`formal_evidence` 表示协议、报告和完整性检查均已归档；`pilot_complete` 表示实验已完成并能指导下一步，但证据范围不足以支撑主结论；`failed_gate` 是被保留的负结果，不等于代码失败。

## 6. 观察到的现象

### 6.1 ACT：Graph 有一定帮助，但 Graph accuracy 不是 control utility

| ACT 输入条件 | 60-rollout success |
|---|---:|
| Flat | 30.0% |
| Teacher Graph | 35.0% |
| Predicted Graph，random-init estimator | 40.0% |
| Predicted Graph，Reflect-pretrained estimator | 41.7% |

这个实验支持一个有限结论：结构化交互信息可以成为有用的 inductive bias。但 Teacher Graph 没有成为性能上界，Reflect estimator 虽然更接近 teacher，也没有稳定超过 random-init estimator。这说明：

$$
\text{Graph correctness} \neq \text{policy usability}.
$$

它不证明 predicted Graph 天生优于 ground truth。更合理的候选解释是分布尺度、连续平滑性、时序跳变和 ACT 的输入兼容性不同。

### 6.2 SmolVLA：Contact、StableGrasp 与 Phase 在早期 SFT 后进入动作路径

下面是 `action_expert_input`、episode-group cross-fit 的 **probe utility**。正值且 confidence interval 通过门限才算可访问；不同因素使用不同原始指标，因此这里展示相对各自最强 shortcut 的 utility，而不是把 AUPRC、F1 和 MAE 直接比较。

| Checkpoint | Geometry | Contact | StableGrasp | Phase |
|---|---:|---:|---:|---:|
| Pretrained | −0.0333 | −0.0575 | −0.0753 | −0.0018 |
| D25@16k | −0.0292 | +0.0392 | +0.0806 | +0.1768 |
| D50@16k | −0.0319 | +0.0342 | +0.0868 | +0.1712 |
| D100@16k | −0.0324 | +0.0320 | +0.0825 | +0.1520 |
| D100@66k | −0.0317 | +0.0347 | +0.0843 | +0.1582 |

原始指标也呈现同一趋势。例如 StableGrasp AUPRC 从 pretrained 的 0.781 上升到 D25@16k 的 0.937，D100@66k 为 0.941；Phase Macro-F1 从 0.331 上升到约 0.49–0.51。Geometry 没有超过预注册 shortcut。

因此现有证据支持：

- frozen upstream 条件下，Contact、StableGrasp 和 Phase 的线性可访问性在早期 SFT 快速出现；
- 更大数据覆盖和更长优化没有继续显著提高这些因素的可访问性，后期主要表现为 plateau 或小幅非单调变化；
- raw action representation 在后期仍持续漂移，所以 representation drift 与 semantic accessibility 不是同一个量；
- Entity 被 task/instruction shortcut 严重混淆，NextRelation 在 held-out task folds 中存在 class-support 问题，这两项不能形成主结果。

### 6.3 线性可访问不等于沿线性方向控制

在四个 longitudinal checkpoints 上，StableGrasp 的 fold-held-out rank-one intervention 均没有超过 same-rank matched-random action effect。关键 paired change 为：

| 对比 | $\Delta U$ | 95% episode-cluster CI |
|---|---:|---:|
| Pretrained → D25@16k | −0.000257 | [−0.000353, −0.000168] |
| D25@16k → D100@16k | +0.000079 | [+0.000016, +0.000141] |
| D100@16k → D100@66k | −0.000047 | [−0.000102, +0.000006] |

为了排除“自训练模型不会做任务”的混淆，又使用官方 `lerobot/smolvla_libero` checkpoint 做 positive-control screen。该模型在 `libero_spatial` task 0 上成功 9/10 次，但结果仍然是：

- Contact AUPRC 0.963，超过最强 shortcut 的 utility 为 +0.044；
- StableGrasp AUPRC 0.962，utility 为 +0.105；
- Phase Macro-F1 0.463，utility 为 +0.130；
- StableGrasp 与 Contact targeted erasure 都比 matched random **更弱**，对应 CI 分别为 [−0.000173, −0.000072] 和 [−0.000205, −0.000097]。

这否定的是“pooled linear-probe direction 就是策略实际使用的控制方向”，不是“策略完全没有使用抓取或接触信息”。

### 6.4 无监督 SAE：模型存在稳定、跨任务且强动作相关的非线性特征

Protocol-v5 不使用 Contact、StableGrasp 或 Phase 标签选特征。它在官方 9/10-success checkpoint 的 720D `action_expert_input` 上训练 3 个 Top-K sparse autoencoders：dictionary width 1440、每个 state 激活 32 个特征、5000 updates。候选特征只按跨 seed decoder/activation 一致性、task/episode coverage 与 entropy 冻结，然后才查看语义标签和动作。

最新 server run 的主要结果是：

- 3 个 SAE 的 held-out explained variance 为 0.9974–0.9979；
- 1440 个 dictionary atoms 中只有 187–217 个在 validation 中存活，约 81.7% inactive，说明 dictionary collapse 明显；
- seed-0 特征被归为 92 个 broad、149 个 task-concentrated、10 个 episode-concentrated、12 个 intermediate 和 1177 个 inactive；
- 8 个冻结候选的相邻帧变化/episode 内随机打乱变化比值为 0.051–0.132，而 broad features 的中位数为 0.525；由于该指标没有参与候选选择，这是候选具有较强时序连续性的探索性证据；
- 冻结得到 8 个跨 seed、跨 task/episode 候选，其中 4 个在 BH correction 后比 matched random 更影响 first action。

| SAE feature | Target − random action effect | 95% episode-cluster CI | 主要动作分量 |
|---:|---:|---:|---|
| 517 | +0.0801 | [+0.0612, +0.1028] | gripper magnitude |
| 694 | +0.2293 | [+0.1854, +0.2760] | gripper magnitude，另有 translation |
| 977 | +0.0898 | [+0.0716, +0.1088] | gripper magnitude |
| 981 | +0.3600 | [+0.3214, +0.3948] | gripper magnitude 与 translation |

所有四个特征的 episode-cluster mean 都为正，activation norm ratio 为 0.968–0.989，没有依靠 gross OOD shift 获得效应。它们没有造成 gripper command sign flip，主要改变连续 gripper 输出强度，因此更像 motor gating/intensity feature，而不是简单的“开/关夹爪神经元”。

候选冻结后进行的语义关联显示，feature 981 与 Phase 最相关（$\eta^2=0.502$），feature 694 对 Contact/StableGrasp 的关联相对更高，但这些关联都不等于清晰的一对一语义概念。另一个 feature 561 对 Contact/StableGrasp 关联更强，却没有通过动作因果门限。这个现象初步表明：

$$
\text{semantic association ranking} \neq \text{causal action ranking}.
$$

Protocol-v5 目前是通过 kill gate 的 pilot。其 compact machine-readable evidence 尚需从服务器归档到仓库，因此还没有提升为 paper-level formal evidence。

## 7. 当前能够成立的结论

现有结果形成了一条逐步收紧的证据链：

1. **显式结构有时能帮助控制。** ACT pilot 中 predicted Graph 相对 Flat 提高了平均成功率，但提升不由 Graph prediction accuracy 单独解释。
2. **SFT 选择性改变 action-proximal representation。** Contact、StableGrasp 和 Phase 在早期 expert-only SFT 后变得可线性读出，而 Geometry 没有出现同样变化。
3. **可访问性不等于线性因果坐标。** 即使在成功的官方 SmolVLA 中，Contact/StableGrasp probe 方向也没有比 matched random 更强地影响动作。
4. **策略内部仍存在真正动作相关的结构。** label-blind SAE 找到跨 seed、task 和 episode 可重复的特征，其中 4 个对动作的影响显著超过 matched random。
5. **人工 ontology 与 policy-native features 只部分对齐。** 当前最有价值的研究问题不再是“Graph 对不对”，而是“人可解释的物理变量如何对应模型实际采用的控制坐标”。

最稳妥的项目结论是：

> **物理交互信息可以在 VLA latent 中变得可访问，但策略实际使用的控制表示未必沿人工定义因素的线性方向组织。成功策略表现出跨任务、时序平滑且动作相关的稀疏特征，这些特征与 Contact、StableGrasp 和 Phase 只有部分对应。**

## 8. 目前不能得出的结论

- 不能说 Graph 普遍优于 Flat；ACT 结果只是一套受控任务中的机制证据。
- 不能把 official checkpoint 的 9/10 写成 LIBERO benchmark success；它只是 `libero_spatial` task 0 的成功策略门检。
- 不能说线性 probe 方向失败就证明策略不使用 Contact 或 StableGrasp。
- 不能说 SAE feature 已经对 closed-loop success 有益；目前只测了离线 action sensitivity。
- 不能把一个 seed-0 reference dictionary 中发现的特征直接当作稳定神经机制；还需要跨 SAE seed 的独立 intervention replication。
- 不能声称完整发现了 SmolVLA 的 feature inventory；当前 SAE 有明显 dead-feature/dictionary-collapse 问题。
- 不能声称结果适用于端到端 VLA fine-tuning、RL plasticity 或其他 VLA 架构。

## 9. 下一步

当前不训练新 SFT、不启动 RL，也不扩展 ontology。最小的判别性实验是：

1. 在另外两个 SAE seeds 上独立复现 feature 694 和 981 的 action effect；
2. 为每个 feature 使用多个正交 matched-random directions，并加入同频率/同范数的非候选 SAE atom control；
3. 只有离线效应通过上述复现，才从相同 LIBERO initial states 做 paired original / feature ablation / matched-control rollouts；
4. 闭环门限通过后，再单独设计这些 policy-native features 的 longitudinal SFT trajectory。

这一步将决定论文是继续研究“interaction semantics 如何映射到 policy-native causal features”，还是停止该机制路线并 pivot。

## 10. 代码与复现入口

服务器从零配置、数据盘、离线 Hugging Face cache、LIBERO assets 和训练/续跑命令见 [SERVER_RUNBOOK.md](SERVER_RUNBOOK.md)。科学状态与 claim gate 见 [ccfa.yaml](ccfa.yaml)。已归档的轻量证据包包括：

- [LIBERO State Bank](docs/results/libero_state_bank_formal/README.md)
- [SmolVLA Protocol-v3](docs/results/libero_smolvla_protocol_v3/README.md)
- [Official SmolVLA positive control](docs/results/libero_smolvla_positive_control/README.md)

Protocol-v5 的执行入口：

```bash
export MUJOCO_GL=egl
export HF_HOME=/root/autodl-tmp/gripper-mujoco-hf-cache
export HF_LEROBOT_HOME="$HF_HOME/lerobot"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

CONFIG=configs/representation_study/libero_smolvla_linux_cuda.yaml

.venv-lerobot/bin/python -m interaction_vla.representation_study \
  libero features discover --config "$CONFIG"

.venv-lerobot/bin/python -m interaction_vla.representation_study \
  libero features intervene --config "$CONFIG" --max-states 512 --batch-size 32

.venv-lerobot/bin/python -m interaction_vla.representation_study \
  libero features report --config "$CONFIG" --max-states 512
```

测试：

```bash
HF_HOME=/tmp/gripper-mujoco-pytest-hf-cache \
PYTHONPYCACHEPREFIX=/tmp/gripper-mujoco-lerobot-pycache \
  .venv-lerobot/bin/python -m pytest -q \
  tests/interaction_vla/representation_study/libero
```

模型权重、optimizer state、原始数据、latent arrays、per-state action cache 和 rollout videos 不进入 Git。仓库只保存代码、配置、split/manifest、统计报告与少量审计可视化。

# Franka Panda 真实接触抓取与 Graph/Flat 实验设计

日期：2026-08-02

## 目标

建立一个完整的 Franka Panda 多物体 pick-and-place 实验：机械臂通过 IK 与关节
position/impedance controller 执行 7D Cartesian delta-pose policy action；双指夹爪只能依靠 MuJoCo
碰撞、摩擦和重力抓起物体；实时 viewer 同时显示 Agent、Wrist、Side、Top 四个视角。

该实验用于验证受控的 representation hypothesis：在使用相同物理状态、训练数据、控制器、
动作头、参数预算与 paired evaluation cases 时，显式 Graph representation 是否比 Flat
vector 更 object-aware、gripper-aware，并在多物体与拥挤交互中取得更好的闭环行为。

本阶段不需要 VLA。Agent/Wrist/Side/Top 四路同步 RGB-D 会被显示并可记录；其中
Agent/Wrist 共同定义未来 dual-camera observation contract，但图像不进入当前 Graph/Flat policy。

## 对先前方案的替代关系

本规格完全替代 `2026-08-02-franka-four-view-visualization-design.md` 中的确定性吸附方案。
不得通过 weld、equality attachment、mocap object、持有状态下逐帧写 object qpos，或任何
“物体跟随夹爪”的脚本实现抓取。

现有 `KinematicTabletopEnv`、checkpoint、报告和 GIF 保留为旧 representation baseline，
但不作为真实物理实验的 authoritative backend。真实物理结果写入独立输出目录，不能与旧
kinematic 数值混写。

## 分阶段范围

### 第一阶段：受控物理验证

- 完整 Franka Panda、7 个 arm joints、2 个 finger joints。
- 6D end-effector SE(3) motion 与 parallel-gripper state 的完整 7D action。
- 相同尺寸、质量、惯量和摩擦参数的圆角方块或圆柱。
- 2、3 物体训练；4、5 物体 count-OOD；4、5 物体 crowded-OOD。
- Scripted physics expert 正式采集，keyboard teleoperation 作为演示和 recovery 补充。
- Graph、Flat、edge-shuffled Graph 对照。
- 四视角 dashboard、native viewer、GIF、四路同步 RGB-D 记录和 Agent/Wrist dual-camera observation。

### 第二阶段：物体类别泛化

在第一阶段 controller 与 representation pipeline 稳定后，另行设计 mug、罐子等 household
mesh 的 shape/mass/contact generalization。第二阶段不属于本次实施计划，避免把 shape
难度与 Graph representation 混入第一阶段结论。

## 官方资产

使用 Google DeepMind MuJoCo Menagerie 的 `franka_emika_panda`，固定到 commit
`71f066ad0be9cd271f7ed58c030243ef157af9f4`。只保留该模型所需 MJCF、mesh、README 与
LICENSE，不拉取其他机器人。

项目场景可以使用一个明确记录的 `panda_integration.xml`，只允许为组合场景移除长度不再
匹配的 upstream viewer keyframe。不得改动官方 mesh、joint range、inertia、actuator、
finger collision geometry 或 material。

## 物理环境

新增 `FrankaContactEnv`，它直接拥有一个 `MjModel/MjData`，并成为物理实验 snapshot、contact、
success 和 termination reason 的唯一权威来源。旧 `KinematicTabletopEnv` 不参与物理 episode。

### 时间与控制频率

- MuJoCo timestep：`0.002 s`（500 Hz）。
- Policy/data frequency：20 Hz。
- 每个 policy action 执行 25 个 physics substeps。
- Reset 后先执行有限的 settling steps，让物体稳定落在桌面上，再向 policy 暴露第一帧。

### 确定性与受控 physics randomization

Canonical Stage-1 split 使用固定的质量、惯量、摩擦、joint damping、actuator gain、object geometry
和 solver 参数。同一个 environment seed 必须产生完全相同的 layout 与 physics parameter sample。

Controlled-randomization split 只改变显式列入配置的参数，并把实际 sample 写入 episode metadata：

- object mass scale：`[0.8, 1.2]`。
- object/table/finger friction scale：`[0.8, 1.2]`。
- arm joint damping scale：`[0.9, 1.1]`。

每个参数由 environment seed 的独立 RNG stream 采样。Flat、Graph、expert 与 edge-shuffle 必须在
paired case 中复用同一个 physics sample。训练、ID evaluation 与 physics-OOD 的 randomization
开关和范围分别写入 config，不允许隐式使用全局 RNG。第一阶段主 representation 结论使用
canonical + crowded/count OOD；controlled-randomization 作为单独 robustness slice 报告。

### 动作接口

Graph、Flat、scripted expert 与 teleop 都使用相同的 7 维 Cartesian delta-pose action：

```text
(dx, dy, dz, drx, dry, drz, gripper_open)
```

- policy、expert 与 teleop 的 7 个输入维度都使用 `[-1, 1]` command space；环境将前三维
  映射为每个 20 Hz step 最大 `0.02 m` 的 Cartesian target delta。
- `drx/dry/drz` 是 body-frame axis-angle rotation vector；controller 通过 SO(3) exponential map
  更新 orientation target，而不是把它解释成绝对 Euler angles；三个 command 分量共同按范数
  限幅后映射为每步最大 `0.05236 rad`（3°）的 rotation-vector delta。
- translation 与 rotation 分别使用独立的 per-step 限幅；rotation 默认最大 `0.05236 rad`（3°）。
- `gripper_open >= 0.5` 映射到 finger open target；`< 0.5` 映射到 close target，与现有
  open/close 语义保持一致。训练标签只写精确的 `0` 或 `1`。
- viewer、expert、teleop 和 dataset 共用这一语义，禁止在任何边界再次反转。

旧 4D checkpoint 与新 7D action head 不兼容，必须保留但不能载入 physics experiment。

### Arm controller

每个 policy step 先在 SE(3) 上更新 Cartesian target，再由包含 position 与 orientation error 的
warm-started 6D damped-least-squares IK 产生 7 个 joint position targets。官方 Panda position
actuators 或等价的限幅 impedance controller 在 25 个 substeps 中追踪这些 targets。

Controller 必须：

- 裁剪 joint target 到官方 joint range。
- 限制单个 policy step 的 joint-target 变化与 actuator force。
- 使用上一 physics state warm start，禁止每步 reset robot qpos。
- 在 IK 未达到 tolerance 时保留有限合法 target，记录 `ik_limited`，不产生 NaN。
- 对 Graph、Flat、expert 和 teleop 使用相同实现及参数。

### Contact objects

第一阶段物体使用同一 collision primitive 与 material，只以颜色、位置、target flag 和名称区分。
每个物体拥有 freejoint、非零 mass/inertia、gravity、接触和摩擦。桌面与 receptacle 具有碰撞
geometry。所有 episode reset 之后：

- 不得直接写活动物体 qpos/qvel。
- 不得创建 object-to-hand equality/weld。
- 不得根据“held object”脚本设置物体 pose。
- 物体只能通过 MuJoCo integration、contact force 和 gravity 移动。

## Snapshot 与 interaction graph

每个 20 Hz observation 从当前 MuJoCo state 构造：

- gripper TCP pose、linear/angular velocity、finger opening。
- 每个物体的真实 body pose、freejoint velocity、size、movable/target flag。
- receptacle 与 support pose/size。
- 由 `data.contact` 解析的 fingertip-object、object-table、object-receptacle contacts。
- 每个有效 interaction pair 的相对 position、relative orientation rotation vector、relative
  linear/angular velocity、normal force 与 tangential force magnitude。
- support relation 由接触与高度共同判断。

Graph 与 Flat 必须从同一个 `SceneSnapshot` 构建。Flat 继续包含与 Graph 相同的节点、边和 mask
信息，只移除显式 message passing；不得让 Graph 获得额外 contact、target 或 pose 信号。

Interaction edge payload 固定为 18 维：relative position 3、relative orientation rotvec 3、
relative linear velocity 3、relative angular velocity 3、distance 1、contact state 1、normal
force 1、tangential force 1、stable-grasp relation 1、support relation 1。所有 force 都来自
`mj_contactForce`，无 contact 的 edge force 为零。Flat payload 展平相同的 18 维 edge values。

`held_object` 不再由距离阈值决定。一个物体被认为 stable grasped，必须满足：

1. 左右 fingertip collision groups 同时接触同一个物体。
2. 物体底部离开桌面至少 `0.01 m`。
3. 前两项连续保持至少 10 个 physics frames（20 ms）。

Wrong-object、drop 与 release 都从该物理状态机推导，而不是影响模拟动力学。

## Task termination

- `SUCCESS`：目标物体与 receptacle 接触并稳定停留；双指已打开；TCP 已抬离物体。
- `WRONG_OBJECT`：非目标物体达到 stable-grasp 条件。
- `DROPPED`：曾 stable-grasp 的目标失去双指接触，并在 receptacle 外重新接触桌面或跌落。
- `TIMEOUT`：达到 episode 最大 policy steps。
- 物体或机器人状态出现非有限值、穿透或越界时以明确 physics-failure 诊断结束，不伪装成 policy timeout。

## Scripted physics expert

Expert 保留 approach、align、close、lift、transport、release、retreat 阶段，但转移条件改为真实状态：

- close 阶段等待左右指尖与目标物体形成 bilateral contact，而不是等待脚本 `held_object`。
- bilateral contact 后立即进入 lift；物体离桌且双指接触连续满足 10 个 physics frames 后才确认
  stable grasp。若 lift 期间接触丢失或物体未离桌，重新打开、抬升、对齐并重试。
- transport 监测 contact loss；掉落后不得继续假装持有。
- release 在 receptacle 上方打开手指，并等待物体由重力落下、稳定接触 receptacle。

Expert 可以读取完整 simulator state，以提供示范；Graph/Flat 仍只读取声明的 observation。训练前
必须先通过 expert gate，否则停止表示比较并先修 controller/environment。

## Teleoperation

Dashboard 支持：

- `W/A/S/D`：Cartesian XY target。
- `R/F`：Cartesian Z target。
- `Q/E`：绕 TCP Z 轴的正/负 rotation-vector delta。
- 方向键上下：绕 TCP X 轴的正/负 rotation-vector delta。
- 方向键左右：绕 TCP Y 轴的正/负 rotation-vector delta。
- `Space`：切换 finger open/close。
- `Z`：丢弃当前 episode 并 reset。
- `Escape` 或关闭窗口：安全结束。

Teleop translation、rotation 和 gripper command 使用与 policy 完全相同的 7D action scale、
SO(3) composition 与限幅。Teleop episode 标记 `trajectory_source=teleop`；默认作为 recovery
数据，只有在 Flat 与 Graph 使用完全相同的 episode manifest 时才能进入正式训练。

## 四视角 viewer

单个 GLFW/MuJoCo window 使用同一个 `MjModel/MjData` 渲染 2×2 dashboard：

```text
┌──────────────────────┬──────────────────────┐
│ Agent View           │ Wrist / Egocentric   │
├──────────────────────┼──────────────────────┤
│ Side View            │ Top View             │
└──────────────────────┴──────────────────────┘
```

- Agent：看到完整 Panda、桌面、所有物体和 receptacle。
- Wrist：固定在 hand/TCP 上，朝夹爪与接触区域观察。
- Side：检查接触、lift height、drop 和 release。
- Top：检查 crowded layout、target selection 和放置位置。

Dashboard overlay 显示 controller、policy step、episode reason、IK status、finger contact、stable
grasp 与 target identity。`--view native` 保留自由相机。GIF 使用与 dashboard 相同的四路 frame。

语义配色保持目标绿色、最近干扰物橙色、其他活动物体蓝色；颜色只改变 visual geom，不改变
collision、friction、mass 或 target feature。

## 数据采集

正式 state/Graph 数据继续保存为可审计的 episode arrays，并新增：

- `trajectory_source`：`scripted` 或 `teleop`。
- contact/stable-grasp diagnostics。
- physics/controller failure flags。
- scene/controller version 与物理参数 hash。

四个 camera 在同一个 physics state、同一个 policy timestamp 同步采样，并以 20 FPS 保存 RGB-D：

- `observation.agent.rgb` 与 `observation.agent.depth`。
- `observation.wrist.rgb` 与 `observation.wrist.depth`。
- `observation.side.rgb` 与 `observation.side.depth`。
- `observation.top.rgb` 与 `observation.top.depth`。

RGB shape 为 256×256×3、`uint8`；metric depth shape 为 256×256、`float32`。未来 dual-camera
policy observation 明确定义为 Agent RGB-D + Wrist RGB-D；Side/Top 仍被记录用于诊断与视频，
但不进入 dual-camera policy。所有 RGB-D 本阶段都不进入 Graph/Flat state policy。无论是否
启用图像写盘，state trajectory 都必须相同；渲染不能改变 physics stepping。

## Recovery augmentation

Recovery 必须是物理合法的 episode initialization，而不是 rollout 中瞬移物体。允许在 reset
阶段确定性构造：

- gripper alignment offset。
- premature close。
- off-center grasp approach。
- transport target offset。

Reset 完成并开始 rollout 后，所有恢复都只能通过 action、controller、contact 和 gravity 发生。
Recovery seed、variant、source episode 和 initial state 写入 manifest。

## 实验公平性

真实物理实验写入 `outputs/interaction_graph_physics/`。Flat 与 Graph 必须共享：

- 成功 scripted demonstrations 和可选 teleop/recovery manifest。
- episode-level train/validation/test split。
- action normalization、batch order、optimizer、action head 与 training epochs。
- representation parameter budget 与 model seeds。
- 相同的 7D action head、translation/rotation normalization 与 per-dimension loss weighting。
- controller、physics parameters、initial state seeds 与 paired evaluation cases。

训练 2、3 物体；评估 normal ID、4/5 count-OOD 与 4/5 crowded-OOD。第一轮使用 3 个 model
seeds。Graph checkpoint 还要在相同 cases 上进行 valid-edge shuffle ablation。

核心指标：

- 完整 pick-and-place success。
- bilateral fingertip contact rate。
- stable-lift rate。
- wrong-object stable-grasp rate。
- contact-loss/drop rate。
- successful placement rate。
- episode length 与 controller/IK failure rate。
- edge shuffle 后的 success 与 grasp 变化。

旧实验的“平均 Graph−Flat 至少 +10 pp 且每 seed 改善”保留为方向性标准，但只有 shared
controller 通过 expert gate 后才解释 representation 差异。

## Expert gate 与验收

开始模型训练前必须满足：

1. Scripted expert 在声明的 normal 与 crowded validation seeds 上成功率至少 90%。
2. stable grasp 的目标同时接触左右指尖，并真实离开桌面。
3. 物体运动来自 contact/gravity；运行时检测不到 weld、attachment 或 object qpos rewrite。
4. release 后目标由重力落入 receptacle 并稳定。
5. 同 seed 初始化完全一致；rollout 在浮点容差内可复现。
6. 所有 joint、qpos、qvel、contact force 和 observations 有限。

若 expert gate 未通过，当前阶段的完成定义是诊断并修好物理/controller；不得继续训练并把 shared
control failure 当作 Graph/Flat 结果。

## 测试

- 资产/模型：官方模型来源、LICENSE、joint/finger/camera/geom/contact 参数和 500 Hz timestep。
- Controller：IK reachability、joint/force limits、20 Hz→25 substeps、无每步 qpos reset。
- Contact：双指接触、单指不算 stable grasp、lift/drop/release 状态机。
- Interaction logging：contact state、normal/tangential force、relative SE(3) pose 与 velocity
  和 MuJoCo state/contact force 一致；Graph/Flat 18D edge payload parity。
- No-attachment audit：无 object weld/equality/mocap；step 代码不写 active object qpos。
- Snapshot：pose/velocity/contact 与 MuJoCo data 一致；Graph/Flat payload parity。
- Expert：normal/crowded gate、失败重试与 recovery initialization。
- Viewer：四 camera RGB、wrist 跟随、2×2 layout、native mode、keyboard mapping。
- Dataset：四路 RGB-D 同步、metric depth、20 FPS、Agent/Wrist dual-camera observation contract、
  trajectory source、manifest parity 与 render/physics isolation。
- Experiment：paired cases、edge shuffle、旧 outputs 保留、新 namespace 独立。
- Mac acceptance：`mjpython` dashboard 与 native viewer；真实 animated four-view GIF。

## 错误处理

- Missing assets/license/camera/contact geom：启动前给出精确路径或名称。
- IK limited：保持有限 joint target、在 diagnostics/UI 标记；超过连续阈值则 physics failure。
- NaN、爆炸速度、严重穿透：立即停止 episode 并保存诊断 state。
- 单指接触或夹爪闭合但未 lift：判为 failed grasp，expert 重试；不设置 held object。
- CoreGraphics 不可用：纯逻辑测试继续；真实渲染 test 在沙箱内显式 skip，在 Mac 图形会话验收。
- `mjpython` uv 动态库：沿用 `interaction_vla.macos_mjpython` 修复入口。

## 成功标准

- 用户能在四画面 dashboard 中看到完整 Panda 通过 7D SE(3) control 和真实双指接触抓起、
  运输并释放物体。
- 移除 friction 或禁用任一指尖接触会使 grasp gate 失败，证明不是吸附。
- Scripted expert 达到 90% gate 后，能够重新采集并训练公平的 Graph/Flat physics baseline。
- Agent/Wrist dual-camera observation 与四路同步 RGB-D 均可记录，但当前表示结论不依赖 VLA。
- 旧 kinematic artifacts 完整保留，新 physics outputs 与报告可单独复现。

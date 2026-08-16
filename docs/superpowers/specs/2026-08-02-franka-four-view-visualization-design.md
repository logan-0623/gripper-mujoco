# Franka Panda IK 与四视角完整演示设计

> **已废弃：** 用户随后选择完全接触物理抓取。本规格中的确定性吸附和
> kinematic-authoritative mirror 不再实施。替代规格为
> `2026-08-02-franka-contact-physics-design.md`。

日期：2026-08-02

## 目标

在不改变当前 Graph/Flat representation experiment 的前提下，把抽象球形 gripper
替换为完整的 Franka Panda 外观、7 关节 IK 联动和双指夹爪，并提供单窗口四画面
实时 viewer 与四画面 GIF 导出。

本阶段不训练像素策略，也不引入 VLA。Graph 与 Flat 仍读取相同的 privileged simulator
state，输出相同的 `(dx, dy, dz, gripper_open)`。Franka、IK 和 camera 是统一的执行/观察
层，不能成为两个 representation 之间的新变量。

## 非目标

- 不把 RGB 或语言输入当前 Graph/Flat policy。
- 不使用接触动力学决定是否抓取；抓取仍由现有距离、夹爪状态和确定性吸附逻辑决定。
- 不修改现有 checkpoint、normalization statistics、训练数据或已生成的评估报告。
- 不复刻参考项目的 Robotis OMY 控制器或旧 notebook。
- 不在本阶段实现键盘 teleoperation 或 LeRobotDataset 图像写入；camera API 会为下一阶段保留稳定接口。

## 机器人资产

使用 Google DeepMind MuJoCo Menagerie 的 `franka_emika_panda` 模型，只引入该模型运行
需要的 MJCF、mesh 和模型目录内许可证。来源与版本记录写入资产目录 README，不拉取整个
Menagerie 仓库。Panda 的原始模型文件尽量保持不变；项目特有的桌面、物体、camera 和
必要的 wrist-camera attachment 放在独立 scene 文件或明确标记的集成文件中。

## 状态与控制边界

`KinematicTabletopEnv` 继续作为任务状态和成功判定的唯一权威来源：

1. Graph/Flat/expert 根据 `SceneSnapshot` 产生 Cartesian action。
2. kinematic backend 执行动作、确定抓取/释放、更新物体状态并判定终止原因。
3. Franka MuJoCo mirror 接收更新后的 snapshot。
4. 统一的坐标变换把 gripper、物体和 receptacle 从 policy workspace 映射到 Panda 可达空间。
5. damped-least-squares IK 从上一帧关节角 warm start，求解 7 个 Panda arm joint。
6. `gripper_open` 映射到两个 finger joint；吸附物体的位置仍严格来自 snapshot。

这个边界保证加入 Franka 前后，同一个 checkpoint、seed 和 layout 的 action、任务状态、
成功/失败原因与 episode 长度完全一致。Affine scene transform 必须对 gripper、物体和
receptacle 一致应用，以保留相对位置、最近邻关系和目标身份。

## IK 行为

- 使用 MuJoCo Jacobian 实现 damped-least-squares position IK，并加入温和的 neutral-pose
  regularization；第一阶段保持固定的向下抓取 orientation。
- 每步从上一帧解开始，限制迭代次数并裁剪到官方 joint range。
- 已声明的 normal、crowded 与 recovery workspace 必须在选定 scene transform 后可达。
- 若某一帧未达到容差，保留最后一个有限、合法的关节解，继续显示任务；dashboard 显示
  `IK limited` 状态，不允许 NaN、关节越界或 viewer 崩溃。
- 测试场景中的初始点、全部采样物体位置、receptacle 和现有 README rollout 必须达到
  预设 position tolerance。

## 场景与相机

场景包含完整 Panda、双指夹爪、桌面、receptacle、最多五个语义着色物体以及四个 camera：

- `agentview`：固定外部斜视角，能同时看到完整机械臂、桌面、物体和 receptacle。
- `wristview`：固定在 hand/TCP 上，随 IK 结果移动和旋转，朝夹爪下方观察。
- `sideview`：固定侧视角，用于检查高度、抬升和夹爪闭合。
- `topview`：固定俯视角，用于检查目标选择、拥挤度和 object identity。

沿用当前语义配色：目标绿色、最近干扰物橙色、其他活动物体蓝色、非活动物体隐藏或灰色。

## Viewer 与导出

### 四画面 dashboard

现有 `viewer` 子命令默认打开单个 2×2 GLFW/MuJoCo dashboard：

```text
┌──────────────────────┬──────────────────────┐
│ Agent View           │ Wrist / Egocentric   │
├──────────────────────┼──────────────────────┤
│ Side View            │ Top View             │
└──────────────────────┴──────────────────────┘
```

每个 panel 直接由同一个 `MjModel/MjData` 和固定 camera 渲染，避免四份模拟状态漂移。窗口顶部
或 panel overlay 显示 controller、step、termination status 和 IK status。Dashboard 在
macOS 上继续通过修复后的 `.venv/bin/mjpython` 启动。

通过 `--view native` 保留自由旋转的原生 MuJoCo viewer，便于检查 mesh、关节和相机位置。

### GIF

新增单 controller 的四画面 rollout GIF 导出，帧布局与 dashboard 一致。输出首先覆盖
scripted expert，再支持真实 Flat/Graph checkpoint。现有 Flat/Graph side-by-side GIF
继续保留；其单 panel camera 可选择 `agentview` 或 `wristview`，默认 `agentview`。

提前结束的 rollout 保持最终帧；GIF 标签必须包含 controller、step、termination reason
和 IK status。README 嵌入一个 crowded expert 四视角 GIF，并保留已有的 Flat/Graph 对比 GIF。

## 模块边界

- `interaction_vla/assets/franka_emika_panda/`：受许可约束的 Panda MJCF/mesh 与来源说明。
- `interaction_vla/assets/franka_tabletop.xml`：项目场景、物体、灯光与固定 camera。
- `interaction_vla/franka.py`：坐标映射、IK solver、snapshot-to-MuJoCo 同步与状态报告。
- `interaction_vla/mujoco_env.py`：保留任务 mirror 接口；委托 Franka scene/IK，不承载 dashboard UI。
- `interaction_vla/visualize.py`：session、CLI、dashboard 调度和 GIF frame composition。
- 独立测试文件覆盖资产、IK、同步、四画面与 CLI，避免把大量 Franka 测试塞进现有环境测试。

## 错误处理

- 资产缺失或许可证文件缺失：启动前给出精确路径与恢复说明。
- camera 名称缺失、framebuffer 尺寸不足：模型加载测试直接失败，不推迟到 GIF 导出时。
- IK 输入包含非有限值：立即拒绝并指出 snapshot/target。
- 单帧 IK 未收敛：有限降级为最后合法姿态并在 UI 标记，不改变任务 backend。
- Mac `mjpython` 动态库问题：沿用 `interaction_vla.macos_mjpython` 修复命令和可操作错误提示。
- CoreGraphics 不可用：offscreen 测试显式 skip；真实 Mac 图形会话执行 smoke 验证。

## 测试与验收

1. 资产测试：scene 在 MuJoCo 3.3.4 加载，7 个 arm joint、2 个 finger joint、4 个 camera、
   TCP/site 和许可证均存在。
2. IK 测试：初始点、目标采样范围、receptacle 与 README 固定 case 收敛；关节始终有限且在 range 内。
3. 同步测试：物体/目标身份、finger 开合、隐藏物体、吸附物体与 snapshot 一致。
4. 隔离测试：加入 Franka mirror 前后，同 action sequence 的 snapshot、termination reason 和
   episode length 完全一致；Graph/Flat checkpoint loader 与输出 shape 不变。
5. Camera 测试：四路 RGB 均为正确尺寸、非空且互不相同；wrist view 随 TCP 移动。
6. UI/CLI 测试：dashboard/native/GIF 参数、缺失 checkpoint、无效 camera 与输出路径均有明确行为。
7. 真实 Mac smoke：通过 `mjpython` 完成一个 expert crowded rollout；导出可由 Pillow 打开的
   animated four-view GIF。
8. 回归：完整 `tests/interaction_vla` 通过；编译检查通过；现有 checkpoint、报告和 GIF 保留。

## 成功标准

- 用户能在一个窗口同时看到完整 Panda 的 agent、wrist、side、top 四个实时视角。
- arm 与当前 gripper target 连续联动，双指状态与 policy 命令一致，抓取物体无视觉跳离。
- 原命令只需增加可选 `--view`，默认即可获得四画面完整演示。
- 可用一条命令导出 README 四视角 GIF。
- 当前 Graph/Flat 数值实验没有因为可视化层而改变。

## 后续方向

下一阶段可把 `agentview` 与 `wristview` 以 256×256、20 FPS 写入 LeRobotDataset，并增加视觉
encoder/VLA adapter。该阶段必须作为独立实验，以免把 perception improvement 与 Graph
representation improvement 混在同一结论中。

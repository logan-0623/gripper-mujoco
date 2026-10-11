# 换指令复核：指令敏感是 SmolVLA 的普遍现象，还是我们训练的产物？

- **运行：** 2026-10-11，代码 `3acf187`，见 [run.sh](run.sh)。脚本 [instruction_swap_eval.py](../../../scripts/instruction_swap_eval.py)（新增 `--suite`、`--lineage` 和 `paraphrase` 条件），每格 40 个初态，`--trace`。
- **原始数据：** 服务器 `/root/autodl-tmp/instruction_swap_double_check_20261011/`。
- **本目录：** 各组的 `summary.json`、`contract.json`（含所有指令原文）；[lifted.json](lifted.json) 统计每格被抬起超过 5 cm 的物体。

## 为什么要复核

F13 发现：去掉或替换指令后，成功率在所有 checkpoint 都降到 0–1/160。文献中 LIBERO-PRO、LIBERO-Plus 报告的是相反的现象（OpenVLA、π0 等模型对指令不敏感），所以要排除三种可能：

1. **流水线混杂。** SmolVLA 的位置编码按非 padding token 累加。空指令只有 1 个 token，乱码有 34–49 个，都在训练指令的 5–22 token 范围之外。换成另一任务的指令时长度正常，但它是另一个训练过的指令。
   → **加入改写（paraphrase）条件：** 同一物体、同一目标、换说法，长度控制在 12–21 个 token。
2. **只是我们这条谱系的问题。** → **B：** 官方 SmolVLA（`lerobot/smolvla_libero` @ `31d453f`），Spatial 0–3。
3. **只是 Spatial 的问题。** Spatial 的场景里有两个相同的碗，必须靠指令区分。→ **A：** 我们的 25k 在 LIBERO-Object 0–3 上评测（Object 也在训练数据 `lerobot/libero` 中）。

## 结果

**Spatial，各 160 个回合，括号内为按任务 0/1/2/3 拆分**

| 模型 | 正确指令 | 空指令 | 另一任务指令 | 改写指令 |
|---|---:|---:|---:|---:|
| 官方 SmolVLA | 137 | 1 | 6 | 40 (35/0/2/3) |
| 我们的 15k | 128 | 0 | 1 | 42 (34/0/0/8) |
| 我们的 25k | 136 | 0 | 0 | 46 (36/0/1/9) |

（我们模型的空指令和另一任务指令两列来自 [换指令归档](../libero_smolvla_instruction_swap_20261011/README.md)；正确指令与之共用。）

**改写指令下，抬起目标碗的回合数 / 成功数**

| 任务 | 改写 | 官方 | 15k | 25k |
|---|---|---|---|---|
| 0 | "grab the black bowl that sits between the plate and the ramekin and put it onto the plate" | 40 / 35 | 38 / 34 | 38 / 36 |
| 1 | "grab the black bowl beside the ramekin …" | 2 / 0 | 0 / 0 | 0 / 0 |
| 2 | "grab the black bowl in the middle of the table …" | 31 / 2 | 33 / 0 | 39 / 1 |
| 3 | "grab the black bowl sitting on top of the cookie box …" | 35 / 3 | 39 / 8 | 38 / 9 |

**官方模型在空指令下：** 成功 1/160，但 task 0/1 分别有 33/35 次抬起目标碗，稳定抓取共 74 次。我们的 25k 是 35 次，现象相同，官方模型更明显。

**LIBERO-Object，我们的 25k，各 160 个回合**

| | 正确 | 空 | 另一任务 | 改写 |
|---|---:|---:|---:|---:|
| 成功 | 143 | 0 | 40 | 75 |
| 接触 | 160 | 0 | 40 | 147 |

- **空指令：** 一次都没有接触物体（Spatial 中空指令仍能抓起碗）。
- **另一任务的指令：**
  - 只有场景 2 成功 40/40。那条指令要的 bbq sauce 不在场景 2 中，策略拿的是场景 2 自己的目标 salad dressing。
  - 场景 0 和场景 3 里**确实有**指令要的物体（cream cheese、alphabet soup），但 80 个回合中一次都没有拿起它。场景 0 拿起 salad dressing 16 次，场景 3 拿起 chocolate pudding 35 次，两个都不是指令要的。
  - 场景 1 中指令要的物体不存在，40 个回合什么都没拿。
- **改写**（加名词 can/box/bottle，pick up → grab，place … in → put … into）：
  - 成功数（task 0/1/2/3）：8/33/20/14；
  - 抬起正确物体的次数：17/34/32/17。

## 结论

1. **指令敏感不是我们训练的产物。** 官方 SmolVLA 在每个条件下都与我们的谱系一致：空指令 1/160，另一任务指令 6/160，改写 40/160。按任务拆分的模式几乎相同。
2. **不是位置编码分布外造成的。** 改写指令的长度在训练范围内，Spatial 的成功率仍从 128–137 降到 40–46，Object 从 143 降到 75。不过空指令在 Spatial 降到 0、在改写下只降到约 30%，所以超出长度范围可能确实额外贡献了一部分。
3. **策略依赖的是指令的具体措辞，而不是语义。**
   - Spatial task 1 把 "next to" 换成 "beside" 后，三个模型都完全不动作（抬起 0–2 次）。
   - task 0 保留了 "between the plate and the ramekin" 这一整段，几乎不受影响。
   - Object 中，指令要的物体就在场景里，策略也从未拿起它（0/80）。
4. **Spatial 中，改写后能抓起、却放不下。** task 2/3 中 31–39/40 个回合抬起了正确的碗，但只有 0–9 次成功。这与空指令下"能抓起、不能完成"（F13）一致，说明抓取之后的阶段对语言更敏感，进一步支持做 Q7。
5. **与文献的差别。** LIBERO-PRO、LIBERO-Plus 中"对指令不敏感"的模型是 OpenVLA、π0 系列，我们这次没有测。对 SmolVLA（官方配方，在 LIBERO 上微调）来说，"忽略指令、只看布局"并不成立；它更像是把整句指令当作检索轨迹的键，措辞一变就匹配不上。

## 不能推出

- 每个任务只有一条手写改写，各任务的差别也可能来自改动了哪些词，而不只是"改写本身"。要区分，需要每个任务做多条改写，并逐词替换。
- Object 只测了 25k；只有一个训练 seed。官方模型只有一个 checkpoint。
- Object 中场景 0 拿起 salad dressing、场景 3 拿起 chocolate pudding，是不是去了"记忆位置"还没有验证。`instruction_trace_analysis.py` 目前只支持 Spatial 的碗。
- 改写下"放不下"只记录了成功与否和抬起与否，还没有区分是掉落、放错位置，还是停住不动。

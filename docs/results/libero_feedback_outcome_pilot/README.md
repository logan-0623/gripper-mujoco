# LIBERO Spatial 动作结果反馈：第一批配对可行性试跑

状态：三个 checkpoint 全部完成（2026-10-05）。5k 的首次运行在容器重启时中断，后来用同一脚本和参数重跑，输出在 `step_005000_rerun_20261005/`。三个 checkpoint 的 `paired_summary.json` 和 `contract.json` 都已归档在本目录。

## 研究用途

核对同一训练谱系的 SmolVLA 是否能在一次真实抓取成功与短暂执行失败后，产生可测量的不同闭环行为。这是 [研究问题](../../research-questions.md) 的可行性步骤，不是表示机制、训练规律或方法增益验证。

## 固定合同

- 模型：seed=1000、25k/10k/5k，来源为 [lineage.json](../libero_smolvla_acquisition_timeline/lineage.json)；本次未训练新权重。
- 环境：服务器现有 LIBERO Spatial task 0、1；每任务初态 ID 0–9；相同原生指令和成功条件。
- 两分支：baseline 按策略执行；open 在已确认稳定抓取且目标比初始高度抬起至少 2 cm 后，将连续 8 个实际夹爪命令设为 `-1`。其他动作维度照策略执行。分叉前动作及 MuJoCo flattened state 按哈希比较；控制器内部状态尚未单独序列化核对。
- 评测合同：LeRobot 原生评测入口、`n_action_steps=10`、`num_steps=10`、`empty_cameras=1`、单环境、同步、固定策略噪声 seed；最多 280 环境步。
- 输出：每步物理事件、实际执行动作、episode 成功、分叉步、分叉前状态/动作哈希。期望 `3 checkpoints × 2 tasks × 10 states × 2 branches = 120` 条 continuation。触发不到的回合单独记为未触发，不作为有效分叉对。
- 服务器路径：`/root/autodl-tmp/smolvla-official-reproduction-v2/acquisition/feedback_outcome_pilot10/`。运行脚本 `run.sh`，日志 `run.log`，每个 checkpoint 的 `paired_summary.json` 为原始汇总。
- 运行代码：[feedback_branch_pilot.py](../../../scripts/feedback_branch_pilot.py)。服务器脚本与本地文件最初上传时 SHA-256 一致；服务器本身仍保留其他未提交改动，本次只新增该脚本及试验输出。

## 已完成 smoke

25k、task 0、初态 0：baseline 与 open 的触发步均为 75，分叉前 MuJoCo flattened state 哈希及动作前缀哈希一致。baseline 在 103 步成功；open 的 8 步夹爪张开造成记录到的掉落，在 280 步内记录到再次稳定抓取，最终没有完成任务。这仅证明配对、扰动和事件记录路径能工作；一次再次抓稳不等于成功恢复。

原始 smoke 文件：

```text
/root/autodl-tmp/smolvla-official-reproduction-v2/acquisition/feedback_outcome_smoke_25k_task0_state0_retry1/paired_summary.json
```

首次 smoke 因当前 LeRobot CLI 把 `--env.task_ids=[0]` 解析失败，在策略评测开始前终止，未产生有效轨迹。上传脚本随后仅把该参数改为当前环境接受的 `--env.task_ids 0`，并重新试跑。

## 三个 checkpoint 的结果（每个任务 10 对，描述性）

"有效"指配对达到触发条件，且分叉前的状态与动作哈希一致。后面四列只在有效对中统计。

| ckpt | 任务 | 有效对 | baseline 成功（全部 10 对） | 掉落 | 再次抓稳 | open 分支成功 | 同一批有效对的 baseline 成功 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5k | 0 | 0 | 0/10 | — | — | — | — |
| 5k | 1 | 3 | 2/10 | 3 | 0 | 0 | 2 |
| 10k | 0 | 9 | 8/10 | 8 | 2 | 1 | 8 |
| 10k | 1 | 5 | 5/10 | 5 | 1 | 0 | 5 |
| 25k | 0 | 8 | 8/10 | 7 | 3 | 0 | 8 |
| 25k | 1 | 10 | 8/10 | 10 | 10 | 9 | 8 |

**读法**
- 5k 几乎抓不起来，很少达到触发条件，所以无法检验它的恢复能力。
- Spatial 1 掉落后的恢复能力在 10k 还没有（0/5），到 25k 已经具备（9/10）。它出现在 10k→25k 之间，正好是离线单步误差差别不大、闭环成功率却继续上升的那一段（见 [Q2](../libero_smolvla_q0b_q6a_q1_20261005/README.md)）。
- Spatial 0 在 10k 和 25k 都很少能恢复（1/9、0/8）。还需要逐条看视频，确认失败是否是物理上无法恢复。

## 数据支持与解释边界

服务器扩展时间轴 `timeline_v2_states0_39_summary/report.json` 与仓库 README 的 160 回合/阶段统计一致；同目录本地 JSON/CSV 是旧 40 回合/阶段批次。两批分母分开使用。

旧时间轴事件显示自然的“接触但未稳定抓取”存在：10k task 0 为 12/40，25k task 0 为 8/40。自然掉落和记录到的再次抓取很少。训练数据 revision `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4` 中，Spatial task 0、1 各有 45 条演示；根据第 7 维动作的夹爪闭合次数，分别有 5/45、4/45 条出现多次闭合。此计数不能区分真实恢复、命令抖动或其他动作，训练集是否提供足够的恢复监督仍未核实。

本试跑采用执行器短暂张开，属于人为扰动。扰动后的状态可能超出演示分布；其失败不能单独归因为缺少反馈表征。运行完成后首先检查每对是否触发、分叉前已记录的状态和动作是否匹配、open 分支是否真的产生掉落，以及是否有基线成功率与可区分的行为轨迹。

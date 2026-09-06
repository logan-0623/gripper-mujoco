# Protocol-v5 SAE server result archive

本目录归档服务器已有的 SAE pilot 结果。本次没有独立复验，没有重新训练或运行策略推理；文件校验仅确认下载前后的字节一致，不构成实验复现。结果仍为 pilot，尚未验证闭环效用。

原始 JSON 和逐状态动作效应 NPZ 保持原样，合计 609,226 bytes（约 595 KiB）。三个 SAE 字典权重未上传；它们的服务器路径、大小和 SHA-256 记录在 [archive_manifest.json](archive_manifest.json) 中。完整重跑仍需要服务器上的权重、State Bank、latent cache 和官方策略 checkpoint。

| 文件 | 内容 |
| --- | --- |
| [report.json](report.json) | 原始 gate 决策：8 个候选，4 个通过动作效应门限 |
| [discovery.json](discovery.json) | 数据分组、训练设置、三个 seed 的重建指标与输入绑定 |
| [candidates.json](candidates.json) | 无标签冻结候选、跨 seed 匹配和选择阈值 |
| [feature_profiles.json](feature_profiles.json) | 特征覆盖、作用范围和时序统计 |
| [action_sensitivity_n_0512.json](action_sensitivity_n_0512.json) | 动作效应、episode-cluster CI、BH 校正与支持集 |
| [action_effects_n_0512.npz](action_effects_n_0512.npz) | 原始逐状态 target/random 效应与分组数据，非模型权重 |
| `models/seed_*.json` | 三个字典的训练元数据和模型校验和，非模型权重 |
| [archive_manifest.json](archive_manifest.json) | 归档来源、时间、完整文件清单和验证范围 |
| [SHA256SUMS](SHA256SUMS) | 本目录 9 个原始结果文件的 SHA-256 |

服务器来源目录为 `/root/gripper-mujoco/outputs/representation_study/libero_smolvla/protocol_v5/sparse_features/`。原始报告中的相对路径仍指向服务器项目根目录；在此归档中查看时，去掉 `outputs/representation_study/libero_smolvla/protocol_v5/sparse_features/` 前缀即可找到对应文件，字典权重除外。

归档时逐文件核对了服务器 SHA-256，并检查 JSON 可解析及 NPZ 容器完整性。仓库原有 `ccfa.yaml` 中 discovery、action-sensitivity 和 effects 的三个校验和均与本次取得的文件一致。未重算统计检验，也未完成其他 SAE seeds 的独立动作干预复验。

在本目录运行 `shasum -a 256 -c SHA256SUMS` 可复查归档文件完整性。报告中的 `authorize_separate_longitudinal_design` 是原实验 gate 的输出；项目后续实验授权状态仍以 `ccfa.yaml` 为准。

# 第二条谱系：seed 2000（2026-10-14）

- **原因：** seed 1000 谱系（5k–25k）随旧服务器实例释放而丢失，没有任何副本。这条谱系既是复现用的第二个 seed，也是此后所有实验的新基础。
- **配置：** 与 seed 1000 完全相同（[run_smolvla_official_reproduction.sh](../../../scripts/run_smolvla_official_reproduction.sh)，代码 `c14e320`），只改了两处：
  - `SEED=2000`；
  - 每 2500 步保存一次（seed 1000 是每 5000 步）。
- **初始权重：** 与 seed 1000 是同一份 smolvla_base，即 [seed 1000 的 lineage.json](../libero_smolvla_acquisition_timeline/lineage.json) 中的 `base_checkpoint`。现在位于服务器的 `/root/autodl-tmp/models/smolvla_base_lineage`，model.safetensors 的 sha256 为 `7cd549ac…01eb`。
- **运行：** [run.sh](run.sh)。共 25000 步，用时 5 小时 45 分，速度 1.21 步/秒，退出码 0。
- **loss：**
  - 第 25k 步：0.305（seed 1000 为 0.311）；
  - 去掉 padding 后：0.259（seed 1000 为 0.265）。
- **服务器路径：** `/root/autodl-tmp/smolvla-official-reproduction-v2/full_25k_seed2000/checkpoints/0XX500/pretrained_model`。
- **优化器状态：** 只保留 010000 和 025000 两个点的 `training_state`。

## 哈希（`pretrained_model/model.safetensors`，sha256）

| 步数 | sha256 |
|---|---|
| 2500 | `26f5eaa63415677436cfc16905232835632c588c3d46d77d440314fcb80577dc` |
| 5000 | `971996f95ce3f323cc9a97614dff5b68d2d79e61c1b0c07667501197811f564d` |
| 7500 | `47ebb46033da754ab22631ad9af953ce1e8f95f43c7b264ab5c7b1741d4dad31` |
| 10000 | `03975d62e8029ea4e664b72ad8d6482dac5abb08972967b02fee864fed238a46` |
| 12500 | `52f399c3505d997ed14d024fde05e541067b22999b36b14ce6c19e35feb7cecf` |
| 15000 | `b5f378ee43102be23e040d8f5e1d67241e6087629c931f662fa8e2e40dae1d65` |
| 17500 | `b6ae3a6f9f6a23aa98d119b7aaf72aa2b2f57915996e1c1ab527c5c346da4288` |
| 20000 | `f2b3e37652766a96b8d1e14796a2f7ac333fa0bc239dec9dbdc6835effa809a1` |
| 22500 | `dab4eba96df8e6e8be7007ade3a5f7a4a573d8dfa50d15625f42c8ec35eedea8` |
| 25000 | `c7c9b16b167ab4022ef7cbbdfe7c35ac50bfdafa5eba8917682442e377efd1b7` |

- **train_config.json：** 10 个点相同，sha256 为 `b7b9baba…9a7e`。
- **归一化器：** 10 个点相同，`policy_preprocessor*.safetensors` 的 sha256 为 `bd239834…1ee4`。混合实验的处理器一致性检查可以通过。

# Graph-Conditioned ACT Implementation Plan

> **Execution:** Implement this plan in the current session on `main`, one test-first
> task at a time. Keep user-generated `outputs/` and `MUJOCO_LOG.TXT` out of commits.

**Goal:** Add a reproducible four-condition experiment that tests whether a frozen,
vision-estimated interaction graph improves ACT closed-loop continuous control.

**Architecture:** A new `interaction_vla.graph_control` package owns the compact 75D
token schema, frozen Graph inference/cache, LeRobot dataset adapter, paired ACT
training, and closed-loop comparison. ACT consumes the token through its native
`observation.environment_state` input. Existing graph-fine-tuning and LeRobot bridge
modules remain the source of checkpoints, dataset metadata, environment construction,
and action execution.

**Stack:** Python 3.11, PyTorch, LeRobot ACT, Hugging Face datasets/LeRobotDataset,
NumPy, MuJoCo, PyYAML, pytest.

---

## Task 1: Token schema and experiment configuration

**Files:**

- Create: `interaction_vla/graph_control/__init__.py`
- Create: `interaction_vla/graph_control/schema.py`
- Create: `interaction_vla/graph_control/config.py`
- Create: `tests/interaction_vla/graph_control/__init__.py`
- Create: `tests/interaction_vla/graph_control/test_schema.py`
- Create: `tests/interaction_vla/graph_control/test_config.py`

- [ ] Write failing schema tests that assert the named slices total exactly 75,
  reject non-finite/wrong-shape tokens, and keep every condition on one feature
  contract.
- [ ] Write failing configuration tests for the exact condition set, seed-matched
  full-data Graph checkpoint paths, output isolation, positive cache batch size, and
  valid evaluation cells.
- [ ] Run:

  ```bash
  .venv/bin/python -m pytest tests/interaction_vla/graph_control/test_schema.py tests/interaction_vla/graph_control/test_config.py -q
  ```

  Confirm the new imports fail.
- [ ] Implement immutable schema/config dataclasses and strict YAML parsing. Expose
  condition names `flat`, `predicted_random`, `predicted_reflect`, and
  `oracle_current`.
- [ ] Re-run the focused tests and commit:

  ```bash
  git add interaction_vla/graph_control tests/interaction_vla/graph_control
  git commit -m "feat: define graph control token contract"
  ```

## Task 2: Causal token packing and frozen Graph inference

**Files:**

- Create: `interaction_vla/graph_control/features.py`
- Create: `tests/interaction_vla/graph_control/test_features.py`
- Modify: `interaction_vla/graph_finetune/data.py`
- Modify: `tests/interaction_vla/graph_finetune/test_data.py`

- [ ] Write failing tests for predicted token packing: sigmoid binary heads,
  softmax categorical heads, checkpoint normalization for semantic/residual values,
  correct distractor risk selection, and deterministic float32 output.
- [ ] Write a future-leakage regression test proving `oracle_current` accepts only
  current entity/visibility/relation/semantic fields and obtains every goal slice
  from predicted Graph output. A payload containing the posthoc teacher
  `relation_goal` must be rejected or structurally impossible to pass.
- [ ] Write a public RGB preprocessing test matching the resize behavior used during
  Graph fine-tuning.
- [ ] Run the feature/data tests and confirm they fail for the missing API.
- [ ] Expose a small public image preprocessing helper in `graph_finetune.data` and
  implement `FrozenGraphRuntime`, flat/predicted/oracle-current token packers, stored
  normalization reconstruction, and language tokenization from checkpoint vocabulary.
- [ ] Re-run focused tests and commit:

  ```bash
  git add interaction_vla/graph_control/features.py interaction_vla/graph_finetune/data.py tests/interaction_vla/graph_control/test_features.py tests/interaction_vla/graph_finetune/test_data.py
  git commit -m "feat: build causal graph control tokens"
  ```

## Task 3: Immutable token cache and split provenance

**Files:**

- Create: `interaction_vla/graph_control/cache.py`
- Create: `tests/interaction_vla/graph_control/test_cache.py`
- Reuse: `interaction_vla/lerobot_bridge/provenance.py`
- Reuse: `interaction_vla/lerobot_bridge/sidecar.py`

- [ ] Write failing tests using a tiny local dataset for global-row ordering, exact
  graph split alignment, immutable destination refusal, atomic `.npz` write, cache
  checksum stability, and bounded graph inference batches.
- [ ] Add mismatch tests for dataset fingerprint, split-manifest hash, checkpoint
  hash/seed/initialization/fraction, token schema, dimension, condition, and row list.
- [ ] Add an oracle-current cache test that reads current teacher fields but does not
  load `annotation.tc_tig.relation_goal`.
- [ ] Implement cache construction/validation and concise JSON manifest writing.
  Store only `row_indices`, finite `[N,75]` float32 tokens, and provenance metadata.
- [ ] Re-run focused tests and commit:

  ```bash
  git add interaction_vla/graph_control/cache.py tests/interaction_vla/graph_control/test_cache.py
  git commit -m "feat: cache graph control features"
  ```

## Task 4: LeRobot dataset adapter with native ACT environment state

**Files:**

- Create: `interaction_vla/graph_control/dataset.py`
- Create: `tests/interaction_vla/graph_control/test_dataset.py`
- Modify only if required: `interaction_vla/lerobot_bridge/act_smoke.py`
- Modify only if required: `tests/interaction_vla/lerobot_bridge/test_act_smoke.py`

- [ ] Write failing tests for a dataset wrapper that indexes the cache by the sample's
  global `index`, adds a 75D `observation.environment_state`, and leaves dual RGB,
  10D proprioception, language/task, and action values byte-equivalent.
- [ ] Assert the metadata proxy advertises one `FeatureType.ENV` feature and delegates
  original stats/episode metadata. Assert no teacher annotation key enters an ACT
  sample.
- [ ] Add an ACT construction/forward test showing all four conditions have identical
  parameter counts and accept the same environment-state feature layout.
- [ ] Implement the dataset and metadata adapters. Factor narrowly reusable ACT bundle
  helpers only when the existing public boundary cannot support the adapter.
- [ ] Re-run focused plus existing ACT smoke tests and commit:

  ```bash
  git add interaction_vla/graph_control/dataset.py tests/interaction_vla/graph_control/test_dataset.py interaction_vla/lerobot_bridge/act_smoke.py tests/interaction_vla/lerobot_bridge/test_act_smoke.py
  git commit -m "feat: feed graph tokens to ACT"
  ```

## Task 5: Paired four-condition ACT training and checkpoint reload

**Files:**

- Create: `interaction_vla/graph_control/training.py`
- Create: `tests/interaction_vla/graph_control/test_training.py`
- Reuse: `interaction_vla/lerobot_bridge/act_smoke.py`

- [ ] Write failing pairing tests: every condition for a seed has the same initial
  shared ACT state hash, train/validation rows, sample order, optimizer settings,
  fixed epoch count.
- [ ] Add a regression test that the graph fine-tune split manifest is used directly;
  the legacy `pilot_episode_split` result must cause a mismatch error.
- [ ] Add checkpoint tests binding condition, cache hash, graph checkpoint hash,
  dataset fingerprint, feature metadata, source fingerprint, and split hash. Reload
  must refuse any altered binding.
- [ ] Add a one-update save/reload test for all four conditions and a formal comparison
  aggregation test across three seeds.
- [ ] Implement paired loaders, seed reset, fixed-epoch training, validation,
  atomic checkpoint/report writes, and nonempty output refusal. Reuse existing ACT
  update and OOM fallback behavior instead of duplicating it.
- [ ] Re-run focused training tests plus existing ACT training tests and commit:

  ```bash
  git add interaction_vla/graph_control/training.py tests/interaction_vla/graph_control/test_training.py
  git commit -m "feat: train paired graph-conditioned ACT policies"
  ```

## Task 6: Causal closed-loop Graph providers and paired evaluation

**Files:**

- Create: `interaction_vla/graph_control/rollout.py`
- Create: `tests/interaction_vla/graph_control/test_rollout.py`
- Reuse: `interaction_vla/lerobot_bridge/rollout.py`
- Reuse: `interaction_vla/lerobot_bridge/teacher.py`

- [ ] Write failing online-provider tests for flat, predicted-random,
  predicted-Reflect, and oracle-current behavior. Verify the oracle extracts only the
  current MuJoCo snapshot/current camera frame and uses predicted goal fields.
- [ ] Write paired-case tests crossing normal/crowded layouts with 2/3 objects and
  proving every condition receives identical environment seeds.
- [ ] Write metric aggregation tests for success, bilateral wrong-object interaction,
  wrong stable grasp, drop, timeout, episode length, IK scale, clipping, and gripper
  switching. Preserve per-case rows and policy-seed-level paired deltas.
- [ ] Implement runtime checkpoint loading, observation augmentation, chunked ACT
  action execution, causal graph refresh, per-case JSONL, and aggregate report.
- [ ] Re-run focused rollout tests plus existing bridge rollout tests and commit:

  ```bash
  git add interaction_vla/graph_control/rollout.py tests/interaction_vla/graph_control/test_rollout.py
  git commit -m "feat: evaluate graph-conditioned ACT rollouts"
  ```

## Task 7: CLI, macOS configs, and concise README workflow

**Files:**

- Create: `interaction_vla/graph_control/cli.py`
- Create: `interaction_vla/graph_control/__main__.py`
- Create: `configs/graph_control_act_smoke_macos.yaml`
- Create: `configs/graph_control_act_pilot_macos.yaml`
- Create: `tests/interaction_vla/graph_control/test_cli.py`
- Modify: `README.md`

- [ ] Write failing CLI tests for `inspect`, `cache`, `smoke`, `compare`, and
  `evaluate`; all expected failures must return compact JSON and nonzero status.
- [ ] Add repository-config tests locking four conditions, smoke one-update behavior,
  formal 3-seed pairing, full-data seed-matched Graph paths, Graph split manifest,
  and normal/crowded x 2/3 evaluation cells.
- [ ] Implement CLI dispatch and both configs. `inspect` performs all dependency,
  split, checkpoint, and leakage checks without writing training artifacts.
- [ ] Update README with the shortest runnable sequence and explicitly label smoke as
  engineering-only and `oracle_current` as privileged perception.
- [ ] Re-run focused CLI/config tests and commit:

  ```bash
  git add interaction_vla/graph_control/cli.py interaction_vla/graph_control/__main__.py configs/graph_control_act_smoke_macos.yaml configs/graph_control_act_pilot_macos.yaml tests/interaction_vla/graph_control/test_cli.py README.md
  git commit -m "docs: add graph-conditioned ACT workflow"
  ```

## Task 8: End-to-end verification

- [ ] Run all new unit/integration tests:

  ```bash
  .venv-lerobot/bin/python -m pytest tests/interaction_vla/graph_control -q
  ```

- [ ] Run affected regression tests:

  ```bash
  .venv-lerobot/bin/python -m pytest tests/interaction_vla/graph_finetune tests/interaction_vla/lerobot_bridge -q
  ```

- [ ] Run the full repository test suite:

  ```bash
  .venv-lerobot/bin/python -m pytest -q
  ```

- [ ] Validate CLI dependencies against the user's completed pilot artifacts:

  ```bash
  .venv-lerobot/bin/python -m interaction_vla.graph_control inspect --config configs/graph_control_act_smoke_macos.yaml
  ```

- [ ] Generate all four seed-0 caches and execute the real one-update/reload smoke:

  ```bash
  .venv-lerobot/bin/python -m interaction_vla.graph_control cache --config configs/graph_control_act_smoke_macos.yaml
  .venv-lerobot/bin/python -m interaction_vla.graph_control smoke --config configs/graph_control_act_smoke_macos.yaml
  ```

- [ ] Run `git diff --check`, inspect `git status --short`, and ensure runtime outputs
  plus `MUJOCO_LOG.TXT` remain unstaged.
- [ ] Commit any verification-only fixes separately. Do not claim success until the
  real smoke reload report and full test output are both present.

## User-run formal experiment

After the engineering smoke passes, the user runs:

```bash
.venv-lerobot/bin/python -m interaction_vla.graph_control inspect \
  --config configs/graph_control_act_pilot_macos.yaml

.venv-lerobot/bin/python -m interaction_vla.graph_control cache \
  --config configs/graph_control_act_pilot_macos.yaml

.venv-lerobot/bin/python -m interaction_vla.graph_control compare \
  --config configs/graph_control_act_pilot_macos.yaml

.venv-lerobot/bin/python -m interaction_vla.graph_control evaluate \
  --config configs/graph_control_act_pilot_macos.yaml
```

The scientific conclusion is based on the three-seed paired report, never on the
one-update smoke.

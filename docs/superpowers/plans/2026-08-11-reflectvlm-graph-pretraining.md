# ReflectVLM Graph Pretraining Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Mac-runnable ReflectVLM adapter and multi-task visual graph pretraining/evaluation workflow without changing the existing ACT controller.

**Architecture:** A new `interaction_vla.graph_pretrain` package safely parses ReflectVLM metadata into a fixed semantic graph target, creates leakage-safe group splits, and lazily decodes RGB only for requested batches. A compact RGB/history encoder predicts graph fields, while a config-driven pipeline owns training, metrics, checkpoint provenance, and JSON CLI output.

**Tech Stack:** Python 3.12, NumPy, PyTorch, Pillow, PyYAML, Hugging Face `datasets`, pytest.

---

### Task 1: Semantic target and ReflectVLM adapter

**Files:**
- Create: `interaction_vla/graph_pretrain/__init__.py`
- Create: `interaction_vla/graph_pretrain/schema.py`
- Create: `interaction_vla/graph_pretrain/reflectvlm.py`
- Test: `tests/interaction_vla/graph_pretrain/test_reflectvlm.py`

- [ ] **Step 1: Write failing parser and split tests**

Create a representative row with four bricks, status strings, upright flags, dependency
tuples, action history, and a `pick up yellow` oracle action. Assert that
`parse_reflect_metadata` returns ordered state labels, target/in-hand slots, adjacency,
phase, and relation goal IDs. Add malformed-action and `set()` cases. Build at least six
groups and assert `grouped_split_indices` has no group overlap and is deterministic.

- [ ] **Step 2: Verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/interaction_vla/graph_pretrain/test_reflectvlm.py -q
```

Expected: collection fails because `interaction_vla.graph_pretrain` does not exist.

- [ ] **Step 3: Implement the fixed target and safe parser**

Define `ReflectGraphTarget` with validated NumPy arrays and these vocabularies:

```python
STATE_NAMES = ("unknown", "ready", "blocked", "bad", "done")
UPRIGHT_NAMES = ("unknown", "false", "true")
PHASE_NAMES = ("pick", "place", "reorient", "insert")
```

Map source actions to existing relation-goal IDs:

```python
ACTION_GOALS = {
    "pick up": ("pick", "establish", "co_motion"),
    "put down": ("place", "break", "co_motion"),
    "reorient": ("reorient", "increase", "alignment"),
    "insert": ("insert", "establish", "containment"),
}
```

Use `ast.literal_eval`, with explicit handling for `None`, JSON `null`, and `set()`.
Order `brick_2` through `brick_5`, exclude the board, map dependency IDs to a 4x4
directed matrix, and return the normalized history separately from graph labels.

Implement deterministic group allocation by sorting unique `(board_id, env_seed)` keys
with SHA-256 of the split seed and allocating non-empty train/validation/test group sets.

- [ ] **Step 4: Verify GREEN**

Run the Task 1 test file and expect all tests to pass.

### Task 2: Vocabulary and lazy Torch dataset

**Files:**
- Modify: `interaction_vla/graph_pretrain/reflectvlm.py`
- Modify: `tests/interaction_vla/graph_pretrain/test_reflectvlm.py`

- [ ] **Step 1: Write failing vocabulary and image tests**

Assert that vocabulary IDs are built only from training histories, unknown validation
tokens map to `<unk>`, sequences are padded/truncated deterministically, and
`ReflectTorchDataset` converts a PIL image to finite CHW float RGB while returning all
graph labels as tensors. Use a source object that counts image accesses and assert split
preparation does not decode images.

- [ ] **Step 2: Verify RED**

Run the Task 1 test file. Expected: missing `Vocabulary`, `prepare_corpus`, and
`ReflectTorchDataset` imports.

- [ ] **Step 3: Implement training-only vocabulary and lazy decoding**

Tokenize with lowercase alphanumeric/underscore tokens. Reserve ID 0 for `<pad>` and 1
for `<unk>`. Read metadata columns eagerly, parse targets once, build vocabulary from
train indices only, and keep the original source plus integer indices for lazy image
lookup. Resize with Pillow and normalize RGB to `[0, 1]`.

- [ ] **Step 4: Verify GREEN**

Run the Task 1 test file and expect all tests to pass.

### Task 3: Graph estimator and objective

**Files:**
- Create: `interaction_vla/graph_pretrain/model.py`
- Test: `tests/interaction_vla/graph_pretrain/test_model.py`

- [ ] **Step 1: Write failing shape and optimizer tests**

Construct a two-example batch and assert output shapes for target `(B,4)`, in-hand
`(B,5)`, states `(B,4,5)`, upright `(B,4,3)`, dependencies `(B,4,4)`, phase `(B,4)`,
operator `(B,5)`, predicate `(B,7)`, and embedding `(B,D)`. Assert a complete loss dict
is finite, backward produces non-zero gradients, and one optimizer update changes a
parameter.

- [ ] **Step 2: Verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/interaction_vla/graph_pretrain/test_model.py -q
```

Expected: collection fails because the model module is missing.

- [ ] **Step 3: Implement compact RGB/history fusion and prediction heads**

Use three strided convolutions plus adaptive pooling for RGB. Use `nn.Embedding` and
masked mean pooling for history. Fuse both embeddings with a two-layer MLP and attach
the eight prediction heads. Implement categorical cross-entropy, off-diagonal masked
dependency BCE, and a summed `total` loss.

- [ ] **Step 4: Verify GREEN**

Run the model tests and expect all tests to pass.

### Task 4: Configuration, metrics, training, and checkpoint reload

**Files:**
- Create: `interaction_vla/graph_pretrain/config.py`
- Create: `interaction_vla/graph_pretrain/pipeline.py`
- Create: `configs/reflectvlm_graph_pretrain_macos.yaml`
- Test: `tests/interaction_vla/graph_pretrain/test_pipeline.py`

- [ ] **Step 1: Write failing config and synthetic end-to-end tests**

Load a temporary YAML config, train one epoch on synthetic grouped rows, assert finite
losses and artifact creation, reload the checkpoint, evaluate held-out rows, and assert
all metrics are finite and bounded. Assert an incompatible schema version and a
non-finite optimizer result fail explicitly.

- [ ] **Step 2: Verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/interaction_vla/graph_pretrain/test_pipeline.py -q
```

Expected: collection fails because config and pipeline modules are missing.

- [ ] **Step 3: Implement config, metrics, and artifact pipeline**

Use validated frozen dataclasses for dataset, model, and training sections. Load the HF
source lazily inside `load_source_dataset`, keeping `datasets` optional until a real
command needs it. Train with AdamW, validate each epoch, retain the lowest-validation-
loss state, and atomically publish:

```text
checkpoint.pt
training_summary.json
split_manifest.json
evaluation.json
```

Store schema version, vocabulary, config, source repo/split, split seed, and state dict
inside the checkpoint. Evaluation reports mean loss, categorical accuracies,
dependency precision/recall/F1, and exact operator-plus-predicate goal accuracy.

- [ ] **Step 4: Verify GREEN**

Run all `tests/interaction_vla/graph_pretrain` tests and expect all tests to pass.

### Task 5: JSON CLI and offline error contract

**Files:**
- Create: `interaction_vla/graph_pretrain/cli.py`
- Create: `interaction_vla/graph_pretrain/__main__.py`
- Test: `tests/interaction_vla/graph_pretrain/test_cli.py`

- [ ] **Step 1: Write failing parser/dispatch tests**

Assert `inspect`, `train`, and `evaluate --checkpoint` parse correctly; missing required
arguments return usage status 2; runtime failures return status 1 and one JSON error;
successful dispatch prints JSON-serializable output.

- [ ] **Step 2: Verify RED**

Run the CLI test file. Expected: collection fails because the CLI module is missing.

- [ ] **Step 3: Implement the CLI**

Follow the existing `lerobot_bridge.cli` error boundary. Add `--partition` with
validation/test choices for evaluate. Do not import Hugging Face datasets while merely
building the parser or displaying help.

- [ ] **Step 4: Verify GREEN**

Run all graph-pretrain tests and expect all tests to pass.

### Task 6: User commands and final verification

**Files:**
- Modify: `README.md`
- Modify: `requirements-lerobot-macos.txt`
- Modify: `requirements-lerobot-macos.lock.txt`
- Test: `tests/interaction_vla/lerobot_bridge/test_config.py`

- [ ] **Step 1: Write a failing dependency contract test**

Assert the LeRobot requirement and lock both contain the exact supported Hugging Face
`datasets` version already installed in the Mac environment.

- [ ] **Step 2: Verify RED, then pin the dependency**

Run the dependency contract test and confirm it fails for the missing explicit pin.
Add the pin without changing Torch, TorchCodec, LeRobot, or MuJoCo versions, then rerun
the test and expect it to pass.

- [ ] **Step 3: Add concise README commands**

Add one section after the ACT experiment with `inspect`, `train`, and `evaluate`
commands, the output paths, and the explicit statement that this checkpoint predicts
semantic graphs rather than continuous robot actions.

- [ ] **Step 4: Run verification**

Run:

```bash
.venv-lerobot/bin/python -m interaction_vla.graph_pretrain --help
.venv/bin/python -m pytest tests/interaction_vla/graph_pretrain -q
.venv/bin/python -m pytest -q
git diff --check
```

Expected: help lists all commands; graph-pretrain tests pass; the full suite has zero
failures; diff check is clean.

- [ ] **Step 5: Commit only implementation files**

Stage the new package, tests, config, requirements, README, and plan. Do not stage
MuJoCo logs, checkpoints, datasets, or evaluation outputs generated by the user's
completed experiment.

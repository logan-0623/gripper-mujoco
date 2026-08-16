# Training Progress Bar Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one epoch-level `tqdm` bar with representation, seed, weighted training MSE, speed, and ETA to every CLI training run.

**Architecture:** Keep progress reporting inside `train_policy` behind a default-off `show_progress` library option. The existing CLI path enables it through `train_from_config`; the training loop updates the bar only after an epoch has completed, without touching sampling, optimization, metrics, or checkpoint state.

**Tech Stack:** Python 3.12, PyTorch, tqdm, pytest

---

### Task 1: Specify progress behavior with a failing test

**Files:**
- Modify: `tests/interaction_vla/test_train.py`
- Modify: `tests/interaction_vla/test_smoke_pipeline.py`

- [x] **Step 1: Add a focused progress-spy test**

Add a test that monkeypatches `interaction_vla.train.tqdm`, trains a tiny Graph policy for two epochs with `show_progress=True`, and asserts:

```python
assert progress.kwargs["desc"] == "graph seed=7"
assert progress.kwargs["total"] == 2
assert progress.kwargs["unit"] == "epoch"
assert len(progress.postfixes) == 2
assert all("mse" in postfix for postfix in progress.postfixes)
```

Extend the smoke pipeline test with `capsys` and assert that the `train_from_config`
calls emit `proprio seed=0`, `flat seed=0`, `graph seed=0`, and `mse=` on stderr.

- [x] **Step 2: Run the focused test and confirm RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider \
  tests/interaction_vla/test_train.py::test_train_policy_reports_epoch_progress -q
```

Expected: FAIL because `train_policy` does not accept `show_progress` yet.

### Task 2: Add tqdm and implement epoch progress

**Files:**
- Modify: `requirements-macos.txt`
- Modify: `interaction_vla/train.py`

- [x] **Step 1: Declare and install tqdm**

Add:

```text
tqdm>=4.66
```

Install the declared dependency into `.venv` without changing other dependencies:

```bash
uv pip install --python .venv/bin/python 'tqdm>=4.66'
```

- [x] **Step 2: Add the default-off library option**

Import `tqdm` from `tqdm.auto` and extend `train_policy` with:

```python
show_progress: bool = False,
```

When enabled, wrap `range(epochs)` with:

```python
tqdm(
    range(epochs),
    desc=f"{representation} seed={seed}",
    total=epochs,
    unit="epoch",
    dynamic_ncols=True,
)
```

After each completed epoch, update the bar using the already-computed weighted MSE:

```python
progress.set_postfix(mse=f"{weighted_loss_sum / weight_sum:.6f}")
```

- [x] **Step 3: Enable progress for the CLI path**

Pass this argument from `train_from_config`:

```python
show_progress=True,
```

- [x] **Step 4: Run the focused test and confirm GREEN**

Run the focused pytest command from Task 1. Expected: `1 passed`.

### Task 3: Verify compatibility

**Files:**
- Verify: `interaction_vla/train.py`
- Verify: `tests/interaction_vla/test_train.py`

- [x] **Step 1: Run all training tests**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider \
  tests/interaction_vla/test_train.py -q
```

Expected: all tests pass, including deterministic resume equivalence.

- [x] **Step 2: Run the complete project suite**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider \
  tests/interaction_vla -q
```

Expected: all tests pass with only the existing platform skips.

- [x] **Step 3: Compile project and tests**

```bash
PYTHONPYCACHEPREFIX=/tmp/interaction_graph_tqdm_pycache \
  .venv/bin/python -m compileall -q interaction_vla tests/interaction_vla
```

Expected: exit code 0 with no output.

- [x] **Step 4: Run the isolated end-to-end smoke pipeline**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider \
  tests/interaction_vla/test_smoke_pipeline.py -q
```

Expected: pass; the test captures the three `train_from_config` progress bars and checks their labels and MSE postfix without touching formal recovery-pilot outputs.

No Git commit step is included because `/Users/loganluo/lerobot-mujoco` is not a Git worktree.

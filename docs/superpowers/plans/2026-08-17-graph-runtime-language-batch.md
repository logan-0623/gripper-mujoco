# Graph Runtime Language Batch Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make predicted Graph v2 cache generation and rollout pass language tokens and masks to the estimator with shape `[1, max_language_tokens]`.

**Architecture:** Preserve the existing one-dimensional `Vocabulary.encode()` output and the estimator's strict batched input contract. Add the batch dimension only in `FrozenGraphRuntime`, the single-sample inference adapter shared by cache generation and rollout.

**Tech Stack:** Python 3.12, NumPy, PyTorch, pytest

---

### Task 1: Reproduce and fix runtime language batching

**Files:**
- Modify: `tests/interaction_vla/graph_control/test_features.py`
- Modify: `interaction_vla/graph_control/features.py:204-213`

- [ ] **Step 1: Write the failing regression assertion**

Extend `_RecordingModel` to record language tensor shapes:

```python
class _RecordingModel:
    def __init__(self) -> None:
        self.previous: list[np.ndarray] = []
        self.language_shapes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def __call__(self, agent, wrist, state, tokens, mask, previous):
        self.previous.append(previous.detach().cpu().numpy().copy())
        self.language_shapes.append((tuple(tokens.shape), tuple(mask.shape)))
        return _outputs()
```

Add this assertion to
`test_frozen_runtime_recurrence_uses_previous_prediction_and_reset`:

```python
assert runtime.model.language_shapes == [((1, 4), (1, 4))] * 3
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
.venv-lerobot/bin/python -m pytest \
  tests/interaction_vla/graph_control/test_features.py::test_frozen_runtime_recurrence_uses_previous_prediction_and_reset \
  -q
```

Expected: FAIL because the recorded shapes are `(4,)`, not `(1, 4)`.

- [ ] **Step 3: Add the batch dimension at the inference boundary**

Change the two language tensor arguments in
`FrozenGraphRuntime.predict_token()`:

```python
torch.from_numpy(tokens[None]).to(self.device),
torch.from_numpy(mask[None]).to(self.device),
```

- [ ] **Step 4: Run focused Graph control feature tests and verify GREEN**

Run:

```bash
.venv-lerobot/bin/python -m pytest \
  tests/interaction_vla/graph_control/test_features.py \
  -q
```

Expected: all tests pass.

- [ ] **Step 5: Verify a real saved estimator checkpoint**

Run one `FrozenGraphRuntime.predict_token()` call using
`outputs/graph_finetune/mujoco_graph_v2/reflectvlm_init/fraction_1/seed_0/checkpoint.pt`.
Expected: the call returns a finite token with shape `(89,)` and does not raise a
language-shape error.

- [ ] **Step 6: Run the complete test suite**

Run:

```bash
.venv-lerobot/bin/python -m pytest -q
```

Expected: all available tests pass.

- [ ] **Step 7: Commit the fix**

```bash
git add \
  interaction_vla/graph_control/features.py \
  tests/interaction_vla/graph_control/test_features.py \
  docs/superpowers/plans/2026-08-17-graph-runtime-language-batch.md
git commit -m "fix: batch graph runtime language inputs"
```

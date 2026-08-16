# Subset Physics Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add provenance-safe `--model-seeds` filtering, fail-fast selected-checkpoint preflight, and a global rollout tqdm to physical evaluation.

**Architecture:** Resolve the requested seed subset without changing the loaded YAML, preload and validate all selected Flat/Graph checkpoints before simulation, then execute the existing paired rollout order through one progress reporter. Preserve the current default behavior when no seed filter is supplied and record the actual evaluation scope in the report.

**Tech Stack:** Python 3.12, PyTorch, MuJoCo, tqdm, pytest

---

### Task 1: Specify seed filtering, CLI parsing, and checkpoint preflight

**Files:**
- Modify: `tests/interaction_vla/test_physics_evaluate.py`
- Modify: `interaction_vla/physics_evaluate.py`

- [x] **Step 1: Add failing seed-selection and CLI tests**

Import `build_parser` and `resolve_evaluation_model_seeds`, then add these tests:

```python
def test_model_seed_override_selects_seed_zero_without_changing_config() -> None:
    config = load_config("configs/physics_recovery_pilot_macos.yaml")
    assert resolve_evaluation_model_seeds(config, None) == (0, 1, 2)
    assert resolve_evaluation_model_seeds(config, (0,)) == (0,)
    with pytest.raises(ValueError, match="unique"):
        resolve_evaluation_model_seeds(config, (0, 0))
    with pytest.raises(ValueError, match="configured"):
        resolve_evaluation_model_seeds(config, (9,))


def test_physics_evaluate_parser_accepts_model_seed_subset() -> None:
    args = build_parser().parse_args(
        ["--config", "configs/physics_recovery_pilot_macos.yaml", "--model-seeds", "0"]
    )
    assert args.model_seeds == [0]
```

- [x] **Step 2: Add a failing selected-checkpoint preflight test**

Use a temporary output directory containing only empty seed-0 Flat and Graph checkpoint files. Construct a config with `dataclasses.replace(config, output_dir=str(tmp_path))`, monkeypatch `load_training_checkpoint` to return `(object(), object(), payload)` where `payload` contains the 7D physical contract, matching physical hashes, matching `training_provenance`, the representation parsed from the path, and `model_seed=0`. Call:

```python
loaded = preload_evaluation_checkpoints(
    config,
    model_seeds=(0,),
    device="cpu",
    physical_hashes=physical_hashes,
    expected_training_provenance=training_provenance,
)
assert set(loaded) == {(0, "flat"), (0, "graph")}
```

Then remove the selected Graph file, clear the fake loader's call list, call the helper again, and assert `FileNotFoundError` is raised and the loader call list remains empty. This proves missing paths are checked before any model load.

- [x] **Step 3: Run the focused tests and confirm RED**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider \
  tests/interaction_vla/test_physics_evaluate.py -q
```

Expected: collection error because the new helpers do not exist.

- [x] **Step 4: Implement seed selection and fail-fast preload**

Add:

```python
def resolve_evaluation_model_seeds(
    config: ExperimentConfig, requested: Iterable[int] | None
) -> tuple[int, ...]:
    selected = tuple(config.train.model_seeds if requested is None else requested)
    if not selected:
        raise ValueError("at least one evaluation model seed is required")
    if len(set(selected)) != len(selected):
        raise ValueError("evaluation model seeds must be unique")
    unknown = tuple(seed for seed in selected if seed not in config.train.model_seeds)
    if unknown:
        raise ValueError(f"evaluation model seeds are not configured: {unknown}")
    return selected
```

Add `preload_evaluation_checkpoints(...)` that constructs every selected Flat/Graph path, reports all missing selected paths before loading any model, then validates checkpoint contract, identity, physical hashes, and exact training provenance before returning a mapping keyed by `(model_seed, representation)`.

Add `build_parser()` with `--model-seeds`, using `nargs="+"` and `type=int`.

- [x] **Step 5: Run focused tests and confirm GREEN**

Run the Task 1 focused command. Expected: all tests in `test_physics_evaluate.py` pass.

### Task 2: Add global rollout progress and scope metadata

**Files:**
- Modify: `tests/interaction_vla/test_physics_evaluate.py`
- Modify: `interaction_vla/physics_evaluate.py`

- [x] **Step 1: Add a failing mocked evaluation integration test**

Monkeypatch the config, case generator, provenance builders, checkpoint preloader, rollout function, and module-level `tqdm`. Invoke:

```python
evaluate_from_config("ignored.yaml", model_seeds=(0,), show_progress=True)
```

Use two fake cases and assert the progress spy receives `total=6`, six `update(1)` calls, and postfixes naming `flat`, `graph`, and `graph_edge_shuffle`. Assert the written report contains:

```python
{
    "model_seeds": [0],
    "case_count": 2,
    "rollout_count": 6,
    "policy_variants": ["flat", "graph", "graph_edge_shuffle"],
}
```

- [x] **Step 2: Run the integration test and confirm RED**

Run the new test by node id. Expected: FAIL because `evaluate_from_config` lacks the new keyword arguments and scope output.

- [x] **Step 3: Implement rollout progress and scope output**

Extend:

```python
def evaluate_from_config(
    config_path: str | Path,
    *,
    model_seeds: Iterable[int] | None = None,
    show_progress: bool = False,
) -> Path:
```

Preload the selected checkpoints, create `tqdm(total=len(cases) * len(selected_seeds) * 3, desc="physics eval", unit="rollout", dynamic_ncols=True)` only when progress is enabled, update it after each result, and close it in `finally`. Add `evaluation_scope` to the JSON report. Update `main()` to pass parsed seeds and `show_progress=True`.

- [x] **Step 4: Run the integration and complete physics-evaluation tests**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider \
  tests/interaction_vla/test_physics_evaluate.py -q
```

Expected: all tests pass.

### Task 3: Verify compatibility and provide the user command

**Files:**
- Verify: `interaction_vla/physics_evaluate.py`
- Verify: `tests/interaction_vla/test_physics_evaluate.py`

- [x] **Step 1: Verify the actual seed-0 checkpoint preflight without rollouts**

Run a read-only Python command that loads the recovery-pilot config, resolves `(0,)`, computes current gate and dataset provenance, and calls `preload_evaluation_checkpoints`. Expected keys: `(0, "flat")` and `(0, "graph")`.

- [x] **Step 2: Run the complete suite**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider \
  tests/interaction_vla -q
```

Expected: all tests pass with only existing platform skips.

- [x] **Step 3: Compile project and tests**

```bash
PYTHONPYCACHEPREFIX=/tmp/interaction_graph_subset_eval_pycache \
  .venv/bin/python -m compileall -q interaction_vla tests/interaction_vla
```

Expected: exit code 0 with no output.

No Git commit step is included because `/Users/loganluo/lerobot-mujoco` is not a Git worktree.

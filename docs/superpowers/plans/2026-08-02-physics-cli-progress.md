# Physics CLI Progress Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make expert validation and physical base/recovery collection display phase-level tqdm bars so all four seed-0 workflow commands show progress.

**Architecture:** Add opt-in `show_progress` parameters to the two library entry points and enable them only from CLI `main()`. Expert validation owns one case bar; physical collection owns sequential base-acceptance and recovery-attempt bars, with `finally`-based closure and spy-tested postfix semantics.

**Tech Stack:** Python 3.12, tqdm, pytest, MuJoCo.

---

### Task 1: Expert Gate Progress

**Files:**
- Modify: `tests/interaction_vla/test_validate_physics_expert.py`
- Modify: `interaction_vla/validate_physics_expert.py`

- [x] **Step 1: Add a failing tqdm-spy test**

Monkeypatch `make_validation_cases`, `run_validation_case`, and module-level
`tqdm`; call `validate_expert_from_config(config_path, show_progress=True)`; assert the
spy receives `total=2`, `desc="expert gate"`, two updates, cumulative
`passed/failed` postfixes, and `close()`.

- [x] **Step 2: Run the focused test and verify RED**

Run: `.venv/bin/python -m pytest tests/interaction_vla/test_validate_physics_expert.py -q`

Expected: failure because `show_progress` and module-level `tqdm` do not exist.

- [x] **Step 3: Implement the gate bar**

```python
from tqdm.auto import tqdm

def validate_expert_from_config(
    config_path, *, output=None, show_progress: bool = False
) -> Path:
    progress = tqdm(total=len(cases), desc="expert gate", unit="case",
                    dynamic_ncols=True) if show_progress else None
    results = []
    passed = 0
    try:
        for case in cases:
            result = run_validation_case(config, case)
            results.append(result)
            passed += int(result.success)
            if progress is not None:
                progress.set_postfix(condition=case.condition,
                                     objects=case.object_count, seed=case.seed,
                                     success=int(result.success), passed=passed,
                                     failed=len(results) - passed)
                progress.update(1)
    finally:
        if progress is not None:
            progress.close()
```

Pass `show_progress=True` from `main()` and keep the default false.

- [x] **Step 4: Run validation tests and verify GREEN**

Run: `.venv/bin/python -m pytest tests/interaction_vla/test_validate_physics_expert.py -q`

Expected: all tests pass.

### Task 2: Base and Recovery Collection Progress

**Files:**
- Modify: `tests/interaction_vla/test_physics_data.py`
- Modify: `interaction_vla/physics_data.py`

- [x] **Step 1: Add failing progress helper tests**

Use a lightweight progress spy and monkeypatched episode collection to exercise
one rejected then one accepted base attempt, followed by accepted/rejected
recovery attempts. Assert the base bar advances only for acceptance, recovery
advances for every attempt, both postfix dictionaries expose counters/reasons,
and both bars close.

- [x] **Step 2: Run collection tests and verify RED**

Run: `.venv/bin/python -m pytest tests/interaction_vla/test_physics_data.py -q`

Expected: failure because `collect_from_config` rejects `show_progress` and no
progress instances are created.

- [x] **Step 3: Implement sequential bars with guaranteed closure**

Add `from tqdm.auto import tqdm`, extend `collect_from_config` with
`show_progress: bool = False`, and create `base_progress` immediately before
the current base attempt loop with `total=config.train.episodes`,
`desc="base data"`, `unit="episode"`, and `dynamic_ncols=True`. Enclose that
loop in `try/finally`, set its postfix after every attempt, call `update(1)`
only after an accepted episode is saved, and close it in the `finally` block.

Build a tuple of accepted source records whose seeds belong to the training
split. Create `recovery_progress` with an exact total of source count times
`variants_per_episode`, `desc="recovery"`, `unit="attempt"`, and
`dynamic_ncols=True`. Enclose all recovery attempts in `try/finally`; inside
each attempt, use another `try/finally` so accepted episodes, ordinary rejected
episodes, and `PhysicsRecoveryRejected` exceptions each set a reason postfix
and advance exactly once. Close the recovery bar in the outer `finally`.

Pass `show_progress=True` from physical data `main()`; leave manifest contents,
retry limits, seed splitting, and acceptance rules unchanged.

- [x] **Step 4: Run collection tests and verify GREEN**

Run: `.venv/bin/python -m pytest tests/interaction_vla/test_physics_data.py -q`

Expected: all tests pass.

### Task 3: Documentation and Full Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/plans/2026-08-02-physics-cli-progress.md`

- [x] **Step 1: Document automatic progress for all four commands**

Add a short note beside the seed-0 workflow: gate shows cases, collection shows
base/recovery phases, training shows epochs, and evaluation shows rollouts.

- [x] **Step 2: Run focused progress tests**

Run: `.venv/bin/python -m pytest tests/interaction_vla/test_validate_physics_expert.py tests/interaction_vla/test_physics_data.py tests/interaction_vla/test_train.py tests/interaction_vla/test_physics_evaluate.py -q`

Expected: all selected tests pass.

- [x] **Step 3: Run the full suite and compilation**

Run: `.venv/bin/python -m pytest -q`

Expected: zero failures.

Run: `.venv/bin/python -m compileall -q interaction_vla tests/interaction_vla`

Expected: exit code 0 and no output.

- [x] **Step 4: Regenerate a formal pilot gate outside user outputs**

Run the pilot validation command with `--output /tmp/interaction_vla_pilot_gate_progress_probe.json`; require `passed=true`, normal/crowded rates at least 0.9, and visually confirm the case-level tqdm output reaches 100 percent.

- [x] **Step 5: Mark the plan complete and hand off unchanged commands**

Check every box only after fresh command evidence. No Git integration step is
available because the project is not a Git repository.

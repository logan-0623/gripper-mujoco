# LIBERO Adjacent-Stage Paired Deltas Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add paired bootstrap confidence intervals for consecutive SmolVLA training stages to the existing Probe v2 report without invalidating probe-cell caches.

**Architecture:** A new report-enrichment module validates the existing Probe v2 report, loads full paired payloads from `.cells`, reuses the existing Pretrained-to-SFT-25 rows, computes only missing adjacent comparisons, and atomically updates `report.json`. The CLI runs enrichment after probe fitting and before report inspection; `probe_runner.py` remains unchanged so existing cell bindings stay valid.

**Tech Stack:** Python 3.12, NumPy, pytest, existing LIBERO Probe v2 JSON artifacts and atomic JSON writer.

---

### Task 1: Define adjacent comparison semantics with failing tests

**Files:**
- Create: `tests/interaction_vla/representation_study/libero/test_probe_transitions.py`

- [ ] **Step 1: Write a failing test for the fixed ordered pairs**

```python
from interaction_vla.representation_study.libero.probe_transitions import (
    ADJACENT_STAGE_PAIRS,
)


def test_adjacent_stage_pairs_are_preregistered_in_training_order() -> None:
    assert ADJACENT_STAGE_PAIRS == (
        ("pretrained", "sft_25"),
        ("sft_25", "sft_50"),
        ("sft_50", "sft_100"),
    )
```

- [ ] **Step 2: Run the test and verify RED**

Run: `.venv-lerobot/bin/python -m pytest tests/interaction_vla/representation_study/libero/test_probe_transitions.py -q`

Expected: FAIL because `probe_transitions` does not exist.

- [ ] **Step 3: Add failing pure-grid tests**

Create synthetic full rows with matched state IDs, clusters, targets, and probe seeds. Assert that the grid:

```python
assert {(row["reference_stage"], row["destination_stage"]) for row in rows} == {
    ("pretrained", "sft_25"),
    ("sft_25", "sft_50"),
    ("sft_50", "sft_100"),
}
assert sft25_to_sft50["improvement"] > 0
assert sft50_to_sft100["status"] == "not_available"
```

Also assert that Geometry reverses the lower-is-better interval correctly and that the existing Pretrained-to-SFT-25 row is reused byte-for-byte except for list copying.

- [ ] **Step 4: Run the tests and verify they fail for missing behavior**

Run: `.venv-lerobot/bin/python -m pytest tests/interaction_vla/representation_study/libero/test_probe_transitions.py -q`

Expected: FAIL on missing adjacent-grid API.

### Task 2: Implement report-only adjacent inference

**Files:**
- Create: `interaction_vla/representation_study/libero/probe_transitions.py`
- Test: `tests/interaction_vla/representation_study/libero/test_probe_transitions.py`

- [ ] **Step 1: Implement the fixed pair contract and deterministic grid**

Implement the following public contract:

```python
ADJACENT_STAGE_PAIRS = tuple(zip(STUDY_STAGES[:-1], STUDY_STAGES[1:], strict=True))


def build_adjacent_stage_delta_grid(
    rows: Sequence[Mapping[str, object]],
    *,
    existing_reference_deltas: Sequence[Mapping[str, object]],
    split_name: str,
    config: LiberoStudyConfig,
) -> list[dict[str, object]]:
    """Return every adjacent Stage × Tap × Factor comparison in fixed order."""
```

For `pretrained -> sft_25`, copy the matching current reference-delta row. For later pairs, call `_paired_stage_delta` with a deterministic `_matched_probe_seed` offset of `30_000 + pair_index`. Always emit the complete Tap × Factor grid, including explicit unavailable rows.

- [ ] **Step 2: Run focused tests and verify GREEN**

Run: `.venv-lerobot/bin/python -m pytest tests/interaction_vla/representation_study/libero/test_probe_transitions.py -q`

Expected: PASS.

- [ ] **Step 3: Add failing enrichment tests**

Test that `enrich_adjacent_stage_deltas(config)`:

```python
before = json.loads(report_path.read_text())
after = enrich_adjacent_stage_deltas(config)
assert all(after[key] == before[key] for key in before)
assert after["adjacent_stage_pairs"] == [
    {"reference_stage": "pretrained", "destination_stage": "sft_25"},
    {"reference_stage": "sft_25", "destination_stage": "sft_50"},
    {"reference_stage": "sft_50", "destination_stage": "sft_100"},
]
assert after == enrich_adjacent_stage_deltas(config)
```

Use temporary Probe v2 reports and cell artifacts; do not touch real outputs.

- [ ] **Step 4: Implement validated atomic enrichment**

The enrichment must first call `inspect_probe_report(config)`, load deterministic `.cells/<stage>/<tap>/<split>/<factor>.json` paths, validate every loaded row is a mapping, add the six adjacent-analysis fields, and write with `write_json_atomic`.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run: `.venv-lerobot/bin/python -m pytest tests/interaction_vla/representation_study/libero/test_probe_transitions.py -q`

Expected: PASS.

### Task 3: Integrate enrichment into existing CLI commands

**Files:**
- Modify: `interaction_vla/representation_study/libero/cli.py`
- Modify: `tests/interaction_vla/representation_study/libero/test_cli.py`

- [ ] **Step 1: Write failing dispatch tests**

Patch `run_probe_study`, `inspect_probe_report`, and `enrich_adjacent_stage_deltas` and assert:

```python
assert dispatch(run_args) == enriched_report
assert calls == ["run", "enrich"]
assert dispatch(report_args) == enriched_report
assert calls == ["inspect", "enrich"]
```

- [ ] **Step 2: Run dispatch tests and verify RED**

Run: `.venv-lerobot/bin/python -m pytest tests/interaction_vla/representation_study/libero/test_cli.py -q`

Expected: FAIL because dispatch does not invoke enrichment.

- [ ] **Step 3: Implement minimal CLI sequencing**

For `probes run`, call `run_probe_study(config)` and then enrichment. For `probes report`, validate with `inspect_probe_report(config)` and then enrichment. Do not modify `probe_runner.py`.

- [ ] **Step 4: Run CLI and transition tests and verify GREEN**

Run: `.venv-lerobot/bin/python -m pytest tests/interaction_vla/representation_study/libero/test_cli.py tests/interaction_vla/representation_study/libero/test_probe_transitions.py -q`

Expected: PASS.

### Task 4: Document execution and verify repository integrity

**Files:**
- Modify: `SERVER_RUNBOOK.md`
- Modify: `README.md`

- [ ] **Step 1: Document evidence and concurrency semantics**

Document that `probes report` enriches the report without refitting probes, that Pretrained/SFT-25/SFT-50 adjacent inference can run CPU-side while SFT-100 uses the GPU, and that latent extraction must wait until SFT training finishes.

- [ ] **Step 2: Run focused and full verification**

Run:

```bash
.venv-lerobot/bin/python -m pytest tests/interaction_vla/representation_study/libero -q
.venv-lerobot/bin/python -m pytest -q
.venv-lerobot/bin/python -m compileall -q interaction_vla
git diff --check
```

Expected: all tests pass, compile succeeds, and diff check is clean.

- [ ] **Step 3: Verify the real server migration path remains cache-only**

Run `probes report` on the server after pulling. Confirm no probe progress bars appear, existing `.cells` mtimes remain unchanged, and the report contains complete SFT-25-to-SFT-50 rows plus unavailable SFT-50-to-SFT-100 rows.

- [ ] **Step 4: Commit and push**

Commit only the implementation, tests, plan, and documentation. Preserve unrelated untracked user files. Push the verified commit to `main`.

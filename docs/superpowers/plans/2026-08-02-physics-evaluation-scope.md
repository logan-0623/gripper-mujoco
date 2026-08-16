# Physics Evaluation Scope Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make physical evaluation run Flat and Graph by default, retain edge shuffle as an opt-in ablation, and support a deterministic per-count episode override for fast comparisons.

**Architecture:** Extend case generation with a validated count override, then derive rollout count and report scope from a dynamic policy-variant list. Keep the existing rollout, aggregation, checkpoint provenance, paired-state validation, and output paths unchanged.

**Tech Stack:** Python 3.12, argparse, PyTorch, MuJoCo, tqdm, pytest.

---

### Task 1: Deterministic Case-Count Override

**Files:**
- Modify: `tests/interaction_vla/test_physics_evaluate.py`
- Modify: `interaction_vla/physics_evaluate.py:134-174`

- [x] **Step 1: Write the failing case-count test**

Extend `test_physics_cases_are_unique_and_cover_id_count_and_crowded` with a
pilot-config count check and add a separate validation test:

```python
def test_physics_case_count_can_be_overridden_without_changing_groups() -> None:
    config = load_config("configs/physics_recovery_pilot_macos.yaml")

    full = make_physics_evaluation_cases(config)
    quick = make_physics_evaluation_cases(config, episodes_per_count=5)

    assert len(full) == 160
    assert len(quick) == 40
    assert {case.condition for case in quick} == {
        "id_normal",
        "count_ood",
        "crowded_ood",
        "controlled_randomization",
    }
    assert quick == make_physics_evaluation_cases(config, episodes_per_count=5)


@pytest.mark.parametrize("episodes_per_count", [0, -1])
def test_physics_case_count_override_must_be_positive(
    episodes_per_count: int,
) -> None:
    config = load_config("configs/physics_recovery_pilot_macos.yaml")

    with pytest.raises(ValueError, match="episodes_per_count must be positive"):
        make_physics_evaluation_cases(
            config,
            episodes_per_count=episodes_per_count,
        )
```

- [x] **Step 2: Run the new tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest \
  tests/interaction_vla/test_physics_evaluate.py::test_physics_case_count_can_be_overridden_without_changing_groups \
  tests/interaction_vla/test_physics_evaluate.py::test_physics_case_count_override_must_be_positive \
  -q
```

Expected: failures because `make_physics_evaluation_cases` does not accept
`episodes_per_count`.

- [x] **Step 3: Implement the validated override**

Change the case factory signature and resolve the loop bound before creating
groups:

```python
def make_physics_evaluation_cases(
    config: ExperimentConfig,
    *,
    episodes_per_count: int | None = None,
) -> tuple[PhysicsEvaluationCase, ...]:
    if config.backend != "franka_contact":
        raise ValueError("physical evaluation requires backend=franka_contact")
    resolved_episodes_per_count = (
        config.eval.episodes_per_count
        if episodes_per_count is None
        else int(episodes_per_count)
    )
    if resolved_episodes_per_count < 1:
        raise ValueError("episodes_per_count must be positive")
```

Replace the existing inner range with:

```python
for episode_index in range(resolved_episodes_per_count):
```

Do not change the `SeedSequence` inputs, groups, layouts, randomization flags,
or case IDs.

- [x] **Step 4: Run the focused tests and verify GREEN**

Run:

```bash
.venv/bin/python -m pytest \
  tests/interaction_vla/test_physics_evaluate.py::test_physics_cases_are_unique_and_cover_id_count_and_crowded \
  tests/interaction_vla/test_physics_evaluate.py::test_physics_case_count_can_be_overridden_without_changing_groups \
  tests/interaction_vla/test_physics_evaluate.py::test_physics_case_count_override_must_be_positive \
  -q
```

Expected: all selected tests pass.

### Task 2: Default Flat/Graph and Opt-In Edge Shuffle

**Files:**
- Modify: `tests/interaction_vla/test_physics_evaluate.py:326-434`
- Modify: `interaction_vla/physics_evaluate.py:574-689`

- [x] **Step 1: Change the integration test to require two variants by default**

Update the `make_physics_evaluation_cases` monkeypatch so it accepts the new
keyword:

```python
monkeypatch.setattr(
    physics_evaluate_module,
    "make_physics_evaluation_cases",
    lambda _config, *, episodes_per_count=None: cases,
)
```

Call the evaluator with a visible override:

```python
report_path = evaluate_from_config(
    "ignored.yaml",
    model_seeds=(0,),
    episodes_per_count=5,
    show_progress=True,
)
```

Replace the progress and scope assertions with:

```python
assert progress.kwargs["total"] == 4
assert progress.updates == 4
assert progress.closed
assert {postfix["policy"] for postfix in progress.postfixes} == {
    "flat",
    "graph",
}
assert report["evaluation_scope"] == {
    "model_seeds": [0],
    "episodes_per_count": 5,
    "case_count": 2,
    "rollout_count": 4,
    "policy_variants": ["flat", "graph"],
}
assert report["graph_vs_edge_shuffle"] == {"by_model_seed": {}}
```

- [x] **Step 2: Add an opt-in edge-shuffle integration test**

Extract the existing monkeypatch setup inside
`test_subset_evaluation_reports_scope_and_rollout_progress` into the local test
body only once by parameterizing the test:

```python
@pytest.mark.parametrize(
    ("include_edge_shuffle", "expected_variants"),
    [
        (False, ["flat", "graph"]),
        (True, ["flat", "graph", "graph_edge_shuffle"]),
    ],
)
def test_subset_evaluation_reports_scope_and_rollout_progress(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    include_edge_shuffle: bool,
    expected_variants: list[str],
) -> None:
```

Pass `include_edge_shuffle=include_edge_shuffle` to `evaluate_from_config` and
derive the exact assertions from `expected_variants`:

```python
expected_rollouts = len(cases) * len(expected_variants)
assert progress.kwargs["total"] == expected_rollouts
assert progress.updates == expected_rollouts
assert {postfix["policy"] for postfix in progress.postfixes} == set(
    expected_variants
)
assert report["evaluation_scope"] == {
    "model_seeds": [0],
    "episodes_per_count": 5,
    "case_count": len(cases),
    "rollout_count": expected_rollouts,
    "policy_variants": expected_variants,
}
if include_edge_shuffle:
    assert set(report["graph_vs_edge_shuffle"]["by_model_seed"]) == {"0"}
else:
    assert report["graph_vs_edge_shuffle"] == {"by_model_seed": {}}
```

- [x] **Step 3: Run the parameterized test and verify RED**

Run:

```bash
.venv/bin/python -m pytest \
  tests/interaction_vla/test_physics_evaluate.py::test_subset_evaluation_reports_scope_and_rollout_progress \
  -q
```

Expected: failures because `evaluate_from_config` lacks both new options and
still hardcodes three variants.

- [x] **Step 4: Implement dynamic policy variants and rollout totals**

Extend the evaluator signature:

```python
def evaluate_from_config(
    config_path: str | Path,
    *,
    model_seeds: Iterable[int] | None = None,
    include_edge_shuffle: bool = False,
    episodes_per_count: int | None = None,
    show_progress: bool = False,
) -> Path:
```

Resolve cases and policy variants immediately after loading the config:

```python
resolved_episodes_per_count = (
    config.eval.episodes_per_count
    if episodes_per_count is None
    else int(episodes_per_count)
)
cases = make_physics_evaluation_cases(
    config,
    episodes_per_count=resolved_episodes_per_count,
)
policy_variants = ["flat", "graph"]
if include_edge_shuffle:
    policy_variants.append("graph_edge_shuffle")
```

Change the tqdm total to:

```python
total=len(cases) * len(selected_seeds) * len(policy_variants)
```

Change the ablation branch to:

```python
if include_edge_shuffle and representation == "graph":
```

Write the actual scope to the report:

```python
report["evaluation_scope"] = {
    "model_seeds": list(selected_seeds),
    "episodes_per_count": resolved_episodes_per_count,
    "case_count": len(cases),
    "rollout_count": len(results),
    "policy_variants": policy_variants,
}
```

- [x] **Step 5: Run the integration test and verify GREEN**

Run:

```bash
.venv/bin/python -m pytest \
  tests/interaction_vla/test_physics_evaluate.py::test_subset_evaluation_reports_scope_and_rollout_progress \
  -q
```

Expected: both parameter combinations pass, with totals four and six.

### Task 3: CLI Arguments and User Commands

**Files:**
- Modify: `tests/interaction_vla/test_physics_evaluate.py:1-75`
- Modify: `interaction_vla/physics_evaluate.py:692-712`
- Modify: `README.md:153-188`

- [x] **Step 1: Write failing parser and main-forwarding tests**

Add `import sys` to the test module. Extend the parser test:

```python
def test_physics_evaluate_parser_accepts_scope_overrides() -> None:
    args = build_parser().parse_args(
        [
            "--config",
            "configs/physics_recovery_pilot_macos.yaml",
            "--model-seeds",
            "0",
            "--episodes-per-count",
            "5",
            "--include-edge-shuffle",
        ]
    )

    assert args.model_seeds == [0]
    assert args.episodes_per_count == 5
    assert args.include_edge_shuffle is True
```

Add a main-forwarding test:

```python
def test_physics_evaluate_main_forwards_scope_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report = tmp_path / "report.json"
    captured: dict[str, object] = {}

    def fake_evaluate(config_path: str, **kwargs: object) -> Path:
        captured.update({"config_path": config_path, **kwargs})
        return report

    monkeypatch.setattr(physics_evaluate_module, "evaluate_from_config", fake_evaluate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "physics_evaluate",
            "--config",
            "configs/physics_recovery_pilot_macos.yaml",
            "--model-seeds",
            "0",
            "--episodes-per-count",
            "5",
            "--include-edge-shuffle",
        ],
    )

    physics_evaluate_module.main()

    assert captured == {
        "config_path": "configs/physics_recovery_pilot_macos.yaml",
        "model_seeds": [0],
        "include_edge_shuffle": True,
        "episodes_per_count": 5,
        "show_progress": True,
    }
```

- [x] **Step 2: Run the CLI tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest \
  tests/interaction_vla/test_physics_evaluate.py::test_physics_evaluate_parser_accepts_scope_overrides \
  tests/interaction_vla/test_physics_evaluate.py::test_physics_evaluate_main_forwards_scope_overrides \
  -q
```

Expected: parser failure because both arguments are unknown.

- [x] **Step 3: Implement parser and forwarding**

Add the arguments in `build_parser`:

```python
parser.add_argument("--episodes-per-count", type=int)
parser.add_argument("--include-edge-shuffle", action="store_true")
```

Forward them from `main`:

```python
evaluate_from_config(
    args.config,
    model_seeds=args.model_seeds,
    include_edge_shuffle=args.include_edge_shuffle,
    episodes_per_count=args.episodes_per_count,
    show_progress=True,
)
```

- [x] **Step 4: Run all physics evaluation tests and verify GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/interaction_vla/test_physics_evaluate.py -q
```

Expected: all tests in the module pass.

- [x] **Step 5: Document default, quick, and ablation commands**

Replace the seed-0 evaluation command in `README.md` with the fast two-policy
command:

```bash
.venv/bin/python -m interaction_vla.physics_evaluate \
  --config configs/physics_recovery_pilot_macos.yaml \
  --model-seeds 0 \
  --episodes-per-count 5
```

Document that it runs 80 rollouts; omitting `--episodes-per-count` runs the full
320 Flat/Graph rollouts; adding `--include-edge-shuffle` runs 120 quick or 480
full rollouts. State that evaluation overwrites only the evaluation CSV/report,
not checkpoints or collected episodes.

### Task 4: Full Verification and Review

**Files:**
- Modify: `docs/superpowers/plans/2026-08-02-physics-evaluation-scope.md`

- [x] **Step 1: Run focused evaluator tests**

Run:

```bash
.venv/bin/python -m pytest tests/interaction_vla/test_physics_evaluate.py -q
```

Expected: zero failures.

- [x] **Step 2: Run the complete regression suite**

Run:

```bash
.venv/bin/python -m pytest -q
```

Expected: zero failures.

- [x] **Step 3: Compile source and tests**

Run:

```bash
.venv/bin/python -m compileall -q interaction_vla tests/interaction_vla
```

Expected: exit code zero and no output.

- [x] **Step 4: Verify case-count arithmetic without running rollouts**

Run:

```bash
.venv/bin/python -c 'from interaction_vla.config import load_config; from interaction_vla.physics_evaluate import make_physics_evaluation_cases; c=load_config("configs/physics_recovery_pilot_macos.yaml"); print(len(make_physics_evaluation_cases(c)), len(make_physics_evaluation_cases(c, episodes_per_count=5)))'
```

Expected output:

```text
160 40
```

- [x] **Step 5: Request a read-only code review**

Ask the reviewer to check default/opt-in semantics, exact tqdm totals, report
scope, paired-state preservation, validation order, CLI forwarding, and README
commands. Fix any Critical or Important findings, then rerun Steps 1 through 4.

- [x] **Step 6: Mark this plan complete and hand off commands**

Check every box only after fresh evidence. Do not offer Git integration because
the workspace is not a Git repository.

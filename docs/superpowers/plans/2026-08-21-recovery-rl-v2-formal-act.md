# Recovery RL v2 Formal ACT Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the selected recovery-RL backend as a reproducible ACT mechanism study with SFT, continued-SFT, Oracle-State, RL-head, and RL-representation conditions; measure nominal retention, recovery improvement, and probe trajectories at six immutable checkpoints.

**Architecture:** Consume only passing foundation gates and their bound distribution/backend/anchoring manifests. A new 1,200-state bank supports primary time-resolved probes; a formal trainer instantiates three actor conditions with one selected backend; paired evaluators and reports keep behavior, decodability, and plasticity separate.

**Tech Stack:** Python 3.12, PyTorch 2.10, NumPy, MuJoCo, LeRobot 0.6.1, pytest.

---

## Entry conditions

This plan cannot begin until the foundation completion gate in
`docs/superpowers/plans/2026-08-21-recovery-rl-v2-foundation.md` passes. The formal
code must read, hash, and validate these artifacts:

```text
gates/distribution.json
gates/backend.json
gates/oracle.json
gates/anchoring.json
manifests/cases.json
manifests/oracle_normalization.json
```

The selected backend and anchoring variant are inputs, not new command-line choices.
The frozen SFT and continued-SFT checkpoints come from the two read-only checkpoint
paths bound by the v2 config; formal code never writes into either parent tree.

## File map

Create:

- `interaction_vla/representation_study/state_bank/v2_builder.py`: balanced 1,200-state bank.
- `interaction_vla/representation_study/rl/formal.py`: condition/seed training orchestration.
- `interaction_vla/representation_study/rl/timeline.py`: six-checkpoint measurement schedule.
- `interaction_vla/representation_study/rl/formal_evaluation.py`: curve and held-out paired evaluation.
- `interaction_vla/representation_study/rl/formal_report.py`: evidence rows, statistics, and gates.
- focused pytest files for each component.

Modify:

- `interaction_vla/representation_study/extraction.py`: accept immutable snapshot checkpoints.
- `interaction_vla/representation_study/probes/training.py`: register v2 factor roles without changing v1 defaults.
- `interaction_vla/representation_study/cli.py`: add formal subcommands.
- `README.md`: add server execution order after verification.
- `ccfa.yaml`: update status only after real artifacts exist.

### Task 1: State Bank v2 selection and validation

**Files:**
- Create: `interaction_vla/representation_study/state_bank/v2_builder.py`
- Create: `tests/interaction_vla/representation_study/test_state_bank_v2.py`

- [ ] **Step 1: Write failing balance and leakage tests**

```python
def test_state_bank_v2_has_exact_primary_balance(tmp_path: Path) -> None:
    report = build_state_bank_v2(fake_case_manifest(), output_dir=tmp_path, seed=11)
    assert report.record_count == 1200
    assert report.stratum_counts == {"nominal": 400, "perturbation": 400, "recovery": 400}


def test_state_bank_v2_has_no_source_seed_overlap(tmp_path: Path) -> None:
    build_state_bank_v2(fake_case_manifest(), output_dir=tmp_path, seed=11)
    split = load_v2_split(tmp_path / "split.json")
    assert split.source_seeds("train").isdisjoint(split.source_seeds("validation"))
    assert split.source_seeds("train").isdisjoint(split.source_seeds("test"))
    assert split.source_seeds("validation").isdisjoint(split.source_seeds("test"))
```

- [ ] **Step 2: Verify RED**

Run the new test file; expect import failure.

- [ ] **Step 3: Implement exact grouped selection**

```python
PRIMARY_STRATA = {"nominal": 400, "perturbation": 400, "recovery": 400}
PRIMARY_FACTORS = ("geometry", "phase", "recovery_state", "next_relation")
SECONDARY_FACTORS = ("contact", "stable_grasp")


def build_state_bank_v2(manifest: RecoveryCaseManifest, *, output_dir: Path, seed: int) -> StateBankV2Report:
    records = collect_nonterminal_records(manifest)
    selected = grouped_stratified_select(
        records, counts=PRIMARY_STRATA, group_key="source_seed", seed=seed
    )
    validate_source_seed_isolation(selected)
    write_v2_bank_atomic(output_dir, selected, manifest_hash=manifest.sha256)
    return validate_state_bank_v2(output_dir)
```

Selection runs after source-seed partitioning. Every test recovery kind must contain
multiple independent groups. Reject insufficient accepted states rather than
duplicating frames. Store primary/secondary roles in `ontology.json`; entity remains
descriptive but is excluded from primary rows.

- [ ] **Step 4: Test and commit**

```bash
HF_HOME=/tmp/gripper-mujoco-pytest-hf-cache \
  .venv-lerobot/bin/python -m pytest -q \
  tests/interaction_vla/representation_study/test_state_bank_v2.py
git add interaction_vla/representation_study/state_bank/v2_builder.py \
  tests/interaction_vla/representation_study/test_state_bank_v2.py
git commit -m "feat: build balanced recovery State Bank v2"
```

### Task 2: Formal condition and seed orchestration

**Files:**
- Create: `interaction_vla/representation_study/rl/formal.py`
- Create: `tests/interaction_vla/representation_study/test_formal_recovery_study.py`

- [ ] **Step 1: Write failing matrix tests**

```python
def test_formal_matrix_has_registered_conditions_and_three_seeds() -> None:
    matrix = formal_matrix(base_seed=2057736129)
    assert tuple(matrix.conditions) == (
        "sft", "continued_sft", "oracle_state", "rl_head", "rl_representation"
    )
    assert len(matrix.training_seeds) == 3


def test_formal_training_requires_all_foundation_gates(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="anchoring gate"):
        run_formal_training(
            fake_formal_config(tmp_path), condition="rl_head", seed_index=0, resume=False
        )
```

- [ ] **Step 2: Verify RED**

Run the formal study tests; expect import failure.

- [ ] **Step 3: Implement immutable run identities**

```python
FORMAL_CONDITIONS = (
    "sft", "continued_sft", "oracle_state", "rl_head", "rl_representation"
)

@dataclass(frozen=True)
class FormalRun:
    condition: str
    seed_index: int
    seed: int
    backend: str
    anchoring: str
    output_dir: Path
    binding: str
```

Derive seeds with `SeedSequence((base_seed, seed_index, 0x524C5632))`. Bind parent
checkpoint, selected backend, anchor variant, case manifest, State Bank, reward,
normalization, and trainable groups.

Create one reference timeline for each non-RL control. Its six entries all bind the
same immutable parent checkpoint and are marked `constant_control: true`; no copied
model weights and no extra supervised updates are produced in v2. In reports,
continued SFT therefore means the already completed 1,000-step continued-SFT
checkpoint, not a newly retuned control. Behavioral evaluation still uses every
registered policy seed, while a constant control's probe measurement is performed
once and is never counted as three independent representation runs.

- [ ] **Step 4: Implement condition actors and backend dispatch**

`oracle_state` uses `OracleResidualActor`; `rl_head` freezes every ACT parameter;
`rl_representation` unfreezes only the registered late-fusion group. SFT and
continued-SFT are evaluation-only controls.

```python
def make_formal_backend(run: FormalRun, runtime: LeRobotPolicyBackend):
    if run.backend == "ppo":
        return PPOV2.for_formal_run(run, runtime)
    if run.backend == "sac":
        return SAC.for_formal_run(run, runtime)
    raise ValueError(f"selected backend is incompatible: {run.backend}")
```

Train exactly 20,480 environment steps and save only through `SnapshotStore`.

- [ ] **Step 5: Test and commit**

```bash
HF_HOME=/tmp/gripper-mujoco-pytest-hf-cache \
  .venv-lerobot/bin/python -m pytest -q \
  tests/interaction_vla/representation_study/test_formal_recovery_study.py
git add interaction_vla/representation_study/rl/formal.py \
  tests/interaction_vla/representation_study/test_formal_recovery_study.py
git commit -m "feat: orchestrate formal ACT recovery runs"
```

### Task 3: Time-resolved checkpoint measurement

**Files:**
- Create: `interaction_vla/representation_study/rl/timeline.py`
- Modify: `interaction_vla/representation_study/extraction.py`
- Modify: `interaction_vla/representation_study/probes/training.py`
- Create: `tests/interaction_vla/representation_study/test_recovery_timeline.py`

- [ ] **Step 1: Write failing schedule tests**

```python
def test_timeline_registers_six_linear_and_two_mlp_measurements() -> None:
    timeline = measurement_timeline()
    assert timeline.linear_steps == (0, 4096, 8192, 12288, 16384, 20480)
    assert timeline.mlp_steps == (0, 20480)


def test_timeline_rejects_incomplete_snapshot(tmp_path: Path) -> None:
    snapshot = tmp_path / "step_004096"
    snapshot.mkdir()
    with pytest.raises(ValueError, match="COMPLETED"):
        measure_snapshot(snapshot, fake_measurement_context())
```

- [ ] **Step 2: Verify RED**

Run timeline tests; expect missing module.

- [ ] **Step 3: Load immutable snapshots as manifests**

```python
@dataclass(frozen=True)
class TimelinePoint:
    condition: str
    seed_index: int
    environment_steps: int
    snapshot: Path
    snapshot_hash: str
```

Extend extraction with a snapshot-backed manifest that requires `COMPLETED`, verifies
the formal-run binding, and adds environment steps to every output path. Preserve v1
manifest behavior.

- [ ] **Step 4: Run registered probes and ledger them**

At all six points run linear geometry, phase, recovery state/type, and next-relation
probes. At steps 0 and 20,480 also run the fixed shallow MLP. Contact and stable
grasp remain secondary final metrics. Write a ledger row only after latent and every
required probe report passes.

For `constant_control` timelines, hash-check all six entries against the same parent
checkpoint, extract latents once, and reference the single measurement artifact at
the remaining steps. The report renders a constant trajectory but retains one
independent probe estimate.

- [ ] **Step 5: Test and commit**

```bash
HF_HOME=/tmp/gripper-mujoco-pytest-hf-cache \
  .venv-lerobot/bin/python -m pytest -q \
  tests/interaction_vla/representation_study/test_recovery_timeline.py \
  tests/interaction_vla/representation_study/test_extraction.py \
  tests/interaction_vla/representation_study/test_probes.py
git add interaction_vla/representation_study/rl/timeline.py \
  interaction_vla/representation_study/extraction.py \
  interaction_vla/representation_study/probes/training.py \
  tests/interaction_vla/representation_study/test_recovery_timeline.py
git commit -m "feat: measure recovery representations over training"
```

### Task 4: Nominal-retention and recovery-improvement curves

**Files:**
- Create: `interaction_vla/representation_study/rl/formal_evaluation.py`
- Extend: `tests/interaction_vla/representation_study/test_formal_recovery_study.py`

- [ ] **Step 1: Write failing paired-evaluation tests**

```python
def test_curve_evaluation_uses_same_cases_at_every_checkpoint(tmp_path: Path) -> None:
    report = evaluate_formal_timeline(fake_completed_timeline(tmp_path), fake_curve_manifest())
    ids = [tuple(point.case_ids) for point in report.points]
    assert len(set(ids)) == 1


def test_final_evaluation_has_fifty_cases_per_distribution(tmp_path: Path) -> None:
    report = evaluate_formal_final(fake_final_snapshot(tmp_path), fake_final_manifest())
    assert report.nominal.episodes == 50
    assert report.recovery.episodes == 50
```

- [ ] **Step 2: Verify RED**

Run formal study tests; expect missing evaluator.

- [ ] **Step 3: Implement curve and final contracts**

Curve evaluation uses the same 30 nominal and 30 recovery development cases at all
six checkpoints. Final evaluation uses 50 nominal and 50 recovery held-out cases.

```python
@dataclass(frozen=True)
class DistributionOutcome:
    episodes: int
    success_rate: float
    timeout_rate: float
    drop_rate: float
    wrong_object_rate: float
    mean_steps: float
    mean_residual_norm: float
    action_clipping_rate: float
    action_smoothness: float
```

Preserve paired rows with condition, training seed, step, case id, source seed,
family, intervention kind, policy seed, reward terms, action diagnostics, and outcome.

- [ ] **Step 4: Test and commit**

```bash
HF_HOME=/tmp/gripper-mujoco-pytest-hf-cache \
  .venv-lerobot/bin/python -m pytest -q \
  tests/interaction_vla/representation_study/test_formal_recovery_study.py
git add interaction_vla/representation_study/rl/formal_evaluation.py \
  tests/interaction_vla/representation_study/test_formal_recovery_study.py
git commit -m "feat: evaluate retention and recovery curves"
```

### Task 5: Formal statistics and evidence report

**Files:**
- Create: `interaction_vla/representation_study/rl/formal_report.py`
- Create: `tests/interaction_vla/representation_study/test_formal_recovery_report.py`

- [ ] **Step 1: Write failing report tests**

```python
def test_report_keeps_evidence_axes_separate(tmp_path: Path) -> None:
    report = build_formal_report(fake_complete_formal_artifacts(tmp_path))
    assert {row["axis"] for row in report.rows} >= {"accessible", "useful", "plasticity"}
    assert all(row["unit"] == "episode" for row in report.rows if row["axis"] == "useful")


def test_expansion_gate_requires_two_of_three_seed_directions() -> None:
    assert expansion_gate([0.10, 0.05, -0.02]).passed is True
    assert expansion_gate([0.10, -0.01, -0.04]).passed is False
```

- [ ] **Step 2: Verify RED**

Run report tests; expect missing module.

- [ ] **Step 3: Implement primary metrics and paired statistics**

For each condition and seed calculate recovery normalized AUC, nominal retention,
final paired recovery delta, final paired nominal delta, and probe trajectories. Use
paired episode bootstrap clustered by policy seed for final effects.

- [ ] **Step 4: Implement descriptive trajectory associations**

Join probe and success changes by condition, training seed, and checkpoint. Report
Spearman associations as descriptive unless at least three seeds and multiple time
points exist. Never label them causal.

- [ ] **Step 5: Write machine-readable artifacts**

```text
formal/result_rows.json
formal/curve_rows.json
formal/probe_trajectory_rows.json
formal/pairwise_effects.json
formal/study_report.json
formal/study_report.md
```

Missing required artifacts set `complete: false`; never substitute smoke results.

- [ ] **Step 6: Test and commit**

```bash
HF_HOME=/tmp/gripper-mujoco-pytest-hf-cache \
  .venv-lerobot/bin/python -m pytest -q \
  tests/interaction_vla/representation_study/test_formal_recovery_report.py
git add interaction_vla/representation_study/rl/formal_report.py \
  tests/interaction_vla/representation_study/test_formal_recovery_report.py
git commit -m "feat: report formal recovery evidence"
```

### Task 6: Formal CLI, documentation, and verification

**Files:**
- Modify: `interaction_vla/representation_study/cli.py`
- Modify: `tests/interaction_vla/representation_study/test_cli.py`
- Modify: `README.md`
- Modify only after real runs: `ccfa.yaml`

- [ ] **Step 1: Write failing formal CLI tests**

```python
@pytest.mark.parametrize("command", ["state-bank", "train", "measure", "evaluate", "report"])
def test_formal_commands_parse(command: str) -> None:
    args = build_parser().parse_args(
        ["recovery-rl", "formal", command, "--config", "v2.yaml"]
    )
    assert args.family == "recovery-rl"
    assert args.command == "formal"
    assert args.formal_command == command
```

- [ ] **Step 2: Verify RED**

Run CLI tests; expect argparse rejection.

- [ ] **Step 3: Implement formal subcommands**

`train` requires `--condition`, `--seed-index`, and optional `--resume`; `measure`
and `evaluate` accept the same run identity; `state-bank` and `report` operate on
the bound study. All commands validate foundation gates first.

- [ ] **Step 4: Document exact server order**

```bash
.venv-lerobot/bin/python -m interaction_vla.representation_study recovery-rl formal state-bank --config "$CONFIG"

for condition in oracle_state rl_head rl_representation
do
  for seed_index in 0 1 2
  do
    .venv-lerobot/bin/python -m interaction_vla.representation_study recovery-rl formal train \
      --config "$CONFIG" --condition "$condition" --seed-index "$seed_index"
    .venv-lerobot/bin/python -m interaction_vla.representation_study recovery-rl formal measure \
      --config "$CONFIG" --condition "$condition" --seed-index "$seed_index"
    .venv-lerobot/bin/python -m interaction_vla.representation_study recovery-rl formal evaluate \
      --config "$CONFIG" --condition "$condition" --seed-index "$seed_index"
  done
done

.venv-lerobot/bin/python -m interaction_vla.representation_study recovery-rl formal report --config "$CONFIG"
```

SFT and continued-SFT are evaluated without RL training commands.

```bash
for condition in sft continued_sft
do
  .venv-lerobot/bin/python -m interaction_vla.representation_study recovery-rl formal measure \
    --config "$CONFIG" --condition "$condition" --seed-index 0
  for seed_index in 0 1 2
  do
    .venv-lerobot/bin/python -m interaction_vla.representation_study recovery-rl formal evaluate \
      --config "$CONFIG" --condition "$condition" --seed-index "$seed_index"
  done
done
```

- [ ] **Step 5: Run fresh verification**

```bash
HF_HOME=/tmp/gripper-mujoco-pytest-hf-cache \
  .venv-lerobot/bin/python -m pytest -q \
  tests/interaction_vla/representation_study

HF_HOME=/tmp/gripper-mujoco-pytest-hf-cache \
  .venv-lerobot/bin/python -m pytest -q
```

Expected: zero failures and only declared environment-dependent skips.

- [ ] **Step 6: Commit orchestration**

```bash
git add interaction_vla/representation_study/cli.py \
  tests/interaction_vla/representation_study/test_cli.py \
  README.md
git commit -m "docs: run formal ACT recovery study"
```

Update `ccfa.yaml` in a separate commit only after real artifacts establish gate
status; code tests cannot mark experiments complete.

## Formal completion gate

The ACT protocol is ready for a separate SmolVLA plan only when:

- three seeds finish for Oracle-State, RL-head, and RL-representation;
- every run has all six immutable checkpoints;
- paired nominal/recovery final evaluations are complete;
- candidate representation nominal forgetting is no worse than 10 percentage points;
- RL-representation-minus-RL-head agrees in at least two of three seeds;
- time-resolved primary probes are joined to success curves;
- the report is `complete: true` and the full test suite passes.

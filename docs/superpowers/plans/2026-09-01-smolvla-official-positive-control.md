# SmolVLA Official Positive-Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (\`- [ ]\`) syntax for tracking.

**Goal:** Add a zero-training, storage-light official SmolVLA positive-control experiment that decides whether StableGrasp functional recruitment survives in a successful LIBERO policy and routes a failed result to one Contact replication or a declared pivot.

**Architecture:** Keep Protocol-v3 immutable. Add one focused \`positive_control.py\` orchestration module under the existing LIBERO study, reuse the audited State Bank, cross-fit cell runner, and intervention primitives, and store all new artifacts under \`protocol_v4/positive_control/\`. Extend latent extraction with an optional tap subset so the server stores only \`action_expert_input\`.

**Tech Stack:** Python 3.12, PyTorch, NumPy, existing LeRobot 0.6.1 adapters, pytest, YAML/JSON artifact contracts.

---

## File map

- Create \`interaction_vla/representation_study/libero/positive_control.py\`: plan binding, one-condition probe, intervention orchestration, and kill/pivot report.
- Modify \`interaction_vla/representation_study/libero/latents.py\`: optional validated semantic-tap subset; current all-tap behavior remains the default.
- Modify \`interaction_vla/representation_study/libero/cli.py\`: five positive-control subcommands and dispatch.
- Create \`tests/interaction_vla/representation_study/libero/test_positive_control.py\`: pure artifact, gate, and binding tests.
- Modify \`tests/interaction_vla/representation_study/libero/test_cli.py\`: parser and dispatch coverage.
- Modify \`SERVER_RUNBOOK.md\`: official evaluation import and positive-control server commands.
- Modify \`ccfa.yaml\`: register implementation-only Protocol-v4 without changing historical Protocol-v3 statuses.

### Task 1: Selective semantic-tap extraction

**Files:**
- Modify: \`interaction_vla/representation_study/libero/latents.py:200-325\`
- Test: \`tests/interaction_vla/representation_study/libero/test_positive_control.py\`

- [ ] **Step 1: Write the failing tap-selection tests**

\`\`\`python
import pytest

from interaction_vla.representation_study.libero.latents import validate_requested_taps
from interaction_vla.representation_study.libero.taps import SEMANTIC_TAPS


def test_requested_taps_default_to_all_semantic_taps() -> None:
    assert validate_requested_taps(None) == tuple(SEMANTIC_TAPS)


def test_requested_taps_are_unique_known_and_ordered() -> None:
    assert validate_requested_taps(("action_expert_input",)) == (
        "action_expert_input",
    )
    with pytest.raises(ValueError, match="unknown semantic tap"):
        validate_requested_taps(("missing",))
    with pytest.raises(ValueError, match="duplicate semantic tap"):
        validate_requested_taps(("pre_action", "pre_action"))
\`\`\`

- [ ] **Step 2: Run the test to verify it fails**

\`\`\`bash
.venv/bin/python -m pytest \
  tests/interaction_vla/representation_study/libero/test_positive_control.py \
  -q
\`\`\`

Expected: collection fails because \`validate_requested_taps\` does not exist.

- [ ] **Step 3: Add the minimal selector and thread it through extraction**

\`\`\`python
def validate_requested_taps(taps: Sequence[str] | None) -> tuple[str, ...]:
    selected = tuple(SEMANTIC_TAPS if taps is None else taps)
    if not selected:
        raise ValueError("at least one semantic tap is required")
    if len(set(selected)) != len(selected):
        raise ValueError("duplicate semantic tap")
    unknown = sorted(set(selected) - set(SEMANTIC_TAPS))
    if unknown:
        raise ValueError(f"unknown semantic tap: {unknown}")
    return selected
\`\`\`

Add \`taps: Sequence[str] | None = None\` to
\`extract_smolvla_latents_from_checkpoint\`, build writers only for the validated
selection, and write only selected cache and tap-metadata entries. Leave every
existing call unchanged so it still extracts all four taps.

- [ ] **Step 4: Run focused and existing latent tests**

\`\`\`bash
.venv/bin/python -m pytest \
  tests/interaction_vla/representation_study/libero/test_positive_control.py \
  tests/interaction_vla/representation_study/libero/test_latents.py \
  -q
\`\`\`

Expected: PASS.

- [ ] **Step 5: Commit**

\`\`\`bash
git add interaction_vla/representation_study/libero/latents.py \
  tests/interaction_vla/representation_study/libero/test_positive_control.py
git commit -m "feat: support selective SmolVLA latent taps"
\`\`\`

### Task 2: Bind the official checkpoint and LeRobot evaluation

**Files:**
- Create: \`interaction_vla/representation_study/libero/positive_control.py\`
- Test: \`tests/interaction_vla/representation_study/libero/test_positive_control.py\`

- [ ] **Step 1: Write failing plan-binding tests**

\`\`\`python
import json
from pathlib import Path

import pytest

from interaction_vla.representation_study.libero.positive_control import (
    official_success_rate,
    positive_control_decision,
)


def test_official_success_rate_reads_lerobot_percent(tmp_path: Path) -> None:
    report = tmp_path / "eval_info.json"
    report.write_text(json.dumps({"overall": {"pc_success": 70.0, "n_episodes": 10}}))
    assert official_success_rate(report) == (0.70, 10)


def test_official_success_rate_rejects_invalid_values(tmp_path: Path) -> None:
    report = tmp_path / "eval_info.json"
    report.write_text(json.dumps({"overall": {"pc_success": float("nan"), "n_episodes": 10}}))
    with pytest.raises(ValueError, match="finite"):
        official_success_rate(report)


def test_floor_policy_stops_before_probe_claims() -> None:
    result = positive_control_decision(
        success_rate=0.20,
        accessible=True,
        specificity_passed=True,
        usage_ci=(0.1, 0.2),
        factor="stable_grasp",
    )
    assert result["decision"] == "failed_policy_floor"
\`\`\`

- [ ] **Step 2: Run and verify failure**

\`\`\`bash
.venv/bin/python -m pytest \
  tests/interaction_vla/representation_study/libero/test_positive_control.py \
  -q
\`\`\`

Expected: import failure for \`positive_control\`.

- [ ] **Step 3: Implement immutable plan creation**

\`\`\`python
POSITIVE_CONTROL_SCHEMA = "libero_smolvla_official_positive_control_v1"
PRIMARY_TAP = "action_expert_input"
SUPPORTED_FACTORS = ("stable_grasp", "contact")


def official_success_rate(path: Path) -> tuple[float, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    overall = payload.get("overall")
    if not isinstance(overall, Mapping):
        raise ValueError("LeRobot evaluation has no overall metrics")
    percent = float(overall.get("pc_success", float("nan")))
    episodes = int(overall.get("n_episodes", 0))
    if not np.isfinite(percent) or not 0.0 <= percent <= 100.0 or episodes <= 0:
        raise ValueError("LeRobot evaluation metrics must be finite and non-empty")
    return percent / 100.0, episodes
\`\`\`

\`plan_positive_control(config, checkpoint, eval_dir)\` must require
\`checkpoint/config.json\` and \`eval_dir/eval_info.json\`; calculate checkpoint,
evaluation, State Bank, config, and implementation hashes; record the open-source
runtime contract; fail at or below 0.20 success; and atomically create
\`protocol_v4/positive_control/plan.json\` while rejecting changed bindings.

- [ ] **Step 4: Run tests**

\`\`\`bash
.venv/bin/python -m pytest \
  tests/interaction_vla/representation_study/libero/test_positive_control.py \
  -q
\`\`\`

Expected: PASS.

- [ ] **Step 5: Commit**

\`\`\`bash
git add interaction_vla/representation_study/libero/positive_control.py \
  tests/interaction_vla/representation_study/libero/test_positive_control.py
git commit -m "feat: bind official SmolVLA positive control"
\`\`\`

### Task 3: Extract the official action-expert cache and run cross-fit cells

**Files:**
- Modify: \`interaction_vla/representation_study/libero/positive_control.py\`
- Test: \`tests/interaction_vla/representation_study/libero/test_positive_control.py\`

- [ ] **Step 1: Write failing artifact-routing test**

\`\`\`python
from interaction_vla.representation_study.libero.positive_control import (
    positive_control_root,
)


def test_positive_control_uses_protocol_v4_without_touching_v3(tmp_path: Path) -> None:
    root = positive_control_root(tmp_path)
    assert root == tmp_path / "protocol_v4" / "positive_control"
    assert "protocol_v3" not in str(root)
\`\`\`

- [ ] **Step 2: Run and verify failure**

\`\`\`bash
.venv/bin/python -m pytest \
  tests/interaction_vla/representation_study/libero/test_positive_control.py \
  -q
\`\`\`

Expected: \`positive_control_root\` is missing.

- [ ] **Step 3: Implement extraction and probe orchestration**

\`extract_positive_control(config, batch_size)\` calls:

\`\`\`python
extract_smolvla_latents_from_checkpoint(
    config,
    checkpoint=str(plan["checkpoint"]),
    checkpoint_id=f"official:{plan['checkpoint_sha256'][:16]}",
    checkpoint_hash=str(plan["checkpoint_sha256"]),
    output_dir=positive_control_root(config.output_dir) / "latents",
    label="official_smolvla_libero",
    batch_size=batch_size,
    runtime_binding=True,
    report_schema="libero_smolvla_positive_control_latents_v1",
    report_fields={"plan_sha256": _file_sha256(plan_path)},
    taps=(PRIMARY_TAP,),
)
\`\`\`

\`run_positive_control_probe(config)\` reuses the existing State Bank,
\`build_crossfit_manifest(..., split_name="episode_group")\`,
\`run_crossfit_cell\`, existing probe seed offsets, and shortcut baselines. Run only
\`stable_grasp\`, \`contact\`, \`phase\`, and \`geometry\`. Store immutable cells
under \`probe/cells/<factor>.json.gz\` and write \`probe/report.json\`.
Accessibility keeps the Protocol-v3 definition.

- [ ] **Step 4: Run tests and existing cross-fit tests**

\`\`\`bash
.venv/bin/python -m pytest \
  tests/interaction_vla/representation_study/libero/test_positive_control.py \
  tests/interaction_vla/representation_study/libero/test_crossfit_probes.py \
  -q
\`\`\`

Expected: PASS.

- [ ] **Step 5: Commit**

\`\`\`bash
git add interaction_vla/representation_study/libero/positive_control.py \
  tests/interaction_vla/representation_study/libero/test_positive_control.py
git commit -m "feat: probe official SmolVLA interaction state"
\`\`\`

### Task 4: Reuse factor intervention primitives and encode the kill decision

**Files:**
- Modify: \`interaction_vla/representation_study/libero/positive_control.py\`
- Modify: \`interaction_vla/representation_study/libero/recruitment.py\`
- Test: \`tests/interaction_vla/representation_study/libero/test_positive_control.py\`
- Test: \`tests/interaction_vla/representation_study/libero/test_recruitment.py\`

- [ ] **Step 1: Write failing decision tests**

\`\`\`python
def test_positive_usage_authorizes_longitudinal_training() -> None:
    result = positive_control_decision(
        success_rate=0.70,
        accessible=True,
        specificity_passed=True,
        usage_ci=(0.01, 0.03),
        factor="stable_grasp",
    )
    assert result["decision"] == "continue_official_longitudinal"


def test_stablegrasp_usage_failure_routes_once_to_contact() -> None:
    result = positive_control_decision(
        success_rate=0.70,
        accessible=True,
        specificity_passed=True,
        usage_ci=(-0.01, 0.01),
        factor="stable_grasp",
    )
    assert result["decision"] == "replicate_contact_once"


def test_contact_failure_kills_recruitment_and_names_pivot() -> None:
    result = positive_control_decision(
        success_rate=0.70,
        accessible=True,
        specificity_passed=True,
        usage_ci=(-0.01, 0.01),
        factor="contact",
    )
    assert result["decision"] == "pivot_interaction_supervised_sft"
    assert not result["authorize_longitudinal_training"]
\`\`\`

- [ ] **Step 2: Run and verify failure**

\`\`\`bash
.venv/bin/python -m pytest \
  tests/interaction_vla/representation_study/libero/test_positive_control.py \
  -q
\`\`\`

Expected: decision assertions fail until all branches are implemented.

- [ ] **Step 3: Extract only general intervention execution from recruitment**

Do not change Protocol-v3 bindings or results. Move the already-tested
one-checkpoint operations into a parameterized helper:

\`\`\`python
@dataclass(frozen=True)
class SingleCheckpointRecruitment:
    condition: str
    checkpoint: Path
    checkpoint_sha256: str
    latent_root: Path
    probe_root: Path
    output_root: Path
    factor: str
    tap: str = "action_expert_input"
\`\`\`

The helper reuses \`binary_raw_probe\`, \`consensus_direction\`, \`_target_delta\`,
\`same_norm_random_delta\`, \`specificity_gate\`, \`FinalDenoisingDeltaHook\`,
\`_predict\`, \`_action_effect\`, and episode-clustered intervals. Existing
Protocol-v3 files are never rewritten.

For the official control, write:

\`\`\`text
intervention/<factor>/specificity.json
intervention/<factor>/actions.npz
intervention/<factor>/action_sensitivity.json
\`\`\`

The primary usage interval is total first-action targeted-minus-random displacement.

- [ ] **Step 4: Implement final report decision**

\`\`\`python
def positive_control_decision(
    *,
    success_rate: float,
    accessible: bool,
    specificity_passed: bool,
    usage_ci: tuple[float, float],
    factor: str,
) -> dict[str, object]:
    if success_rate <= 0.20:
        decision = "failed_policy_floor"
    elif not accessible:
        decision = (
            "replicate_contact_once"
            if factor == "stable_grasp"
            else "pivot_interaction_supervised_sft"
        )
    elif not specificity_passed:
        decision = "failed_specificity"
    elif usage_ci[0] > 0.0:
        decision = "continue_official_longitudinal"
    elif factor == "stable_grasp":
        decision = "replicate_contact_once"
    else:
        decision = "pivot_interaction_supervised_sft"
    return {
        "decision": decision,
        "authorize_longitudinal_training": (
            decision == "continue_official_longitudinal"
        ),
    }
\`\`\`

\`report_positive_control\` binds plan, latent, probe, specificity, and action
reports by hash and writes one immutable \`report.json\`.

- [ ] **Step 5: Run recruitment and positive-control tests**

\`\`\`bash
.venv/bin/python -m pytest \
  tests/interaction_vla/representation_study/libero/test_positive_control.py \
  tests/interaction_vla/representation_study/libero/test_recruitment.py \
  -q
\`\`\`

Expected: PASS and no Protocol-v3 fixture changes.

- [ ] **Step 6: Commit**

\`\`\`bash
git add interaction_vla/representation_study/libero/positive_control.py \
  interaction_vla/representation_study/libero/recruitment.py \
  tests/interaction_vla/representation_study/libero/test_positive_control.py \
  tests/interaction_vla/representation_study/libero/test_recruitment.py
git commit -m "feat: gate official SmolVLA functional recruitment"
\`\`\`

### Task 5: CLI, CCFA registry, and server commands

**Files:**
- Modify: \`interaction_vla/representation_study/libero/cli.py\`
- Modify: \`tests/interaction_vla/representation_study/libero/test_cli.py\`
- Modify: \`SERVER_RUNBOOK.md\`
- Modify: \`ccfa.yaml\`

- [ ] **Step 1: Write failing CLI parser test**

\`\`\`python
def test_positive_control_cli_has_five_gated_commands() -> None:
    parser = build_parser()
    config = "configs/representation_study/libero_smolvla_linux_cuda.yaml"
    plan = parser.parse_args([
        "libero", "positive-control", "plan", "--config", config,
        "--checkpoint", "/tmp/model", "--eval-dir", "/tmp/eval",
    ])
    assert plan.libero_command == "plan"
    for command in ("extract", "probe", "intervene", "report"):
        args = parser.parse_args([
            "libero", "positive-control", command, "--config", config,
        ])
        assert args.libero_command == command
\`\`\`

- [ ] **Step 2: Run and verify parser failure**

\`\`\`bash
.venv/bin/python -m pytest \
  tests/interaction_vla/representation_study/libero/test_cli.py::test_positive_control_cli_has_five_gated_commands \
  -q
\`\`\`

Expected: argparse rejects \`positive-control\`.

- [ ] **Step 3: Add parser and dispatch**

Add:

\`\`\`text
positive-control plan --checkpoint PATH --eval-dir PATH
positive-control extract --batch-size 32
positive-control probe
positive-control intervene --factor stable_grasp|contact --max-states 1600 --batch-size 32
positive-control report --factor stable_grasp|contact
\`\`\`

Dispatch lazily imports \`positive_control.py\`. No command starts training or paired
closed-loop evaluation.

- [ ] **Step 4: Register implementation state and document exact commands**

Add to \`ccfa.yaml\`:

\`\`\`yaml
- id: SMOLVLA_OFFICIAL_POSITIVE_CONTROL
  name: "Official SmolVLA functional-recruitment kill test"
  status: implementation_only
  role: successful_policy_kill_test
  requires: [LIBERO_STATE_BANK, SMOLVLA_FUNCTIONAL_RECRUITMENT]
  gate: official_success_then_stablegrasp_accessibility_specificity_usage
\`\`\`

Document the offline Hugging Face environment, official LeRobot eval directory,
five commands, expected artifact paths, and the instruction not to train the
full-SFT trajectory unless \`authorize_longitudinal_training\` is true.

- [ ] **Step 5: Run CLI tests**

\`\`\`bash
.venv/bin/python -m pytest \
  tests/interaction_vla/representation_study/libero/test_cli.py \
  -q
\`\`\`

Expected: PASS.

- [ ] **Step 6: Commit**

\`\`\`bash
git add interaction_vla/representation_study/libero/cli.py \
  tests/interaction_vla/representation_study/libero/test_cli.py \
  SERVER_RUNBOOK.md ccfa.yaml
git commit -m "docs: add official SmolVLA decision protocol"
\`\`\`

### Task 6: Full verification and server handoff

**Files:**
- Verify only; fix only files listed above if a check fails.

- [ ] **Step 1: Run the focused LIBERO representation suite**

\`\`\`bash
.venv/bin/python -m pytest \
  tests/interaction_vla/representation_study/libero/test_positive_control.py \
  tests/interaction_vla/representation_study/libero/test_latents.py \
  tests/interaction_vla/representation_study/libero/test_crossfit_probes.py \
  tests/interaction_vla/representation_study/libero/test_recruitment.py \
  tests/interaction_vla/representation_study/libero/test_cli.py \
  -q
\`\`\`

Expected: PASS.

- [ ] **Step 2: Verify immutable evidence and formatting**

\`\`\`bash
git diff 55f0bc1 -- docs/results/libero_smolvla_protocol_v3 \
  outputs/representation_study/libero_smolvla/protocol_v3
git diff --check
\`\`\`

Expected: no Protocol-v3 diff and no whitespace errors.

- [ ] **Step 3: Run no-GPU CLI smoke checks**

\`\`\`bash
.venv/bin/python -m interaction_vla.representation_study \
  libero positive-control plan --help
.venv/bin/python -m interaction_vla.representation_study \
  libero positive-control intervene --help
\`\`\`

Expected: both commands exit 0 and show only registered options.

- [ ] **Step 4: Inspect the final diff**

\`\`\`bash
git status --short
git diff --stat
\`\`\`

Expected: only deliberate implementation changes; unrelated untracked files remain
unstaged.


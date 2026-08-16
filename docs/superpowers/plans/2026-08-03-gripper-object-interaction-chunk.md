# Gripper–Object Interaction Chunking v3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fair Flat-versus-Graph physical manipulation experiment whose only method variable is the representation encoder, using 200 source-split expert demonstrations, exactly 25% recovery loss mass, a shared H=8 temporal controller, and strict in-box placement.

**Architecture:** Keep the existing Franka contact environment, graph schema, matched Flat/Graph encoders, expert gate, and 7D action contract. Add focused modules for source-level splitting, strict placement geometry, sequence training, and shared chunked rollout control; route training, evaluation, dashboard, native viewer, and GIF export through those contracts. Preserve all old output directories and put v3 artifacts under isolated `interaction_chunk_*` paths.

**Tech Stack:** Python 3.12, PyTorch/MPS training, deterministic CPU rollout, MuJoCo 3.x contact physics, NumPy, PyYAML, tqdm, pytest, Pillow.

---

## Execution constraints

- The workspace is not a Git repository. Do not initialize Git and do not add
  synthetic commit steps. Each task ends with a focused verification checkpoint.
- Use `apply_patch` for source, test, config, and documentation edits.
- Follow TDD for every behaviour change: add one failing test, observe the expected
  failure, implement the minimum contract, then rerun the focused tests.
- Do not run the 200-episode pilot automatically. Run only the isolated smoke
  chain; hand the full pilot commands to the user.
- Before editing, record the baseline:

```bash
.venv/bin/python -m pytest -q
```

Expected baseline at plan creation: `261 passed, 4 skipped`.

## File structure

**Create:**

- `interaction_vla/placement.py` — receptacle geometry, projected containment,
  base/wall contact classification, and strict placement diagnostics.
- `interaction_vla/source_split.py` — deterministic source-seed split, recovery
  source selection, manifest loading, disjointness validation, and split hash.
- `interaction_vla/sequence_training.py` — H=8 window dataset, phase-stratified
  75/25 batch sampler, masked sequence loss, and sequence MSE.
- `interaction_vla/chunked_controller.py` — shared CPU learned-policy controller,
  overlapping-chunk aggregation, gripper hysteresis, and IK projection.
- `configs/physics_interaction_chunk_smoke_macos.yaml` — 10-source, one-epoch
  current-provenance smoke.
- `configs/physics_interaction_chunk_pilot_macos.yaml` — 200-source, 80-epoch,
  seed-0 pilot.
- `tests/interaction_vla/test_placement.py`
- `tests/interaction_vla/test_source_split.py`
- `tests/interaction_vla/test_sequence_training.py`
- `tests/interaction_vla/test_chunked_controller.py`

**Modify:**

- `interaction_vla/config.py`
- `interaction_vla/contact_physics.py`
- `interaction_vla/physics_env.py`
- `interaction_vla/physics_data.py`
- `interaction_vla/models/policy.py`
- `interaction_vla/train.py`
- `interaction_vla/physics_provenance.py`
- `interaction_vla/physics_evaluate.py`
- `interaction_vla/physics_visualize.py`
- `interaction_vla/validate_physics_expert.py`
- `README.md`
- focused tests named in each task.

---

### Task 1: Add an explicit v3 configuration contract

**Files:**

- Modify: `interaction_vla/config.py:28-294`
- Create: `configs/physics_interaction_chunk_smoke_macos.yaml`
- Create: `configs/physics_interaction_chunk_pilot_macos.yaml`
- Modify: `tests/interaction_vla/test_config.py`

- [ ] **Step 1: Write failing configuration validation tests**

Add tests that define all fixed, shared temporal parameters and reject a non-CPU
v3 rollout:

```python
from dataclasses import replace

import pytest

from interaction_vla.config import SequenceConfig, load_config


def test_sequence_config_encodes_the_shared_h8_controller() -> None:
    config = SequenceConfig(
        enabled=True,
        horizon=8,
        future_loss_decay=0.9,
        temporal_decay=0.25,
        gripper_close_threshold=0.35,
        gripper_open_threshold=0.65,
        recovery_loss_fraction=0.25,
        rollout_device="cpu",
    )
    assert config.horizon == 8
    assert config.recovery_loss_fraction == 0.25
    assert config.rollout_device == "cpu"


@pytest.mark.parametrize(
    "changes",
    [
        {"horizon": 0},
        {"future_loss_decay": 0.0},
        {"future_loss_decay": 1.1},
        {"temporal_decay": -0.1},
        {"gripper_close_threshold": 0.7, "gripper_open_threshold": 0.6},
        {"recovery_loss_fraction": 1.0},
        {"rollout_device": "mps"},
    ],
)
def test_enabled_sequence_config_rejects_unfair_or_invalid_values(changes) -> None:
    base = SequenceConfig(enabled=True, horizon=8, recovery_loss_fraction=0.25)
    with pytest.raises(ValueError):
        replace(base, **changes)


def test_interaction_chunk_configs_are_isolated_and_fair() -> None:
    smoke = load_config("configs/physics_interaction_chunk_smoke_macos.yaml")
    pilot = load_config("configs/physics_interaction_chunk_pilot_macos.yaml")
    assert smoke.train.episodes == 10
    assert smoke.train.batch_size == 8
    assert pilot.train.episodes == 200
    assert pilot.train.batch_size == 64
    assert pilot.train.model_seeds == (0,)
    assert pilot.sequence.horizon == 8
    assert pilot.sequence.recovery_loss_fraction == 0.25
    assert pilot.recovery.training_source_fraction == 0.25
    assert pilot.recovery.benchmark_enabled is True
    assert "interaction_chunk_pilot" in pilot.output_dir
    assert "terminal_recovery_pilot" not in pilot.output_dir
```

- [ ] **Step 2: Run the new tests and observe RED**

```bash
.venv/bin/python -m pytest \
  tests/interaction_vla/test_config.py::test_sequence_config_encodes_the_shared_h8_controller \
  tests/interaction_vla/test_config.py::test_interaction_chunk_configs_are_isolated_and_fair -q
```

Expected: import/config-file failures because `SequenceConfig` and the two v3
YAML files do not exist.

- [ ] **Step 3: Implement the typed sequence and recovery configuration**

Add these dataclass fields and validation to `interaction_vla/config.py`:

```python
@dataclass(frozen=True)
class SequenceConfig:
    enabled: bool = False
    horizon: int = 1
    future_loss_decay: float = 0.9
    temporal_decay: float = 0.25
    gripper_close_threshold: float = 0.35
    gripper_open_threshold: float = 0.65
    recovery_loss_fraction: float = 0.0
    rollout_device: str = "cpu"

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError("sequence.horizon must be positive")
        if not 0.0 < self.future_loss_decay <= 1.0:
            raise ValueError("sequence.future_loss_decay must be within (0, 1]")
        if not math.isfinite(self.temporal_decay) or self.temporal_decay < 0.0:
            raise ValueError("sequence.temporal_decay must be finite and non-negative")
        if not (
            0.0 <= self.gripper_close_threshold
            < self.gripper_open_threshold
            <= 1.0
        ):
            raise ValueError("sequence gripper thresholds must satisfy 0 <= close < open <= 1")
        if not 0.0 <= self.recovery_loss_fraction < 1.0:
            raise ValueError("sequence.recovery_loss_fraction must be within [0, 1)")
        if self.rollout_device not in {"cpu", "mps"}:
            raise ValueError("sequence.rollout_device must be cpu or mps")
        if self.enabled and self.rollout_device != "cpu":
            raise ValueError("enabled sequence experiments require rollout_device=cpu")


@dataclass(frozen=True)
class RecoveryConfig:
    enabled: bool = False
    variants_per_episode: int = 0
    min_acceptance_rate: float = 0.0
    training_source_fraction: float = 1.0
    benchmark_enabled: bool = False

    def __post_init__(self) -> None:
        if not 0.0 < self.training_source_fraction <= 1.0:
            raise ValueError("recovery.training_source_fraction must be within (0, 1]")
```

Add `sequence: SequenceConfig = field(default_factory=SequenceConfig)` to
`ExperimentConfig`, parse a top-level `sequence:` YAML mapping in `load_config`,
and enforce these cross-field rules:

```python
if self.sequence.enabled and self.model.action_dim != 7:
    raise ValueError("sequence control currently requires physical 7D actions")
if self.sequence.enabled and not self.recovery.enabled:
    raise ValueError("sequence v3 requires configured recovery augmentation")
if self.sequence.enabled and self.sequence.recovery_loss_fraction == 0.0:
    raise ValueError("sequence v3 requires a positive recovery loss fraction")
```

- [ ] **Step 4: Add exact smoke and pilot YAML files**

Use the terminal-recovery physics settings unchanged. The smoke-specific fields
must be:

```yaml
name: interaction_graph_physics_interaction_chunk_smoke
seed: 42
device: auto
backend: franka_contact
max_objects: 5
data_dir: outputs/interaction_graph_physics/interaction_chunk_smoke/data
output_dir: outputs/interaction_graph_physics/interaction_chunk_smoke
train:
  object_counts: [2, 3]
  episodes: 10
  batch_size: 8
  epochs: 1
  learning_rate: 0.003
  model_seeds: [0]
recovery:
  enabled: true
  variants_per_episode: 4
  min_acceptance_rate: 0.5
  training_source_fraction: 0.25
  benchmark_enabled: true
sequence:
  enabled: true
  horizon: 8
  future_loss_decay: 0.9
  temporal_decay: 0.25
  gripper_close_threshold: 0.35
  gripper_open_threshold: 0.65
  recovery_loss_fraction: 0.25
  rollout_device: cpu
```

Copy the remaining `eval`, `environment`, `physics`, and `recording` sections
from `configs/physics_terminal_recovery_smoke_macos.yaml` without changing the
scene or controller. Use a 16-wide model for smoke.

The pilot-specific fields must be:

```yaml
name: interaction_graph_physics_interaction_chunk_pilot
seed: 42
device: auto
backend: franka_contact
max_objects: 5
data_dir: outputs/interaction_graph_physics/interaction_chunk_pilot/data
output_dir: outputs/interaction_graph_physics/interaction_chunk_pilot
train:
  object_counts: [2, 3]
  episodes: 200
  batch_size: 64
  epochs: 80
  learning_rate: 0.001
  model_seeds: [0]
recovery:
  enabled: true
  variants_per_episode: 4
  min_acceptance_rate: 0.8
  training_source_fraction: 0.25
  benchmark_enabled: true
sequence:
  enabled: true
  horizon: 8
  future_loss_decay: 0.9
  temporal_decay: 0.25
  gripper_close_threshold: 0.35
  gripper_open_threshold: 0.65
  recovery_loss_fraction: 0.25
  rollout_device: cpu
```

Copy the remaining sections from
`configs/physics_terminal_recovery_pilot_macos.yaml`, retaining the 64-wide,
two-message-round model and 180-step environment/evaluation horizons.

- [ ] **Step 5: Run configuration tests GREEN**

```bash
.venv/bin/python -m pytest tests/interaction_vla/test_config.py -q
```

Expected: all configuration tests pass, including legacy configs that default to
disabled H=1 behaviour.

---

### Task 2: Replace any-receptacle contact with strict physical placement

**Files:**

- Create: `interaction_vla/placement.py`
- Modify: `interaction_vla/contact_physics.py:13-233`
- Modify: `interaction_vla/physics_env.py:35-640`
- Create: `tests/interaction_vla/test_placement.py`
- Modify: `tests/interaction_vla/test_contact_physics.py`
- Modify: `tests/interaction_vla/test_physics_env.py`
- Modify: `tests/interaction_vla/test_validate_physics_expert.py`

- [ ] **Step 1: Write pure geometry tests for containment**

Create `tests/interaction_vla/test_placement.py` with axis-aligned, rotated, and
exterior cases:

```python
import numpy as np

from interaction_vla.placement import strict_containment


def test_strict_containment_requires_the_complete_object_inside() -> None:
    rotation = np.eye(3)
    inside = strict_containment(
        local_center=np.asarray((0.030, 0.0, 0.028)),
        relative_rotation=rotation,
        object_half_extents=np.asarray((0.022, 0.022, 0.022)),
        inner_half_extents=np.asarray((0.055, 0.055)),
    )
    outside = strict_containment(
        local_center=np.asarray((0.040, 0.0, 0.028)),
        relative_rotation=rotation,
        object_half_extents=np.asarray((0.022, 0.022, 0.022)),
        inner_half_extents=np.asarray((0.055, 0.055)),
    )
    assert inside.fully_contained
    assert inside.containment_margin.min() == pytest.approx(0.003)
    assert not outside.fully_contained
    assert outside.containment_margin[0] < 0.0


def test_rotated_object_uses_projected_half_extents() -> None:
    angle = np.deg2rad(45.0)
    rotation = np.asarray(
        (
            (np.cos(angle), -np.sin(angle), 0.0),
            (np.sin(angle), np.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    result = strict_containment(
        local_center=np.asarray((0.025, 0.0, 0.028)),
        relative_rotation=rotation,
        object_half_extents=np.asarray((0.022, 0.022, 0.022)),
        inner_half_extents=np.asarray((0.055, 0.055)),
    )
    assert result.projected_half_extents[0] == pytest.approx(0.0311127)
    assert not result.fully_contained
```

Import `pytest` in the new file.

- [ ] **Step 2: Run geometry tests RED**

```bash
.venv/bin/python -m pytest tests/interaction_vla/test_placement.py -q
```

Expected: module import failure because `interaction_vla.placement` does not
exist.

- [ ] **Step 3: Implement strict projected containment**

Create `interaction_vla/placement.py` with these public types and function:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ContainmentResult:
    local_center: np.ndarray
    projected_half_extents: np.ndarray
    containment_margin: np.ndarray
    fully_contained: bool


def strict_containment(
    *,
    local_center: np.ndarray,
    relative_rotation: np.ndarray,
    object_half_extents: np.ndarray,
    inner_half_extents: np.ndarray,
) -> ContainmentResult:
    center = np.asarray(local_center, dtype=np.float64)
    rotation = np.asarray(relative_rotation, dtype=np.float64)
    object_half = np.asarray(object_half_extents, dtype=np.float64)
    inner_half = np.asarray(inner_half_extents, dtype=np.float64)
    if center.shape != (3,) or rotation.shape != (3, 3):
        raise ValueError("placement pose must be a 3D centre and 3x3 rotation")
    if object_half.shape != (3,) or inner_half.shape != (2,):
        raise ValueError("placement extents must have shapes (3,) and (2,)")
    if not all(np.isfinite(value).all() for value in (center, rotation, object_half, inner_half)):
        raise ValueError("placement geometry must be finite")
    if np.any(object_half <= 0.0) or np.any(inner_half <= 0.0):
        raise ValueError("placement extents must be positive")
    projected = np.abs(rotation[:2]) @ object_half
    margin = inner_half - np.abs(center[:2]) - projected
    return ContainmentResult(
        local_center=center.copy(),
        projected_half_extents=projected,
        containment_margin=margin,
        fully_contained=bool(np.all(margin >= 0.0)),
    )
```

- [ ] **Step 4: Write failing contact classification tests**

Extend `ContactDiagnostics` fixtures and add a MuJoCo test that places one object
on the receptacle base and another against an exterior wall. Assert:

```python
assert "object_0" in diagnostics.object_receptacle_base
assert "object_0" not in diagnostics.object_receptacle_wall
assert "object_1" in diagnostics.object_receptacle_wall
assert diagnostics.object_receptacle == frozenset(("object_0", "object_1"))
```

Expected raw labels are `receptacle_base` and `receptacle_wall`; the existing
`object_receptacle` union remains for graph-schema compatibility.

- [ ] **Step 5: Run contact tests RED**

```bash
.venv/bin/python -m pytest \
  tests/interaction_vla/test_contact_physics.py \
  tests/interaction_vla/test_placement.py -q
```

Expected: `ContactDiagnostics` lacks base/wall fields.

- [ ] **Step 6: Implement base/wall contact classification**

Extend the dataclass without removing the old union:

```python
@dataclass(frozen=True)
class ContactDiagnostics:
    left_objects: frozenset[str]
    right_objects: frozenset[str]
    object_table: frozenset[str]
    object_receptacle: frozenset[str]
    interactions: tuple[InteractionSignal, ...]
    object_receptacle_base: frozenset[str] = frozenset()
    object_receptacle_wall: frozenset[str] = frozenset()
```

Append the two new defaulted fields after the existing non-default
`interactions` field so all current five-position constructor calls retain their
meaning.

In `ContactParser.parse`, track base and wall sets separately. Change semantic
labelling before the descendant fallback:

```python
if geom_name == "receptacle_base":
    return "receptacle_base"
if geom_name.startswith("receptacle_wall_"):
    return "receptacle_wall"
if self._is_descendant(body_id, self.receptacle_body_id):
    return "receptacle"
```

When an object contacts `receptacle_base` or `receptacle_wall`, add it to the
specific set and the compatibility union. Accumulate the graph interaction under
the existing semantic endpoint `receptacle`, so Flat and Graph input dimensions
remain unchanged.

- [ ] **Step 7: Write failing strict environment success tests**

Add in-memory MuJoCo setup helpers to `test_physics_env.py` and cover both sides
of the success contract:

```python
def test_exterior_wall_contact_never_accumulates_strict_placement() -> None:
    env = make_env()
    env.reset(seed=11, object_count=2, target_index=0)
    set_free_object_pose(env, "object_0", position=(0.583, -0.12, 0.273))
    advance_zero_actions(env, count=12, gripper_open=True)
    assert env.contact_diagnostics.object_receptacle_wall == frozenset(("object_0",))
    assert env.last_placement.fully_contained is False
    assert env.last_placement.strict_stable is False
    assert env._placement_frames == 0


def test_base_contact_full_containment_can_reach_strict_success() -> None:
    env = make_env()
    env.reset(seed=11, object_count=2, target_index=0)
    set_free_object_pose(env, "object_0", position=(0.67, -0.12, 0.273))
    place_tcp_above_target(env, height=0.10, gripper_open=True)
    result = advance_zero_actions(env, count=12, gripper_open=True)
    assert env.last_placement.fully_contained
    assert env.last_placement.base_contact
    assert result.reason is TerminationReason.SUCCESS
```

The test helpers may change qpos only during fixture setup, before rollout; keep
the existing no-attachment rollout audit unchanged.

- [ ] **Step 8: Implement strict placement tracking in the environment**

Add `PlacementDiagnostics` and a model-derived geometry factory to
`placement.py`:

```python
@dataclass(frozen=True)
class PlacementDiagnostics:
    local_center: np.ndarray
    projected_half_extents: np.ndarray
    containment_margin: np.ndarray
    fully_contained: bool
    base_contact: bool
    wall_contact: bool
    wall_only_contact: bool
    stable_frames: int
    strict_stable: bool
```

At environment initialization derive inner X/Y faces from wall geom local
positions and half-sizes, yielding `(0.055, 0.055)` for the current box. At each
physics substep:

1. transform target pose into receptacle local coordinates;
2. call `strict_containment` with the target geom half-size;
3. require base contact, full containment, linear speed `< 0.05`, and angular
   speed `< 0.05` before incrementing `_placement_frames`;
4. reset the counter otherwise;
5. expose `last_placement` and include these exact keys in `_info`:

```python
info.update(
    {
        "strict_placement": self.last_placement.strict_stable,
        "stable_placement": self.last_placement.strict_stable,
        "wall_only_receptacle_contact": self.last_placement.wall_only_contact,
        "containment_margin_x": float(self.last_placement.containment_margin[0]),
        "containment_margin_y": float(self.last_placement.containment_margin[1]),
        "target_receptacle_local_position": self.last_placement.local_center.copy(),
    }
)
```

Retain open gripper and TCP-above-target checks in `_is_success`, but require
`last_placement.strict_stable` instead of any receptacle contact.

- [ ] **Step 9: Run strict placement and expert validation tests GREEN**

```bash
.venv/bin/python -m pytest \
  tests/interaction_vla/test_placement.py \
  tests/interaction_vla/test_contact_physics.py \
  tests/interaction_vla/test_physics_env.py \
  tests/interaction_vla/test_validate_physics_expert.py -q
```

Expected: strict geometry tests pass; wall-only contact cannot produce success;
the expert gate tests use the new success predicate.

---

### Task 3: Make source seed the only split unit

**Files:**

- Create: `interaction_vla/source_split.py`
- Modify: `interaction_vla/physics_data.py:455-780`
- Create: `tests/interaction_vla/test_source_split.py`
- Modify: `tests/interaction_vla/test_physics_data.py`

- [ ] **Step 1: Write failing exact split and inheritance tests**

Create `tests/interaction_vla/test_source_split.py`:

```python
import pytest

from interaction_vla.source_split import (
    SourceSplit,
    deterministic_source_split,
    select_training_recovery_sources,
    validate_derived_sources,
)


def test_two_hundred_sources_split_exactly_160_20_20() -> None:
    seeds = tuple(range(10_000, 10_200))
    first = deterministic_source_split(seeds, seed=42)
    second = deterministic_source_split(reversed(seeds), seed=42)
    assert first == second
    assert len(first.train) == 160
    assert len(first.validation) == 20
    assert len(first.test) == 20
    assert set(first.train).isdisjoint(first.validation)
    assert set(first.train).isdisjoint(first.test)
    assert set(first.validation).isdisjoint(first.test)


def test_training_recovery_uses_exactly_twenty_five_percent_of_train_sources() -> None:
    split = deterministic_source_split(range(200), seed=42)
    selected = select_training_recovery_sources(
        split.train,
        fraction=0.25,
        seed=42,
    )
    assert len(selected) == 40
    assert set(selected).issubset(split.train)


def test_derived_artifact_cannot_cross_its_source_split() -> None:
    split = SourceSplit(train=(1,), validation=(2,), test=(3,))
    validate_derived_sources(
        split,
        training_recovery_sources=(1,),
        benchmark_sources=(2, 3),
    )
    with pytest.raises(ValueError, match="training recovery"):
        validate_derived_sources(
            split,
            training_recovery_sources=(2,),
            benchmark_sources=(3,),
        )
```

- [ ] **Step 2: Run source split tests RED**

```bash
.venv/bin/python -m pytest tests/interaction_vla/test_source_split.py -q
```

Expected: module import failure.

- [ ] **Step 3: Implement the canonical source split module**

Create `interaction_vla/source_split.py` with a canonical sorted representation:

```python
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SourceSplit:
    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]

    def __post_init__(self) -> None:
        groups = tuple(set(group) for group in (self.train, self.validation, self.test))
        if any(len(group) == 0 for group in groups):
            raise ValueError("source split groups must not be empty")
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise ValueError("source split groups must be disjoint")

    def payload(self) -> dict[str, list[int]]:
        return {
            "train": list(self.train),
            "validation": list(self.validation),
            "test": list(self.test),
        }


def deterministic_source_split(seeds: Iterable[int], *, seed: int) -> SourceSplit:
    values = tuple(sorted(int(value) for value in seeds))
    if len(set(values)) != len(values):
        raise ValueError("source seeds must be unique")
    if len(values) < 10:
        raise ValueError("source split requires at least ten successful sources")
    rng = np.random.default_rng(np.random.SeedSequence((seed, 0x53504C54)))
    shuffled = np.asarray(values, dtype=np.int64)[rng.permutation(len(values))]
    validation_count = int(round(len(values) * 0.10))
    test_count = int(round(len(values) * 0.10))
    validation = tuple(sorted(int(value) for value in shuffled[:validation_count]))
    test = tuple(
        sorted(int(value) for value in shuffled[validation_count:validation_count + test_count])
    )
    train = tuple(sorted(int(value) for value in shuffled[validation_count + test_count:]))
    return SourceSplit(train=train, validation=validation, test=test)


def select_training_recovery_sources(
    train_sources: Iterable[int], *, fraction: float, seed: int
) -> tuple[int, ...]:
    values = tuple(sorted(int(value) for value in train_sources))
    count = int(round(len(values) * fraction))
    if count < 1:
        raise ValueError("training recovery source selection is empty")
    rng = np.random.default_rng(np.random.SeedSequence((seed, 0x52454353)))
    selected = np.asarray(values, dtype=np.int64)[rng.permutation(len(values))[:count]]
    return tuple(sorted(int(value) for value in selected))
```

Also implement `validate_derived_sources`, atomic `save_source_split`,
`load_source_split`, and `source_split_hash`. The saved JSON must contain
`train`, `validation`, `test`, `training_recovery_sources`, and
`benchmark_sources`; hash canonical JSON with sorted keys and compact separators.

- [ ] **Step 4: Write failing physical collector manifest tests**

Extend the existing fake-episode collector test to collect 10 successful smoke
sources and assert:

```python
source_payload = json.loads((data_dir / "source_split.json").read_text())
assert len(source_payload["train"]) == 8
assert len(source_payload["validation"]) == 1
assert len(source_payload["test"]) == 1
assert len(source_payload["training_recovery_sources"]) == 2

training_records = json.loads((data_dir / "recovery_manifest.json").read_text())
benchmark_records = json.loads(
    (data_dir / "recovery_benchmark_manifest.json").read_text()
)
assert {record["source_seed"] for record in training_records}.issubset(
    source_payload["training_recovery_sources"]
)
assert {record["source_seed"] for record in benchmark_records}.issubset(
    source_payload["validation"] + source_payload["test"]
)
assert all(record["source_split"] == "train" for record in training_records)
assert {record["source_split"] for record in benchmark_records} == {
    "validation",
    "test",
}
```

Use deterministic fake successes for every recovery attempt so the assertion is
about source routing, not physics acceptance.

- [ ] **Step 5: Run collector manifest test RED**

```bash
.venv/bin/python -m pytest \
  tests/interaction_vla/test_physics_data.py::test_v3_collection_routes_recovery_by_source_split -q
```

Expected: `source_split.json` and benchmark manifest are absent.

- [ ] **Step 6: Refactor physical recovery collection into two explicit groups**

After all successful base episodes are collected, compute and atomically save the
source split before attempting recovery:

```python
split = deterministic_source_split(
    (int(record["seed"]) for record in records),
    seed=config.seed,
)
training_recovery_sources = select_training_recovery_sources(
    split.train,
    fraction=config.recovery.training_source_fraction,
    seed=config.seed,
)
benchmark_sources = tuple(sorted(split.validation + split.test))
validate_derived_sources(
    split,
    training_recovery_sources=training_recovery_sources,
    benchmark_sources=benchmark_sources,
)
save_source_split(
    data_dir / "source_split.json",
    split,
    training_recovery_sources=training_recovery_sources,
    benchmark_sources=benchmark_sources,
)
```

Then add `source_seed` and `source_split` to every in-memory base manifest record
and atomically rewrite `manifest.json`. Do not recompress or mutate the already
saved base NPZs: their existing `metadata.seed` is the source seed and the two
canonical JSON manifests supply split membership. Recovery NPZ metadata is
created after the split is known and must include both `source_seed` and
`source_split`.

Extract the repeated recovery loop into `_collect_recovery_group` with this exact
interface:

```python
def _collect_recovery_group(
    *,
    config,
    records_by_seed: Mapping[int, Mapping[str, object]],
    source_seeds: tuple[int, ...],
    source_split_by_seed: Mapping[int, str],
    gate_hash: str,
    manifest_path: Path,
    rejection_path: Path,
    quality_path: Path,
    filename_prefix: str,
    show_progress: bool,
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
```

Call it once for selected train sources using `recovery_manifest.json` and once
for validation/test sources using `recovery_benchmark_manifest.json`. Save
accepted NPZs with prefixes `train_recovery_` and `benchmark_recovery_`.
Every record contains `source_seed`, `source_split`, `variant_id`, `kind`,
`object_count`, `frames`, `reason`, and `path` when accepted. Write separate
`recovery_quality.json` and `recovery_benchmark_quality.json` files and enforce
the existing per-kind quality gate for both groups.

- [ ] **Step 7: Run source split and collector tests GREEN**

```bash
.venv/bin/python -m pytest \
  tests/interaction_vla/test_source_split.py \
  tests/interaction_vla/test_physics_data.py -q
```

Expected: all source routing, progress, quality, and legacy collection tests pass.

---

### Task 4: Make saved source manifests authoritative during training

**Files:**

- Modify: `interaction_vla/source_split.py`
- Modify: `interaction_vla/train.py`
- Modify: `tests/interaction_vla/test_source_split.py`
- Modify: `tests/interaction_vla/test_train.py`

- [ ] **Step 1: Write failing tests for manifest-only data selection**

Create physical episode fixtures whose filenames are deliberately shuffled. Base
NPZ metadata contains its existing `seed`; base manifest records contain
`source_seed` and `source_split`; recovery NPZ metadata and records contain both.
Assert that
`resolve_training_data` uses `source_split.json`, `recovery_manifest.json`, and
`recovery_benchmark_manifest.json` instead of recomputing a trajectory split:

```python
def test_training_selection_obeys_saved_source_manifests(tmp_path: Path) -> None:
    layout = write_source_manifest_fixture(tmp_path, source_count=10)
    selection = resolve_training_data(layout.data_dir, include_recovery=True)
    assert source_seeds(selection.base_train_paths) == set(layout.split.train)
    assert source_seeds(selection.validation_paths) == set(layout.split.validation)
    assert source_seeds(selection.test_paths) == set(layout.split.test)
    assert source_seeds(selection.recovery_train_paths) == set(
        layout.training_recovery_sources
    )
    assert not set(selection.recovery_train_paths) & set(
        selection.recovery_benchmark_paths
    )


def test_training_selection_rejects_split_metadata_mismatch(tmp_path: Path) -> None:
    layout = write_source_manifest_fixture(tmp_path, source_count=10)
    rewrite_base_manifest_source_split(layout.manifest_path, row=0, value="test")
    with pytest.raises(ValueError, match="source split mismatch"):
        resolve_training_data(layout.data_dir, include_recovery=True)
```

Add a separate statistics test that makes validation, test, and recovery actions
large enough to detect leakage:

```python
def test_statistics_are_fit_from_training_base_episodes_only(tmp_path: Path) -> None:
    layout = write_statistics_leakage_fixture(tmp_path)
    selection = resolve_training_data(layout.data_dir, include_recovery=True)
    statistics = TrainingStatistics.fit(selection.base_train_paths)
    np.testing.assert_allclose(statistics.action_mean, np.zeros(7), atol=1e-6)
```

- [ ] **Step 2: Run the manifest and statistics tests RED**

```bash
.venv/bin/python -m pytest \
  tests/interaction_vla/test_train.py::test_training_selection_obeys_saved_source_manifests \
  tests/interaction_vla/test_train.py::test_statistics_are_fit_from_training_base_episodes_only -q
```

Expected: the current trainer regenerates its own split and fits statistics from
the combined path list.

- [ ] **Step 3: Add strict manifest loaders and validation**

In `source_split.py`, implement typed manifest loading and this validation
contract:

```python
@dataclass(frozen=True)
class SourceDataLayout:
    split: SourceSplit
    base_by_split: Mapping[str, tuple[Path, ...]]
    training_recovery_paths: tuple[Path, ...]
    benchmark_recovery_paths: tuple[Path, ...]
    training_recovery_sources: tuple[int, ...]
    benchmark_sources: tuple[int, ...]
    source_split_hash: str
    training_recovery_manifest_hash: str
    benchmark_recovery_manifest_hash: str


def load_source_data_layout(data_dir: Path) -> SourceDataLayout:
```

Resolve paths relative to `data_dir` and require every file to exist. For base
episodes, require NPZ `metadata.seed` to equal manifest `source_seed` and manifest
`source_split` to equal `source_split.json`. For recovery episodes, additionally
require NPZ recovery metadata to match the record's source seed, split, variant,
and kind. Reject duplicate source membership, a missing source, an unknown
source, train recovery derived from validation/test, or benchmark recovery
derived from train. Hash canonical JSON content rather than filesystem
modification times.

- [ ] **Step 4: Replace implicit splitting in `resolve_training_data`**

Change `TrainingDataSelection` to make every group explicit:

```python
@dataclass(frozen=True)
class TrainingDataSelection:
    base_train_paths: tuple[Path, ...]
    recovery_train_paths: tuple[Path, ...]
    validation_paths: tuple[Path, ...]
    test_paths: tuple[Path, ...]
    recovery_benchmark_paths: tuple[Path, ...]
    source_split_hash: str
    recovery_manifest_hash: str
    recovery_benchmark_manifest_hash: str
```

`resolve_training_data` must call only `load_source_data_layout`; remove its call
to random `split_episode_seeds`. Preserve a legacy path only when
`config.sequence.enabled` is false. For v3, missing manifests are a hard error.
Fit `TrainingStatistics` from `base_train_paths` only, then apply those frozen
statistics to base, recovery, validation, and test samples.

- [ ] **Step 5: Run focused training selection tests GREEN**

```bash
.venv/bin/python -m pytest \
  tests/interaction_vla/test_source_split.py \
  tests/interaction_vla/test_train.py -q
```

Expected: all source membership, no-leakage, statistics, and legacy trainer tests
pass.

---

### Task 5: Build H=8 windows and an exact 75/25 training objective

**Files:**

- Create: `interaction_vla/sequence_training.py`
- Create: `tests/interaction_vla/test_sequence_training.py`

- [ ] **Step 1: Write failing tests for episode-bounded action chunks**

Use a five-frame episode with actions equal to their frame index. Assert exact
right padding and masks at the last two starts:

```python
def test_sequence_windows_never_cross_episode_boundaries(tmp_path: Path) -> None:
    episode = write_numbered_episode(tmp_path / "episode.npz", frames=5)
    dataset = EpisodeSequenceDataset(
        base_paths=(episode,),
        recovery_paths=(),
        statistics=identity_statistics(action_dim=7),
        horizon=3,
    )
    penultimate = dataset[3]
    np.testing.assert_array_equal(penultimate["horizon_mask"], [True, True, False])
    np.testing.assert_allclose(penultimate["action_chunk"][:, 0], [3.0, 4.0, 0.0])
    final = dataset[4]
    np.testing.assert_array_equal(final["horizon_mask"], [True, False, False])
```

Also assert that the observation, graph, contact state, and proprioception always
come from the chunk's start frame and that each sample exposes `sample_group`
equal to `0` for base and `1` for recovery.

- [ ] **Step 2: Run the window test RED**

```bash
.venv/bin/python -m pytest \
  tests/interaction_vla/test_sequence_training.py::test_sequence_windows_never_cross_episode_boundaries -q
```

Expected: `interaction_vla.sequence_training` does not exist.

- [ ] **Step 3: Implement the sequence dataset**

Add an immutable window index and cache each episode once per dataset instance:

```python
@dataclass(frozen=True)
class SequenceWindow:
    episode_index: int
    start: int
    sample_group: int
    phase: str


class EpisodeSequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        *,
        base_paths: tuple[Path, ...],
        recovery_paths: tuple[Path, ...],
        statistics: TrainingStatistics,
        horizon: int,
    ) -> None:
```

Use `from __future__ import annotations` and import `TrainingStatistics` only
under `typing.TYPE_CHECKING`; `train.py` imports this new dataset, so a runtime
back-import from `sequence_training.py` would create a circular import.

Construct one window per real frame. Return normalized graph and flat inputs,
raw physical `action_chunk` shaped `[H, 7]`, boolean `horizon_mask` shaped `[H]`,
integer `sample_group`, `phase`, `source_seed`, `episode_path`, and `frame_index`.
Padding is zero in raw action space and is excluded by the mask. The trainer must
normalize both predictions and targets with the same train-base-only action
statistics immediately before loss computation.

- [ ] **Step 4: Write failing exact-mixture sampler tests**

For pilot batch size 64 assert every yielded batch has exactly 48 base and 16
recovery indices. For smoke batch size 8 assert exactly 6 base and 2 recovery.
Assert phases are sampled uniformly inside both base and recovery groups,
sampling is deterministic for the same model seed and epoch, and `set_epoch(1)`
changes the order:

```python
@pytest.mark.parametrize("batch_size,expected", [(64, (48, 16)), (8, (6, 2))])
def test_stratified_sampler_has_exact_recovery_mass(batch_size, expected) -> None:
    sampler = StratifiedSequenceBatchSampler(
        base_indices_by_phase={
            "approach": tuple(range(0, 40)),
            "grasp": tuple(range(40, 80)),
            "transport": tuple(range(80, 120)),
        },
        recovery_indices_by_phase={
            "lift": (120, 121),
            "transport": (122, 123),
            "retreat": (124, 125),
        },
        batch_size=batch_size,
        recovery_fraction=0.25,
        seed=7,
    )
    for batch in sampler:
        groups = [index >= 120 for index in batch]
        assert groups.count(False) == expected[0]
        assert groups.count(True) == expected[1]
```

- [ ] **Step 5: Implement deterministic phase-stratified batches**

`StratifiedSequenceBatchSampler` computes group counts exactly and rejects a
fraction that cannot be represented as integer counts for the chosen batch size.
Set the epoch length to `ceil(total_base_windows / base_per_batch)`. For each
group, divide its per-batch quota across non-empty phases so phase counts differ
by at most one; rotate the remainder phase across batches and epochs. Sample
windows uniformly inside each phase with deterministic seeded cycles, reshuffling
before reuse. This applies to base and recovery independently and keeps scarce
phases from changing the required 75/25 group mass. The test must also verify
that cumulative phase counts over a complete sampler epoch differ by at most one
when the group quota permits it.

- [ ] **Step 6: Write failing masked group-loss tests**

Create a batch where base per-sample sequence MSE is `1.0`, recovery is `4.0`,
and later horizon slots are masked. Assert the objective is computed per sample,
then per group:

```python
loss = sequence_behavior_cloning_loss(
    prediction,
    target,
    horizon_mask,
    sample_group,
    future_loss_decay=0.9,
    recovery_loss_fraction=0.25,
)
assert loss.base.item() == pytest.approx(1.0)
assert loss.recovery.item() == pytest.approx(4.0)
assert loss.total.item() == pytest.approx(0.75 * 1.0 + 0.25 * 4.0)
```

Assert duplicating a long recovery trajectory does not change its total group
coefficient, and assert a batch missing either required group raises an error.

- [ ] **Step 7: Implement the masked sequence loss**

For horizon weights `w_k = future_loss_decay ** k`, first average squared error
over the seven action dimensions, then normalize each sample by the sum of its
valid horizon weights. Compute base and recovery sample means separately and
return:

```python
@dataclass(frozen=True)
class SequenceLoss:
    total: torch.Tensor
    base: torch.Tensor
    recovery: torch.Tensor


total = (1.0 - recovery_loss_fraction) * base_mean
total = total + recovery_loss_fraction * recovery_mean
```

- [ ] **Step 8: Run the complete sequence-training module GREEN**

```bash
.venv/bin/python -m pytest tests/interaction_vla/test_sequence_training.py -q
```

Expected: window boundary, exact mixture, deterministic phase balancing, and
masked-loss tests pass.

---

### Task 6: Give both representations the same H=8 prediction head

**Files:**

- Modify: `interaction_vla/models/policy.py`
- Modify: `tests/interaction_vla/test_policy.py`

- [ ] **Step 1: Write failing shape and parity tests**

Add tests for a public chunk API and legacy H=1 behaviour:

```python
def test_action_policy_predicts_a_bounded_h8_chunk() -> None:
    policy = build_action_policy(
        representation="graph",
        max_nodes=8,
        policy_hidden_dim=32,
        action_dim=7,
        action_mode="cartesian_7d",
        action_horizon=8,
    )
    scene, proprioception = example_policy_inputs(batch_size=2)
    chunk = policy.predict_action_chunk(scene, proprioception)
    assert chunk.shape == (2, 8, 7)
    assert torch.all(chunk.abs() <= 1.0)


def test_h1_forward_contract_is_backward_compatible() -> None:
    policy = build_action_policy(
        representation="flat",
        max_nodes=8,
        policy_hidden_dim=32,
        action_dim=7,
        action_mode="cartesian_7d",
        action_horizon=1,
    )
    scene, proprioception = example_policy_inputs(batch_size=2)
    assert policy(scene, proprioception).shape == (2, 7)
```

Build Flat and Graph with the same seed and assert they use the same temporal
head class, layer widths, activation, output transform, parameter count, and
bit-identical initial `action_head` and `proprio_encoder` state dictionaries.
The only allowed architecture difference is `policy.scene_encoder`. Retain the
existing `test_physics_flat_payload_contains_the_exact_same_graph_values` and
matched-encoder 10% parameter-budget tests in the full-suite gate.

- [ ] **Step 2: Run policy tests RED**

```bash
.venv/bin/python -m pytest \
  tests/interaction_vla/test_policy.py::test_action_policy_predicts_a_bounded_h8_chunk \
  tests/interaction_vla/test_policy.py::test_h1_forward_contract_is_backward_compatible -q
```

Expected: `action_horizon` and `predict_action_chunk` are unsupported.

- [ ] **Step 3: Implement the shared chunk head**

Add `action_horizon` to `ActionPolicy` and `build_action_policy`. Keep the same
encoder-specific forward path and change only the final shared head width:

```python
self.action_horizon = int(action_horizon)
self.action_dim = int(action_dim)
self.action_head = nn.Sequential(
    nn.Linear(2 * embedding_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, self.action_horizon * self.action_dim),
)

def predict_action_chunk(
    self,
    scene: SceneBatch | None,
    proprioception: Tensor,
) -> Tensor:
    proprio_context = self.proprio_encoder(proprioception)
    scene_context = self._encode_scene(scene, proprio_context)
    raw = self.action_head(torch.cat((scene_context, proprio_context), dim=-1))
    raw = raw.reshape(-1, self.action_horizon, self.action_dim)
    if self.action_mode == "legacy_cartesian_4d":
        xyz = torch.tanh(raw[:, :, :3]) * self.xyz_action_scale
        gripper = torch.sigmoid(raw[:, :, 3:4])
        return torch.cat((xyz, gripper), dim=-1)
    pose_command = torch.tanh(raw[:, :, :6])
    gripper = torch.sigmoid(raw[:, :, 6:7])
    return torch.cat((pose_command, gripper), dim=-1)

def forward(self, scene: SceneBatch | None, proprioception: Tensor) -> Tensor:
    chunk = self.predict_action_chunk(scene, proprioception)
    return chunk[:, 0] if self.action_horizon == 1 else chunk
```

Do not add an RNN, Transformer, representation-specific decoder, auxiliary loss,
or graph-only temporal feature. The temporal head is shared infrastructure, not
an experimental contribution.

- [ ] **Step 4: Run all policy tests GREEN**

```bash
.venv/bin/python -m pytest tests/interaction_vla/test_policy.py -q
```

Expected: H=8, H=1, encoder parameter matching, edge shuffle, and information
parity tests pass.

---

### Task 7: Integrate sequence sampling and loss into the trainer

**Files:**

- Modify: `interaction_vla/train.py`
- Modify: `tests/interaction_vla/test_train.py`

- [ ] **Step 1: Write a failing one-epoch v3 trainer test**

Train Flat and Graph on the same tiny deterministic fixture and inspect the
checkpoint and training summary:

```python
@pytest.mark.parametrize("representation", ["flat", "graph"])
def test_v3_trainer_uses_shared_h8_objective(tmp_path: Path, representation: str) -> None:
    config_path = write_tiny_v3_training_fixture(tmp_path)
    checkpoint_path = train_from_config(config_path, representation, model_seed=0)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["model_kwargs"]["action_horizon"] == 8
    assert checkpoint["temporal_contract"]["contribution"] == "shared_infrastructure"
    assert checkpoint["temporal_contract"]["recovery_loss_fraction"] == 0.25
    assert checkpoint["data_provenance"]["source_split_hash"]
    assert checkpoint["data_provenance"]["recovery_benchmark_manifest_hash"]
    summary = json.loads(checkpoint_path.with_name("training_summary.json").read_text())
    assert summary["effective_loss_mass"] == {"base": 0.75, "recovery": 0.25}
```

Add a paired parity test asserting Flat and Graph summaries have identical source
hashes, sample counts, steps per epoch, batch composition, sequence settings,
optimizer settings, model seed, and head specification.

- [ ] **Step 2: Run the trainer test RED**

```bash
.venv/bin/python -m pytest \
  tests/interaction_vla/test_train.py::test_v3_trainer_uses_shared_h8_objective -q
```

Expected: the current trainer builds one-step datasets and checkpoints.

- [ ] **Step 3: Add the v3 data loader without changing legacy training**

When `config.sequence.enabled` is true:

1. Load `TrainingDataSelection` from saved manifests.
2. Fit statistics using only `base_train_paths`.
3. Construct `EpisodeSequenceDataset` from train base and train recovery paths.
4. Group base and recovery dataset indices by their start-frame phase, then
   construct `StratifiedSequenceBatchSampler` using the configured batch size,
   exact recovery fraction, and model seed.
5. Build `DataLoader(dataset, batch_sampler=sampler)` with no second shuffle.

Keep the existing `EpisodeFrameDataset` and weighted one-step path unchanged for
non-v3 configs.

- [ ] **Step 4: Train with the masked group objective and epoch tqdm**

Call `sampler.set_epoch(epoch)` before each epoch and use
`policy.predict_action_chunk(scene, proprioception)`. Normalize the predicted and
target chunks with the same `TrainingStatistics`, then backpropagate
`SequenceLoss.total`. Keep the existing epoch-level `tqdm`, but show all three
losses and the exact composition:

```python
progress.set_postfix(
    mse=f"{epoch_total:.6f}",
    base=f"{epoch_base:.6f}",
    recovery=f"{epoch_recovery:.6f}",
    mix="48+16" if config.train.batch_size == 64 else "6+2",
)
```

Save per-epoch `total_mse`, `base_mse`, and `recovery_mse`; do not report the
weighted total as an unqualified global MSE.

- [ ] **Step 5: Save the fairness and temporal contracts in every checkpoint**

Add these immutable metadata blocks:

```python
checkpoint["representation_contract"] = {
    "experimental_variable": "encoder_only",
    "state_information": "identical",
    "temporal_head": "identical",
    "rollout_controller": "identical",
}
checkpoint["temporal_contract"] = {
    "contribution": "shared_infrastructure",
    "horizon": config.sequence.horizon,
    "future_loss_decay": config.sequence.future_loss_decay,
    "temporal_decay": config.sequence.temporal_decay,
    "gripper_close_threshold": config.sequence.gripper_close_threshold,
    "gripper_open_threshold": config.sequence.gripper_open_threshold,
    "recovery_loss_fraction": config.sequence.recovery_loss_fraction,
    "rollout_device": config.sequence.rollout_device,
}
```

Include all three manifest hashes, ordered source seed lists, base/recovery window
counts, phase counts, exact batch group counts, and the hash of train-base-only
statistics in `data_provenance`. Resume must reject a changed manifest, split,
temporal setting, or statistics hash.

- [ ] **Step 6: Run trainer and resume tests GREEN**

```bash
.venv/bin/python -m pytest tests/interaction_vla/test_train.py -q
```

Expected: v3 Flat/Graph parity, exact loss mixture, source provenance, resume
rejection, MPS selection tests, and all legacy training tests pass.

---

### Task 8: Add one shared chunked rollout controller

**Files:**

- Create: `interaction_vla/chunked_controller.py`
- Create: `tests/interaction_vla/test_chunked_controller.py`

- [ ] **Step 1: Write failing overlap-aggregation tests**

Use synthetic H=3 chunks issued at policy steps 0, 1, and 2. At step 2 the valid
predictions are indices `[2, 1, 0]` from those chunks. Assert the six Cartesian
dimensions use age weights `exp(-temporal_decay * age)`:

```python
def test_temporal_ensemble_uses_all_predictions_for_the_current_step() -> None:
    ensemble = TemporalActionEnsembler(
        horizon=3,
        temporal_decay=0.25,
        gripper_close_threshold=0.35,
        gripper_open_threshold=0.65,
    )
    ensemble.add(0, constant_chunk((1.0, 1.0, 1.0), gripper=1.0))
    ensemble.add(1, constant_chunk((2.0, 2.0, 2.0), gripper=1.0))
    ensemble.add(2, constant_chunk((3.0, 3.0, 3.0), gripper=1.0))
    action, diagnostics = ensemble.action_for_step(2)
    weights = np.exp(-0.25 * np.arange(3))
    expected = np.average((3.0, 2.0, 1.0), weights=weights)
    np.testing.assert_allclose(action[:6], expected)
    assert diagnostics.ensemble_size == 3
```

Add boundary tests proving expired chunks are discarded and H=1 exactly matches
the current prediction.

- [ ] **Step 2: Write failing gripper hysteresis tests**

Assert an initial open state closes only at `<= 0.35`, stays closed for values in
the deadband, and opens only at `>= 0.65`. Assert `reset(gripper_open=True)`
clears old chunks and restores the requested discrete state:

```python
def test_gripper_hysteresis_holds_state_inside_deadband() -> None:
    ensemble = make_ensemble(initially_open=True)
    assert ensemble.resolve_gripper(0.34) == 0.0
    assert ensemble.resolve_gripper(0.50) == 0.0
    assert ensemble.resolve_gripper(0.66) == 1.0
    assert ensemble.resolve_gripper(0.50) == 1.0
```

- [ ] **Step 3: Run temporal tests RED**

```bash
.venv/bin/python -m pytest tests/interaction_vla/test_chunked_controller.py -q
```

Expected: the controller module does not exist.

- [ ] **Step 4: Implement the pure temporal ensemble**

Add these public diagnostics and classes:

```python
@dataclass(frozen=True)
class ChunkControllerDiagnostics:
    ensemble_size: int
    raw_first_action: np.ndarray
    aggregated_action: np.ndarray
    raw_gripper_score: float
    gripper_command: float
    gripper_switch_count: int
    smoothing_delta_norm: float
    ik_projection_scale: float


class TemporalActionEnsembler:
    def add(self, issued_step: int, chunk: np.ndarray) -> None:
    def action_for_step(
        self, step: int
    ) -> tuple[np.ndarray, ChunkControllerDiagnostics]:
    def reset(self, *, gripper_open: bool = True) -> None:
```

Validate each chunk has shape `[H, 7]` and finite values. For current step `t`,
retain every issued chunk at `tau` satisfying `0 <= t - tau < H`, select row
`t - tau`, and assign weight `exp(-temporal_decay * (t - tau))`. Normalize the
weights and average only the first six dimensions. Average the gripper scores
with the same weights, then emit exactly `0.0` or `1.0` through hysteresis. Clip
Cartesian output to `[-1, 1]`; never modify MuJoCo state. Define
`raw_first_action` as row 0 of the chunk issued at the current step,
`aggregated_action` as the 7D command after temporal aggregation/hysteresis but
before IK, and `smoothing_delta_norm` as the L2 difference between their first
six dimensions. Increment `gripper_switch_count` only when the emitted discrete
state changes.

- [ ] **Step 5: Write a failing end-to-end controller test**

Use a fake H=8 policy, real `SceneGraphBuilder`, identity statistics, and a fake
environment/controller. Assert:

```python
action, diagnostics = controller.act(env)
assert policy.last_device.type == "cpu"
assert action.shape == (7,)
assert action.dtype == np.float32
assert np.all(np.abs(action[:6]) <= 1.0)
assert action[6] in {0.0, 1.0}
assert diagnostics.ik_projection_scale <= 1.0
```

Assert `controller.reset(env)` initializes hysteresis from the physical finger
state, resets the policy-step index, and removes all previously issued chunks.

- [ ] **Step 6: Implement `ChunkedPolicyController`**

Give the learned controller one construction path:

```python
class ChunkedPolicyController:
    def __init__(
        self,
        *,
        policy: ActionPolicy,
        statistics: TrainingStatistics,
        builder: SceneGraphBuilder,
        horizon: int,
        temporal_decay: float,
        gripper_close_threshold: float,
        gripper_open_threshold: float,
        device: torch.device | str = "cpu",
        edge_shuffle: bool = False,
        edge_shuffle_seed: int = 0,
    ) -> None:

    @torch.no_grad()
    def act(
        self, env: FrankaContactEnv
    ) -> tuple[np.ndarray, ChunkControllerDiagnostics]:
```

Require `device.type == "cpu"` for enabled v3 experiments. In `act`, build the
same `physics_v2` graph and 23D proprioception used by training, normalize both
with checkpoint statistics, call
`policy.predict_action_chunk(scene, proprioception)`, add the physical action
chunk to the ensemble, aggregate the current step, and finally call
`project_cartesian_action(env.controller, action)`. Update the returned
diagnostics with the projection scale. Keep edge shuffling deterministic by
policy step and expose it only through the same controller option used by the
evaluator.

Do not import `physics_evaluate` from this module. Move the small
`shuffle_valid_physics_edges` adapter into `chunked_controller.py` and have it
call the generic `evaluate.shuffle_valid_edge_assignments`; otherwise evaluator
importing the controller would create a circular import.

- [ ] **Step 7: Run shared-controller tests GREEN**

```bash
.venv/bin/python -m pytest tests/interaction_vla/test_chunked_controller.py -q
```

Expected: numerical aggregation, hysteresis, reset, CPU, graph input, H=1, edge
shuffle, clipping, and IK projection tests pass.

---

### Task 9: Evaluate interaction first and strict task completion second

**Files:**

- Modify: `interaction_vla/contact_physics.py`
- Modify: `interaction_vla/physics_data.py`
- Modify: `interaction_vla/physics_evaluate.py`
- Modify: `tests/interaction_vla/test_contact_physics.py`
- Modify: `tests/interaction_vla/test_physics_data.py`
- Modify: `tests/interaction_vla/test_physics_evaluate.py`

- [ ] **Step 1: Write failing target-specific interaction metric tests**

The existing `ever_bilateral_contact` can be satisfied by a distractor. Add a
fixture that grips `object_1` while the target is `object_0`, then grips the
target. Assert the tracker distinguishes the two events:

```python
    assert first.ever_bilateral_contact
    assert first.first_bilateral_object == "object_1"
    assert first.ever_bilateral_wrong_object
    assert not first.ever_bilateral_target_contact
    assert second.ever_bilateral_target_contact
    assert second.first_bilateral_target_substep == 7
```

Track the first bilateral object, first target bilateral substep, first
stable-target substep, total stable-target substeps, longest consecutive
stable-target run, any bilateral wrong-object interaction, and target drop.
These counters describe gripper–object understanding and are the primary result;
they do not change graph feature dimensions.

- [ ] **Step 2: Run the target interaction test RED**

```bash
.venv/bin/python -m pytest \
  tests/interaction_vla/test_contact_physics.py::test_tracker_separates_target_and_distractor_bilateral_contact -q
```

Expected: the target-specific tracker fields are missing.

- [ ] **Step 3: Extend grasp diagnostics without changing policy inputs**

Add backward-compatible fields to `GraspState` and internal counters to
`StableGraspTracker`. Increment tracker time on every physics substep. Define
`target_first_contact` as `first_bilateral_object == target_name`; a missing first
bilateral object is false, not successful. Keep the existing contact and
`InteractionSignal` schema exactly unchanged so both representations receive the
same information as before.

- [ ] **Step 4: Write failing recovery benchmark reconstruction tests**

Load one accepted record from `recovery_benchmark_manifest.json`, reconstruct its
`PhysicsRecoverySpec`, and prepare the deterministic intervention state twice.
Assert identical qpos, qvel, initial-state hash, target, source split, and recovery
kind. Assert no expert action is used after the intervention state is returned.

- [ ] **Step 5: Extract a public deterministic recovery-start helper**

Refactor the prefix and intervention portion of `collect_physics_episode` into:

```python
@dataclass(frozen=True)
class PreparedRecoveryStart:
    snapshot: SceneSnapshot
    source_seed: int
    source_split: str
    variant_id: int
    kind: str
    interaction_baseline: Mapping[str, int | bool | str | None]


def prepare_physics_recovery_start(
    env: FrankaContactEnv,
    expert: PhysicsScriptedExpert,
    *,
    spec: PhysicsRecoverySpec,
    object_count: int,
    source_split: str,
) -> PreparedRecoveryStart:
```

Use this helper in both recovery collection and benchmark evaluation. It may run
the scripted expert only to reach the declared trigger and apply the deterministic
physical intervention. Once it returns, the learned controller owns every
subsequent action. Never restore a saved qpos or modify object qpos during the
rollout.

Do not reset `StableGraspTracker` at handoff: its current stable/contact state is
part of the recovery observation seen during data collection. Instead, snapshot
all cumulative interaction counters in `interaction_baseline` and report only
post-handoff deltas for held-out recovery. Add a test where the expert prefix has
already set `ever_stable_target=True` and the learned policy immediately fails;
the policy's post-handoff stable-grasp result must be false rather than receiving
expert credit.

- [ ] **Step 6: Replace one-step evaluator inference with the shared controller**

For v3, `preload_evaluation_checkpoints` must load every policy on CPU and verify
the checkpoint temporal contract against the config. In
`rollout_physics_policy`, construct and reset `ChunkedPolicyController`, then use
only `controller.act(env)` for learned actions. Remove the duplicated graph
normalization, direct one-step policy call, manual clipping, and separate IK
projection from the v3 branch. Keep those operations only in the legacy H=1
branch.

Use an evaluator-local `InteractionRolloutTracker` initialized at learned-policy
handoff. For ID its baseline is zero. For held-out recovery it subtracts the
prepared cumulative counters and observes bilateral/stable/wrong-object state
only during learned steps. Never derive a learned-policy endpoint directly from
an `ever_*` flag that was already true at handoff.

Add benchmark cases from `recovery_benchmark_manifest.json` under condition
`heldout_recovery`. Add CLI controls:

```text
--conditions id_normal heldout_recovery
--model-seeds 0
--episodes-per-count 5
```

The default v3 evaluation includes `id_normal` and `heldout_recovery`; OOD and
edge-shuffle conditions run only when explicitly requested. `tqdm` total must be
computed after condition filtering and its postfix must include condition,
objects, policy, seed, strict success, and stable-target grasp.

- [ ] **Step 7: Expand episode results into primary and secondary endpoints**

Extend `PhysicsEpisodeResult` with explicit values while retaining its current
constructor fields and legacy JSON keys for old reports. Give appended fields
backward-compatible defaults so legacy fixtures remain valid:

```python
@dataclass(frozen=True)
class PhysicsEpisodeResult:
    first_bilateral_object: str | None = None
    target_first_contact: bool = False
    target_bilateral_contact: bool = False
    stable_target_grasp: bool = False
    first_bilateral_target_substep: int | None = None
    first_stable_target_substep: int | None = None
    stable_target_substeps: int = 0
    longest_stable_target_run: int = 0
    wrong_object_interaction: bool = False
    dropped_target: bool = False
    strict_containment: bool = False
    receptacle_base_contact: bool = False
    receptacle_wall_contact: bool = False
    strict_placement: bool = False
    gripper_released: bool = False
    tcp_retreated: bool = False
    strict_task_success: bool = False
    containment_margin_x: float | None = None
    containment_margin_y: float | None = None
    rollout_device: str = "cpu"
    mean_ensemble_size: float = 1.0
    gripper_switch_count: int = 0
```

Use the existing `stable_lift`, `wrong_object_stable_grasp`, and
`transport_progress_rate` fields as their canonical values rather than adding
duplicate fields with different names. For v3, legacy `success` aliases
`strict_task_success`, `placement` aliases `strict_placement`, and `dropped`
aliases `dropped_target`.

In `report.json`, group the metrics under:

```json
{
  "primary_interaction": {
    "target_first_contact_rate": 0.0,
    "target_bilateral_contact_rate": 0.0,
    "stable_target_grasp_rate": 0.0,
    "stable_lift_rate": 0.0,
    "grasp_given_contact_rate": 0.0,
    "mean_stable_target_substeps": 0.0,
    "wrong_object_interaction_rate": 0.0,
    "wrong_object_stable_grasp_rate": 0.0,
    "target_drop_rate": 0.0,
    "mean_transport_progress_rate": 0.0
  },
  "secondary_task": {
    "strict_placement_rate": 0.0,
    "released_and_retreated_rate": 0.0,
    "strict_task_success_rate": 0.0,
    "wall_contact_without_containment_rate": 0.0
  }
}
```

Compute `grasp_given_contact_rate` only over episodes that reached bilateral
target contact and return `null` when the denominator is zero. Write every raw
field to the episode CSV. Report Flat-minus-Graph or Graph-minus-Flat paired
deltas with an explicit sign convention, paired only by the same case ID and
model seed.

Write `evaluation/action_diagnostics.jsonl` with one row per learned policy step:
case ID, representation, model seed, step, raw current chunk row 0, aggregated
7D action before IK, executed 7D action, ensemble size, smoothing delta,
gripper score, discrete gripper state, cumulative gripper switches, and IK scale.
Store this file's SHA-256 and relative path in `report.json`; do not embed the
full step trace in the report itself.

- [ ] **Step 8: Write evaluator regression tests**

Add tests proving:

1. the previously diagnosed exterior-box seed cannot count as strict placement;
2. evaluator loads learned policies on CPU even when training config says `auto`;
3. Flat and Graph invoke the same controller class and temporal parameters;
4. held-out recovery sources are disjoint from all train sources;
5. expert-prefix counters cannot credit a held-out recovery policy;
6. interaction and secondary metrics aggregate independently;
7. `--conditions id_normal heldout_recovery` excludes OOD and edge shuffle;
8. each progress update occurs exactly once per completed rollout.

- [ ] **Step 9: Run evaluation tests GREEN**

```bash
.venv/bin/python -m pytest \
  tests/interaction_vla/test_contact_physics.py \
  tests/interaction_vla/test_physics_data.py \
  tests/interaction_vla/test_physics_evaluate.py -q
```

Expected: target-specific primary metrics, strict secondary metrics, source-clean
held-out recovery, controller sharing, CPU parity, filtering, CSV, and tqdm tests
pass.

---

### Task 10: Route dashboard, viewer, and GIF export through the same controller

**Files:**

- Modify: `interaction_vla/physics_visualize.py`
- Modify: `tests/interaction_vla/test_physics_visualize.py`

- [ ] **Step 1: Write a failing visualization/evaluation parity test**

Load one tiny H=8 checkpoint and the same deterministic case in both
`rollout_physics_policy` and `PhysicsVisualizationSession`. Advance the session
without rendering and assert identical executed actions, step count, termination
reason, strict placement flag, final object pose, and final TCP pose.

- [ ] **Step 2: Run parity test RED**

```bash
.venv/bin/python -m pytest \
  tests/interaction_vla/test_physics_visualize.py::test_learned_visualization_matches_evaluator_rollout -q
```

Expected: visualization still performs an independent one-step policy call.

- [ ] **Step 3: Replace learned visualization state with the controller**

Change `PhysicsVisualizationSession` learned fields to:

```python
learned_controller: ChunkedPolicyController | None = None
last_chunk_diagnostics: ChunkControllerDiagnostics | None = None
```

During `create`, load the checkpoint on CPU, validate its representation,
physical provenance, source hashes, and temporal contract, then build the same
controller factory used by evaluation. During `reset`, reset the learned
controller after the environment. During `advance`, call only
`learned_controller.act(self.env)` for Flat/Graph. Remove `_policy_action` from
the v3 route.

- [ ] **Step 4: Add strict and temporal diagnostics to overlays**

For learned policies, show `H8`, current ensemble size, discrete gripper command,
IK scale, target bilateral/stable contact, containment margins, base/wall
contact, strict placement, release/retreat, and termination reason. Keep expert
and teleoperation overlays compatible.

- [ ] **Step 5: Add tqdm to non-interactive GIF export**

Wrap rendered policy frames with `tqdm(desc="GIF <controller>", unit="frame")`.
For Flat-versus-Graph comparison use one bar whose total is the shared maximum
step count and whose postfix includes each policy's reason, stable grasp, and
strict placement. Do not put tqdm in the interactive GLFW dashboard/viewer event
loop.

- [ ] **Step 6: Run all visualization tests GREEN**

```bash
.venv/bin/python -m pytest tests/interaction_vla/test_physics_visualize.py -q
```

Expected: evaluator/session action parity, reset, CPU, overlay, four-view camera,
dashboard, viewer, GIF, paired GIF, and progress tests pass.

---

### Task 11: Make physical, data, and learned-rollout provenance auditable

**Files:**

- Modify: `interaction_vla/physics_provenance.py`
- Modify: `interaction_vla/physics_data.py`
- Modify: `interaction_vla/train.py`
- Modify: `interaction_vla/physics_evaluate.py`
- Modify: `interaction_vla/physics_visualize.py`
- Modify: `interaction_vla/validate_physics_expert.py`
- Modify: `tests/interaction_vla/test_validate_physics_expert.py`
- Modify: `tests/interaction_vla/test_physics_data.py`
- Modify: `tests/interaction_vla/test_train.py`
- Modify: `tests/interaction_vla/test_physics_evaluate.py`
- Modify: `tests/interaction_vla/test_physics_visualize.py`
- Modify: `tests/interaction_vla/test_no_attachment_audit.py`

- [ ] **Step 1: Write failing source-hash and stale-checkpoint tests**

Assert the physical controller hash covers strict placement and the learned
rollout hash covers chunk aggregation:

```python
assert "placement.py" in physics_control_module_names()
assert "chunked_controller.py" in learned_rollout_module_names()
assert "models/policy.py" in learned_rollout_module_names()
assert "sequence_training.py" in training_pipeline_module_names()
assert "source_split.py" in training_pipeline_module_names()
```

Create a valid checkpoint fixture, change only its stored temporal decay or
learned-rollout hash, and assert training resume, evaluation, and visualization
all reject it before creating an environment.

- [ ] **Step 2: Run provenance tests RED**

```bash
.venv/bin/python -m pytest \
  tests/interaction_vla/test_validate_physics_expert.py \
  tests/interaction_vla/test_physics_data.py \
  tests/interaction_vla/test_train.py \
  tests/interaction_vla/test_physics_evaluate.py \
  tests/interaction_vla/test_physics_visualize.py -q
```

Expected: no learned-rollout source hash or temporal-contract validator exists.

- [ ] **Step 3: Separate physical-gate and learned-rollout hashes**

Keep `controller_source_hash()` responsible for code that can change expert
validity or generated physical states. Add `placement.py` to its module list.
Expose immutable module-name functions for testing.

Add a second hash:

```python
def learned_rollout_source_hash() -> str:
    module_root = Path(__file__).parent
    return _hash_named_files(
        (name, module_root / name)
        for name in (
            "chunked_controller.py",
            "physics_action_safety.py",
            "models/policy.py",
            "models/encoders.py",
            "graph/builder.py",
            "graph/schema.py",
        )
    )
```

Add `training_pipeline_source_hash()` over `train.py`, `sequence_training.py`,
`source_split.py`, `models/policy.py`, and `models/encoders.py`. Save it in the
checkpoint and require it when resuming; evaluation and visualization retain it
as reported provenance but use the learned-rollout hash to validate execution
compatibility.

Do not include evaluation formatting or README files in either hash. Source split
and recovery manifest content are covered by their canonical data hashes from
Task 4.

Add `rollout_integrity_hash=learned_rollout_source_hash()` to
`expected_gate_hashes` and the expert-gate report. This binds the gate's
no-attachment audit to the exact learned rollout sources; modifying the chunked
controller therefore makes the old gate stale even if expert behaviour itself is
unchanged. Include the differing `rollout_integrity_hash` key in the stale-gate
error details.

- [ ] **Step 4: Validate one checkpoint contract everywhere**

Save `training_pipeline_source_hash` and `learned_rollout_source_hash` beside
`representation_contract`, `temporal_contract`, `training_provenance`, and
physical gate hashes. Implement:

```python
def validate_temporal_checkpoint(
    payload: Mapping[str, object],
    *,
    config: ExperimentConfig,
) -> None:
```

Require exact equality for horizon, loss decay, smoothing decay, both gripper
thresholds, recovery loss fraction, rollout device, contribution label, and
source hash. Call it from resume, evaluator preload, dashboard, native viewer,
and GIF creation. A mismatch is a hard error with the differing key in the
message; never silently fall back to one-step control.

- [ ] **Step 5: Expand the no-attachment audit**

Add `ChunkedPolicyController.act`, `TemporalActionEnsembler.action_for_step`, and
`prepare_physics_recovery_start` to the inspected rollout methods. Assert none
writes `data.qpos`, `data.qvel`, object mocap, weld/equality state, or object
actuators. Keep fixture-only qpos helpers confined to tests. Add an audit test
that monkeypatches the chunked controller with an object qpos write and proves
the expert gate fails.

- [ ] **Step 6: Run provenance and physical-integrity tests GREEN**

```bash
.venv/bin/python -m pytest \
  tests/interaction_vla/test_validate_physics_expert.py \
  tests/interaction_vla/test_physics_data.py \
  tests/interaction_vla/test_train.py \
  tests/interaction_vla/test_physics_evaluate.py \
  tests/interaction_vla/test_physics_visualize.py \
  tests/interaction_vla/test_no_attachment_audit.py -q
```

Expected: changed physical code invalidates the expert gate, changed temporal or
rollout code invalidates learned checkpoints, and the no-attachment audit covers
every learned and recovery rollout path.

---

### Task 12: Document and exercise the complete v3 workflow

**Files:**

- Modify: `README.md`
- Modify: `tests/interaction_vla/test_config.py`
- Modify: `tests/interaction_vla/test_physics_evaluate.py`

- [ ] **Step 1: Write a failing CLI/config workflow test**

Parse every command used below with the module's argument parser or CLI smoke
fixture. Assert collection, expert validation, Flat training, Graph training,
evaluation, dashboard, and GIF export accept the v3 config and required flags.
Assert the evaluator default for v3 contains only `id_normal` and
`heldout_recovery`, and all long-running non-interactive commands enable tqdm.

- [ ] **Step 2: Update README around the actual hypothesis**

Replace the terminal-recovery pilot as the recommended experiment with this
precise statement:

> 在物理状态信息、source episode、共享 H=8 时序头、优化预算和 rollout
> controller 完全相同的条件下，显式 interaction Graph encoder 是否比 canonical
> Flat encoder 更容易学到 gripper-aware、object-aware 操作策略？

State clearly that action chunking, exponential smoothing, gripper hysteresis,
expert gate, and recovery generation are shared infrastructure, not claimed
contributions. Document primary interaction metrics first and strict complete-task
metrics second. Do not claim Graph wins until a generated v3 report supports it.

- [ ] **Step 3: Add the full pilot command block with progress bars**

Use these exact commands:

```bash
.venv/bin/python -m interaction_vla.validate_physics_expert \
  --config configs/physics_interaction_chunk_pilot_macos.yaml

.venv/bin/python -m interaction_vla.physics_data collect \
  --config configs/physics_interaction_chunk_pilot_macos.yaml

.venv/bin/python -m interaction_vla.train \
  --config configs/physics_interaction_chunk_pilot_macos.yaml \
  --representation flat \
  --model-seed 0

.venv/bin/python -m interaction_vla.train \
  --config configs/physics_interaction_chunk_pilot_macos.yaml \
  --representation graph \
  --model-seed 0

.venv/bin/python -m interaction_vla.physics_evaluate \
  --config configs/physics_interaction_chunk_pilot_macos.yaml \
  --model-seeds 0 \
  --conditions id_normal heldout_recovery \
  --episodes-per-count 5
```

Explain that the first command must be rerun after this implementation because
strict placement and controller provenance intentionally make the old expert gate
stale. Each command already enables tqdm in its CLI entry point.

- [ ] **Step 4: Add live and GIF visualization commands**

Use CPU-learned rollout through the shared controller:

```bash
.venv/bin/python -m interaction_vla.physics_visualize dashboard \
  --config configs/physics_interaction_chunk_pilot_macos.yaml \
  --controller graph \
  --checkpoint outputs/interaction_graph_physics/interaction_chunk_pilot/graph/seed_0/checkpoint.pt \
  --layout crowded \
  --object-count 3 \
  --seed 2140049

.venv/bin/mjpython -m interaction_vla.physics_visualize export-comparison-gif \
  --config configs/physics_interaction_chunk_pilot_macos.yaml \
  --flat-checkpoint outputs/interaction_graph_physics/interaction_chunk_pilot/flat/seed_0/checkpoint.pt \
  --graph-checkpoint outputs/interaction_graph_physics/interaction_chunk_pilot/graph/seed_0/checkpoint.pt \
  --layout crowded \
  --object-count 3 \
  --seed 2140049 \
  --output docs/media/interaction_chunk_flat_vs_graph.gif
```

Verify the actual subcommand name and flags in `physics_visualize.py`; if the
existing paired exporter uses another name, update its parser and compatibility
alias so the documented `export-comparison-gif` command is accepted. Mention the
macOS `mjpython` libpython fallback already documented in the project, without
changing the user's virtual environment or deleting its cache.

- [ ] **Step 5: Run all automated tests before the physical smoke chain**

```bash
.venv/bin/python -m compileall -q interaction_vla tests/interaction_vla
.venv/bin/python -m pytest -q
```

Expected: compilation succeeds and the full suite passes with only the existing
platform skips or explicitly documented new platform skips.

- [ ] **Step 6: Check isolated output paths before running the smoke chain**

```bash
test ! -e outputs/interaction_graph_physics/interaction_chunk_smoke/expert_gate.json
test ! -e outputs/interaction_graph_physics/interaction_chunk_smoke/data/manifest.json
```

If either check fails, stop and preserve the user's artifacts; create a copied
smoke YAML with a new timestamped output suffix instead of overwriting them.

- [ ] **Step 7: Run the isolated physical smoke chain**

```bash
.venv/bin/python -m interaction_vla.validate_physics_expert \
  --config configs/physics_interaction_chunk_smoke_macos.yaml

.venv/bin/python -m interaction_vla.physics_data collect \
  --config configs/physics_interaction_chunk_smoke_macos.yaml

.venv/bin/python -m interaction_vla.train \
  --config configs/physics_interaction_chunk_smoke_macos.yaml \
  --representation flat \
  --model-seed 0

.venv/bin/python -m interaction_vla.train \
  --config configs/physics_interaction_chunk_smoke_macos.yaml \
  --representation graph \
  --model-seed 0

.venv/bin/python -m interaction_vla.physics_evaluate \
  --config configs/physics_interaction_chunk_smoke_macos.yaml \
  --model-seeds 0 \
  --conditions id_normal heldout_recovery \
  --episodes-per-count 1

.venv/bin/python -m interaction_vla.physics_visualize export-comparison-gif \
  --config configs/physics_interaction_chunk_smoke_macos.yaml \
  --flat-checkpoint outputs/interaction_graph_physics/interaction_chunk_smoke/flat/seed_0/checkpoint.pt \
  --graph-checkpoint outputs/interaction_graph_physics/interaction_chunk_smoke/graph/seed_0/checkpoint.pt \
  --layout normal \
  --object-count 2 \
  --seed 2140049 \
  --output outputs/interaction_graph_physics/interaction_chunk_smoke/evaluation/smoke_flat_vs_graph.gif
```

Expected: each command displays tqdm; the source split is exactly 8/1/1; two
train sources generate recovery augmentation; validation/test recovery appears
only in the benchmark manifest; both checkpoints record H=8, 75/25 loss mass,
CPU rollout, and identical non-encoder contracts; evaluation writes JSON and CSV.
The paired exporter writes a GIF through the same controller and its parity test
has already compared the corresponding non-rendered rollout action by action.
The one-epoch smoke is an integration check, not evidence that either
representation is better.

- [ ] **Step 8: Inspect smoke artifacts with read-only assertions**

```bash
.venv/bin/python -c 'import json, pathlib; p=pathlib.Path("outputs/interaction_graph_physics/interaction_chunk_smoke/data/source_split.json"); d=json.loads(p.read_text()); assert [len(d[k]) for k in ("train", "validation", "test")] == [8, 1, 1]; print(p)'
.venv/bin/python -c 'import json, pathlib; p=pathlib.Path("outputs/interaction_graph_physics/interaction_chunk_smoke/evaluation/report.json"); d=json.loads(p.read_text()); assert d["rollout_contract"]["device"] == "cpu"; assert "primary_interaction" in d; assert "secondary_task" in d; print(p)'
.venv/bin/python -c 'import pathlib; p=pathlib.Path("outputs/interaction_graph_physics/interaction_chunk_smoke/evaluation/smoke_flat_vs_graph.gif"); assert p.stat().st_size > 0; print(p)'
```

- [ ] **Step 9: Request code review and address substantive findings**

Invoke the `requesting-code-review` skill against the approved spec and this
plan. Fix every Critical or Important finding, rerun its focused regression test,
then rerun:

```bash
.venv/bin/python -m pytest -q
```

- [ ] **Step 10: Hand the pilot to the user without running it**

Report the exact changed files, full-test result, smoke artifact paths, and any
known limitation. Tell the user to run the README pilot commands in order. The
first decision point after evaluation is the primary interaction section; the
secondary acceptance target is Graph seed-0 ID strict-task success at or above
50%, but it must be reported as an observed result rather than guaranteed in
advance.

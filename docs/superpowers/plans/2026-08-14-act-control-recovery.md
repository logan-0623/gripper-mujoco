# ACT Control Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover a measurable Flat ACT closed-loop control floor on the existing 50-episode LeRobot dataset before any Graph v2 policy training.

**Architecture:** Keep the standard dual-RGB, 10D proprioception, 7D local-Cartesian action, and eight-step ACT training target. Add deterministic shuffled sampling, an ImageNet-initialized ResNet18, one-action receding-horizon execution, truthful loss/action diagnostics, and a bounded train-seen/held-out gate. All new outputs use `outputs/graph_control/act_recovery/` and never overwrite the completed v1 experiment.

**Tech Stack:** Python 3.12, PyTorch 2.10, torchvision ResNet18, LeRobot 0.6.1 ACT, MuJoCo 3.3.4, NumPy, pytest, YAML.

**Design reference:** `docs/superpowers/specs/2026-08-14-act-control-recovery-graph-v2-design.md`

---

## File map

- Modify `interaction_vla/lerobot_bridge/config.py`: recovery and ACT initialization contracts.
- Modify `interaction_vla/lerobot_bridge/act_smoke.py`: pretrained ACT construction, deterministic loaders, order hashes, and truthful KL metrics.
- Modify `interaction_vla/lerobot_bridge/rollout.py`: configurable receding-horizon action queue and reusable loaded-policy rollout.
- Create `interaction_vla/lerobot_bridge/interaction_phase.py`: shared causal six-phase labels for diagnostics and Graph v2.
- Create `interaction_vla/lerobot_bridge/act_diagnostics.py`: held-out chunk/action metrics.
- Create `interaction_vla/lerobot_bridge/act_recovery.py`: deterministic recovery cases, aggregate gates, and atomic report.
- Modify `interaction_vla/lerobot_bridge/cli.py`: `act-diagnose` and `act-recovery` commands.
- Modify `interaction_vla/graph_control/training.py`: reuse the same shuffled loader and include batch-order hashes in pairing checks.
- Create `configs/lerobot_act_recovery_macos.yaml`: bounded one-seed recovery experiment.
- Modify `README.md`: mark v1 control result complete/failed and expose only the next recovery commands.
- Modify or create focused tests under `tests/interaction_vla/lerobot_bridge/` and `tests/interaction_vla/graph_control/`.

### Task 1: Version the recovery configuration

**Files:**
- Modify: `interaction_vla/lerobot_bridge/config.py`
- Create: `configs/lerobot_act_recovery_macos.yaml`
- Modify: `tests/interaction_vla/lerobot_bridge/test_config.py`

- [ ] **Step 1: Write failing configuration tests**

Add these tests:

```python
def test_recovery_config_uses_pretrained_receding_horizon_act() -> None:
    config = load_bridge_config("configs/lerobot_act_recovery_macos.yaml")

    assert config.act.chunk_size == 8
    assert config.act.n_action_steps == 1
    assert config.act.shuffle_train is True
    assert (
        config.act.pretrained_backbone_weights
        == "ResNet18_Weights.IMAGENET1K_V1"
    )
    assert config.recovery is not None
    assert config.recovery.train_seen_cases == 10
    assert config.recovery.heldout_cases == 20
    assert config.recovery.heldout_attempt_multiplier == 10
    assert config.required_smoke_report is None
    assert config.recovery.train_success_threshold == pytest.approx(0.8)
    assert config.recovery.heldout_success_threshold == pytest.approx(0.3)


def test_action_horizon_must_fit_inside_chunk() -> None:
    config = load_bridge_config("configs/lerobot_act_smoke_macos.yaml")

    with pytest.raises(ValueError, match="n_action_steps"):
        replace(config.act, n_action_steps=0)
    with pytest.raises(ValueError, match="n_action_steps"):
        replace(config.act, n_action_steps=9)
```

- [ ] **Step 2: Run the tests and verify the missing contract**

Run:

```bash
.venv-lerobot/bin/python -m pytest \
  tests/interaction_vla/lerobot_bridge/test_config.py -q
```

Expected: FAIL because `shuffle_train`, `pretrained_backbone_weights`, and
`BridgeConfig.recovery` do not exist and the current horizon validation requires eight
executed actions.

- [ ] **Step 3: Add the exact configuration types**

Add to `config.py`:

```python
@dataclass(frozen=True)
class ACTRecoveryConfig:
    output_dir: Path
    train_seen_cases: int = 10
    heldout_cases: int = 20
    heldout_attempt_multiplier: int = 10
    heldout_master_seed: int = 2057736129
    max_steps: int = 180
    train_success_threshold: float = 0.8
    heldout_success_threshold: float = 0.3

    def __post_init__(self) -> None:
        if (
            self.train_seen_cases < 1
            or self.heldout_cases < 1
            or self.heldout_attempt_multiplier < 1
            or self.heldout_master_seed < 0
            or self.max_steps < 1
        ):
            raise ValueError(
                "ACT recovery counts/max_steps must be positive and seed non-negative"
            )
        for name in ("train_success_threshold", "heldout_success_threshold"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"recovery.{name} must lie within [0, 1]")
```

Extend `ACTBridgeConfig` with:

```python
shuffle_train: bool = False
pretrained_backbone_weights: str | None = None
```

Replace the fixed horizon check with:

```python
if self.chunk_size != 8:
    raise ValueError("the ACT bridge requires an 8-step training chunk")
if not 1 <= self.n_action_steps <= self.chunk_size:
    raise ValueError("ACT n_action_steps must lie within [1, chunk_size]")
if self.pretrained_backbone_weights not in {
    None,
    "ResNet18_Weights.IMAGENET1K_V1",
}:
    raise ValueError("unsupported ACT pretrained_backbone_weights")
```

Add `recovery: ACTRecoveryConfig | None` to `BridgeConfig`. In
`load_bridge_config`, pop `recovery`, convert `output_dir` to `Path`, and construct the
dataclass only when the section is present:

```python
recovery_raw = raw.pop("recovery", None)
recovery = None
if recovery_raw is not None:
    recovery_values = dict(recovery_raw)
    recovery_values["output_dir"] = Path(recovery_values["output_dir"])
    recovery = ACTRecoveryConfig(**recovery_values)
```

- [ ] **Step 4: Create the recovery YAML**

Create `configs/lerobot_act_recovery_macos.yaml` with:

```yaml
seed: 42
source_config: configs/physics_pilot_macos.yaml
expert_gate: outputs/interaction_graph_physics/pilot/expert_gate.json
required_smoke_report: null
dataset:
  repo_id: local/franka_lerobot_act_pilot
  root: outputs/lerobot/franka_lerobot_act_pilot
  episodes: 50
  object_counts: [2, 3]
  task: Pick up the green target object and place it inside the receptacle.
  fps: 20
  image_size: [256, 256]
  state_dim: 10
  action_dim: 7
  max_attempt_multiplier: 10
teacher:
  distractor_count: 2
  replacement_margin: 0.10
  replacement_frames: 3
  dropout_frames: 3
  safety_margin_m: 0.01
  goal_horizon: 8
  goal_improvement_margin: 0.05
act:
  output_dir: outputs/graph_control/act_recovery/train
  device: auto
  chunk_size: 8
  n_action_steps: 1
  batch_size: 2
  num_workers: 0
  steps: null
  epochs: 10
  maximum_epochs: 10
  learning_rate: 0.00001
  seed: 0
  dim_model: 256
  dim_feedforward: 1024
  encoder_layers: 2
  vae_encoder_layers: 2
  shuffle_train: true
  pretrained_backbone_weights: ResNet18_Weights.IMAGENET1K_V1
recovery:
  output_dir: outputs/graph_control/act_recovery/evaluation
  train_seen_cases: 10
  heldout_cases: 20
  heldout_attempt_multiplier: 10
  heldout_master_seed: 2057736129
  max_steps: 180
  train_success_threshold: 0.80
  heldout_success_threshold: 0.30
```

The recovery config deliberately sets `required_smoke_report: null`: changing the ACT
training and rollout source invalidates the old smoke report by design. The fresh
`act-check` command in Task 7 is the bounded engineering gate for this new code path,
so the formal run does not depend on a stale v1 source fingerprint.

- [ ] **Step 5: Run the focused tests**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add interaction_vla/lerobot_bridge/config.py \
  configs/lerobot_act_recovery_macos.yaml \
  tests/interaction_vla/lerobot_bridge/test_config.py
git commit -m "feat: configure bounded ACT control recovery"
```

### Task 2: Build ACT with pretrained vision and deterministic shuffled batches

**Files:**
- Modify: `interaction_vla/lerobot_bridge/act_smoke.py`
- Modify: `tests/interaction_vla/lerobot_bridge/test_act_smoke.py`
- Modify: `tests/interaction_vla/lerobot_bridge/test_act_training.py`

- [ ] **Step 1: Write failing ACT construction and order tests**

Add:

```python
def test_configured_act_uses_recovery_backbone_and_horizon() -> None:
    bridge = load_bridge_config("configs/lerobot_act_recovery_macos.yaml")
    config = _act_config(
        device=torch.device("cpu"),
        architecture="configured",
        bridge_config=bridge,
    )

    assert config.chunk_size == 8
    assert config.n_action_steps == 1
    assert config.pretrained_backbone_weights == bridge.act.pretrained_backbone_weights


def test_seeded_train_loader_is_shuffled_and_reproducible() -> None:
    dataset = list(range(24))
    first = list(iter_seeded_batches(dataset, batch_size=3, seed=17))
    second = list(iter_seeded_batches(dataset, batch_size=3, seed=17))
    different = list(iter_seeded_batches(dataset, batch_size=3, seed=18))

    flatten = lambda batches: [int(value) for batch in batches for value in batch]
    assert flatten(first) == flatten(second)
    assert flatten(first) != list(range(24))
    assert flatten(first) != flatten(different)


def test_optimizer_update_records_real_kld_loss() -> None:
    metric = optimizer_metric_from_loss_dict(
        total_loss=3.0,
        loss_dict={"l1_loss": 1.5, "kld_loss": 0.25},
    )

    assert metric == {"loss": 3.0, "l1_loss": 1.5, "kld_loss": 0.25}


def test_formal_training_summary_contains_reload_check(trained_summary) -> None:
    assert trained_summary["reload_max_abs_error"] <= 1e-5
```

Import the named helpers from `act_smoke.py` in the test modules.

- [ ] **Step 2: Verify the tests fail**

Run:

```bash
.venv-lerobot/bin/python -m pytest \
  tests/interaction_vla/lerobot_bridge/test_act_smoke.py \
  tests/interaction_vla/lerobot_bridge/test_act_training.py -q
```

Expected: FAIL because the configured ACT still hard-codes random vision, eight
executed actions, fixed-order loading, and the wrong KL key.

- [ ] **Step 3: Pass configuration through to LeRobot ACT**

In `_act_config`, preserve the tiny network-free test architecture and use the bridge
values for configured runs:

```python
n_action_steps = 8 if architecture == "test" else bridge_config.act.n_action_steps
pretrained_weights = (
    None
    if architecture == "test"
    else bridge_config.act.pretrained_backbone_weights
)
return ACTConfig(
    device=device.type,
    push_to_hub=False,
    chunk_size=8,
    n_action_steps=n_action_steps,
    vision_backbone="resnet18",
    pretrained_backbone_weights=pretrained_weights,
    dim_model=dim_model,
    dim_feedforward=dim_feedforward,
    n_encoder_layers=encoder_layers,
    n_vae_encoder_layers=vae_encoder_layers,
    use_vae=True,
    optimizer_lr=optimizer_lr,
)
```

Require the configured torchvision file to be cached before `make_policy`, so training
cannot silently switch to random initialization and unrelated model errors retain
their original message:

```python
def require_cached_backbone_weights(identifier: str | None) -> Path | None:
    if identifier is None:
        return None
    from urllib.parse import urlparse

    from torchvision.models import get_weight

    weight = get_weight(identifier)
    filename = Path(urlparse(weight.url).path).name
    cached = Path(torch.hub.get_dir()) / "checkpoints" / filename
    if not cached.is_file():
        raise FileNotFoundError(
            "ACT pretrained backbone is not cached: "
            f"{identifier}; expected {cached}"
        )
    return cached
```

Call `require_cached_backbone_weights(config.pretrained_backbone_weights)` immediately
before `make_policy`, then construct the policy normally. Extend `ACTBundle` with
`backbone_weights_sha256: str | None`; set it to `sha256_file(cached)` when the helper
returns a path and to `None` for the network-free test architecture. Store both
`pretrained_backbone_weights` and `backbone_weights_sha256` in
`bridge_checkpoint.json` and `training_summary.json`. Add a test that writes a fake
cached file, monkeypatches `torch.hub.get_dir`, and asserts the exact SHA-256 is
recorded; a missing file must raise `FileNotFoundError` before `make_policy` is called.

- [ ] **Step 4: Add one deterministic loader helper**

Add:

```python
def seeded_train_loader(
    dataset: Any, *, batch_size: int, seed: int, shuffle: bool
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        drop_last=True,
        num_workers=0,
    )


def iter_seeded_batches(
    dataset: Any, *, batch_size: int, seed: int
) -> list[object]:
    return list(
        seeded_train_loader(
            dataset,
            batch_size=batch_size,
            seed=seed,
            shuffle=True,
        )
    )
```

Replace both training `DataLoader(..., shuffle=False)` constructions in
`run_one_batch_check` and `train_once` with `seeded_train_loader`; use
`shuffle=config.act.shuffle_train` for configured training and keep validation
unshuffled.

- [ ] **Step 5: Correct the KL metric and add an order digest**

Add:

```python
def optimizer_metric_from_loss_dict(
    *, total_loss: float, loss_dict: dict[str, Any]
) -> dict[str, float]:
    return {
        "loss": float(total_loss),
        "l1_loss": _loss_value(loss_dict, "l1_loss"),
        "kld_loss": _loss_value(loss_dict, "kld_loss"),
    }


def row_order_sha256(rows: list[int]) -> str:
    values = np.asarray(rows, dtype=np.int64)
    return hashlib.sha256(values.tobytes()).hexdigest()
```

Build `_optimizer_update`'s loss fields from `optimizer_metric_from_loss_dict` and
remove the false `kl_loss` field. At each epoch boundary, concatenate
`source_row_indices`, compute `row_order_sha256`, and store `epoch_order_hashes` in the
training summary.

Factor the prediction/reload comparison currently embedded in `run_one_batch_check`
into `checkpoint_reload_max_abs_error(bundle, checkpoint, raw_batch, device)`. After
formal training saves its checkpoint, create a fixed unshuffled one-batch loader,
compare the in-memory policy with `ACTPolicy.from_pretrained(...,
local_files_only=True)` on that raw batch, reject nonfinite error or error above
`1e-5`, and write the value as `reload_max_abs_error` before the final
`training_summary.json` write. The helper must not download or reconstruct weights
from the network.

- [ ] **Step 6: Re-run focused tests**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add interaction_vla/lerobot_bridge/act_smoke.py \
  tests/interaction_vla/lerobot_bridge/test_act_smoke.py \
  tests/interaction_vla/lerobot_bridge/test_act_training.py
git commit -m "fix: train ACT with shuffled pretrained vision"
```

### Task 3: Execute one action from every predicted chunk

**Files:**
- Modify: `interaction_vla/lerobot_bridge/rollout.py`
- Modify: `interaction_vla/graph_control/rollout.py`
- Modify: `tests/interaction_vla/lerobot_bridge/test_rollout.py`
- Modify: `tests/interaction_vla/graph_control/test_rollout.py`

- [ ] **Step 1: Replace the eight-action queue expectation with horizon tests**

Add:

```python
def test_chunk_queue_requeries_after_configured_action_horizon() -> None:
    calls = 0

    def predict() -> np.ndarray:
        nonlocal calls
        calls += 1
        return np.full((8, 7), calls, dtype=np.float32)

    queue = ActionChunkQueue(chunk_size=8, n_action_steps=1)
    selected = [queue.next(predict) for _ in range(3)]

    assert calls == 3
    assert [item.queue_index for item in selected] == [0, 0, 0]
    assert [float(item.action[0]) for item in selected] == [1.0, 2.0, 3.0]


def test_chunk_queue_can_retain_legacy_eight_action_execution() -> None:
    calls = 0

    def predict() -> np.ndarray:
        nonlocal calls
        calls += 1
        return np.zeros((8, 7), dtype=np.float32)

    queue = ActionChunkQueue(chunk_size=8, n_action_steps=8)
    for _ in range(9):
        queue.next(predict)

    assert calls == 2
```

- [ ] **Step 2: Run rollout tests and verify failure**

Run:

```bash
.venv-lerobot/bin/python -m pytest \
  tests/interaction_vla/lerobot_bridge/test_rollout.py \
  tests/interaction_vla/graph_control/test_rollout.py -q
```

Expected: FAIL because `ActionChunkQueue` has no `n_action_steps` argument.

- [ ] **Step 3: Implement the bounded queue**

Use:

```python
class ActionChunkQueue:
    def __init__(self, *, chunk_size: int, n_action_steps: int) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if not 1 <= n_action_steps <= chunk_size:
            raise ValueError("n_action_steps must lie within [1, chunk_size]")
        self.chunk_size = int(chunk_size)
        self.n_action_steps = int(n_action_steps)
        self.reset()

    def reset(self) -> None:
        self._chunk: np.ndarray | None = None
        self._index = 0

    def next(self, predict: Callable[[], np.ndarray]) -> QueuedAction:
        if self._chunk is None or self._index >= self.n_action_steps:
            chunk = np.asarray(predict(), dtype=np.float32)
            expected = (self.chunk_size, 7)
            if chunk.shape != expected:
                raise ValueError(f"predicted action chunk must have shape {expected}")
            if not np.isfinite(chunk).all():
                raise ValueError("predicted action chunk must be finite")
            self._chunk = chunk.copy()
            self._index = 0
        queue_index = self._index
        action = self._chunk[queue_index].copy()
        raw_chunk = self._chunk.copy()
        self._index += 1
        return QueuedAction(action, raw_chunk, queue_index)
```

Instantiate it in both standard and Graph rollout paths with:

```python
queue = ActionChunkQueue(
    chunk_size=config.act.chunk_size,
    n_action_steps=config.act.n_action_steps,
)
```

- [ ] **Step 4: Re-run rollout tests**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add interaction_vla/lerobot_bridge/rollout.py \
  interaction_vla/graph_control/rollout.py \
  tests/interaction_vla/lerobot_bridge/test_rollout.py \
  tests/interaction_vla/graph_control/test_rollout.py
git commit -m "fix: use receding-horizon ACT rollout"
```

### Task 4: Add action-direction diagnostics

**Files:**
- Create: `interaction_vla/lerobot_bridge/interaction_phase.py`
- Create: `interaction_vla/lerobot_bridge/act_diagnostics.py`
- Create: `tests/interaction_vla/lerobot_bridge/test_interaction_phase.py`
- Create: `tests/interaction_vla/lerobot_bridge/test_act_diagnostics.py`

- [ ] **Step 1: Write numerical metric tests**

Create the test file with:

```python
import numpy as np
import pytest

from interaction_vla.lerobot_bridge.act_diagnostics import action_chunk_metrics


def test_action_chunk_metrics_mask_padding_and_report_direction() -> None:
    target = np.zeros((1, 3, 7), dtype=np.float32)
    predicted = np.zeros_like(target)
    target[0, 0, :3] = (1.0, 2.0, 0.0)
    predicted[0, 0, :3] = (1.0, -2.0, 0.0)
    target[0, 1, :3] = (0.5, 0.0, 0.0)
    predicted[0, 1, :3] = (0.5, 0.0, 0.0)
    predicted[0, 2] = 1000.0
    is_pad = np.asarray([[False, False, True]])

    metrics = action_chunk_metrics(predicted, target, is_pad)

    assert metrics["valid_actions"] == 2
    assert metrics["translation_mae_y"] == pytest.approx(2.0)
    assert metrics["first_translation_mae_y"] == pytest.approx(4.0)
    assert metrics["first_translation_sign_accuracy"] == pytest.approx(2.0 / 3.0)
    assert metrics["translation_direction_cosine"] < 1.0
    assert metrics["gripper_mae"] == pytest.approx(0.0)


def test_action_chunk_metrics_reject_nonfinite_or_empty_input() -> None:
    values = np.zeros((1, 2, 7), dtype=np.float32)
    with pytest.raises(ValueError, match="valid action"):
        action_chunk_metrics(values, values, np.ones((1, 2), dtype=np.bool_))
    values[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        action_chunk_metrics(values, np.zeros_like(values), np.zeros((1, 2), dtype=np.bool_))


def test_causal_phase_labels_use_only_current_and_previous_relations() -> None:
    relation = phase_relation_fixture()
    first = causal_phase_ids(relation)
    changed_future = relation.copy()
    changed_future[-1, :, :] = 100.0
    second = causal_phase_ids(changed_future)

    np.testing.assert_array_equal(first[:-1], second[:-1])
    assert set(first.tolist()) <= set(range(6))
```

Import `causal_phase_ids` from the new `interaction_phase.py`; the fixture provides
finite `[T, 8, 24]` current-relation values spanning approach through release.

- [ ] **Step 2: Verify failure**

Run:

```bash
.venv-lerobot/bin/python -m pytest \
  tests/interaction_vla/lerobot_bridge/test_interaction_phase.py \
  tests/interaction_vla/lerobot_bridge/test_act_diagnostics.py -q
```

Expected: FAIL because both new modules do not exist.

- [ ] **Step 3: Implement shared causal phase labels**

Create `interaction_phase.py` with `PHASE_NAMES`, `PHASE_IDS`,
`causal_phase_step(values, previous)`, and `causal_phase_ids(relation)`. Validate one
finite `[8, 24]` frame in the step function and a finite `[T, 8, 24]` array in the
sequence function. The step function applies this exact current/backward-only state
machine:

```python
contact = values[0, PROBABILITY_0] >= 0.45
co_motion = values[0, PROBABILITY_1] >= 0.70
grasped = contact and co_motion
placed = values[1, PROBABILITY_0] >= 0.50
on_support = values[2, PROBABILITY_0] >= 0.50
target_distance = float(np.linalg.norm(values[0, RELATIVE_POSITION]))
goal_distance = float(np.linalg.norm(values[1, RELATIVE_POSITION]))
support_gap = float(values[2, SIGNED_MARGIN_0])
if placed and not grasped:
    current = PHASE_IDS["release"]
elif placed and grasped:
    current = PHASE_IDS["place"]
elif grasped and (on_support or support_gap < 0.06):
    current = PHASE_IDS["lift"]
elif grasped and goal_distance > 0.08:
    current = PHASE_IDS["transport"]
elif grasped:
    current = PHASE_IDS["place"]
elif target_distance <= 0.07:
    current = PHASE_IDS["grasp"]
else:
    current = PHASE_IDS["approach"]
if previous == PHASE_IDS["release"] and current == PHASE_IDS["approach"]:
    current = previous
```

`causal_phase_step` returns `current`. `causal_phase_ids` initializes `previous` to
approach, calls the step function per frame, appends the returned ID, and updates
`previous`. The module imports only NumPy and the public teacher channel constants.

- [ ] **Step 4: Implement the pure metric function**

Create `act_diagnostics.py` with the validated core:

```python
from __future__ import annotations

from typing import Any

import numpy as np


def action_chunk_metrics(
    predicted: object, target: object, action_is_pad: object
) -> dict[str, Any]:
    prediction = np.asarray(predicted, dtype=np.float32)
    truth = np.asarray(target, dtype=np.float32)
    padding = np.asarray(action_is_pad, dtype=np.bool_)
    if prediction.ndim != 3 or prediction.shape[-1] != 7 or prediction.shape != truth.shape:
        raise ValueError("predicted and target actions must share shape [batch, chunk, 7]")
    if padding.shape != prediction.shape[:2]:
        raise ValueError("action_is_pad must have shape [batch, chunk]")
    if not np.isfinite(prediction).all() or not np.isfinite(truth).all():
        raise ValueError("action diagnostics require finite values")
    valid = ~padding
    if not np.any(valid):
        raise ValueError("action diagnostics require at least one valid action")
    errors = np.abs(prediction - truth)
    translation_prediction = prediction[..., :3][valid]
    translation_truth = truth[..., :3][valid]
    denominator = np.linalg.norm(translation_prediction, axis=1) * np.linalg.norm(
        translation_truth, axis=1
    )
    directional = denominator > 1e-8
    cosines = np.divide(
        np.sum(translation_prediction * translation_truth, axis=1),
        denominator,
        out=np.zeros_like(denominator),
        where=directional,
    )
    first_valid = ~padding[:, 0]
    if not np.any(first_valid):
        raise ValueError("action diagnostics require a valid first action")
    first_errors = errors[first_valid, 0]
    first_prediction = prediction[first_valid, 0, :3]
    first_truth = truth[first_valid, 0, :3]
    first_denominator = np.linalg.norm(first_prediction, axis=1) * np.linalg.norm(
        first_truth, axis=1
    )
    first_directional = first_denominator > 1e-8
    first_cosines = np.divide(
        np.sum(first_prediction * first_truth, axis=1),
        first_denominator,
        out=np.zeros_like(first_denominator),
        where=first_directional,
    )
    first_sign = np.sign(prediction[first_valid, 0, :3]) == np.sign(
        truth[first_valid, 0, :3]
    )
    return {
        "valid_actions": int(valid.sum()),
        "translation_mae_x": float(errors[..., 0][valid].mean()),
        "translation_mae_y": float(errors[..., 1][valid].mean()),
        "translation_mae_z": float(errors[..., 2][valid].mean()),
        "rotation_mae": float(errors[..., 3:6][valid].mean()),
        "gripper_mae": float(errors[..., 6][valid].mean()),
        "first_translation_mae_x": float(first_errors[:, 0].mean()),
        "first_translation_mae_y": float(first_errors[:, 1].mean()),
        "first_translation_mae_z": float(first_errors[:, 2].mean()),
        "first_rotation_mae": float(first_errors[:, 3:6].mean()),
        "first_gripper_mae": float(first_errors[:, 6].mean()),
        "translation_direction_cosine": float(cosines[directional].mean())
        if np.any(directional)
        else 0.0,
        "first_translation_direction_cosine": float(
            first_cosines[first_directional].mean()
        )
        if np.any(first_directional)
        else 0.0,
        "first_translation_sign_accuracy": float(first_sign.mean()),
    }
```

- [ ] **Step 5: Add checkpoint/dataset evaluation**

In the same module, add `evaluate_checkpoint_actions(config_path, checkpoint)` that:

1. loads the 40/5/5 split with `pilot_episode_split`;
2. loads train and validation `LeRobotDataset` views;
3. loads the checkpoint once with `load_act_runtime`;
4. runs unshuffled batches through preprocessor, `predict_action_chunk`, and
   postprocessor;
5. passes postprocessed predictions, raw `action`, and `action_is_pad` to
   `action_chunk_metrics`;
6. writes `action_diagnostics.json` below `config.recovery.output_dir` with separate
   `train` and `validation` records.

Use the exact batch extraction:

```python
processed = preprocessor(raw_batch)
with torch.inference_mode():
    normalized = policy.predict_action_chunk(processed)
    predicted = postprocessor(normalized).detach().cpu().numpy()
target = raw_batch["action"].detach().cpu().numpy()
padding = raw_batch["action_is_pad"].detach().cpu().numpy()
```

Concatenate arrays across batches before calling the pure metric function, so means
are weighted by valid actions rather than by batch count.

Load each episode's `annotation.tc_tig.relation_values` through the hashed teacher
manifest, compute `causal_phase_ids`, and map every global dataset row to its current
phase. Partition first-action predictions/targets by that row phase and add
`by_phase: {phase_name: metrics}` to both train and validation records. Teacher arrays
remain diagnostics-only and are never passed to ACT preprocessing or forward.

- [ ] **Step 6: Run tests**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add interaction_vla/lerobot_bridge/interaction_phase.py \
  interaction_vla/lerobot_bridge/act_diagnostics.py \
  tests/interaction_vla/lerobot_bridge/test_interaction_phase.py \
  tests/interaction_vla/lerobot_bridge/test_act_diagnostics.py
git commit -m "feat: report ACT directional action errors"
```

### Task 5: Add the bounded closed-loop recovery gate

**Files:**
- Modify: `interaction_vla/lerobot_bridge/rollout.py`
- Create: `interaction_vla/lerobot_bridge/act_recovery.py`
- Create: `tests/interaction_vla/lerobot_bridge/test_act_recovery.py`

- [ ] **Step 1: Write deterministic case and gate tests**

Create:

```python
import pytest

from interaction_vla.lerobot_bridge.act_recovery import (
    RecoveryCase,
    aggregate_recovery,
    heldout_candidates,
    train_seen_cases,
)


def test_train_seen_cases_use_only_training_normal_two_object_episodes() -> None:
    manifest = [
        {"episode_index": index, "seed": 100 + index, "object_count": 2 if index % 2 == 0 else 3}
        for index in range(20)
    ]
    cases = train_seen_cases(manifest, train_episodes=range(20), count=5)

    assert len(cases) == 5
    assert all(case.partition == "train_seen" for case in cases)
    assert all(case.layout == "normal" and case.object_count == 2 for case in cases)
    assert len({case.seed for case in cases}) == 5


def test_heldout_candidate_schedule_is_deterministic_and_disjoint() -> None:
    first = heldout_candidates(master_seed=17, count=200, forbidden_seeds={1, 2, 3})
    second = heldout_candidates(master_seed=17, count=200, forbidden_seeds={1, 2, 3})

    assert first == second
    assert len(first) == 200
    assert not ({case.seed for case in first} & {1, 2, 3})


def test_recovery_requires_both_prespecified_gates() -> None:
    records = [
        {"partition": "train_seen", "success": index < 8, "termination_reason": "success" if index < 8 else "timeout"}
        for index in range(10)
    ] + [
        {"partition": "heldout", "success": index < 6, "termination_reason": "success" if index < 6 else "timeout"}
        for index in range(20)
    ]

    report = aggregate_recovery(
        records,
        train_threshold=0.8,
        heldout_threshold=0.3,
    )

    assert report["passed"] is True
    assert report["train_seen"]["success_rate"] == pytest.approx(0.8)
    assert report["heldout"]["success_rate"] == pytest.approx(0.3)


def test_recovery_rejects_unverified_checkpoint_summary(tmp_path) -> None:
    summary = tmp_path / "training_summary.json"
    summary.write_text('{"reload_max_abs_error": 0.01}', encoding="utf-8")

    with pytest.raises(ValueError, match="reload"):
        require_verified_training_summary(summary)
```

- [ ] **Step 2: Verify failure**

Run:

```bash
.venv-lerobot/bin/python -m pytest \
  tests/interaction_vla/lerobot_bridge/test_act_recovery.py -q
```

Expected: FAIL because `act_recovery.py` does not exist.

- [ ] **Step 3: Implement deterministic cases and aggregation**

Create the module with:

```python
@dataclass(frozen=True)
class RecoveryCase:
    case_id: str
    partition: str
    seed: int
    layout: str = "normal"
    object_count: int = 2


def train_seen_cases(
    manifest: Sequence[Mapping[str, object]],
    *,
    train_episodes: Iterable[int],
    count: int,
) -> tuple[RecoveryCase, ...]:
    allowed = {int(value) for value in train_episodes}
    selected = sorted(
        (
            int(record["episode_index"]),
            int(record["seed"]),
        )
        for record in manifest
        if int(record["episode_index"]) in allowed
        and int(record["object_count"]) == 2
    )
    if len(selected) < count:
        raise ValueError("not enough train-seen normal two-object episodes")
    return tuple(
        RecoveryCase(f"train_seen_{episode:03d}", "train_seen", seed)
        for episode, seed in selected[:count]
    )


def heldout_candidates(
    *, master_seed: int, count: int, forbidden_seeds: set[int]
) -> tuple[RecoveryCase, ...]:
    cases: list[RecoveryCase] = []
    replicate = 0
    while len(cases) < count:
        seed = int(
            np.random.SeedSequence((master_seed, 0x48454C44, replicate)).generate_state(
                1, dtype=np.uint32
            )[0]
        )
        replicate += 1
        if seed in forbidden_seeds:
            continue
        cases.append(RecoveryCase(f"heldout_{len(cases):03d}", "heldout", seed))
    return tuple(cases)


def aggregate_recovery(
    records: Sequence[Mapping[str, object]],
    *,
    train_threshold: float,
    heldout_threshold: float,
) -> dict[str, object]:
    groups: dict[str, dict[str, object]] = {}
    for partition in ("train_seen", "heldout"):
        selected = [record for record in records if record["partition"] == partition]
        if not selected:
            raise ValueError(f"recovery report is missing {partition} records")
        rate = float(np.mean([bool(record["success"]) for record in selected]))
        groups[partition] = {
            "cases": len(selected),
            "successes": sum(bool(record["success"]) for record in selected),
            "success_rate": rate,
            "termination_counts": dict(
                Counter(str(record["termination_reason"]) for record in selected)
            ),
        }
    return {
        "passed": bool(
            groups["train_seen"]["success_rate"] >= train_threshold
            and groups["heldout"]["success_rate"] >= heldout_threshold
        ),
        **groups,
    }
```

Import `Counter`, `dataclass`, `Iterable`, `Mapping`, `Sequence`, and NumPy.

- [ ] **Step 4: Factor loaded-policy rollout without changing action semantics**

In `rollout.py`, add:

```python
@dataclass
class LoadedACTRuntime:
    checkpoint: Path
    policy: Any
    preprocessor: Any
    postprocessor: Any

    def reset(self) -> None:
        if hasattr(self.policy, "reset"):
            self.policy.reset()
```

Extract the existing environment loop into:

```python
def rollout_loaded_policy(
    config: BridgeConfig,
    runtime: LoadedACTRuntime,
    *,
    seed: int,
    object_count: int,
    layout: LayoutMode,
    max_steps: int,
    recorder: RolloutGIFRecorder | None = None,
) -> dict[str, object]:
    if max_steps < 1:
        raise ValueError("rollout max_steps must be positive")
    env = _make_env(config, max_steps=max_steps)
    validate_finger_joint_ranges(env.model)
    snapshot = env.reset(
        seed=seed,
        object_count=object_count,
        layout_mode=layout,
    )
    capture = DualViewCapture(
        env.model,
        width=config.dataset.image_size[1],
        height=config.dataset.image_size[0],
    )
    runtime.reset()
    queue = ActionChunkQueue(
        chunk_size=config.act.chunk_size,
        n_action_steps=config.act.n_action_steps,
    )
    gripper = BinaryGripperHysteresis(
        close_threshold=0.4,
        open_threshold=0.6,
        initially_open=True,
    )
    projection_scales: list[float] = []
    clipped_steps = 0
    reason = TerminationReason.RUNNING
    try:
        for _ in range(max_steps):
            camera_frame = capture.capture(env, include_teacher=False)
            state = EndEffectorStateCodec.encode_snapshot(
                snapshot, env.proprioception()
            )
            observation = policy_observation(
                agent_rgb=camera_frame.views["agent"].rgb,
                wrist_rgb=camera_frame.views["wrist"].rgb,
                state=state,
            )
            selected = queue.next(
                lambda: _predict_chunk(
                    policy=runtime.policy,
                    preprocessor=runtime.preprocessor,
                    postprocessor=runtime.postprocessor,
                    observation=observation,
                )
            )
            raw = selected.action.copy()
            action = raw.copy()
            action[:6] = np.clip(action[:6], -1.0, 1.0)
            clipped_steps += int(np.any(action[:6] != raw[:6]))
            action[6] = gripper.resolve(float(raw[6]))
            rotation = EndEffectorStateCodec.quaternion_to_matrix(
                snapshot.gripper.orientation
            )
            world = LocalCartesianActionCodec.decode(action, rotation)
            projection = project_cartesian_action(env.controller, world)
            projection_scales.append(float(projection.scale))
            transition = env.step(projection.action)
            snapshot = transition.snapshot
            reason = transition.reason
            record_rollout_gif_frame(
                recorder,
                camera_frame,
                step=len(projection_scales) - 1,
                gripper_open=gripper.is_open,
                terminal_reason=reason.value,
            )
            if transition.done:
                break
    finally:
        capture.close()
    steps = len(projection_scales)
    return {
        "success": reason is TerminationReason.SUCCESS,
        "termination_reason": reason.value,
        "steps": steps,
        "mean_ik_projection_scale": float(np.mean(projection_scales))
        if steps
        else 1.0,
        "action_clipping_rate": float(clipped_steps / steps) if steps else 0.0,
        "gripper_switch_count": gripper.switch_count,
    }
```

Change `_make_env` to accept `max_steps: int | None = None` and pass
`source.environment.max_steps if max_steps is None else max_steps` to
`FrankaContactEnv`. `rollout_checkpoint` becomes a load-once wrapper around this
function. It creates the optional recorder, passes it to `rollout_loaded_policy`, calls
`write()` afterward, adds `gif` and `gif_frames` to the result, and atomically writes
the existing `rollout.json`. Recovery rollouts pass `recorder=None` and do not write
per-case JSON files.

- [ ] **Step 5: Add the real recovery runner**

Implement `evaluate_recovery(config_path, checkpoint)` in `act_recovery.py`:

```python
config = load_bridge_config(config_path)
if config.recovery is None:
    raise ValueError("bridge config does not define recovery")
training_summary = require_verified_training_summary(
    Path(checkpoint) / "training_summary.json"
)
split = pilot_episode_split(total_episodes=config.dataset.episodes, seed=config.act.seed)
manifest = json.loads(
    (config.dataset.root / "meta" / "teacher_manifest.json").read_text(encoding="utf-8")
)
seen = train_seen_cases(
    manifest,
    train_episodes=split["train"],
    count=config.recovery.train_seen_cases,
)
all_dataset_seeds = {int(record["seed"]) for record in manifest}
candidates = heldout_candidates(
    master_seed=config.recovery.heldout_master_seed,
    count=(
        config.recovery.heldout_cases
        * config.recovery.heldout_attempt_multiplier
    ),
    forbidden_seeds=all_dataset_seeds,
)
expert_seen = [
    {**asdict(case), **rollout_expert_case(
        config, case, max_steps=config.recovery.max_steps
    )}
    for case in seen
]
expert_candidates = []
unseen = []
for case in candidates:
    result = rollout_expert_case(
        config, case, max_steps=config.recovery.max_steps
    )
    expert_candidates.append({**asdict(case), **result})
    if result["success"]:
        unseen.append(case)
    if len(unseen) == config.recovery.heldout_cases:
        break
if (
    not all(record["success"] for record in expert_seen)
    or len(unseen) != config.recovery.heldout_cases
):
    failure = {
        "passed": False,
        "failure_stage": "expert_case_gate",
        "expert_train_seen": expert_seen,
        "expert_heldout_candidates": expert_candidates,
    }
    _write_json_atomic(
        config.recovery.output_dir / "recovery_report.json", failure
    )
    return failure
device = resolve_device(config.act.device)
policy, preprocessor, postprocessor, metadata = _load_checkpoint_bundle(
    config=config,
    checkpoint=Path(checkpoint),
    device=device,
)
runtime = LoadedACTRuntime(Path(checkpoint), policy, preprocessor, postprocessor)
records = []
for case in (*seen, *unseen):
    result = rollout_loaded_policy(
        config,
        runtime,
        seed=case.seed,
        object_count=case.object_count,
        layout=LayoutMode(case.layout),
        max_steps=config.recovery.max_steps,
    )
    records.append({**asdict(case), **result})
report = aggregate_recovery(
    records,
    train_threshold=config.recovery.train_success_threshold,
    heldout_threshold=config.recovery.heldout_success_threshold,
)
report.update(
    {
        "checkpoint": Path(checkpoint).as_posix(),
        "checkpoint_reload_max_abs_error": training_summary[
            "reload_max_abs_error"
        ],
        "heldout_master_seed": config.recovery.heldout_master_seed,
        "heldout_candidate_limit": len(candidates),
        "heldout_candidates_inspected": len(expert_candidates),
        "heldout_selected_case_ids": [case.case_id for case in unseen],
        "records": records,
        "expert_train_seen": expert_seen,
        "expert_heldout_candidates": expert_candidates,
    }
)
_write_json_atomic(config.recovery.output_dir / "recovery_report.json", report)
return report
```

`require_verified_training_summary` requires a real JSON object with finite
`reload_max_abs_error <= 1e-5`; a missing, malformed, or unverified formal checkpoint
fails before creating an environment. Add the same held-out selection provenance to
the `expert_case_gate` failure report so failed qualification is auditable.

Add this expert check and run it on the identical cases before loading ACT:

```python
def rollout_expert_case(
    config: BridgeConfig, case: RecoveryCase, *, max_steps: int
) -> dict[str, object]:
    env = _make_env(config, max_steps=max_steps)
    snapshot = env.reset(
        seed=case.seed,
        object_count=case.object_count,
        layout_mode=LayoutMode(case.layout),
    )
    expert = PhysicsScriptedExpert(config.source.physics)
    expert.reset(seed=case.seed)
    reason = TerminationReason.RUNNING
    for step in range(max_steps):
        action = expert.act(snapshot, env.contact_diagnostics, env.grasp_state)
        transition = env.step(action)
        snapshot = transition.snapshot
        reason = transition.reason
        if transition.done:
            break
    return {
        "case_id": case.case_id,
        "success": reason is TerminationReason.SUCCESS,
        "termination_reason": reason.value,
        "steps": step + 1,
    }
```

The selected held-out set is therefore the first 20 expert successes in a deterministic
200-candidate stream. Candidate failures remain in the report, and learned-policy
behavior never participates in selection. The helper uses only normal `env.reset` and
`env.step`; it never changes MuJoCo state directly.

- [ ] **Step 6: Run focused tests**

Run the Step 2 command plus:

```bash
.venv-lerobot/bin/python -m pytest \
  tests/interaction_vla/lerobot_bridge/test_rollout.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add interaction_vla/lerobot_bridge/rollout.py \
  interaction_vla/lerobot_bridge/act_recovery.py \
  tests/interaction_vla/lerobot_bridge/test_act_recovery.py \
  tests/interaction_vla/lerobot_bridge/test_rollout.py
git commit -m "feat: gate ACT on train-seen and held-out control"
```

### Task 6: Expose the commands and preserve paired Graph training invariants

**Files:**
- Modify: `interaction_vla/lerobot_bridge/cli.py`
- Modify: `interaction_vla/graph_control/training.py`
- Modify: `tests/interaction_vla/lerobot_bridge/test_cli.py`
- Modify: `tests/interaction_vla/graph_control/test_training.py`

- [ ] **Step 1: Write failing CLI and pairing tests**

Assert the parser accepts:

```python
diagnose = build_parser().parse_args(
    ["act-diagnose", "--config", "recovery.yaml", "--checkpoint", "checkpoint"]
)
recovery = build_parser().parse_args(
    ["act-recovery", "--config", "recovery.yaml", "--checkpoint", "checkpoint"]
)
assert diagnose.command == "act-diagnose"
assert recovery.command == "act-recovery"
```

Extend the pairing fixture summaries with `epoch_order_hashes` and assert a one-byte
difference raises `ValueError` mentioning that field.

- [ ] **Step 2: Run focused tests and verify failure**

```bash
.venv-lerobot/bin/python -m pytest \
  tests/interaction_vla/lerobot_bridge/test_cli.py \
  tests/interaction_vla/graph_control/test_training.py -q
```

Expected: FAIL because the commands and paired order field are absent.

- [ ] **Step 3: Add the CLI subcommands**

Add both parsers with required `--config` and `--checkpoint`. Dispatch them to
`evaluate_checkpoint_actions` and `evaluate_recovery` respectively:

```python
if args.command == "act-diagnose":
    return evaluate_checkpoint_actions(args.config, args.checkpoint)
if args.command == "act-recovery":
    return evaluate_recovery(args.config, args.checkpoint)
```

- [ ] **Step 4: Use the shared seeded loader for Graph-conditioned ACT**

Replace the fixed-order loader in `graph_control/training.py` with:

```python
loader = seeded_train_loader(
    conditioned_train,
    batch_size=batch_size,
    seed=seed,
    shuffle=True if bridge_config is None else bridge_config.act.shuffle_train,
)
```

Store per-epoch `epoch_order_hashes` and add the field to
`assert_paired_summaries.paired_fields`. This guarantees future Flat/oracle/predicted
conditions see identical row order.

- [ ] **Step 5: Re-run focused tests**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add interaction_vla/lerobot_bridge/cli.py \
  interaction_vla/graph_control/training.py \
  tests/interaction_vla/lerobot_bridge/test_cli.py \
  tests/interaction_vla/graph_control/test_training.py
git commit -m "feat: expose reproducible ACT recovery workflow"
```

### Task 7: Document, verify, and run the real Mac gate

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace the stale “next Graph ACT experiment” section**

State the audited v1 outcome: 240 rollouts, zero successes, a train-seen Flat timeout,
and no valid Graph-control conclusion. Make the next commands exactly:

```bash
.venv-lerobot/bin/python -m interaction_vla.lerobot_bridge act-check \
  --config configs/lerobot_act_recovery_macos.yaml

.venv-lerobot/bin/python -m interaction_vla.lerobot_bridge act-train \
  --config configs/lerobot_act_recovery_macos.yaml

.venv-lerobot/bin/python -m interaction_vla.lerobot_bridge act-diagnose \
  --config configs/lerobot_act_recovery_macos.yaml \
  --checkpoint outputs/graph_control/act_recovery/train/checkpoint

.venv-lerobot/bin/python -m interaction_vla.lerobot_bridge act-recovery \
  --config configs/lerobot_act_recovery_macos.yaml \
  --checkpoint outputs/graph_control/act_recovery/train/checkpoint
```

State that Graph v2 commands remain blocked until `recovery_report.json` has
`passed=true`.

- [ ] **Step 2: Run all network-free tests**

```bash
HF_HOME=/tmp/gripper-mujoco-hf-cache \
  .venv-lerobot/bin/python -m pytest tests/interaction_vla -q
```

Expected: all tests pass with zero failures. The duplicate FFmpeg Objective-C warning
may still print on macOS; it is not a test failure.

- [ ] **Step 3: Cache the official ResNet18 weights**

Run with the user's proxy only if the weight is not already cached:

```bash
.venv-lerobot/bin/python -c 'from torchvision.models import ResNet18_Weights, resnet18; resnet18(weights=ResNet18_Weights.IMAGENET1K_V1); print("cached")'
```

Expected: `cached`. An SSL or proxy failure stops the real training step without
falling back to random initialization.

- [ ] **Step 4: Run the bounded training and diagnostics**

Run the four README commands in order. Expected engineering evidence:

- finite one-batch loss and gradient;
- 10 completed epochs with non-sequential, reproducible `epoch_order_hashes`;
- checkpoint reload error at or below `1e-5`;
- finite train and validation directional metrics;
- 30 closed-loop recovery records.

- [ ] **Step 5: Apply the prespecified stopping rule**

Read:

```bash
.venv-lerobot/bin/python -c 'import json; p=json.load(open("outputs/graph_control/act_recovery/evaluation/recovery_report.json")); print(json.dumps({"passed":p["passed"],"train_seen":p["train_seen"]["success_rate"],"heldout":p["heldout"]["success_rate"]}, indent=2))'
```

The command must print `passed: true`, with `train_seen >= 0.8` and
`heldout >= 0.3`, before starting Graph v2 policy work. A boundary example is:

```json
{
  "passed": true,
  "train_seen": 0.8,
  "heldout": 0.3
}
```

Values may exceed those minima. If `passed=false`, stop and use the phase-partitioned
action diagnostics to choose exactly one follow-up ablation: phase-balanced sampling
first, then 200 demonstrations, then recovery/DAgger data.

- [ ] **Step 6: Commit documentation after verified commands match it**

```bash
git add README.md
git commit -m "docs: make ACT recovery the next experiment"
```

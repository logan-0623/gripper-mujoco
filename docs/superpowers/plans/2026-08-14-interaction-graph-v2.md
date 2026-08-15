# Interaction Graph v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the non-directional 75D Graph v1 control token with a causal, coordinate-invariant 89D Interaction Graph v2 and evaluate oracle value before training vision-estimated Graph policies.

**Architecture:** Upgrade the existing `graph_finetune` and `graph_control` paths in place while preserving completed v1 artifacts as immutable results. Derive v2 labels from current and preceding teacher frames, train a temporally conditioned visual-language graph estimator, feed its frozen 89D token through ACT's native environment-state input, and enforce recovery/oracle/prediction stopping gates. Version checks reject every v1 checkpoint and cache in the v2 path.

**Tech Stack:** Python 3.12, NumPy, PyTorch 2.10, LeRobot 0.6.1 ACT, MuJoCo 3.3.4, Hugging Face dataset/checkpoint conventions, pytest, YAML.

**Prerequisite:** `outputs/graph_control/act_recovery/evaluation/recovery_report.json` from `2026-08-14-act-control-recovery.md` must contain `passed=true`.

**Design reference:** `docs/superpowers/specs/2026-08-14-act-control-recovery-graph-v2-design.md`

---

## File map

- Modify `interaction_vla/graph_finetune/schema.py`: v2 target schema, token layout, normalization, and exact version constants.
- Modify `interaction_vla/graph_finetune/data.py`: causal v2 target extraction, previous-token samples, and training-only normalization.
- Modify `interaction_vla/graph_finetune/model.py`: geometry, phase, trend, and previous-Graph model heads plus losses and Reflect transfer.
- Modify `interaction_vla/graph_finetune/pipeline.py`: v2 metrics, split preparation, checkpoint provenance, and paired fine-tuning.
- Modify `interaction_vla/graph_finetune/cli.py`: add a non-training `split` command.
- Modify `interaction_vla/graph_control/schema.py`: re-export the versioned 89D layout and define staged conditions.
- Modify `interaction_vla/graph_control/features.py`: predicted/oracle v2 packing and stateful frozen inference.
- Modify `interaction_vla/graph_control/cache.py`: chronological per-episode caches and oracle caches without a Graph checkpoint.
- Modify `interaction_vla/graph_control/config.py`: recovery prerequisite and oracle/full condition matrices.
- Modify `interaction_vla/graph_control/training.py`: dynamic paired condition sets and v2 checkpoint binding.
- Modify `interaction_vla/graph_control/rollout.py`: causal oracle provider, stateful predicted provider, and staged aggregation.
- Modify `interaction_vla/graph_control/pipeline.py`: oracle-first and full predicted workflows.
- Create three v2 Mac configs under `configs/`.
- Update focused tests in both Graph packages and simplify `README.md` commands.

### Task 1: Define the exact v2 schema and 89D token

**Files:**
- Modify: `interaction_vla/graph_finetune/schema.py`
- Modify: `interaction_vla/graph_control/schema.py`
- Modify: `tests/interaction_vla/graph_finetune/test_data.py`
- Modify: `tests/interaction_vla/graph_control/test_schema.py`

- [ ] **Step 1: Write failing schema tests**

Replace v1 dimension assertions with:

```python
from interaction_vla.graph_finetune.schema import (
    GRAPH_SCHEMA_VERSION,
    TOKEN_DIM,
    TOKEN_FEATURE_NAMES,
    TOKEN_SCHEMA_VERSION,
    TOKEN_SLICES,
    GraphV2Targets,
)


def test_graph_v2_token_layout_is_exact_and_stable() -> None:
    expected = {
        "entity_presence": (0, 6),
        "entity_visibility": (6, 18),
        "relation_presence": (18, 26),
        "gripper_target_geometry": (26, 34),
        "target_receptacle_geometry": (34, 44),
        "distractor_geometry": (44, 58),
        "phase": (58, 64),
        "relation_trends": (64, 68),
        "next_relation": (68, 76),
        "relation_operator": (76, 81),
        "predicate": (81, 88),
        "goal_residual": (88, 89),
    }
    assert GRAPH_SCHEMA_VERSION == "mujoco_interaction_graph_v2"
    assert TOKEN_SCHEMA_VERSION == "interaction_graph_control_v2"
    assert {name: (value.start, value.stop) for name, value in TOKEN_SLICES.items()} == expected
    assert TOKEN_DIM == 89
    assert len(TOKEN_FEATURE_NAMES) == 89
    assert len(set(TOKEN_FEATURE_NAMES)) == 89


def test_v2_targets_reject_wrong_shapes_and_nonfinite_values() -> None:
    valid = make_graph_v2_targets(frames=3)
    assert valid.gripper_target_geometry.shape == (3, 8)
    assert valid.target_receptacle_geometry.shape == (3, 10)
    assert valid.distractor_geometry.shape == (3, 2, 7)
    assert valid.phase.shape == (3,)
    assert valid.relation_trends.shape == (3, 4)

    with pytest.raises(ValueError, match="gripper_target_geometry"):
        replace(valid, gripper_target_geometry=np.zeros((3, 7), dtype=np.float32))
```

`make_graph_v2_targets` is a test helper that fills every required field with finite,
correctly typed arrays and valid categorical IDs.

- [ ] **Step 2: Run schema tests and verify v1 failure**

```bash
.venv-lerobot/bin/python -m pytest \
  tests/interaction_vla/graph_finetune/test_data.py \
  tests/interaction_vla/graph_control/test_schema.py -q
```

Expected: FAIL because the current graph schema is `mujoco_semantic_graph_v1` and the
control token is 75D.

- [ ] **Step 3: Replace the graph target dataclass and centralize token slices**

In `graph_finetune/schema.py`, define:

```python
GRAPH_SCHEMA_VERSION = "mujoco_interaction_graph_v2"
TOKEN_SCHEMA_VERSION = "interaction_graph_control_v2"

_TOKEN_WIDTHS = (
    ("entity_presence", 6),
    ("entity_visibility", 12),
    ("relation_presence", 8),
    ("gripper_target_geometry", 8),
    ("target_receptacle_geometry", 10),
    ("distractor_geometry", 14),
    ("phase", 6),
    ("relation_trends", 4),
    ("next_relation", 8),
    ("relation_operator", 5),
    ("predicate", 7),
    ("goal_residual", 1),
)


def _token_slices() -> Mapping[str, slice]:
    cursor = 0
    result: dict[str, slice] = {}
    for name, width in _TOKEN_WIDTHS:
        result[name] = slice(cursor, cursor + width)
        cursor += width
    return MappingProxyType(result)


TOKEN_SLICES = _token_slices()
TOKEN_DIM = TOKEN_SLICES["goal_residual"].stop
TOKEN_FEATURE_NAMES = tuple(
    f"{name}_{index}"
    for name, bounds in TOKEN_SLICES.items()
    for index in range(bounds.stop - bounds.start)
)
```

Import `PHASE_NAMES` and `PHASE_IDS` from
`interaction_vla.lerobot_bridge.interaction_phase`; do not create a second phase
vocabulary.

Replace `MuJoCoGraphTargets` with the frozen `GraphV2Targets` fields:

```python
entity_mask: np.ndarray                 # [T, 6] bool
entity_visibility: np.ndarray           # [T, 6, 2] float32
relation_mask: np.ndarray               # [T, 8] bool
gripper_target_geometry: np.ndarray     # [T, 8] float32
target_receptacle_geometry: np.ndarray  # [T, 10] float32
distractor_geometry: np.ndarray         # [T, 2, 7] float32
phase: np.ndarray                       # [T] int64
relation_trends: np.ndarray             # [T, 4] float32
goal_relation: np.ndarray               # [T] int64
goal_operator: np.ndarray               # [T] int64
goal_predicate: np.ndarray              # [T] int64
goal_residual: np.ndarray                # [T] float32
```

Validate all shapes/dtypes/finiteness, phase IDs within `[0, 6)`, the existing goal
category ranges, probabilities/confidences within `[0, 1]`, and inactive distractor
slots as zero.

- [ ] **Step 4: Make Graph control import the central contract**

Replace the duplicated layout in `graph_control/schema.py` with imports from
`graph_finetune.schema`, and define:

```python
ORACLE_CONDITIONS = ("flat", "oracle_graph_v2")
ALL_CONDITIONS = (
    "flat",
    "oracle_graph_v2",
    "predicted_random_v2",
    "predicted_reflect_v2",
)
CONDITIONS = ALL_CONDITIONS


def validate_token(value: object) -> np.ndarray:
    token = np.asarray(value, dtype=np.float32)
    if token.shape != (TOKEN_DIM,) or not np.isfinite(token).all():
        raise ValueError(f"Graph v2 token must be finite with shape [{TOKEN_DIM}]")
    return token.copy()
```

- [ ] **Step 5: Re-run schema tests**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add interaction_vla/graph_finetune/schema.py \
  interaction_vla/graph_control/schema.py \
  tests/interaction_vla/graph_finetune/test_data.py \
  tests/interaction_vla/graph_control/test_schema.py
git commit -m "feat: define the 89D interaction graph v2 schema"
```

### Task 2: Derive causal geometry, phase, trends, and next goals

**Files:**
- Modify: `interaction_vla/graph_finetune/data.py`
- Modify: `tests/interaction_vla/graph_finetune/test_data.py`
- Modify: `tests/interaction_vla/lerobot_bridge/test_teacher.py`

- [ ] **Step 1: Write failing geometry and causality tests**

Add tests that construct one sidecar episode and assert:

```python
targets = graph_v2_targets(arrays)

np.testing.assert_allclose(
    targets.gripper_target_geometry[:, :3],
    arrays["annotation.tc_tig.relation_values"][:, 0, :3],
)
np.testing.assert_allclose(
    targets.target_receptacle_geometry[:, 3:6],
    arrays["annotation.tc_tig.relation_values"][:, 1, :3],
)
assert targets.relation_trends[0].tolist() == [0.0, 0.0, 0.0, 0.0]
assert set(targets.phase.tolist()) <= set(range(6))
```

Make two copies of the arrays that differ only in
`annotation.tc_tig.relation_goal`; assert every v2 target is exactly equal. This proves
the future-window v1 goal is ignored.

Add a rigid-transform test using `transform_snapshot_passive`: extract geometry before
and after a random translation/yaw and assert all v2 metric fields are equal within
`1e-6`.

- [ ] **Step 2: Run and verify failure**

```bash
.venv-lerobot/bin/python -m pytest \
  tests/interaction_vla/graph_finetune/test_data.py \
  tests/interaction_vla/lerobot_bridge/test_teacher.py -q
```

Expected: FAIL because `graph_v2_targets` and causal phase/goal labels do not exist.

- [ ] **Step 3: Extract the exact metric groups**

Implement `graph_v2_targets(arrays)` using current teacher fields only. Let
`relation = arrays["annotation.tc_tig.relation_values"]`, `pose` be entity pose, and
decode rotation-6D with `EndEffectorStateCodec.decode_rotation`.

For every frame, populate:

```python
gripper_target = np.stack(
    (
        relation[:, 0, 0],
        relation[:, 0, 1],
        relation[:, 0, 2],
        np.linalg.norm(relation[:, 0, :3], axis=1),
        closing_speed(relation[:, 0, :3], relation[:, 0, 6:9]),
        relation[:, 0, PROBABILITY_0],
        relation[:, 0, PROBABILITY_1],
        relation[:, 0, CONFIDENCE],
    ),
    axis=1,
).astype(np.float32)
```

Here `PROBABILITY_0` is contact probability and `PROBABILITY_1` is co-motion
probability; do not relabel either as an oracle stable-grasp bit.

Compute `target_to_goal_action` as
`R_gripper.T @ (p_receptacle - p_target)`. Populate target-receptacle fields in this
exact order:

```text
action-frame delta XYZ,
receptacle-frame target offset XYZ,
min(x_margin, y_margin),
-abs(bottom_gap),
containment probability,
confidence
```

For distractor slot 0 use relation 3, for slot 1 use relation 5. Each seven-value row
is:

```text
gripper-frame delta XYZ, clearance, collision risk, target-confusion risk, confidence
```

Use relation channels `SIGNED_MARGIN_0`, `RISK_1`, `RISK_0`, and `CONFIDENCE`; zero an
inactive slot.

- [ ] **Step 4: Reuse the causal phase state machine**

Import `causal_phase_ids`, `causal_phase_step`, `PHASE_IDS`, and `PHASE_NAMES` from
`interaction_vla.lerobot_bridge.interaction_phase`. Call it on the episode's current
relation tensor. The ACT recovery plan already tests that changing a future relation
frame cannot change earlier labels. Keep the Graph data input audit rejecting expert
controller phase, action, future frame, success, and termination values.

- [ ] **Step 5: Implement backward trends and causal goals**

Build scalar series for gripper-target distance, grasp confidence
`contact * co_motion`, target-goal distance, and minimum active distractor clearance.
Use `np.diff(..., prepend=first_value)` so frame zero is exactly zero.

Map phases to current next goals:

```python
PHASE_GOALS = {
    PHASE_IDS["approach"]: (0, OPERATOR_IDS["establish"], PREDICATE_IDS["proximity"]),
    PHASE_IDS["grasp"]: (0, OPERATOR_IDS["establish"], PREDICATE_IDS["enclosure"]),
    PHASE_IDS["lift"]: (0, OPERATOR_IDS["establish"], PREDICATE_IDS["co_motion"]),
    PHASE_IDS["transport"]: (1, OPERATOR_IDS["establish"], PREDICATE_IDS["proximity"]),
    PHASE_IDS["place"]: (1, OPERATOR_IDS["establish"], PREDICATE_IDS["containment"]),
    PHASE_IDS["release"]: (0, OPERATOR_IDS["break"], PREDICATE_IDS["co_motion"]),
}
```

Store the raw signed residual as follows: negative gripper-target distance for
approach, `-(1 - contact)` for grasp, `-(1 - co_motion)` for lift, negative
target-receptacle distance for transport, `-(1 - containment)` for place, and negative
co-motion probability for release. Distance residuals remain in metres until the
training-only normalizer is available; probability residuals are already
dimensionless. Never read the sidecar `relation_goal` array. In particular, the place phase continues to describe
the task-relevant target-to-receptacle relation; it must not switch to the unrelated
global support surface.

Add a rollout-safe tracker that applies the same formulas one frame at a time:

```python
class CausalGraphV2Tracker:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._previous_scalars: np.ndarray | None = None
        self._previous_phase = PHASE_IDS["approach"]

    def update(self, frame: TeacherFrame) -> GraphV2Targets:
        arrays = teacher_frame_arrays(frame)
        current = current_graph_v2_target(
            arrays,
            previous_scalars=self._previous_scalars,
            previous_phase=self._previous_phase,
        )
        self._previous_scalars = graph_v2_trend_scalars(current).copy()
        self._previous_phase = int(current.phase[0])
        return current
```

`teacher_frame_arrays` converts one `TeacherFrame` into one-row arrays;
`current_graph_v2_target` contains the single-frame formulas also used by
`graph_v2_targets`. `graph_v2_targets` must iterate frames through this tracker rather
than maintain a separate batch-only state machine. `current_graph_v2_target` calls
`causal_phase_step(current_relation, previous_phase)`, ensuring offline labels and
rollout oracle tokens share byte-identical logic.

- [ ] **Step 6: Re-run causality tests**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add interaction_vla/graph_finetune/data.py \
  tests/interaction_vla/graph_finetune/test_data.py \
  tests/interaction_vla/lerobot_bridge/test_teacher.py
git commit -m "feat: derive causal object-centric graph v2 targets"
```

### Task 3: Fit training-only scales and provide previous Graph input

**Files:**
- Modify: `interaction_vla/graph_finetune/schema.py`
- Modify: `interaction_vla/graph_finetune/data.py`
- Modify: `tests/interaction_vla/graph_finetune/test_data.py`

- [ ] **Step 1: Write failing normalization and temporal sample tests**

Add:

```python
def test_v2_normalization_uses_selected_train_episodes_only(prepared_corpus) -> None:
    first = prepared_corpus.for_training_fraction(0.25, seed=0)
    changed = prepared_corpus_with_test_geometry_offset(prepared_corpus, offset=1000.0)
    second = changed.for_training_fraction(0.25, seed=0)

    np.testing.assert_array_equal(
        first.normalization.state_mean, second.normalization.state_mean
    )
    np.testing.assert_array_equal(
        first.normalization.state_std, second.normalization.state_std
    )
    assert first.normalization.workspace_scale == second.normalization.workspace_scale
    assert first.normalization.velocity_scale == second.normalization.velocity_scale


def test_dataset_previous_token_is_zero_only_at_episode_start(training_corpus) -> None:
    dataset = MuJoCoGraphDataset(
        training_corpus,
        partition="train",
        image_size=32,
        max_language_tokens=8,
    )
    first = dataset[item_for_frame(dataset, episode=0, frame=0)]
    second = dataset[item_for_frame(dataset, episode=0, frame=1)]

    assert first["previous_graph"].shape == (89,)
    assert torch.count_nonzero(first["previous_graph"]) == 0
    assert torch.count_nonzero(second["previous_graph"]) > 0
```

- [ ] **Step 2: Run and verify failure**

Run:

```bash
.venv-lerobot/bin/python -m pytest \
  tests/interaction_vla/graph_finetune/test_data.py -q
```

Expected: FAIL because v1 normalization has `(8, 10)` relation statistics and samples
contain no previous Graph.

- [ ] **Step 3: Replace normalization with the v2 scale contract**

Define:

```python
@dataclass(frozen=True)
class GraphV2Normalization:
    state_mean: np.ndarray
    state_std: np.ndarray
    workspace_scale: float
    velocity_scale: float

    def __post_init__(self) -> None:
        state_mean = _float_array(self.state_mean, (10,), "state_mean")
        state_std = _float_array(self.state_std, (10,), "state_std")
        if np.any(state_std <= 0.0):
            raise ValueError("state_std must be positive")
        for name in ("workspace_scale", "velocity_scale"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        object.__setattr__(self, "state_mean", state_mean)
        object.__setattr__(self, "state_std", state_std)
```

Fit only selected training episodes. Use the maximum 95th percentile across all
active relative-position/distance/clearance magnitudes, clamped to at least `0.10 m`,
as `workspace_scale`. Use the 95th percentile absolute closing speed, clamped to at
least `0.05 m/s`, as `velocity_scale`.

- [ ] **Step 4: Add deterministic oracle token packing**

Add `pack_oracle_target(targets, frame, normalization)` to `schema.py`. It writes masks,
visibility, normalized geometry, one-hot phase/goal values, trends normalized by the
same workspace scale where metric, and clipped goal residual into the exact 89D
slices. Divide position, distance, margin, and clearance fields by `workspace_scale`;
divide closing speed by `velocity_scale`; divide trend indices 0, 2, and 3 by
`workspace_scale` while leaving grasp-confidence trend index 1 unchanged. Divide
approach/transport goal residuals by `workspace_scale`, leave the other residuals
dimensionless, then clip every residual to `[-1, 1]`. Validate the final token before
return. Use the same normalization helper for model target tensors so oracle and
predicted tokens have identical units.

- [ ] **Step 5: Add previous-frame token to every model sample**

Extend `MODEL_BATCH_KEYS` with `previous_graph`. In `MuJoCoGraphDataset.__getitem__`:

```python
previous_graph = (
    np.zeros(TOKEN_DIM, dtype=np.float32)
    if frame == 0
    else pack_oracle_target(target, frame - 1, normalization)
)
```

Return `torch.from_numpy(previous_graph)`. Return the normalized current geometry
groups and unmodified bounded probabilities/confidences as their named target tensors.

- [ ] **Step 6: Re-run data tests**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add interaction_vla/graph_finetune/schema.py \
  interaction_vla/graph_finetune/data.py \
  tests/interaction_vla/graph_finetune/test_data.py
git commit -m "feat: add train-only graph scales and temporal input"
```

### Task 4: Train the spatially aware v2 visual-language estimator

**Files:**
- Modify: `interaction_vla/graph_finetune/model.py`
- Modify: `tests/interaction_vla/graph_finetune/test_model.py`

- [ ] **Step 1: Write failing output, loss, and transfer tests**

Assert a two-example forward pass returns exactly:

```python
expected = {
    "entity_mask_logits": (2, 6),
    "entity_visibility": (2, 6, 2),
    "relation_mask_logits": (2, 8),
    "gripper_target_geometry": (2, 8),
    "target_receptacle_geometry": (2, 10),
    "distractor_geometry": (2, 2, 7),
    "phase_logits": (2, 6),
    "relation_trends": (2, 4),
    "goal_relation_logits": (2, 8),
    "goal_operator_logits": (2, 5),
    "goal_predicate_logits": (2, 7),
    "goal_residual": (2,),
    "graph_embedding": (2, 128),
}
assert {name: tuple(value.shape) for name, value in outputs.items()} == expected
```

Assert the loss has finite keys `entity_mask`, `entity_visibility`, `relation_mask`,
`gripper_target`, `target_receptacle`, `distractor`, `phase`, `temporal_trend`, three
goal classifications, `goal_residual`, and `total`. Assert inactive distractor target
changes do not alter the loss.

Add a deterministic `SpatialSoftmaxPool` test using two synthetic feature maps whose
single activation is shifted horizontally. Assert their pooled X moments differ in
the same direction while the output shape remains fixed. This is the regression test
that prevents global average pooling from silently returning.

```python
pool = SpatialSoftmaxPool(channels=1, keypoints=1, output_dim=4)
with torch.no_grad():
    pool.attention.weight.fill_(4.0)
    pool.attention.bias.zero_()
left = torch.zeros(1, 1, 3, 5)
right = torch.zeros_like(left)
left[0, 0, 1, 1] = 1.0
right[0, 0, 1, 3] = 1.0
_, left_xy = pool.summarize(left)
_, right_xy = pool.summarize(right)
assert right_xy[0, 0, 0] > left_xy[0, 0, 0]
assert pool(left).shape == (1, 4)
```

Add a controlled two-view test whose `spatial_view_fusion` selects only its first
half, then swap agent and wrist spatial inputs and assert the fused value changes.
This prevents a regression back to averaging view-dependent pixel coordinates.

Assert paired random/Reflect initialization starts with byte-identical spatial pool,
spatial view-fusion, new geometry, phase, trend, state, and previous-Graph heads. Only
the existing compatible image encoder, language embeddings, fusion, operator, and
predicate parameters may differ after transfer.

- [ ] **Step 2: Run and verify failure**

```bash
.venv-lerobot/bin/python -m pytest \
  tests/interaction_vla/graph_finetune/test_model.py -q
```

Expected: FAIL because the model exposes the v1 relation head, globally pools image
features, and has no temporal input.

- [ ] **Step 3: Preserve spatial layout and replace the model heads**

Keep the existing `image_encoder` unchanged so its global semantic branch and every
Reflect checkpoint key remain compatible. Add a shared spatial pool that taps the
output after its first six convolution/ReLU layers, before global pooling:

```python
class SpatialSoftmaxPool(nn.Module):
    def __init__(
        self,
        *,
        channels: int,
        keypoints: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.attention = nn.Conv2d(channels, keypoints, kernel_size=1)
        self.projection = nn.Linear(keypoints * (channels + 2), output_dim)

    def summarize(self, features: Tensor) -> tuple[Tensor, Tensor]:
        batch, channels, height, width = features.shape
        weights = self.attention(features).flatten(2).softmax(dim=-1)
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(
                -1.0, 1.0, height, device=features.device, dtype=features.dtype
            ),
            torch.linspace(
                -1.0, 1.0, width, device=features.device, dtype=features.dtype
            ),
            indexing="ij",
        )
        coordinates = torch.stack((grid_x, grid_y), dim=-1).reshape(-1, 2)
        moments = torch.einsum("bkn,nd->bkd", weights, coordinates)
        appearance = torch.einsum(
            "bkn,bcn->bkc", weights, features.flatten(2)
        )
        return appearance, moments

    def forward(self, features: Tensor) -> Tensor:
        appearance, moments = self.summarize(features)
        return self.projection(
            torch.cat((appearance, moments), dim=-1).flatten(1)
        )
```

Instantiate `SpatialSoftmaxPool(channels=128, keypoints=6,
output_dim=image_embedding_dim)` once and share it across agent and wrist views. Add:

```python
self.spatial_view_fusion = nn.Sequential(
    nn.Linear(2 * image_embedding_dim, image_embedding_dim),
    nn.ReLU(),
)
```

Use this helper to return the transferred global semantic vector and the
location-sensitive vector separately:

```python
def _encode_image(self, image: Tensor) -> tuple[Tensor, Tensor]:
    spatial_features = self.image_encoder[:6](image)
    semantic_context = self.image_encoder[6:](spatial_features)
    spatial_context = self.spatial_pool(spatial_features)
    return semantic_context, spatial_context
```

Retain `state_encoder`, `token_embedding`, and `fusion` names and add:

```python
self.previous_graph_encoder = nn.Sequential(
    nn.Linear(TOKEN_DIM, image_embedding_dim),
    nn.ReLU(),
)
self.gripper_target_head = nn.Linear(graph_embedding_dim, 8)
self.target_receptacle_head = nn.Linear(graph_embedding_dim, 10)
self.distractor_head = nn.Linear(graph_embedding_dim, 14)
self.phase_head = nn.Linear(graph_embedding_dim, 6)
self.trend_head = nn.Linear(graph_embedding_dim, 4)
```

Extend `forward` with `previous_graph: Tensor`, validate shape `[batch, 89]`, and fuse
it additively so the existing Reflect fusion tensor shapes remain compatible:

```python
agent_semantic, agent_spatial = self._encode_image(agent_rgb)
wrist_semantic, wrist_spatial = self._encode_image(wrist_rgb)
semantic_context = (agent_semantic + wrist_semantic) * 0.5
spatial_context = self.spatial_view_fusion(
    torch.cat((agent_spatial, wrist_spatial), dim=-1)
)
visual_context = (
    semantic_context
    + spatial_context
    + state_context
    + self.previous_graph_encoder(previous_graph)
)
graph_embedding = self.fusion(torch.cat((visual_context, language), dim=-1))
```

Apply sigmoid only to bounded geometry slices: indices 5:8 of gripper-target, 8:10 of
target-receptacle, and 4:7 of each distractor. Leave normalized vectors, margins,
distances, speeds, and trends unconstrained.

- [ ] **Step 4: Implement masked, weighted v2 loss**

Define and checkpoint this constant mapping:

```python
LOSS_WEIGHTS = {
    "entity_mask": 1.0,
    "entity_visibility": 1.0,
    "relation_mask": 1.0,
    "gripper_target": 2.0,
    "target_receptacle": 2.0,
    "distractor": 1.0,
    "phase": 1.0,
    "temporal_trend": 0.5,
    "goal_relation": 1.0,
    "goal_operator": 1.0,
    "goal_predicate": 1.0,
    "goal_residual": 0.5,
}
```

Use BCE for masks, MSE for visibility, Smooth L1 for geometry/trends/residual, and
cross-entropy for phase/goals. Mask gripper-target with relation 0, target-receptacle
with relation 1, distractor 0 with relation 3, and distractor 1 with relation 5. Sum
`LOSS_WEIGHTS[name] * losses[name]` into `total`.

- [ ] **Step 5: Update Reflect transfer exclusions**

Continue strict-loading the compatible `image_encoder`, matching language tokens,
`fusion`, `operator_head`, and `predicate_head` from the Reflect checkpoint. Explicitly
include `spatial_pool`, `spatial_view_fusion`, `previous_graph_encoder`, all geometry heads,
`phase_head`, `trend_head`, `entity_mask_head`, `entity_visibility_head`,
`relation_mask_head`, `goal_relation_head`, `residual_head`, and `state_encoder` in the
identically-random transfer report. Assert every listed tensor is byte-equal between
the paired models before fine-tuning.

- [ ] **Step 6: Re-run model tests**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add interaction_vla/graph_finetune/model.py \
  tests/interaction_vla/graph_finetune/test_model.py
git commit -m "feat: train temporal graph v2 geometry heads"
```

### Task 5: Version fine-tuning, metrics, and split preparation

**Files:**
- Modify: `interaction_vla/graph_finetune/config.py`
- Modify: `interaction_vla/graph_finetune/pipeline.py`
- Modify: `interaction_vla/graph_finetune/cli.py`
- Create: `configs/mujoco_graph_v2_finetune_macos.yaml`
- Modify: `tests/interaction_vla/graph_finetune/test_pipeline.py`
- Modify: `tests/interaction_vla/graph_finetune/test_cli.py`

- [ ] **Step 1: Write failing checkpoint and split tests**

Assert:

```python
split = split_from_config("configs/mujoco_graph_v2_finetune_macos.yaml")
payload = json.loads(Path(split["path"]).read_text())
assert payload["schema_version"] == "mujoco_interaction_graph_v2"
assert set(payload["episode_indices"]) == {"train", "validation", "test"}

with pytest.raises(ValueError, match="schema"):
    load_finetune_checkpoint(v1_checkpoint, device="cpu")
```

Update evaluation assertions to require geometry MAE per group, phase accuracy, trend
MAE, causal goal metrics, and total held-out loss.

- [ ] **Step 2: Run and verify failure**

```bash
.venv-lerobot/bin/python -m pytest \
  tests/interaction_vla/graph_finetune/test_pipeline.py \
  tests/interaction_vla/graph_finetune/test_cli.py -q
```

Expected: FAIL because the v2 split command and metrics do not exist.

- [ ] **Step 3: Add v2 evaluation metrics and checkpoint fields**

Replace semantic-relation MAE with:

```text
gripper_target_geometry_mae
target_receptacle_geometry_mae
distractor_geometry_mae
phase_accuracy
relation_trend_mae
goal_relation_accuracy
goal_operator_accuracy
goal_predicate_accuracy
goal_exact_accuracy
goal_residual_mae
```

Checkpoint payloads must contain:

```python
{
    "schema_version": GRAPH_SCHEMA_VERSION,
    "token_schema_version": TOKEN_SCHEMA_VERSION,
    "token_dim": TOKEN_DIM,
    "token_feature_names": list(TOKEN_FEATURE_NAMES),
    "loss_weights": LOSS_WEIGHTS,
    "causal_goal_labels": True,
    "future_relation_goal_used": False,
}
```

`load_finetune_checkpoint` checks all five fields in addition to the existing model,
vocabulary, normalization, split, and state contracts. A v1 checkpoint fails before
model construction.

Add `required_oracle_report: Path | None = None` to the fine-tune `TrainingConfig` and
convert it to `Path` in the loader when present. `split` and `inspect` do not require
the report, because they prepare the oracle experiment. `compare` requires the file,
loads it as JSON, checks `oracle_gate.passed is True`, and binds its SHA-256 into every
v2 Graph checkpoint and comparison report.

- [ ] **Step 4: Add a non-training split command**

Factor split-manifest construction from `compare_with_source` into
`write_split_manifest(config, corpus)`. Implement:

```python
def split_from_config(path: str | Path) -> dict[str, object]:
    config = load_graph_finetune_config(path)
    source, records, sidecars = _load_local_inputs(config)
    corpus = _prepare(config, source, records, sidecars)
    destination = config.training.output_dir / "split_manifest.json"
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        expected = split_manifest_payload(config, corpus)
        if existing != expected:
            raise ValueError("existing Graph v2 split manifest is incompatible")
    else:
        _write_json_atomic(destination, split_manifest_payload(config, corpus))
    return {"passed": True, "path": destination, "episodes": corpus.splits}
```

Expose
`python -m interaction_vla.graph_finetune split --config configs/mujoco_graph_v2_finetune_macos.yaml`
in the CLI.

- [ ] **Step 5: Create the v2 fine-tune config**

```yaml
dataset:
  repo_id: local/franka_lerobot_act_pilot
  root: outputs/lerobot/franka_lerobot_act_pilot
  reflect_checkpoint: outputs/graph_pretrain/reflectvlm/checkpoint.pt
  split_seed: 17
  split_ratios: [0.8, 0.1, 0.1]
model:
  image_size: 128
  max_language_tokens: 32
  image_embedding_dim: 128
  text_embedding_dim: 64
  graph_embedding_dim: 128
training:
  output_dir: outputs/graph_finetune/mujoco_graph_v2
  required_oracle_report: outputs/graph_control/graph_v2_oracle/runs/evaluation/report.json
  device: auto
  batch_size: 16
  num_workers: 0
  epochs: 10
  learning_rate: 0.0003
  weight_decay: 0.0001
  fractions: [0.1, 0.25, 1.0]
  seeds: [0, 1, 2]
```

- [ ] **Step 6: Re-run pipeline tests**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add interaction_vla/graph_finetune/config.py \
  interaction_vla/graph_finetune/pipeline.py \
  interaction_vla/graph_finetune/cli.py \
  configs/mujoco_graph_v2_finetune_macos.yaml \
  tests/interaction_vla/graph_finetune/test_pipeline.py \
  tests/interaction_vla/graph_finetune/test_cli.py
git commit -m "feat: version graph v2 fine-tuning and metrics"
```

### Task 6: Pack predicted/oracle tokens and run stateful inference

**Files:**
- Modify: `interaction_vla/graph_control/features.py`
- Modify: `interaction_vla/graph_control/cache.py`
- Modify: `tests/interaction_vla/graph_control/test_features.py`
- Modify: `tests/interaction_vla/graph_control/test_cache.py`

- [ ] **Step 1: Write failing packing and recurrence tests**

Assert predicted output packing produces 89 finite values; all entity/relation/phase
and goal distributions sum correctly; an oracle target produces the exact normalized
local vectors; and a frozen runtime sends the previous predicted token to the next
model call but resets to zeros at an episode boundary.

Add cache tests that provide two episodes with two frames each and assert the runtime
event log is:

```python
assert runtime.events == [
    ("reset",),
    ("predict", 0, 0),
    ("predict", 0, 1),
    ("reset",),
    ("predict", 1, 0),
    ("predict", 1, 1),
]
```

- [ ] **Step 2: Run and verify failure**

```bash
.venv-lerobot/bin/python -m pytest \
  tests/interaction_vla/graph_control/test_features.py \
  tests/interaction_vla/graph_control/test_cache.py -q
```

Expected: FAIL because v1 packing reads relation channels 12:22 and inference is
stateless/batched across arbitrary rows.

- [ ] **Step 3: Replace predicted token packing**

`pack_predicted` writes sigmoid entity/relation masks, visibility, all three predicted
geometry groups, softmax phase and goal distributions, trends, and residual into the
central slices. Validate bounded fields and the final shape.

`pack_oracle_current` becomes:

```python
def pack_oracle_current(
    targets: GraphV2Targets,
    *,
    frame_index: int,
    normalization: GraphV2Normalization,
) -> np.ndarray:
    return pack_oracle_target(targets, frame_index, normalization)
```

Unlike v1, oracle goal/phase fields are exact causal one-hot values and do not come
from a Reflect estimator.

- [ ] **Step 4: Make frozen prediction stateful**

Add to `FrozenGraphRuntime`:

```python
def reset(self) -> None:
    self.previous_graph = np.zeros(TOKEN_DIM, dtype=np.float32)


@torch.inference_mode()
def predict_token(
    self,
    *,
    agent_rgb: object,
    wrist_rgb: object,
    state: object,
    task: str,
) -> np.ndarray:
    outputs = self.model(
        prepared_agent,
        prepared_wrist,
        prepared_state,
        language_tokens,
        language_mask,
        torch.from_numpy(self.previous_graph[None]).to(self.device),
    )
    token = pack_predicted(outputs, sample_index=0)
    self.previous_graph = token.copy()
    return token
```

Initialize with `reset()` and reject any checkpoint whose v2 schema, token layout, or
normalization is incompatible.

- [ ] **Step 5: Cache in episode chronology**

Change `build_token_cache` to accept `episode_rows: Mapping[int, Sequence[int]]`.
Iterate sorted episodes, call `runtime.reset()`, then iterate each episode's rows in
frame order. Validate the sample's `episode_index`, `frame_index`, and global `index`
before `predict_token`. Oracle uses the precomputed episode `GraphV2Targets`; Flat
writes zeros. Store cache schema `graph_control_cache_v2`, token schema, feature names,
and ordered row hash in its manifest.

Update `CacheProvenance.__post_init__` with the exact checkpoint rule:

```python
checkpoint_free = {"flat", "oracle_graph_v2"}
if self.condition in checkpoint_free:
    if any(
        value is not None
        for value in (
            self.graph_checkpoint_sha256,
            self.graph_initialization,
            self.graph_fraction,
            self.graph_seed,
        )
    ):
        raise ValueError(f"{self.condition} cache must not bind a Graph checkpoint")
else:
    if any(
        value is None
        for value in (
            self.graph_checkpoint_sha256,
            self.graph_initialization,
            self.graph_fraction,
            self.graph_seed,
        )
    ):
        raise ValueError("predicted Graph v2 cache must bind a Graph checkpoint")
```

Bind `TOKEN_SCHEMA_VERSION` in every cache provenance payload so a 75D v1 cache fails
before array loading.

- [ ] **Step 6: Re-run feature/cache tests**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add interaction_vla/graph_control/features.py \
  interaction_vla/graph_control/cache.py \
  tests/interaction_vla/graph_control/test_features.py \
  tests/interaction_vla/graph_control/test_cache.py
git commit -m "feat: cache causal stateful graph v2 tokens"
```

### Task 7: Implement the oracle-first ACT stage

**Files:**
- Modify: `interaction_vla/graph_control/config.py`
- Modify: `interaction_vla/graph_control/training.py`
- Modify: `interaction_vla/graph_control/rollout.py`
- Modify: `interaction_vla/graph_control/pipeline.py`
- Create: `configs/graph_v2_act_oracle_macos.yaml`
- Modify: `tests/interaction_vla/graph_control/test_config.py`
- Modify: `tests/interaction_vla/graph_control/test_training.py`
- Modify: `tests/interaction_vla/graph_control/test_rollout.py`
- Modify: `tests/interaction_vla/graph_control/test_pipeline.py`

- [ ] **Step 1: Write failing staged-condition and oracle-gate tests**

Assert the oracle config accepts exactly `flat, oracle_graph_v2`, requires a passing
ACT recovery report, trains byte-identical ACT initial states with identical epoch
order hashes, and computes:

```python
assert report["oracle_gate"] == {
    "passed": True,
    "success_delta": 0.10,
    "wrong_object_stable_grasp_delta": 0.0,
    "required_success_delta": 0.10,
}
```

for a fixture where Flat succeeds on 3/20 and oracle succeeds on 5/20 with no extra
wrong-object grasp. Assert 4/20 oracle success fails because the improvement is only
five percentage points.

- [ ] **Step 2: Run and verify failure**

```bash
.venv-lerobot/bin/python -m pytest \
  tests/interaction_vla/graph_control/test_config.py \
  tests/interaction_vla/graph_control/test_training.py \
  tests/interaction_vla/graph_control/test_rollout.py \
  tests/interaction_vla/graph_control/test_pipeline.py -q
```

Expected: FAIL because Graph control requires the old four conditions and oracle still
depends on a Reflect checkpoint.

- [ ] **Step 3: Permit only the two prespecified matrices**

Validate:

```python
if self.conditions not in {ORACLE_CONDITIONS, ALL_CONDITIONS}:
    raise ValueError("conditions must be the oracle pair or full Graph v2 matrix")
```

Add `required_recovery_report: Path` to `GraphControlConfig`; read it before inspect,
cache, train, or evaluate and require `passed is True` plus the exact 0.8/0.3 threshold
fields. Make `graph_runs_root` optional for the oracle matrix and required for the full
matrix. `graph_checkpoint` returns a path only for predicted conditions.

Permit the oracle evaluation cell and the full matrix without accepting arbitrary
layouts:

```python
if self.layouts not in {("normal",), ("normal", "crowded")}:
    raise ValueError("evaluation layouts must be normal or normal/crowded")
if self.object_counts not in {(2,), (2, 3)}:
    raise ValueError("evaluation object_counts must be [2] or [2, 3]")
if len(self.layouts) == 1 and self.object_counts != (2,):
    raise ValueError("oracle evaluation requires the normal two-object cell")
```

- [ ] **Step 4: Generalize cache/training pairing to configured conditions**

Replace global loops and exact-set checks with `config.conditions` or an explicit
`conditions` argument. `assert_paired_summaries` compares initial ACT hash, parameter
count, source rows, epoch order hashes, epochs, and extension decisions for the active
matrix. Bind `TOKEN_SCHEMA_VERSION`, `TOKEN_DIM=89`, feature names, cache hash, recovery
report hash, and condition into every ACT checkpoint.

- [ ] **Step 5: Make oracle rollout fully causal**

Replace `OracleCurrentTokenProvider` with `OracleGraphV2TokenProvider`. It owns a
`TCTIGTeacherExtractor` and a `CausalGraphV2Tracker`; `reset()` resets both, and each
`token()` call extracts only the current frame, updates the backward-only tracker, and
packs the resulting oracle target. It does not instantiate `FrozenGraphRuntime` and
does not read a sidecar `relation_goal`.

Predicted providers call `runtime.reset()` at episode reset and `predict_token()` once
per environment step. The rollout queue from the recovery plan must report
`queue_index=0` on every step.

- [ ] **Step 6: Implement dynamic aggregation and the oracle gate**

Aggregate only configured conditions. For the oracle matrix compute:

```python
success_delta = oracle["success_rate"] - flat["success_rate"]
wrong_delta = (
    oracle["wrong_object_stable_grasp_rate"]
    - flat["wrong_object_stable_grasp_rate"]
)
oracle_gate = {
    "passed": success_delta >= 0.10 and wrong_delta <= 0.0,
    "success_delta": success_delta,
    "wrong_object_stable_grasp_delta": wrong_delta,
    "required_success_delta": 0.10,
}
```

Persist `oracle_gate` in the paired evaluation report and use process exit status 1
when the scientific gate fails, while retaining the complete report.

- [ ] **Step 7: Create the oracle config**

```yaml
bridge_config: configs/lerobot_act_recovery_macos.yaml
required_recovery_report: outputs/graph_control/act_recovery/evaluation/recovery_report.json
split_manifest: outputs/graph_finetune/mujoco_graph_v2/split_manifest.json
graph_runs_root: null
conditions: [flat, oracle_graph_v2]
seeds: [0]
cache:
  directory: outputs/graph_control/graph_v2_oracle/cache
  batch_size: 1
training:
  output_dir: outputs/graph_control/graph_v2_oracle/runs
  smoke_steps: 1
  formal_epochs: 10
evaluation:
  layouts: [normal]
  object_counts: [2]
  cases_per_cell: 20
  master_seed: 2057736129
  max_steps: 180
```

- [ ] **Step 8: Re-run oracle stage tests**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add interaction_vla/graph_control/config.py \
  interaction_vla/graph_control/training.py \
  interaction_vla/graph_control/rollout.py \
  interaction_vla/graph_control/pipeline.py \
  configs/graph_v2_act_oracle_macos.yaml \
  tests/interaction_vla/graph_control
git commit -m "feat: gate graph v2 with an oracle ACT comparison"
```

### Task 8: Add predicted random/Reflect comparisons

**Files:**
- Modify: `interaction_vla/graph_control/config.py`
- Modify: `interaction_vla/graph_control/pipeline.py`
- Create: `configs/graph_v2_act_pilot_macos.yaml`
- Modify: `tests/interaction_vla/graph_control/test_pipeline.py`

- [ ] **Step 1: Write failing full-matrix tests**

Use three seed summaries and assert the report contains:

```text
oracle_graph_v2 - flat
predicted_random_v2 - flat
predicted_reflect_v2 - flat
predicted_reflect_v2 - predicted_random_v2
oracle_graph_v2 - predicted_reflect_v2
```

Assert `oracle_gap_recovered` is
`(predicted_reflect_success - flat_success) / (oracle_success - flat_success)` when the
denominator is positive, and `null` otherwise. Assert crowded cells are labeled
`training_distribution: ood`.

- [ ] **Step 2: Run and verify failure**

```bash
.venv-lerobot/bin/python -m pytest \
  tests/interaction_vla/graph_control/test_pipeline.py -q
```

Expected: FAIL because v1 contrast names and reports are fixed.

- [ ] **Step 3: Require the oracle report before predicted work**

For the full condition matrix, require
`outputs/graph_control/graph_v2_oracle/runs/evaluation/report.json` with
`oracle_gate.passed=true`. Bind its SHA-256 into caches, ACT checkpoints, and the final
report. Do not start Graph fine-tuning or predicted ACT training when it is false.
Add `required_oracle_report: Path | None` to `GraphControlConfig`; require `None` for
the oracle matrix and a real file for `ALL_CONDITIONS`. Parse YAML `null` as `None`
without passing it through `Path`.

- [ ] **Step 4: Produce full v2 contrasts and distribution labels**

Update aggregation to use the five contrasts above, preserve per-case paired deltas,
and compute `oracle_gap_recovered`. Label normal cells `id` and crowded cells `ood`
because the current demonstrations contain only normal layouts. Keep policy seed as
the independent replication unit.

- [ ] **Step 5: Create the full pilot config**

```yaml
bridge_config: configs/lerobot_act_recovery_macos.yaml
required_recovery_report: outputs/graph_control/act_recovery/evaluation/recovery_report.json
required_oracle_report: outputs/graph_control/graph_v2_oracle/runs/evaluation/report.json
split_manifest: outputs/graph_finetune/mujoco_graph_v2/split_manifest.json
graph_runs_root: outputs/graph_finetune/mujoco_graph_v2
conditions:
  - flat
  - oracle_graph_v2
  - predicted_random_v2
  - predicted_reflect_v2
seeds: [0, 1, 2]
cache:
  directory: outputs/graph_control/graph_v2_pilot/cache
  batch_size: 1
training:
  output_dir: outputs/graph_control/graph_v2_pilot/runs
  smoke_steps: 1
  formal_epochs: 10
evaluation:
  layouts: [normal, crowded]
  object_counts: [2, 3]
  cases_per_cell: 5
  master_seed: 2057736129
  max_steps: 180
```

- [ ] **Step 6: Re-run the full-matrix tests**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add interaction_vla/graph_control/config.py \
  interaction_vla/graph_control/pipeline.py \
  configs/graph_v2_act_pilot_macos.yaml \
  tests/interaction_vla/graph_control/test_pipeline.py
git commit -m "feat: compare predicted interaction graph v2 policies"
```

### Task 9: Verify the staged workflow and simplify README commands

**Files:**
- Modify: `README.md`
- Modify: `tests/interaction_vla/graph_control/test_cli.py`
- Modify: `tests/interaction_vla/graph_control/test_dataset.py`
- Modify: `tests/interaction_vla/graph_control/conftest.py`

- [ ] **Step 1: Remove remaining v1 assumptions from shared fixtures**

Update synthetic cache/token fixtures to allocate `TOKEN_DIM` rather than literal 75,
use `TOKEN_FEATURE_NAMES` rather than fabricated v1 names, and use
`ORACLE_CONDITIONS` or `ALL_CONDITIONS` rather than literal old condition strings.
Update CLI fixtures to point at `configs/graph_v2_act_oracle_macos.yaml`. Add these
explicit legacy assertions:

```python
with pytest.raises(ValueError, match="conditions.*Graph v2"):
    load_graph_control_config("configs/graph_control_act_pilot_macos.yaml")

assert GraphDatasetMetadata(base.meta).features[
    "observation.environment_state"
]["shape"] == [89]
```

Run:

```bash
.venv-lerobot/bin/python -m pytest \
  tests/interaction_vla/graph_control/test_cli.py \
  tests/interaction_vla/graph_control/test_dataset.py -q
```

Expected: PASS.

- [ ] **Step 2: Run the complete network-free suite**

```bash
HF_HOME=/tmp/gripper-mujoco-hf-cache \
  .venv-lerobot/bin/python -m pytest tests/interaction_vla -q
```

Expected: all tests pass with zero failures.

- [ ] **Step 3: Verify v1 incompatibility explicitly**

Run an automated test and a local command that attempt to load one v1 fine-tune
checkpoint and one 75D cache. Expected: both fail with a compact schema-version error,
not a tensor-size traceback. Confirm completed v1 JSON reports remain untouched.

- [ ] **Step 4: Prepare the v2 split**

```bash
.venv-lerobot/bin/python -m interaction_vla.graph_finetune split \
  --config configs/mujoco_graph_v2_finetune_macos.yaml
```

Expected: a leakage-free 40/5/5 episode split at
`outputs/graph_finetune/mujoco_graph_v2/split_manifest.json`.

- [ ] **Step 5: Run only the oracle stage first**

```bash
.venv-lerobot/bin/python -m interaction_vla.graph_control inspect \
  --config configs/graph_v2_act_oracle_macos.yaml

.venv-lerobot/bin/python -m interaction_vla.graph_control cache \
  --config configs/graph_v2_act_oracle_macos.yaml

.venv-lerobot/bin/python -m interaction_vla.graph_control smoke \
  --config configs/graph_v2_act_oracle_macos.yaml

.venv-lerobot/bin/python -m interaction_vla.graph_control compare \
  --config configs/graph_v2_act_oracle_macos.yaml

.venv-lerobot/bin/python -m interaction_vla.graph_control evaluate \
  --config configs/graph_v2_act_oracle_macos.yaml
```

Expected prerequisite for predicted Graph work: oracle success improves Flat by at
least 10 percentage points and wrong-object stable grasp does not increase.

- [ ] **Step 6: Run Graph fine-tuning only after the oracle gate**

```bash
.venv-lerobot/bin/python -m interaction_vla.graph_finetune inspect \
  --config configs/mujoco_graph_v2_finetune_macos.yaml

.venv-lerobot/bin/python -m interaction_vla.graph_finetune compare \
  --config configs/mujoco_graph_v2_finetune_macos.yaml
```

Expected: random and Reflect runs share splits, new-head initialization, loader order,
optimizer budget, and held-out rows; checkpoints report
`future_relation_goal_used=false`.

- [ ] **Step 7: Run predicted Graph v2 only after both gates**

```bash
.venv-lerobot/bin/python -m interaction_vla.graph_control inspect \
  --config configs/graph_v2_act_pilot_macos.yaml

.venv-lerobot/bin/python -m interaction_vla.graph_control cache \
  --config configs/graph_v2_act_pilot_macos.yaml

.venv-lerobot/bin/python -m interaction_vla.graph_control compare \
  --config configs/graph_v2_act_pilot_macos.yaml

.venv-lerobot/bin/python -m interaction_vla.graph_control evaluate \
  --config configs/graph_v2_act_pilot_macos.yaml
```

Expected: 3 seeds x 4 conditions, complete paired records, nonzero predicted control
before interpreting Reflect-versus-random, and explicit ID/OOD labels.

- [ ] **Step 8: Rewrite README around the actual next unpassed gate**

Keep the project description and completed experiment conclusions concise. Show only
the commands for the earliest unpassed stage. Move later commands under a short
“after this report passes” subsection and link the two implementation plans and the
v2 design.

- [ ] **Step 9: Commit verified documentation and fixture migration**

```bash
git add README.md \
  tests/interaction_vla/graph_control/test_cli.py \
  tests/interaction_vla/graph_control/test_dataset.py \
  tests/interaction_vla/graph_control/conftest.py
git commit -m "docs: document the gated interaction graph v2 workflow"
```

# MuJoCo Graph Fine-tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a paired random-initialization versus ReflectVLM-initialization experiment that learns a coordinate-invariant TC-TIG graph from LeRobot agent RGB, wrist RGB, 10D end-effector state, and language.

**Architecture:** A new `interaction_vla.graph_finetune` package aligns standard LeRobot frames with teacher sidecars at episode boundaries, derives training-only normalization, and exposes a multimodal graph estimator. The estimator preserves the ReflectVLM RGB/text/fusion dimensions so compatible weights transfer exactly, while MuJoCo-specific graph heads remain paired random controls. A config-driven pipeline trains both conditions on identical episode splits and reports held-out metric deltas.

**Tech Stack:** Python 3.12, NumPy, PyTorch, Pillow/TorchVision-free resizing, PyYAML, LeRobot 0.6.1, pytest.

---

### Task 1: Coordinate-invariant target and episode allocation

**Files:**
- Create: `interaction_vla/graph_finetune/__init__.py`
- Create: `interaction_vla/graph_finetune/schema.py`
- Create: `interaction_vla/graph_finetune/data.py`
- Create: `tests/interaction_vla/graph_finetune/__init__.py`
- Create: `tests/interaction_vla/graph_finetune/test_data.py`

- [ ] **Step 1: Write failing target and episode-split tests**

Create five manifest records with distinct episode seeds and synthetic
`tc_tig_teacher_v1` arrays. Assert the target selects relation channels 12 through 21
and never exposes pose, velocity, depth, segmentation, action, or success. Assert the
five-episode split contains 3/1/1 whole episodes and is deterministic.

```python
def test_semantic_target_selects_only_coordinate_invariant_channels() -> None:
    arrays = teacher_arrays(frames=3)
    target = semantic_targets(arrays)
    np.testing.assert_array_equal(
        target.relation_semantics,
        arrays["annotation.tc_tig.relation_values"][:, :, 12:22],
    )
    assert target.relation_semantics.shape == (3, 8, 10)


def test_episode_split_is_deterministic_and_never_splits_frames() -> None:
    records = manifest_records(5)
    first = split_episode_indices(records, seed=7, ratios=(0.6, 0.2, 0.2))
    second = split_episode_indices(records, seed=7, ratios=(0.6, 0.2, 0.2))
    assert first == second
    assert {name: len(value) for name, value in first.items()} == {
        "train": 3,
        "validation": 1,
        "test": 1,
    }
    assert set(first["train"]).isdisjoint(first["validation"] + first["test"])
```

- [ ] **Step 2: Run the tests and verify RED**

```bash
.venv/bin/python -m pytest tests/interaction_vla/graph_finetune/test_data.py -q
```

Expected: collection fails because `interaction_vla.graph_finetune` does not exist.

- [ ] **Step 3: Implement the schema and deterministic allocation**

Define this contract in `schema.py`:

```python
SCHEMA_VERSION = "mujoco_semantic_graph_v1"
SEMANTIC_CHANNELS = tuple(range(12, 22))

@dataclass(frozen=True)
class MuJoCoGraphTargets:
    entity_mask: np.ndarray              # [T, 6]
    entity_visibility: np.ndarray        # [T, 6, 2]
    relation_mask: np.ndarray            # [T, 8]
    relation_semantics: np.ndarray       # [T, 8, 10]
    goal_relation: np.ndarray            # [T]
    goal_operator: np.ndarray            # [T]
    goal_predicate: np.ndarray           # [T]
    goal_residual: np.ndarray            # [T]
```

Validate finite values, exact dtypes/shapes, binary masks, categorical ID ranges, and
frame-count agreement. Implement:

```python
def semantic_targets(arrays: Mapping[str, np.ndarray]) -> MuJoCoGraphTargets:
    values = np.asarray(
        arrays["annotation.tc_tig.relation_values"], dtype=np.float32
    )
    goals = np.asarray(arrays["annotation.tc_tig.relation_goal"], dtype=np.float32)
    return MuJoCoGraphTargets(
        entity_mask=np.asarray(
            arrays["annotation.tc_tig.entity_mask"], dtype=np.bool_
        ),
        entity_visibility=np.asarray(
            arrays["annotation.tc_tig.entity_visibility"], dtype=np.float32
        ),
        relation_mask=np.asarray(
            arrays["annotation.tc_tig.relation_mask"], dtype=np.bool_
        ),
        relation_semantics=values[:, :, SEMANTIC_CHANNELS],
        goal_relation=goals[:, 0].astype(np.int64),
        goal_operator=goals[:, 1].astype(np.int64),
        goal_predicate=goals[:, 2].astype(np.int64),
        goal_residual=goals[:, 3].astype(np.float32),
    )

def split_episode_indices(
    records: Sequence[Mapping[str, object]],
    *, seed: int, ratios: tuple[float, float, float],
) -> dict[str, list[int]]:
    if len(records) < 3:
        raise ValueError("episode split requires at least three episodes")
    ordered = sorted(
        records,
        key=lambda record: hashlib.sha256(
            f"{seed}:{record['episode_index']}:{record['seed']}".encode()
        ).digest(),
    )
    validation_count = max(1, int(np.floor(len(ordered) * ratios[1])))
    test_count = max(1, int(np.floor(len(ordered) * ratios[2])))
    train_count = len(ordered) - validation_count - test_count
    if train_count < 1:
        raise ValueError("split ratios leave no training episode")
    values = [int(record["episode_index"]) for record in ordered]
    return {
        "train": sorted(values[:train_count]),
        "validation": sorted(values[train_count : train_count + validation_count]),
        "test": sorted(values[train_count + validation_count :]),
    }

def select_training_fraction(
    episodes: Sequence[int], *, fraction: float, seed: int
) -> list[int]:
    if not episodes or not 0.0 < fraction <= 1.0:
        raise ValueError("fraction requires episodes and must lie within (0, 1]")
    ordered = sorted(
        (int(value) for value in episodes),
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).digest(),
    )
    return sorted(ordered[: max(1, math.ceil(len(ordered) * fraction))])
```

Hash `f"{seed}:{episode_index}:{teacher_seed}"` with SHA-256. Allocate counts while
keeping all partitions non-empty. Fraction selection hashes complete training episodes
and keeps `max(1, ceil(N*fraction))`.

- [ ] **Step 4: Verify GREEN**

Run the Task 1 test file and expect all tests to pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add interaction_vla/graph_finetune tests/interaction_vla/graph_finetune
git commit -m "feat: define MuJoCo semantic graph targets"
```

### Task 2: Aligned LeRobot/teacher corpus and training-only statistics

**Files:**
- Modify: `interaction_vla/graph_finetune/data.py`
- Modify: `tests/interaction_vla/graph_finetune/test_data.py`

- [ ] **Step 1: Write failing alignment and leakage tests**

Use a source whose samples contain agent/wrist tensors, a 10D state, language,
episode/frame indices, and a deliberately present `action`. Assert the dataset output
contains only model inputs plus graph labels. Assert sidecar frame 2 joins only to
source frame 2. Fit statistics with an extreme held-out episode and verify it cannot
alter training statistics or vocabulary.

```python
def test_dataset_aligns_teacher_by_episode_and_frame_and_drops_action() -> None:
    corpus = prepare_corpus(source(), records(), sidecars(), split_seed=3)
    dataset = MuJoCoGraphDataset(
        corpus.for_training_fraction(1.0, seed=9),
        partition="train", image_size=32, max_language_tokens=16,
    )
    sample = dataset[2]
    assert set(sample) == MODEL_BATCH_KEYS
    assert "action" not in sample
    assert sample["goal_relation"].item() == expected_goal(episode=0, frame=2)


def test_statistics_and_vocabulary_use_selected_training_episodes_only() -> None:
    corpus = prepare_corpus(
        source_with_extreme_test_episode(), records(), sidecars(), split_seed=3
    )
    prepared = corpus.for_training_fraction(0.5, seed=9)
    assert np.max(np.abs(prepared.normalization.state_mean)) < 10.0
    assert "heldoutword" not in prepared.vocabulary.token_to_id
```

- [ ] **Step 2: Verify RED**

Run Task 2 tests. Expected: missing corpus, normalization, and dataset APIs.

- [ ] **Step 3: Implement alignment, vocabulary, and normalization**

Reuse `graph_pretrain.reflectvlm.Vocabulary`, preserving the pretrained token order and
appending sorted tokens from selected MuJoCo training tasks. Add:

```python
@dataclass(frozen=True)
class GraphNormalization:
    state_mean: np.ndarray               # [10]
    state_std: np.ndarray                # [10]
    relation_mean: np.ndarray            # [8, 10]
    relation_std: np.ndarray             # [8, 10]
    residual_mean: float
    residual_std: float

@dataclass(frozen=True)
class PreparedMuJoCoCorpus:
    source: Any
    records: tuple[dict[str, object], ...]
    targets: dict[int, MuJoCoGraphTargets]
    tasks: dict[int, str]
    splits: dict[str, list[int]]
    pretrained_vocabulary: Vocabulary

    def for_training_fraction(self, fraction: float, seed: int) -> TrainingCorpus:
        selected = select_training_fraction(
            self.splits["train"], fraction=fraction, seed=seed
        )
        vocabulary = extend_vocabulary(
            self.pretrained_vocabulary,
            [self.tasks[episode] for episode in selected],
        )
        normalization = fit_normalization(
            self.source, self.targets, selected
        )
        return TrainingCorpus(
            corpus=self, selected_train_episodes=tuple(selected),
            vocabulary=vocabulary, normalization=normalization,
        )
```

Compute per-relation statistics from active training masks only and clamp every
standard deviation to at least `1e-4`. `MuJoCoGraphDataset` resizes RGB with
`torch.nn.functional.interpolate`, normalizes state/semantic/residual targets, and
returns only `MODEL_BATCH_KEYS`. Audit raw standard samples against an explicit
allowlist before extraction.

Real loading uses `LeRobotDataset(repo_id, root=root)` without action delta timestamps,
`load_teacher_sidecar` for SHA-verified targets, and the teacher manifest for episode
seeds and frame counts.

- [ ] **Step 4: Verify GREEN**

Run all data tests and expect them to pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add interaction_vla/graph_finetune/data.py tests/interaction_vla/graph_finetune/test_data.py
git commit -m "feat: align LeRobot frames with graph teachers"
```

### Task 3: Multimodal estimator, objective, and exact transfer

**Files:**
- Create: `interaction_vla/graph_finetune/model.py`
- Create: `tests/interaction_vla/graph_finetune/test_model.py`

- [ ] **Step 1: Write failing model and paired-transfer tests**

Assert output shapes `(B,6)`, `(B,6,2)`, `(B,8)`, `(B,8,10)`, `(B,8)`, `(B,5)`,
`(B,7)`, and `(B,)` for entity mask, visibility, relation mask, semantics, goal
relation/operator/predicate, and residual respectively.

Create a real `ReflectGraphEstimator` checkpoint, build paired MuJoCo models from the
same seed, and assert:

```python
assert torch.equal(pretrained.image_encoder[0].weight, reflect.image_encoder[0].weight)
assert not torch.equal(random.image_encoder[0].weight, reflect.image_encoder[0].weight)
assert torch.equal(random.state_encoder[0].weight, pretrained.state_encoder[0].weight)
assert torch.equal(random.relation_head.weight, pretrained.relation_head.weight)
assert transfer.copied_token_count > 0
```

Assert all component losses are finite, inactive slots do not affect regression loss,
backward produces gradients, and one optimizer update changes parameters.

- [ ] **Step 2: Verify RED**

```bash
.venv/bin/python -m pytest tests/interaction_vla/graph_finetune/test_model.py -q
```

Expected: collection fails because `model.py` is missing.

- [ ] **Step 3: Implement the estimator, loss, and transfer report**

Implement:

```python
class MuJoCoGraphEstimator(nn.Module):
    def forward(
        self, agent_rgb: Tensor, wrist_rgb: Tensor, state: Tensor,
        language_tokens: Tensor, language_mask: Tensor,
    ) -> dict[str, Tensor]:
        agent = self.image_encoder(agent_rgb)
        wrist = self.image_encoder(wrist_rgb)
        state_context = self.state_encoder(state)
        mask = language_mask.to(agent.dtype).unsqueeze(-1)
        words = self.token_embedding(language_tokens) * mask
        language = words.sum(1) / mask.sum(1).clamp_min(1.0)
        embedding = self.fusion(
            torch.cat(((agent + wrist) * 0.5 + state_context, language), dim=-1)
        )
        batch = agent_rgb.shape[0]
        return {
            "entity_mask_logits": self.entity_mask_head(embedding),
            "entity_visibility": self.entity_visibility_head(embedding).reshape(batch, 6, 2),
            "relation_mask_logits": self.relation_mask_head(embedding),
            "relation_semantics": self.relation_head(embedding).reshape(batch, 8, 10),
            "goal_relation_logits": self.goal_relation_head(embedding),
            "goal_operator_logits": self.operator_head(embedding),
            "goal_predicate_logits": self.predicate_head(embedding),
            "goal_residual": self.residual_head(embedding).squeeze(-1),
            "graph_embedding": embedding,
        }

def graph_finetune_loss(
    outputs: Mapping[str, Tensor], batch: Mapping[str, Tensor]
) -> dict[str, Tensor]:
    entity_mask = batch["entity_mask"].bool()
    relation_mask = batch["relation_mask"].bool()
    losses = {
        "entity_mask": F.binary_cross_entropy_with_logits(
            outputs["entity_mask_logits"], batch["entity_mask"].float()
        ),
        "entity_visibility": F.mse_loss(
            outputs["entity_visibility"][entity_mask],
            batch["entity_visibility"][entity_mask],
        ),
        "relation_mask": F.binary_cross_entropy_with_logits(
            outputs["relation_mask_logits"], batch["relation_mask"].float()
        ),
        "relation_semantics": F.smooth_l1_loss(
            outputs["relation_semantics"][relation_mask],
            batch["relation_semantics"][relation_mask],
        ),
        "goal_relation": F.cross_entropy(
            outputs["goal_relation_logits"], batch["goal_relation"]
        ),
        "goal_operator": F.cross_entropy(
            outputs["goal_operator_logits"], batch["goal_operator"]
        ),
        "goal_predicate": F.cross_entropy(
            outputs["goal_predicate_logits"], batch["goal_predicate"]
        ),
        "goal_residual": F.smooth_l1_loss(
            outputs["goal_residual"], batch["goal_residual"]
        ),
    }
    losses["total"] = torch.stack(tuple(losses.values())).sum()
    return losses

def initialize_paired_models(
    *, model_config: ModelConfig, vocabulary: Vocabulary,
    reflect_checkpoint: Path, seed: int,
) -> tuple[MuJoCoGraphEstimator, MuJoCoGraphEstimator, TransferReport]:
    torch.manual_seed(seed)
    random_model = MuJoCoGraphEstimator(model_config, len(vocabulary.tokens))
    pretrained_model = copy.deepcopy(random_model)
    payload = load_reflect_payload(reflect_checkpoint, model_config)
    report = transfer_reflect_weights(
        pretrained_model, vocabulary=vocabulary, payload=payload
    )
    return random_model, pretrained_model, report
```

Copy the three-convolution image encoder, fusion MLP, operator/predicate heads, and
matching token rows. Validate `reflect_semantic_graph_v1`, dimensions, tensor shapes,
and finite weights before modifying the paired model. Report immutable copied/skipped
tensor names and token counts.

- [ ] **Step 4: Verify GREEN**

Run model tests and then all graph-finetune tests; expect all to pass.

- [ ] **Step 5: Commit Task 3**

```bash
git add interaction_vla/graph_finetune/model.py tests/interaction_vla/graph_finetune/test_model.py
git commit -m "feat: transfer ReflectVLM into MuJoCo graph estimator"
```

### Task 4: Configuration, training, metrics, and paired comparison

**Files:**
- Create: `interaction_vla/graph_finetune/config.py`
- Create: `interaction_vla/graph_finetune/pipeline.py`
- Create: `tests/interaction_vla/graph_finetune/test_pipeline.py`

- [ ] **Step 1: Write failing config and end-to-end tests**

Load a temporary config with five synthetic episodes, fraction `1.0`, seed `0`, and
one epoch. Run a complete comparison and assert both checkpoints reload, use identical
row indices, preserve initialization labels, and produce finite metrics and deltas.

```python
report = compare_with_source(config, source(), records(), sidecars())
assert report["passed"] is True
assert report["paired_runs"] == 1
assert set(report["conditions"]) == {"random_init", "reflectvlm_init"}
assert report["runs"][0]["random_init"]["test_examples"] > 0
assert math.isfinite(report["runs"][0]["delta"]["goal_exact_accuracy"])
```

Add tests rejecting an incompatible teacher schema, missing ReflectVLM checkpoint,
non-finite loss, different condition row orders, and non-empty run directory.

- [ ] **Step 2: Verify RED**

Run the pipeline test. Expected: config and pipeline modules are missing.

- [ ] **Step 3: Implement configuration, training, and artifacts**

Define frozen `DatasetConfig`, `ModelConfig`, `TrainingConfig`, and
`GraphFinetuneConfig`. Use this smoke YAML contract:

```yaml
dataset:
  repo_id: local/franka_lerobot_act_smoke
  root: outputs/lerobot/franka_lerobot_act_smoke
  reflect_checkpoint: outputs/graph_pretrain/reflectvlm/checkpoint.pt
  split_seed: 17
  split_ratios: [0.6, 0.2, 0.2]
model:
  image_size: 128
  max_language_tokens: 32
  image_embedding_dim: 128
  text_embedding_dim: 64
  graph_embedding_dim: 128
training:
  output_dir: outputs/graph_finetune/mujoco_smoke
  device: auto
  batch_size: 8
  num_workers: 0
  epochs: 1
  learning_rate: 0.0003
  weight_decay: 0.0001
  fractions: [1.0]
  seeds: [0]
```

Implement atomic checkpoint/JSON publication, lowest-validation-loss selection,
checkpoint compatibility, and all design metrics. Create one `TrainingCorpus` per
fraction and reuse identical row order and loader seed for both conditions. Write
`split_manifest.json`, condition run artifacts, and aggregate mean/std plus paired
deltas in `comparison.json` without selecting a best seed.

- [ ] **Step 4: Verify GREEN**

Run pipeline and all graph-finetune tests; expect all to pass.

- [ ] **Step 5: Commit Task 4**

```bash
git add interaction_vla/graph_finetune/config.py interaction_vla/graph_finetune/pipeline.py tests/interaction_vla/graph_finetune/test_pipeline.py
git commit -m "feat: add paired graph fine-tuning pipeline"
```

### Task 5: JSON CLI, Mac configs, and commands

**Files:**
- Create: `interaction_vla/graph_finetune/cli.py`
- Create: `interaction_vla/graph_finetune/__main__.py`
- Create: `configs/mujoco_graph_finetune_smoke_macos.yaml`
- Create: `configs/mujoco_graph_finetune_pilot_macos.yaml`
- Create: `tests/interaction_vla/graph_finetune/test_cli.py`
- Modify: `tests/interaction_vla/graph_finetune/test_pipeline.py`
- Modify: `README.md`

- [ ] **Step 1: Write failing CLI/config tests**

Assert `inspect`, `compare`, and `evaluate` parse required arguments, runtime failures
return status 1 with one JSON object, and usage failures return status 2. Assert the
smoke config uses five-episode ratios, one fraction/seed, and one epoch; assert pilot
uses 0.8/0.1/0.1, three fractions/seeds, and the fifty-episode root.

```python
args = build_parser().parse_args([
    "evaluate", "--config", "config.yaml", "--checkpoint", "run/checkpoint.pt",
    "--partition", "test",
])
assert args.command == "evaluate"
assert args.partition == "test"
```

- [ ] **Step 2: Verify RED**

Run the CLI test. Expected: CLI/config files are missing.

- [ ] **Step 3: Implement CLI, configs, and README**

Follow the graph-pretrain JSON error boundary. Document these smoke commands:

```bash
.venv-lerobot/bin/python -m interaction_vla.graph_finetune inspect \
  --config configs/mujoco_graph_finetune_smoke_macos.yaml
.venv-lerobot/bin/python -m interaction_vla.graph_finetune compare \
  --config configs/mujoco_graph_finetune_smoke_macos.yaml
```

The pilot section first runs existing LeRobot `collect` and `validate` with
`configs/lerobot_act_pilot_macos.yaml`, then graph-finetune `inspect` and `compare`.
State that smoke and one language task do not prove transfer or language generalization.

- [ ] **Step 4: Verify GREEN**

Run all graph-finetune tests and CLI help in the LeRobot environment; expect success.

- [ ] **Step 5: Commit Task 5**

```bash
git add README.md configs/mujoco_graph_finetune_smoke_macos.yaml configs/mujoco_graph_finetune_pilot_macos.yaml interaction_vla/graph_finetune tests/interaction_vla/graph_finetune
git commit -m "docs: add MuJoCo graph fine-tuning commands"
```

### Task 6: Real smoke and final verification

**Files:**
- Modify only a Task 1-5 file when a failing regression test identifies a defect.

- [ ] **Step 1: Inspect the real local smoke dataset**

```bash
HF_HOME=/tmp/gripper-mujoco-hf-cache HF_HUB_OFFLINE=1 \
.venv-lerobot/bin/python -m interaction_vla.graph_finetune inspect \
  --config configs/mujoco_graph_finetune_smoke_macos.yaml
```

Expected: five episodes, 581 aligned frames, a 3/1/1 split, compatible transfer,
forbidden-input audit passed, and `passed=true`.

- [ ] **Step 2: Run the real paired smoke**

```bash
HF_HOME=/tmp/gripper-mujoco-hf-cache HF_HUB_OFFLINE=1 \
.venv-lerobot/bin/python -m interaction_vla.graph_finetune compare \
  --config configs/mujoco_graph_finetune_smoke_macos.yaml
```

Expected: both conditions finish, reload, evaluate the same held-out episode, and
write `outputs/graph_finetune/mujoco_smoke/comparison.json` with `passed=true`. Do not
interpret the smoke delta as scientific evidence.

- [ ] **Step 3: Run both test environments**

```bash
PYTHONPYCACHEPREFIX=/tmp/gripper-mujoco-pycache \
.venv/bin/python -m pytest tests/interaction_vla -q

HF_HOME=/tmp/gripper-mujoco-hf-cache \
PYTHONPYCACHEPREFIX=/tmp/gripper-mujoco-lerobot-pycache \
.venv-lerobot/bin/python -m pytest \
  tests/interaction_vla/lerobot_bridge \
  tests/interaction_vla/graph_pretrain \
  tests/interaction_vla/graph_finetune -q

uv pip check --python .venv-lerobot/bin/python
git diff --check
```

Expected: all tests pass, packages are compatible, and Git finds no whitespace errors.

- [ ] **Step 4: Audit the scientific report boundary**

Confirm `comparison.json` preserves both conditions, every fraction/seed, identical
test example counts, signed deltas, and no claim that smoke proves transfer. Confirm
checkpoints contain no teacher image/depth/segmentation arrays.

- [ ] **Step 5: Commit only implementation files**

Do not stage `MUJOCO_LOG.TXT`, physics data/checkpoints, LeRobot datasets, ReflectVLM
artifacts, or graph-finetune runtime outputs.

```bash
git status --short
git add README.md configs/mujoco_graph_finetune_smoke_macos.yaml configs/mujoco_graph_finetune_pilot_macos.yaml interaction_vla/graph_finetune tests/interaction_vla/graph_finetune
git commit -m "feat: compare ReflectVLM and random graph initialization"
```

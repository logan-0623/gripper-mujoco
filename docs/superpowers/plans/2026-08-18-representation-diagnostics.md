# Representation Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic `graph_control diagnose` command that analyzes the existing row-aligned Graph v2 caches on shared states and publishes provenance-bound per-feature, per-group, temporal, categorical, Teacher-distance, and episode-clustered uncertainty metrics.

**Architecture:** Pure numerical functions live in a new `interaction_vla/graph_control/diagnostics.py` module and accept NumPy arrays plus explicit episode boundaries. Configuration and cache/dataset loading remain in the existing config and pipeline modules; the CLI only parses `--partition` and delegates. Reports are atomically published beneath a new diagnostics output root and never mutate caches or checkpoints.

**Tech Stack:** Python 3.12, NumPy, PyTorch/LeRobot only for existing dataset loading, pytest, YAML, JSON/JSONL.

---

## File structure

- Create `interaction_vla/graph_control/diagnostics.py`: pure validation, metric, bootstrap, and report-composition functions.
- Modify `interaction_vla/graph_control/config.py`: strict optional diagnostics configuration.
- Modify `interaction_vla/graph_control/pipeline.py`: load aligned partition tokens and atomically publish diagnostics.
- Modify `interaction_vla/graph_control/cli.py`: expose `diagnose --partition`.
- Modify `configs/graph_v2_act_pilot_macos.yaml`: enable the local completed-pilot diagnostics output.
- Modify `configs/graph_v2_act_pilot_linux_cuda.yaml`: use the corresponding isolated CUDA diagnostics output.
- Create `tests/interaction_vla/graph_control/test_diagnostics.py`: pure numerical and report tests.
- Modify `tests/interaction_vla/graph_control/test_config.py`: diagnostics config validation.
- Modify `tests/interaction_vla/graph_control/test_pipeline.py`: aligned cache loading and atomic publication.
- Modify `tests/interaction_vla/graph_control/test_cli.py`: parser and dispatch coverage.

### Task 1: Strict diagnostics configuration

**Files:**
- Modify: `tests/interaction_vla/graph_control/test_config.py`
- Modify: `interaction_vla/graph_control/config.py`
- Modify: `configs/graph_v2_act_pilot_macos.yaml`
- Modify: `configs/graph_v2_act_pilot_linux_cuda.yaml`

- [ ] **Step 1: Write the failing config tests**

Add a diagnostics block to `_write_config` and assert exact parsing:

```python
diagnostics:
  output_dir: outputs/graph_control/graph_v2_oracle/diagnostics
  bootstrap_samples: 2000
  bootstrap_seed: 2057736129
  max_lag: 3
  active_epsilon: 1.0e-6
```

```python
assert config.diagnostics.output_dir == Path(
    "outputs/graph_control/graph_v2_oracle/diagnostics"
)
assert config.diagnostics.bootstrap_samples == 2000
assert config.diagnostics.bootstrap_seed == 2057736129
assert config.diagnostics.max_lag == 3
assert config.diagnostics.active_epsilon == 1.0e-6
```

Add parametrized invalid cases for zero bootstrap samples, negative seed/lag,
nonpositive epsilon, and an unknown diagnostics field. Add one backward-compatibility
test that removes the complete diagnostics block and asserts
`config.diagnostics is None`.

- [ ] **Step 2: Run the config tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/interaction_vla/graph_control/test_config.py -q
```

Expected: FAIL because `GraphControlConfig` has no `diagnostics` field and the strict
top-level parser rejects the block.

- [ ] **Step 3: Add the minimal config dataclass and parser**

Add:

```python
@dataclass(frozen=True)
class DiagnosticsConfig:
    output_dir: Path
    bootstrap_samples: int = 2000
    bootstrap_seed: int = 2057736129
    max_lag: int = 3
    active_epsilon: float = 1.0e-6

    def __post_init__(self) -> None:
        if self.bootstrap_samples < 1:
            raise ValueError("diagnostics.bootstrap_samples must be positive")
        if self.bootstrap_seed < 0:
            raise ValueError("diagnostics.bootstrap_seed must be non-negative")
        if self.max_lag < 0:
            raise ValueError("diagnostics.max_lag must be non-negative")
        if not np.isfinite(self.active_epsilon) or self.active_epsilon <= 0.0:
            raise ValueError("diagnostics.active_epsilon must be finite and positive")
```

Import NumPy in `config.py`, add
`diagnostics: DiagnosticsConfig | None = None` to `GraphControlConfig`, allow the
top-level `diagnostics` key, validate its five exact fields, and construct it only when
the block is present.

Add the approved block to the two pilot configs. The Linux/CUDA path must be
`outputs/graph_control/graph_v2_pilot_cuda/diagnostics`, not the macOS output.

- [ ] **Step 4: Run the config tests and verify GREEN**

Run the same pytest command. Expected: all config tests PASS.

- [ ] **Step 5: Commit the config slice**

```bash
git add interaction_vla/graph_control/config.py \
  tests/interaction_vla/graph_control/test_config.py \
  configs/graph_v2_act_pilot_macos.yaml \
  configs/graph_v2_act_pilot_linux_cuda.yaml
git commit -m "feat: configure graph representation diagnostics"
```

### Task 2: Episode-safe alignment and descriptive metrics

**Files:**
- Create: `tests/interaction_vla/graph_control/test_diagnostics.py`
- Create: `interaction_vla/graph_control/diagnostics.py`

- [ ] **Step 1: Write failing tests for aligned inputs and descriptive metrics**

Define synthetic rows with two episodes and assert that validation accepts only
contiguous zero-based frame indices within each episode:

```python
layout = validate_episode_layout(
    row_indices=np.array([4, 5, 9, 10, 11]),
    episode_indices=np.array([2, 2, 7, 7, 7]),
    frame_indices=np.array([0, 1, 0, 1, 2]),
)
assert layout.episode_slices == (slice(0, 2), slice(2, 5))
assert layout.episode_ids == (2, 7)
```

Reject duplicate rows, changing episode order, nonzero first frames, frame gaps, and
nonfinite/wrong-shaped tokens.

For values `[-1, 0, 1, 2]`, assert count, mean, population standard deviation,
min/max, NumPy linear quantiles p05/p25/p50/p75/p95, robust range, active fraction,
and both saturation fractions from `feature_distribution`.

- [ ] **Step 2: Run the focused tests and verify RED**

```bash
.venv/bin/python -m pytest \
  tests/interaction_vla/graph_control/test_diagnostics.py -q
```

Expected: collection ERROR because the diagnostics module does not exist.

- [ ] **Step 3: Implement validation and descriptive metrics**

Create immutable `EpisodeLayout` and these public pure functions:

```python
DIAGNOSTICS_SCHEMA_VERSION = "graph_representation_diagnostics_v1"

@dataclass(frozen=True)
class EpisodeLayout:
    row_indices: np.ndarray
    episode_indices: np.ndarray
    frame_indices: np.ndarray
    episode_ids: tuple[int, ...]
    episode_slices: tuple[slice, ...]

def validate_episode_layout(...) -> EpisodeLayout: ...
def validate_tokens(tokens: object, *, rows: int) -> np.ndarray: ...
def feature_distribution(values: object, *, active_epsilon: float) -> dict[str, float | int]: ...
```

Use float64 internally for metrics, `np.quantile(..., method="linear")`, population
standard deviation (`ddof=0`), and JSON-safe Python scalars. Freeze copies stored in
`EpisodeLayout` to prevent accidental mutation.

- [ ] **Step 4: Run the diagnostics tests and verify GREEN**

Run the focused pytest command. Expected: PASS.

- [ ] **Step 5: Commit the numerical foundation**

```bash
git add interaction_vla/graph_control/diagnostics.py \
  tests/interaction_vla/graph_control/test_diagnostics.py
git commit -m "feat: validate aligned graph diagnostic inputs"
```

### Task 3: Temporal, categorical, and effective-rank metrics

**Files:**
- Modify: `tests/interaction_vla/graph_control/test_diagnostics.py`
- Modify: `interaction_vla/graph_control/diagnostics.py`

- [ ] **Step 1: Write failing temporal tests**

Use two episodes where the boundary jump is large and assert it is excluded:

```python
tokens = np.array([[0.0], [1.0], [100.0], [102.0], [105.0]])
metrics = temporal_feature_metrics(tokens[:, 0], layout)
assert metrics == {
    "first_difference_count": 3,
    "first_difference_mae": 2.0,
    "first_difference_rms": pytest.approx(np.sqrt(14.0 / 3.0)),
    "second_difference_count": 1,
    "second_difference_mae": 1.0,
}
```

Add categorical tests with zero-vector missing frames. Assert valid-frame entropy,
flip numerator/denominator/rate, mean dwell length, and false flips defined as a
predicted label change while the Teacher label stays unchanged on the same valid
transition.

Add effective-rank tests: constant group returns `0.0`, one varying dimension returns
`1.0`, and two equal independent dimensions return approximately `2.0`.

- [ ] **Step 2: Run the focused tests and verify RED**

Expected: FAIL because the new metric functions are absent.

- [ ] **Step 3: Implement the metrics**

Add:

```python
def temporal_feature_metrics(values: object, layout: EpisodeLayout) -> dict[str, float | int]: ...
def categorical_sequence_metrics(
    probabilities: object,
    layout: EpisodeLayout,
    *,
    teacher_probabilities: object | None = None,
) -> dict[str, float | int | None]: ...
def covariance_effective_rank(values: object, *, epsilon: float = 1.0e-12) -> float: ...
```

Treat a categorical row as valid only when its finite probability sum is strictly
positive. Normalize valid rows before entropy. Count transitions only when both
adjacent rows are valid and lie in the same episode. Define dwell segments separately
per episode.

- [ ] **Step 4: Run focused tests and verify GREEN**

- [ ] **Step 5: Commit the temporal metrics**

```bash
git add interaction_vla/graph_control/diagnostics.py \
  tests/interaction_vla/graph_control/test_diagnostics.py
git commit -m "feat: measure graph temporal consistency"
```

### Task 4: Teacher distances and bounded lag correlation

**Files:**
- Modify: `tests/interaction_vla/graph_control/test_diagnostics.py`
- Modify: `interaction_vla/graph_control/diagnostics.py`

- [ ] **Step 1: Write failing Teacher-comparison tests**

Test exact vector distances, cosine missingness for zero vectors, and a predicted
sequence delayed by one step. Assert the best correlation lag is `1`, correlation is
`1.0`, and no pair crosses an episode boundary. Assert a constant feature returns
`lag_zero=None`, `best_lag=None`, and `best_correlation=None`.

- [ ] **Step 2: Run focused tests and verify RED**

- [ ] **Step 3: Implement comparison functions**

Add:

```python
def teacher_distance_metrics(
    predicted: object, teacher: object
) -> dict[str, float | int | None]: ...

def lagged_feature_correlation(
    predicted: object,
    teacher: object,
    layout: EpisodeLayout,
    *,
    max_lag: int,
) -> dict[str, float | int | None]: ...
```

Use the convention that positive lag means the predicted sequence follows the
Teacher sequence. Compute Pearson correlation only with at least two aligned pairs and
nonzero variance. Choose the best lag by highest correlation, then smallest absolute
lag, then negative before positive for a remaining tie.

- [ ] **Step 4: Run focused tests and verify GREEN**

- [ ] **Step 5: Commit Teacher comparisons**

```bash
git add interaction_vla/graph_control/diagnostics.py \
  tests/interaction_vla/graph_control/test_diagnostics.py
git commit -m "feat: compare predicted and teacher graph tokens"
```

### Task 5: Deterministic episode-clustered bootstrap

**Files:**
- Modify: `tests/interaction_vla/graph_control/test_diagnostics.py`
- Modify: `interaction_vla/graph_control/diagnostics.py`

- [ ] **Step 1: Write failing bootstrap tests**

Use episode values `{2: 0.0, 7: 1.0, 9: 2.0}` and assert two calls with the same
seed return byte-equal dictionaries, the estimate is `1.0`, the interval is ordered,
and changing `bootstrap_samples` is reflected in the report. Reject empty values,
nonfinite values, zero samples, and negative seeds.

- [ ] **Step 2: Run focused tests and verify RED**

- [ ] **Step 3: Implement clustered bootstrap**

```python
def cluster_bootstrap_mean(
    episode_values: Mapping[int, float],
    *,
    samples: int,
    seed: int,
) -> dict[str, float | int]: ...
```

Use `np.random.default_rng(seed)`, resample episode identifiers with replacement,
compute one mean per replicate, and use linear 2.5/97.5 percentiles. Preserve the
point estimate from the unresampled episodes.

- [ ] **Step 4: Run focused tests and verify GREEN**

- [ ] **Step 5: Commit uncertainty metrics**

```bash
git add interaction_vla/graph_control/diagnostics.py \
  tests/interaction_vla/graph_control/test_diagnostics.py
git commit -m "feat: add clustered graph metric intervals"
```

### Task 6: Compose complete condition and episode reports

**Files:**
- Modify: `tests/interaction_vla/graph_control/test_diagnostics.py`
- Modify: `interaction_vla/graph_control/diagnostics.py`

- [ ] **Step 1: Write failing report-composition tests**

Construct 89D synthetic Teacher and predicted tokens using `TOKEN_SLICES`. Assert:

- all 89 `TOKEN_FEATURE_NAMES` appear in ordered feature reports;
- all 12 named slices appear in group reports;
- categorical groups contain entropy/flip metrics;
- predicted groups contain Teacher distances while Teacher groups do not describe
  self-distance as estimator error;
- every aggregate with episode values contains a clustered interval;
- the per-episode output contains condition, estimator seed, episode, frames, feature
  metrics, and group metrics;
- flat is summarized but excluded from Teacher-distance and lag comparisons.

- [ ] **Step 2: Run focused tests and verify RED**

- [ ] **Step 3: Implement report composition**

Add explicit group sets and one top-level pure function:

```python
CATEGORICAL_GROUPS = ("phase", "next_relation", "relation_operator", "predicate")
HARD_BINARY_GROUPS = ("entity_presence", "relation_presence")

def build_representation_diagnostics(
    *,
    condition_tokens: Mapping[tuple[int, str], np.ndarray],
    teacher_tokens: np.ndarray,
    layout: EpisodeLayout,
    partition: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
    max_lag: int,
    active_epsilon: float,
) -> tuple[dict[str, object], list[dict[str, object]]]: ...
```

Require the complete active condition matrix for every estimator seed. Deduplicate
Teacher statistics and preserve predicted estimator seeds separately. Derive distinct
bootstrap sub-seeds from `(base seed, condition, estimator seed, metric path)` using
SHA-256 rather than Python's randomized `hash()`.

- [ ] **Step 4: Run focused tests and verify GREEN**

- [ ] **Step 5: Refactor repeated feature/group traversal while tests stay green**

- [ ] **Step 6: Commit report composition**

```bash
git add interaction_vla/graph_control/diagnostics.py \
  tests/interaction_vla/graph_control/test_diagnostics.py
git commit -m "feat: compose graph representation diagnostic reports"
```

### Task 7: Pipeline loading and atomic publication

**Files:**
- Modify: `tests/interaction_vla/graph_control/test_pipeline.py`
- Modify: `interaction_vla/graph_control/pipeline.py`

- [ ] **Step 1: Write failing pipeline tests**

Add tests that monkeypatch `_context` and `_load_cache_matrix` with a synthetic source,
split, and `TokenCache` matrix. Assert:

- requesting `test` selects only test rows in split order;
- cache row mismatch raises before metrics run;
- missing diagnostics config raises an actionable error;
- an existing nonempty output directory fails before loading context;
- report and JSONL publication is atomic when report writing fails;
- success writes `report.json` and ordered `per_episode.jsonl` and returns their paths.

The fake source exposes only:

```python
source.hf_dataset = {
    "episode_index": [0, 0, 1, 1],
    "frame_index": [0, 1, 0, 1],
}
```

- [ ] **Step 2: Run pipeline tests and verify RED**

```bash
.venv/bin/python -m pytest \
  tests/interaction_vla/graph_control/test_pipeline.py -q
```

Expected: FAIL because `diagnose_from_config` is absent.

- [ ] **Step 3: Implement aligned loading and publication**

In `pipeline.py`, add:

```python
def diagnose_from_config(
    path: str | Path, *, partition: str | None = None
) -> dict[str, object]: ...
```

Validate the partition against `train`, `validation`, and `test`; default to `test`.
Preflight `<diagnostics.output_dir>/<partition>` before `_context`. Use the split row
list as the canonical ordering, require every cache to contain exactly those rows,
extract episode/frame columns without decoding images, call
`build_representation_diagnostics`, and publish with `_atomic_output_directory` plus
`_write_json_atomic`. Write every per-episode record with sorted JSON keys and `fsync`.

Bind the final report to config path, partition, split SHA-256, cache SHA-256 values,
dataset fingerprint, token schema/dimension, diagnostics schema, source fingerprint,
bootstrap controls, output paths, row count, and episode count.

- [ ] **Step 4: Run pipeline tests and verify GREEN**

- [ ] **Step 5: Commit the diagnostics pipeline**

```bash
git add interaction_vla/graph_control/pipeline.py \
  tests/interaction_vla/graph_control/test_pipeline.py
git commit -m "feat: publish graph representation diagnostics"
```

### Task 8: CLI command and JSON behavior

**Files:**
- Modify: `tests/interaction_vla/graph_control/test_cli.py`
- Modify: `interaction_vla/graph_control/cli.py`

- [ ] **Step 1: Write failing CLI tests**

Extend the parser test with:

```python
parsed = parser.parse_args(
    ["diagnose", "--config", "config.yaml", "--partition", "validation"]
)
assert parsed.command == "diagnose"
assert parsed.partition == "validation"
```

Monkeypatch `diagnose_from_config`, invoke `main`, and assert it receives
`Path("config.yaml")` and `partition="validation"`. Assert an invalid partition is a
single JSON `CLIUsageError` with exit code 2.

- [ ] **Step 2: Run CLI tests and verify RED**

- [ ] **Step 3: Implement CLI parsing and dispatch**

Import `diagnose_from_config`. Add a dedicated parser with choices
`train`, `validation`, and `test`, default `None`, while leaving existing commands
unchanged. Dispatch diagnose explicitly because it receives a keyword partition:

```python
if args.command == "diagnose":
    return diagnose_from_config(args.config, partition=args.partition)
```

- [ ] **Step 4: Run CLI tests and verify GREEN**

- [ ] **Step 5: Commit the CLI**

```bash
git add interaction_vla/graph_control/cli.py \
  tests/interaction_vla/graph_control/test_cli.py
git commit -m "feat: expose graph diagnostics command"
```

### Task 9: Run the existing cache diagnostic and verify the artifact

**Files:**
- No production file changes unless verification exposes a defect.

- [ ] **Step 1: Run focused tests in the base environment**

```bash
.venv/bin/python -m pytest \
  tests/interaction_vla/graph_control/test_diagnostics.py \
  tests/interaction_vla/graph_control/test_config.py \
  tests/interaction_vla/graph_control/test_pipeline.py \
  tests/interaction_vla/graph_control/test_cli.py -q
```

Expected: PASS.

- [ ] **Step 2: Run the complete graph-control suite in LeRobot**

```bash
.venv-lerobot/bin/python -m pytest tests/interaction_vla/graph_control -q
```

Expected: PASS.

- [ ] **Step 3: Run diagnostics on the completed pilot cache**

```bash
.venv-lerobot/bin/python -m interaction_vla.graph_control diagnose \
  --config configs/graph_v2_act_pilot_macos.yaml \
  --partition test
```

Expected: JSON with `passed: true`, `rows: 639`, `episodes: 5`, and report paths below
`outputs/graph_control/graph_v2_pilot/diagnostics/test/`.

- [ ] **Step 4: Verify deterministic numerical output without overwriting**

Load `report.json` twice in a read-only Python command and assert every number is
finite, all three predicted estimator seeds are present, Teacher statistics occur
once, and all 89 features/12 groups are present. The command itself must refuse a
second write to the completed output directory.

- [ ] **Step 5: Record verification evidence in the final handoff**

Include exact test counts, the diagnostic artifact path, and the top-level integrity
summary. Do not interpret metric values as scientific support until the report is
reviewed separately.

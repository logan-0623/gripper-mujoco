# Physics Evaluation Scope Design

## Objective

Make the normal seed-0 comparison evaluate only the two representations the
user is currently testing: Flat and Graph. Keep Graph edge shuffling available
as an explicit structural ablation, and allow a smaller case count for fast
iteration without editing the experiment YAML.

## Root Cause

`evaluate_from_config` currently hardcodes three rollouts per evaluation case:
Flat, Graph, and Graph with shuffled edges. The pilot config creates 160 cases,
so selecting model seed 0 still runs `160 * 3 = 480` rollouts. The
`--model-seeds` option limits checkpoint seeds only; it does not limit policy
variants.

## Considered Approaches

1. Delete edge-shuffle evaluation. This makes the default smaller but removes a
   useful future ablation and breaks existing report workflows.
2. Make edge shuffle opt-in and add a CLI case-count override. This is the
   recommended design because the everyday command becomes Flat versus Graph,
   while the structural ablation remains reproducible when requested.
3. Add a separate quick-evaluation YAML. This avoids new CLI arguments but
   duplicates the pilot configuration and makes it easier for the quick config
   to drift.

The implementation will use approach 2.

## Library Interface

`evaluate_from_config` gains two quiet, explicit options:

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

`make_physics_evaluation_cases` gains the same optional
`episodes_per_count` override. `None` uses `config.eval.episodes_per_count`;
an integer must be positive. The override changes only the evaluation seed
count. It does not modify the loaded config, training data, checkpoints,
physics parameters, case groups, object counts, or deterministic seed formula.

## CLI Interface

The existing command defaults to Flat and Graph only:

```bash
.venv/bin/python -m interaction_vla.physics_evaluate \
  --config configs/physics_recovery_pilot_macos.yaml \
  --model-seeds 0
```

With the unchanged pilot YAML, this runs 160 cases for two policies, or 320
rollouts. A fast comparison uses five episodes per condition/object-count pair:

```bash
.venv/bin/python -m interaction_vla.physics_evaluate \
  --config configs/physics_recovery_pilot_macos.yaml \
  --model-seeds 0 \
  --episodes-per-count 5
```

There are eight condition/object-count pairs, so this runs `8 * 5 * 2 = 80`
rollouts. The optional ablation is enabled explicitly:

```bash
.venv/bin/python -m interaction_vla.physics_evaluate \
  --config configs/physics_recovery_pilot_macos.yaml \
  --model-seeds 0 \
  --episodes-per-count 5 \
  --include-edge-shuffle
```

That command runs `8 * 5 * 3 = 120` rollouts. All variants remain on the same
paired initial states.

## Evaluation and Report Flow

The evaluator builds `policy_variants` dynamically as `("flat", "graph")` or
`("flat", "graph", "graph_edge_shuffle")`. The tqdm total is the exact product
of case count, selected model-seed count, and policy-variant count. Graph edge
shuffle rollouts execute only when requested.

`evaluation_scope` records:

- selected model seeds;
- resolved `episodes_per_count`;
- case count;
- rollout count;
- the exact policy-variant list.

For schema compatibility, `graph_vs_edge_shuffle` remains in the report. When
the ablation is disabled, its `by_model_seed` mapping is empty. `graph_vs_flat`
is always populated for a valid Flat/Graph evaluation.

As before, each run replaces `evaluation/episodes.csv` and
`evaluation/report.json`; the report therefore always describes the most recent
scope. No data recollection or checkpoint retraining is required.

## Error Handling

`episodes_per_count` values below one raise `ValueError` before any checkpoint
loading or rollout begins. Existing model-seed, checkpoint provenance, dataset
provenance, and paired-initial-state validation remain unchanged. Progress bars
still close through `finally` when a rollout raises.

## Testing

Tests will prove that:

- the default evaluator runs exactly Flat and Graph and reports two variants;
- enabling edge shuffle restores the third rollout and ablation metrics;
- `episodes_per_count=5` creates 40 pilot cases and therefore 80 default
  seed-0 rollouts;
- non-positive overrides fail before evaluation;
- CLI arguments are forwarded correctly;
- tqdm totals and report scope exactly match the executed rollout count;
- the full regression suite and compilation pass.

## Non-goals

- Changing policy architecture, training, checkpoints, success criteria, IK,
  physics, or recovery data.
- Claiming end-to-end Graph superiority from the current zero-success report.
- Removing edge-shuffle support or changing the pilot YAML defaults.

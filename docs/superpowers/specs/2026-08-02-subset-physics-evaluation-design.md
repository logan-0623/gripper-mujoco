# Subset Physics Evaluation Design

## Goal

Allow a quick, explicitly scoped physical comparison using only selected trained model seeds while preserving the experiment configuration and checkpoint provenance. The immediate target is seed 0 with Flat, Graph, and Graph edge-shuffle evaluation.

## Command

```bash
.venv/bin/python -m interaction_vla.physics_evaluate \
  --config configs/physics_recovery_pilot_macos.yaml \
  --model-seeds 0
```

`--model-seeds` accepts one or more integer seeds. Omitting it retains the configured full evaluation over all seeds.

## Evaluation Scope

For each selected seed, evaluate every configured case with:

1. Flat
2. Graph
3. Graph with valid edge assignments shuffled

With the current recovery-pilot configuration, seed 0 therefore runs 200 cases × 3 policy variants = 600 rollouts.

## Provenance and Validation

- The CLI seed filter does not modify the YAML file, so config, gate, dataset, and checkpoint provenance remain unchanged.
- Selected seeds must be unique, non-empty, and present in `config.train.model_seeds`.
- Before the first rollout, load and validate every selected Flat and Graph checkpoint, including backend, feature schema, model identity, expert-gate hashes, and training-dataset provenance.
- Any missing or stale selected checkpoint fails immediately. Unselected seeds are neither required nor inspected.

## Progress Reporting

The CLI enables one global epoch-style `tqdm` bar over rollouts. Its total is computed before evaluation. The postfix reports the active seed, policy variant, condition, object count, and most recent success result; tqdm provides throughput and ETA.

Library calls remain quiet by default through a `show_progress=False` option.

## Output

Continue writing `evaluation/episodes.csv` and `evaluation/report.json`. Add an `evaluation_scope` object to the report containing:

- selected model seeds;
- number of configured cases;
- number of completed rollouts;
- enabled policy variants.

This makes a seed-0 exploratory report distinguishable from a full three-seed experiment. A later complete evaluation may intentionally overwrite these files with a new scope record.

## Compatibility and Testing

- Existing calls without seed selection preserve full configured behavior.
- Test CLI parsing for `--model-seeds 0`.
- Test subset evaluation requires only the selected checkpoints and reports the selected scope.
- Test checkpoint preflight occurs before any rollout.
- Test progress total and per-rollout updates.
- Run the full project suite and Python compilation.

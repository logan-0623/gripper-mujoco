# Physics CLI Progress Design

## Objective

Ensure every command in the seed-0 physical recovery workflow displays useful
`tqdm` progress by default when invoked through its CLI:

1. expert validation;
2. base and recovery data collection;
3. Flat/Graph training;
4. paired physical evaluation.

Training and evaluation already provide progress bars. This change adds matching
behavior to expert validation and physical data collection without changing
simulation, data, gate, checkpoint, or evaluation semantics.

## Interface

Library functions remain quiet by default for tests and programmatic use:

```python
validate_expert_from_config(..., show_progress=False)
collect_from_config(..., show_progress=False)
```

Their CLI `main()` functions pass `show_progress=True`. Existing commands and
arguments do not change, and no additional flag is required.

All bars use `tqdm.auto.tqdm`, dynamic terminal width, and stderr output. Every
bar is closed through `finally`, including validation failures and collection
exceptions.

## Expert Gate Progress

Expert validation creates one bar:

```text
expert gate: 40/40 cases
```

The total equals the deterministic validation case count. The bar advances once
after every completed `run_validation_case`, whether that case succeeds or
fails. Its postfix contains:

- `condition`: normal or crowded;
- `objects`: object count;
- `seed`: environment seed;
- `success`: `0` or `1`;
- `passed`: cumulative successful case count;
- `failed`: cumulative failed case count.

The final gate report, threshold, exit status, hashes, and episode ordering are
unchanged.

## Physical Data Collection Progress

Collection uses two sequential bars.

### Base demonstrations

```text
base data: 50/50 episodes
```

The total is `config.train.episodes`. It advances only when a successful base
episode is accepted and written to the manifest, because the user is waiting for
50 usable demonstrations rather than a fixed number of attempts. Its postfix is
updated after every attempt and contains:

- `attempts`;
- `accepted`;
- `rejected`;
- `objects`;
- `reason`.

Rejected attempts therefore remain visible even though they do not advance the
accepted-episode bar. If the maximum attempt limit is reached, the bar closes
before the existing `RuntimeError` is raised.

### Post-grasp recovery

When recovery is enabled, collection creates a second bar:

```text
recovery: 120/120 attempts
```

Its total is the number of training-split source episodes multiplied by
`recovery.variants_per_episode`. It advances once for every recovery attempt,
including accepted trajectories, explicit `PhysicsRecoveryRejected` cases, and
episodes that finish without task success. Its postfix contains:

- `kind`;
- `source_seed`;
- `accepted`;
- `rejected`;
- `reason`.

Validation/test source seeds are excluded before the total is computed. The bar
does not appear when recovery is disabled.

## Existing Training and Evaluation Bars

No behavior change is made to:

- training: one epoch-level bar per representation/model seed with MSE postfix;
- evaluation: one global rollout bar with seed, policy variant, condition,
  object count, and success postfix.

The README describes all four workflow commands as automatically showing
progress.

## Testing

Tests inject lightweight tqdm spies and run no long physical rollout. They prove:

- expert validation creates a bar with the exact case total, advances once per
  result, publishes success/failure counts, and closes;
- base collection advances only for accepted episodes but updates postfix after
  rejected attempts;
- recovery collection totals training sources times three, advances on both
  acceptance and rejection, and closes;
- default library calls remain quiet;
- both CLI entry points pass `show_progress=True`;
- existing train/evaluation progress tests continue to pass;
- full regression and compilation pass.

## Non-goals

- simulation-step progress updates;
- nested per-episode substep bars;
- changing retry limits or rejection behavior;
- changing training/evaluation progress semantics;
- adding a new command-line flag;
- changing provenance hashes solely to ignore these source changes. The normal
  provenance system remains authoritative, so modifying validation or collection
  source may make an existing expert gate stale.

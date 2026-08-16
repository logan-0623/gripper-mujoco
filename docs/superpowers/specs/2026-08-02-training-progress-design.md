# Training Progress Bar Design

## Goal

Show one concise epoch-level `tqdm` progress bar for every CLI training run so Flat and Graph experiments expose their progress, current training MSE, throughput, and estimated remaining time.

## Behavior

- `python -m interaction_vla.train` enables the progress bar by default.
- The bar description identifies the representation and model seed, for example `graph seed=0`.
- The total is the number of epochs requested for the current invocation.
- After every epoch, the postfix reports that epoch's weighted training MSE.
- Resume runs show progress for the additional epochs requested by that invocation while preserving the checkpoint's cumulative epoch and step counters.
- The final checkpoint path remains the CLI's normal stdout result; `tqdm` continues to use stderr.

## API and Dependencies

- Add `tqdm>=4.66` to `requirements-macos.txt`.
- Add a `show_progress` option to `train_policy`, defaulting to `False` so library use and tests remain quiet.
- `train_from_config` passes `show_progress=True`, making the existing CLI commands display progress without requiring a new flag.

## Compatibility

The progress wrapper must not change data ordering, epoch seeding, optimization steps, metrics JSONL content, checkpoints, training provenance, or numerical results. It only observes completed epochs and displays the already-computed weighted MSE.

## Verification

- A focused test verifies that enabling progress creates an epoch bar with the correct description, total, and MSE postfix.
- Existing deterministic resume and uninterrupted-training tests must continue to pass.
- The complete `tests/interaction_vla` suite and Python compilation must pass.

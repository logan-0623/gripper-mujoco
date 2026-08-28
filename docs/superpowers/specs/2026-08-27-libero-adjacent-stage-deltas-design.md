# LIBERO Adjacent-Stage Paired Delta Design

## Goal

Extend the formal SmolVLA Probe v2 report with paired bootstrap confidence
intervals for consecutive training stages, without invalidating or recomputing
the existing probe-cell cache.

## Evidence semantics

The report keeps its existing absolute-reference comparisons:

- `pretrained -> sft_25`
- `pretrained -> sft_50`
- `pretrained -> sft_100`

It additionally records the preregistered adjacent trajectory:

- `pretrained -> sft_25`
- `sft_25 -> sft_50`
- `sft_50 -> sft_100`

Both comparison families use the same held-out states, group identities,
factor targets, normalization scales, label universes, and matched probe seeds.
The primary task-group split bootstraps tasks; the secondary episode-group split
bootstraps episodes. Geometry retains its lower-is-better sign conversion.

Adjacent comparisons distinguish three longitudinal patterns that comparisons
against Pretrained alone cannot resolve:

- continued strengthening;
- saturation or preservation;
- weakening or reorganization.

A confidence interval containing zero is reported as inconclusive rather than
as evidence of exact saturation.

## Artifact contract

The existing Probe v2 report remains the single source of truth and preserves
all current fields. An additive enrichment step writes:

- `adjacent_stage_deltas` for the task-group split;
- `secondary_adjacent_stage_deltas` for the episode-group split;
- `adjacent_stage_pairs` with the fixed ordered comparisons;
- `adjacent_stage_delta_interpretation` describing metric direction;
- `adjacent_stage_delta_schema_version`;
- `adjacent_stage_delta_analysis_sha256` binding the enrichment implementation.

Every delta row retains the current `_paired_stage_delta` payload: status,
metric, direction, point delta, improvement, paired seed deltas, matched seeds,
group count, bootstrap valid rate, confidence level, and interval bounds.
Unavailable SFT-100 cells remain explicit `not_available` rows until that stage
exists; they are never encoded as zero.

## Architecture

Add a report-only module that reads the already validated Probe v2 report and
the full rows stored under `probes/protocol_v2/.cells/`. It calls the existing
paired-delta implementation and atomically enriches `report.json`.

The already computed `pretrained -> sft_25` rows are copied from the existing
absolute-reference delta arrays. Only genuinely missing adjacent comparisons
are bootstrapped: currently `sft_25 -> sft_50`, and later
`sft_50 -> sft_100`. This avoids duplicating valid inference work.

The CLI invokes this enrichment after `probes run` and before returning
`probes report`. The cell-generating implementation in `probe_runner.py` is not
modified, so its implementation hash and the existing 307 MB server cache stay
valid. The new analysis has its own implementation hash in the report.

The enrichment is CPU-only and may run while SFT-100 occupies the GPU. It must
not launch latent extraction or another probe-fitting process concurrently with
SFT training. On the current 16-core server, the SFT dataloader retains its four
workers and the adjacent analysis uses the remaining CPU capacity without
changing any training hyperparameter.

## Determinism and validation

Bootstrap seeds are deterministic functions of the study seed, tap, factor,
split, and adjacent-pair index. The enrichment rejects reports whose standard
Probe v2 bindings fail validation, rejects malformed cell artifacts, and writes
the report atomically.

Tests cover:

- the exact ordered adjacent-stage pairs;
- SFT-25 to SFT-50 paired deltas with matched states and seeds;
- lower-is-better Geometry interval direction;
- explicit unavailable rows for a missing SFT-100 stage;
- preservation of all existing report fields;
- deterministic repeated enrichment;
- reuse of existing `pretrained -> sft_25` rows without recomputation;
- CLI integration for both `probes run` and `probes report`.

## Scope

This change performs no SFT, latent extraction, probe fitting, closed-loop
evaluation, intervention, or RL. It only adds the missing longitudinal
inference layer to already computed Probe v2 evidence.

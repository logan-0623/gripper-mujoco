# Control-Effective Representation Diagnostics Design

Date: 2026-08-24  
Target: ICRA 2027  
Status: approved design, implementation pending

## Purpose

Extend the LIBERO–SmolVLA longitudinal representation study so that it can
distinguish representation accessibility, compact retention, temporal and
controlled dynamical organization, functional action use, and closed-loop
utility. The extension borrows the effective-theory evaluation perspective
from Representational Effective Theory (RET), while preserving the scientific
question and evidence boundaries of this robotics project.

The primary question remains:

> What physically grounded interaction information exists before manipulation
> SFT, what is selectively acquired or reorganized during SFT, where does it
> become action-proximal, and which factors are merely decodable versus
> functionally consequential for closed-loop control?

The project does not seek a single scalar definition of a universally good
representation. It measures a factor-specific profile:

```text
accessible
  -> compactly retained
  -> dynamically coherent under action
  -> functionally used by the policy
  -> useful in paired closed-loop control
```

No arrow is treated as a logical implication.

## Scientific boundary relative to RET

RET studies low-dimensional macrostates learned from frozen LLM hidden-state
trajectories and evaluates abstraction, approximate closure, behavior
prediction, and steering. This project transfers only those evaluation
principles that remain scientifically valid for embodied control.

Robot dynamics are controlled rather than autonomous. The relevant transition
model is therefore:

```text
z_(t+h) = T(z_t, a_t, language) + residual
```

and not an unconditional `z_(t+1) = T(z_t)` model. The action-free predictor is
retained only as a baseline. Language is task-constant within an episode but is
part of the conditioning contract where it is not already encoded in the tap.

The first implementation does not train a RET/JEPA encoder and does not add a
macrostate to the policy input. It operates on the preregistered native SmolVLA
latent caches. An action-conditioned learned macrostate is a conditional later
experiment and requires the native-diagnostic gate described below.

## Preserved experimental contract

The following parts of the existing study remain unchanged:

- one shared simulator-grounded LIBERO State Bank;
- stages `pretrained`, `sft_25`, `sft_50`, and `sft_100`;
- semantic taps `vision_output`, `multimodal_fusion`,
  `action_expert_input`, and `pre_action`;
- factors `entity`, `geometry`, `contact`, `stable_grasp`, `phase`, and
  `next_relation`;
- task-group primary and episode-group secondary splits;
- episode-grouped, task-balanced, nested SFT subsets;
- linear probes as the primary accessibility measurement;
- factor-aligned interventions and paired closed-loop evaluation;
- the prohibition on frame-level random splits and frame-level
  pseudo-replication;
- the separation between ACT controlled-mechanism evidence and the modern-VLA
  main study;
- the explicit stop before RL.

Existing State Banks, checkpoints, latent caches, reports, and legacy
experiments are never overwritten or silently reinterpreted.

## Research questions

### RQ1: Accessibility and pathway

Which interaction factors are linearly accessible at each training stage and
semantic tap?

This is answered by the existing `Stage x Tap x Factor` probe grid.

### RQ2: Longitudinal reorganization

Does SFT strengthen an existing encoding, introduce a newly accessible
encoding, move a factor toward action-proximal taps, or reorganize the encoding
without changing within-stage probe quality?

This is answered by matched-state representation similarity and cross-stage
probe transfer.

### RQ3: Compact retention

Can the interaction factor and action-relevant information be retained through
a low-dimensional bottleneck, or is it dispersed through high-dimensional
latent detail?

This is answered by train-only PCA compression curves with matched random
projection and constant/simple baselines.

### RQ4: Controlled dynamical organization

Can a compact representation predict its future under the executed action,
and does action conditioning improve prediction over state-only dynamics?

This is answered by action-conditioned, multi-horizon transition prediction.

### RQ5: Temporal event alignment

Is a representation stable within a coherent interaction phase while changing
at genuine contact, grasp, phase, and next-relation boundaries?

This is answered by joint smoothness and boundary-discrimination diagnostics.
Smoothness alone is never treated as evidence of a useful macrostate.

### RQ6: Functional and closed-loop relevance

For the same predefined interaction factor, does a distribution-matched
intervention disrupt the factor, alter the action, and alter paired closed-loop
behavior?

This remains the existing intervention and paired-evaluation contract.

## Analysis 1: Cross-stage reorganization

All comparisons use identical State Bank IDs. States are balanced by task and
canonical phase before similarity estimation so long `approach` segments do not
dominate the result.

### Linear CKA

For each tap and stage pair, compute linear centered-kernel alignment on the
same state matrix. Report the point estimate and cluster-bootstrap confidence
interval using tasks for the primary split and episodes for the secondary
split.

CKA is descriptive representational similarity. It does not identify the
factor responsible for similarity or establish functional use.

### Cross-stage probe transfer

For each tap, factor, source stage, and destination stage:

1. fit preprocessing and a linear probe on the source-stage training groups;
2. freeze both;
3. evaluate on the destination-stage held-out states with identical state IDs;
4. compare against the destination within-stage probe under the same metric.

The primary reorganization quantity is:

```text
transfer_gap = destination_within_stage_score - source_to_destination_score
```

Geometry uses normalized MAE and R2 with signs handled explicitly; categorical
factors use their preregistered factor metrics. Transfer is suppressed when
tensor dimensionality or factor support is incompatible rather than silently
projected.

## Analysis 2: Compactness and retention

PCA is fit on the training partition only. Primary bottleneck dimensions are:

```text
16, 32, 64, 128
```

Dimensions greater than the tap dimension or training rank are marked
inapplicable. The same training groups, normalization, hyperparameter-selection
rules, and test groups used by native probes are reused.

For factor `f` and bottleneck dimension `d`, report:

```text
retention_f(d) = compressed_probe_utility(d) / native_probe_utility
```

where `probe_utility` is a metric-specific improvement over the registered
constant/simple baseline, not a raw ratio of metrics with incompatible zeros or
directions. Geometry errors are converted to improvement over the train-mean
baseline before retention is calculated.

Also report:

- effective rank of each native latent matrix;
- explained variance at each PCA dimension;
- factor-retention curve and area under the dimension curve;
- action-prediction retention under the same bottleneck;
- matched Gaussian random projection as a compression control.

Task-ID, instruction, normalized time, majority, and train-mean baselines remain
visible where applicable. A compact representation must retain nontrivial
factor or action information; low variance or easy self-prediction is not
sufficient.

## Analysis 3: Action-conditioned closure

The analysis uses consecutive State Bank records from the same episode only.
Pairs never cross episode boundaries or missing-frame gaps. Primary horizons at
20 Hz are:

```text
1, 5, 10 frames
```

For every stage, tap, bottleneck, and horizon, compare:

1. constant future-mean predictor;
2. normalized-time predictor;
3. state-only predictor `T(z_t)`;
4. state-action predictor `T(z_t, a_t)`;
5. state-action-language predictor where language is not already encoded;
6. causal moving-average representation baseline;
7. random-projection representation baseline.

The primary predictor is linear ridge regression selected on grouped validation
data. A two-layer MLP is a capacity check. Predictor capacity and selection grid
are matched across representation baselines.

Metrics are held-out R2 and normalized prediction error. The central controlled
dynamics quantity is:

```text
action_conditioning_gain = R2[T(z_t, a_t)] - R2[T(z_t)]
```

Multi-horizon degradation is reported rather than selecting the easiest
horizon. Confidence intervals resample tasks or episodes, never individual
frames.

This measurement is called `controlled predictability`. The stronger term
`approximate closure` is reserved for a compact macrostate whose prediction is
not materially improved by adding its source microstate under a matched
predictor. That microstate-sufficiency test is conditional on a later learned
macrostate implementation.

## Analysis 4: Temporal consistency and event boundaries

For every native representation and registered compression, report:

- within-phase first-difference norm;
- geometry-aligned second-difference jitter;
- contact flip rate;
- stable-grasp flip rate;
- phase flip rate;
- next-relation flip rate;
- representation effective rank and per-dimension variance;
- event-boundary AUPRC using latent change magnitude;
- within-phase versus true-boundary change ratio.

True boundaries are derived from the fixed privileged annotations. Boundary
metrics are computed separately for contact, stable grasp, phase, and next
relation before any aggregate is shown. Isolated annotation noise remains
visible in the State Bank audit rather than being optimized away during latent
analysis.

A representation is temporally coherent only if it combines low within-phase
variation with strong true-boundary discrimination and noncollapsed variance.

## Primary result object

The original `Stage x Tap x Factor` tensor remains the primary accessibility
result. New outputs are linked views rather than a single enlarged tensor:

1. `Stage x Tap x Factor`: within-stage accessibility;
2. `SourceStage x DestinationStage x Tap x Factor`: probe transfer;
3. `Stage x Tap x Bottleneck x Factor`: compact retention;
4. `Stage x Tap x Bottleneck x Horizon`: controlled predictability;
5. `Stage x Tap x BoundaryType`: temporal event alignment;
6. `Stage x Tap x Factor x Intervention`: action sensitivity and closed-loop
   utility.

No global representation-quality score is produced. The final analysis reports
factor-specific Pareto profiles and explicitly identifies trade-offs.

## Artifact layout

All new artifacts live under the formal run without changing old paths:

```text
outputs/representation_study/libero_smolvla/
  longitudinal/
    cka/report.json
    probe_transfer/report.json
  compression/
    projections/
    report.json
  dynamics/
    transition_pairs/
    predictors/
    report.json
  temporal/
    report.json
  effective_representation/
    report.json
```

Every report binds the State Bank manifest, split manifest, checkpoint hashes,
tap contract, latent-cache hashes, config hash, and implementation hash. Caches
are resumable and are reused only when all bindings match.

## Configuration

Add a versioned `representation_diagnostics` section to the LIBERO configs:

```yaml
representation_diagnostics:
  bottleneck_dims: [16, 32, 64, 128]
  horizons: [1, 5, 10]
  primary_projection: pca
  compression_control: gaussian_random_projection
  transition_model: ridge
  transition_capacity_check: shallow_mlp
  cka_sampling: task_phase_balanced
  boundary_factors: [contact, stable_grasp, phase, next_relation]
  bootstrap_samples: 2000
  confidence_level: 0.95
```

Smoke uses fewer bootstrap samples and may restrict dimensions/horizons, but
never substitutes for formal evidence.

## CLI contract

Extend the existing lazy-loaded LIBERO command family with:

```text
libero representations longitudinal
libero representations compress
libero representations dynamics
libero representations temporal
libero representations report
```

Each command fails closed if the State Bank, stage checkpoint, latent cache,
split, or probe prerequisite is missing or stale. Dry-run and smoke modes may
validate contracts but cannot create a passing formal report.

## Gates

### Gate D1: Shared-latent integrity

- exact state-ID equality across stages for every tap;
- finite tensors and stable dimensions;
- valid task/episode split bindings;
- no state from a source episode crosses a split.

### Gate D2: Longitudinal validity

- task/phase-balanced CKA sample manifest is frozen before metrics;
- probe preprocessing is fit only on source training groups;
- destination test data never participates in model selection;
- unsupported transfer cells are explicit.

### Gate D3: Compression validity

- PCA/random projections are fit on training groups only;
- native and compressed probes share evaluation groups;
- collapse diagnostics are finite;
- retention is defined relative to the correct factor baseline.

### Gate D4: Dynamics validity

- transition pairs remain within episodes and contain no frame gaps;
- actions are aligned to the pre-action state and next state;
- grouped validation selects predictor hyperparameters;
- all horizons and preregistered baselines are reported;
- the action-conditioning comparison uses matched predictor capacity.

### Gate D5: Temporal validity

- boundary labels come only from the frozen privileged annotation;
- event support is reported per split and factor;
- smoothness, boundary discrimination, and collapse checks are jointly present.

### Gate D6: Functional relevance

- existing probe and intervention-specificity gates pass;
- paired rollouts share initial state, language, seed, and inference-noise plan;
- action displacement is not described as closed-loop utility;
- utility claims require paired task-level outcomes and confidence intervals.

## Conditional learned macrostate

An action-conditioned RET/JEPA-style encoder is implemented only if native
diagnostics establish all of the following:

1. at least one factor is accessible but poorly retained by linear compression;
2. controlled predictability varies meaningfully across stages or taps;
3. temporal smoothness and boundary sensitivity expose a nontrivial trade-off;
4. the additional model answers a scientific question that PCA and native taps
   cannot answer.

If admitted, the learned macrostate is a post-hoc diagnostic adapter, not a
policy input. It is trained only on training trajectories, conditioned on
actions, compared against native, PCA, random projection, temporal mean, and
matched autoencoder baselines, and evaluated on unchanged held-out State Bank
states. Learned representations for different stages use identical
architectures, data budgets, seeds, and selection rules.

## Expected paper evidence

The optimized evidence package consists of:

1. State Bank coverage and annotation audit;
2. full accessibility heatmaps;
3. cross-stage CKA and probe-transfer matrices;
4. compact-retention curves;
5. controlled-predictability and temporal-boundary diagnostics;
6. factor-aligned action intervention;
7. paired closed-loop utility and failure-mode contrasts;
8. ACT/Graph-v2 results as controlled mechanism motivation rather than the
   main modern-VLA evidence.

The intended contribution is a physically grounded, longitudinal, and
control-linked representation study. Generic probing, generic activation
intervention, or the observation that decodability need not imply causality are
not claimed as standalone novelty.

## Implementation sequence

1. update `ccfa.yaml` and config schemas without changing existing bindings;
2. implement shared diagnostic data loading and grouped sampling;
3. implement CKA and cross-stage probe transfer;
4. implement train-only compression and retention metrics;
5. implement aligned transition-pair construction and controlled predictors;
6. implement temporal event diagnostics;
7. implement the linked report and CLI commands;
8. add unit tests, cache-binding tests, leakage tests, and smoke integration;
9. update README and server runbook commands;
10. stop before learned RET and RL.

## Success condition

The implementation is complete when the smoke pipeline can produce validated,
reproducible artifacts for all five native diagnostic views from one shared
State Bank and one shared set of stage latent caches, while the formal pipeline
can execute the same protocol without altered scientific definitions. No
scientific finding is claimed until the formal reports contain real results.

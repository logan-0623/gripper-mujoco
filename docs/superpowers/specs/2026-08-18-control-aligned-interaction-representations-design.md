# Control-Aligned Interaction Representations Design

Status: approved direction; written specification pending final review

## 1. Purpose

This design turns the completed Graph v2 ACT experiment into a controlled study of
which properties make a structured intermediate representation usable by a
continuous visuomotor policy.

The primary question is:

> What makes a structured interaction representation usable for continuous
> visuomotor control?

The implementation must connect three levels of evidence without claiming causality
from observational associations alone:

```text
shared-state representation properties
    -> policy sensitivity and utilization
    -> on-policy interaction events and terminal outcomes
```

The completed four-condition result is preserved as the motivating observation. It
does not, by itself, establish that prediction smoothness or any other representation
property caused the performance differences.

## 2. Scientific terminology and claims

The existing `oracle_graph_v2` artifact and configuration identifiers remain unchanged
for checkpoint compatibility. User-facing reports, documentation, plots, and paper
text call this condition the **Privileged Teacher Graph** or **Teacher Graph**. It is
not described as a policy-performance upper bound.

The first progressive ablation uses the name **Entity+Geometry**, not Object-only.
The current 89D schema stores object position primarily through pairwise local
geometry and does not contain an independent object-slot pose representation. Calling
a masked subset of that schema Object-only would overstate what the experiment
isolates.

The strongest claim allowed before the new experiments is:

> Different structured representations produce different closed-loop outcomes, so
> privileged correctness alone may be insufficient to characterize control utility.

The stronger claim that aggregate representation correctness does not predict control
utility is enabled only if the shared-state diagnostics, policy interventions, and
closed-loop results support it.

## 3. Scope

This work includes:

- offline diagnostics over the already frozen, row-aligned Graph token caches;
- policy sensitivity diagnostics using existing ACT checkpoints;
- optional step-level tracing during closed-loop evaluation;
- failure-conditioned, event-aligned association analysis;
- a fixed-width progressive representation ablation;
- machine-readable JSON/JSONL reports and concise README commands;
- deterministic tests for metrics, trace schemas, masking, shuffling, and provenance.

This work does not initially include:

- a new object detector, segmentation model, or slot encoder;
- a new Graph estimator schema;
- retraining ReflectVLM;
- SmolVLA, pi0, a world model, sim-to-real, or a real-robot experiment;
- causal language for observational failure associations;
- using inference-only masking as a substitute for separately trained ablations.

## 4. Existing evidence and available artifacts

The current pilot provides three policy seeds and four conditions:

```text
flat
oracle_graph_v2              # reported as Privileged Teacher Graph
predicted_random_v2
predicted_reflect_v2
```

For each seed and condition, the token cache contains 6,722 row-aligned 89D tokens.
The cache row identifiers cover complete LeRobot episodes and bind the dataset,
split manifest, Graph checkpoint, token schema, and prerequisite reports by hash.

The current evaluation report contains 60 rollouts per condition but stores only
episode-level metrics. It cannot answer when a Graph error occurred, how the action
changed, or which interaction event preceded a failure. Step-level traces therefore
require re-evaluation with existing checkpoints but do not require policy retraining.

## 5. Selected architecture

The feature is implemented inside `interaction_vla.graph_control` as four bounded
units:

1. `diagnostics.py` computes representation statistics from frozen shared-state
   caches and compares predicted tokens with the Teacher token.
2. `sensitivity.py` evaluates standardized token interventions against frozen ACT
   checkpoints on fixed dataset observations.
3. `tracing.py` validates and writes step-level closed-loop records without making
   the rollout loop responsible for statistical analysis.
4. `failure_analysis.py` converts completed traces into event-aligned association
   reports with clustered uncertainty estimates.

The existing `pipeline.py` remains responsible for configuration, provenance,
loading datasets/checkpoints, and atomic publication. The CLI gains commands that
delegate to these focused modules.

Progressive ablations reuse the current Graph cache and ACT training pipeline through
explicit token transforms. The transform is part of cache/checkpoint provenance, so
an ablation checkpoint cannot be loaded as a different condition.

## 6. Stage A: shared-state representation diagnostics

### 6.1 Comparison population

Diagnostics operate on a selected dataset partition, defaulting to `test`. Every
condition must use exactly the row indices listed in the existing split manifest.
The loader rejects missing, duplicated, reordered, or condition-specific rows.

Teacher and predicted representations are compared on the same observation rows.
Flat is summarized only as a zero-input control and is excluded from Teacher-distance
metrics.

Episode boundaries come from the split manifest and LeRobot `episode_index` and
`frame_index` fields. Temporal differences never cross episode boundaries.

### 6.2 Metric groups

For every token feature and every named token slice, diagnostics report:

- mean, standard deviation, minimum, maximum, and finite-value count;
- p05, p25, median, p75, and p95;
- p95-p05 robust dynamic range;
- active fraction, defined as the fraction whose absolute value exceeds `1e-6`;
- saturation fractions at or beyond `-0.99` and `0.99`;
- first-difference mean absolute value and root mean square value;
- second-difference mean absolute value, computed only where three consecutive
  frames exist;
- lag-0 and best lagged Pearson correlation against Teacher for lags `[-3, 3]`, when
  both sequences have nonzero variance.

At slice level the report additionally contains:

- mean per-frame L1, L2, and cosine distance to Teacher;
- covariance effective rank using the entropy of normalized eigenvalues;
- categorical entropy for `phase`, `next_relation`, `relation_operator`, and
  `predicate` distributions;
- categorical flip rate, mean dwell length, and false-flip rate relative to Teacher;
- entity/relation hard-state flip rate after a fixed 0.5 threshold.

`goal_residual`, geometry, and relation-trend slices are treated as continuous. Soft
categorical slices are summarized both as probabilities and by `argmax`. Zero-vector
categorical inputs are recorded as missing, not assigned category zero.

### 6.3 Aggregation and uncertainty

Raw per-frame values are not treated as independent samples for uncertainty.
Condition summaries include episode-level metric values and nonparametric 95% cluster
bootstrap intervals sampled over episodes with a fixed seed. The report states the
number of episodes, frames, and policy/estimator seeds contributing to each result.

Estimator seeds are reported separately and summarized by their mean and sample
standard deviation. Teacher tokens are identical across estimator seeds and are
deduplicated before aggregate statistics.

### 6.4 Outputs

The command is:

```bash
.venv-lerobot/bin/python -m interaction_vla.graph_control diagnose \
  --config configs/graph_v2_act_pilot_macos.yaml \
  --partition test
```

It publishes atomically to:

```text
outputs/graph_control/graph_v2_pilot/diagnostics/test/
    report.json
    per_episode.jsonl
```

The report records cache hashes, split hash, token schema, diagnostic schema,
partition, bootstrap seed, and implementation source fingerprint.

## 7. Stage B: frozen-policy sensitivity diagnostics

### 7.1 Purpose

Sensitivity diagnostics ask whether a trained ACT policy responds to each Graph
group. They do not, alone, establish that the group improves closed-loop success.

The analysis uses fixed dataset observations from the test partition. For each
policy seed and representation condition, it loads the existing frozen ACT policy and
computes the first predicted action under controlled changes to the Graph token.
Images, proprioception, language, preprocessing, and policy state are held fixed.

### 7.2 Interventions

The baseline is the unmodified cached token. Each named representation group receives
two interventions:

1. **Group masking:** replace the group with its neutral zero value.
2. **Standardized finite difference:** add and subtract `0.25` training-standard-
   deviation units per continuous feature, clipped to the observed training p01/p99
   range. Categorical probability groups use a probability-preserving interpolation
   toward the uniform distribution rather than independent coordinate perturbation.

Policy state is reset before every counterfactual prediction. This prevents ACT
internal state or action queues from leaking between interventions.

### 7.3 Metrics

For every policy seed and group, report:

- L1 and L2 change in the first 6D end-effector command;
- translation and rotation changes separately;
- gripper-command absolute change;
- action-direction cosine change;
- fraction of observations whose translation sign changes;
- sensitivity normalized by the actual standardized perturbation magnitude.

The report retains per-episode values and cluster-bootstrap intervals. A large value
means the policy is sensitive to the group, not that the sensitivity is beneficial.

### 7.4 Outputs

The command is:

```bash
.venv-lerobot/bin/python -m interaction_vla.graph_control sensitivity \
  --config configs/graph_v2_act_pilot_macos.yaml \
  --partition test
```

It publishes under `diagnostics/test/sensitivity/` and binds every result to the ACT
checkpoint and cache hashes.

## 8. Stage C: step-level closed-loop trace

### 8.1 Evaluation behavior

Tracing is enabled explicitly in the graph-control config. A traced evaluation uses
the same paired case schedule, policy checkpoints, receding-horizon action execution,
and episode summary metrics as the existing evaluation. Tracing must not change the
action selected by the policy.

For every policy observation, the rollout obtains both:

- `policy_token`: the token supplied to the current policy condition;
- `teacher_token`: the causal Privileged Teacher token computed from the same current
  MuJoCo snapshot.

The Teacher tracker is reset once per episode and updated once per policy step,
independently of which token conditions the policy. Predicted runtimes and Teacher
tracking must not share mutable temporal state.

### 8.2 Trace record schema

Every JSONL record contains:

```text
trace_schema_version
episode_id / case_id / environment_seed
condition / policy_seed / layout / object_count / training_distribution
step / phase
policy_token / teacher_token
graph_error_by_group
raw_action / clipped_action / executed_world_action
action_was_clipped / ik_projection_scale / gripper_command
end_effector_position / end_effector_orientation
target_relative_position / receptacle_relative_position
minimum_distractor_clearance
target_contact / stable_target_grasp
wrong_object_contact / stable_wrong_object_grasp
events
done / termination_reason
```

All arrays have fixed documented shapes and finite numeric values. Contact booleans
and distances are derived from causal current state. Terminal outcome is repeated only
on the final step; it is never supplied to the policy.

`graph_error_by_group` contains L1/L2 distance for continuous groups and hard-label
agreement for categorical groups. Flat traces retain the Teacher comparison but are
not described as Graph prediction errors.

### 8.3 Atomic and scalable output

Each episode is written to a separate temporary JSONL file and atomically renamed
after successful completion. A manifest binds trace files to checkpoint hashes,
case schedule, config hash, trace schema, and source fingerprint. An interrupted run
may resume only complete manifest-compatible episodes; incompatible or partial files
are rejected rather than silently reused.

Traces publish to a new output root instead of overwriting the completed scientific
artifact:

```text
outputs/graph_control/graph_v2_pilot/traced_evaluation/
    manifest.json
    episodes.jsonl
    report.json
    traces/<policy-seed>/<condition>/<case-id>.jsonl
```

The command is:

```bash
.venv-lerobot/bin/python -m interaction_vla.graph_control trace \
  --config configs/graph_v2_act_pilot_macos.yaml
```

## 9. Stage D: failure-conditioned analysis

### 9.1 Outcomes

The primary terminal outcomes are:

```text
success
timeout
target_drop
wrong_object_stable_grasp
```

Wrong-object interaction without a stable grasp remains a secondary event. Analyses
with fewer than five positive outcome episodes are marked underpowered and report
counts/effect sizes without inferential language.

### 9.2 Error exposure

For each Graph group, error exposure is summarized in event-aligned windows:

- 10 steps before first target contact;
- grasp establishment through lift;
- transport;
- 10 steps before release;
- final 20 steps before termination.

The analysis also records maximum error, mean error, false flips, and duration above
a threshold fitted from the training partition only.

### 9.3 Association score and uncertainty

The descriptive score is named **Failure Association Score**:

```text
P(failure | high error exposure) - P(failure | low error exposure)
```

High error exposure is defined by the training-partition p75 threshold and cannot be
selected using evaluation outcomes. Confidence intervals use a fixed-seed clustered
bootstrap over `(policy_seed, case_id)`. Reports include raw contingency counts,
risk difference, risk ratio when defined, and interval estimates.

This score is never called causal. Causal wording is reserved for controlled token
interventions or separately trained representation ablations.

The command is:

```bash
.venv-lerobot/bin/python -m interaction_vla.graph_control failure-analysis \
  --config configs/graph_v2_act_pilot_macos.yaml \
  --traces outputs/graph_control/graph_v2_pilot/traced_evaluation
```

## 10. Stage E: progressive representation ablation

### 10.1 Fixed Graph source

The primary progressive ablation uses `predicted_random_v2` as the single frozen
Graph source. This avoids selecting Reflect initialization after observing its small
mean success advantage, keeps semantic pretraining out of the primary representation
claim, and ensures that every non-flat ablation differs only in which fields of the
same estimated token are exposed to ACT.

Within policy seed, `entity_geometry`, `interaction_state`, `full_graph`, and
`shuffled_graph` bind the same random-initialized Graph estimator checkpoint. The
existing Teacher and Reflect results remain separate diagnostic comparisons; they are
not mixed into this progressive hierarchy. A later confirmatory run may repeat the
prespecified hierarchy with the Teacher source, but it is not part of the first
ablation gate.

### 10.2 Conditions

The first fixed-schema ablation is:

| Condition | Entity/visibility | Continuous geometry | Current relations/phase | Trends/transition |
|---|---:|---:|---:|---:|
| `flat` | no | no | no | no |
| `entity_geometry` | yes | yes | no | no |
| `interaction_state` | yes | yes | yes | no |
| `full_graph` | yes | yes | yes | yes |
| `shuffled_graph` | yes | yes | yes | yes, but correspondence is destroyed |

The group masks are:

- entity/visibility: `entity_presence`, `entity_visibility`;
- continuous geometry: `gripper_target_geometry`,
  `target_receptacle_geometry`, `distractor_geometry`;
- current interaction: `relation_presence`, `phase`;
- temporal/transition: `relation_trends`, `next_relation`,
  `relation_operator`, `predicate`, `goal_residual`.

All conditions retain an 89D `observation.environment_state`, identical ACT parameter
count, identical initial ACT weights within policy seed, identical row order, epochs,
and train/validation/test split.

### 10.3 Shuffled control

Shuffling uses a deterministic episode-level permutation within the training
partition, stratified by episode length quartile. A source episode's complete token
sequence is paired with a different observation episode, and the sequence is
linearly indexed to the destination episode length without interpolation of token
values. This preserves within-sequence temporal structure and approximate marginal
distribution while breaking observation-token correspondence.

Validation uses an independently derived fixed permutation from the same master seed.
No training/validation partition shares an episode or permutation target. The
permutation manifest is stored and hashed.

Closed-loop evaluation cannot permute against an on-policy future episode. Instead,
each evaluation case is deterministically assigned a different complete token
sequence from the frozen test-partition cache reservoir, stratified by sequence-length
quartile. The reservoir sequence is indexed by normalized episode progress using only
the current step and configured `max_steps`; values are selected by nearest index and
are never interpolated. Source sequences are balanced across paired cases and are
independent of the current camera observation, environment seed, action, and outcome.

Thus the shuffled policy is trained and evaluated with observation-independent token
sequences drawn from the same frozen Graph source. The test-time source schedule is
declared before rollout and stored in the manifest. A test-time-only shuffle applied
to a normally trained policy is reported separately as a sensitivity diagnostic.

### 10.4 Output isolation

The ablation receives separate macOS and Linux/CUDA configs and writes to a new root:

```text
outputs/graph_control/control_alignment_ablation/
```

It never overwrites the completed Graph v2 pilot. At least three policy seeds are
required. The report uses policy seed as the replication unit and retains paired-case
deltas. No condition is promoted based on one seed alone.

The ablation commands use a separate strict configuration that references the
completed Graph v2 config:

```bash
.venv-lerobot/bin/python -m interaction_vla.graph_control ablation-inspect \
  --config configs/control_alignment_ablation_macos.yaml

.venv-lerobot/bin/python -m interaction_vla.graph_control ablation-cache \
  --config configs/control_alignment_ablation_macos.yaml

.venv-lerobot/bin/python -m interaction_vla.graph_control ablation-smoke \
  --config configs/control_alignment_ablation_macos.yaml

.venv-lerobot/bin/python -m interaction_vla.graph_control ablation-compare \
  --config configs/control_alignment_ablation_macos.yaml

.venv-lerobot/bin/python -m interaction_vla.graph_control ablation-evaluate \
  --config configs/control_alignment_ablation_macos.yaml
```

## 11. Configuration

The existing config remains valid. New optional sections use strict unknown-field
validation:

```yaml
diagnostics:
  output_dir: outputs/graph_control/graph_v2_pilot/diagnostics
  partition: test
  bootstrap_samples: 2000
  bootstrap_seed: 2057736129
  max_lag: 3
  active_epsilon: 1.0e-6

trace:
  enabled: false
  output_dir: outputs/graph_control/graph_v2_pilot/traced_evaluation
  resume: true

ablation:
  output_dir: outputs/graph_control/control_alignment_ablation/runs
  cache_dir: outputs/graph_control/control_alignment_ablation/cache
  conditions: [flat, entity_geometry, interaction_state, full_graph, shuffled_graph]
  shuffle_seed: 2057736129
```

Commands that require a section fail with an actionable error when it is absent.
Existing `inspect`, `cache`, `smoke`, `compare`, and `evaluate` behavior remains
backward compatible.

## 12. Error handling and provenance

All outputs use atomic publication. The pipeline rejects:

- mismatched token schemas or feature ordering;
- cache rows that differ across compared conditions;
- temporal calculations crossing episode boundaries;
- missing or incompatible ACT checkpoints;
- nonfinite token, action, geometry, or metric values;
- a trace whose checkpoint/config/case hashes do not match the requested run;
- an ablation transform not recorded in cache and checkpoint provenance;
- overwrite of completed diagnostics, traces, or ablation artifacts.

Reports contain `passed: true` only when structural and numerical validation passes.
This flag means artifact integrity, not that a scientific hypothesis was supported.

## 13. Testing strategy

Development follows test-first red-green-refactor cycles.

Unit tests cover:

- episode-safe first and second differences;
- zero-variance and missing-category behavior;
- categorical flips, false flips, dwell lengths, entropy, and effective rank;
- lagged correlation without crossing episode boundaries;
- deterministic clustered bootstrap;
- probability-preserving categorical perturbations;
- reset-isolated policy counterfactuals;
- exact trace shapes, finite values, and final-step outcome rules;
- resumable atomic episode traces and incompatibility rejection;
- event-window extraction and underpowered failure outcomes;
- deterministic group masks and partition-safe shuffled mappings;
- cache/checkpoint provenance for every ablation condition;
- backward compatibility of the existing Graph v2 config and CLI.

Integration tests use synthetic token caches and fake policy/environment runtimes.
One bounded local smoke command verifies the real existing cache and checkpoints when
the LeRobot environment is installed. Full MuJoCo reruns and ablation training remain
explicit user-run experiments because they are long-running.

## 14. Documentation and experiment order

The README exposes commands in this order:

1. shared-state `diagnose`;
2. frozen-policy `sensitivity`;
3. traced evaluation;
4. `failure-analysis`;
5. ablation inspect/cache/smoke/compare/evaluate.

The documentation distinguishes commands implemented and verified locally from
long-running experiments that the user must execute on macOS or Linux/CUDA. It also
states which outputs are reusable and which commands require a new isolated output
directory.

## 15. Success criteria

Implementation is complete when:

1. existing caches produce a deterministic, provenance-bound diagnostics report;
2. sensitivity analysis loads existing ACT checkpoints and produces finite grouped
   action-change metrics without changing checkpoints;
3. traced evaluation records causal per-step Teacher and policy tokens without
   changing selected actions or existing episode summary definitions;
4. failure analysis produces event-aligned descriptive associations with clustered
   intervals and underpowered warnings;
5. the five progressive ablations can be trained and evaluated with identical ACT
   capacity and paired experimental controls;
6. tests pass in the base environment, with LeRobot-dependent tests passing in
   `.venv-lerobot`;
7. README commands and output paths match the implementation exactly.

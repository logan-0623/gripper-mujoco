# Gripper–Object Interaction Representation Experiment v3

**Date:** 2026-08-03  
**Status:** Ready for user review  
**Scope:** A fair Flat-versus-Graph representation experiment with a shared
short-horizon controller and a strict physical pick-and-place task.

## 1. Representation Hypothesis

The sole method hypothesis is:

> Given identical physical state information, source episodes, temporal policy
> head, optimization budget, and rollout controller, an explicit interaction
> Graph encoder learns gripper-aware and object-aware manipulation more easily
> than a Flat vector encoder.

The experiment does not claim action chunking, temporal smoothing, recovery
augmentation, or the scripted expert as contributions. They are fixed training
and control infrastructure shared by Flat and Graph.

The primary evidence concerns gripper–object interaction: selecting the correct
object, forming bilateral contact, establishing a stable grasp, lifting the
target, and transporting it toward the receptacle. Complete strict pick-and-place
success is required as a secondary task-level result.

## 2. Motivation and Current Failure Evidence

The terminal-recovery pilot exposed three problems that prevent the current
report from supporting the hypothesis cleanly:

1. The training split contains 40 full base demonstrations and 148 retained
   recovery suffixes. By episode count, recovery dominates the training set and
   shifts the experiment away from normal interaction learning.
2. The Graph checkpoint has lower normalized training MSE than Flat and produces
   some stable lifts, but its closed-loop placement is not reliable. In the
   generated comparison GIF, the CPU Graph rollout for seed `337141941` drops the
   target at step 113 with target-to-receptacle XY offset approximately
   `[-0.0946, 0.0452]` metres, outside the receptacle footprint.
3. The current placement predicate counts contact with any receptacle geometry,
   including exterior walls. Evaluation also uses the configured `auto` device
   while visualization loads the policy on CPU, so report and GIF can diverge on
   the same checkpoint and seed.

The v3 experiment fixes these contracts before interpreting task success.

## 3. Non-Negotiable Fairness Invariant

The experimental variable is `representation encoder` only.

Both encoders receive the same masked node features, edge features, relative
poses, contact signals, target flags, and proprioception. Flat receives that
payload in a fixed canonical flattened order; Graph additionally consumes the
explicit edge index to perform message passing. Flat is not weakened by removing
an observed state field, and Graph is not given an extra sensor modality.

Flat and Graph must share all of the following:

- raw state fields and feature schema;
- source-seed split and retained sequence windows;
- action normalization and target construction;
- H=8 action-chunk head architecture and hidden dimensions;
- initialization seed for corresponding shared head layers;
- future-action loss, optimizer, batch schedule, epochs, and stopping rule;
- 75/25 base/recovery batch mixture;
- temporal ensemble, gripper hysteresis, clipping, and IK projection;
- rollout device and deterministic evaluation cases;
- strict placement and termination predicates.

No horizon, smoother, head, optimizer, or controller parameter may be tuned
separately for Flat and Graph. Reports and checkpoint metadata must contain:

```text
representation_variable: encoder_only
```

## 4. Source-Level Dataset Split

### 4.1 Base data

Collect 200 successful normal-layout expert demonstrations with object counts 2
and 3. The independent expert gate and physical-contact requirements remain in
force. No suction, weld, object qpos edit, or scripted attachment is allowed.

The split unit is the successful base episode source, identified by
`source_seed`. A deterministic source-seed mapping assigns exactly:

- 160 source seeds to train;
- 20 source seeds to validation;
- 20 source seeds to test.

The mapping is created before recovery trajectories or action windows are
constructed. Every artifact derived from a source inherits its source split:

```text
base episode
  -> recovery variants
  -> H=8 sequence windows
  -> RGB-D sidecars, if enabled
```

Random trajectory-level or window-level splitting is forbidden. The collector
writes `source_split.json`, and loading fails if a source seed appears in more
than one split.

### 4.2 Training recovery augmentation

Recovery is auxiliary data, not the primary objective. Four deterministic
recovery kinds remain available, including post-placement reclose. Training
recovery is generated only from a deterministic 25% subset of train sources:
40 of the 160 train source seeds. Up to four balanced variants are attempted per
selected source, with the existing per-kind quality gate.

The number and length of accepted recovery episodes do not determine their
training influence. Every batch has 48 base and 16 recovery samples at batch
size 64, so recovery contributes exactly 25% of samples and loss mass. Within
each group, start phases are sampled uniformly, then windows within a phase are
sampled uniformly. A missing recovery group, source leakage, or inability to
construct the 48/16 batch fails training before the first optimizer step.

Each sequence sample's masked horizon loss is first normalized by the sum of its
valid future-step weights. Training then computes the mean of the 48 normalized
base losses and the mean of the 16 normalized recovery losses and combines them
as `0.75 * base_loss + 0.25 * recovery_loss`. Variable sequence length therefore
cannot change the configured group contribution.

Normalization statistics are computed from train-source base frames only.
Recovery observations and actions are normalized using those fixed statistics.

### 4.3 Held-out recovery benchmark

Recovery variants derived from validation and test source seeds are evaluation
cases only. Their NPZ files and case manifests are stored separately from the
training data and rejected by the training loader. They measure hard-state
recovery without allowing recovery data to become the contribution.

## 5. Shared Short-Horizon Policy

### 5.1 Sequence target

At policy step `t`, both policies receive the current physical observation and
predict:

```text
[a_t, a_(t+1), ..., a_(t+7)] with shape [8, 7]
```

Each action remains the existing 7D Cartesian delta pose plus parallel-gripper
command. Windows never cross episode boundaries. Near episode termination, a
boolean horizon mask marks unavailable targets; actions are not repeated or
zero-padded into the loss as artificial supervision.

The masked sequence loss uses fixed future decay:

```text
horizon_weight(k) = 0.9^k,  k in [0, 7]
```

The per-step 7D error remains normalized MSE. Horizon masks, future weights,
phase-balanced group sampling, and the 75/25 group mass are applied identically
to Flat and Graph.

### 5.2 Temporal action aggregation

Every rollout step predicts a new H=8 chunk. Predictions from overlapping chunks
that refer to the current target timestep are aggregated. For the six Cartesian
dimensions, the fixed age weight is:

```text
temporal_weight(age) = exp(-0.25 * age)
```

Newer chunks therefore receive more weight. The resulting 6D delta is clipped to
the existing action bounds and passed through the existing IK projection.

The gripper dimension is aggregated as a weighted vote followed by hysteresis:

- aggregate `g >= 0.65`: command open;
- aggregate `g <= 0.35`: command closed;
- `0.35 < g < 0.65`: retain the previously executed discrete gripper state.

The gripper state and temporal buffer reset at every environment reset. No
learned controller-specific smoothing parameter is allowed.

### 5.3 One shared rollout controller

A single `ChunkedPolicyController` owns observation normalization, chunk
prediction, temporal buffers, Cartesian aggregation, gripper hysteresis, action
clipping, and IK projection. Evaluation, dashboard, native viewer, and GIF export
must all use this controller rather than duplicate policy-action logic.

Training may use MPS. All learned-policy evaluation and visualization rollouts
use CPU for deterministic parity and record `rollout_device: cpu`. Given the same
checkpoint, config, seed, and object count, report and GIF must agree on final
step, termination reason, strict-placement state, and final target pose within
numeric tolerance.

## 6. Strict Receptacle Placement

Contact parsing distinguishes `receptacle_base` from the four walls. A target is
strictly contained only when all conditions below hold:

1. Transform the target centre and orientation into the receptacle local frame.
2. Compute the target box's projected half extents on the receptacle local X and
   Y axes from its current orientation.
3. Require `abs(local_xy) + projected_half_extent` to lie within both inner wall
   faces. Merely placing the target centre inside is insufficient.
4. Require target contact with `receptacle_base`. Wall-only or exterior-wall
   contact does not count.
5. Require low linear and angular velocity for 10 consecutive physics substeps.

Strict task success additionally requires the gripper to be open and the TCP to
be at least 0.08 m above the target after release. This retains the required
release-and-retreat behaviour.

The report records both `strict_placement_rate` and
`wall_only_receptacle_contact_rate`. The old any-receptacle-contact placement
predicate cannot produce success in the v3 environment contract.

## 7. Metrics and Experimental Interpretation

### 7.1 Primary interaction metrics

- `target_first_contact_rate`: first meaningful bilateral object contact is the
  target;
- `bilateral_target_contact_rate`;
- `stable_target_grasp_rate`;
- `stable_lift_rate`;
- `wrong_object_interaction_rate`;
- `transport_progress_rate` toward the receptacle.

These metrics directly test gripper awareness and object awareness.

### 7.2 Secondary task metrics

- `strict_placement_rate`;
- `strict_success_rate`;
- `wall_only_receptacle_contact_rate`;
- release completion and upward-retreat completion;
- drop, premature-open, post-placement-reclose, IK-failure, and physics-failure
  rates.

The initial ID acceptance target is Graph `strict_success_rate >= 0.50` on normal
2/3-object cases. This is a project gate for a usable simple task, not a claimed
statistical theorem.

### 7.3 Evaluation levels

1. ID normal with 2 and 3 objects;
2. crowded/OOD interaction with 4 and 5 objects;
3. held-out recovery cases from validation/test sources.

The primary comparison is always `Flat-H8 versus Graph-H8` on paired cases.
An H=1 controller may be retained as an internal diagnostic but is not a core
result and is not presented as a competing method.

## 8. Configuration, Outputs, and Provenance

Add isolated smoke and pilot configurations:

- `configs/physics_interaction_chunk_smoke_macos.yaml`;
- `configs/physics_interaction_chunk_pilot_macos.yaml`.

The pilot writes only under:

```text
outputs/interaction_graph_physics/interaction_chunk_pilot/
```

Existing terminal-recovery and recovery-pilot artifacts remain untouched.

Dataset, checkpoint, report, and GIF provenance bind at least:

- expert-gate hash and scene/controller hashes;
- complete source-seed split mapping;
- dataset content and recovery benchmark hashes;
- base/recovery group mass and batch composition;
- H=8, future decay 0.9, temporal decay 0.25;
- gripper thresholds 0.35/0.65;
- strict-placement contract version;
- rollout device;
- `representation_variable=encoder_only`.

A mismatch is a hard error; stale gates, datasets, checkpoints, and reports are
not silently reused.

## 9. Diagnostics and Failure Handling

- Collection retains tqdm progress and writes manifests/rejections before a
  quality failure is raised.
- Training reports separate base and recovery sequence loss as well as their
  effective loss mass.
- Rollouts record raw first-chunk action, aggregated action, temporal ensemble
  size, Cartesian smoothing delta, gripper vote, discrete gripper state, and
  gripper switch count.
- Evaluation rows include final target-to-receptacle local pose, projected
  extents, base/wall contacts, containment margin, and rollout device.
- Viewer overlay displays termination reason, strict placement, containment
  margin, gripper state, and temporal ensemble size.

These diagnostics distinguish representation failure from control smoothing,
IK, contact, and metric failures.

## 10. Verification Requirements

Automated tests must cover:

1. deterministic 160/20/20 source-level split and strict source disjointness;
2. recovery/window inheritance of the source split;
3. rejection of validation/test recovery by the training loader;
4. exact 48/16 base/recovery batch composition and 75/25 loss mass;
5. Flat/Graph raw state-information parity;
6. H=8 target alignment, boundary masks, and masked future-decay loss;
7. Cartesian temporal aggregation and buffer reset;
8. gripper hysteresis, including no chatter in the dead band;
9. strict containment under rotated target orientation;
10. exterior-wall and wall-only contacts cannot create placement or success;
11. stable base contact plus open gripper and retreat can create success;
12. identical report/viewer/GIF controller results for a fixed CPU case;
13. provenance rejection for every v3 control/data contract change;
14. no object qpos/qvel write, weld, suction, or attachment during rollout;
15. an end-to-end smoke chain: expert gate, collection, chunk training for Flat
    and Graph, paired ID evaluation, and GIF replay parity.

## 11. Completion Criteria

Implementation is complete only when:

- the full automated suite passes;
- the isolated smoke chain passes with current provenance;
- source leakage and recovery loss-mass audits pass;
- strict placement rejects the reproduced exterior-drop case;
- report and GIF agree on the same fixed CPU rollout;
- the 200-base pilot commands are documented with tqdm;
- no pilot training is automatically started without the user's explicit run.

The v3 pilot is then interpreted in this order: first interaction metrics, second
strict complete-task success, and finally held-out recovery robustness.

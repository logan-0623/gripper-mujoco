# ACT Control Recovery and Interaction Graph v2 Design

Status: approved design

## 1. Purpose

This design repairs the control floor of the current Graph-conditioned ACT pilot and
then replaces the control token with an Interaction Graph v2 that contains the compact
directional geometry needed by a continuous controller.

The scientific pipeline remains:

```text
ReflectVLM semantic graph pretraining
    -> MuJoCo graph fine-tuning
    -> frozen graph estimation from RGB + wrist RGB + proprioception + language
    -> LeRobot ACT continuous control
```

The design amends, rather than erases, the completed
`2026-08-11-graph-conditioned-act-design.md` experiment. Its zero-success result is
retained as evidence that the original control and representation choices were not
sufficient.

## 2. Evidence motivating the revision

The completed pilot cannot support a Graph-versus-Flat control claim:

- all four conditions produced zero successes over 240 paired rollouts;
- the Flat policy timed out on an exact training episode reset, so the failure is not
  explained only by crowded or held-out distribution shift;
- on the first frame of that training episode, the Flat policy's largest error was the
  target-directed translation axis, consistent with action averaging or visual
  underfitting;
- ACT trained a ResNet18 vision encoder from scratch on only 40 training episodes;
- training traversed adjacent frames in the same fixed sequential order every epoch;
- rollout executed all eight predicted actions open loop before observing again;
- the reported zero KL value was a logging defect: LeRobot returns `kld_loss`, while
  the bridge recorded `kl_loss`;
- the 75D Graph v1 token selected relation channels 12 through 21 and therefore
  omitted relative position and rotation. It described margins, risks, and task
  errors, but not the direction of the next Cartesian correction.

Dataset action replay already reconstructs expert trajectories successfully. The
action codec is therefore not the first repair target.

## 3. Approaches considered

### 3.1 Selected: recover Flat control, then introduce Graph v2

This path first establishes that ACT can learn a closed-loop controller, then tests an
oracle Graph v2, and only then measures vision-estimated Graph v2. It adds moderate
engineering work but keeps failures attributable to control, representation, or graph
estimation.

### 3.2 Rejected: tune ACT while retaining Graph v1

Shuffle, pretrained vision, and receding-horizon execution could recover Flat control,
but Graph v1 would still lack directional interaction geometry. A negative oracle
result would remain unsurprising and scientifically weak.

### 3.3 Deferred: scale demonstrations before fixing the pipeline

Collecting 200--500 demonstrations or adding DAgger may eventually be useful, but it
is expensive and does not correct fixed-order batches, open-loop execution, or the
missing Graph direction fields. Data scaling is a fallback after the bounded recovery
experiment.

## 4. Scope and non-goals

This revision includes:

- deterministic ACT sampling and correct diagnostics;
- pretrained visual features and one-step receding-horizon execution;
- train-seen and held-out control gates;
- a versioned, compact Interaction Graph v2;
- causal oracle and vision-estimated v2 tokens;
- staged comparisons that stop before expensive experiments when a prerequisite
  fails.

It does not:

- copy full MuJoCo state into the policy;
- use actions, future frames, future teacher goals, success, or termination as model
  inputs;
- claim language generalization from the current single-instruction dataset;
- immediately rerun the 4-condition by 3-seed main experiment;
- add SmolVLA or pi0 before ACT establishes a usable control floor.

## 5. Stage A: ACT control recovery

### 5.1 Paired deterministic sampling

The training loader shuffles frames with a seed-owned `torch.Generator`. Every Graph
condition for a policy seed receives the same epoch permutations. The run manifest
records the generator seed and an order hash for every epoch. A mismatch between
paired conditions aborts comparison.

This changes the current fixed sequence of adjacent frames without weakening paired
experimental control. Episode splits remain unchanged and episode-level.

### 5.2 Pretrained vision

ACT initializes its shared ResNet18 camera backbone with the official ImageNet
weights supported by the installed LeRobot/torchvision versions. Checkpoints and run
summaries record the exact weight identifier and a hash of the initialized backbone.
Flat and Graph conditions use byte-identical initial ACT parameters for a seed.

Missing weights fail with an actionable message instead of silently falling back to
random initialization. Network download is an explicit setup step; training and
evaluation remain local after the weights are cached.

### 5.3 Receding-horizon execution

Training continues to predict an eight-action chunk because chunk supervision is the
ACT representation being evaluated. Closed-loop rollout executes only the first
action, obtains new RGB, wrist RGB, proprioception, and Graph input, and queries ACT
again. Thus:

```text
chunk_size = 8
n_action_steps = 1
```

The first recovery experiment does not enable temporal ensembling, so its effect is
not confounded with the horizon correction. It may be evaluated later as a separately
named ablation.

### 5.4 Diagnostics and loss reporting

The bridge records LeRobot's actual `kld_loss` key. Each validation report also
contains:

- total, L1, and KL loss;
- translation MAE per axis for the first action and complete chunk;
- translation direction cosine and sign accuracy;
- rotation and gripper MAE;
- metrics partitioned by expert interaction phase;
- train-seen and held-out closed-loop termination counts.

No new action loss is introduced in the first recovery run. If the recovery gate
fails after the loader, visual initialization, and horizon fixes, phase-balanced
sampling is tried next. Demonstration scaling and recovery/DAgger data follow only if
that ablation still fails.

### 5.5 Recovery gates

The first run uses Flat ACT and one seed. It must satisfy both prespecified gates:

1. at least 8 successes over 10 exact train-seen `normal / 2-object` resets;
2. at least 30% success over a fixed held-out `normal / 2-object` seed set.

Expert rollouts must pass the same cases, and checkpoint reload must reproduce the
same policy outputs within the existing numerical tolerance. If either control gate
fails, Graph v2 ACT training stops and the report identifies the next bounded
recovery ablation.

## 6. Interaction Graph v2 contract

### 6.1 Design questions

The Graph is restricted to answering five policy questions:

1. which object is the target;
2. how the gripper currently relates to that target;
3. how the target relates to the receptacle;
4. which distractors can cause a wrong grasp or collision;
5. which interaction phase is active and which relation should change next.

It is task-conditioned, object-centered, invariant to the global coordinate frame,
temporally consistent, and estimable from policy observations.

### 6.2 Coordinate convention

Metric vectors are divided by a training-split workspace scale. They are represented
in a local interaction frame:

```text
gripper_to_target = R_gripper^T (p_target - p_gripper)
target_to_goal_action = R_gripper^T (p_receptacle - p_target)
target_in_receptacle = R_receptacle^T (p_target - p_receptacle)
```

Applying the same rigid transform to the whole scene leaves these values unchanged.
The first two vectors directly specify a direction compatible with the existing
gripper-local Cartesian action codec. The third describes placement geometry without
depending on world coordinates.

No absolute object pose, joint state, contact force, depth map, segmentation mask, or
MuJoCo body identifier enters the token.

### 6.3 Versioned 89D token

The new schema is `interaction_graph_control_v2` and has 89 float32 values:

| Group | Size | Contents |
|---|---:|---|
| entity presence | 6 | gripper, target, receptacle, support, two distractors |
| dual-view visibility | 12 | per-entity support from agent and wrist RGB |
| relation presence | 8 | active task-relation slots |
| gripper-target geometry | 8 | local delta XYZ, distance, closing speed, grasp probability, contact probability, confidence |
| target-receptacle geometry | 10 | action-frame delta XYZ, receptacle-frame offset XYZ, horizontal/vertical containment margins, placed probability, confidence |
| distractor geometry | 14 | for two distractors: local delta XYZ, clearance, collision risk, target-confusion risk, confidence |
| phase distribution | 6 | approach, grasp, lift, transport, place, release |
| relation trends | 4 | signed change of target distance, grasp confidence, goal distance, and minimum clearance |
| next relation | 8 | soft distribution over the relation to change |
| next operator | 5 | establish, break, increase, preserve, decrease |
| next predicate | 7 | existing proximity-through-clearance vocabulary |
| normalized residual | 1 | signed magnitude of the requested relation change |

Continuous confidence and categorical distributions remain soft. Missing or occluded
entities are masked; their metric regressions do not contribute to loss. A v1 token or
checkpoint can never be loaded as v2 by dimension alone: schema, ordered field names,
normalization, and provenance hashes must all match.

### 6.4 Causal temporal state

Phase labels and trends are derived only from the current and preceding observations.
The phase teacher uses current relation predicates with hysteresis; it does not expose
the scripted expert's internal phase variable. Trends use backward differences only.
The next-relation label is generated from the task template and current causal phase,
not from a future action or trajectory window.

This preserves the requirement that every Graph v2 field could, in principle, be
estimated during a real robot rollout.

## 7. Graph v2 estimator and transfer

The estimator receives only:

- current agent RGB;
- current wrist RGB;
- the existing 10D end-effector/proprioceptive observation, which contains the
  six-DoF end-effector pose representation plus gripper state;
- current task language;
- the previous predicted Graph state when temporal consistency is enabled.

Teacher sidecars provide label-only geometry during MuJoCo fine-tuning. The forward
input audit rejects actions, future frames, teacher graphs, simulator poses, contact
forces, segmentation, success, and termination fields.

ReflectVLM pretraining transfers the compatible visual/language trunk and semantic
heads. The new metric-geometry and phase/trend heads begin identically random in the
random and Reflect conditions and learn from MuJoCo labels. This keeps the intended
pretraining question honest: ReflectVLM supplies semantic interaction structure, not
unavailable metric supervision.

Training adds masked Smooth L1 geometry losses, phase cross-entropy, and a backward
temporal-consistency loss to the existing entity, relation, operator, predicate, and
residual objectives. Loss weights and normalization are fitted on the training
partition only and stored in the checkpoint.

## 8. Control conditions

All ACT conditions have the same 89D `observation.environment_state` interface and
identical parameter count:

1. `flat`: all 89 values are zero;
2. `oracle_graph_v2`: exact causal current geometry and relation state from MuJoCo,
   with no future information;
3. `predicted_random_v2`: frozen estimator initialized without ReflectVLM transfer;
4. `predicted_reflect_v2`: frozen estimator initialized from ReflectVLM-compatible
   weights.

The oracle is deliberately privileged and non-deployable. Its purpose is to test
whether the representation is useful before visual estimation error is introduced.

## 9. Staged experiment and stopping rules

Experiments run in this order:

1. recover Flat ACT with one seed and pass both Section 5.5 gates;
2. compare `oracle_graph_v2` against Flat on fixed paired normal cases;
3. only if oracle improves Flat, compare predicted Graph v2 against oracle;
4. only if predicted control is nonzero, compare Reflect initialization against
   random initialization;
5. only after these gates, run three policy seeds across normal/crowded and 2/3-object
   conditions.

Oracle Graph v2 must improve paired success over Flat by at least 10 percentage points
without increasing wrong-object stable grasp before the representation is considered
control-useful. If it does not, the Graph schema or ACT fusion is revised rather than
spending more compute on the visual predictor.

Predicted Graph reports both absolute success and the fraction of the
oracle-minus-Flat improvement it recovers. Reflect transfer is evaluated across seeds,
not from a selected checkpoint or a single favorable metric.

Crowded results remain out-of-distribution evidence while training data contains only
normal layouts. They must be labeled as such. A later balanced crowded-data experiment
is a separate data-generalization study.

## 10. Artifacts and provenance

New work writes to new destinations and never overwrites the completed v1 pilot:

```text
outputs/graph_control/act_recovery/
outputs/graph_finetune/mujoco_graph_v2/
outputs/graph_control/graph_v2_pilot/
```

Each report binds dataset, split, source, pretrained-backbone, graph schema,
normalization, cache, policy initialization, batch-order, checkpoint, and evaluation
seed hashes. Training destinations refuse nonempty directories unless an explicit,
safe resume contract matches every provenance field.

## 11. Verification

Network-free automated tests cover:

- deterministic shuffled order and identical order hashes across paired conditions;
- correct `kld_loss` reporting;
- eight-step training targets with exactly one executed rollout action per query;
- pretrained-backbone identity and byte-identical paired ACT initialization;
- 89D field order, masks, finite values, checkpoint incompatibility with v1, and
  forbidden-input rejection;
- invariance of every geometric field under random global rigid transforms;
- causal phase/trend construction with no future indexing;
- Graph-estimator output shapes, masked objectives, normalization isolation, and
  checkpoint reload;
- paired recovery/oracle/predicted aggregation and stopping rules.

The real macOS verification sequence is bounded: one-batch tests, a reloaded Flat
checkpoint, 10 train-seen rollouts, and the fixed held-out normal set. Full multi-seed
training is not presented as the next command until these gates pass.


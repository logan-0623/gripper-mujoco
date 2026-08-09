# Task-Conditioned Temporal Interaction Graph with Relational Goals

Status: approved design
Date: 2026-08-09

## 1. Research question

The current project has established `Graph > Flat` as an initial representation
comparison, but that comparison does not identify which graph information is useful.
The next study asks a narrower and falsifiable question:

> Which task-conditioned, object-centered, coordinate-invariant, temporally
> consistent, and visually estimable interaction relations improve robot-policy
> generalization and sample efficiency?

The graph must not reproduce the full MuJoCo state. It must answer five policy
questions:

1. Which object is the task target?
2. What is the current gripper-target relationship?
3. What is the current target-receptacle relationship?
4. Which distractors create wrong-grasp or collision risk?
5. Which interaction phase is active, how are the relations changing, and which
   relation should change next?

The resulting representation is called the Task-Conditioned Temporal Interaction
Graph with Relational Goals, abbreviated TC-TIG.

## 2. Scope and non-goals

The first study identifies valuable relation groups under controlled behavior
cloning. It does not optimize every component for maximum task success, train an
RGB perception stack, or introduce discrete action tokenization.

The task supplies the target and receptacle identities. A short history of visually
estimable graphs must infer the interaction phase and predict the next relation
goal. The first implementation extracts visual-observable quantities from simulator
ground truth, while explicitly excluding privileged signals, and then tests the
same representation under synthetic perception errors.

FAST-style discrete action tokens are a follow-up experiment. The first TC-TIG
study keeps the existing shared `H=8` continuous action chunk so that action
representation does not confound relation-value measurements.

## 3. Alternatives considered

### 3.1 Dense continuous relative graph

This option replaces absolute state with pairwise relative geometry but retains a
dense graph. It is simple, but it can still reproduce almost all simulator state,
does not state why a relation matters, and gives weak targeted ablations.

### 3.2 Typed temporal graph with relational goals

This is the selected design. It keeps only task-relevant entities and typed
relations, models relation histories, and predicts both a discrete relation
transition and its continuous residual. It preserves geometry while enabling
question-specific causal ablations.

### 3.3 Symbolic event graph

This option represents only predicates and phase events. It is interpretable, but
hard thresholds are brittle under visual noise and discard geometry needed for
fine Cartesian control.

## 4. System boundary and data flow

The data flow is:

```text
task instruction
  -> bind target and receptacle
  -> construct final task-goal relations

RGB-D or visual-observable simulator extractor
  -> detect and track entities
  -> select task-relevant entities
  -> construct current sparse typed graph G_t

causal graph history G_(t-W+1:t), W=8
  -> per-relation temporal encoders
  -> sparse typed message passing
  -> phase belief and next relation goal
  -> H=8 local-frame continuous action chunk
  -> shared action safety and MuJoCo controller
```

No rollout input may use future frames. Future demonstration frames are available
only to construct training labels.

## 5. Entity schema

The graph contains the following role-bound entities:

| Entity | Cardinality | Visual-observable node attributes |
| --- | ---: | --- |
| `gripper` | 1 | entity type, task role, aperture, aperture rate, confidence |
| `target` | 1 | entity type, target role, size, shape/symmetry descriptor, visibility, track confidence and age |
| `receptacle` | 1 | entity type, receptacle role, internal dimensions, entrance direction, visibility and confidence |
| `support` | 1 | entity type, support role, surface normal, visible boundary and confidence |
| `distractor` | `K=2` | entity type, distractor role, size, shape/symmetry descriptor, visibility, track confidence, age and risk score |

Nodes must not contain world-frame position, absolute quaternion, world-frame
velocity, MuJoCo body or joint identifiers, contact forces, stable-grasp flags,
success flags, or oracle phase labels.

Shape attributes use object-local dimensions and a symmetry/observability mask so
that an unobservable rotation axis does not produce a falsely precise orientation.

## 6. Dynamic distractor selection

All visible non-target objects are scored using four geometry-derived values in
`[0, 1]`:

1. target-neighborhood wrong-grasp risk;
2. occupancy of the gripper closing region;
3. approach swept-volume collision risk;
4. transport swept-volume collision risk.

The initial deterministic score is their equal-weight mean. The approach corridor
connects the current gripper pose to the task-bound target grasp region. The
transport corridor connects the tracked target pose to the task-bound receptacle
placement region. Both corridors use oriented bounding volumes enlarged by the
gripper/object safety margin, initially fixed at `0.01 m` for every primary
comparison.

The two highest-risk tracked objects become distractor nodes. A challenger replaces
the lowest-ranked retained distractor only when its normalized risk is at least
`0.10` higher for three consecutive frames. A missing retained track is propagated
with a constant-velocity estimate for at most three frames while its confidence
decays linearly to zero. These constants are part of checkpoint provenance.

Sensitivity to `K` is measured later with `K=1`, `K=2`, and `K=4`; it is not part
of the primary relation ladder.

## 7. Sparse typed relation schema

The graph contains canonical directed relation records. The encoder may create
reverse message directions internally, but the input does not create unrelated
all-pairs edges.

### 7.1 Manipulation relation: `gripper -> target`

This relation contains the target pose in the gripper frame, signed surface gap,
grasp-axis alignment error, finger-to-target clearances, relative linear and
angular motion, a continuous probability that the target lies in the closing
region, and a geometry-and-motion-derived co-motion score.

It must not contain force, simulator contact, held-object identity, or an oracle
stable-grasp indicator.

### 7.2 Placement relation: `target -> receptacle`

This relation contains the target pose in the receptacle frame, signed containment
margins, bottom gap, orientation error, relative motion, and soft containment and
support probabilities.

### 7.3 Support relation: `target -> support`

This relation contains bottom gap along gravity, projected overlap, relative
vertical velocity, and a soft support likelihood.

### 7.4 Risk relations: `distractor -> gripper` and `distractor -> target`

These relations contain closing-region distance, minimum signed swept-volume
clearance, time-to-collision when defined, target proximity, visual ambiguity, and
wrong-grasp and collision risk scores. The same relation encoder is shared across
all distractor slots.

### 7.5 Clearance relation: `gripper -> receptacle`

This relation contains the gripper pose in the receptacle frame, signed entrance
and wall clearances, retreat-height residual, and collision risk. It lets the
policy represent terminal release and retreat instead of treating placement as the
end of the task.

Every discrete predicate is accompanied by its continuous signed margin and an
observation confidence. Hard predicates are evaluation and label-generation tools,
not the sole policy input.

## 8. Coordinate invariance and action equivariance

All relation geometry is expressed in a physically meaningful local frame:

- gripper-target geometry uses the gripper frame;
- target-receptacle geometry uses the receptacle frame;
- risk geometry uses the approach or transport path frame;
- support geometry uses gravity-aligned height and projected overlap.

A passive global-coordinate reparameterization must leave the graph numerically
unchanged when all poses, frames, and gravity are transformed consistently.

An active physical transformation of a tabletop scene has translation and yaw
symmetry, not arbitrary `SE(3)` symmetry, because gravity and support are physical
directions. Tests therefore require action equivariance under scene translation
and rotation about gravity. They do not claim that physically rotating the table
away from gravity is an invariant task.

The policy predicts each 7D Cartesian action in the current gripper frame:

```text
[delta_position_gripper(3), delta_rotation_gripper(3), gripper_open(1)]
```

Execution converts the six-dimensional local pose delta into the controller's
convention. Flat, Graph, and every relation ablation share this conversion, IK
projection, temporal ensemble, and gripper hysteresis.

## 9. Temporal representation

The graph history length is `W=8` policy frames. Every edge record includes its
current value, observation confidence, visibility mask, first difference, robust
short-window trend, and the duration of active soft predicates.

Each relation type has a lightweight causal GRU. GRU weights are shared across
relations of the same type, including distractor slots. Current temporal relation
embeddings then enter a sparse typed message-passing encoder. The encoder emits a
global task embedding and one embedding for every active relation.

Entity identity comes from the tracker, never from a simulator object name.
Occlusion propagation and Top-K hysteresis preserve relation slots across short
observation failures. Every temporal computation is causal and covered by a
future-leakage test.

## 10. Relational goal and phase belief

The next relation goal is represented as:

```text
(active_edge, operator, predicate, continuous_residual, confidence)
```

The operator vocabulary is:

```text
establish, preserve, break, increase, decrease
```

The predicate vocabulary is:

```text
proximity, alignment, enclosure, co_motion,
containment, support, clearance
```

Examples include:

```text
(gripper->target, establish, alignment, grasp_pose_error)
(target->receptacle, establish, containment, containment_residual)
(gripper->target, break, co_motion, release_residual)
(gripper->receptacle, increase, clearance, retreat_residual)
```

Phase is not an input label. The model's phase belief is an interpretable projection
of the relation-goal distribution:

| Phase belief | Dominant relation change |
| --- | --- |
| approach | decrease proximity |
| align | establish alignment |
| grasp | establish enclosure or co-motion |
| transport/place | establish containment or support |
| release | break co-motion |
| retreat | increase clearance |

## 11. Demonstration-derived supervision

Each predicate has a continuous normalized relation error. At time `t`, the label
generator examines the next action chunk and computes:

```text
improvement(predicate) = error(t) - min(error(t+1:t+H))
```

The task template supplies only valid relation prerequisites and the final goal;
it does not supply the current phase. Among valid candidates, the label is the
confidence-weighted relation with the greatest normalized improvement. The
operator follows from the desired error direction, and the continuous residual is
the local relation change at the best future step. If no candidate clears the
minimum normalized improvement margin of `0.05`, the previous active relation
receives `preserve`.

The labeler may inspect future visual-observable relation values during training.
It must not read scripted-expert phase, MuJoCo contact forces, stable-grasp flags,
success, or termination reason.

The model is optimized with:

```text
L = L_action_chunk
  + lambda_discrete * L_relation_choice
  + lambda_residual * L_relation_residual
```

The action head consumes the predicted soft relation-goal embedding. It never
receives a teacher-forced goal at training time. Loss weights are selected once on
the validation split from the Cartesian product
`lambda_discrete, lambda_residual in {0.1, 0.3, 1.0}` and frozen across all primary
comparisons.

## 12. Controlled representation ladder

All primary variants share demonstrations, nested source subsets, batch ordering,
model seeds, temporal action head, controller, optimizer budget, and evaluation
cases.

| Variant | Representation change |
| --- | --- |
| `R0 Geometry` | Single-frame task binding plus manipulation, placement, support and clearance geometry |
| `R1 + Risk` | Add Top-K distractor risk entities and relations |
| `R2 + Temporal` | Add the tracked `W=8` relation history, trends, durations and confidence |
| `R3 + Goal` | Add the relation-goal head and auxiliary supervision; this is full TC-TIG |

The ladder uses the same typed encoder, with unavailable relation groups masked.
One additional parameter-matched comparison applies Flat and Graph encoders to the
same complete `R3` payload. This separates the value of relation information from
the value of message passing.

The existing `physics_v2` full-state graph is reported only as a privileged
historical reference and cannot support the main visual-interaction claim.

## 13. Targeted causal corruptions

The complete `R3` policy receives the following inference-time corruptions while
feature marginals and case seeds remain fixed:

| Policy question | Targeted corruption | Primary diagnostics |
| --- | --- | --- |
| Which object? | swap target and distractor roles | wrong-object interaction |
| Gripper-target relation? | shuffle manipulation edges across paired cases | bilateral contact, stable grasp and lift |
| Target-receptacle relation? | shuffle placement edges across paired cases | containment, support and transport progress |
| Which distractor is risky? | remove risk edges or shuffle distractor endpoints | crowded collision and wrong grasp |
| Which phase and trend? | reverse temporal order or freeze relation deltas | recovery, reclose and action oscillation |
| Which relation next? | shuffle active-edge or predicate logits | relation progress and strict success |

A relation group is not considered causally useful unless adding it helps and its
targeted corruption reverses the corresponding benefit.

## 14. Data scale and evaluation conditions

Sample efficiency uses nested training-source subsets of `40`, `80`, and `160`
base demonstrations. The discovery study uses three model seeds. A confirmation
study compares the selected full representation with the strongest baseline using
five model seeds. Evaluation seeds are fixed and paired across every policy.

The evaluation suite contains:

- in-distribution normal layouts;
- object-count OOD;
- crowded OOD;
- object size, orientation, and receptacle-placement OOD;
- held-out recovery interventions;
- synthetic perception perturbations.

The perception perturbation levels are:

| Level | Position sigma | Rotation sigma | Relative size sigma | Per-frame dropout | Per-frame ID switch |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean | 0 mm | 0 deg | 0% | 0% | 0% |
| low | 5 mm | 2 deg | 2% | 2% | 0.2% |
| medium | 10 mm | 5 deg | 5% | 5% | 1% |
| high | 20 mm | 10 deg | 10% | 10% | 3% |

All noise is applied before relation construction so that geometry, risk selection,
tracking, and confidence respond consistently.

## 15. Metrics and value criterion

Primary metrics are strict task success, bilateral target contact, stable target
grasp/lift, strict placement, wrong-object interaction, collision/near-miss,
transport progress, relation-progress monotonicity, and success-versus-data curve
area.

A relation group is designated valuable only when all of the following hold:

1. Its incremental addition produces a paired bootstrap 95% confidence interval
   excluding zero on its preassigned primary metric.
2. Its targeted deletion or shuffle produces a paired bootstrap 95% confidence
   interval excluding zero in the degradation direction on that same metric.
3. At least two of three discovery model seeds agree in direction.
4. The benefit remains under the medium perception perturbation level.
5. It does not regress ID success or a safety metric by more than three percentage
   points.

The five-seed confirmation result, not discovery screening, determines the final
claim. Each paired interval uses 10,000 case-level bootstrap resamples, shared
across paired policies.

## 16. Diagnostics and visualization

Every learned-policy step records:

- tracked entities, visibility, confidence and track age;
- Top-K risk ranking and replacement events;
- current relation values, trends and predicate durations;
- phase-belief distribution;
- relation-goal edge, operator, predicate, residual and candidate probabilities;
- relation progress;
- raw and executed action chunks;
- IK projection, action saturation and termination reason.

The dashboard overlays the active relation, largest-risk distractor, relation-goal
confidence and continuous residual. A failed rollout must be classifiable as a
perception/tracking, relation-selection, relation-geometry, action-control, or
terminal-condition failure.

## 17. Verification requirements

The implementation must include tests for:

- rejection of every forbidden privileged feature;
- invariance under passive global-coordinate changes;
- action equivariance under physical translation and yaw about gravity;
- invariance to distractor input ordering;
- temporal causality and absence of future leakage;
- relation-label generation without expert phase, force, success, or termination;
- stable identity across three-frame dropout;
- Top-K hysteresis and deterministic tie-breaking;
- masks and confidence under missing observations;
- targeted corruptions that preserve shapes and feature marginals;
- checkpoint provenance covering schema version, relation vocabulary, coordinate
  convention, tracker settings, risk settings and perception perturbations.

## 18. Acceptance boundary

The TC-TIG implementation is ready for the discovery experiment only after all
verification tests pass, the existing physical expert/data provenance gate remains
valid or is deliberately regenerated, and a smoke run produces complete relation
and action diagnostics without privileged inputs.

Implementation planning begins only after this specification is reviewed.

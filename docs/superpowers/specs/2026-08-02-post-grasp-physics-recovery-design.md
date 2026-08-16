# Post-Grasp Physics Recovery Augmentation Design

## Objective

Replace the current reset-only physical recovery augmentation with deterministic,
post-grasp interventions that teach the same Flat and Graph behavior-cloning
policies how to keep the target grasped, move it toward the receptacle, align it
over the receptacle, and release only at the goal.

This change addresses the observed seed-0 failure: the Graph policy achieved a
stable target lift, then moved away from the receptacle and opened the gripper
while still about 0.41 m from the goal. It does not change the 7D action space,
scene graph, Flat input, model architecture, contact physics, task success rule,
or evaluation cases.

## Chosen Approach

Each base episode in the training split produces exactly three recovery
attempts, one for each post-grasp intervention. With the pilot's approximately
40 training episodes, this yields at most approximately 120 accepted recovery
episodes. The three kinds are therefore balanced by construction instead of
depending on random seeds or Python hash order.

The alternatives rejected for this iteration are:

- one seeded intervention per source episode, which is faster but only
  approximately balanced and supplies too few post-grasp examples;
- loss reweighting of ordinary TRANSPORT frames, which cannot expose the policy
  to wrong-way, partially opened, or misaligned physical states.

## Recovery Specification

`PhysicsRecoveryKind` is replaced with three values:

1. `wrong_way_transport`;
2. `premature_open`;
3. `receptacle_misalignment`.

`PhysicsRecoverySpec` records the source seed, globally unique variant ID,
recovery kind, trigger phase, deterministic planar direction, translation step
count, and gripper-open simulation substep count. All generated vectors and
counts are validated and included in episode metadata.

`make_physics_recovery_spec(source_seed, variant_id)` remains deterministic. The
kind is selected by `variant_id % 3`. Since collection allocates three
consecutive variant IDs per training source episode, every source receives one
of every kind. Direction signs and intervention magnitudes use a NumPy generator
seeded only from `source_seed`, `variant_id`, and a fixed namespace constant.

## Intervention Semantics

All interventions occur after a real bilateral, stable target grasp. They use
the existing Cartesian controller and MuJoCo stepping; they never modify object
qpos, create welds, use suction, or disable contact dynamics.

### Wrong-way transport

The trigger is the first valid TRANSPORT state while the target remains stably
grasped and is more than 0.15 m from the receptacle center. The collector applies
three closed-gripper Cartesian actions whose XY direction has negative dot
product with the target-to-receptacle direction. The total commanded lateral
displacement is 0.06 m. Recording starts after the intervention, when the expert
must command motion back toward the receptacle while keeping the gripper closed.

### Premature open

The trigger is a valid TRANSPORT state more than 0.15 m from the receptacle. The
collector holds Cartesian pose, commands the gripper open, and advances one raw
MuJoCo substep (2 ms at the configured 500 Hz physics rate) rather than a
full 20 Hz policy step.
This produces a physically partially opened finger state while limiting the
chance of an unrecoverable drop. Recording begins immediately afterward and the
expert's first label must close the gripper.

The one-substep duration is evidence-based: for the deterministic seed-11
contact state, the finger gap increases while bilateral contact remains; at two
substeps bilateral contact is already lost. The intervention is accepted only
if finger separation increased, the target
has not contacted the table or receptacle, and bilateral target contact remains.
Otherwise the recovery attempt is rejected instead of silently storing an
unrecoverable or unchanged state.

### Receptacle misalignment

The trigger is a TRANSPORT state with a stable target grasp and target-to-goal XY
distance at most 0.10 m. Before the expert can release, the collector applies
two closed-gripper Cartesian actions tangential to the current target-to-goal
direction, producing a 0.04 m lateral error without moving away along the radial
goal direction. Recording begins after the intervention. The expert must move
back to the receptacle center, release there, and retreat.

## Collection Data Flow

`collect_physics_episode` performs a normal environment and expert reset even
when a recovery specification is present. The collector runs the unperturbed
expert prefix without saving policy frames until the recovery trigger is met.
It then applies the unlabelled intervention, validates its postconditions, and
starts saving pre-action observations paired with expert corrective actions.

For base episodes, collection remains unchanged and saves the full trajectory.
For recovery episodes:

1. the pre-grasp prefix is omitted;
2. intervention actions are not stored as demonstrations;
3. the first stored observation is the post-intervention physical state;
4. all stored actions are expert corrections;
5. only trajectories ending in task success enter `recovery_manifest.json`;
6. missing triggers, failed intervention postconditions, drops, timeouts, IK
   failures, and other physics failures enter `recovery_rejections.json`.

Every accepted recovery archive must contain at least one frame. Its metadata
records `trajectory_kind="recovery"`, `source_seed`, `variant_id`,
`perturbation_kind`, the actual trigger phase, intervention parameters, and the
current expert-gate hash.

## Expert Behavior

The expert continues to use observable state and contact diagnostics. It gains
post-grasp resynchronization rules applied before phase-specific control:

- a stable or bilateral target away from the receptacle selects LIFT when too
  low and TRANSPORT when high enough;
- TRANSPORT always commands a closed gripper;
- a partially opened but still bilaterally held target is immediately commanded
  closed before transport continues;
- RELEASE is permitted only when the held target is within the receptacle's
  center tolerance;
- loss of the target follows the existing physical regrasp/retry path and cannot
  be labelled as successful transport.

No recovery kind or expert phase is added to policy observations.

## Evaluation Metrics

Each rollout records three new episode-level values:

- `transport_progress`: after the first stable target grasp, the non-negative
  reduction in target-to-receptacle XY distance from that starting distance to
  the minimum later distance, measured in meters;
- `transport_progress_rate`: `transport_progress / start_distance`, clipped to
  `[0, 1]`; episodes without a stable target grasp receive zero for both values;
- `premature_open`: true when the policy commands `g >= 0.5` after its first
  stable target grasp while the target is farther than 0.065 m in XY from the
  receptacle center.

Aggregates expose mean transport progress, mean transport progress rate, and
premature-open episode rate overall, by policy, and by condition. Paired reports
add Graph-minus-Flat and Graph-minus-edge-shuffle deltas for transport progress
rate and premature-open rate. A lower premature-open delta is favorable, while
higher transport progress is favorable.

These metrics diagnose partial improvement; they do not replace task success as
the primary endpoint.

## Configuration and Provenance

`configs/physics_recovery_pilot_macos.yaml` changes
`recovery.variants_per_episode` from 1 to 3. The smoke recovery configuration is
updated consistently so the end-to-end smoke test covers all three kinds with a
small dataset.

The implementation changes provenance-hashed recovery, environment, expert, and
data modules as well as the pilot configuration. Existing expert gates, episode
archives, recovery manifests, checkpoints, and evaluation reports are therefore
stale. The user must rerun expert validation and collect a fresh dataset before
training seed 0. New files may replace files at the configured output paths;
existing artifacts are not deleted by the implementation.

## Expected Module Changes

```text
interaction_vla/physics_recovery.py   three deterministic post-grasp specs
interaction_vla/physics_env.py        bounded unlabelled physics intervention API
interaction_vla/physics_expert.py     post-grasp observable-state resynchronization
interaction_vla/physics_data.py       trigger, intervention, suffix-only collection
interaction_vla/physics_evaluate.py   transport and premature-open metrics
configs/physics_recovery_pilot_macos.yaml
configs/physics_recovery_smoke_macos.yaml
tests/interaction_vla/test_physics_recovery.py
tests/interaction_vla/test_physics_env.py
tests/interaction_vla/test_physics_expert.py
tests/interaction_vla/test_physics_data.py
tests/interaction_vla/test_physics_evaluate.py
tests/interaction_vla/test_smoke_pipeline.py
README.md
```

## Testing and Acceptance

Automated tests must prove, through red-green TDD:

- deterministic, balanced construction of the three recovery specs;
- each trigger rejects the wrong phase, missing stable grasp, or invalid goal
  distance;
- unlabelled interventions advance MuJoCo without incrementing the policy-step
  counter or directly modifying object qpos;
- premature-open intervention increases finger separation while retaining the
  target for an accepted deterministic smoke seed;
- recovery archives contain only post-intervention frames and the first label
  keeps the gripper closed while either restoring safe lift height or correcting
  transport toward the receptacle;
- all three recovery kinds can produce successful accepted trajectories on fixed
  smoke seeds;
- manifest counts, unique variant IDs, split isolation, metadata, and rejection
  logging remain auditable;
- evaluation metrics and paired deltas have exact synthetic expected values;
- the physical smoke pipeline covers collection, augmentation, Flat/Graph
  training, and checkpoint validation;
- the full test suite and Python compilation pass.

Before handing off training commands, run a small deterministic real-MuJoCo
collection check and report accepted/rejected recovery counts by kind. The
implementation is not allowed to claim that learned success improves before the
user retrains and evaluates the new checkpoints.

## Non-goals

- DAgger or learned-policy rollouts during data generation;
- changing Graph/Flat feature content or network capacity;
- adding RGB inputs or VLA training;
- changing contact friction, grasp thresholds, placement thresholds, or task
  success criteria to improve scores;
- adding suction, weld constraints, deterministic attachment, or object-state
  edits;
- claiming a paper-level Graph advantage from seed 0.

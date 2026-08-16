# Terminal Release-and-Retreat Recovery Design

## Goal

Create a clean second physical dataset that teaches Flat and Graph policies to
finish the task after placing the target in the receptacle. The target behavior
is: keep the fingers open, move the TCP vertically away from the placed object,
and satisfy the existing physical success rule. The experiment continues to
compare representations under identical observations, actions, demonstrations,
training settings, IK projection, and paired evaluation cases.

This is a data-quality correction, not a new controller or a hand-coded success
reflex. All object motion continues to emerge from MuJoCo contact physics.

## Evidence and Root Cause

The deterministic Graph seed-0 ID case `id_normal:2:2057736129` reaches stable
placement at policy step 112 with the fingers open. At step 119 the learned
policy closes the fingers again. It then remains close to the receptacle until
the 180-step timeout. The rollout has no IK projection or physics failure in the
terminal segment.

The current dataset contains release and retreat frames, but it does not contain
the learned failure state: target supported by the receptacle, TCP too low, and
fingers reclosed. It also records `expert.phase` after `expert.act()`. Because
`act()` may transition the state machine while producing the previous phase's
action, some closed transport actions are labeled `release`, and some initial
open-in-place actions are labeled `retreat`. Phase labels are used for inverse
frequency sample weighting, so this timing error makes the terminal supervision
less precise even though phase is not a policy input.

## Chosen Approach

Build a new dataset from scratch with two coordinated changes:

1. record the action's phase before calling `expert.act()`; and
2. add a deterministic `post_placement_reclose` recovery kind.

The existing three recovery kinds remain unchanged. The new kind is the fourth
variant for every training-split source episode. Flat and Graph consume exactly
the same resulting files and sample weights.

## Correct Phase-to-Action Alignment

For every recorded frame, collection performs this order:

```text
snapshot and diagnostics
-> action_phase = expert.phase.value
-> action = expert.act(...)
-> store snapshot, action, and action_phase
-> env.step(action)
```

This means a transport action that causes the expert to enter `release` remains
labeled `transport`; an open-in-place action that enters `retreat` remains labeled
`release`; and upward open actions are labeled `retreat`. Prefix actions that are
discarded before a recovery intervention remain unrecorded.

## Terminal Recovery Specification

Add `PhysicsRecoveryKind.POST_PLACEMENT_RECLOSE` after the three existing enum
members. Its deterministic contract is:

- trigger phase: `retreat`;
- target-to-receptacle XY distance: at most `0.065 m`;
- target must be in receptacle contact;
- target no longer needs a stable bilateral grasp;
- intervention: five full policy steps of
  `[0, 0, -1, 0, 0, 0, 0]`;
- resulting commanded descent: at most `0.10 m` before controller/workspace
  limits;
- fingers must measurably close;
- the target must remain receptacle-supported and must not become a stable
  gripper grasp.

The intervention uses `FrankaContactEnv.advance_intervention` with the normal 25
MuJoCo substeps per policy step. It does not write object qpos, create a weld, or
use suction. If target support is lost, a stable grasp appears, the fingers do not
close, or a physical failure occurs, the attempt is rejected with a specific
reason.

After the intervention the expert remains in `retreat`. Collection begins at the
perturbed state, so the first retained targets are open-gripper, upward Cartesian
actions. Recording continues until the unchanged environment success condition
is reached. The approach, grasp, transport, and intervention frames are not
written into this recovery episode.

`PhysicsRecoverySpec` gains `close_descent_steps`. Existing recovery kinds require
it to be zero; `post_placement_reclose` requires exactly five. Its metadata is
stored in every recovery NPZ and manifest record.

## Dataset Quality Gate

`RecoveryConfig` gains `min_acceptance_rate`, defaulting to `0.0` for backward
compatibility and validated within `[0, 1]`. The new pilot config sets it to
`0.80`.

Collection counts attempted and accepted trajectories separately for each
recovery kind. After all attempts, every configured kind must:

- appear in `recovery_quality.json`, including an explicit zero-attempt entry;
- have at least one accepted successful trajectory; and
- meet `accepted / attempted >= min_acceptance_rate`.

Failure raises with the per-kind counts and leaves the manifest and rejection log
available for diagnosis. The tqdm display continues to advance once per attempt
and includes the current kind plus accepted/rejected totals.

Each unlabelled intervention step returns its controller diagnostics and any
physics-failure subtype. Collection rejects an IK-limited intervention or a
physics failure immediately, preserving the subtype in the rejection reason
instead of allowing the next labelled policy step to hide it.

The base-data contract remains 50 successful normal-layout episodes with object
counts 2 and 3. Only source episodes in the deterministic training split generate
recovery trajectories; validation and test sources remain augmentation-free.

## Isolated Configuration and Provenance

Add two configurations:

- `configs/physics_terminal_recovery_smoke_macos.yaml` for a one-epoch pipeline
  check; and
- `configs/physics_terminal_recovery_pilot_macos.yaml` for 50 base episodes,
  four recovery variants per training source, 80 epochs, and model seeds 0/1/2.

The pilot writes only below:

```text
outputs/interaction_graph_physics/terminal_recovery_pilot/
```

The current `recovery_pilot` gate, data, checkpoints, ID sanity report, and full
report remain untouched.

Changing the collector and recovery generator changes the controller provenance
hash by design. The new configuration therefore requires a new expert gate before
collection. Every new episode stores that gate hash, training provenance hashes
the new manifest and selected NPZ contents, and old checkpoints are rejected for
the new experiment.

## Training and Evaluation

The first comparison trains only model seed 0 for Flat and Graph. Architectures,
normalization, optimizer, batch size, epochs, phase balancing, 7D action head, and
IK-safe evaluation remain unchanged.

Before any OOD evaluation, run the paired `id_normal` sanity set with five episodes
per object count and write it to an isolated report. Report the existing control,
stable-lift, placement, and success metrics. Add
`post_placement_reclose_rate`: the fraction of episodes in which a learned action
commands closed fingers after stable placement has already occurred.

The v2 seed-0 checkpoint is useful enough to proceed when:

- both representations have physics-failure rate at most 10%;
- Graph stable-lift and placement rates are at least 10%;
- Graph achieves at least one real success in the ten ID episodes; and
- no success is created by an evaluation-time release or retreat override.

This is an engineering gate, not evidence that Graph is superior. If placement
increases but success remains zero, the next experiment should add temporal
context or action chunks rather than more copies of terminal frames.

## Error Handling

- Invalid recovery-kind parameter combinations fail before simulation.
- Terminal recovery cannot trigger without receptacle support.
- An intervention that disturbs the placed object is rejected, not retained.
- A non-successful corrective suffix is written only to the rejection log.
- Per-kind acceptance failure stops collection before training.
- Collection and training never delete or overwrite the existing experiment.

## Testing

Tests cover:

- phase labels refer to the phase active before action generation;
- all four recovery specs are deterministic and balanced per source;
- the terminal trigger requires retreat plus receptacle support;
- the terminal intervention closes and lowers the real Franka gripper without
  moving object qpos directly;
- retained terminal recovery begins with open/up action and reaches success;
- per-kind acceptance gates reject missing or low-quality recovery groups;
- smoke collection writes all four kinds with valid provenance;
- Flat and Graph select identical new episode manifests;
- evaluation records post-placement reclose behavior; and
- the full existing test suite remains green.

## Non-Goals

- No scripted release/retreat override during learned rollout.
- No change to the environment success condition.
- No action history, recurrent state, transformer, or action chunks in this data
  iteration.
- No RGB/VLA input change.
- No Graph-only supervision or representation-specific recovery data.
- No OOD or paper-level claim until the ID completion gate is passed.

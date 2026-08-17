# Non-finite contact collection recovery

## Problem

During LeRobotDataset collection, MuJoCo can rarely return a non-finite value
from `mj_contactForce` even though the collector has already committed earlier
episodes. `FrankaContactEnv.step` currently constructs strict
`InteractionSignal` values before it can classify that substep as a physics
failure. The schema therefore raises `ValueError`, aborting the entire
collection and leaving the dataset marked `INCOMPLETE`.

FFmpeg's `moving the moov atom` message is normal MP4 finalization and is not
part of this failure.

## Selected design

Treat a non-finite MuJoCo contact force as a typed physics failure. The contact
parser will detect the invalid value and raise a dedicated exception instead
of clamping or dropping the contact. The environment step and intervention
paths will translate that exception into the existing
`TerminationReason.PHYSICS_FAILURE` contract with the reason
`non_finite_contact_force`.

The collector already rejects every non-success termination: it clears the
uncommitted episode buffer, records the rejected attempt, and advances to the
next deterministic seed. Previously committed episodes remain unchanged.

## Scientific constraints

- Never replace `NaN` or `Inf` force with zero.
- Never accept a rollout whose contact signal became invalid.
- Preserve strict finite-force validation in `InteractionSignal`.
- Preserve all successful-rollout behavior, seeds, labels, and thresholds.
- Do not add partial-dataset resume support in this fix.

## Error handling

The typed exception is caught only at the physics-environment boundary. Other
`ValueError` and schema failures continue to abort, so real programming or
data-contract errors are not hidden. A failure transition uses the last valid
snapshot and reports `physics_failure=non_finite_contact_force`.

## Verification

Regression tests will prove that:

1. the contact parser rejects a non-finite MuJoCo force with the typed error;
2. `FrankaContactEnv.step` returns a physics-failure transition rather than
   raising;
3. the collector clears and rejects such an episode through its existing
   non-success path;
4. the complete interaction_vla test suite still passes.

The existing 23-episode server dataset remains an incomplete artifact and must
be moved aside before a fresh collection with the repaired code.

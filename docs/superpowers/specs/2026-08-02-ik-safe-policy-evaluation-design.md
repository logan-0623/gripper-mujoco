# IK-Safe Learned-Policy Evaluation Design

## Objective

Make learned Flat and Graph policy rollouts reach a meaningful manipulation
outcome instead of terminating primarily because a predicted Cartesian delta is
temporarily outside the Franka IK tolerance. The same deterministic safety layer
must be applied to both representations, and the report must expose how often it
intervened so that control difficulty is not mistaken for representation quality.

This change does not claim to repair behavior-cloning covariate shift. It isolates
IK reachability from representation comparison and adds a small ID-only sanity
workflow that shows whether either learned policy can actually lift an object.

## Root-Cause Evidence

The completed 80-rollout report has a 93.75% aggregate physics-failure rate, with
every recorded physics failure classified as `ik_limited`. On the same
`id_normal:2:3178023727` case:

- Flat predicts persistent, often saturated positive X/Z motion, reaches 20
  consecutive IK-limited steps, and terminates at step 143.
- Graph predicts persistent positive X/Z motion plus increasing rotation, reaches
  20 consecutive IK-limited steps, and terminates at step 99.
- The scripted expert uses the same environment and controller, has zero
  IK-limited steps, and succeeds at step 156.

A no-source-change probe tried pose scales `(1.0, 0.5, 0.25, 0.125, 0.0)` before
each learned action. Both policies then reached the 180-step timeout with no
physics failure. Neither achieved a stable lift in that case. This confirms that
deterministic backtracking removes the control termination without fabricating a
successful manipulation result.

## Considered Approaches

### 1. Evaluation-layer deterministic IK backtracking — selected

Preview the raw learned action at decreasing pose scales and execute the first
IK-feasible candidate. Preserve the gripper command. Apply the identical scale
schedule to Flat and Graph, log every intervention, and use the same helper in
learned-policy visualization.

This keeps the current expert gate, dataset, and checkpoints valid because neither
the physical controller nor any module in `_PHYSICS_CONTROL_MODULES` changes. Its
limitation is intentional: a policy that repeatedly asks to leave the reachable
workspace will stall and time out rather than be declared successful.

### 2. Controller-level IK projection

Put backtracking inside `FrankaCartesianController.apply_action`. This would make
the behavior universal for expert, teleoperation, collection, and evaluation, but
it changes the hashed physical controller. The gate would become stale and the
strict chain would require expert revalidation, data recollection, and retraining.
That cost is not justified for the current representation pilot.

### 3. Diagnostics followed immediately by DAgger or retraining

Keep raw-policy evaluation untouched and collect recovery data from learned-policy
failures. This addresses covariate shift directly, but the current Graph/Flat
comparison remains dominated by IK termination until a new dataset and two new
models are ready. It is a future training improvement, not the smallest valid
next experiment.

## Architecture

Create `interaction_vla/physics_action_safety.py`, which is intentionally absent
from `_PHYSICS_CONTROL_MODULES`. It exposes:

```python
@dataclass(frozen=True)
class IKProjectionResult:
    raw_action: np.ndarray
    action: np.ndarray
    scale: float
    raw_diagnostics: ControllerDiagnostics
    projected_diagnostics: ControllerDiagnostics


def project_cartesian_action(
    controller: FrankaCartesianController,
    action: np.ndarray,
    *,
    scales: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125, 0.0),
) -> IKProjectionResult:
```

The function validates a finite 7D action and a strictly decreasing scale schedule
that starts at `1.0` and ends at `0.0`. For each scale it multiplies only the six
pose coordinates, preserves the gripper coordinate, and calls the existing
controller IK path without advancing MuJoCo physics. It selects the first
diagnostic with `ik_limited=false`. The caller immediately sends the selected
action through `env.step`, which reapplies it before any physics step, so preview
control writes cannot affect simulated motion. A zero-pose action is expected to
be feasible; if it is not, the helper raises instead of silently executing an
unsafe command.

`physics_evaluate.py` enables the projector by default for Flat, Graph, and Graph
edge-shuffle variants. `physics_visualize.py` uses it only for learned controllers;
expert and teleoperation behavior remain unchanged.

## Diagnostics and Report Semantics

Each `PhysicsEpisodeResult` records:

- `action_saturation_rate`: fraction of policy steps with any raw pose component
  at absolute value at least `0.95`;
- `ik_projection_rate`: fraction of steps whose selected scale is below `1.0`;
- `zero_pose_projection_rate`: fraction whose selected scale is `0.0`;
- `mean_ik_projection_scale`: mean selected scale; and
- the existing terminal reason and physical manipulation outcomes.

Aggregate metrics include the four fields above and a
`termination_reason_counts` mapping. The evaluation scope records
`ik_projection_enabled`, the exact scale schedule, and selected conditions. A
report therefore distinguishes “Graph manipulates better” from “Graph merely
needed less safety intervention.”

The projector can be disabled with `--disable-ik-projection` to reproduce the old
raw-policy behavior. This flag applies to every learned representation in the run;
there is no per-representation setting.

## ID Sanity Workflow

Add `--conditions` with choices `id_normal`, `count_ood`, `crowded_ood`, and
`controlled_randomization`. Filtering happens after deterministic case generation,
so paired seeds and initial states do not change.

Add optional `--output` so an ID-only report does not overwrite the full report.
The recommended command is:

```bash
.venv/bin/python -m interaction_vla.physics_evaluate \
  --config configs/physics_recovery_pilot_macos.yaml \
  --model-seeds 0 \
  --episodes-per-count 5 \
  --conditions id_normal \
  --output outputs/interaction_graph_physics/recovery_pilot/evaluation/id_sanity_report.json
```

The report adds `learned_policy_sanity` per policy:

- `control_passed`: physics-failure rate is at most 10%;
- `manipulation_passed`: stable-lift rate is at least 10%; and
- `passed`: both conditions hold.

These are pilot engineering gates, not statistical significance claims. If control
passes but manipulation fails, the next step is recovery-data collection or
closed-loop imitation learning rather than more OOD evaluation.

## Error Handling and Provenance

- Invalid actions or scale schedules fail before a rollout step.
- Projection never changes gripper state and never edits object state.
- Both representations always share the same projection configuration.
- The report records whether projection was enabled, preventing accidental
  comparison with the previous raw-policy report.
- No config, controller, environment, expert, scene, dataset, gate, or checkpoint
  file is modified. The current provenance chain remains valid.

## Testing and Verification

Tests first prove that the projector:

- returns the raw action when full scale is feasible;
- selects the first feasible lower scale;
- preserves the gripper command;
- rejects invalid schedules; and
- raises when even zero pose is IK-limited.

Evaluation tests cover diagnostics aggregation, condition filtering, parser/main
forwarding, output routing, projection disablement, and paired policy use. Viewer
tests cover learned-only projection and overlay reporting. Existing expert,
controller, provenance, evaluation, and migration tests must remain green.

After the full suite passes, run the ID sanity command above. Success means the
production report has low physics-failure rates and transparently reports the
projection burden. A nonzero stable-lift result is evidence to continue; zero
stable lift is a diagnosed learning limitation, not a reason to weaken success
criteria.

## Non-Goals

- Improving checkpoint weights without retraining.
- Claiming Graph superiority from safety-filtered control alone.
- Changing success, contact, stable-grasp, or placement definitions.
- Modifying the hashed physical controller or migrating provenance again.
- Automatically suppressing OOD evaluation based on the sanity result.

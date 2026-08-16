# Crowded OOD and Deterministic Recovery Augmentation Design

## Objective

Extend the existing state-only Graph-versus-Flat behavior-cloning experiment in two ordered stages:

1. evaluate the already-trained policies on a harder, deterministic interaction-crowding OOD condition;
2. add deterministic recovery demonstrations generated only from the training split, retrain the same policies, and evaluate them on the unchanged cases.

The extension must preserve the original representation contract. Graph and Flat continue to receive identical privileged current-state information, use the same data, batches, shared action head, optimizer settings, training seeds, and paired evaluation cases. No visual model, world model, obstacle system, or SmolVLA modification is introduced.

## Scientific Questions

The extension separates three questions that must not be conflated:

1. Does interaction crowding make object selection harder than object-count OOD alone?
2. Does deterministic recovery data improve closed-loop grasping for both representations?
3. Under the same crowded cases and recovery data, does Graph retain an advantage over Flat, and does edge shuffling remove that advantage?

Stage A answers the first question without retraining. Stage B answers the second and third questions using new checkpoints and exactly the same evaluation cases.

## Crowded OOD Scene Generation

### Layout modes

`KinematicTabletopEnv.reset()` gains an explicit layout mode:

- `normal`: the existing sampler with a minimum pairwise center distance of `0.12` meters;
- `crowded`: a deterministic target-centered cluster used only for evaluation.

The default remains `normal`, so existing collection and tests keep their behavior.

### Deterministic crowded sampler

For `crowded` reset:

1. initialize one NumPy random generator from the environment seed;
2. choose the target index before placing objects;
3. sample the target in the existing spawn region;
4. choose one non-target object as the anchor distractor;
5. sample its polar angle from the same generator and place its center between `0.085` and `0.105` meters from the target center;
6. reject placements outside the spawn region;
7. place remaining objects with at least `0.12` meters separation from every already placed non-anchor object and at least `0.085` meters from the target/anchor pair;
8. fail with an actionable `RuntimeError` after 500 deterministic attempts.

The target and anchor cubes therefore do not overlap, but small Cartesian prediction errors can move the gripper closer to the distractor. The layout does not modify the target flag, grasp radius, action bounds, or success rule.

The same `(seed, object_count, layout_mode)` must produce numerically identical snapshots. The scripted expert must retain 100 percent success on a deterministic smoke set of crowded scenes; otherwise a scene is invalid as an evaluation case rather than meaningfully difficult.

## Evaluation Conditions

`EvaluationCase` gains a named condition and layout mode. Evaluation contains:

- `id_normal`: normal layouts with two and three objects;
- `count_ood`: normal layouts with four and five objects;
- `crowded_ood`: crowded layouts with four and five objects.

Each condition has its own deterministic seed namespace, preventing accidental scene reuse while keeping every case identical across policies and model seeds. The existing 20 episodes per object count remain the pilot default.

Every `EpisodeResult` records condition, layout mode, object count, environment seed, representation, model seed, and ablation. Reports add:

- metrics by policy and condition;
- metrics by policy, condition, and object count;
- paired Graph-minus-Flat success deltas by model seed and condition;
- wrong-object deltas for crowded OOD;
- edge-shuffled Graph results on the exact same cases.

The existing count-OOD result remains available, so increased difficulty can be attributed to interaction crowding rather than merely more objects.

## Stage A: Frozen-Checkpoint Complexity Baseline

Stage A uses the six existing pilot checkpoints—Flat and Graph for model seeds 0, 1, and 2—without any training or normalization change. A new `configs/crowded_ood_macos.yaml` points evaluation at a separate output directory so the original report remains intact.

The Stage A report establishes:

- how much crowded OOD lowers Flat and Graph success relative to normal four-/five-object count OOD;
- whether Graph reduces wrong-object selection in crowded scenes;
- whether edge shuffling specifically damages crowded-scene behavior.

This baseline must be produced before recovery data are generated.

## Recovery Augmentation

### Split isolation

Recovery trajectories are generated only after the base episode-level split has been computed. The augmentation command reads the authoritative base manifest, recomputes the split with the same configuration seed and fractions used by training, and writes that source split beside the recovery manifest. It accepts only source seeds in the training set. Validation and held-out test episodes are never perturbed or copied into recovery data.

Recovery data live in `recovery_manifest.json`, separate from `manifest.json`. Training loads the base training paths plus recovery paths. Normalization statistics and phase weights are refit on that combined training-only set. Validation, held-out action evaluation, and all closed-loop evaluation cases remain unchanged.

The command is:

```bash
python -m interaction_vla.data augment-recovery --config configs/recovery_macos.yaml
```

Running the command twice with the same configuration must reproduce identical arrays and replace the recovery manifest atomically.

### Recovery trajectory construction

Each base training seed produces one recovery variant in the pilot. The perturbation family is selected by a stable round-robin mapping of `(source_seed, variant_id)` so the dataset has balanced coverage rather than depending on Python hash order.

The four perturbation families are:

1. `align_offset`: during ALIGN, move the open gripper laterally by `0.04–0.06` meters and vertically upward by `0.02–0.04` meters;
2. `failed_close`: immediately before CLOSE, move the gripper laterally outside the grasp radius and set it closed without attaching an object;
3. `lift_offset`: while holding the target during LIFT, move the gripper and attached object laterally by `0.03–0.05` meters and downward by `0.02` meters;
4. `transport_offset`: while holding the target during TRANSPORT, move the gripper and attached object laterally by `0.05–0.07` meters away from the receptacle path.

Signs, exact magnitudes, and injection step are derived from a NumPy generator seeded from `source_seed` and `variant_id`. Perturbations change simulator state without consuming an expert-labelled action step. A held object always moves consistently with the gripper.

Only post-perturbation recovery frames are saved. This avoids duplicating the unperturbed prefix and prevents ordinary movement frames from overwhelming the recovery signal.

### Recovery-aware expert

The scripted expert gains state-consistency guards:

- if it is in CLOSE without holding the target and the gripper is no longer aligned, it returns to ALIGN and opens the gripper;
- if the gripper is closed without holding anything, it opens while moving back toward the appropriate target waypoint;
- if it is holding the target below lift height, it selects LIFT;
- if it is holding the target away from the receptacle, it selects TRANSPORT;
- it releases only while holding the target over the receptacle.

These guards also make the expert robust to synthetic states without adding expert phase or privileged future information to policy inputs.

Every recovery trajectory must terminate in success within the configured step limit. Failed recovery generation is written to `recovery_rejections.json` and excluded from the recovery manifest.

### Storage metadata

Episode metadata gains optional fields with backward-compatible defaults:

- `trajectory_kind`: `base` or `recovery`;
- `source_seed`;
- `variant_id`;
- `perturbation_kind`;
- `injection_phase`.

The scene graph, proprioception, and action tensors remain unchanged, so Graph and Flat still consume exactly the same inputs.

## Stage B: Recovery Training and Fixed Evaluation

`configs/recovery_macos.yaml` uses:

- `outputs/interaction_vla/pilot/data` as the unchanged authoritative base dataset and writes recovery files beside it under a separate filename prefix and manifest;
- the same 50 base collection attempts and episode split as the pilot;
- two- and three-object normal layouts only;
- one recovery variant per base training episode;
- the same encoder dimensions, optimizer, 80 epochs, and model seeds 0, 1, and 2;
- the exact Stage A `id_normal`, `count_ood`, and `crowded_ood` evaluation case seeds.

Outputs are isolated under `outputs/interaction_vla/recovery/`. The experiment trains proprioception-only, Flat, and Graph policies, although the primary paired comparison remains Graph versus Flat.

The final report contains both augmentation stages. It records, for every model seed:

- recovery-minus-baseline success change for each representation and condition;
- Graph-minus-Flat success delta after recovery training;
- crowded wrong-object rate;
- held-out physical and normalized action MSE;
- Graph versus edge-shuffled Graph deltas.

## Interpretation Criteria

The extension is exploratory. It must report results even when they do not favor Graph.

Evidence that the new OOD condition is meaningful requires:

- the scripted expert succeeds on all accepted crowded cases;
- at least one learned representation performs worse on `crowded_ood` than on `count_ood` before recovery training.

Evidence that recovery augmentation helps requires a positive mean crowded-OOD grasp or success change over the corresponding Stage A checkpoint. Per-seed values must be shown; an average alone is insufficient.

The existing representation criterion remains strict:

- at least three model seeds;
- mean Graph-minus-Flat crowded-OOD success improvement of at least 10 percentage points;
- every seed improves in the same direction;
- no material ID regression;
- edge shuffling lowers Graph crowded-OOD success.

If the criterion is not met, the report must say so directly. Lower wrong-object rate or lower offline error may be described only as a narrower object-awareness signal.

## Configuration and Module Changes

Expected changes are limited to:

```text
interaction_vla/config.py       crowded/recovery configuration fields
interaction_vla/env.py          deterministic normal/crowded layouts and consistent perturbation API
interaction_vla/expert.py       recovery-aware state guards
interaction_vla/data.py         recovery generation, metadata, manifests, and CLI
interaction_vla/train.py        combined base/recovery training paths
interaction_vla/evaluate.py     condition-aware paired reports and stage comparison
configs/crowded_ood_macos.yaml  frozen-checkpoint Stage A evaluation
configs/recovery_macos.yaml     Stage B augmentation/training/evaluation
tests/interaction_vla/          unit and end-to-end coverage
```

No existing notebook or LeRobot/VLA training entry point is modified.

## Testing

Tests must cover:

- deterministic crowded resets;
- target-anchor distance bounds and non-overlap;
- normal layout backward compatibility;
- expert success for two through five objects in both valid layout modes;
- deterministic perturbations and held-object consistency;
- failed-close recovery reopening and realigning;
- recovery generation from training seeds only;
- recovery `.npz` metadata round trip;
- Graph/Flat loading the exact same combined augmented frames;
- condition-aware paired cases and metrics;
- frozen-checkpoint Stage A smoke evaluation;
- CPU collection → recovery augmentation → training → crowded evaluation;
- MPS one-step training when MPS is available;
- full regression and static compilation.

## Failure Handling

Invalid crowded placements and failed expert recoveries are recorded with seed, object count, condition or perturbation kind, and error. Manifests are updated atomically. Non-finite tensors and losses remain fatal. Existing checkpoint/data formats remain readable through metadata defaults.

## Explicit Non-Goals

- training on crowded layouts;
- changing node or edge features;
- changing grasp radius or success thresholds to improve scores;
- adding obstacles or collision planning;
- learning perturbations;
- DAgger policy rollouts in this iteration;
- RGB graph extraction, a world model, or SmolVLA integration.

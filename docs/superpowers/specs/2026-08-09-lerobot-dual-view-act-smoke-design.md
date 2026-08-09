# LeRobot v3 Dual-View Dataset and ACT Smoke Integration

Status: implementation design approved; awaiting written-spec review

Date: 2026-08-09

## 1. Objective

Add an isolated, reproducible bridge from the existing Franka MuJoCo environment
to a Hugging Face LeRobotDataset v3 dataset with:

- agent RGB;
- wrist RGB;
- a continuous end-effector observation state;
- local relative Cartesian actions;
- episode-level natural-language tasks;
- privileged teacher-only geometry for future TC-TIG perception training;
- an ACT training, checkpoint, and closed-loop rollout smoke test.

The first milestone proves that data collection, dataset loading, policy training,
checkpoint loading, and MuJoCo rollout form a valid engineering loop. It does not
claim useful task performance from a five-episode smoke dataset.

## 2. Current project facts

The existing physical datasets contain graph tensors, 23-dimensional
proprioception, 7-dimensional actions, expert phases, and privileged interaction
diagnostics. They do not contain RGB because their recording configuration was
disabled.

The existing `MultiViewRecorder` can synchronously render agent, wrist, side, and
top RGB-D views. The existing collection loop records state and action before the
action is executed. The new bridge preserves that timing convention but records
only agent and wrist RGB as policy inputs.

Existing experiment directories and checkpoint provenance remain untouched.

## 3. Scope

The subproject includes:

1. an isolated LeRobot dependency environment;
2. a standalone scripted-expert collector using the existing Franka environment;
3. LeRobotDataset v3 creation and validation;
4. teacher-only RGB-D, segmentation, calibration, object, and relation metadata;
5. 10D end-effector observation encoding;
6. 7D gripper-frame relative action encoding and rollout decoding;
7. an ACT single-batch check, 500-step smoke train, local checkpoint reload, and
   one closed-loop MuJoCo rollout;
8. local Hugging Face-compatible dataset and checkpoint interfaces.

The subproject does not include:

- language-conditioned ACT architecture changes;
- a learned RGB-to-TC-TIG model;
- SmolVLA or pi0 training;
- teleoperation data;
- recovery demonstrations;
- multi-task language-grounding claims;
- automatic publication to the Hugging Face Hub.

## 4. Alternatives considered

### 4.1 Replay existing NPZ episodes and render missing images

This preserves action trajectories but requires deterministic replay audits and a
separate reconstruction path for recovery starts. It also inherits older dataset
semantics. It is rejected for the first milestone.

### 4.2 Standalone direct LeRobot writer

This is the selected approach. A new collector reuses `FrankaContactEnv`,
`PhysicsScriptedExpert`, action safety, and rendering components while writing a
new dataset. It avoids modifying or overwriting the established physics data
pipeline.

### 4.3 Full LeRobot Robot or EnvHub plugin

This is useful for later distribution and unified LeRobot evaluation, but it
expands the first milestone into a packaging and plugin project. It is deferred
until the local dataset and ACT loop are proven.

## 5. Dependency isolation

The existing `.venv` must not be upgraded. The new integration uses
`.venv-lerobot` with:

- Python 3.12;
- `lerobot==0.6.1`;
- PyTorch `>=2.10,<2.11` for macOS arm64;
- the LeRobot `dataset` and `training` extras;
- the platform-compatible torchvision and video-decoding dependencies;
- the existing system FFmpeg 7.1.1 installation.

The implementation records the fully resolved package versions in a lock file or
frozen requirements artifact. It never installs CUDA wheels or copies dependency
pins from the older community MuJoCo tutorial.

Startup diagnostics record Python, LeRobot, PyTorch, torchvision, MuJoCo, FFmpeg,
platform, and accelerator availability. MPS is used only when
`torch.backends.mps.is_available()` is true in the user's process. Otherwise ACT
smoke training falls back to CPU without treating the fallback as an error.

## 6. Output isolation

The two dataset roots are:

```text
outputs/lerobot/franka_lerobot_act_smoke
outputs/lerobot/franka_lerobot_act_pilot
```

The smoke dataset contains five accepted base episodes. The pilot dataset contains
50 accepted base episodes and is created only after the smoke acceptance checks
pass.

Neither collector accepts an existing non-empty output root. A user must select a
new dataset root to rerun collection. No command in this subproject deletes or
rewrites an existing dataset.

## 7. Dataset layout

The root remains loadable by the standard LeRobotDataset v3 reader:

```text
<dataset-root>/
  data/
  videos/
  meta/
    info.json
    stats.json
    tasks.jsonl
    tc_tig_calibration.json
    tc_tig_teacher_schema.json
    provenance.json
    teacher_manifest.json
  teacher/
    episode_000000.npz
    ...
```

The additional `teacher/` files and custom metadata are ignored by an ordinary
LeRobot loader. They are consumed only by the future TC-TIG perception dataset
wrapper.

Dataset completion is represented by `meta/provenance.json` containing
`"complete": true`. A newly created or interrupted dataset contains an
`INCOMPLETE` marker, and the validator refuses to train from it.

## 8. Standard policy features

The model-visible feature contract is:

| Key | Storage dtype and shape | Loaded tensor shape | Meaning |
| --- | --- | --- | --- |
| `observation.images.agent` | video, `uint8[256,256,3]` | `float32[3,256,256]` after policy preprocessing | fixed agent RGB |
| `observation.images.wrist` | video, `uint8[256,256,3]` | `float32[3,256,256]` after policy preprocessing | wrist RGB |
| `observation.state` | `float32[10]` | `float32[10]` | EE position, rotation-6D, gripper aperture |
| `action` | `float32[7]` | `float32[7]` | local relative pose action and gripper command |
| `task` | LeRobot task metadata | string/tokenized by compatible policies | episode instruction |

The dataset FPS is 20. Every frame has a timestamp equal to
`policy_step / 20.0` relative to episode start.

The first dataset uses the canonical task:

```text
Pick up the green target object and place it inside the receptacle.
```

The constant instruction verifies language storage and policy compatibility. It
does not provide evidence that a policy uses language. Language variation and
target grounding require a later multi-task dataset.

## 9. End-effector observation state

`observation.state` is:

```text
[position_base(3), rotation_6d_base(6), gripper_aperture(1)]
```

Rotation uses the first two columns of the end-effector rotation matrix in
column-major order:

```text
[R00, R10, R20, R01, R11, R21]
```

Decoding applies Gram-Schmidt orthogonalization and a right-handed cross product.
Tests cover round-trip accuracy, orthogonality, and finite values.

The gripper aperture is the mean physical finger opening normalized to `[0, 1]`
using the MuJoCo joint range. It is continuous, not the current binary open flag.

The state is called a 6-DoF end-effector state because it represents an SE(3)
pose; its neural vector has nine pose values plus one aperture value. Euler angles
are not used.

## 10. Action representation

The stored action is:

```text
[delta_position_gripper(3), delta_rotation_gripper(3), gripper_command(1)]
```

The current expert and controller already express the rotation vector in the
gripper/body frame. The bridge converts only translation:

```text
delta_position_gripper = R_gripper.T @ delta_position_world
delta_position_world = R_gripper @ delta_position_gripper
```

The six pose components retain the existing normalized action scale in `[-1, 1]`.
The gripper command remains binary in `{0, 1}`. The normal LeRobot policy
normalizer operates after this representation conversion.

Training and rollout use the same versioned processor pair. A round-trip test must
prove that converting an expert action to dataset space and back produces the
same controller command within `1e-6` absolute tolerance.

## 11. Frame synchronization

Every policy step executes exactly this sequence:

1. read the current MuJoCo snapshot;
2. synchronously render agent RGB and wrist RGB;
3. render teacher-only metric depth and instance segmentation for both views;
4. encode the current 10D end-effector state;
5. compute teacher entities and relations from the same snapshot;
6. compute the expert action;
7. transform the action into gripper-local dataset coordinates;
8. append one LeRobot frame and one teacher frame with the same episode-local
   frame index, timestamp, and state hash;
9. execute the original expert action in the existing controller;
10. advance 25 MuJoCo substeps.

No post-action image may be paired with a pre-action state or action. The final
transition does not create an extra observation-only row.

## 12. Collection configuration

The smoke and pilot configurations share:

- backend `franka_contact`;
- normal layouts;
- object counts 2 and 3, sampled by the existing deterministic schedule;
- image size 256 by 256;
- policy rate 20 Hz;
- existing physical timestep and 25 substeps;
- maximum 180 policy steps;
- the existing expert gate and provenance validation;
- only successful base expert episodes;
- no recovery generation;
- no domain randomization in the first smoke.

The collector counts only accepted episodes toward 5 or 50. Every rejected
attempt records seed, object count, frame count, termination reason, and relevant
physics diagnostics.

## 13. Teacher-only data

All teacher-only fields are stored in the episode NPZ sidecar, not in LeRobot's
Parquet rows. This keeps the standard dataset sample equal to the policy feature
contract in Section 8 and prevents a privileged field from reaching ACT through a
permissive collator. Each sidecar contains the compact per-frame fields:

```text
annotation.tc_tig.entity_pose
annotation.tc_tig.entity_size
annotation.tc_tig.entity_role
annotation.tc_tig.visibility
annotation.tc_tig.relation_values
annotation.tc_tig.relation_goal
annotation.tc_tig.entity_mask
annotation.tc_tig.relation_mask
```

The keys retain the `annotation.tc_tig.*` namespace inside the NPZ. A versioned
`meta/tc_tig_teacher_schema.json` defines six canonical entity slots (gripper,
target, receptacle, support, and two distractors), eight canonical relation slots,
the feature widths, numeric vocabularies, padding values, coordinate frames, and
dtype of every array. The same sidecar also stores the large or audit-only arrays:

- agent and wrist metric depth as `float32`;
- agent and wrist instance segmentation as `uint16` IDs;
- episode-local frame index;
- timestamp;
- MuJoCo state hash;
- raw camera matrices required to audit derived labels.

Entity and relation semantics follow the approved TC-TIG specification. Relation
goal labels are postprocessed after the full successful episode is available,
because label generation may inspect the next action chunk. Future access is used
only for labels, never for model inputs.

Teacher generation may use MuJoCo segmentation, depth, geometry, and object poses.
It must not place contact force, scripted-expert phase, stable-grasp state, success,
or termination reason in either the teacher schema or the policy feature contract.
A forbidden-feature audit inspects the standard schema, teacher schema, and final
batch passed to ACT.

## 14. Camera calibration

`meta/tc_tig_calibration.json` records:

- MuJoCo camera names;
- image width and height;
- vertical field of view and derived intrinsic matrix;
- fixed agent-camera extrinsic in the robot base frame;
- wrist-camera extrinsic in the gripper frame;
- coordinate, quaternion, rotation-6D, and pixel-axis conventions;
- near and far depth planes;
- hashes of the scene XML and relevant camera configuration.

At runtime, the wrist camera base-frame pose is reconstructed from the 10D
end-effector state and the fixed wrist-to-gripper calibration. This is the geometry
interface later used by the RGB-token-to-object-slot model.

## 15. Component boundaries

The implementation is organized around independently testable units:

1. `EndEffectorStateCodec`: MuJoCo pose and fingers to/from the 10D state.
2. `LocalCartesianActionCodec`: controller action to/from the stored 7D action.
3. `DualViewCapture`: synchronized policy RGB and teacher RGB-D/segmentation.
4. `TCTIGTeacherExtractor`: compact entity and relation teacher records.
5. `LeRobotEpisodeWriter`: standard v3 frames, tasks, finalization, and metadata.
6. `TeacherSidecarWriter`: compact TC-TIG labels, raw lossless audit arrays, and
   hashes.
7. `FrankaLeRobotCollector`: episode orchestration only.
8. `LeRobotDatasetValidator`: schema, alignment, hashes, provenance, and replay.
9. `FrankaACTRolloutAdapter`: policy preprocessing, action decoding, controller,
   and rollout diagnostics.

No unit reaches into another unit's internals. Model-visible feature selection is
defined once and shared by training, validation, and rollout.

## 16. Transaction and failure behavior

Collection creates the requested, previously nonexistent dataset root and
immediately adds an `INCOMPLETE` marker. This root is the durable recovery and
audit location; it is not renamed or silently discarded after a failure. An
episode becomes visible to the dataset only after both the LeRobot episode and its
teacher sidecar have been flushed and their frame counts and hashes agree.

On normal completion, the writer calls `finalize()`, validates every episode,
writes final provenance with `"complete": true`, and removes the `INCOMPLETE`
marker. If collection or finalization fails, the marker remains and training is
refused. Existing output roots are never deleted, cleared, or overwritten.

Rejected expert attempts do not allocate a permanent episode index. The rejection
log is append-only within the new dataset root.

## 17. Provenance

The dataset provenance binds:

- repository Git commit;
- LeRobotDataset format version;
- Python and package versions;
- experiment configuration hash;
- scene XML and asset hash;
- expert and controller source hashes;
- expert gate hash;
- camera calibration hash;
- state codec version;
- action codec version;
- teacher schema version;
- task text and task ID mapping;
- accepted episode seeds and object counts;
- every teacher sidecar path, frame count, and SHA-256.

ACT checkpoints additionally bind the full LeRobot dataset fingerprint and policy
input/output feature contract.

## 18. ACT smoke protocol

### 18.1 One-batch integration check

The checker loads a batch through the standard LeRobotDataset and policy processor,
then performs:

1. ACT forward pass;
2. finite loss assertion;
3. backward pass;
4. one optimizer step;
5. local checkpoint save;
6. checkpoint reload through the Hugging Face-compatible local pretrained API;
7. evaluation-mode output equivalence on a fixed batch.

### 18.2 Five-episode smoke train

The smoke policy uses:

- both RGB keys;
- the 10D state;
- the 7D action;
- `chunk_size=8`;
- `n_action_steps=8`;
- batch size 2, falling back to 1 only on a caught out-of-memory error;
- `num_workers=0` on macOS;
- MPS when available, otherwise CPU;
- 500 optimizer steps;
- local logging;
- W&B disabled;
- no Hub upload.

An out-of-memory fallback is recorded in the training summary and retries exactly
once with batch size 1 after recreating the policy and optimizer. It does not
silently continue from a partially updated state.

### 18.3 Closed-loop rollout

The rollout adapter loads the local checkpoint, resets a deterministic normal
two-object case, and executes at most 180 policy steps using the existing action
safety and controller. It records raw action chunks, selected actions, local/world
conversion, action bounds, IK projection, and termination reason.

The smoke rollout must contain finite, correctly shaped actions and terminate
without schema, device, image, checkpoint, or controller errors. Task success is
reported but is not an acceptance requirement.

## 19. Pilot gate

The 50-episode pilot is blocked until all smoke acceptance checks pass. Pilot
training uses 5 epochs first, followed by evaluation. It may extend to 10 epochs
only when validation loss is still decreasing and the extension is recorded. It
does not use an arbitrary large step count.

Language-conditioned ACT, SmolVLA, pi0, and the learned TC-TIG visual head remain
blocked until the dataset/checkpoint/rollout loop is stable.

## 20. Verification requirements

Unit tests cover:

- rotation-6D encode/decode and orthogonality;
- gripper aperture normalization;
- local/world action round-trip;
- action bounds and gripper semantics;
- camera intrinsic and extrinsic construction;
- selected-view capture and pixel shape;
- metric depth and segmentation types;
- teacher feature masks and forbidden fields;
- deterministic task metadata and episode seed schedules.

Integration tests cover:

- a two-frame dataset finalized and loaded by standard LeRobotDataset;
- RGB decode shape and dtype;
- state/action shapes;
- exactly aligned timestamps, frame indices, and teacher state hashes;
- rejection of `INCOMPLETE` data;
- rejection of a missing or mismatched teacher sidecar;
- deterministic action replay from the recorded initial state;
- one ACT forward/backward and optimizer step;
- local pretrained checkpoint save/reload;
- one finite-action MuJoCo rollout;
- the existing project test suite.

## 21. Acceptance criteria

The subproject is complete only when:

1. all five accepted smoke episodes load with the standard LeRobotDataset API;
2. each sample exposes two `[3,256,256]` images, one `[10]` state, one `[7]`
   action, a timestamp, and readable task metadata;
3. every standard frame has exactly one teacher frame with matching index,
   timestamp, and state hash;
4. action conversion round-trips within `1e-6` absolute tolerance;
5. deterministic replay matches every recorded 10D observation state with maximum
   absolute error at most `1e-5` under the recorded software and scene hashes;
6. ACT completes a finite forward/backward update;
7. the local checkpoint reloads and produces the same fixed-batch evaluation
   output with maximum absolute error at most `1e-5` on the same device;
8. the learned-policy adapter completes a finite-action MuJoCo rollout;
9. the old data, training, evaluation, and dashboard tests remain green;
10. no model batch contains teacher-only or forbidden privileged fields.

## 22. Deferred VLA and visual-graph path

After this milestone, model-specific token providers expose a shared interface:

```text
VisualTokenBatch[batch, view, token, channel]
```

ACT supplies ResNet feature-map tokens, SmolVLA supplies SigLIP/SmolVLM visual
tokens, and pi0 supplies PaliGemma visual tokens. A task-conditioned cross-view
object-slot decoder combines these tokens with language, the 10D end-effector
state, and camera calibration to estimate objects and uncertainty. An analytic
relation layer then constructs TC-TIG.

The first visual-graph study freezes the vision backbone and compares the same
backbone with and without the graph bottleneck. SmolVLA local inference and PEFT
smoke follow ACT. Pi0 retains the same dataset and checkpoint adapter but trains on
remote NVIDIA hardware with sufficient memory.

## 23. Source compatibility policy

The community `lerobot-mujoco-tutorial` is a behavioral reference for MuJoCo
collection, two-camera data, ACT, pi0, and SmolVLA rollout. Its Python 3.10, CUDA
PyTorch 2.6, MuJoCo 3.1.6, and pinned historical LeRobot dependency are not copied.
The integration targets the versioned official LeRobot v3 interfaces defined in
this specification.

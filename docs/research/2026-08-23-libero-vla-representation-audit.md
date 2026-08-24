# LIBERO–VLA Interaction Representation Audit

Date: 2026-08-23  
Target: ICRA 2027  
Mode: scientific and implementation audit before code changes

## Scientific decision

The primary question is no longer whether a graph-conditioned ACT policy beats a flat ACT policy. The project will study how a fixed, physically grounded interaction vocabulary changes across VLA training stages and computation-path taps, and whether a factor is merely accessible, functionally used, or useful in closed loop.

The primary analysis tensor is:

`training stage × semantic tap × interaction factor`.

The measurement chain is:

`accessible → action-sensitive/functionally used → closed-loop useful`.

No implication between adjacent terms is assumed. The interaction graph is an annotation and measurement vocabulary, not a required policy architecture. Online RL and recovery labels are out of scope until the offline and intervention gates pass.

## 1. Reusable repository components

- `interaction_vla/representation_study/backends/`: LeRobot policy loading, deterministic action inference, and ACT/SmolVLA/π0 adapters. The generic policy/checkpoint binding contract is reusable.
- `interaction_vla/representation_study/taps/`: hook registration, intervention hooks, and latent serialization. Hook lifecycle and artifact binding are reusable; SmolVLA semantic tap definitions and masked pooling are not.
- `interaction_vla/representation_study/state_bank/io.py`: atomic JSON/JSONL writes and content hashing.
- `interaction_vla/representation_study/state_bank/validation.py`: episode-group leakage concepts and immutable manifest patterns.
- `interaction_vla/representation_study/extraction.py`: resumable latent caching pattern keyed by immutable state and checkpoint identities.
- `interaction_vla/representation_study/probes/`: frozen linear/shallow-MLP training infrastructure and checkpointed reports. Metrics and target masking must be extended for LIBERO.
- `interaction_vla/representation_study/interventions.py` and `evaluation.py`: intervention hook and paired-evaluation scaffolding. Existing global zero/mean/random ablations are controls, not the new factor-aligned primary intervention.
- `interaction_vla/representation_study/sft.py`: deterministic LeRobotDataset sampling and resumable checkpoint pattern. A new nested episode-subset planner is required.
- Existing Graph-v2 annotation code provides useful naming and tests, but its Franka scene assumptions and 89-dimensional policy-token schema must not define the LIBERO ontology.

The upstream LeRobot 0.6.1 installation contains its LIBERO environment adapter. The local macOS environment does not contain `hf-libero`; deterministic simulator replay is therefore a Linux-only integration path with dependency-free unit tests on macOS.

The formal profile uses the official `lerobot/libero` LeRobotDataset. Every training stage consumes the same immutable dataset revision and shared State Bank, so camera encoding is held fixed across the longitudinal comparison. The earlier `HuggingFaceVLA/libero` choice was retired on 2026-08-23 after server validation showed two operational threats to the protocol: roughly 35GB of image-parquet materialization exceeded the available cache volume, and its published episode file pointers were unsuitable for reliable episode-subset SFT. This source change does not reinterpret any earlier ACT/Graph result; the LIBERO study remains `not_started` until its replay and alignment gates pass.

## 2. Experiment-state classification

| Experiment | Status | Scientific role | Preserved artifact/observation |
|---|---|---|---|
| Graph-v2 ACT multi-seed control study | `formal_evidence` | Controlled mechanism evidence, not the modern-VLA contribution | 3 policy seeds, 60 rollouts/condition; flat 30.0%, teacher 35.0%, predicted-random 40.0%, predicted-Reflect 41.7% success |
| Graph-v2 diagnostics and intervention controls | `pilot_complete` | Motivates correctness/decodability/action sensitivity distinctions | Offline action sensitivity and descriptive failure traces; not closed-loop causal utility |
| ReflectVLM semantic graph pretraining | `pilot_complete` | Secondary transfer observation | Reflect is closer to teacher on graph targets, but does not reliably improve ACT success over random initialization |
| ACT stagewise representation study | `pilot_complete` | Controlled mechanism pilot | Artifact-complete, but 20 rollout/stage and recovery/backend confounds prevent a formal RL representation claim |
| Recovery RL v2 distribution calibration | `failed_gate` | Negative calibration result | Frozen-SFT recovery success remains about 0.84–0.85 for every severity; target band was 0.30–0.50 |
| Formal recovery PPO/SAC protocol | `implementation_only` | Frozen extension infrastructure | Code and validation exist; no positive scientific result is claimed |
| Existing local-Franka SmolVLA adapter | `implementation_only` | Engineering precursor | It binds SmolVLA to the custom two-camera Franka dataset, not LIBERO |
| π0 adapter | `implementation_only` | Future targeted replication | No experiment claimed |
| LIBERO replay/annotation/State Bank | `not_started` | New blocking evidence path | No current repository artifact supplies a shared privileged LIBERO State Bank |
| SmolVLA pretrained/SFT-25/50/100 LIBERO study | `not_started` | Main modern-VLA study | Missing State Bank, nested subsets, checkpoints, semantic latent cache, and probe reports |

`complete: true` in an old report denotes completeness under that report's own artifact contract. It is not reinterpreted as completion of the new LIBERO scientific study.

## 3. Missing components

1. A raw-LIBERO-to-LeRobot episode alignment manifest.
2. Deterministic replay from the original LIBERO HDF5 `states`, `actions`, and `model_file`.
3. A versioned task-semantics registry that maps BDDL predicates and simulator bodies/geoms to interaction roles.
4. Privileged geometry, contact, stable-grasp, phase, and next-relation annotation with applicability masks.
5. A LIBERO State Bank schema independent of the old Franka nominal/recovery strata.
6. Task-group and episode-group split manifests over the same State Bank.
7. Replay and label-distribution gates plus human-readable timelines.
8. Nested `D25 ⊂ D50 ⊂ D100` episode manifests and stage/checkpoint metadata.
9. SmolVLA-specific semantic taps with explicit masks, denoising-call selection, and pooling.
10. Cluster-aware probe metrics and confidence intervals.
11. Factor-aligned intervention and paired LIBERO evaluation interfaces.

## 4. Scientific risks

- **Task-semantics mismatch.** LIBERO Goal and Long contain articulation, activation, and multi-goal tasks; `receptacle`, `stable grasp`, `lift`, and `place` are not universally applicable. Every factor/subfield therefore has an applicability mask. Undefined values are never encoded as negative labels.
- **Intent ambiguity.** A conjunctive BDDL goal does not define an execution order. `NextRelation` for multi-goal tasks requires a versioned, manually reviewed task plan. The initial formal scope supports the relocation tasks in LIBERO Spatial and Object; Goal/Long tasks fail closed until their plans are reviewed.
- **Entity shortcut.** Target identity can be inferred from task ID or instruction. Entity results require task-ID-only and instruction-only baselines and cannot alone support a visual-grounding claim.
- **Probe shortcut.** Increased probe quality establishes accessibility, not policy use or utility.
- **Intervention OOD.** Whole-vector zeroing is only an OOD sanity control. Primary interventions must preserve scale and be factor aligned.
- **Pseudo-replication.** Frames are not independent replicates. Confidence intervals and comparisons use episode/task sampling units.
- **SFT fraction confound.** Fractions must be nested and task balanced. Each stage starts independently from the same immutable base snapshot and uses a fixed epoch budget, not fixed steps, for the primary data-scale comparison.
- **Replay drift.** Annotation is invalid if state/action playback does not match the stored trajectory within the registered tolerances.
- **Outcome availability.** The public nominal demonstration records do not expose a trustworthy success/failure field aligned to every pre-action observation. The audit reports whether the terminal goal relation is observed, but does not relabel its absence as failure. Failed/partial outcome supervision must come from a source that explicitly records it.

## 5. Data-leakage risks and controls

- Frame-random splitting is prohibited.
- Each source episode belongs to exactly one partition in the episode-group split.
- Each task belongs to exactly one partition in the main task-group split; tasks are stratified by LIBERO suite before assignment so suite coverage is not left to a global shuffle.
- SFT subset episodes and State Bank evaluation episodes are recorded and checked for overlap according to the declared study protocol.
- State IDs include source revision and episode/frame identity; duplicate observation references are rejected.
- Labels are computed once and shared across all checkpoints. No checkpoint-specific reannotation is permitted.
- Probe feature/target normalization is fit on the probe-training partition only and bound by report provenance. Policy preprocessing retains the immutable official LeRobotDataset statistics for every stage so normalization does not vary with SFT fraction; this shared metadata-level exposure is disclosed rather than described as episode-level training.
- Hyperparameter/model selection uses validation groups; the test split is report-only.
- Task-group support is audited per factor. A metric is suppressed when a held-out partition lacks a meaningful label support set.

## 6. Exact State Bank schema

Schema version: `libero_interaction_state_bank_v1`.

Stable identifier:

`libero:{suite}:{task_id}:{source_episode_id}:{frame_id}:{source_revision_prefix}`.

Each record contains:

- identity: `state_id`, suite, task ID/name, source episode ID, LeRobot episode/index, frame/timestep, source revision, optional simulator seed/init-state index;
- observation reference: global RGB key, optional wrist RGB key, language instruction, robot state, action, timestamp, dataset row;
- replay reference: raw HDF5 file/demo, simulator-state row, action row, model-XML hash, initial-state hash;
- entities: target, goal entity, support/source entity, and distractor semantic IDs;
- geometry: `T_gripper_target` and `T_target_goal`, each represented by relative translation plus continuous rotation-6D, plus gripper-target and target-goal distances;
- contact: gripper-target contact and target-goal/support contact;
- stable grasp;
- canonical phase;
- structured next relation;
- per-factor and per-subfield applicability masks;
- provenance: config, ontology, raw source, LeRobot source, and annotator hashes.

Relative transforms are:

`T_gripper_target = inverse(T_world_gripper) @ T_world_target`

`T_target_goal = inverse(T_world_target) @ T_world_goal`.

World-coordinate positions may be retained only as privileged replay diagnostics, never as the primary geometry probe target.

## 7. Exact Phase definition

Canonical vocabulary:

`approach`, `align_precontact`, `contact`, `secure`, `actuate`, `lift`, `transport`, `place`, `release_retreat`.

Labels are event based, with configurable hysteresis; elapsed timestep is never a rule input.

For relocation tasks, rules are evaluated in this order:

1. `release_retreat`: goal predicate is satisfied, stable grasp is false, and gripper-target contact is absent or clearance is increasing.
2. `place`: target is at the goal relation and is still grasped/contacted.
3. `transport`: stable grasp is true, target is off its source support, and target is not at the goal-near threshold.
4. `lift`: stable grasp is true, source support has been lost, and displacement from the episode's initial target pose is below the configured lift/transport threshold.
5. `secure`: stable grasp is true while source support remains.
6. `contact`: gripper-target contact is true but stable grasp is false.
7. `align_precontact`: gripper-target surface distance is inside the approach threshold but no contact exists.
8. `approach`: remaining applicable pre-contact states.

For articulation/activation tasks, the applicable reduced sequence is `approach → align_precontact → contact → actuate → release_retreat`. This path is enabled only by a reviewed task-semantics entry. Multi-goal tasks also require a reviewed ordered subgoal plan.

Default thresholds are registered in configuration, not hard-coded: a 5-frame stable window at the aligned trajectory cadence, 15 mm relative-translation drift, 15-degree relative-rotation drift, 5 mm object co-motion, 10 mm initial-pose displacement for the lift/transport boundary, 50 mm pre-contact surface distance, and 10 mm hysteresis. Replay control frequency is separately configured and the LeRobot observation FPS is recorded from dataset metadata. These are annotation defaults subject to the distribution/timeline gate, not claims of universal physical constants.

## 8. Exact StableGrasp definition

`StableGrasp` is applicable only to a graspable movable target. It is true when all conditions hold over the configured temporal window:

1. contact exists between the target and at least the configured number of distinct gripper finger groups (default two);
2. the gripper is closing or aperture is below the configured grasp-aperture threshold;
3. target pose relative to the gripper remains within translation and rotation drift thresholds;
4. either the target co-moves with the end effector by the minimum displacement or the target is lifted/off its source support.

Missing history produces `not_evaluable`, not `false`. Contact alone is insufficient. The report includes event durations, isolated one-frame positives, transition counts, and sensitivity to the registered thresholds.

## 9. Exact NextRelation definition

`NextRelation` is a structured intended transition:

```text
active_goal_index
subject_role
predicate
object_role
operator
```

`operator` is one of `establish`, `maintain`, `increase`, `decrease`, or `clear`. Predicates are versioned and include `near`, `contact`, `stable_grasp`, `off_support`, `on`, `inside`, `open`, `closed`, `powered_on`, `powered_off`, and `clearance`.

For a single-goal relocation task:

1. far/no contact → establish `near(gripper,target)`;
2. near/no contact → establish `contact(gripper,target)`;
3. contact/not stable → establish `stable_grasp(gripper,target)`;
4. stable/on source → establish `off_support(target,source)`;
5. stable/off source/not near goal → establish `near(target,goal)`;
6. near goal/goal unsatisfied → establish the BDDL goal predicate, e.g. `on(target,goal)` or `inside(target,goal)`;
7. goal satisfied/still grasped → clear `contact(gripper,target)`;
8. released → increase `clearance(gripper,target)`.

This is not the next frame's phase. For an unordered multi-goal conjunction, an active goal cannot be inferred uniquely from BDDL; annotation fails closed unless a reviewed task plan supplies the order.

## 10. Proposed SmolVLA taps

The primary pooling strategy is preregistered as `valid_token_mean`. Padding and invalid visual tokens are excluded. Alternative strategies may be implemented as named sensitivity analyses but cannot replace the primary after results are observed.

| Semantic tap | Exact SmolVLA tensor/module | Call selection | Primary reduction |
|---|---|---|---|
| `vision_output` | per-view output of `model.vlm_with_expert.embed_image`, after vision encoder and connector | one call per configured view | valid visual-token mean per view, then fixed ordered concatenation of view means |
| `multimodal_fusion` | normalized prefix hidden state returned by the first prefix/prefill call to `model.vlm_with_expert.forward` | first prefix call only | mean over valid image, language, and state prefix tokens using `prefix_pad_masks` |
| `action_expert_input` | output of `model.action_time_mlp_out` | final registered denoising evaluation | mean over the action-chunk axis |
| `pre_action` | input to `model.action_out_proj` (`suffix_out`) | final registered denoising evaluation | mean over the action-chunk axis |

The current generic `model.vlm_with_expert` last-call hook conflates prefix fusion with repeated denoising suffix calls and is not acceptable for the new study. Every latent manifest records module path, tensor shape before/after pooling, mask shape, pooling rule, denoising-call rule, checkpoint hash, and deterministic inference-noise seed.

## 11. Directory and configuration structure

```text
configs/representation_study/
  libero_smolvla_smoke_linux_cuda.yaml
  libero_smolvla_linux_cuda.yaml

interaction_vla/representation_study/libero/
  config.py
  schema.py
  interfaces.py
  task_semantics.py
  replay.py
  contacts.py
  annotation.py
  splits.py
  state_bank.py
  audit.py
  visualize.py
  stages.py
  training.py
  taps.py
  latents.py
  probes.py
  probe_runner.py
  interventions.py
  evaluation.py
  sources.py
  runtime.py
  collector.py

outputs/representation_study/libero_smolvla/
  source_alignment/
  state_bank/
  splits/
  audit/
  timelines/
  stages/
  latents/
  probes/
  interventions/
  evaluation/
```

The old `outputs/representation_study/icra/` and `icra_rl_v2/` trees remain immutable.

## 12. Gates and dependency graph

```text
scientific audit
  → source availability/alignment gate
  → deterministic replay gate
  → annotation semantics/distribution/timeline gate
  → shared State Bank + task/episode leakage gates
  → nested SFT subset manifests
  → checkpoint identity/status gate
  → semantic tap contract + latent cache gate
  → linear-probe/baseline/reliability gate
  → MLP capacity check + Stage×Tap×Factor report
  → factor-aligned intervention specificity gate
  → paired closed-loop utility evaluation
  ─X→ RL (explicitly stopped in this pass)
```

Required checks before moving forward include raw/LeRobot action and frame alignment; replay state error; finite labels; temporal contact/grasp sanity; source metadata retention; task/episode leakage; shared state-ID equality across checkpoints; exact label support/distributions; manual timelines; latent shape/mask/call provenance; exact Stage×Tap×Factor coverage; episode/task bootstrap units; intervention target disruption and non-target preservation; and paired initial-state identity.

## 13. Planned files

Create the new `interaction_vla/representation_study/libero/` package and matching tests under `tests/interaction_vla/representation_study/libero/`; add two isolated configs; extend the existing CLI with a lazy-loaded `libero` command family; add the Linux LIBERO/SmolVLA dependency extras; update `ccfa.yaml`, README commands, and research/design/implementation documents. Existing ACT, Graph-v2, ReflectVLM, and recovery-RL code and output paths are not rewritten.

## Blocking assumptions

There is no scientifically valid way to recover privileged contacts, object poses, or exact simulator replay from the public LeRobot video/state/action columns alone. A formal State Bank therefore requires the matching original LIBERO HDF5 demonstrations containing simulator states, actions, and model XML. The implementation must fail clearly if that source is absent. It must not approximate privileged labels from RGB and call them ground truth.

The initial formal annotation scope is LIBERO Spatial and Object. Goal and Long support is an explicit extension gate because their articulation and multi-goal task plans require reviewed semantics. This limitation is reported as coverage, not hidden by default labels.

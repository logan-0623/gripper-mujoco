# MuJoCo Graph Fine-tuning Design

## Scope

This phase tests whether the completed ReflectVLM semantic-graph checkpoint improves
MuJoCo graph learning. It adds a graph estimator and a paired initialization
experiment over the existing standard LeRobot dataset plus its teacher sidecars. It
does not train ACT, generate continuous actions, or use a control rollout as evidence.

The experiment has two gates:

1. a five-episode smoke dataset verifies the software path;
2. a fifty-episode pilot measures sample efficiency for random and ReflectVLM
   initialization on identical episode splits.

## Decision

Three approaches were considered:

1. Transfer the compatible ReflectVLM trunk into a MuJoCo TC-TIG estimator and compare
   it against the same architecture with random initialization. Selected because it
   isolates initialization while preserving the target graph used by the later policy.
2. Reuse every ReflectVLM output head and remap its labels. Rejected because
   ReflectVLM object-index slots and MuJoCo task-role slots do not have equivalent
   semantics.
3. Connect the checkpoint directly to ACT. Deferred because simultaneously changing
   perception, graph representation, and control would make the source of any result
   uninterpretable.

## Input and Leakage Boundary

The estimator receives only fields available to a future policy:

- current agent RGB;
- current wrist RGB;
- the 10D end-effector state already defined by the LeRobot bridge;
- the language task string stored by `LeRobotDataset`.

The action, future frames, teacher graph, depth, instance segmentation, simulator
poses, contact forces, expert phase, success flag, and termination reason are never
model inputs. Teacher sidecars are label-only data and remain outside the standard
LeRobot policy sample.

All splitting is at episode level. A deterministic hash of the split seed and teacher
episode seed assigns complete episodes to train, validation, or test. Training
fractions select complete episodes only from the fixed training partition. Vocabulary,
end-effector normalization, relation normalization, and residual normalization are fit
only on the selected training episodes.

## MuJoCo Semantic Graph Target

The target keeps the existing `tc_tig_teacher_v1` task-role topology:

- six entity slots: gripper, target, receptacle, support, and two distractors;
- eight typed relation slots covering gripper-target, target-receptacle,
  target-support, distractor risk, and gripper-receptacle clearance;
- a next-relation goal containing relation, operator, predicate, and signed future
  residual.

The estimator predicts:

- `entity_mask[6]` and `entity_visibility[6,2]`;
- `relation_mask[8]`;
- `relation_semantics[8,10]`, containing only signed margins, probabilities, risks,
  and task errors from relation feature channels 12 through 21;
- `goal_relation`, `goal_operator`, `goal_predicate`, and `goal_residual`.

Relative position, relative rotation, linear velocity, angular velocity, simulator
contact state, raw depth, segmentation, and teacher measurement-quality channels are
deliberately excluded. The output therefore answers which
task roles are present, how the gripper, target, receptacle, and distractors relate,
and which relation should change next without copying full MuJoCo state.

## Model and Transfer Contract

`MuJoCoGraphEstimator` uses one weight-shared RGB encoder for agent and wrist images.
The two image embeddings are averaged. A small 10D state encoder produces a residual
in the same image-embedding space. The language encoder uses the ReflectVLM token
embedding layout, and a fusion MLP produces the graph embedding used by MuJoCo-specific
heads.

Both initialization conditions start from the same deterministic base parameters. The
`reflectvlm_init` condition then copies only shape- and semantics-compatible values:

- the complete shared RGB encoder;
- token embeddings for vocabulary tokens present in the ReflectVLM checkpoint;
- the fusion MLP;
- operator and predicate classifier heads, whose vocabularies are already shared.

The state encoder and MuJoCo-specific entity, relation, relation-ID, and residual
heads remain identically random in both conditions. Loading emits a transfer
report containing every copied tensor, skipped tensor, and copied-token count. A
schema or dimensional mismatch fails instead of silently performing a partial transfer.

## Objective and Metrics

Training uses binary cross-entropy for entity/relation masks, mean squared error for
entity visibility, Smooth L1 for normalized active relation semantics and goal
residual, and cross-entropy for the three categorical goal fields. Inactive entity and
relation slots are excluded from regression losses.

Evaluation reports:

- entity-mask and relation-mask precision, recall, and F1;
- entity-visibility MAE;
- semantic-relation MAE in original units, both overall and per relation slot;
- relation, operator, and predicate accuracy;
- exact relation-plus-operator-plus-predicate goal accuracy;
- signed goal-residual MAE;
- total held-out loss.

The primary transfer comparison is the paired `reflectvlm_init - random_init` delta for
exact goal accuracy, relation-mask F1, and semantic-relation MAE. Lower MAE is better;
higher classification metrics are better.

## Experiment Matrix

The smoke configuration uses the existing five-episode
`local/franka_lerobot_act_smoke` dataset, a 3/1/1 episode split, seed 0, the full
training partition, and a small model/training budget. A smoke `passed=true` means
decoding, alignment, transfer, optimization, checkpoint reload, and evaluation work;
it is not a scientific result.

The pilot configuration uses `local/franka_lerobot_act_pilot`, fifty episodes, a
40/5/5 episode split, training fractions 0.10, 0.25, and 1.00, and model seeds 0, 1,
and 2. Every condition shares the same episode allocation, fraction selection, loader
order, optimizer, and number of epochs. Reports preserve per-seed values rather than
selecting the best seed. Pilot inspection fails clearly until the existing LeRobot
bridge has collected and validated all fifty episodes.

ReflectVLM pretraining counts as evidence of improved sample efficiency only if the
pretrained condition improves the prespecified primary metrics across seeds, with the
largest expected benefit at 0.10 or 0.25 data. A single smoke run, one favorable metric,
or a seed-selected result is insufficient. Because the current dataset contains one
language instruction, this phase validates a language-conditioned interface but does
not establish language generalization.

## Commands and Artifacts

The new module exposes:

```text
python -m interaction_vla.graph_finetune inspect --config <yaml>
python -m interaction_vla.graph_finetune compare --config <yaml>
python -m interaction_vla.graph_finetune evaluate --config <yaml> --checkpoint <pt>
```

`inspect` validates the standard dataset, teacher hashes, frame alignment, episode
split, label ranges, and ReflectVLM checkpoint compatibility without training.
`compare` executes the frozen experiment matrix and writes condition checkpoints plus
one paired comparison. `evaluate` reloads one compatible checkpoint on validation or
test episodes.

Artifacts are written below:

```text
outputs/graph_finetune/mujoco_smoke/
outputs/graph_finetune/mujoco_pilot/
  split_manifest.json
  random_init/fraction_<value>/seed_<seed>/
  reflectvlm_init/fraction_<value>/seed_<seed>/
  comparison.json
```

Each run directory contains `checkpoint.pt`, `training_summary.json`, and
`evaluation.json`. Checkpoints store dataset and teacher provenance, split seed,
training fraction, initialization condition, transfer report, normalization values,
vocabulary, config, and model state.

## Failure Handling

The pipeline rejects missing or stale teacher manifests, sidecar hash mismatches,
frame misalignment, frame-level split leakage, empty episode partitions, incompatible
ReflectVLM checkpoints, forbidden model-input keys, non-finite labels or losses,
unsupported relation schemas, and existing non-empty run directories. CLI failures
return a non-zero status and one compact JSON error.

## Verification

Network-free tests use synthetic LeRobot-like samples and sidecars. They cover episode
splitting, fraction selection, vocabulary and normalization isolation, sample/teacher
alignment, forbidden-input auditing, output shapes, masked objectives, exact transfer
provenance, paired initialization, checkpoint reload, metrics, CLI errors, and a tiny
end-to-end comparison. A real smoke run uses the existing local five-episode dataset
and the completed ReflectVLM checkpoint before the implementation is declared ready.

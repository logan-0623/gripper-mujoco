# ReflectVLM Graph Pretraining Design

## Scope

This phase adds a self-contained visual interaction-graph pretraining workflow for
`yunhaif/ReflectVLM-data-expert`. It does not convert symbolic actions into robot
actions, modify ACT, or consume MuJoCo privileged state. The output is a checkpoint
and evaluation report that can later initialize the MuJoCo graph estimator.

## Decision

Three approaches were considered:

1. Convert ReflectVLM directly to `LeRobotDataset`. Rejected because it has no wrist
   view, end-effector state, control rate, or continuous action.
2. Fine-tune a generative VLM to emit JSON graphs. Deferred because it adds tokenizer,
   decoder, and large-checkpoint dependencies before the graph contract is validated.
3. Train a lightweight discriminative multi-task graph estimator. Selected because it
   preserves the available supervision, runs on Apple Silicon, and produces measurable
   graph outputs without pretending the data is robot control data.

## Input and Leakage Boundary

The model input is only the current RGB image and the previous symbolic action
history. `action_description`, `oracle_action`, `agent_action`, object annotations,
`next_image`, and `final_goal_image` are supervision-only fields. Training and
evaluation reject rows whose required labels cannot be parsed instead of silently
inventing labels.

The source repository exposes one `train` split. The adapter creates deterministic
train, validation, and test partitions by hashing `(board_id, env_seed)`. All steps
and resets from one board/environment group stay in one partition.

## Canonical Semantic Graph Target

Each example produces a fixed, serializable target over at most six task objects. Rows
with fewer objects use padded slots plus an explicit `object_mask[6]`:

- `target_index`: object operated on by the next oracle action;
- `in_hand_index`: one of six object slots or `none`;
- `state_ids[6]`: `unknown`, `ready`, `blocked`, `bad`, or `done`;
- `upright_ids[6]`: `unknown`, `false`, or `true`;
- `dependency[6,6]`: directed task dependency adjacency;
- `phase_id`: `pick`, `place`, `reorient`, or `insert`;
- `goal_operator_id` and `goal_predicate_id`: values drawn from the existing teacher
  schema's relation-goal vocabularies.

The adapter orders physical objects by their source brick index and excludes
`brick_1` (the board) from the six object slots. The graph is task-conditioned: the
next oracle action supplies labels, never model inputs. This semantic target
intentionally omits poses, contact forces, simulator success, and full environment
state.

## Model

`ReflectGraphEstimator` contains:

- a small convolutional RGB encoder operating on resized images;
- a learned token embedding with masked mean pooling for action history;
- a fused graph embedding;
- independent heads for target, in-hand object, per-object state, upright state,
  dependency edges, interaction phase, goal operator, and goal predicate.

Training uses cross-entropy for categorical heads and masked binary cross-entropy for
dependency edges. The checkpoint stores model weights, vocabulary, config, schema
version, source repository, and split seed. No pretrained weights are downloaded.

## Commands and Artifacts

The module exposes:

```text
python -m interaction_vla.graph_pretrain inspect --config <yaml>
python -m interaction_vla.graph_pretrain train --config <yaml>
python -m interaction_vla.graph_pretrain evaluate --config <yaml> --checkpoint <pt>
```

`inspect` validates schema and split isolation without training. `train` writes
`checkpoint.pt`, `training_summary.json`, and `split_manifest.json`. `evaluate` writes
`evaluation.json` with loss, target accuracy, in-hand accuracy, state accuracy,
dependency F1, phase accuracy, and exact relation-goal accuracy.

The default Mac config uses the Hugging Face dataset, 128 px RGB, a small batch,
automatic CPU/MPS selection, bounded epochs, and an optional row limit for smoke runs.

## Failure Handling

Parsing uses `ast.literal_eval`, explicitly handles null and empty-set encodings, and
never uses `eval`. Invalid object identifiers, missing target objects, unknown action
verbs, duplicate object slots, split overlap, non-finite losses, incompatible
checkpoints, and missing optional dependencies raise actionable errors. CLI failures
return a non-zero exit status and a compact JSON error.

## Verification

Unit tests use in-memory rows and synthetic images, so they require no network. They
cover safe parsing, semantic labels, leakage-safe grouped splitting, batching, model
output shapes, finite optimizer updates, checkpoint reload, metrics, and CLI errors.
A synthetic end-to-end smoke trains and evaluates a tiny dataset. The existing full
test suite must remain green.

## Scientific Gate

This phase is successful when held-out groups can be evaluated reproducibly and the
checkpoint reloads exactly. Accuracy is reported rather than assumed. ReflectVLM
pretraining is considered useful for transfer only if a later MuJoCo comparison shows
that pretrained initialization improves graph-label metrics over random initialization
on identical seed splits. Continuous-control claims require separate ACT comparisons
with Flat, oracle Graph, and predicted Graph inputs.

# Graph-Conditioned ACT Continuous-Control Design

## Objective

Test whether the compact, vision-estimated interaction graph improves continuous
closed-loop manipulation beyond the existing dual-RGB and 10D end-effector ACT
baseline. The experiment compares the representation, not a larger policy: every
condition uses the same ACT architecture, initialization seed, episode split, batch
order, optimizer, and action horizon.

This stage evaluates control utility. It does not claim language generalization,
because the current pilot dataset contains one instruction.

## Chosen integration

ACT receives the graph as its native `observation.environment_state` token. The
10D `observation.state` remains pure proprioception. This is preferable to appending
Graph values to robot state because it preserves the semantic boundary and ACT gives
environment state a separate encoder token. It is preferable to a new graph
cross-attention architecture because changing ACT would confound representation
quality with architecture capacity.

All four conditions expose a 75D environment-state feature, so parameter counts are
identical. The graph estimator is frozen and its outputs are cached before ACT
training.

## Compact 75D control token

The token contains only task-relevant, coordinate-invariant interaction information:

| Slice | Size | Meaning |
|---|---:|---|
| entity presence | 6 | gripper, target, receptacle, support, two distractors |
| dual-view entity visibility | 12 | whether each task entity is visually supported |
| relation presence | 8 | which TC-TIG relation slots are active |
| gripper-target semantics | 10 | margins, probabilities, risks, and errors |
| target-receptacle semantics | 10 | containment and placement state |
| distractor risks | 8 | two risk channels for four distractor relations |
| next relation distribution | 8 | which relation should change next |
| relation operator distribution | 5 | establish, break, increase, preserve, decrease |
| predicate distribution | 7 | proximity through clearance |
| normalized residual | 1 | predicted magnitude/direction of the next change |

The token excludes world coordinates, object poses, contact forces, depth,
segmentation, actions, expert phase, success labels, and termination state.
Categorical predictions remain soft distributions rather than argmax IDs. Semantic
values and the residual use the normalization stored in the Graph checkpoint.

## Experimental conditions

1. `flat`: the 75D token is zero. RGB and proprioception are unchanged.
2. `predicted_random`: token from the seed-matched random-initialized MuJoCo Graph
   checkpoint.
3. `predicted_reflect`: token from the seed-matched ReflectVLM-initialized checkpoint.
4. `oracle_current`: current entity and relation fields come from the causal MuJoCo
   teacher; next-goal fields still come from the seed-matched Reflect estimator.

`oracle_current` is a privileged current-perception upper bound, not a deployable
policy. It must never consume `annotation.tc_tig.relation_goal`, because that label is
computed using a future trajectory window. During rollout it uses only the current
snapshot and current camera frame. This condition answers whether current graph
perception, rather than goal prediction, is the bottleneck.

## Split and pairing contract

The ACT experiment reuses
`outputs/graph_finetune/mujoco_pilot/split_manifest.json`: 40 training, 5 validation,
and 5 test episodes. It must not use the older ACT-specific permutation, because that
would place images used to fine-tune the Graph estimator into the ACT held-out set.

Each policy seed uses the full-data Graph checkpoint with the same seed. All four ACT
conditions for a seed begin with byte-identical shared ACT parameters. Training and
validation rows, action chunks, epoch count, batch order, learning rate, and early
extension decisions are paired. A mismatch aborts the experiment.

## Cache and provenance

Graph token caches are immutable `.npz` files indexed by the standard LeRobot global
row index. Each cache binds:

- dataset fingerprint;
- split-manifest hash;
- token schema and dimension;
- Graph checkpoint hash, schema, initialization, fraction, and seed;
- exact row indices and finite float32 token matrix;
- whether fields are predicted, zero, or causal-oracle.

Cache generation decodes images lazily in bounded batches. ACT checkpoints bind the
cache hash and condition in `bridge_checkpoint.json`. Reload refuses mismatched data,
Graph checkpoints, feature layouts, or source code.

## Training and evaluation

The engineering smoke performs one optimizer update and checkpoint reload for every
condition on the pilot dataset. The formal pilot trains 4 conditions x 3 seeds with
the existing ACT hyperparameters: batch size 2 and exactly five epochs. The baseline's
condition-specific validation-extension rule is disabled here because it could give
one representation more optimizer updates than another and break the paired comparison.

Closed-loop evaluation uses a fixed paired seed schedule crossing normal/crowded
layouts and 2/3 objects. Every condition receives the identical environment case.
Primary metrics are task success, wrong-object bilateral interaction, wrong-object
stable grasp, target drop, and timeout. Secondary diagnostics include episode length,
IK projection scale, action clipping, and gripper switching. Reports retain per-case
rows and paired deltas; the three policy seeds are the independent replication unit.

The primary scientific contrast is `predicted_reflect - flat`. Secondary contrasts
are `predicted_reflect - predicted_random` (pretraining transfer) and
`oracle_current - predicted_reflect` (current-graph perception gap). No conclusion is
drawn from the engineering smoke.

## CLI and outputs

The module exposes:

```text
python -m interaction_vla.graph_control inspect --config ...
python -m interaction_vla.graph_control cache --config ...
python -m interaction_vla.graph_control smoke --config ...
python -m interaction_vla.graph_control compare --config ...
python -m interaction_vla.graph_control evaluate --config ...
```

Smoke outputs live under `outputs/graph_control/act_smoke/`; formal checkpoints,
per-case rollout records, and aggregate reports live under
`outputs/graph_control/act_pilot/`. Commands refuse nonempty training destinations
instead of overwriting user results.

## Failure boundaries and tests

Unit tests cover token layout, forbidden-field rejection, future-goal exclusion,
cache/checkpoint hashes, split alignment, condition pairing, ACT environment-state
feature/reload, and paired evaluation aggregation. Integration tests use tiny local
LeRobot datasets and synthetic Graph checkpoints. A real one-update smoke on macOS is
required before presenting the formal pilot commands.

The experiment cannot establish language generalization, policy-family
generalization, or real-robot transfer. Those require multiple instructions and later
SmolVLA/pi0 experiments.

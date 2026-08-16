# Interaction-Graph Policy Design

## Objective

Build a Mac-friendly experiment that tests one representation hypothesis:

> Given the same current scene information and training budget, a policy that encodes gripper-object-support interactions as a graph learns more reliable object-aware and gripper-aware manipulation than a policy that receives the same information as one flat vector.

The first implementation predicts actions directly. It is not a world model and does not predict future states. A frozen SmolVLA adapter is a later extension after the lightweight comparison works locally.

## Scope

The first version implements a state-only, one-step Cartesian behavior-cloning pilot for tabletop pick-and-place. It uses privileged simulator state to isolate representation quality from visual perception and language grounding. A task instruction is converted to a target-object flag and supplied identically to both representations.

The pilot supports two to five movable objects, one receptacle, and the robot gripper. Training scenes contain two or three objects. Four- and five-object scenes are reserved for object-count generalization tests.

The existing tutorial notebooks remain intact. New research code lives under `interaction_vla/`, with command-line entry points for collecting data, training, and evaluating policies.

## Non-goals

- End-to-end SmolVLA or PI0 fine-tuning on the local Mac.
- Learning a graph from RGB images.
- Predicting future world states.
- Claiming paper-level evidence from the pilot dataset.
- Adding navigation, deformable objects, or multi-robot interaction.

## Representation Contract

Every observation is converted to a fixed-capacity `SceneGraph` with masks. The maximum is eight nodes: one gripper, up to five movable objects, one receptacle, and one support surface. Edges are the complete directed graph without self-edges, for at most 56 edges.

Each node stores only current-time information:

- node type;
- position and orientation;
- linear and angular velocity;
- approximate size;
- gripper-open value;
- movable flag;
- target-object flag.

Each directed edge stores:

- relative position;
- relative velocity;
- Euclidean distance;
- contact flag;
- gripper-holding-object flag;
- object-supported-by-receptacle/surface flag.

The `GraphEncoder` performs shared node encoding and masked message passing. The `FlatEncoder` flattens the exact same padded node features, edge features, and masks in canonical order and passes them through an MLP. Both encoders emit the same-dimensional scene embedding. Their trainable parameter counts must remain within 10 percent.

## Policy Architecture

The Mac pilot uses a compact shared action head:

```text
scene tensors -> FlatEncoder or GraphEncoder -> scene embedding
robot proprioception -------------------------^
                                                     -> next Cartesian/gripper action
```

The Graph and Flat variants share the proprioception encoder, temporal/action head, loss, optimizer settings, phase-balancing weights, training batches, and action normalization. Rare close/release frames receive inverse-frequency phase weights so gripper transitions are not dominated by movement frames. The only model-specific component is the scene encoder.

The first required mode is state-only. Frozen visual/language context and a later SmolVLA adapter remain future work; they can reuse the scene-graph schema without changing this first comparison.

## Environment and Data Generation

A deterministic tabletop environment exposes the same reset/step/snapshot contract for fast tests and MuJoCo-backed collection. A scripted expert executes the phases:

```text
approach -> align -> close -> lift -> transport -> release -> retreat
```

The observation at time `t` is captured before `action_t` is applied:

```text
snapshot_t -> graph_t -> expert/policy action_t -> store pair -> environment step
```

This ordering prevents future contact or grasp information from leaking into the input.

Each episode records the environment seed, object count, target object, robot state, scene tensors, expert action, expert phase, and terminal outcome. Invalid spawns, inverse-kinematics failures, timeouts, wrong-object grasps, and drops are retained as explicit result codes or rejected with a logged reason; they are never silently discarded.

Two configurations are provided:

- `pilot_macos`: 50 collection attempts and small models for rapid local validation;
- `main_macos`: 300 collection attempts and larger models.

Both configurations reserve about 10 percent of collected episodes for validation and 10 percent for held-out action evaluation. Collection updates its manifest atomically after each successful episode and records deterministic spawn/runtime rejections separately.

## Evaluation

Three policies are compared:

1. a proprioception-only reference policy;
2. Flat-conditioned policy;
3. Graph-conditioned policy.

The primary metric is closed-loop task success. Secondary metrics are wrong-object grasp rate, grasp-establishment rate, drop rate, completion steps, on-policy expert disagreement, and action error on fixed held-out expert frames. Results are reported separately for two, three, four, and five objects.

Flat and Graph use the same data, environment seeds, optimizer, training steps, action head, and model seeds. At least three training seeds are run. Test episodes are split at the episode/scene level, never at the frame level.

The practical pilot criterion is:

- Graph improves four- to five-object success by at least 10 absolute percentage points over Flat;
- all three model seeds improve in the same direction;
- Graph does not materially regress on two- to three-object scenes.

An edge-shuffle evaluation checks whether the trained Graph policy actually uses interaction structure. If shuffling valid edges leaves performance unchanged, the result does not support the representation hypothesis.

Improved training loss or offline action error alone is insufficient evidence.

## Mac Compatibility

The runtime selects Apple MPS when PyTorch exposes it and otherwise uses CPU. CUDA is never assumed. Headless collection is the default. Pilot models use small batches and checkpoint/resume support. Dependencies are split so graph-schema and environment tests can run without importing the VLA stack.

The existing CUDA-specific PyTorch requirement is replaced by platform-neutral installation guidance rather than forcing CUDA wheels on macOS.

## Module Boundaries

```text
interaction_vla/
  graph/schema.py       typed padded scene graph and validation
  graph/builder.py      snapshot-to-graph conversion
  env.py                deterministic tabletop environment contract
  expert.py             scripted expert and phase/result codes
  data.py               episode collection, storage, and splits
  models/encoders.py    FlatEncoder and GraphEncoder
  models/policy.py      shared one-step action policy
  train.py              reproducible training/checkpoint loop
  evaluate.py           paired closed-loop evaluation and reports
  config.py             validated YAML configuration
configs/
  pilot_macos.yaml
  main_macos.yaml
tests/
  interaction_vla/
```

Each module accepts simple typed values rather than reaching into notebook globals. MuJoCo-specific state extraction is isolated behind the environment snapshot interface.

## Testing and Failure Handling

Unit tests verify:

- deterministic scene generation from a seed;
- schema shapes, masks, finite values, and maximum capacities;
- correct relative, contact, holding, and support edges;
- numerical identity of Graph and Flat raw inputs before encoding;
- graph pooled-output invariance to node permutation;
- no post-action state in the stored current observation;
- device selection on CPU and MPS-capable systems.

Integration tests verify:

- collecting short two- through five-object episodes;
- overfitting a two-episode dataset;
- checkpoint save/resume;
- one CPU training and closed-loop evaluation smoke run;
- MPS smoke execution when MPS is available.

Configuration and schema violations fail early with actionable messages. Non-finite losses stop training immediately. Evaluation always saves per-episode rows in addition to aggregate metrics.

## Future Extensions

After the Mac pilot validates the pipeline, the same scene embedding can be injected as a condition token into a frozen SmolVLA backbone. Later work can remove privileged target flags, infer nodes and edges from RGB, add a next-state auxiliary head, and scale the task family for a paper-grade study.

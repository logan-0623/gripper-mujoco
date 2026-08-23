# LIBERO Interaction Representation Study Design

## Purpose

Build a shared, simulator-grounded LIBERO State Bank and a model-neutral analysis pipeline for longitudinal SmolVLA representation study. The system must separate accessibility, action sensitivity, and closed-loop utility for the same interaction factors while preserving all legacy ACT/Graph evidence.

## Scope

This implementation includes deterministic replay, privileged annotation, State Bank construction and audit, grouped splits, SmolVLA stage planning, semantic taps, latent caching, probes, factor-aligned intervention infrastructure, and paired LIBERO evaluation infrastructure. It stops before online RL and does not generate recovery labels from nominal demonstrations.

## Source contract

Two sources are bound:

1. a standard `LeRobotDataset` revision for policy observations, actions, language, and episode indexing;
2. a matching original LIBERO HDF5 source for simulator states, model XML, contacts, object poses, and deterministic replay.

Episodes are aligned by task metadata, action length, and the complete action sequence. Proprioception is retained and validated as part of the LeRobot observation schema, but is not used to force a raw-HDF5 match. Ambiguous matches fail. Every mapping and source hash is stored in `source_alignment/manifest.json`.

## Domain coverage

Formal v1 covers the single-goal relocation tasks in LIBERO Spatial and Object. The schema supports factor applicability and task-family plugins. Goal and Long tasks are admitted only after a reviewed semantics entry defines active entities, goal predicates, and an ordered task plan where needed.

## Core interfaces

```python
class ReplayAdapter(Protocol):
    def replay_episode(self, source: EpisodeSource) -> Iterable[ReplayFrame]: ...

class InteractionAnnotator(Protocol):
    def annotate_episode(self, frames: Sequence[ReplayFrame]) -> Sequence[InteractionLabels]: ...

class StateBank(Protocol):
    def records(self) -> Iterable[StateRecord]: ...

class PolicyAdapter(Protocol):
    def act(self, observation: Mapping[str, Tensor]) -> Tensor: ...

class LatentTapRegistry(Protocol):
    def capture(self, policy: PolicyAdapter, observation: Mapping[str, Tensor]) -> Mapping[str, TapValue]: ...

class ProbeRunner(Protocol):
    def fit_and_evaluate(self, bank: StateBank, latents: LatentCache) -> ProbeReport: ...

class InterventionRunner(Protocol):
    def intervene(self, factor: str, latent: TapValue, control: str) -> TapValue: ...

class ClosedLoopEvaluator(Protocol):
    def paired_rollout(self, case: InitialStateCase, intervention: InterventionSpec) -> PairedOutcome: ...
```

Existing backend and hashing utilities implement portions of these contracts and remain reused.

## Annotation model

All labels carry applicability masks. Entity, geometry, contact, stable grasp, phase, and next relation follow the exact definitions in `docs/research/2026-08-23-libero-vla-representation-audit.md`. Recovery is absent from the schema.

The task-semantics registry binds:

- task ID/name and BDDL hash;
- task family;
- target, goal, source, gripper, and distractor body/geom selectors;
- goal predicate and arguments;
- ordered subgoals when logically required;
- factor applicability;
- canonical phase transition rules.

Unknown or ambiguous entity selectors and plans are errors in formal mode. Smoke mode may use synthetic fixtures, but it may not emit a passing scientific audit for unsupported real tasks.

## Replay gate

Replay begins from the raw demonstration model XML and flattened simulator state, then steps the recorded action sequence. Per-transition flattened-state L2/max error and source action alignment are recorded; the flattened state already includes the simulator's robot and object state. A configurable quantile and maximum threshold gate determine pass/fail. A failed replay episode is preserved in the report and excluded from an accepted bank; a formal bank fails if acceptance or task-coverage requirements are not met.

## Stable grasp and event hysteresis

Stable grasp is windowed and cannot be true from contact alone. Phase rules use simulator contact events, the windowed stability signal, the privileged goal predicate, and hysteretic near relations. Initial frames without enough history are not evaluable for the windowed target; raw contact transitions and isolated positives are reported for manual inspection rather than silently smoothed.

## Splits

The bank has two immutable split manifests:

- `task_group.json`: disjoint task groups stratified by suite, primary generalization split;
- `episode_group.json`: disjoint episodes stratified within tasks, secondary in-task split.

Every record appears exactly once in each manifest; no source episode spans partitions. The smoke profile uses three tasks per suite and three held-out episodes per task so both split schemes have non-empty train/validation/test groups. The manifests report per-factor label support and suppress invalid comparisons rather than silently dropping missing classes.

## Stage contract

Stages are `pretrained`, `sft_25`, `sft_50`, and `sft_100`. Subsets are deterministic, episode-grouped, task-balanced, and nested. The primary comparison uses a fixed epoch budget so steps scale with data fraction. Stage manifests record base model, dataset revision, subset manifest/hash, seed, epochs, steps, code/config hashes, checkpoint path/hash, and one of `not_run`, `running`, `complete`, or `failed`.

No absent checkpoint is synthesized. A pretrained Hub identifier is resolved to an immutable revision before latent extraction.

## Semantic taps

The four tap roles and their exact SmolVLA tensors are preregistered in the audit document. `valid_token_mean` is primary. Captures are keyed by state ID, checkpoint ID, tap, pooling rule, and deterministic inference-noise seed. Partial caches are resumable and only finalized after exact state coverage and tensor validation.

## Probe protocol

Linear probes are primary; a one-hidden-layer MLP is a capacity check. Targets use applicability masks. Metrics are macro-F1/balanced accuracy for Entity, macro-F1 for Phase and NextRelation, normalized MAE and R² for Geometry, and AUPRC plus balanced accuracy for Contact/StableGrasp. Majority/mean, task-ID, exact-instruction, and normalized-time-bin baselines are included where applicable. Exact instruction and task ID remain separately named even when they coincide; a learned language-only baseline is a separately registered extension. `Accessible=true` requires the task/episode-bootstrap confidence interval, not only the point estimate, to clear the strongest registered simple baseline.

Confidence intervals resample tasks for the main split and episodes for the secondary split. Frames are never bootstrap units. Hyperparameters are selected on validation groups from a preregistered grid.

Generated timeline images do not automatically pass the semantic annotation gate. A separate explicit approval command revalidates the State Bank and every image hash before recording the human-review assertion; probes require that assertion.

## Intervention protocol

The primary factor-aligned intervention operates in the row space of a frozen linear probe and uses matched donor states to replace factor-associated coordinates while preserving the orthogonal component. Matching is constrained by suite/task family and declared nuisance factors. Distribution checks compare norm, mean, variance, and Mahalanobis distance.

Controls are original, matched random, matched mean, instruction shuffle where applicable, and whole-latent zero as OOD sanity only. The intervention report must show target-factor disruption and non-target specificity before any closed-loop utility run is considered valid.

## Paired closed-loop evaluation

Original and intervened rollouts share the exact LIBERO task, model XML/init state, language, seed, and policy inference-noise schedule. Reports include action deltas by translation/rotation/gripper, success, completion steps, contact/grasp failure, object-selection error, placement failure, and task-specific failure categories. Success contrasts use paired task/episode bootstrap intervals.

Action displacement establishes action sensitivity only. Closed-loop outcome changes are required for a usefulness statement.

## Artifact states

Every artifact uses `implementation_only`, `not_run`, `running`, `failed_gate`, `pilot_complete`, or `formal_evidence`. Code existence never promotes an experiment. Old reports remain under their existing roots and retain their original schema meanings.

## Stop condition

This work stops after paired closed-loop evaluation infrastructure and any explicitly requested non-RL executions. No recovery label, PPO/SAC tuning, residual reward design, or RL representation experiment is included.

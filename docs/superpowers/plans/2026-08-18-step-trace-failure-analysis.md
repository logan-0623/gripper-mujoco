# Step Trace and Failure-Conditioned Analysis Plan

**Goal:** Re-evaluate existing frozen Graph ACT policies with resumable, causal
step-level traces, then measure descriptive associations between Graph error exposure
and closed-loop failure modes.

## Task 1: Strict trace configuration and schema

- Add optional `trace` config with isolated output directory and explicit resume
  behavior; preserve compatibility of configs that omit it.
- Implement Graph-group error metrics, fixed-shape trace record validation, atomic
  per-episode JSONL writing, and complete-file loading in `tracing.py`.
- Record observation-time tokens/geometry separately from post-action contacts,
  events, and termination state.

## Task 2: Trace-capable rollout without policy changes

- Refactor the current rollout to compute the policy token once before each action,
  preserving the existing receding-horizon action path.
- Maintain a fully independent causal Teacher provider that is reset once per
  episode and updated once per policy step.
- Emit raw/local-clipped/world-executed actions, projection/clipping, poses,
  contact/grasp events, group errors, and terminal status through an optional trace
  callback.
- Prove with tests that traced and untraced rollouts select identical actions.

## Task 3: Resumable trace pipeline and CLI

- Predeclare the case schedule and bind it to config, split, dataset, cache/Graph,
  ACT checkpoint, trace-schema, and implementation hashes.
- Atomically write one trace file per `(policy_seed, condition, case_id)`.
- Resume only complete, schema-valid, manifest-compatible episode files; reject
  partial or incompatible artifacts.
- Aggregate the same episode outcomes as the original evaluation and publish
  `episodes.jsonl`, `report.json`, and a completed manifest.
- Add `graph_control trace --config ...` with episode progress.

## Task 4: Failure-conditioned analysis

- Implement event-aligned exposure windows and train-derived p75 high-error
  thresholds without using evaluation outcomes.
- Report counts, risk difference, risk ratio, and clustered uncertainty for success,
  timeout, target drop, and wrong-object stable grasp.
- Mark comparisons with fewer than five positive outcomes as underpowered and avoid
  causal wording.
- Add `graph_control failure-analysis --config ... --traces ...` and atomically
  publish a machine-readable report.

## Task 5: Verification

- Run pure schema/metric tests, rollout equivalence tests, resume/incompatibility
  tests, the complete Graph-control suite, and a one-case bounded real trace smoke.
- Do not start the full 240-episode Mac trace automatically; document the resumable
  Mac and preferred Linux/CUDA commands.


# Frozen-Policy Sensitivity Implementation Plan

**Goal:** Measure which Graph v2 token groups existing frozen ACT policies actually
use, without retraining or weakening the normal checkpoint validation path.

## Task 1: Retrospective checkpoint compatibility

- Keep `load_graph_act_checkpoint` strict by default.
- Add an explicit analysis-only compatibility mode that may ignore only the nested
  `graph_control.source_fingerprint` mismatch.
- Continue requiring exact dataset, feature, ACT config, codec, LeRobot version,
  cache, split/report, condition, and seed metadata.
- Return an audit record containing the stored and current Graph-control source
  fingerprints whenever compatibility mode is used.
- Cover strict rejection, scoped acceptance, and rejection of every other mismatch
  with unit tests.

## Task 2: Token interventions and action metrics

- Add `interaction_vla.graph_control.sensitivity` with pure functions for group
  masking, continuous finite differences, categorical probability-preserving
  perturbations, and 7D first-action change metrics.
- Treat an all-zero categorical slice as missing and leave it zero.
- Clip continuous perturbations to train-cache p01/p99 bounds.
- Reset policy state before every baseline or counterfactual prediction.
- Develop the module test-first with deterministic synthetic arrays.

## Task 3: Frozen-policy inference and report composition

- Select a deterministic, episode-balanced subset of the requested partition.
- Load every existing policy checkpoint with analysis-only compatibility auditing.
- Hold images, proprioception, language/preprocessing, and source row fixed while
  changing only `observation.environment_state`.
- Report masking and finite-difference effects per Graph group, policy seed,
  condition, and episode, with clustered 95% intervals.
- Bind reports to ACT checkpoint hashes, token cache hashes, split hash, selected
  rows, and compatibility audits.

## Task 4: Configuration, CLI, and verification

- Extend `diagnostics` config with bounded sensitivity batch/sample/perturbation
  settings and an isolated sensitivity output directory below the partition report.
- Add `graph_control sensitivity --config ... --partition ...`.
- Publish a compact terminal summary plus full `report.json` and
  `per_episode.jsonl` atomically.
- Run focused tests, the complete Graph-control suite, and a bounded real smoke on
  the existing Mac pilot checkpoint matrix.


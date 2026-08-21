# ICRA Representation Study: Phases 3–7 Implementation Plan

**Implementation status (2026-08-20):** Phases 3–7 are implemented and code-verified.
The fixed State Bank is populated. Formal ACT/SmolVLA SFT, residual-RL, probe, and
intervention experiments remain intentionally unrun; they are the next evidence gate.

**Goal:** Turn the representation-study contracts into a runnable, evidence-bound
pipeline that measures how interaction structure changes from pretrained to SFT and
residual RL policies. ACT remains the controlled mechanism study; SmolVLA is the
required modern-VLA validation; pi0 is an optional external-validity adapter.

**Scientific invariant:** Graph labels are a measurement ontology. They are never
silently appended to a policy input in this study. Every stage is evaluated on the
same immutable State Bank and the same latent-tap roles.

## Phase 3 — Fixed State Bank and ontology

1. Add a strict representation-study config loader and a single module CLI.
2. Build expert-support records from the existing LeRobotDataset and TC-TIG
   sidecars, using the existing episode-level train/validation/test split.
3. Build policy-shift records by deterministic replay of one configured rollout
   condition. Bind each replayed state to its trace file, environment seed, layout,
   object count, and executed action prefix.
4. Assign nominal, perturbation, recovery, and terminal strata with versioned,
   inspectable rules. Report expert-support and policy-shift domains separately.
5. Materialize replay observations into immutable NPZ shards only when extraction
   needs pixels. Validate replayed end-effector positions against the trace.
6. Publish `records.jsonl`, `split.json`, `manifest.json`, `ontology.json`, and
   `validation_report.json` atomically.

## Phase 4 — Latent extraction and frozen probes

1. Implement a generic forward-hook collector with deterministic tensor pooling.
2. Implement ACT, SmolVLA, and pi0 adapters with the four fixed scientific tap
   roles: visual, fused, policy input, and action proximal.
3. Extract one latent row per State Bank state and bind it to checkpoint/config,
   tap identity, State Bank hash, source-tree hash, dtype, and shape.
4. Train frozen linear probes as the primary measurement and shallow MLP probes as
   a secondary capacity check for entity, geometry, contact, stable grasp, phase,
   next relation, and recovery state.
5. Fit probes on State Bank train, select only training hyperparameters on
   validation, and report test once. Report class-balanced accuracy/F1 for
   categorical targets and MAE/R2 for continuous targets.

## Phase 5 — Access, use, and causal intervention

1. Separate representation accessibility (probe score) from policy use.
2. Support tap-level zero, matched-random, and mean replacement interventions.
3. Apply interventions to a frozen checkpoint and report action displacement
   offline; use closed-loop rollout deltas when the backend exposes a compatible
   runtime.
4. Bind every intervention report to its probe artifact and stage manifest.

## Phase 6 — SFT and residual RL stages

1. Register pretrained, SFT, continued-SFT, RL-head, and RL-representation stage
   manifests without changing legacy ACT outputs.
2. Implement residual control as `a = clip(a_sft + alpha * delta_a_rl)`.
3. Train a compact Gaussian actor/critic with PPO over a frozen base-policy latent
   for RL-head. For RL-representation, additionally unfreeze only the configured
   late representation group.
4. Use sparse task success as the primary reward; allow the configured minimal
   progress shaping only as a secondary run.
5. Save resume-safe optimizer, RNG, environment-step, return, and evaluation state.
6. Report normalized learning-curve AUC as primary plasticity, with final return
   gain and steps-to-threshold as secondary metrics.

## Phase 7 — Modern VLA validation and study report

1. Add SmolVLA SFT configuration/launch support using the standard LeRobotDataset
   and Hugging Face checkpoint interface.
2. Reuse the same State Bank extraction, probes, interventions, residual-RL
   contract, and statistics through the SmolVLA backend adapter.
3. Add an optional pi0 adapter with the same contracts; it must fail clearly when
   its checkpoint or hardware requirements are absent and is not a required gate.
4. Aggregate stagewise `C` (correctness/accessibility), `U` (closed-loop control
   utility), and `P` (RL plasticity) without claiming causality from correlation.
5. Produce machine-readable result rows for the ICRA tables and a concise study
   report. Preserve all existing ACT Graph-vs-Flat artifacts as prior controlled
   evidence.

## Required verification

- Unit tests cover schemas, replay/selection determinism, episode leakage,
  ontology targets, hook pooling, probe splits, interventions, residual-action
  bounds, PPO resume state, AUC/statistics, and backend manifest compatibility.
- CPU smoke tests do not download large model weights.
- ACT end-to-end smoke uses the existing local dataset/checkpoint.
- SmolVLA/pi0 adapter inspection works without constructing their full weights.
- Full existing test suite remains green and README commands match the CLI parser.

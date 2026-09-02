# SmolVLA Official Positive-Control Kill Test

## Objective

Determine whether the failed Protocol-v3 StableGrasp recruitment result was caused by
closed-loop-incompetent expert-only SFT checkpoints or by the absence of measurable
StableGrasp dependence in a successful SmolVLA policy.

This is a zero-training decision experiment. It adds the public
`lerobot/smolvla_libero` checkpoint as a successful-policy positive control while
preserving Protocol-v3 as immutable historical evidence.

## Scientific status before this experiment

Protocol-v3 established that StableGrasp accessibility increased during early
expert-only SFT. Its preregistered functional-recruitment gate failed because the
StableGrasp-targeted first-action displacement did not exceed the same-rank,
same-norm matched-random intervention at any tested checkpoint.

The old checkpoints used a frozen upstream pathway and were previously evaluated
with an invalid LIBERO rollout protocol. The corrected open-source LeRobot protocol
produces substantially higher success for the official checkpoint. The positive
control therefore separates policy floor from intervention/factor failure.

## Scope

Primary cell:

```text
official smolvla_libero
× action_expert_input
× StableGrasp
× episode-group cross-fit
```

Controls:

- Contact, Phase, and Geometry probe disruption for specificity;
- same-rank, same-norm orthogonal random intervention;
- matched-mean intervention as a secondary control;
- zero intervention only as an out-of-distribution sanity check.

The experiment does not retrain a policy, regenerate the State Bank, alter
Protocol-v3 artifacts, run RL, or run paired closed-loop interventions.

## Inputs

- Existing audited 13,603-state LIBERO State Bank.
- Existing episode-group cross-fit fold construction and shortcut baselines.
- Existing StableGrasp, Contact, Phase, and Geometry annotations.
- Local immutable snapshot of `lerobot/smolvla_libero`.
- A completed official LeRobot evaluation report generated with:
  - `policy.n_action_steps=10`;
  - `policy.empty_cameras=1`;
  - LIBERO camera-name mapping;
  - one environment per stateful policy;
  - `eval.recording=false`.

All artifacts are bound by SHA-256. Absolute server paths are supplied through the
CLI and are not stored in configuration defaults.

## Artifact layout

```text
outputs/representation_study/libero_smolvla/protocol_v4/positive_control/
├── plan.json
├── latents/
│   └── action_expert_input/
├── probe/
├── intervention/
└── report.json
```

Protocol-v3 remains unchanged under `protocol_v3/`.

## Commands

The implementation exposes one new `positive-control` command family. `evaluate`
must generate the immutable checkpoint/command/evaluation contract; historical
`eval_info.json` files without that provenance are not accepted:

```text
libero positive-control evaluate
libero positive-control plan
libero positive-control extract
libero positive-control probe
libero positive-control intervene
libero positive-control report
```

`evaluate` invokes the upstream LeRobot evaluator with the preregistered runtime
contract and seals its exact command, checkpoint hash, and output hash.

`plan` records the checkpoint hash, evaluation-report hash, observed baseline
success, State Bank hash, policy configuration, and exact open-source rollout
contract. It fails if the checkpoint or evaluation result is incomplete.

`extract` reuses the existing SmolVLA adapter and deterministic inference noise but
stores only `action_expert_input`. It is resumable at the state-row level.

`probe` reuses Protocol-v3 episode-group folds, matched probe seeds, linear probe,
and shortcut baselines. It computes StableGrasp accessibility and the three control
factor cells without creating a second probe protocol.

`intervene` reuses the fold-held-out StableGrasp basis and the existing final
denoising hook. It evaluates 1,600 deterministically selected states unless
overridden for a smoke run.

`report` applies the gates below and records exactly one decision.

## Gates

### G0 — Successful-policy baseline

The imported official evaluation must have baseline success greater than 0.20.
This is a floor screen, not a benchmark claim. A result at or below the threshold
fails the experiment because destructive functional dependence is not estimable.

### G1 — Accessibility

The Protocol-v3 cross-fit StableGrasp probe must exceed the strongest preregistered
shortcut baseline on held-out episode groups. No probe is trained on its evaluation
fold.

### G2 — Specificity

The StableGrasp-targeted intervention must disrupt StableGrasp probe behavior more
than the matched-random intervention in aggregate. It must not collapse all control
factor probes or move latent norms outside the registered natural-support bounds.

### G3 — Functional usage

The primary quantity is

```text
U_SG = first-action displacement(targeted)
       - first-action displacement(matched-random)
```

The primary statistic is total first-action displacement with an episode-clustered
95% bootstrap interval. Functional usage passes only when the interval lower bound
is strictly positive.

Translation L2, rotation L2, gripper absolute delta, gripper-command flip rate,
full action chunk, and phase-conditioned effects are secondary diagnostics. They do
not override the primary gate.

## Decision policy

### StableGrasp passes G0–G3

The longitudinal idea survives. The next experiment is one official-style full-SFT
trajectory with model checkpoints at approximately 0, 6.25k, 12.5k, and 25k
updates. Accessibility and functional usage are then measured at every checkpoint.

### StableGrasp passes accessibility but fails functional usage

Run exactly one preregistered secondary replication using Contact at the same tap,
states, folds, controls, and statistic. Do not tune the StableGrasp intervention.

### StableGrasp and Contact both fail functional usage

Stop the natural functional-recruitment study. Do not train the longitudinal
official-style trajectory. The candidate pivot is interaction-supervised SFT:
whether explicit Contact/StableGrasp supervision can turn accessible interaction
information into control-aligned representations and improve low-data closed-loop
performance.

### Accessibility fails

Treat the selected factor/tap ontology as invalid for this policy. Contact receives
the same single replication. Failure of both factors kills this measurement route.

## Storage and compute

Only one 720-dimensional pooled tap is cached. At 13,603 states in float32, the raw
latent matrix is approximately 40 MB before metadata. No new checkpoint or optimizer
state is produced.

The full-SFT trajectory is prohibited until G3 passes. If authorized later, only
model artifacts are retained at scientific checkpoints; optimizer states are kept
only as needed for active resume and are not publication artifacts.

## Implementation changes

- Add a focused official positive-control module under
  `interaction_vla/representation_study/libero/`.
- Add the five CLI subcommands to the existing LIBERO command tree.
- Let the existing latent extractor accept an explicit tap subset while preserving
  its current all-tap default.
- Reuse existing cross-fit and StableGrasp intervention functions; do not implement
  another evaluator or probe library.
- Add one unit-test file for plan binding, tap selection, and gate decisions, plus
  CLI parser coverage.
- Add server commands and artifact descriptions to `SERVER_RUNBOOK.md`.
- Update `ccfa.yaml` without changing historical experiment records.

No dependency is added.

## Error handling and immutability

- Existing artifacts with different bindings cause a hard failure.
- Missing evaluation summaries, stale checkpoint hashes, altered State Bank hashes,
  or incompatible action-expert tensors stop the experiment.
- Partial latent extraction resumes only when every immutable binding matches.
- Failed gates are written as failed evidence; they are not silently retried with a
  new threshold, factor, tap, or intervention norm.

## Tests

The smallest required checks are:

1. plan rejects a missing or changed checkpoint and a floor-level evaluation;
2. selected-tap extraction writes only `action_expert_input` while the old all-tap
   path remains unchanged;
3. report passes only when G0–G3 pass;
4. StableGrasp failure routes to Contact exactly once;
5. StableGrasp and Contact failure routes to the declared pivot and forbids full-SFT
   longitudinal expansion;
6. Protocol-v3 files are unchanged by all positive-control commands.

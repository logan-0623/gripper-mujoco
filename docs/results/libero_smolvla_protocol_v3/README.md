# SmolVLA LIBERO Longitudinal Protocol v3

This directory preserves the lightweight, machine-readable evidence bundle from
the completed protocol-v3 server run. The JSON files are copied unchanged from
their experiment output paths.

## Evidence status

- The cross-fitted probe report passed its integrity, paired-delta, and
  identical-latent gates.
- It contains 384 cells: 288 complete, 96 not estimable, and zero failed or
  unrun cells.
- The 96 non-estimable cells are Task-group Entity and NextRelation cells whose
  held-out folds contain classes absent from their training folds.
- The StableGrasp specificity report passed, but the preregistered offline
  action-sensitivity gate failed: none of the four checkpoints showed larger
  first-action displacement under the targeted intervention than under the
  matched-random control.
- The package therefore supports accessibility and offline action-sensitivity
  claims, but not positive functional recruitment or closed-loop utility.
- The primary report SHA-256 is
  `39bb49875be240fc72effc743cc28e629fd2e1b0c929e00e85a4e2d0de8a581d`.

The primary result is
[`protocol_v3/probes/crossfit_v1/report.json`](protocol_v3/probes/crossfit_v1/report.json).
Its paired folds, longitudinal condition plan, and latent-integrity gate are
stored beside it under [`protocol_v3/`](protocol_v3/).

The frozen StableGrasp follow-up is under
[`protocol_v3/recruitment/stable_grasp/n_1600/`](protocol_v3/recruitment/stable_grasp/n_1600/):

- `specificity.json`: factor-specificity gate, SHA-256
  `30d6d25ac8baef0cc2b52e4eae6582fda3f9c594e1cbc89a63f7ebc2e8cc5f8c`;
- `action_sensitivity.json`: checkpoint-wise action effects, SHA-256
  `87c9fc9e47860caa5b1bea5a894dc37b694fec0f48a93399faf23002abc46119`;
- `cached_analysis.json`: paired checkpoint deltas, state-conditioned effects,
  action components, and full-chunk secondary analysis, SHA-256
  `bc462bbd9ad394df917d4e0c97d75add12d5f563462064aff787be0fe9a6278b`.

For the primary first-action metric, paired `after − before` episode-cluster
estimates were −0.0002566 for Pretrained→D25@16k (95% CI
[−0.0003532, −0.0001680]), +0.0000793 for D25@16k→D100@16k
([+0.0000157, +0.0001411]), and −0.0000468 for D100@16k→D100@66k
([−0.0001017, +0.0000062]). The matched-random basis is checkpoint-specific,
so these are paired changes in checkpoint-defined `U`, not effects under one
fixed perturbation direction.

The remaining directories preserve the compact provenance chain: deterministic
replay, source alignment, State Bank audit, timeline approval, and the four
checkpoint-stage manifests.

No model weights, optimizer states, latent arrays, per-state action-row caches,
per-cell prediction caches, videos, images, or LIBERO dataset files are included.

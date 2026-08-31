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
- This package supports claims about **representation accessibility only**. It
  does not measure functional use or closed-loop utility.
- The primary report SHA-256 is
  `39bb49875be240fc72effc743cc28e629fd2e1b0c929e00e85a4e2d0de8a581d`.

The primary result is
[`protocol_v3/probes/crossfit_v1/report.json`](protocol_v3/probes/crossfit_v1/report.json).
Its paired folds, longitudinal condition plan, and latent-integrity gate are
stored beside it under [`protocol_v3/`](protocol_v3/).

The remaining directories preserve the compact provenance chain: deterministic
replay, source alignment, State Bank audit, timeline approval, and the four
checkpoint-stage manifests.

No model weights, optimizer states, latent arrays, per-cell prediction caches,
videos, images, or LIBERO dataset files are included.

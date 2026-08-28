# SmolVLA LIBERO Probe Protocol v2 Pilot

This directory preserves the lightweight reports from the completed protocol-v2
server run. The source repository revision was `6a05e8f`.

## Evidence status

- The probe report passed its implemented integrity checks but was not complete:
  80 of 96 primary cells and 80 of 96 secondary cells completed.
- Task-group Entity cells failed because test classes were absent from training.
- Episode-group NextRelation cells failed the corresponding class-support gate.
- Latents were extracted across different server runtimes. This cross-runtime
  confound prevents protocol v2 from supporting the formal longitudinal claim.
- These files therefore remain `pilot_complete`, not formal protocol-v3 evidence.

The exact machine-readable result is [`probe_report.json`](probe_report.json).
The [`stages/`](stages/) directory preserves checkpoint manifests and training
reports for Pretrained, SFT-25, SFT-50, and SFT-100.

No model weights, optimizer states, latent arrays, row caches, videos, or benchmark
data are included. Protocol-v3 reports did not yet exist when this package was
archived on 2026-08-28; they must be added only after the same-runtime gate runs.

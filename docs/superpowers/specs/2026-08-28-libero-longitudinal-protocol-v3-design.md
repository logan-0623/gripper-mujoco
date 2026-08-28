# LIBERO Longitudinal Protocol v3

## Goal

Remove the current cross-server latent-cache confound before interpreting SFT-stage
representation changes. Reuse saved SmolVLA checkpoints to separate supervision
diversity from optimization exposure.

## Conditions

The protocol binds these existing checkpoints by recorded training step:

| Condition | Dataset fraction | Update step |
|---|---:|---:|
| `pretrained` | 0% | 0 |
| `d25_u16070` | 25% | 16070 |
| `d50_u16324` | 50% | 16324 |
| `d100_u16617` | 100% | 16617 |
| `d50_u32650` | 50% | 32650 |
| `d100_u33234` | 100% | 33234 |
| `d100_u49851` | 100% | 49851 |
| `d100_u66470` | 100% | 66470 |

This supports matched-update comparisons near 16k and 32k updates and a D100
optimization trajectory without retraining.

## Scientific gate

Every condition must be extracted from the same State Bank with the same latent
implementation, taps, pooling, Python/PyTorch/LeRobot/Transformers runtime, CUDA
device class, model dtype, and extraction batch size. A differing runtime fingerprint
fails the gate. Existing protocol-v2 artifacts remain immutable.

## Artifacts

Artifacts live below `protocol_v3/`:

- `conditions/manifest.json`: immutable checkpoint/contrast plan;
- `latents/<condition>/<tap>/`: resumable latent caches;
- `latents/<condition>/report.json`: checkpoint and runtime binding;
- `latent_gate/report.json`: cross-condition validation.

Cross-fit probes are deliberately downstream of this gate. They must not consume the
old mixed-runtime caches.

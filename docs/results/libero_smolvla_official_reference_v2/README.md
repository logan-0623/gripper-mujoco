# Official SmolVLA Spatial reference v2

Measured on server commit `ded857c` with official checkpoint tree SHA-256
`3154ece6ac5f6e78bea3617a99f38d4d6eeaaeb43c476f70670310d8d2c4707a`.
All tasks use initial-state IDs 10–19, `n_action_steps=10`, ten denoising
steps, one empty camera, one synchronous environment, and physical-events v2.

| Spatial task | Success | Contact | Stable grasp | Geometric lift | Supported lift | Drop | Recovery |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 8/10 | 10/10 | 8/10 | 9/10 | 8/10 | 0/10 | 0/10 |
| 1 | 7/10 | 10/10 | 9/10 | 7/10 | 7/10 | 0/10 | 0/10 |
| 2 | 7/10 | 10/10 | 10/10 | 9/10 | 9/10 | 2/10 | 1/10 |
| 3 | 8/10 | 10/10 | 9/10 | 9/10 | 9/10 | 1/10 | 0/10 |

The SFT100 audit maps all 254 training episodes to Spatial 0–2 and Object
0–2. Spatial task 3 is therefore a generalization evaluation for this SFT
lineage. The SFT100 checkpoint is hash-complete and resume-ready; the old v1
manifest does not record its planned batch, and the new reproduction protocol
fields remain unresolved.

The real-model fixed-point smoke queried checkpoint 008216 at checkpoint
016435's exact `x_sigma`, epsilon, and sigma values for 8 states and 3 repeats.
All fixed arrays matched exactly. Velocity stage RMS increased from `0.072721`
at sigma 1.0 to `0.126267` at sigma 0.1. This validates the measurement path;
it is not mechanism evidence.

Raw JSON, audit output, and fixed-query bindings are retained beside this file.
The archive checksum is recorded in `SHA256SUMS`.

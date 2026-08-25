# Formal LIBERO Interaction State Bank

This directory records the completed infrastructure gate for the shared LIBERO State Bank used by the SmolVLA longitudinal representation study. The automatic audit passed, and all 12 sampled annotation timelines were manually reviewed and approved.

These artifacts establish replay, annotation, split, and review quality. They are not evidence that one SmolVLA checkpoint has a better representation or higher control success than another; those claims require the later latent, probe, and closed-loop experiments.

## Summary

| Item | Result | Gate |
|---|---:|---:|
| Suites / tasks | 2 / 20 | all configured tasks present |
| Selected episodes | 100 | 5 per task |
| State Bank states | 13,603 | no missing or invalid labels |
| Selected replay acceptance | 100 / 100 | 100% |
| Candidate replay yield | 100 / 120 | 83.3% |
| Replay qpos L2 p95 | 0.00793 | at most 0.01 |
| Replay maximum absolute error | 0.03938 | at most 0.05 |
| Stable grasp without gripper–target contact | 0 | required 0 |
| Annotation timelines | 12 / 12 approved | manual review required |

The main task-group split contains 9,387 train, 2,881 validation, and 1,335 test states with no task or episode overlap. The secondary episode-group split contains 8,196 train, 2,612 validation, and 2,795 test states with no episode overlap; task overlap is intentional for this in-task split.

## Replay caveat

Replay candidates were filtered by the preregistered tolerances before selecting five episodes per task. Rejections were not uniform: `libero_spatial/task_0` and `task_6` each yielded 5 valid episodes from 6 candidates, while `libero_spatial/task_7` yielded 5 from 23. Every other task yielded 5 from 5. The selected State Bank is balanced, but downstream work must report this task-dependent replay compatibility rather than treating the 100 selected episodes as an unfiltered sample.

## Included artifacts

- `state_bank_manifest.json`: immutable identities and State Bank dimensions.
- `state_bank_audit.json`: label distributions, geometry ranges, split checks, and audit gates.
- `collection_report.json`: source alignment and collection provenance.
- `replay_report.json`: per-candidate replay diagnostics and selected episodes.
- `timeline_report.json`: approval state and timeline identities.
- `timelines/`: the 12 lightweight, human-readable review panels.

The 32 MB `records.jsonl`, exact split maps, source-alignment index, episode shards, videos, latents, checkpoints, optimizer state, and model parameters are intentionally excluded from Git. Their identities remain recorded in the manifests so an external artifact release can be verified later.

The run used [`libero_smolvla_linux_cuda.yaml`](../../../configs/representation_study/libero_smolvla_linux_cuda.yaml) at source commit `d62f75c`. It binds the official raw LIBERO demonstrations to an immutable `lerobot/libero` revision; exact source and configuration identities are stored in the JSON reports. The timeline images are derived benchmark observations and remain subject to the upstream dataset's terms in addition to this repository's software license.

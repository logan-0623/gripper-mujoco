# Official SmolVLA positive control

The official checkpoint succeeded in 9/10 rollouts on `libero_spatial` task 0. On the fixed 13,603-state bank, Contact, StableGrasp, and Phase were accessible at `action_expert_input`, but rank-one Contact and StableGrasp erasure changed actions less than same-norm matched-random directions.

This is a failed functional-recruitment gate, not evidence that the policy never uses interaction information. [report.json](report.json) records the exact metrics and hashes; weights, latents, NPZ payloads, and videos are intentionally excluded.

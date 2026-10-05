# SmolVLA capability timeline and label-blind flow discovery

This archive records the current exploratory evidence for one self-trained
SmolVLA lineage (`seed=1000`, checkpoints 5k--25k). The official SmolVLA is an
external capability reference and is not part of this within-lineage timeline.

The experiment has two separate parts:

1. **E0 capability phenotype:** the same 40 simulator initial-state IDs
   (`0--39`) were evaluated for each LIBERO Spatial task and checkpoint. The
   recorder stores contact, stable grasp, geometric lift, supported lift,
   unintended drop, recovery, and success.
2. **E1/E2 internal computation:** a label-blind flow trace uses 512 StateBank
   discovery states, three checkpoint-independent noise repeats, and the same
   solver inputs across checkpoints. Shared PCA/SVD directions are frozen from
   5k→25k differences and then projected through the 15k/20k intermediate
   checkpoints.

## E0 capability phenotype

Each cell contains 40 episodes. The aggregate table is descriptive and is not
a benchmark score.

| checkpoint | contact | stable grasp | geometric lift | supported lift | success |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5k | 110/160 | 31/160 | 31/160 | 23/160 | 20/160 (12.5%) |
| 10k | 160/160 | 117/160 | 114/160 | 104/160 | 95/160 (59.4%) |
| 15k | 159/160 | 143/160 | 142/160 | 141/160 | 134/160 (83.8%) |
| 20k | 160/160 | 141/160 | 145/160 | 141/160 | 132/160 (82.5%) |
| 25k | 158/160 | 146/160 | 146/160 | 143/160 | 138/160 (86.3%) |

Success by task (`task 0`, `task 1`, `task 2`, `task 3`) is:

| checkpoint | task 0 | task 1 | task 2 | task 3 |
| ---: | ---: | ---: | ---: | ---: |
| 5k | 1/40 | 6/40 | 12/40 | 1/40 |
| 10k | 25/40 | 8/40 | 31/40 | 31/40 |
| 15k | 35/40 | 33/40 | 28/40 | 38/40 |
| 20k | 30/40 | 38/40 | 33/40 | 31/40 |
| 25k | 30/40 | 37/40 | 35/40 | 36/40 |

The main behavioral transition is concentrated between 5k and 15k, followed
by task-dependent stabilization. The non-monotonic task curves mean that this
lineage should not be described as a single universal “grasp feature” or as a
monotonic capability proof. The E0 table supplies `S_k`, the behavioral axis
used to interpret later representation changes; it does not establish causal
feature use.

The actual-state identity is part of the contract: `initial_state_count=50`
and the executed IDs are exactly `0--39` for every task and checkpoint. The
summary validator checks duplicates, missing states, episode counts, and
cross-checkpoint pairing.

## E1-v2 flow trace and E2 candidates

The formal flow trace is stored on the authorized server at:

```text
/root/autodl-tmp/smolvla-official-reproduction-v2/acquisition/flow_e1_v2_train512_r3
```

It contains complete natural traces for 5k, 10k, 15k, 20k, and 25k. Each
checkpoint has:

- 512 train discovery states and 48 independent source episodes;
- three shared `epsilon` tensors, generated independently of checkpoint;
- identical state-ID order, `epsilon`, and `sigma` arrays across checkpoints;
- ten solver stages with `x_sigma`, `velocity`, middle/late expert activations,
  final latent action, and postprocessed action;
- atomic shard manifests and binding hashes.

The frozen, label-blind candidates are archived at:

```text
/root/autodl-tmp/smolvla-official-reproduction-v2/acquisition/candidates_flow_e1_v2_train512
```

For both `expert_middle` and `expert_late`, the candidate set contains 12
directions: four `formation`, four `low_change`, and four `matched_random`
controls. Candidate selection uses no Contact, StableGrasp, success, or other
physical labels. The full five-checkpoint trajectories are in each tap's
`trajectory_full5/` directory.

The old 128-state flow traces and their derived candidates were removed from
the active experiment tree because they use an obsolete E1 contract. Their
small binding/report archive is retained at:

```text
/root/autodl-tmp/smolvla-official-reproduction-v2/acquisition/archive_invalid_e1_n128_20260922
```

## Evidence boundary and next gate

The current evidence supports:

- a task-heterogeneous capability trajectory in this training lineage;
- reproducible, checkpoint-paired changes in action-expert computation;
- candidate directions that can be followed through the full training time
  axis without using physical labels for selection.

It does **not** yet support:

- a semantic name for any candidate direction;
- a claim that the VLM, projector, or action expert is the source of the
  change;
- a claim that a candidate is used by the policy for grasping;
- a closed-loop or task-success effect of candidate intervention.

The next authorized stage is an offline action-effect gate on validation
episodes, with candidate suppression matched to same-norm random, low-change,
and no-op controls. Closed-loop rollout remains blocked until that comparison
is inspected and a candidate/stage/dose contract is frozen.

## Files and provenance

- `lineage.json`: checkpoint hashes and shared training contract.
- `report.json`: validated E0 per-task and aggregate event counts.
- `e0_capability_analysis.json`: compact E0 `S_k` analysis and interpretation
  boundary.
- `results.csv`: machine-readable E0 table when included in the archive.
- Server flow/candidate paths above: large raw E1/E2 artifacts, not checked
  into Git.

The E0 source summary is:

```text
/root/autodl-tmp/smolvla-official-reproduction-v2/acquisition/timeline_v2_states0_39_summary/report.json
```

All values here are from the current v2 recorder and summary validator; older
pilot reports in `libero_smolvla_acquisition_flow/` remain historical and must
not be merged with this timeline.

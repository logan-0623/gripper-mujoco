# SmolVLA capability-acquisition timeline

This archive contains the frozen evaluation summary for one self-trained
SmolVLA lineage (`seed=1000`, checkpoints 5k--25k). All checkpoints use the
same 40 paired initial conditions: LIBERO Spatial tasks 0--3, 10 states per
task. The released official model is not part of this comparison.

| step | contact | stable grasp | geometric lift | supported lift | success |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5k | 29/40 | 11/40 | 13/40 | 8/40 | 5/40 (12.5%) |
| 10k | 39/40 | 29/40 | 30/40 | 29/40 | 25/40 (62.5%) |
| 15k | 40/40 | 36/40 | 36/40 | 36/40 | 30/40 (75.0%) |
| 20k | 38/40 | 35/40 | 36/40 | 35/40 | 33/40 (82.5%) |
| 25k | 40/40 | 38/40 | 39/40 | 38/40 | 38/40 (95.0%) |

The largest paired behavioral transition is 5k to 10k: 22 initial conditions
change from failure to success and 2 change from success to failure, for a net
increase of 50 percentage points. Adjacent success changes after 10k are:
10k--15k, 9 gains/4 losses; 15k--20k, 5/2; and 20k--25k, 6/1.

The primary unlabeled representation comparison is therefore frozen as 5k
(weak), 10k (transition), and 25k (strong). The 15k and 20k checkpoints remain
held-out trajectory checks for whether a candidate follows acquisition rather
than merely separating two endpoints. Physical labels are reserved for
post-discovery interpretation and validation.

Files:

- `lineage.json`: checkpoint hashes and shared training contract.
- `report.json`: full per-task and aggregate event counts with source hashes.
- `results.csv`: machine-readable per-task table.

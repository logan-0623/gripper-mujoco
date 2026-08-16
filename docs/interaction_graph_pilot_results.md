# Interaction Graph Pilot: Three-Seed Result

Run date: 2026-08-01  
Configuration: `configs/pilot_macos.yaml` on CPU  
Training: 50 collection attempts, 40 training episodes after episode-level splitting, 80 epochs per representation and seed  
Evaluation: 20 paired closed-loop cases for each of 2, 3, 4, and 5 objects

This is an exploratory pilot result, not paper-level evidence. Flat and Graph used identical privileged state, demonstrations, batch order, shared action head, optimizer settings, evaluation cases, and encoder parameter budget.

## Paired closed-loop success

| Model seed | Flat ID | Graph ID | Graph−Flat ID | Flat OOD | Graph OOD | Graph−Flat OOD | Shuffled-Graph OOD |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 7.5% | 7.5% | 0.0 pp | 2.5% | 2.5% | 0.0 pp | 0.0% |
| 1 | 7.5% | 37.5% | +30.0 pp | 2.5% | 20.0% | +17.5 pp | 5.0% |
| 2 | 0.0% | 5.0% | +5.0 pp | 0.0% | 5.0% | +5.0 pp | 2.5% |

The mean OOD success improvement was **+7.5 percentage points**. The predeclared criterion required at least +10 points and improvement for every seed; seed 0 tied, so the full criterion was **not met**. Graph did not regress on mean ID success.

## Awareness diagnostics

Across the three seeds, mean wrong-object rate was 13.3% for Flat and 5.0% for Graph. Held-out normalized action MSE was also lower for Graph in every seed:

| Model seed | Flat held-out normalized MSE | Graph held-out normalized MSE |
|---:|---:|---:|
| 0 | 0.1437 | 0.0932 |
| 1 | 0.1989 | 0.0376 |
| 2 | 0.1229 | 0.0772 |

Shuffling Graph edge assignments reduced overall success for every seed (5.0%→2.5%, 28.75%→3.75%, and 5.0%→1.25%). This supports the narrower claim that the trained Graph policies used relational edge assignments. However, absolute success remains low and seed variance is large, so the current pilot is suggestive rather than a validation of `Graph > Flat`.

The complete machine-readable report and all 720 per-episode rows are written to `outputs/interaction_vla/pilot/evaluation/report.json` and `episodes.csv` after running the documented commands.

## Frozen Stage A: interaction-crowding OOD

Before generating any recovery trajectories, the same six pilot checkpoints were evaluated on 120 new deterministic cases: 40 `id_normal`, 40 `count_ood`, and 40 `crowded_ood`. Graph edge shuffling was evaluated on the same cases, producing 1,080 episode rows in total. The scripted expert solved all 40 crowded cases, so failures below reflect learned-policy behavior rather than invalid layouts.

| Seed | Flat ID | Graph ID | Flat count OOD | Graph count OOD | Flat crowded OOD | Graph crowded OOD | Graph−Flat crowded | Shuffled Graph crowded |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 7.5% | 7.5% | 0.0% | 2.5% | 2.5% | 5.0% | +2.5 pp | 0.0% |
| 1 | 7.5% | 37.5% | 12.5% | 12.5% | 2.5% | 30.0% | +27.5 pp | 7.5% |
| 2 | 0.0% | 5.0% | 2.5% | 0.0% | 2.5% | 2.5% | 0.0 pp | 0.0% |

The mean crowded-OOD Graph−Flat success delta is **+10.0 percentage points**. The strict criterion is still not met because seed 2 ties rather than improves. Crowding lowers mean Flat success from 5.0% on normal four-/five-object scenes to 2.5%, while Graph's mean rises from 5.0% to 12.5% because of the high-variance seed-1 result. Thus this first crowded condition is meaningfully harder for Flat, but not uniformly harder for every learned policy.

Mean crowded wrong-object rate is 16.7% for both representations. Relative to normal count OOD, it rises from 14.2% for Flat and 7.5% for Graph, confirming that the target-centered distractor changes interaction errors. Shuffling Graph edges reduces crowded success for all three seeds (5.0%→0.0%, 30.0%→7.5%, and 2.5%→0.0%).

The frozen Stage A artifacts are `outputs/interaction_vla/crowded_baseline/evaluation/report.json` and `episodes.csv`. Recovery comparisons must use these exact cases and this baseline file.

## Stage B: deterministic recovery augmentation

Stage B adds 40 deterministic recovery trajectories, one for every base training episode and none from validation or held-out test seeds. All 40 recoveries terminate successfully; generation has zero rejections. The training set contains the original 40 base training episodes plus these 40 post-perturbation trajectories. Crowded layouts remain evaluation-only. PyTorch reported MPS unavailable in this Mac runtime, so the real run used the configured CPU fallback and the MPS-only test was skipped.

| Seed | Flat ID | Graph ID | Flat count OOD | Graph count OOD | Flat crowded OOD | Graph crowded OOD | Graph−Flat crowded | Shuffled Graph crowded |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 5.0% | 20.0% | 5.0% | 25.0% | 0.0% | 25.0% | +25.0 pp | 0.0% |
| 1 | 17.5% | 57.5% | 25.0% | 2.5% | 7.5% | 5.0% | −2.5 pp | 0.0% |
| 2 | 17.5% | 67.5% | 10.0% | 25.0% | 7.5% | 12.5% | +5.0 pp | 0.0% |

The mean Stage B Graph−Flat crowded-OOD success delta is **+9.17 percentage points**. This is below the 10-point threshold, and seed 1 reverses direction, so the strict representation criterion is **not met**. Edge shuffling does lower crowded success for every Graph seed, from 25.0%, 5.0%, and 12.5% to 0.0% in all three cases.

The exact recovery-minus-Stage-A changes show substantial seed variance:

| Seed | Flat crowded success Δ | Flat crowded grasp Δ | Graph crowded success Δ | Graph crowded grasp Δ | Flat crowded wrong object | Graph crowded wrong object |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | −2.5 pp | −25.0 pp | +20.0 pp | +2.5 pp | 12.5% | 2.5% |
| 1 | +5.0 pp | +37.5 pp | −25.0 pp | −35.0 pp | 50.0% | 5.0% |
| 2 | +5.0 pp | +22.5 pp | +10.0 pp | −5.0 pp | 25.0% | 0.0% |

Mean crowded success improves by +2.5 points for Flat and +1.67 points for Graph, satisfying only the exploratory recovery-help signal; the per-seed inconsistency prevents a stronger claim. The clearer object-awareness result is that Stage B crowded wrong-object rate averages **29.2% for Flat versus 2.5% for Graph**.

Graph also has lower base-held-out normalized action MSE for every seed:

| Seed | Flat normalized MSE | Graph normalized MSE | Flat physical MSE | Graph physical MSE |
|---:|---:|---:|---:|---:|
| 0 | 0.2094 | 0.0587 | 0.01218 | 0.00064 |
| 1 | 0.1830 | 0.0451 | 0.00511 | 0.00058 |
| 2 | 0.1219 | 0.0595 | 0.00380 | 0.00300 |

These results support the narrow conclusion that the Graph encoder is more object-aware and uses relational edges in this controlled state-policy experiment. They do not establish a uniform closed-loop success advantage: absolute success is still low, recovery effects are seed-sensitive, and the predeclared criterion fails. The complete Stage B report, strict case-paired Stage A comparison, and all 1,080 episode rows are in `outputs/interaction_vla/recovery/evaluation/report.json` and `episodes.csv`.

## Next experiment

The next controlled improvement would be broader recovery coverage or DAgger-style corrections with more model seeds, while retaining the frozen crowded evaluation. After that, the same graph adapter can be tested inside a VLA such as SmolVLA without changing the core claim into a world-model project. RGB-derived graphs remain a separate perception experiment so perception and representation effects are not mixed into this baseline.

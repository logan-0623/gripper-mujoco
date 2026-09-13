# SmolVLA capability-acquisition and flow pilot

This directory records the exploratory evidence produced for §12 of the
predictive-interaction-state experiment design.

## Capability pilot

The smoke SFT100 lineage was evaluated at checkpoints `004108`, `008216`, and
`016435` on LIBERO Spatial task IDs 0–3. Each cell used 10 paired initial
conditions with seed `2057736129`, `init_states=true`, `n_action_steps=10`,
`num_steps=10`, `empty_cameras=1`, one synchronous environment, and recording
disabled.

| checkpoint | task 0 | task 1 | task 2 | task 3 | total |
| --- | ---: | ---: | ---: | ---: | ---: |
| 004108 | 5/10 | 2/10 | 4/10 | 0/10 | 11/40 (27.5%) |
| 008216 | 1/10 | 5/10 | 1/10 | 0/10 | 7/40 (17.5%) |
| 016435 | 4/10 | 4/10 | 6/10 | 0/10 | 14/40 (35.0%) |

For `008216 -> 016435`, paired outcomes contain 10 gains and 3 losses across
the four tasks (two-sided exact McNemar p=0.0923). Directions differ by task,
and task 3 is a 0/10 floor at every checkpoint. This is evidence of an
exploratory capability difference, not confirmation of monotonic capability
acquisition.

Only task success was recorded. The §12 E0 simulator outcomes for grasp
establishment, lift, hold duration, normal release, and unintended drop were
not collected. E0 therefore remains partial and unresolved.

Exact outcomes and SHA-256 hashes of the server-side `eval_info.json` files
are stored in `capability_pilot.json`. The source paths refer to the authorized
AutoDL server and are provenance records rather than portable local paths.

## Flow trace status

The implementation smoke recorded eight states, three shared epsilon draws,
all ten solver calls, middle/late expert activations, velocity, latent action,
and postprocessed action across the available smoke checkpoints. Paired
natural-trajectory differences were computed successfully. This validates the
small-batch E1 trace path; it does not complete the planned 512-state E1
measurement or the fixed-input-point query.

E2 candidate discovery, E3 observation/noise source tests, E4 causal and
closed-loop interventions, and E5 compression/OOD tests have not started.

## E0 confirmation run

The confirmation run completed on 2026-09-13 after a real-environment smoke.
It compares checkpoints `008216` and `016435` on task IDs 0–3 using frozen
initial-state IDs 10–49: 320 planned rollouts. It records contact, stable
grasp, lift, maximum lift, longest stable hold, release, normal placement,
unintended drop, and success. The physical-event smoke used initial-state ID
10 and produced a consistent contact → stable grasp → lift → normal release →
success trace. Server outputs are written under
`/root/autodl-tmp/gripper-mujoco-rollouts/e0_physical_confirmation_v1`.

The run completed with all eight cells and 320 rollouts. Every cell contains
40 unique initial-state IDs 10–49, and every event record agrees exactly with
the corresponding LeRobot success record. Raw JSON is archived under
`confirmation_raw/`; derived statistics and source hashes are in
`confirmation_report.json`.

| outcome | 008216 | 016435 | task-macro paired delta (95% bootstrap CI) |
| --- | ---: | ---: | ---: |
| Contact | 89/160 | 94/160 | +3.1 pp [-5.0, +10.6] |
| StableGrasp | 45/160 | 42/160 | -1.9 pp [-10.0, +5.6] |
| Lift | 36/160 | 41/160 | +3.1 pp [-5.6, +11.3] |
| Success | 28/160 | 32/160 | +2.5 pp [-5.0, +10.0] |

The global acquisition gate failed because every task-macro interval includes
zero and the directions differ sharply by task. Task 0 StableGrasp/Lift rose
from 2/40 to 15/40, while task 1 StableGrasp fell from 30/40 to 10/40 and
Lift from 22/40 to 9/40. Task 2 improved more moderately; task 3 never reached
contact. The supported phenotype is task-conditioned interaction
reorganization, not a general transition from unable to able.

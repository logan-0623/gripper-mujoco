# Recovery RL Protocol v2 for Interaction-Representation Plasticity

Status: approved by the user on 2026-08-21

## 1. Objective

This protocol replaces the unstable nominal-reset residual-PPO experiment with a
gated recovery-learning study. It asks whether an RL algorithm that is first shown
to improve a privileged residual controller can then reveal a difference between
frozen-policy adaptation and policy-representation adaptation.

The protocol preserves the scientific distinctions used by the project:

```text
C: correctness or decodability
U: closed-loop control utility
P: reward-driven adaptation plasticity
```

It does not assume that `C => U`, `C => P`, or `U => P`. Existing ACT results are
frozen. No additional steps are appended to the completed PPO runs, and their
artifacts are never overwritten.

## 2. Scope and artifact isolation

The existing root remains immutable evidence:

```text
outputs/representation_study/icra/
```

Every v2 configuration, checkpoint, State Bank, replay shard, and report is written
under:

```text
outputs/representation_study/icra_rl_v2/
```

The v2 work is split into three sequential projects:

1. stabilize the ACT recovery-RL protocol;
2. run the formal ACT mechanism study;
3. migrate only the validated protocol to SmolVLA.

The first two projects are covered by this specification. SmolVLA implementation is
blocked by the ACT gate in Section 13.

## 3. Recovery and perturbation distributions

### 3.1 Distribution families

Training resets sample one of three disjoint families:

| Family | Probability | Construction |
| --- | ---: | --- |
| recovery | 0.50 | expert prefix followed by a physical interaction intervention |
| perturbation | 0.30 | bounded phase-conditioned deviation from a reproducible expert prefix |
| nominal rollout | 0.20 | ordinary initial reset evaluated through the online environment |

Recovery contains `wrong_way_transport`, `premature_open`, and
`receptacle_misalignment`. `post_placement_reclose` is excluded from the main
recovery distribution because it occurs after task completion and does not measure
recovery to successful placement.

Perturbations are applied at `approach`, `grasp`, or `lift`. Each perturbation is
physically realized through the environment action interface. No object or robot
state is assigned directly after reset.

### 3.2 Split and calibration contract

Source seeds are split before variants are generated. Calibration, RL training,
curve evaluation, and final evaluation have disjoint source seeds. Variants from one
source seed cannot cross these partitions.

Perturbation severity is selected only on calibration seeds. The accepted version
must place frozen-SFT recovery success within `[0.30, 0.50]`. Once accepted, its
severity parameters, source-seed split, and case manifests are hashed and frozen.
Final evaluation is never used to tune severity or mixture weights.

Rejected physical interventions are logged by source seed, variant, kind, and
reason. A distribution version passes only if each declared family and kind meets a
predeclared minimum acceptance count.

## 4. Compact Oracle-State

Oracle-State is a 36-dimensional task-sufficient diagnostic input. It is not the
MuJoCo state and does not contain an expert action, reward, terminal outcome, or
future relation label.

| Group | Width | Definition |
| --- | ---: | --- |
| gripper-target geometry | 10 | relative translation in gripper frame, relative 6D rotation, Euclidean distance |
| target-receptacle geometry | 6 | relative translation in receptacle frame, distance, upright alignment, placement alignment |
| interaction state | 4 | gripper open fraction, bilateral target contact, stable target grasp, target support |
| distractor risk | 1 | normalized nearest distractor clearance |
| phase | 6 | one-hot approach, grasp, lift, transport, place, release |
| intervention family | 7 | one-hot nominal, approach offset, grasp offset, lift offset, wrong-way transport, premature open, receptacle misalignment |
| recovery scalars | 2 | normalized intervention severity and current progress relative to the intervention-start potential |

All continuous values are finite, clipped to declared physical ranges, and
normalized with constants stored in the case manifest. Geometry is expressed in
local frames so that the diagnostic state is invariant to the global scene frame.

The same Oracle-State schema is used by the Oracle actor and every training-only
critic. This keeps algorithm comparisons aligned while preventing critic gradients
from contaminating the visual policy representation.

## 5. Residual control and reward

Every learned controller uses the existing bounded residual interface:

```text
a_t = clip(a_SFT,t + alpha * delta_t)
```

The seven-dimensional `alpha` vector is identical across algorithms and actor
conditions. Reports retain the unclipped sum, executed local action, residual norm,
clipping event, IK projection, and termination reason.

The shared task reward is:

```text
r_t = r_terminal
      + 0.10 * (gamma * Phi(s_{t+1}) - Phi(s_t))
      - 0.01 * ||alpha * delta_t||_2^2
```

`r_terminal` is `+1` for task success, `-1` for drop or wrong-object termination,
and `0` for timeout or a nonterminal transition. `Phi` is the phase-appropriate
task potential: target-to-gripper progress before stable grasp and
target-to-receptacle progress after stable grasp. Potential inputs come from the
compact privileged state and are not exposed to visual actors.

The terminal reward, progress coefficient, and residual coefficient are identical
for PPO and SAC. They are not retuned after the algorithm screen.

## 6. Actor, critic, and gradient isolation

All critics have independent parameters and consume Compact Oracle-State.

```text
Compact Oracle-State -> critic encoder -> V(s) or twin Q(s, delta)
```

Actors differ only in their observation and declared trainable policy group:

| Actor | Actor input | ACT policy encoder |
| --- | --- | --- |
| Oracle-State | Compact Oracle-State | not used |
| RL-head | frozen ACT action-proximal latent | frozen |
| RL-representation | ACT action-proximal latent | registered late-fusion group trainable |

For PPO, value loss updates only the independent value encoder and value head. For
SAC, Q losses update only the independent Q encoders and Q heads. The ACT encoder
can receive gradients only from the actor objective and the explicit anchoring
terms below. Tests must verify zero critic-loss gradient on all actor and ACT-policy
parameters.

## 7. Behavior anchoring

Each actor update includes a fixed nominal replay batch drawn from a provenance-bound
nominal bank. The nominal batch size equals the online actor minibatch size, capped
at 64.

```text
L_actor_total = L_actor_RL
              + 1.0 * mean(||delta_nominal||_2^2)
              + 0.10 * mean(||z_nominal - stopgrad(z_SFT)||_2^2)
```

The residual task reward already supplies the `0.01 * ||alpha * delta||^2` online
penalty. The latent anchor applies only to RL-representation. The frozen SFT latent
target is cached with dataset, checkpoint, tap, and normalization hashes.

The anchoring development ablation contains:

| Variant | Online residual penalty | Nominal zero-residual anchor | Latent anchor |
| --- | ---: | ---: | ---: |
| no anchor | no | no | no |
| residual only | yes | no | no |
| full anchoring | yes | yes | RL-representation only |

The formal variant is the simplest one whose nominal loss is at most 10 percentage
points and whose recovery AUC is within 0.02 of the best development variant. If no
variant satisfies both conditions, the formal ACT comparison is blocked.

## 8. PPO-SAC algorithm screen

The only off-policy candidate is Soft Actor-Critic. PPO and SAC are screened using
the Oracle-State actor so that perception does not confound algorithm stability.

| Condition | Steps | Development seeds | Evaluation interval |
| --- | ---: | ---: | ---: |
| Oracle-PPO | 8,192 | 2 | 1,024 |
| Oracle-SAC | 8,192 | 2 | 1,024 |

Both algorithms use identical distribution manifests, reward, action scale,
nominal replay, and paired evaluation cases. The screen is reported as backend
selection evidence, not as the paper's main algorithm comparison.

Selection is lexicographic:

1. reject a backend with non-finite loss/action, corrupted resume, or repeated
   simulator-integrity failure;
2. reject a backend whose median nominal success drops by more than 10 percentage
   points from SFT;
3. choose the remaining backend with greater median recovery normalized AUC;
4. if the AUC difference is below 0.02, choose the backend with lower across-seed
   AUC variance;
5. if still tied, choose PPO because its implementation has fewer moving parts in
   the existing pipeline.

## 9. Oracle interface gate

The selected Oracle-State backend must, on both development seeds:

```text
final recovery success - SFT recovery success >= 0.10
final nominal success  - SFT nominal success  >= -0.10
```

It must also improve median recovery AUC over the constant SFT curve. Failure blocks
RL-head and RL-representation. The correct response to failure is to inspect reset,
reward, and action-interface traces, not to add environment steps.

## 10. Formal ACT comparison

After the distribution, algorithm, Oracle, and anchoring gates pass, the formal ACT
matrix is:

| Condition | Role |
| --- | --- |
| SFT | frozen behavior baseline |
| continued SFT | additional supervised-optimization control |
| Oracle-State RL | reward and residual-interface diagnostic ceiling |
| RL-head | online adaptation with frozen ACT representation |
| RL-representation | online adaptation with policy representation plasticity |

The selected RL backend uses 20,480 environment steps for each RL condition. The
completed v1 PPO checkpoints are not resumed. Formal development begins with three
training seeds. A key RL-representation-minus-RL-head direction must agree in at
least two of three seeds before it is expanded to five seeds.

## 11. Time-resolved checkpoints and State Bank v2

### 11.1 Checkpoints

Exactly resumable snapshots are saved at:

```text
0, 4,096, 8,192, 12,288, 16,384, 20,480 environment steps
```

Each snapshot binds actor, critic, target critic where applicable, optimizers,
temperature state, RNG states, environment sampler state, distribution manifest,
normalization, and replay-shard manifest. A snapshot is immutable after its
completion marker is written.

### 11.2 State Bank v2

State Bank v2 contains 1,200 nonterminal states:

```text
400 nominal + 400 perturbation + 400 recovery
```

It is split by source seed before frame selection. Every test recovery kind has
multiple independent episode groups. Terminal frames are excluded from primary
probes.

Primary probe factors are geometry, phase, recovery state/type, and next relation.
Entity is removed from the primary set because the current task makes its entity
mask nearly constant. Contact and stable grasp remain secondary because the current
probes are saturated.

Linear probes are run at all six checkpoints. The fixed shallow MLP is run only at
steps 0 and 20,480 as a capacity sensitivity check.

## 12. Evaluation and statistics

At every checkpoint, learning curves use 30 fixed nominal and 30 fixed recovery
development cases per training seed. Final held-out evaluation uses 50 nominal and
50 recovery cases per seed. Conditions share paired case schedules.

Primary metrics are:

- recovery success curve and normalized AUC;
- nominal retention curve;
- final recovery success delta from SFT;
- final nominal success delta from SFT;
- time-resolved primary linear-probe metrics.

Secondary metrics are timeout, drop, wrong object, residual norm, clipping, action
smoothness, and IK projection. Clipping is an action-saturation diagnostic, not a
safety metric.

The independent behavioral unit is an evaluation episode. Final uncertainty uses
paired bootstrap with policy-seed clustering. Probe reports preserve source-episode
grouping. Correlations between `delta probe(t)` and `delta recovery success(t)` are
descriptive unless they are supported across at least three training seeds and
multiple time checkpoints. No frame is treated as an independent policy outcome.

The main evidence panels are:

1. recovery success versus environment steps;
2. nominal success versus environment steps;
3. primary probe change versus recovery-success change;
4. nominal-retention versus recovery-improvement Pareto plot.

## 13. SmolVLA gate

SmolVLA begins only if all of the following hold:

- the Oracle interface gate passes;
- one RL backend is stable across ACT seeds;
- RL-representation versus RL-head has a consistent direction in at least two of
  three ACT seeds;
- final nominal forgetting is no worse than 10 percentage points;
- at least one primary factor has a time-resolved probe trajectory that can be
  compared with recovery improvement.

SmolVLA then runs only SFT, RL-head, and RL-representation under the selected ACT
protocol. It does not repeat PPO-SAC selection or the full anchoring ablation.

## 14. Implementation boundaries

The implementation is divided into independent modules:

```text
rl/distributions.py    recovery/perturbation case generation and split binding
rl/oracle_state.py     36D codec, normalization, and invariance validation
rl/rewards.py          terminal, potential, and residual reward terms
rl/actors.py           Oracle, frozen-latent, and representation actors
rl/critics.py          independent PPO value and SAC twin-Q encoders
rl/ppo.py              on-policy backend
rl/sac.py              replay-based off-policy backend
rl/anchoring.py        nominal replay and latent drift losses
rl/snapshots.py        immutable time checkpoints and exact resume
rl/evaluation.py       paired nominal/recovery curves and final evaluation
```

The CLI exposes separate commands for distribution calibration, algorithm screen,
Oracle gate, anchoring screen, formal training, time-resolved measurement, and
report aggregation. A later command cannot run unless the prior gate artifact is
present and compatible.

## 15. Error handling and overwrite policy

- Existing v1 output paths are rejected as v2 destinations.
- Completed outputs are never silently overwritten; a new distribution or protocol
  version requires a new output root.
- Resume rejects different config, parent checkpoint, distribution, reward,
  normalization, encoder, algorithm, replay, or case hashes.
- Non-finite simulator state, Oracle-State, reward, action, loss, or gradient stops
  the affected run and writes a structured failure report.
- Intervention preparation rejection advances to a declared replacement case; it
  cannot silently change family proportions.
- Gate reports distinguish execution success from scientific gate success.

## 16. Verification

Unit tests must cover:

- Oracle-State width, finiteness, normalization, and global-frame invariance;
- recovery/perturbation split isolation and deterministic reconstruction;
- reward decomposition and terminal signs;
- residual bounds and penalties;
- zero actor/ACT gradients from critic-only losses;
- nominal replay and latent-anchor behavior;
- one finite PPO update and one finite SAC update;
- replay sampling and exact checkpoint resume;
- immutable checkpoint and v1-output protection;
- metric aggregation and gate decisions.

Integration tests retain only unique critical paths:

- one accepted reset per distribution family;
- one resumable Oracle-PPO screen update;
- one resumable Oracle-SAC screen update;
- one RL-head and one RL-representation update with critic-gradient isolation;
- one time-checkpoint load followed by paired nominal/recovery evaluation.

Smoke runs are implementation checks only and are excluded from scientific tables.

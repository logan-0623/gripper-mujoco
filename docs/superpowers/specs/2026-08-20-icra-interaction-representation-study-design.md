# ICRA Interaction Representation Study Design

Status: approved by the project plan and user direction on 2026-08-20

## 1. Research thesis

The project studies whether reward-driven adaptation selectively reorganizes
action-relevant interaction representations that supervised imitation does not make
equally accessible or usable.

Interaction Graph labels are a measurement ontology, privileged supervision source,
and intervention vocabulary. They are not required policy inputs in the main study.
The scientific objects are kept distinct:

```text
C: representation correctness / decodability
U: behavioral control utility
P: online adaptation plasticity
```

The experiments test whether `C => U`, `C => P`, or `U => P`; none of these
implications is assumed in advance. The implementation also distinguishes four
levels of evidence:

```text
Encoded -> Accessible -> Used -> Useful
```

- **Encoded:** information can be decoded from a frozen latent.
- **Accessible:** a predeclared lightweight probe can recover it on held-out states.
- **Used:** controlled perturbations of that factor measurably change the action.
- **Useful:** the intervention changes closed-loop outcomes.

Probe accuracy alone never supports a causal claim about policy use.

## 2. Experimental roles

### 2.1 ACT: controlled mechanism study

The current ACT pipeline and all completed Graph-input results remain supported.
They provide inexpensive, reproducible control over architecture, trainable groups,
state coverage, and interventions. Existing artifacts are classified as
`legacy_graph_input_study` and retain their original report schemas and paths.

New ACT checkpoints follow these stages:

1. `pretrained`: image encoder initialization before task demonstrations;
2. `sft`: behavior cloning on the nominal demonstration dataset;
3. `continued_sft`: additional supervised updates matched to the RL update budget;
4. `rl_head`: residual RL with the representation encoder frozen;
5. `rl_representation`: residual RL with predeclared late representation layers
   trainable.

The continued-SFT control separates reward-driven changes from additional optimizer
steps and state exposure.

### 2.2 SmolVLA: modern VLA validation

SmolVLA is the main modern-policy validation. Its true observation contract is:

```text
agent RGB + wrist RGB + 6D end-effector state + language -> continuous actions
```

The study uses the same stage names, state bank, ontology, probe protocol, outcome
metrics, and intervention semantics as ACT. Backend-specific tap names may differ,
but their scientific roles are fixed before results are inspected.

### 2.3 pi0: optional backend

pi0 receives the same backend and stage manifest contract. It is not required for
the first ICRA evidence gate and is activated only after ACT and SmolVLA establish a
coherent result. This avoids letting the most resource-intensive backend block the
core study.

## 3. Predeclared questions and hypotheses

- **RQ1:** Which interaction factors are already accessible in pretrained features?
- **RQ2:** Which nominal-task factors become more accessible after SFT?
- **RQ3:** Does RL selectively change contact, phase, next-relation, and recovery
  representations on policy-visited failure states?
- **RQ4:** Which representation changes predict closed-loop improvement and RL
  learning efficiency?

The primary hypotheses are:

- **H1:** SFT primarily improves nominal entity, geometry, and phase readout.
- **H2:** Representation-adapting RL changes recovery and next-relation readout more
  than head-only RL or continued SFT.
- **H3:** Changes in recovery/contact/transition measurements predict normalized RL
  learning-curve AUC better than generic entity measurements.
- **H4:** Continued SFT does not reproduce every reward-driven representation change.

Alternative outcomes remain valid findings: RL may reorganize an existing code,
make it easier to read out, or improve behavior without a measurable latent change.

## 4. Fixed State Bank

Every stage is measured on identical observation states. A state-bank record contains:

```text
state_id, source_episode_id, task_id, condition, phase, seed, provenance
agent_rgb, wrist_rgb, end_effector_state, language
privileged_simulator_state, ontology_labels
```

The bank has four strata:

- `nominal`: states supported by expert demonstrations;
- `perturbation`: controlled deviations around demonstrated trajectories;
- `recovery`: off-nominal states that require corrective interaction;
- `terminal`: success and failure endpoints used only for descriptive checks.

Primary representation analyses use nominal, perturbation, and recovery states.
Results are reported separately for expert-support and policy-shift strata. No
temporal split may place frames from the same episode in both probe train and test.

## 5. Shared backend and tap contract

Every policy family implements:

```text
load_stage(manifest)
encode(batch)
act(batch)
get_latents(batch, taps)
set_trainable_groups(groups)
```

A stage manifest binds backend, stage, checkpoint, config, dataset, state bank,
trainable groups, latent taps, and source revision by hashes. Loading rejects a
stage whose declared backend, taps, or provenance are incompatible.

Predeclared scientific tap roles are:

| Role | ACT | SmolVLA | pi0 |
|---|---|---|---|
| visual | vision backbone | vision output | vision output |
| fused | temporal fusion | multimodal fusion | multimodal fusion |
| policy input | decoder input | action-expert input | action-expert input |
| action proximal | pre-action feature | pre-action feature | pre-action feature |

Concrete module paths are backend version-specific and stored in the manifest. A tap
cannot be added after test results are seen without creating a new registered study.

## 6. Measurements

### 6.1 Accessible information

Frozen latent probes predict:

- entity/target identity;
- relative geometry and alignment;
- contact and stable grasp;
- interaction phase;
- next relation to establish;
- recovery need and recovery type.

Linear probes are primary; a fixed shallow MLP is secondary. Reports include held-out
metrics, calibration where relevant, sample counts, episode grouping, seed variance,
and confidence intervals. Terms such as “decodable” or “recoverable” are used instead
of claiming the policy uses the information.

### 6.2 Policy use

Offline action sensitivity reports raw translation, rotation, gripper, and full-action
changes as primary measurements. Scale-normalized action changes use training action
IQR with a declared floor; they are never divided by a nearly zero token perturbation.

Intervention controls include:

- zero;
- training mean;
- episode-constant random;
- temporally matched random;
- delayed Teacher;
- ontology-specific masking or replacement.

At minimum, zero and temporally matched random controls must pass before a semantic
mechanism claim is allowed. Instruction-only interventions hold image and robot state
fixed while changing language.

### 6.3 Behavioral usefulness and plasticity

Closed-loop metrics include success, timeout, drop, wrong-object grasp, action
clipping, recovery rate, and steps. Residual RL uses

```text
a_t = a_SFT,t + alpha * delta_a_RL,t
```

with matched `rl_head` and `rl_representation` variants. Sparse task success is the
primary reward; minimal progress shaping is secondary and explicitly reported.

Plasticity is measured primarily by normalized learning-curve AUC. Final improvement
and environment steps to a fixed threshold are secondary. No composite plasticity
score is introduced without an independent validation study.

## 7. Statistical contract

- Evaluation episodes, not frames, are the independent outcome units.
- Paired case schedules are used across policy stages and intervention controls.
- Uncertainty uses episode bootstrap with policy-seed clustering.
- Main comparisons report effect sizes and 95% confidence intervals.
- Failure-conditioned analysis tests only three to five predeclared hypotheses,
  excludes terminal frames, and controls the family with Benjamini-Hochberg FDR.
- Development uses 20 episodes per seed-condition; main evaluation uses at least 50,
  preferably 100. Exploratory training uses three seeds; only the key final subset is
  expanded to five seeds.

## 8. Evidence gates

### Gate A: measurement contract

- sensitivity v2 cannot produce explosive near-zero denominators;
- stage manifests and tap registries reject incompatible inputs;
- legacy reports remain readable and unchanged;
- unit tests cover schemas, scale handling, and provenance validation.

### Gate B: ACT Go/No-Go

- fixed State Bank passes leakage and provenance checks;
- probes beat label-matched controls on at least one interaction-critical factor;
- offline sensitivity and at least one closed-loop intervention agree in direction;
- ACT stage comparisons are reproducible across three training seeds.

### Gate C: SmolVLA validation

- ACT yields a predeclared, falsifiable stagewise hypothesis;
- SmolVLA backend exposes fixed taps and identical ontology labels;
- two or three tasks cover nominal pick-place, language-conditioned distractors, and
  recovery-heavy states.

pi0 begins only after Gate C evidence is coherent and resources remain.

## 9. Artifact layout

```text
interaction_vla/representation_study/
  backends/
  taps/
  ontology/
  state_bank/
  probes/
  interventions/
  outcomes/
  statistics/
  rl/
  reports/
  schemas/

configs/representation_study/
outputs/representation_study/
```

The implementation now includes the stage/backend contract, sensitivity v3, the
populated Fixed State Bank, frozen probes, paired interventions, resumable SFT,
residual PPO, ACT/SmolVLA/pi0 adapters, fixed-case evaluation, and machine-readable
study reports. Formal multi-seed training and rollout remain experimental evidence,
not implementation artifacts.

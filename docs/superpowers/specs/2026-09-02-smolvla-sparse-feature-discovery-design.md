# SmolVLA Sparse Feature Discovery

## Question

Can the successful official SmolVLA checkpoint yield sparse, reproducible,
cross-task features at `action_expert_input`, and do any label-blind features
affect actions more than norm-matched random directions?

This is a diagnostic kill test. SAE training, feature interpretability, and
latent steering are not standalone novelty claims.

## Frozen inputs

- Official SmolVLA checkpoint bound by Protocol-v4.
- Protocol-v4 baseline: 9/10 success on `libero_spatial` task 0.
- Shared 13,603-state LIBERO State Bank and its episode-group split.
- Existing pooled `action_expert_input` latent, shape `[13603, 720]`.
- Existing deterministic policy-noise and action-intervention runtime.

No SFT, RL, latent regeneration, new ontology, or closed-loop intervention is
allowed in this stage.

## Discovery method

Train three deterministic Top-K sparse autoencoders on episode-group training
states only:

```text
dictionary width: 1440
active features per state: 32
updates: 5000
batch size: 512
seeds: project seed + {0, 1, 2}
```

Standardization statistics are fitted on training states. Models are saved as
safe NumPy artifacts. Validation states match seed-0 features one-to-one across
seeds using decoder cosine similarity; held-out activation correlation is then
reported. PCA with 32 components is a reconstruction reference, not a sparse
feature baseline.

## Label-blind candidate selection

Rank seed-0 features using only:

- alive activation rate;
- cross-seed decoder cosine and activation correlation;
- task and episode coverage;
- activation-mass entropy.

Freeze the top eight broad, stable features before reading privileged labels or
actions. Contact, StableGrasp, and Phase cannot influence candidate selection.

## Diagnostics

For every selected feature report:

- reconstruction error and explained variance;
- sparsity and dead-feature rate;
- cross-seed reproducibility;
- task/episode coverage and concentration;
- adjacent temporal variation versus within-episode shuffled variation;
- post-selection association with Contact, StableGrasp, and Phase.

Concentration is a memorization warning, not proof of memorization.

## Action intervention

Ablate one selected feature by subtracting its per-state decoder contribution
from the live final-denoising `action_expert_input`. Compare it with an
orthogonal, same-rank, same-norm random perturbation on the same state.

Primary quantity:

```text
U_j = first-action L2(feature ablation) - first-action L2(matched random)
```

Also report translation, rotation, gripper, full-chunk effects, episode-cluster
confidence intervals, and Benjamini-Hochberg correction across eight frozen
candidates.

## Gates

1. Integrity: input hashes, splits, shapes, finite values, and deterministic
   model bindings pass.
2. Discovery: at least four label-blind features meet the preregistered
   stability and broad-coverage thresholds.
3. Causality: at least one frozen candidate has positive episode-cluster CI and
   adjusted `q <= 0.05` for `U_j`.

If discovery fails, stop. If discovery passes but causality fails, stop without
training a trajectory. Only a causal-feature pass authorizes a later,
separately designed longitudinal experiment.

## Artifacts

Runtime artifacts live under
`outputs/representation_study/libero_smolvla/protocol_v5/sparse_features/`.
Git stores only compact reports, candidate/fold manifests, configuration, and
hashes. Models, latents, actions, videos, and NPZ arrays remain outside Git.

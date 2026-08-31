# SmolVLA StableGrasp Longitudinal Recruitment Design

## Question

When does StableGrasp become accessible at SmolVLA's action-expert input, and when does the policy begin to rely on the same factor for action? The object of study is the four-point trajectory `RawDrift(t), Accessibility(t), FunctionalRecruitment(t)`, not generic probing or steering.

## Frozen inputs

Use the existing Protocol-v3 State Bank, folds, cross-fit report, latent caches, and the four conditions `pretrained`, `d25_u16070`, `d100_u16617`, and `d100_u66470`. Accessibility is read from the frozen report. No checkpoint, label, fold, or probe protocol is regenerated.

## Fold-held-out subspace

For each episode-group fold and checkpoint, reconstruct the three archived StableGrasp linear probes with the exact recorded seeds and solver. Require predictions and scores to match the immutable cell artifact. Convert each binary decision vector to raw latent coordinates, normalize and sign-align it, and take the first right singular vector as a rank-one consensus. Evaluation states only use their held-out fold's consensus.

## Intervention

Hook `model.action_time_mlp_out` at its final denoising call. Its tensor is `[B, 50, 720]`; the factor delta is computed in the same 720-dimensional chunk-mean space and broadcast to all 50 tokens. The targeted destructive intervention moves StableGrasp evidence toward the decision boundary with a train-fold natural-support cap. Matched random is rank one, orthogonal, deterministic, uses the same token positions, and has the exact same per-state L2 norm. Matched mean and zero are secondary controls.

## Gates

Specificity precedes policy inference. StableGrasp disruption must exceed matched-random disruption and must not be explained by equal or larger Phase/Contact disruption. Because StableGrasp is phase-coupled, a place-only diagnostic is mandatory. A failed specificity gate stops action and closed-loop claims.

After specificity passes, action sensitivity compares original, targeted, and matched-random actions under identical deterministic denoising noise. First action is primary; translation, rotation, gripper delta, and gripper flips are episode-cluster bootstrapped with task-cluster robustness. Closed loop and RL remain disabled.

## Kill condition

If accessibility and functional recruitment simply co-emerge and plateau, or if the intervention only recovers a Phase direction, record the result and reassess novelty before adding factors. Contact is the only planned replication after a successful StableGrasp result.

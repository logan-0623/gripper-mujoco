# StableGrasp Longitudinal Recruitment Implementation Plan

**Goal:** Implement the smallest leakage-safe offline test of StableGrasp functional recruitment over four frozen SmolVLA checkpoints.

**Architecture:** Reuse Protocol-v3 caches, folds, probe solver, State Bank collation, deterministic inference noise, and SmolVLA loading. Add one recruitment module, one CLI route, and focused tests. Do not alter frozen artifacts.

### Task 1: Bind the frozen evidence

- Add tests for the exact four checkpoint IDs, episode-group StableGrasp cell mapping, exact categorical reconstruction, and tolerance-bounded continuous reconstruction.
- Implement a read-only audit/plan report below `protocol_v3/recruitment/stable_grasp/`.

### Task 2: Build the fold-specific intervention basis

- Add tests for standardized-to-raw binary directions, seed consensus, orthogonal same-norm random controls, and final-call token broadcasting.
- Reconstruct archived probes and require held-out prediction/score equality.
- Produce specificity diagnostics for StableGrasp, Phase, Contact, Geometry, and `place`-only states.

### Task 3: Gate policy action sensitivity

- Add a test that a failed specificity report prevents policy loading.
- Reuse deterministic noise and State Bank collation.
- Hook only the final `action_time_mlp_out` call, run original/target/random, and report first-action plus chunk effects with episode/task clustered intervals.

### Task 4: Verify and hand off

- Run focused recruitment/CLI/config tests, then the LIBERO representation suite.
- Run the read-only audit command locally against archived metadata.
- Document server commands. Stop before closed-loop and RL.

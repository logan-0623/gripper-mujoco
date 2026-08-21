# ICRA Representation Study Phase 2 Implementation Plan

> **For Codex:** Execute test-first. Existing sensitivity v1/v2 artifacts and legacy
> ACT checkpoints are immutable inputs.

**Goal:** Add distribution-aware sensitivity controls and a fixed, leakage-safe State
Bank contract shared by ACT and modern VLA backends.

**Evidence semantics:** Zero and temporally matched random tokens are negative
controls. They do not identify a semantic mechanism by themselves. A factor-specific
intervention is interpretable only when it differs from these controls and later
changes a paired closed-loop outcome.

---

### Task 1: Whole-Graph sensitivity controls

**Files:**

- Modify: `interaction_vla/graph_control/sensitivity.py`
- Modify: `interaction_vla/graph_control/pipeline.py`
- Modify: `tests/interaction_vla/graph_control/test_sensitivity.py`
- Modify: `tests/interaction_vla/graph_control/test_pipeline.py`

- [x] Add a deterministic episode-deranged, normalized-time-matched token transform.
- [x] Add whole-Graph `zero` and `temporally_matched_random` records under the
  `all_tokens` group.
- [x] Use the same episode permutation across representation conditions for a policy
  seed.
- [x] Preserve raw action metrics and v2 action-IQR normalization.
- [x] Record the control seed and semantics in report provenance.

### Task 2: Fixed State Bank schema

**Files:**

- Create: `interaction_vla/representation_study/state_bank/__init__.py`
- Create: `interaction_vla/representation_study/state_bank/schema.py`
- Create: `interaction_vla/representation_study/state_bank/validation.py`
- Create: `tests/interaction_vla/representation_study/test_state_bank.py`

- [x] Define immutable observation references instead of copying RGB into JSON.
- [x] Represent nominal, perturbation, recovery, and terminal strata explicitly.
- [x] Distinguish expert-support from policy-shift states.
- [x] Bind the bank to dataset, ontology, source, and selection-config hashes.
- [x] Reject duplicate state IDs, duplicate dataset rows, non-finite robot state, and
  source-episode leakage across train/validation/test.
- [x] Emit a deterministic canonical manifest payload suitable for hashing.

### Task 3: Configuration and evidence handoff

**Files:**

- Modify: `configs/representation_study/base.yaml`
- Modify: `README.md`
- Modify: `ccfa.yaml`

- [x] Record the completed sensitivity-v2 findings as descriptive evidence.
- [x] Document that `8640` counterfactual records derive from 20 selected states, not
  8640 independent observations.
- [x] Keep the ACT mechanism gate closed until State Bank creation and probe/control
  checks pass.

### Task 4: Verification

- [x] Run focused sensitivity and State Bank tests.
- [x] Run graph-control pipeline/CLI regressions.
- [x] Run the complete `tests/interaction_vla` suite.
- [x] Validate YAML, compile new modules, and run `git diff --check`.

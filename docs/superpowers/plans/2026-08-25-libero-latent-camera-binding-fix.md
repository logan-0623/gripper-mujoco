# LIBERO SmolVLA Latent Camera-Binding Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make pretrained and SFT SmolVLA latent extraction consume LIBERO images through the exact camera-key contract used during SFT, while invalidating every latent row produced by the old loader.

**Architecture:** Put the LIBERO-to-SmolVLA image rename map in one small contract module shared by training and extraction. Extend the LeRobot backend's dataset-bound loader to preserve checkpoint input features and forward the same map to both policy construction and preprocessing. Bind latent caches to all source files that control this runtime path, then verify one real State Bank batch on the server before any full extraction.

**Tech Stack:** Python 3.12, PyTorch, Hugging Face LeRobot/SmolVLA, pytest, SHA-256 artifact bindings.

---

## File structure

- Create `interaction_vla/representation_study/libero/feature_binding.py`: single source of truth for the LIBERO image rename contract.
- Modify `interaction_vla/representation_study/libero/training.py`: import the shared contract while keeping the existing training CLI unchanged.
- Modify `interaction_vla/representation_study/backends/lerobot.py`: preserve checkpoint input features and apply a caller-supplied rename map consistently.
- Modify `interaction_vla/representation_study/libero/latents.py`: pass the shared contract and expand the latent implementation identity.
- Modify `tests/interaction_vla/representation_study/test_lerobot_backends.py`: dataset-bound loader regression tests.
- Modify `tests/interaction_vla/representation_study/libero/test_latents.py`: shared-contract and cache-identity regression tests.

### Task 1: Lock the camera-binding contract with failing tests

**Files:**
- Modify: `tests/interaction_vla/representation_study/test_lerobot_backends.py`
- Modify: `tests/interaction_vla/representation_study/libero/test_latents.py`

- [ ] **Step 1: Add a backend regression test using lightweight LeRobot fakes**

Add a test that creates a checkpoint config whose visual inputs are `camera1`, `camera2`, and `camera3`; monkeypatches `PreTrainedConfig.from_pretrained`, `LeRobotDatasetMetadata`, `make_policy`, and `make_pre_post_processors`; and calls:

```python
backend.load_checkpoint_for_dataset(
    checkpoint,
    repo_id="lerobot/libero",
    dataset_root=tmp_path / "dataset",
    rename_map={
        "observation.images.image": "observation.images.camera1",
        "observation.images.image2": "observation.images.camera2",
    },
)
```

Assert all of the following:

```python
assert set(policy_config.input_features) == {
    "observation.state",
    "observation.images.camera1",
    "observation.images.camera2",
    "observation.images.camera3",
}
assert captured_make_policy["rename_map"] == rename_map
assert captured_preprocessor_overrides["rename_observations_processor"] == {
    "rename_map": rename_map
}
assert captured_preprocessor_overrides["normalizer_processor"]["features"] == {
    **policy.config.input_features,
    **policy.config.output_features,
}
```

- [ ] **Step 2: Add latent binding tests**

Add a test for a new `_latent_implementation_sha256()` helper. Monkeypatch `_file_sha256` to return deterministic per-file digests, change only the digest returned for `backends/lerobot.py`, and assert that the implementation identity changes. Also assert the source list contains `latents.py`, `taps.py`, `feature_binding.py`, and `backends/lerobot.py`.

Add a contract test:

```python
assert LIBERO_SMOLVLA_RENAME_MAP == {
    "observation.images.image": "observation.images.camera1",
    "observation.images.image2": "observation.images.camera2",
}
```

- [ ] **Step 3: Run the focused tests and observe the intended failures**

Run:

```bash
.venv-lerobot/bin/python -m pytest \
  tests/interaction_vla/representation_study/test_lerobot_backends.py \
  tests/interaction_vla/representation_study/libero/test_latents.py -q
```

Expected: the new tests fail because `feature_binding.py`, the `rename_map` loader argument, and `_latent_implementation_sha256()` do not yet exist.

- [ ] **Step 4: Commit the red tests**

```bash
git add \
  tests/interaction_vla/representation_study/test_lerobot_backends.py \
  tests/interaction_vla/representation_study/libero/test_latents.py
git commit -m "test: cover SmolVLA latent camera binding"
```

### Task 2: Share the LIBERO feature-binding contract

**Files:**
- Create: `interaction_vla/representation_study/libero/feature_binding.py`
- Modify: `interaction_vla/representation_study/libero/training.py`

- [ ] **Step 1: Create the shared contract module**

```python
from __future__ import annotations


LIBERO_SMOLVLA_RENAME_MAP: dict[str, str] = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.image2": "observation.images.camera2",
}
```

- [ ] **Step 2: Replace the duplicated training constant with an import**

In `training.py`, remove the local dictionary and add:

```python
from .feature_binding import LIBERO_SMOLVLA_RENAME_MAP
```

Keep `build_stage_training_command()` serialization unchanged so existing stage manifests and completed checkpoints are not rewritten.

- [ ] **Step 3: Run training-command contract tests**

Run:

```bash
.venv-lerobot/bin/python -m pytest \
  tests/interaction_vla/representation_study/libero/test_stages.py \
  tests/interaction_vla/representation_study/libero/test_latents.py -q
```

Expected: the rename-map command test passes; backend/hash tests may remain red until Tasks 3 and 4.

- [ ] **Step 4: Commit the shared contract**

```bash
git add \
  interaction_vla/representation_study/libero/feature_binding.py \
  interaction_vla/representation_study/libero/training.py
git commit -m "refactor: share LIBERO SmolVLA feature binding"
```

### Task 3: Correct dataset-bound checkpoint loading

**Files:**
- Modify: `interaction_vla/representation_study/backends/lerobot.py`
- Test: `tests/interaction_vla/representation_study/test_lerobot_backends.py`

- [ ] **Step 1: Extend the loader signature without changing ACT behavior**

Change the signature to:

```python
def load_checkpoint_for_dataset(
    self,
    checkpoint: str | Path,
    *,
    repo_id: str,
    dataset_root: str | Path,
    rename_map: Mapping[str, str] | None = None,
) -> None:
```

Normalize once after the ACT/residual early return:

```python
feature_rename_map = dict(rename_map or {})
```

- [ ] **Step 2: Preserve the checkpoint input contract and forward the mapping**

Remove both assignments that clear `policy_config.input_features` and `policy_config.output_features`. Construct the policy with:

```python
policy = make_policy(
    policy_config,
    ds_meta=metadata,
    rename_map=feature_rename_map,
)
```

LeRobot's `make_policy` will retain non-empty checkpoint inputs and replace output features with the dataset action schema.

- [ ] **Step 3: Apply the identical rename map to preprocessing**

Add this entry to `preprocessor_overrides`:

```python
"rename_observations_processor": {"rename_map": feature_rename_map},
```

Keep the existing device, normalization, and unnormalization overrides so the loaded checkpoint uses formal LIBERO dataset statistics and action dimensions.

- [ ] **Step 4: Run the backend regression suite**

Run:

```bash
.venv-lerobot/bin/python -m pytest \
  tests/interaction_vla/representation_study/test_lerobot_backends.py -q
```

Expected: all tests pass, including preservation of `camera1/camera2/camera3` and consistent rename forwarding.

- [ ] **Step 5: Commit the loader fix**

```bash
git add \
  interaction_vla/representation_study/backends/lerobot.py \
  tests/interaction_vla/representation_study/test_lerobot_backends.py
git commit -m "fix: preserve SmolVLA camera binding for latents"
```

### Task 4: Bind extraction and caches to the corrected runtime

**Files:**
- Modify: `interaction_vla/representation_study/libero/latents.py`
- Test: `tests/interaction_vla/representation_study/libero/test_latents.py`

- [ ] **Step 1: Pass the formal rename contract during extraction**

Import the shared constant and change the loader call to:

```python
backend.load_checkpoint_for_dataset(
    checkpoint,
    repo_id=config.sources.lerobot_repo_id,
    dataset_root=dataset.root,
    rename_map=LIBERO_SMOLVLA_RENAME_MAP,
)
```

- [ ] **Step 2: Add an explicit implementation identity helper**

Define:

```python
def _latent_implementation_source_paths() -> tuple[Path, ...]:
    return (
        Path(__file__),
        Path(__file__).with_name("taps.py"),
        Path(__file__).with_name("feature_binding.py"),
        Path(__file__).parents[1] / "backends" / "lerobot.py",
    )


def _latent_implementation_sha256() -> str:
    digest = hashlib.sha256()
    for source in _latent_implementation_source_paths():
        digest.update(source.as_posix().encode("utf-8"))
        digest.update(_file_sha256(source).encode("ascii"))
    return digest.hexdigest()
```

Replace the old two-file inline hash with:

```python
implementation_hash = _latent_implementation_sha256()
```

- [ ] **Step 3: Run focused latent tests**

Run:

```bash
.venv-lerobot/bin/python -m pytest \
  tests/interaction_vla/representation_study/libero/test_latents.py -q
```

Expected: all tests pass and changing only the backend loader source changes the latent cache binding.

- [ ] **Step 4: Commit extraction and binding changes**

```bash
git add \
  interaction_vla/representation_study/libero/latents.py \
  tests/interaction_vla/representation_study/libero/test_latents.py
git commit -m "fix: bind LIBERO latents to camera adapter"
```

### Task 5: Verify locally and prepare a non-destructive server rerun

**Files:**
- Modify only if needed: `SERVER_RUNBOOK.md`

- [ ] **Step 1: Run the complete relevant local suite**

```bash
.venv-lerobot/bin/python -m pytest \
  tests/interaction_vla/representation_study/test_lerobot_backends.py \
  tests/interaction_vla/representation_study/libero/test_latents.py \
  tests/interaction_vla/representation_study/libero/test_stages.py \
  tests/interaction_vla/representation_study/libero/test_taps.py \
  tests/interaction_vla/representation_study/libero/test_project_contract.py -q
```

Expected: all selected tests pass with no skipped new regression tests.

- [ ] **Step 2: Run syntax and diff checks**

```bash
.venv-lerobot/bin/python -m compileall -q interaction_vla/representation_study
git diff --check
git status --short
```

Expected: compilation and diff checks succeed; status contains no unrelated staged files.

- [ ] **Step 3: Push the tested commits to `main`**

```bash
git push origin main
```

- [ ] **Step 4: Preserve invalid server cache under an explicit backup name**

After pulling the tested commit on the server, verify the destination does not exist and move the old cache:

```bash
test ! -e outputs/representation_study/libero_smolvla/latents_invalid_camera_binding_20260825
mv \
  outputs/representation_study/libero_smolvla/latents \
  outputs/representation_study/libero_smolvla/latents_invalid_camera_binding_20260825
```

Expected: old rows remain recoverable and the canonical `latents/` path is absent.

- [ ] **Step 5: Execute bounded server smoke checks before full extraction**

Start the real resumable extractor for at most three minutes per stage, accept only normal completion or timeout/interrupt status, and require at least one persisted row for every tap:

```bash
CONFIG=configs/representation_study/libero_smolvla_linux_cuda.yaml
for stage in pretrained sft_25
do
  set +e
  timeout --signal=INT 180 \
    .venv-lerobot/bin/python -m interaction_vla.representation_study \
      libero latents extract --config "$CONFIG" --stage "$stage"
  status=$?
  set -e
  case "$status" in
    0|124|130) ;;
    *) echo "latent smoke failed for $stage with status $status"; exit "$status" ;;
  esac
  for tap in vision_output multimodal_fusion action_expert_input pre_action
  do
    find \
      "outputs/representation_study/libero_smolvla/latents/$stage/$tap/.rows" \
      -name '*.npy' -print -quit | grep -q .
  done
done
```

Expected: both stages persist finite rows for `vision_output`, `multimodal_fusion`, `action_expert_input`, and `pre_action` without the missing-image-feature exception. These rows have the corrected binding and are intentionally resumed by Step 6.

- [ ] **Step 6: Run fresh full extraction only after smoke passes**

```bash
CONFIG=configs/representation_study/libero_smolvla_linux_cuda.yaml
for stage in pretrained sft_25
do
  .venv-lerobot/bin/python -m interaction_vla.representation_study \
    libero latents extract --config "$CONFIG" --stage "$stage"
done
```

Expected: each stage writes a complete `report.json`; every tap covers all 13,603 State Bank IDs under the new implementation binding.

## Self-review

- Spec coverage: checkpoint feature preservation, shared mapping, preprocessing consistency, cache invalidation, non-destructive old-cache handling, and two-stage real-batch smoke validation are all assigned to explicit tasks.
- Placeholder scan: no incomplete markers or unspecified implementation decisions remain.
- Type consistency: `rename_map` is accepted as `Mapping[str, str] | None`, copied to a `dict`, and passed to LeRobot APIs that require `dict[str, str]`; latent helper names and paths are consistent across implementation and tests.

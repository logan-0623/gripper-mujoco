# Physical Provenance Migration Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task by task. Apply `test-driven-development` to each behavior change, then use `verification-before-completion` before reporting success.

**Goal:** Migrate the current pilot gate, 162 manifest-referenced episodes, and the Flat/Graph seed-0 checkpoints onto the verified current provenance chain without changing training arrays or learned checkpoint state.

**Architecture:** Add one standalone, unhashed migration module that performs a read-only preflight, creates and verifies a complete backup, atomically rewrites only approved provenance fields, validates the result through the production loading path, and automatically restores the backup on any post-backup failure. Keep all existing controller-hash inputs unchanged so that the already verified gate remains current.

**Tech Stack:** Python 3.12, pathlib, hashlib/json/shutil/tempfile, NumPy NPZ, PyTorch checkpoints, tqdm, pytest.

**Design reference:** `docs/superpowers/specs/2026-08-02-provenance-migration-design.md`

**Repository note:** This workspace is not a Git worktree, so the commit steps normally included in an implementation plan are intentionally omitted. Every mutation of pilot artifacts is protected by the migrator's timestamped backup and rollback path.

---

## Task 1: Add read-only migration discovery and preflight

**Files:**

- Create: `interaction_vla/migrate_physics_provenance.py`
- Create: `tests/interaction_vla/test_migrate_physics_provenance.py`

### Step 1: Write failing tests for gate normalization and artifact discovery

Create temporary old/current gate JSON files whose only accepted difference is `controller_hash`. Add tests that assert:

```python
def test_gate_equivalence_allows_only_controller_hash_change(tmp_path: Path) -> None:
    old_gate = _write_gate(tmp_path / "old.json", controller_hash="old")
    new_gate = _write_gate(tmp_path / "new.json", controller_hash="new")

    assert_gate_reports_equivalent(old_gate, new_gate)

    report = json.loads(new_gate.read_text())
    report["results"][0]["success"] = False
    new_gate.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="differ outside controller_hash"):
        assert_gate_reports_equivalent(old_gate, new_gate)
```

Build a minimal physical config tree with two base episode files, one recovery file, and Flat/Graph seed-0 checkpoint files. Assert that `prepare_migration(...)` returns paths in deterministic manifest/representation order and performs no writes.

### Step 2: Run the focused tests and confirm they fail

Run:

```bash
.venv/bin/python -m pytest tests/interaction_vla/test_migrate_physics_provenance.py -q
```

Expected: FAIL because the migration module does not exist.

### Step 3: Implement immutable migration planning types and pure helpers

In `interaction_vla/migrate_physics_provenance.py`, add:

```python
@dataclass(frozen=True)
class MigrationPlan:
    config_path: Path
    output_dir: Path
    data_dir: Path
    official_gate: Path
    verified_gate: Path
    backup_dir: Path
    episode_paths: tuple[Path, ...]
    checkpoint_paths: tuple[Path, ...]
    old_gate_hash: str
    new_gate_hash: str
    old_hashes: Mapping[str, str]
    new_hashes: Mapping[str, str]


def prepare_migration(
    config_path: str | Path,
    *,
    verified_gate: str | Path,
    model_seeds: Iterable[int] = (0,),
    backup_dir: str | Path | None = None,
) -> MigrationPlan:
```

Implement these pure/read-only helpers:

- `_sha256(path)` hashes bytes without modifying the file.
- `_load_gate(path)` requires a JSON object with `passed=true`.
- `assert_gate_reports_equivalent(old_gate, verified_gate)` deep-copies both reports, replaces only the old top-level `controller_hash` with the verified value, and requires complete dictionary equality.
- `_checkpoint_paths(output_dir, model_seeds)` resolves both `flat` and `graph` checkpoints for every selected configured seed, rejects duplicate/unknown/empty seeds, and verifies all files before loading any.
- `_episode_paths(data_dir, recovery_enabled)` combines `episode_paths_from_manifest` and `recovery_paths_from_manifest` in manifest order, rejects cross-manifest duplicates, and returns exactly the training-provenance input set.
- `_default_backup_dir(output_dir)` uses a UTC timestamp with microseconds below `provenance_backups/`.

`prepare_migration` must:

1. require `backend=franka_contact`;
2. compare the verified gate against `expected_gate_hashes(config_path)`;
3. require the old and verified gates to match outside `controller_hash`;
4. compute old/new gate artifact SHA-256 values;
5. require every episode's metadata to be `backend=franka_contact`, `feature_schema=physics_v2`, and bound to the old gate artifact hash;
6. require each checkpoint's representation, model seed, physical top-level hashes, `training_provenance.expert_gate_hash`, and all old training-provenance fields to match the current manifests and old episode bytes;
7. reject an existing backup destination; and
8. return the plan without creating directories or changing files.

Use `torch.load(..., map_location="cpu", weights_only=False)` for checkpoint inspection. Reuse `resolve_training_data` and `build_training_provenance` rather than duplicating training provenance logic.

### Step 4: Run focused tests

Run:

```bash
.venv/bin/python -m pytest tests/interaction_vla/test_migrate_physics_provenance.py -q
```

Expected: gate and preflight tests PASS.

---

## Task 2: Implement atomic episode and checkpoint rewrites with exact invariants

**Files:**

- Modify: `interaction_vla/migrate_physics_provenance.py`
- Modify: `tests/interaction_vla/test_migrate_physics_provenance.py`

### Step 1: Write failing episode rewrite tests

Create an NPZ with metadata plus arrays using multiple dtypes. Call `_rewrite_episode_gate_hash(...)` and assert:

```python
assert after_metadata == {**before_metadata, "expert_gate_hash": new_gate_hash}
for name in set(before_arrays) - {"metadata"}:
    assert after_arrays[name].dtype == before_arrays[name].dtype
    assert after_arrays[name].shape == before_arrays[name].shape
    assert np.array_equal(after_arrays[name], before_arrays[name], equal_nan=True)
```

Also assert the helper rejects an unexpected old hash before replacement and leaves the original file digest unchanged.

### Step 2: Write failing recursive checkpoint invariant tests

Build a checkpoint payload containing model tensors, optimizer tensors, nested statistics, epoch/global-step scalars, and provenance. Assert `_rewrite_checkpoint(...)` changes only:

- `expert_gate_hash`;
- `controller_hash`;
- `scene_hash`;
- `config_hash`; and
- `training_provenance`.

Test that `_assert_nested_equal` handles mappings, lists, tuples, NumPy arrays, tensors, scalars, NaNs, and type mismatches. Inject a change into `model_state` and require an invariant error naming that field.

### Step 3: Run the focused tests and confirm they fail

Run:

```bash
.venv/bin/python -m pytest tests/interaction_vla/test_migrate_physics_provenance.py -q
```

Expected: FAIL because rewrite helpers are missing.

### Step 4: Implement same-directory atomic writes

Add an `_atomic_path(destination)` context manager that creates a named temporary file in `destination.parent`, flushes and `fsync`s file contents, yields the path, and uses `os.replace` only after validation succeeds. It must unlink its own temporary file on failure.

Implement `_rewrite_episode_gate_hash(path, old_gate_hash, new_gate_hash)`:

1. load all arrays with `allow_pickle=False` and copy them out of the archive;
2. parse scalar JSON metadata and require the exact old hash;
3. create new sorted-key JSON metadata with only `expert_gate_hash` changed;
4. write `np.savez_compressed` to the atomic temporary path;
5. reload the temporary archive and compare dtype, shape, key set, and values for every non-metadata array;
6. compare before/after metadata dictionaries after normalizing only `expert_gate_hash`; and
7. atomically replace the episode.

Use a temporary filename ending in `.npz`; otherwise NumPy silently appends another suffix and breaks atomic replacement.

Implement `_assert_nested_equal(expected, actual, path)` with exact `torch.equal`/`np.array_equal(..., equal_nan=True)` semantics and clear field paths.

Implement `_rewrite_checkpoint(path, new_physical_hashes, new_training_provenance)`:

1. load the original payload on CPU;
2. make a shallow top-level copy and replace only approved provenance keys;
3. write with `torch.save` to a same-directory temporary file;
4. reload it on CPU;
5. remove approved keys from the old/new copies and recursively require exact equality for all remaining payload state;
6. require the newly loaded approved fields to exactly equal requested values; and
7. atomically replace the checkpoint.

### Step 5: Run focused tests

Run:

```bash
.venv/bin/python -m pytest tests/interaction_vla/test_migrate_physics_provenance.py -q
```

Expected: episode/checkpoint unit tests PASS.

---

## Task 3: Add verified backup and rollback orchestration

**Files:**

- Modify: `interaction_vla/migrate_physics_provenance.py`
- Modify: `tests/interaction_vla/test_migrate_physics_provenance.py`

### Step 1: Write failing backup tests

Add a test for `_create_backup(plan)` that asserts the backup contains:

- `expert_gate.json`;
- a complete `data/` tree, including an unreferenced sentinel file;
- selected checkpoints at `flat/seed_0/checkpoint.pt` and `graph/seed_0/checkpoint.pt`; and
- `backup_manifest.json` whose recorded digest for every potentially modified source equals the digest of both source and backup copies.

Verify a pre-existing backup directory is rejected without overwriting it.

### Step 2: Write a failing rollback test

Monkeypatch `_rewrite_checkpoint` to raise after episode and gate mutation. Call the orchestration entry point and assert:

```python
with pytest.raises(RuntimeError, match="rollback completed"):
    migrate_from_config(...)

assert _tree_digests(output_dir) == original_digests
assert backup_dir.exists()
```

The digest comparison must cover the official gate, the complete data directory, and both selected checkpoints.

### Step 3: Run focused tests and confirm they fail

Run:

```bash
.venv/bin/python -m pytest tests/interaction_vla/test_migrate_physics_provenance.py -q
```

Expected: FAIL because backup/orchestration functions are missing.

### Step 4: Implement backup creation and verification

Add `_create_backup(plan)` using `shutil.copy2`/`shutil.copytree`. Keep artifact-relative paths under the backup root. Write `backup_manifest.json` only after copying; it records:

```json
{
  "schema_version": 1,
  "source_output_dir": "...",
  "files": [
    {
      "relative_path": "data/episode_000000.npz",
      "sha256": "...",
      "will_modify": true
    }
  ]
}
```

The file list covers the original official gate, every file in the complete data directory, and selected checkpoints. Verify every backup copy digest before returning. On backup failure before mutation, leave the incomplete directory for diagnosis and raise without attempting artifact restore.

### Step 5: Implement exact rollback

Add `_restore_backup(plan)` that copies back:

1. the official gate;
2. the complete data tree from backup; and
3. selected checkpoints.

Because mutation never creates or deletes files inside the data tree, copy every backed-up file back in place and verify all original digests from `backup_manifest.json`. If restore verification fails, raise an error that clearly states rollback verification failed and preserves both the original exception and the backup path.

### Step 6: Implement the main migration transaction

Add:

```python
def migrate_from_config(
    config_path: str | Path,
    *,
    verified_gate: str | Path,
    model_seeds: Iterable[int] = (0,),
    backup_dir: str | Path | None = None,
    show_progress: bool = False,
) -> Path:
```

Flow:

1. call `prepare_migration` before any mutation;
2. create and verify the backup;
3. rewrite the 162 manifest-referenced episodes;
4. atomically promote the verified gate bytes to the official gate;
5. reload config, recompute `TrainingDataSelection` and new training provenance from migrated episode bytes;
6. rewrite selected checkpoints;
7. run post-migration validation from Task 4;
8. write the report atomically; and
9. on any exception after a completed backup, restore and verify the original artifacts before raising `RuntimeError("provenance migration failed; rollback completed; backup: ...")` from the original exception.

Do not catch `KeyboardInterrupt`/`SystemExit`; place rollback in a `BaseException` handler so an interrupt after mutation is also restored, then re-raise interrupt types after successful rollback with the backup path added through a note.

### Step 7: Run focused tests

Run:

```bash
.venv/bin/python -m pytest tests/interaction_vla/test_migrate_physics_provenance.py -q
```

Expected: backup and rollback tests PASS.

---

## Task 4: Add production-path post-validation, report, CLI, and tqdm

**Files:**

- Modify: `interaction_vla/migrate_physics_provenance.py`
- Modify: `tests/interaction_vla/test_migrate_physics_provenance.py`

### Step 1: Write failing post-validation tests

Build a complete synthetic migration fixture compatible with `load_training_checkpoint`. After migration, assert:

- `expert_gate_provenance` returns the new gate and current physical hashes;
- `require_episode_gate_provenance` accepts every referenced episode;
- `build_training_provenance` equals the provenance in both checkpoints;
- `preload_evaluation_checkpoints` loads Flat and Graph seed 0;
- non-metadata episode arrays equal their backup copies; and
- checkpoint fields outside approved provenance keys equal their backup copies.

If a full minimal policy checkpoint is too expensive for this test, monkeypatch only `preload_evaluation_checkpoints` and retain real tests for all other validation calls; separately assert it receives both seed-0 representations through its normal seed interface.

### Step 2: Write failing CLI tests

Test parser/defaults and `main()` forwarding:

```python
args = build_parser().parse_args([
    "--config", "pilot.yaml",
    "--verified-gate", "verified.json",
    "--model-seeds", "0",
])
assert args.model_seeds == [0]
```

Monkeypatch `migrate_from_config` and assert `main()` passes `show_progress=True` and prints the report path.

Add a progress test by monkeypatching the module's `tqdm` and asserting total is `len(episode_paths) + len(checkpoint_paths) + 2`: one gate promotion, one final validation.

### Step 3: Run focused tests and confirm they fail

Run:

```bash
.venv/bin/python -m pytest tests/interaction_vla/test_migrate_physics_provenance.py -q
```

Expected: FAIL because post-validation/CLI/reporting is incomplete.

### Step 4: Implement post-validation against production code

Add `_post_validate(plan, new_training_provenance)` that:

1. calls `expert_gate_provenance(config_path, official_gate)`;
2. calls `require_episode_gate_provenance(episode_paths, new_gate_hash)`;
3. recomputes the current selection/provenance and requires exact equality;
4. checks each checkpoint payload's physical and training provenance;
5. calls `preload_evaluation_checkpoints` with selected model seeds and `device="cpu"`;
6. compares every episode non-metadata array against the backup copy; and
7. compares every checkpoint field outside approved provenance keys against the backup copy.

Return an invariant summary mapping for the audit report.

### Step 5: Implement the audit report

Atomically write `output_dir/provenance_migration.json` with:

- schema version and UTC completion timestamp;
- absolute config, verified-gate, output, data, and backup paths;
- old/new gate artifact hashes;
- old/new controller, scene, and config hashes;
- migrated episode/checkpoint counts and relative paths;
- old/new dataset-content hashes;
- manifest and recovery-manifest hashes;
- invariant flags for episode arrays, checkpoint payloads, gate equivalence, backup verification, production checkpoint preload, and rollback availability.

The report is not part of rollback inputs because it is written only after successful validation. If a stale report already exists, preserve it in the backup root before replacement and record that preservation in `backup_manifest.json`.

### Step 6: Add parser, main entry point, and progress bar

Expose:

```bash
.venv/bin/python -m interaction_vla.migrate_physics_provenance \
  --config configs/physics_recovery_pilot_macos.yaml \
  --verified-gate /tmp/interaction_vla_pilot_gate_progress_probe.json \
  --model-seeds 0
```

Parser options:

- required `--config`;
- required `--verified-gate`;
- one-or-more integer `--model-seeds`, default `[0]`;
- optional `--backup-dir`.

The CLI calls `migrate_from_config(..., show_progress=True)`. Use one `tqdm(desc="provenance migration", unit="artifact")` instance, update once after each episode/checkpoint, once after gate promotion, and once after post-validation, and always close it in `finally`. Set postfix `phase` and a short relative path. Quiet library calls create no tqdm object.

### Step 7: Run focused tests

Run:

```bash
.venv/bin/python -m pytest tests/interaction_vla/test_migrate_physics_provenance.py -q
```

Expected: all migration tests PASS.

---

## Task 5: Verify source boundaries and the real pilot preflight

**Files:**

- Modify only if a test exposes a defect: `interaction_vla/migrate_physics_provenance.py`
- Modify only if test coverage needs correction: `tests/interaction_vla/test_migrate_physics_provenance.py`

### Step 1: Prove the migration module did not change current physical hashes

Capture and compare:

```bash
.venv/bin/python - <<'PY'
from interaction_vla.physics_data import expected_gate_hashes
print(expected_gate_hashes("configs/physics_recovery_pilot_macos.yaml"))
PY
```

Expected controller hash:

```text
420431f92ca1269b7a4f2bbd318b1453002cc436f81771f17727a48d26fc5e52
```

### Step 2: Run the real read-only preflight

Run:

```bash
.venv/bin/python - <<'PY'
from interaction_vla.migrate_physics_provenance import prepare_migration

plan = prepare_migration(
    "configs/physics_recovery_pilot_macos.yaml",
    verified_gate="/tmp/interaction_vla_pilot_gate_progress_probe.json",
    model_seeds=(0,),
)
print(len(plan.episode_paths))
print(len(plan.checkpoint_paths))
print(plan.old_gate_hash)
print(plan.new_gate_hash)
PY
```

Expected: `162` episodes, `2` checkpoints, and no artifact digest changes.

### Step 3: Run the full automated verification suite

Run:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q interaction_vla tests/interaction_vla
```

Expected: all tests PASS and compileall exits 0.

### Step 4: Review the implementation against the approved mutation allowlist

Inspect the working tree manually and require:

- only the new migration module, its test file, and documentation changed;
- no file named by `_PHYSICS_CONTROL_MODULES` changed;
- no stale-provenance bypass or `force` option exists;
- report/backup paths stay within the configured output directory unless the caller explicitly supplies `--backup-dir`; and
- every output mutation occurs only after successful preflight and verified backup.

---

## Task 6: Execute the authorized real migration and verify evaluation preflight

**Files/artifacts (intentional runtime mutations):**

- Modify: `outputs/interaction_graph_physics/recovery_pilot/expert_gate.json`
- Modify: the 162 manifest-referenced NPZ files below `outputs/interaction_graph_physics/recovery_pilot/data/`
- Modify: `outputs/interaction_graph_physics/recovery_pilot/flat/seed_0/checkpoint.pt`
- Modify: `outputs/interaction_graph_physics/recovery_pilot/graph/seed_0/checkpoint.pt`
- Create: timestamped `outputs/interaction_graph_physics/recovery_pilot/provenance_backups/<timestamp>/`
- Create: `outputs/interaction_graph_physics/recovery_pilot/provenance_migration.json`

### Step 1: Record original digests outside the mutation set

Before execution, record the official gate, all data-tree files, and selected checkpoint digests in a temporary verification file under `/tmp`. This is independent confirmation in addition to `backup_manifest.json`.

### Step 2: Run the migration once

Run:

```bash
.venv/bin/python -m interaction_vla.migrate_physics_provenance \
  --config configs/physics_recovery_pilot_macos.yaml \
  --verified-gate /tmp/interaction_vla_pilot_gate_progress_probe.json \
  --model-seeds 0
```

Expected: one tqdm bar completes, a permanent backup remains, and the CLI prints the new `provenance_migration.json` path.

### Step 3: Inspect the audit report and backup

Require all invariant flags to be true, episode count `162`, checkpoint count `2`, and backup hashes to match the pre-migration `/tmp` digest record. Confirm the old gate, complete data directory, and selected checkpoints can be recovered from the backup.

### Step 4: Verify production evaluation preflight without running 80 rollouts

Run the normal evaluation command and stop it only after the tqdm line displays `physics eval: 0/80`, proving gate, episode, training-provenance, and both checkpoint preloads succeeded:

```bash
.venv/bin/python -m interaction_vla.physics_evaluate \
  --config configs/physics_recovery_pilot_macos.yaml \
  --model-seeds 0 \
  --episodes-per-count 5
```

Send `Ctrl-C` after `physics eval: 0/80`; evaluation does not mutate the migrated gate/data/checkpoints. If preflight fails before the bar appears, do not rerun migration: inspect `provenance_migration.json` and the preserved backup first.

### Step 5: Re-run focused and full verification after runtime migration

Run:

```bash
.venv/bin/python -m pytest tests/interaction_vla/test_migrate_physics_provenance.py -q
.venv/bin/python -m pytest -q
```

Expected: all tests PASS with the pilot artifacts on the new strict provenance chain.

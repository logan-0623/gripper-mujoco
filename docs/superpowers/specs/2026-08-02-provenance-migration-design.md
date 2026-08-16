# Physical Provenance Migration Design

## Objective

Repair the stale-provenance error without recollecting 50 demonstrations or
retraining the seed-0 Flat and Graph policies. The migration must preserve every
training array and learned tensor exactly, create a recoverable backup before
mutation, and bind the existing artifacts to the current verified expert gate.

## Evidence and Scope

The official gate, all manifest-referenced episodes, and both seed-0
checkpoints form an internally consistent artifact chain with controller hash
`3be419dd16d23bae4c7659ef598c0b443e70081d44d9a970d304f86b2234f5e7`.
The current controller hash is
`420431f92ca1269b7a4f2bbd318b1453002cc436f81771f17727a48d26fc5e52`;
scene and config hashes are unchanged. The old official gate and the new
verified gate use the same 40 cases, and their parsed reports are structurally
equal after replacing only `controller_hash`. Every per-case success, reason,
step count, stable-lift flag, and physics-failure flag is identical.

The data directory is 19MB. Its manifests reference 50 base and 112 recovery
NPZ files. The remaining 54 NPZ files are not training inputs. The two selected
checkpoints are approximately 1MB each.

## Considered Approaches

1. Re-run gate, recollect data, and retrain. This is maximally conservative but
   repeats expensive work even though the verified physical outcomes are
   identical.
2. Add a stale-provenance bypass to evaluation. This avoids artifact writes but
   weakens safety for all future runs and makes the report depend on an unsafe
   flag.
3. Perform a verified, backed-up metadata migration. This keeps strict
   provenance checks enabled, preserves numeric artifacts, and records a
   recoverable audit trail. This is the approved approach.

## Command and Module Boundary

Create `interaction_vla/migrate_physics_provenance.py` with a quiet library
entry point and a tqdm-enabled CLI:

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

The user command is:

```bash
.venv/bin/python -m interaction_vla.migrate_physics_provenance \
  --config configs/physics_recovery_pilot_macos.yaml \
  --verified-gate /tmp/interaction_vla_pilot_gate_progress_probe.json \
  --model-seeds 0
```

The new migration module is not added to `_PHYSICS_CONTROL_MODULES`, and no
existing hashed source module is modified. Creating the migration code therefore
does not invalidate the already verified current gate.

## Preflight Validation

No output artifact is changed until all checks pass:

1. Load the physical config and resolve the official gate, both manifests, all
   162 referenced NPZ files, and the selected Flat/Graph checkpoints.
2. Require both old and verified gate reports to have `passed=true`.
3. Require the verified gate controller, scene, and config hashes to equal the
   current source, scene, and config hashes.
4. Require the old gate scene and config hashes to equal the verified gate.
5. Compare the complete old and verified gate reports after normalizing only
   the old `controller_hash` to the verified value. Any other difference aborts
   migration.
6. Compute the SHA-256 of the old and verified gate files.
7. Require every referenced episode metadata record to contain the old gate
   artifact hash, `backend=franka_contact`, and `feature_schema=physics_v2`.
8. Require every selected checkpoint top-level gate/controller/scene/config
   value and training-provenance gate hash to match the old artifact chain.
9. Require every referenced file and selected checkpoint to exist before
   backup creation.

## Backup and Recovery

The default backup directory is a timestamped child of
`outputs/interaction_graph_physics/recovery_pilot/provenance_backups/`. It
contains:

- the original official `expert_gate.json`;
- a complete copy of the 19MB data directory, including manifests and
  unreferenced files;
- the selected Flat/Graph checkpoint files in their original relative paths;
- `backup_manifest.json` with the original SHA-256 and relative path of every
  file that migration may modify.

The backup directory must not already exist. Mutation begins only after every
backup file is copied and its digest matches the original. If any later step
fails, the migrator restores the official gate, complete data directory, and
selected checkpoints from the backup, verifies their original digests, and
raises an error stating that rollback completed. The successful backup is never
deleted automatically.

## Episode Migration

Only the 162 NPZ files referenced by `manifest.json` and
`recovery_manifest.json` are rewritten. For each archive:

1. Load every array with `allow_pickle=False`.
2. Change only `metadata.expert_gate_hash` from the old gate artifact hash to
   the verified gate artifact hash.
3. Write a compressed temporary NPZ in the same directory.
4. Reload the temporary archive and require exact dtype, shape, and
   `numpy.array_equal` equality for every non-metadata array.
5. Require the metadata dictionaries to differ only at `expert_gate_hash`.
6. Atomically replace the original archive.

The manifest and recovery-manifest contents and ordering do not change.

## Gate and Checkpoint Migration

Promote the verified gate bytes to the official gate through a same-directory
temporary file and atomic replace. Recompute `TrainingDataSelection` and
`training_provenance` using the migrated NPZ bytes and the new official gate
artifact hash.

For each selected Flat/Graph checkpoint, change only:

- top-level `expert_gate_hash`;
- top-level `controller_hash`, `scene_hash`, and `config_hash`;
- the complete `training_provenance` mapping recomputed by existing training
  code.

Before atomic replacement, reload the temporary checkpoint and recursively
require equality for every other payload field. Tensor equality requires the
same dtype, shape, device-normalized CPU value, and `torch.equal` result. This
covers model weights, optimizer state, normalization statistics, epoch count,
global step, representation, model seed, dimensions, and model arguments.

## Post-Migration Verification

The migrator runs the production validation path after all replacements:

- `expert_gate_provenance` must accept the official gate and return current
  hashes;
- `require_episode_gate_provenance` must accept all 162 referenced episodes;
- recomputed training provenance must exactly match both checkpoints;
- `preload_evaluation_checkpoints` must load both seed-0 representations under
  current physical and training provenance;
- every migrated episode non-metadata array must equal its backup copy;
- every checkpoint field outside the approved provenance keys must equal its
  backup copy.

On success, write `provenance_migration.json` containing old/new hashes, backup
path, migrated episode/checkpoint counts, invariant-check results, and the
verified-gate path. The CLI prints this report path.

## Progress Reporting

The CLI displays one `provenance migration` tqdm bar whose total covers the 162
episode rewrites, two checkpoint rewrites, gate promotion, and final validation.
Its postfix reports the current phase and relative artifact path. The library
entry point remains quiet by default, and the bar closes through `finally`.

## Testing

Tests use temporary synthetic gates, NPZ archives, manifests, and checkpoints.
They prove:

- preflight rejects a verified gate with any non-controller report difference;
- episode migration changes only the gate hash in metadata;
- checkpoint migration changes only approved provenance fields;
- successful orchestration creates a complete backup and report;
- injected mid-migration failure restores original file digests;
- the real workspace preflight identifies 162 referenced episodes and two
  seed-0 checkpoints without mutating them;
- the full test suite and compilation pass.

After tests pass, run the migration once on the authorized pilot artifacts,
then run the production 80-rollout evaluation command far enough to complete
preflight and display `physics eval: 0/80`. The migration itself must not launch
training or data collection.

## Non-goals

- Changing model weights, optimizer state, training arrays, manifests, physics,
  controller behavior, success criteria, or the expert policy.
- Migrating unreferenced NPZ files.
- Adding a general stale-provenance bypass.
- Redesigning the controller-hash module list in this migration.

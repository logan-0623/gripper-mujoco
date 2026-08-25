# LIBERO SmolVLA Latent Camera-Binding Fix

## Problem

Formal SmolVLA latent extraction fails because the loaded policy and its preprocessor use different image-key contracts. The SFT checkpoint was trained with:

```text
observation.images.image  -> observation.images.camera1
observation.images.image2 -> observation.images.camera2
```

The saved checkpoint therefore expects `camera1`, `camera2`, and an optional masked `camera3`. The current dataset-bound loader clears those checkpoint input features and rebuilds them from unrenamed LIBERO metadata, so the policy expects `image` and `image2` while the saved preprocessor emits `camera1` and `camera2`.

Server inspection confirmed the mismatch directly:

```text
policy_images       = [observation.images.image, observation.images.image2]
preprocessor_rename = {image: camera1, image2: camera2}
```

The existing incomplete pretrained latent cache was produced with this incorrect runtime binding and must not be mixed with corrected latents.

## Decision

Make dataset-bound inference follow the same feature-binding path as formal training:

1. Preserve the checkpoint's input feature contract.
2. Pass the explicit LIBERO rename map to LeRobot `make_policy`.
3. Apply the same map to the preprocessor's rename step.
4. Bind normalization and output features to the LIBERO dataset metadata without replacing the checkpoint's image feature names.
5. Include the LeRobot backend implementation in the latent-cache scientific binding so loader changes invalidate old caches.

This keeps pretrained and SFT stages comparable under one observation contract and does not modify or retrain any checkpoint.

## Alternatives rejected

### Rename processed tensors back to dataset keys

This would make the current incorrectly rebound policy run, but would not reproduce the feature contract used during SFT. It risks producing executable yet scientifically incomparable latents.

### Separate pretrained and SFT loaders

The base checkpoint needs dataset statistics while the SFT checkpoint already carries a trained processor. Separate branches could be made to work, but they would duplicate policy-binding logic and increase the risk of stage-specific representation differences introduced by the adapter itself.

## Code changes

### Backend

Extend `LeRobotPolicyBackend.load_checkpoint_for_dataset` with an optional rename map. For modern VLAs, retain checkpoint input features, pass the map to `make_policy`, and override the rename processor consistently when rebuilding pre/post processors.

### LIBERO latent extraction

Pass the formal `LIBERO_SMOLVLA_RENAME_MAP` used by the training command into the backend loader.

Expand the latent implementation identity to cover the backend loader and the source of the rename-map contract. An existing cache produced by the old loader must fail its binding check instead of being silently reused.

### Cache handling

Do not delete or overwrite the incomplete cache automatically. Before rerunning on the server, move the old `latents` directory to an explicitly named invalid-camera-binding backup. The corrected run then creates a fresh cache.

## Tests

1. A regression test verifies that dataset-bound loading preserves checkpoint visual feature names and forwards the rename map to `make_policy`.
2. A regression test verifies that the rebuilt preprocessor receives the same rename map.
3. A binding test verifies that backend-loader changes affect the latent implementation identity.
4. Existing LIBERO representation-study tests remain green.
5. A server smoke test loads both pretrained and SFT-25 checkpoints, processes one real State Bank batch, and reaches every preregistered semantic tap.

## Success criteria

- Policy image features remain `camera1`, `camera2`, and optional `camera3` after dataset binding.
- Processed LIBERO batches contain `camera1` and `camera2`.
- One real batch succeeds for pretrained and SFT-25 without the missing-image-feature exception.
- Fresh latent extraction uses a new scientific binding and cannot mix with the old cache.
- SFT-25 training artifacts and all checkpoint bytes remain unchanged.

## Non-goals

- No SFT retraining.
- No change to the State Bank or interaction labels.
- No probe, intervention, closed-loop, or RL protocol change.
- No attempt to salvage scientifically invalid latent rows.

# ACT Rollout GIF Design

## Goal

Extend the existing LeRobot ACT rollout command so a user can run the saved
500-step smoke checkpoint in MuJoCo and receive a visual artifact that proves
which observations were supplied to the policy during closed-loop control.
The artifact is diagnostic evidence, not a task-performance claim.

## User Interface

The existing command gains one optional argument:

```bash
.venv-lerobot/bin/python -m interaction_vla.lerobot_bridge rollout \
  --config configs/lerobot_act_smoke_macos.yaml \
  --checkpoint outputs/lerobot/act_smoke/checkpoint \
  --object-count 2 \
  --gif outputs/lerobot/act_smoke/rollout.gif
```

Without `--gif`, rollout behavior remains unchanged. With `--gif`, the command
writes the GIF atomically and includes its path and frame count in the final
JSON result.

## Visual Contract

Each GIF frame contains the exact agent RGB and wrist RGB images captured for
that policy step, arranged left-to-right at their native 256×256 resolution.
A small header identifies the artifact as an ACT smoke rollout and displays the
policy step, gripper state, and latest terminal status. The two panels are
explicitly labeled `agent RGB` and `wrist RGB`.

The rollout remains at 20 Hz, while the GIF retains every second observation
and plays at 10 FPS. This preserves the full motion duration while controlling
file size. The final rendered frame displays the actual terminal reason. A
timeout or task failure must remain visibly labeled and must never be presented
as a successful policy result.

## Components and Data Flow

`rollout_checkpoint` continues to own environment reset, policy inference,
action chunking, safety projection, and stepping. Immediately after capturing
the two policy views, it passes copies of those images and current diagnostic
text to a small rollout-GIF recorder. The recorder only observes data already
available to the policy loop and cannot affect actions or MuJoCo state.

The recorder builds labeled side-by-side RGB frames in memory, samples them at
the configured display rate, converts them to an indexed palette, and writes a
temporary GIF using Pillow. After a successful write, `os.replace` publishes
the requested destination. If encoding fails, the temporary file is removed
and the rollout command exits with its existing JSON error envelope.

The CLI forwards `--gif` to `rollout_from_config`. The smoke command does not
record a GIF by default, avoiding an unexpected artifact or additional memory
cost in automated gates.

## Error Handling

The destination must end in `.gif`. Its parent directory is created when
needed. Existing GIFs are replaced atomically only after the new file is fully
encoded. Captured images must be matching `uint8` H×W×3 arrays. Invalid frame
data or an unwritable destination causes a clear exception and a nonzero CLI
exit; no partial destination is retained.

## Testing and Acceptance

Unit tests cover frame composition, panel labels, frame sampling, valid GIF
output, temporary-file cleanup, CLI forwarding, and the unchanged no-GIF path.
The existing rollout and full project suites must remain green.

Acceptance requires one real Mac MPS rollout using the smoke checkpoint. The
result must contain 180 finite control steps, produce a readable animated GIF
with both 256×256 views, and truthfully report the observed terminal reason.
The generated GIF and `rollout.json` live under
`outputs/lerobot/act_smoke/` and remain local artifacts.

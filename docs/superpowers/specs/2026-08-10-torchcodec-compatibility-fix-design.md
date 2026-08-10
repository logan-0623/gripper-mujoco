# TorchCodec Compatibility Fix Design

## Goal

Remove the fatal-looking TorchCodec import traceback from the documented
LeRobot rollout command while preserving its existing ACT policy, MuJoCo,
PyAV, checkpoint, and GIF behavior.

## Root Cause

The macOS lock file combines `torch==2.10.0` with `torchcodec==0.11.1`.
TorchCodec's compatibility matrix pairs Torch 2.10 with TorchCodec 0.10 and
Torch 2.11 with TorchCodec 0.11. The incompatible native extension fails to
load, after which LeRobot falls back to PyAV and the rollout completes with
exit code zero. The fallback makes the run functional, but the dependency lock
is invalid and prints a large, misleading traceback on every invocation.

## Change

Add an explicit `torchcodec>=0.10,<0.11` constraint beside the existing
`torch>=2.10,<2.11` constraint and pin the resolved macOS lock entry to
`torchcodec==0.10.0`. No rollout source code, model configuration, dataset, or
checkpoint format changes.

Reinstall the corrected TorchCodec version in `.venv-lerobot` and verify that
it imports against the current Torch and Homebrew FFmpeg installation.

## Regression Protection

Add a dependency-contract test that parses the direct requirements and lock
file and asserts the supported Torch 2.10 / TorchCodec 0.10 pairing. The test
must fail against the current 0.11.1 lock before either requirements file is
changed.

Run the dependency-contract test, the LeRobot bridge suite, and the original
rollout command without auxiliary environment variables. Acceptance requires
a zero exit code, a readable GIF, a passing rollout JSON result, and no
`Could not load libtorchcodec` traceback. Existing PyAV remains installed as a
supported fallback.

## Scope

This fix does not upgrade Torch, LeRobot, or any policy implementation. It also
does not treat `task_success=false` from the 500-step smoke model as an
environment failure; learned task performance remains outside this dependency
repair.

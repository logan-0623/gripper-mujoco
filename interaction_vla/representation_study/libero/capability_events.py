from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from ..state_bank.io import write_json_atomic
from .annotation import AnnotationThresholds, PrivilegedFrame, annotate_relocation_episode
from .runtime import LiberoOffscreenSimulator


@dataclass
class EpisodeEventTracker:
    task_id: int
    initial_state_id: int
    control_freq: int
    thresholds: AnnotationThresholds
    frames: list[PrivilegedFrame]

    def add(self, frame: PrivilegedFrame) -> None:
        self.frames.append(replace(frame, frame_index=len(self.frames)))


def _summarize(tracker: EpisodeEventTracker, labels, *, success: bool) -> dict[str, object]:
    contact = [bool(label.contact.gripper_target) for label in labels]
    stable = [label.stable_grasp is True for label in labels]
    target_z = np.asarray([frame.target_pose[2, 3] for frame in tracker.frames])
    initial_z = float(target_z[0])
    lift = [
        is_stable and z - initial_z >= tracker.thresholds.lift_clearance_m
        for is_stable, z in zip(stable, target_z, strict=True)
    ]

    def first(values: list[bool], start: int = 0) -> int | None:
        return next((i for i, value in enumerate(values[start:], start) if value), None)

    stable_onset = first(stable)
    lift_onset = first(lift)
    release_onset = (
        first([not value for value in contact], stable_onset + 1)
        if stable_onset is not None
        else None
    )
    normal_release = bool(
        release_onset is not None
        and (
            success
            or any(frame.goal_satisfied for frame in tracker.frames[release_onset:])
        )
    )
    max_lift = float(np.max(target_z - initial_z, initial=0.0))
    drop_onset = None
    if lift_onset is not None and not normal_release:
        peak = float(target_z[lift_onset])
        for index in range(lift_onset + 1, len(target_z)):
            peak = max(peak, float(target_z[index]))
            if (
                not contact[index]
                and not tracker.frames[index].goal_satisfied
                and peak - float(target_z[index]) >= tracker.thresholds.lift_clearance_m
            ):
                drop_onset = index
                break

    longest = run = 0
    for value in stable:
        run = run + 1 if value else 0
        longest = max(longest, run)
    return {
        "task_id": tracker.task_id,
        "initial_state_id": tracker.initial_state_id,
        "steps": len(tracker.frames) - 1,
        "contact": any(contact),
        "contact_onset_step": first(contact),
        "stable_grasp": any(stable),
        "stable_grasp_onset_step": stable_onset,
        "lift": any(lift),
        "lift_onset_step": lift_onset,
        "max_target_lift_m": max_lift,
        "longest_stable_hold_steps": longest,
        "longest_stable_hold_s": longest / tracker.control_freq,
        "release_after_grasp": release_onset is not None,
        "release_onset_step": release_onset,
        "normal_release": normal_release,
        "unintended_drop": drop_onset is not None,
        "drop_onset_step": drop_onset,
        "success": bool(success),
    }


def install_libero_event_recorder(
    *, output: Path, suite: str, initial_state_offset: int, thresholds: AnnotationThresholds
) -> None:
    from lerobot.envs.libero import LiberoEnv

    if initial_state_offset < 0:
        raise ValueError("initial_state_offset must be non-negative")
    rows: list[dict[str, object]] = []
    monitors: dict[int, tuple[LiberoOffscreenSimulator, EpisodeEventTracker, bool]] = {}
    original_reset, original_step, original_close = LiberoEnv.reset, LiberoEnv.step, LiberoEnv.close

    def save() -> None:
        write_json_atomic(
            output,
            {
                "schema": "libero_capability_events_v1",
                "suite": suite,
                "initial_state_offset": initial_state_offset,
                "thresholds": thresholds.__dict__,
                "episodes": rows,
            },
        )

    def finish(env, success: bool | None = None) -> None:
        item = monitors.get(id(env))
        if item is None or item[2]:
            return
        adapter, tracker, observed_success = item
        rows.append(_summarize(
            tracker,
            annotate_relocation_episode(tracker.frames, adapter.semantics, thresholds),
            success=observed_success if success is None else success,
        ))
        monitors[id(env)] = (adapter, tracker, True)
        save()

    def reset(env, seed=None, **kwargs):
        finish(env)
        if not getattr(env, "_capability_offset_applied", False):
            env.init_state_id += initial_state_offset
            env._capability_offset_applied = True
        initial_state_id = int(env.init_state_id % len(env._init_states))
        result = original_reset(env, seed=seed, **kwargs)
        adapter = LiberoOffscreenSimulator.from_live_env(
            env._env,
            suite=suite,
            task_id=int(env.task_id),
            task_name=str(env.task),
            language=str(env.task_description),
        )
        tracker = EpisodeEventTracker(
            task_id=int(env.task_id),
            initial_state_id=initial_state_id,
            control_freq=int(env.control_freq),
            thresholds=thresholds,
            frames=[],
        )
        tracker.add(adapter._privileged_frame())
        monitors[id(env)] = (adapter, tracker, False)
        return result

    def step(env, action):
        result = original_step(env, action)
        adapter, tracker, finished = monitors[id(env)]
        adapter._observation = adapter.domain._get_observations()
        tracker.add(adapter._privileged_frame())
        success = bool(result[4].get("is_success", False))
        monitors[id(env)] = (adapter, tracker, finished or success)
        if result[2] or result[3]:
            monitors[id(env)] = (adapter, tracker, False)
            finish(env, success=success)
        return result

    def close(env):
        finish(env)
        return original_close(env)

    LiberoEnv.reset, LiberoEnv.step, LiberoEnv.close = reset, step, close


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events-output", type=Path, required=True)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--initial-state-offset", type=int, default=10)
    parser.add_argument("--stable-window-frames", type=int, default=5)
    parser.add_argument("--lift-clearance-m", type=float, default=0.01)
    args, forwarded = parser.parse_known_args()
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    install_libero_event_recorder(
        output=args.events_output,
        suite=args.suite,
        initial_state_offset=args.initial_state_offset,
        thresholds=AnnotationThresholds(
            stable_window_frames=args.stable_window_frames,
            lift_clearance_m=args.lift_clearance_m,
        ),
    )
    from lerobot.scripts.lerobot_eval import main as lerobot_eval_main

    sys.argv = [sys.argv[0], *forwarded]
    lerobot_eval_main()


if __name__ == "__main__":
    main()

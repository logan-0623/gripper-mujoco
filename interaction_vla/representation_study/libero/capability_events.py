from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field, replace
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
    drop_height_m: float
    frames: list[PrivilegedFrame]
    actions: list[list[float]] = field(default_factory=list)
    requested_initial_state_id: int = -1
    initial_state_count: int = 0

    def add(self, frame: PrivilegedFrame) -> None:
        self.frames.append(replace(frame, frame_index=len(self.frames)))


def _summarize(tracker: EpisodeEventTracker, labels, *, success: bool) -> dict[str, object]:
    contact = [bool(label.contact.gripper_target) for label in labels]
    stable = [label.stable_grasp is True for label in labels]
    target_z = np.asarray([frame.target_pose[2, 3] for frame in tracker.frames])
    initial_z = float(target_z[0])
    geometric_lift = [
        z - initial_z >= tracker.thresholds.lift_clearance_m for z in target_z
    ]
    supported_lift = [
        is_stable and z - initial_z >= tracker.thresholds.lift_clearance_m
        for is_stable, z in zip(stable, target_z, strict=True)
    ]

    def first(values: list[bool], start: int = 0) -> int | None:
        return next((i for i, value in enumerate(values[start:], start) if value), None)

    stable_onset = first(stable)
    geometric_lift_onset = first(geometric_lift)
    lift_onset = first(supported_lift)
    release_steps = [
        index for index in range(1, len(contact))
        if contact[index - 1] and not contact[index] and any(stable[:index])
    ]
    release_onset = release_steps[0] if release_steps else None
    max_lift = float(np.max(target_z - initial_z, initial=0.0))
    drop_onset = None
    if lift_onset is not None:
        for release in release_steps:
            if release <= lift_onset:
                continue
            # Last contact bounds the unobserved release between sampled frames.
            release_z = float(target_z[release - 1])
            for index in range(release, len(target_z)):
                if contact[index] or tracker.frames[index].goal_satisfied:
                    break
                if release_z - float(target_z[index]) >= tracker.drop_height_m:
                    drop_onset = index
                    break
            if drop_onset is not None:
                break
    normal_release_steps = []
    for release in release_steps:
        next_contact = next((i for i in range(release + 1, len(contact)) if contact[i]), len(contact))
        if any(frame.goal_satisfied for frame in tracker.frames[release:next_contact]):
            normal_release_steps.append(release)
    recovery_onset = None if drop_onset is None else first(stable, drop_onset + 1)

    longest = run = 0
    for value in stable:
        run = run + 1 if value else 0
        longest = max(longest, run)
    return {
        "task_id": tracker.task_id,
        "initial_state_id": tracker.initial_state_id,
        "requested_initial_state_id": tracker.requested_initial_state_id,
        "initial_state_count": tracker.initial_state_count,
        "steps": len(tracker.frames) - 1,
        "contact": any(contact),
        "contact_onset_step": first(contact),
        "stable_grasp": any(stable),
        "stable_grasp_onset_step": stable_onset,
        "geometric_lift": any(geometric_lift),
        "geometric_lift_onset_step": geometric_lift_onset,
        "supported_lift": any(supported_lift),
        "supported_lift_onset_step": lift_onset,
        "lift": any(supported_lift),
        "lift_onset_step": lift_onset,
        "max_target_lift_m": max_lift,
        "longest_stable_hold_steps": longest,
        "longest_stable_hold_s": longest / tracker.control_freq,
        "release_after_grasp": release_onset is not None,
        "release_onset_step": release_onset,
        "normal_release": bool(normal_release_steps),
        "normal_release_onset_step": normal_release_steps[0] if normal_release_steps else None,
        "unintended_drop": drop_onset is not None,
        "drop_onset_step": drop_onset,
        "recovered_after_drop": recovery_onset is not None,
        "recovery_onset_step": recovery_onset,
        "success": bool(success),
        "executed_actions": tracker.actions,
    }


def install_libero_event_recorder(
    *,
    output: Path,
    suite: str,
    initial_state_offset: int,
    initial_state_count: int | None,
    thresholds: AnnotationThresholds,
    drop_height_m: float,
) -> None:
    from lerobot.envs.libero import LiberoEnv

    if initial_state_offset < 0 or drop_height_m <= 0:
        raise ValueError("initial_state_offset must be non-negative and drop_height_m positive")
    if initial_state_count is not None and initial_state_count <= 0:
        raise ValueError("initial_state_count must be positive")
    rows: list[dict[str, object]] = []
    monitors: dict[int, tuple[LiberoOffscreenSimulator, EpisodeEventTracker, bool]] = {}
    original_reset, original_step, original_close = LiberoEnv.reset, LiberoEnv.step, LiberoEnv.close

    def save() -> None:
        write_json_atomic(
            output,
            {
                "schema": "libero_capability_events_v4",
                "drop_definition": "post-contact-loss descent from last-contact height; excludes goal satisfaction; not an intent label",
                "suite": suite,
                "initial_state_offset": initial_state_offset,
                "initial_state_count": initial_state_count,
                "thresholds": thresholds.__dict__,
                "drop_height_m": drop_height_m,
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
        state_count = len(env._init_states)
        if not getattr(env, "_capability_offset_applied", False):
            requested_initial_state_id = int(env.init_state_id) + initial_state_offset
            env.init_state_id = requested_initial_state_id
            env._capability_offset_applied = True
        else:
            requested_initial_state_id = int(env.init_state_id)
        if initial_state_count is not None and initial_state_count != state_count:
            raise ValueError(
                f"initial state count differs: contract={initial_state_count}, simulator={state_count}"
            )
        initial_state_id = int(requested_initial_state_id % state_count)
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
            requested_initial_state_id=requested_initial_state_id,
            initial_state_count=state_count,
            control_freq=int(env.control_freq),
            thresholds=thresholds,
            drop_height_m=drop_height_m,
            frames=[],
        )
        tracker.add(adapter._privileged_frame())
        monitors[id(env)] = (adapter, tracker, False)
        return result

    def step(env, action):
        monitors[id(env)][1].actions.append(np.asarray(action, dtype=float).reshape(-1).tolist())
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
    parser.add_argument("--initial-state-count", type=int)
    parser.add_argument("--stable-window-frames", type=int, default=5)
    parser.add_argument("--lift-clearance-m", type=float, default=0.01)
    parser.add_argument("--drop-height-m", type=float, default=0.02)
    parser.add_argument("--rendered-episodes", type=int, default=0)
    args, forwarded = parser.parse_known_args()
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    install_libero_event_recorder(
        output=args.events_output,
        suite=args.suite,
        initial_state_offset=args.initial_state_offset,
        initial_state_count=args.initial_state_count,
        thresholds=AnnotationThresholds(
            stable_window_frames=args.stable_window_frames,
            lift_clearance_m=args.lift_clearance_m,
        ),
        drop_height_m=args.drop_height_m,
    )
    import lerobot.scripts.lerobot_eval as lerobot_eval

    original_eval_all = lerobot_eval.eval_policy_all

    def eval_without_extra_videos(*call_args, **call_kwargs):
        call_kwargs["max_episodes_rendered"] = args.rendered_episodes
        if args.rendered_episodes == 0:
            call_kwargs["videos_dir"] = None
        return original_eval_all(*call_args, **call_kwargs)

    lerobot_eval.eval_policy_all = eval_without_extra_videos

    sys.argv = [sys.argv[0], *forwarded]
    lerobot_eval.main()


if __name__ == "__main__":
    main()

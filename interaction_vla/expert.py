from __future__ import annotations

from enum import Enum

import numpy as np

from .graph.schema import SceneSnapshot


class ExpertPhase(str, Enum):
    APPROACH = "approach"
    ALIGN = "align"
    CLOSE = "close"
    LIFT = "lift"
    TRANSPORT = "transport"
    RELEASE = "release"
    RETREAT = "retreat"


class ScriptedExpert:
    def __init__(self, max_delta: float = 0.04, tolerance: float = 0.012) -> None:
        self.max_delta = max_delta
        self.tolerance = tolerance
        self.phase = ExpertPhase.APPROACH

    def reset(self) -> None:
        self.phase = ExpertPhase.APPROACH

    def act(self, snapshot: SceneSnapshot) -> np.ndarray:
        target = snapshot.target_object
        gripper = snapshot.gripper.position
        receptacle = snapshot.receptacle.position

        recovery_action = self._resynchronize(snapshot)
        if recovery_action is not None:
            return recovery_action

        if self.phase is ExpertPhase.APPROACH:
            waypoint = target.position + np.asarray((0.0, 0.0, 0.16), dtype=np.float32)
            if self._near(gripper, waypoint):
                self.phase = ExpertPhase.ALIGN
            else:
                return self._move(gripper, waypoint, gripper_open=1.0)

        if self.phase is ExpertPhase.ALIGN:
            waypoint = target.position + np.asarray((0.0, 0.0, 0.055), dtype=np.float32)
            if self._near(gripper, waypoint):
                self.phase = ExpertPhase.CLOSE
            else:
                return self._move(gripper, waypoint, gripper_open=1.0)

        if self.phase is ExpertPhase.CLOSE:
            if snapshot.held_object == target.name:
                self.phase = ExpertPhase.LIFT
            else:
                return np.asarray((0.0, 0.0, 0.0, 0.0), dtype=np.float32)

        if self.phase is ExpertPhase.LIFT:
            waypoint = np.asarray((gripper[0], gripper[1], 0.26), dtype=np.float32)
            if gripper[2] >= 0.245:
                self.phase = ExpertPhase.TRANSPORT
            else:
                return self._move(gripper, waypoint, gripper_open=0.0)

        if self.phase is ExpertPhase.TRANSPORT:
            horizontally_aligned = bool(
                np.linalg.norm(gripper[:2] - receptacle[:2]) <= self.tolerance
            )
            waypoint = (
                receptacle + np.asarray((0.0, 0.0, 0.09), dtype=np.float32)
                if horizontally_aligned
                else np.asarray((receptacle[0], receptacle[1], 0.26), dtype=np.float32)
            )
            if self._near(gripper, waypoint):
                self.phase = ExpertPhase.RELEASE
            else:
                return self._move(gripper, waypoint, gripper_open=0.0)

        if self.phase is ExpertPhase.RELEASE:
            if snapshot.held_object is None:
                self.phase = ExpertPhase.RETREAT
            else:
                return np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float32)

        waypoint = np.asarray((gripper[0], gripper[1], 0.28), dtype=np.float32)
        return self._move(gripper, waypoint, gripper_open=1.0)

    def _resynchronize(self, snapshot: SceneSnapshot) -> np.ndarray | None:
        """Regress to a safe phase using observable state only."""

        target = snapshot.target_object
        gripper = snapshot.gripper.position
        target_held = snapshot.held_object == target.name
        over_receptacle = bool(
            np.linalg.norm(gripper[:2] - snapshot.receptacle.position[:2])
            <= self.tolerance
        )
        target_at_goal = bool(
            np.linalg.norm(target.position[:2] - snapshot.receptacle.position[:2])
            <= float(snapshot.receptacle.size[0])
        )

        if target_held:
            if self.phase in {
                ExpertPhase.APPROACH,
                ExpertPhase.ALIGN,
                ExpertPhase.CLOSE,
            }:
                self.phase = ExpertPhase.LIFT
            elif self.phase is ExpertPhase.RELEASE and not over_receptacle:
                self.phase = (
                    ExpertPhase.LIFT
                    if gripper[2] < 0.245
                    else ExpertPhase.TRANSPORT
                )
            elif (
                self.phase is ExpertPhase.TRANSPORT
                and gripper[2] < 0.245
                and not over_receptacle
            ):
                self.phase = ExpertPhase.LIFT
            elif self.phase is ExpertPhase.RETREAT:
                self.phase = ExpertPhase.TRANSPORT
            return None

        if target_at_goal and self.phase in {ExpertPhase.RELEASE, ExpertPhase.RETREAT}:
            return None

        align_waypoint = target.position + np.asarray(
            (0.0, 0.0, 0.055), dtype=np.float32
        )
        if snapshot.gripper.gripper_open < 0.5:
            self.phase = ExpertPhase.ALIGN
            return self._move(gripper, align_waypoint, gripper_open=1.0)

        if self.phase in {
            ExpertPhase.LIFT,
            ExpertPhase.TRANSPORT,
            ExpertPhase.RELEASE,
            ExpertPhase.RETREAT,
        }:
            approach_waypoint = target.position + np.asarray(
                (0.0, 0.0, 0.16), dtype=np.float32
            )
            self.phase = (
                ExpertPhase.ALIGN
                if self._near(gripper, approach_waypoint)
                else ExpertPhase.APPROACH
            )
        elif self.phase is ExpertPhase.CLOSE and not self._near(gripper, align_waypoint):
            self.phase = ExpertPhase.ALIGN
        return None

    def _move(self, current: np.ndarray, target: np.ndarray, *, gripper_open: float) -> np.ndarray:
        delta = np.clip(target - current, -self.max_delta, self.max_delta)
        return np.concatenate((delta, np.asarray((gripper_open,), dtype=np.float32))).astype(np.float32)

    def _near(self, first: np.ndarray, second: np.ndarray) -> bool:
        return bool(np.linalg.norm(first - second) <= self.tolerance)

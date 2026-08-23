from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import yaml

from .schema import FACTORS


RELOCATION_PREDICATES = {"on", "inside", "in"}


@dataclass(frozen=True)
class GoalAtom:
    predicate: str
    arguments: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.predicate or not self.arguments:
            raise ValueError("goal atom must contain a predicate and arguments")


@dataclass(frozen=True)
class TaskSemantics:
    suite: str
    task_id: int
    task_name: str
    task_family: str
    target: str
    goal: str | None
    source: str | None
    distractors: tuple[str, ...]
    goal_predicate: str
    ordered_goals: tuple[GoalAtom, ...]
    applicability: Mapping[str, bool]


def parse_goal_expression(value: object) -> tuple[GoalAtom, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ValueError("goal expression must be a non-empty sequence")
    head = str(value[0]).lower()
    if head == "and":
        return tuple(atom for item in value[1:] for atom in parse_goal_expression(item))
    arguments = tuple(str(item) for item in value[1:])
    return (GoalAtom("inside" if head == "in" else head, arguments),)


class TaskSemanticsRegistry:
    """Fail-closed task semantics for formal v1.

    Spatial and Object are known single-goal relocation suites.  Goal and Long
    are deliberately excluded until their articulation and ordered multi-goal
    plans have been reviewed.
    """

    def __init__(self, reviewed_suites: tuple[str, ...]) -> None:
        self.reviewed_suites = reviewed_suites

    @classmethod
    def default(cls) -> "TaskSemanticsRegistry":
        path = Path(__file__).with_name("task_registry_v1.yaml")
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "libero_interaction_task_registry_v1":
            raise ValueError("LIBERO task registry schema is incompatible")
        return cls(tuple(str(item) for item in payload["formal_suites"]))

    def resolve(
        self,
        *,
        suite: str,
        task_id: int,
        task_name: str,
        language: str,
        goal_atoms: tuple[GoalAtom, ...],
        ordered_plan: tuple[GoalAtom, ...] | None = None,
        source: str = "table",
        distractors: tuple[str, ...] = (),
    ) -> TaskSemantics:
        del language
        if suite not in self.reviewed_suites:
            raise ValueError(f"suite {suite} task semantics are not reviewed")
        if not goal_atoms:
            raise ValueError("task has no goal atoms")
        if len(goal_atoms) > 1 and ordered_plan is None:
            raise ValueError("multi-goal task requires a reviewed ordered task plan")
        plan = ordered_plan or goal_atoms
        if sorted(plan, key=repr) != sorted(goal_atoms, key=repr):
            raise ValueError("ordered task plan must contain exactly the BDDL goal atoms")
        first = plan[0]
        if first.predicate not in RELOCATION_PREDICATES or len(first.arguments) != 2:
            raise ValueError(
                f"task {task_name} is not a reviewed single-goal relocation task"
            )
        target, goal = first.arguments
        return TaskSemantics(
            suite=suite,
            task_id=int(task_id),
            task_name=task_name,
            task_family="relocation",
            target=target,
            goal=goal,
            source=source,
            distractors=tuple(sorted(set(distractors).difference({target, goal, source}))),
            goal_predicate="inside" if first.predicate == "in" else first.predicate,
            ordered_goals=plan,
            applicability={factor: True for factor in FACTORS},
        )

    def coverage(self, tasks: Sequence[tuple[str, int]]) -> dict[str, object]:
        supported = [item for item in tasks if item[0] in self.reviewed_suites]
        unsupported = [item for item in tasks if item[0] not in self.reviewed_suites]
        return {
            "reviewed_suites": list(self.reviewed_suites),
            "supported_tasks": len(supported),
            "unsupported_tasks": [
                {"suite": suite, "task_id": task_id, "reason": "semantics_not_reviewed"}
                for suite, task_id in unsupported
            ],
        }

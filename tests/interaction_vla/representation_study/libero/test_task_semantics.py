import pytest

from interaction_vla.representation_study.libero.task_semantics import (
    GoalAtom,
    TaskSemanticsRegistry,
    parse_goal_expression,
)


def test_single_goal_relocation_semantics_are_supported() -> None:
    registry = TaskSemanticsRegistry.default()
    semantics = registry.resolve(
        suite="libero_spatial",
        task_id=3,
        task_name="pick_up_the_black_bowl_and_place_it_on_the_plate",
        language="pick up the black bowl and place it on the plate",
        goal_atoms=(GoalAtom("on", ("black_bowl", "plate")),),
        distractors=("white_bowl", "black_bowl"),
    )
    assert semantics.task_family == "relocation"
    assert semantics.target == "black_bowl"
    assert semantics.goal == "plate"
    assert semantics.goal_predicate == "on"
    assert semantics.distractors == ("white_bowl",)
    assert semantics.ordered_goals == (GoalAtom("on", ("black_bowl", "plate")),)
    assert semantics.applicability["stable_grasp"]


def test_unreviewed_goal_suite_fails_closed() -> None:
    registry = TaskSemanticsRegistry.default()
    with pytest.raises(ValueError, match="not reviewed"):
        registry.resolve(
            suite="libero_goal",
            task_id=2,
            task_name="open_the_top_drawer",
            language="open the top drawer",
            goal_atoms=(GoalAtom("open", ("top_drawer",)),),
        )


def test_unordered_multi_goal_fails_instead_of_inventing_intention() -> None:
    registry = TaskSemanticsRegistry.default()
    with pytest.raises(ValueError, match="ordered task plan"):
        registry.resolve(
            suite="libero_spatial",
            task_id=99,
            task_name="two_goal_fixture",
            language="put both bowls on the plate",
            goal_atoms=(
                GoalAtom("on", ("black_bowl", "plate")),
                GoalAtom("on", ("white_bowl", "plate")),
            ),
        )


def test_goal_parser_normalizes_single_atom_and_conjunction() -> None:
    assert parse_goal_expression(["on", "black_bowl", "plate"]) == (
        GoalAtom("on", ("black_bowl", "plate")),
    )
    assert parse_goal_expression(
        ["and", ["on", "black_bowl", "plate"], ["inside", "mug", "drawer"]]
    ) == (
        GoalAtom("on", ("black_bowl", "plate")),
        GoalAtom("inside", ("mug", "drawer")),
    )

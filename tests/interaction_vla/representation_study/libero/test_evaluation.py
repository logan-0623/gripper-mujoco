from interaction_vla.representation_study.libero.evaluation import (
    PairedRollout,
    paired_outcome_report,
)


def _row(case: int, original: bool, intervened: bool) -> PairedRollout:
    return PairedRollout(
        case_id=f"case-{case}",
        suite="libero_spatial",
        task_id=case // 2,
        source_episode_id=f"episode-{case}",
        initial_state_sha256=(f"{case:064x}")[-64:],
        inference_seed=100 + case,
        original_success=original,
        intervened_success=intervened,
        original_steps=100,
        intervened_steps=110,
        original_failure=None if original else "placement_failure",
        intervened_failure=None if intervened else "grasp_failure",
        action_delta_translation=0.1,
        action_delta_rotation=0.2,
        action_delta_gripper=0.3,
    )


def test_paired_report_keeps_action_sensitivity_and_closed_loop_utility_distinct() -> None:
    rows = (_row(0, True, False), _row(1, True, True), _row(2, False, False), _row(3, False, True))
    report = paired_outcome_report(rows, bootstrap_samples=200, seed=3)
    assert report["action_sensitive"]
    assert report["mean_action_delta"]["translation"] == 0.1
    assert report["original_success_rate"] == 0.5
    assert report["intervened_success_rate"] == 0.5
    assert report["delta_success"] == 0.0
    assert report["useful_for_closed_loop"] is False
    assert report["bootstrap_unit"] == "task"

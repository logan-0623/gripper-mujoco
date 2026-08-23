import os
from pathlib import Path

import pytest

from interaction_vla.representation_study.libero.replay import (
    load_raw_replay_episode,
    replay_episode,
)
from interaction_vla.representation_study.libero.runtime import LiberoOffscreenSimulator


@pytest.mark.skipif(
    os.environ.get("LIBERO_INTEGRATION_DEMO") is None,
    reason="set LIBERO_INTEGRATION_DEMO to a raw HDF5 file on a Linux LIBERO host",
)
def test_one_real_libero_episode_replays_with_registered_tolerance() -> None:
    pytest.importorskip("libero")
    demo_path = Path(os.environ["LIBERO_INTEGRATION_DEMO"])
    bddl_path = Path(os.environ["LIBERO_INTEGRATION_BDDL"])
    suite = os.environ.get("LIBERO_INTEGRATION_SUITE", "libero_spatial")
    task_id = int(os.environ.get("LIBERO_INTEGRATION_TASK_ID", "0"))
    task_name = os.environ["LIBERO_INTEGRATION_TASK_NAME"]
    language = os.environ["LIBERO_INTEGRATION_LANGUAGE"]
    demo_key = os.environ.get("LIBERO_INTEGRATION_DEMO_KEY", "demo_0")
    episode = load_raw_replay_episode(
        demo_path, suite=suite, task_id=task_id, demo_key=demo_key
    )
    simulator = LiberoOffscreenSimulator(
        suite=suite,
        task_id=task_id,
        task_name=task_name,
        language=language,
        bddl_path=bddl_path,
        seed=0,
        control_freq=20,
    )
    try:
        report = replay_episode(
            episode,
            simulator,
            action_atol=1e-5,
            state_l2_p95_tolerance=0.01,
            state_max_abs_tolerance=0.05,
        )
    finally:
        simulator.close()
    assert report.passed
    assert report.frames
    assert "_privileged_frame" in report.frames[0].observation

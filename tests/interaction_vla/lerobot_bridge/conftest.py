from pathlib import Path

import numpy as np
import pytest

from interaction_vla.lerobot_bridge.dataset_writer import LeRobotEpisodeWriter


@pytest.fixture
def tiny_lerobot_dataset(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "tiny_lerobot"
    repo_id = "local/tiny_lerobot"
    writer = LeRobotEpisodeWriter.create(
        repo_id=repo_id,
        root=root,
        fps=20,
        width=256,
        height=256,
    )
    for index in range(9):
        writer.add_frame(
            agent_rgb=np.full((256, 256, 3), index, dtype=np.uint8),
            wrist_rgb=np.full((256, 256, 3), 2 * index, dtype=np.uint8),
            state=np.full(10, 0.01 * index, dtype=np.float32),
            action=np.zeros(7, dtype=np.float32),
            task="Pick up the green target object and place it inside the receptacle.",
        )
    writer.save_episode()
    writer.finalize()
    return root, repo_id

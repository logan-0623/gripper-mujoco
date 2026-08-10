import numpy as np
import pytest

pytest.importorskip("lerobot")
from lerobot.datasets import LeRobotDataset

from interaction_vla.lerobot_bridge.dataset_writer import LeRobotEpisodeWriter


def test_two_frame_dataset_loads_through_standard_lerobot_api(tmp_path) -> None:
    root = tmp_path / "dataset"
    writer = LeRobotEpisodeWriter.create(
        repo_id="local/test_dual_view",
        root=root,
        fps=20,
        width=256,
        height=256,
    )
    for value in (0, 1):
        writer.add_frame(
            agent_rgb=np.full((256, 256, 3), value, dtype=np.uint8),
            wrist_rgb=np.full((256, 256, 3), value + 1, dtype=np.uint8),
            state=np.zeros(10, dtype=np.float32),
            action=np.zeros(7, dtype=np.float32),
            task="Pick up the green target object and place it inside the receptacle.",
        )
    writer.save_episode()
    writer.finalize()
    writer.finalize()

    assert (root / "data").is_dir()
    assert (root / "videos").is_dir()
    assert (root / "meta").is_dir()
    with pytest.raises(RuntimeError, match="finalization"):
        writer.save_episode()
    with pytest.raises(FileExistsError, match="new dataset root"):
        LeRobotEpisodeWriter.create(
            repo_id="local/test_dual_view",
            root=root,
            fps=20,
            width=256,
            height=256,
        )

    dataset = LeRobotDataset("local/test_dual_view", root=root)
    sample = dataset[0]
    assert len(dataset) == 2
    assert tuple(sample["observation.images.agent"].shape) == (3, 256, 256)
    assert tuple(sample["observation.images.wrist"].shape) == (3, 256, 256)
    assert tuple(sample["observation.state"].shape) == (10,)
    assert tuple(sample["action"].shape) == (7,)
    assert dataset.meta.tasks.index.tolist() == [
        "Pick up the green target object and place it inside the receptacle."
    ]

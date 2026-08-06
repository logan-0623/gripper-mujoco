from pathlib import Path

import mujoco
import numpy as np


def test_franka_scene_contains_robot_fingers_objects_and_cameras() -> None:
    from interaction_vla.franka import FRANKA_SCENE_PATH

    model = mujoco.MjModel.from_xml_path(str(FRANKA_SCENE_PATH))
    assert [model.joint(f"joint{index}").id for index in range(1, 8)]
    assert model.joint("finger_joint1").id >= 0
    assert model.joint("finger_joint2").id >= 0
    assert model.body("hand").id >= 0
    for index in range(5):
        assert model.joint(f"object_{index}_joint").id >= 0
        assert model.geom(f"object_{index}_geom").id >= 0
    for camera in ("agentview", "wristview", "sideview", "topview"):
        assert model.camera(camera).id >= 0
    assert model.vis.global_.offwidth >= 640
    assert model.vis.global_.offheight >= 480


def test_franka_scene_is_a_500hz_gravity_contact_model() -> None:
    from interaction_vla.franka import FRANKA_SCENE_PATH

    model = mujoco.MjModel.from_xml_path(str(FRANKA_SCENE_PATH))

    assert model.opt.timestep == 0.002
    np.testing.assert_allclose(model.opt.gravity, (0.0, 0.0, -9.81))
    assert model.nu == 8
    assert model.geom("table").contype != 0
    assert model.body("receptacle").id >= 0
    for index in range(5):
        joint = model.joint(f"object_{index}_joint")
        assert model.jnt_type[joint.id] == mujoco.mjtJoint.mjJNT_FREE
        assert model.geom(f"object_{index}_geom").contype != 0


def test_scene_has_no_object_attachment_or_object_mocap() -> None:
    from interaction_vla.franka import FRANKA_SCENE_PATH

    model = mujoco.MjModel.from_xml_path(str(FRANKA_SCENE_PATH))

    assert model.neq == 1  # Upstream finger-joint coupling only.
    for index in range(5):
        body = model.body(f"object_{index}")
        assert model.body_mocapid[body.id] == -1


def test_franka_asset_provenance_and_license_are_preserved() -> None:
    from interaction_vla.franka import FRANKA_ASSET_ROOT, FRANKA_COMMIT

    text = (FRANKA_ASSET_ROOT / "README.md").read_text()
    assert FRANKA_COMMIT == "71f066ad0be9cd271f7ed58c030243ef157af9f4"
    assert "71f066ad0be9cd271f7ed58c030243ef157af9f4" in text
    assert "google-deepmind/mujoco_menagerie" in text
    license_path = FRANKA_ASSET_ROOT / "franka_emika_panda" / "LICENSE"
    assert license_path.is_file()
    assert "Apache License" in license_path.read_text()

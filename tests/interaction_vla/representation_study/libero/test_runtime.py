from pathlib import Path
import xml.etree.ElementTree as ET

from interaction_vla.representation_study.libero.runtime import (
    LiberoOffscreenSimulator,
    _model_id2name,
    _model_name2id,
)


class _Named:
    def __init__(self, identifier: int, name: str) -> None:
        self.id = identifier
        self.name = name


class _ModernModel:
    def geom(self, value):
        if isinstance(value, str):
            return _Named(4, value)
        return _Named(int(value), "target_geom")


class _LegacyModel:
    def geom_name2id(self, name: str) -> int:
        assert name == "target_geom"
        return 7

    def geom_id2name(self, identifier: int) -> str:
        assert identifier == 7
        return "target_geom"


def test_mujoco_name_lookup_supports_modern_and_legacy_wrappers() -> None:
    assert _model_name2id(_ModernModel(), "geom", "target_geom") == 4
    assert _model_id2name(_ModernModel(), "geom", 4) == "target_geom"
    assert _model_name2id(_LegacyModel(), "geom", "target_geom") == 7
    assert _model_id2name(_LegacyModel(), "geom", 7) == "target_geom"


def test_runtime_relocates_recorded_assets_before_mujoco_reset(tmp_path: Path) -> None:
    robot_root = tmp_path / "robosuite"
    assets_root = tmp_path / "libero-assets"
    mesh = robot_root / "models/assets/robots/panda/meshes/link0.stl"
    texture = assets_root / "textures/object.png"
    mesh.parent.mkdir(parents=True)
    texture.parent.mkdir(parents=True)
    mesh.touch()
    texture.touch()

    class FakeSim:
        def reset(self) -> None:
            pass

    class FakeEnv:
        def __init__(self) -> None:
            self.sim = FakeSim()
            self.xml = ""

        def reset_from_xml_string(self, xml: str) -> None:
            self.xml = xml

    simulator = object.__new__(LiberoOffscreenSimulator)
    simulator.env = FakeEnv()
    simulator._robosuite_root = robot_root
    simulator._libero_assets_root = assets_root
    simulator.reset_from_xml_string(
        """
        <mujoco><asset>
          <mesh file="/old/robosuite/robosuite/models/assets/robots/panda/meshes/link0.stl"/>
          <texture file="/old/libero/chiliocosm/assets/textures/object.png"/>
        </asset></mujoco>
        """
    )

    files = [
        element.get("file")
        for element in ET.fromstring(simulator.env.xml).iter()
        if element.get("file")
    ]
    assert files == [str(mesh.resolve()), str(texture.resolve())]

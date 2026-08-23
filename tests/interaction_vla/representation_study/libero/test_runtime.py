from interaction_vla.representation_study.libero.runtime import (
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

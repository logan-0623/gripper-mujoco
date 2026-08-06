from __future__ import annotations

import numpy as np
import pytest

from interaction_vla.placement import strict_containment


def test_strict_containment_requires_the_complete_object_inside() -> None:
    rotation = np.eye(3)
    inside = strict_containment(
        local_center=np.asarray((0.030, 0.0, 0.028)),
        relative_rotation=rotation,
        object_half_extents=np.asarray((0.022, 0.022, 0.022)),
        inner_half_extents=np.asarray((0.055, 0.055)),
    )
    outside = strict_containment(
        local_center=np.asarray((0.040, 0.0, 0.028)),
        relative_rotation=rotation,
        object_half_extents=np.asarray((0.022, 0.022, 0.022)),
        inner_half_extents=np.asarray((0.055, 0.055)),
    )

    assert inside.fully_contained
    assert inside.containment_margin.min() == pytest.approx(0.003)
    assert not outside.fully_contained
    assert outside.containment_margin[0] < 0.0


def test_rotated_object_uses_projected_half_extents() -> None:
    angle = np.deg2rad(45.0)
    rotation = np.asarray(
        (
            (np.cos(angle), -np.sin(angle), 0.0),
            (np.sin(angle), np.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    result = strict_containment(
        local_center=np.asarray((0.025, 0.0, 0.028)),
        relative_rotation=rotation,
        object_half_extents=np.asarray((0.022, 0.022, 0.022)),
        inner_half_extents=np.asarray((0.055, 0.055)),
    )

    assert result.projected_half_extents[0] == pytest.approx(0.0311127)
    assert not result.fully_contained


@pytest.mark.parametrize(
    "kwargs",
    [
        {"local_center": np.zeros(2)},
        {"relative_rotation": np.eye(2)},
        {"object_half_extents": np.zeros(3)},
        {"inner_half_extents": np.asarray((0.055, np.nan))},
    ],
)
def test_strict_containment_rejects_invalid_geometry(kwargs) -> None:
    values = {
        "local_center": np.zeros(3),
        "relative_rotation": np.eye(3),
        "object_half_extents": np.full(3, 0.022),
        "inner_half_extents": np.full(2, 0.055),
    }
    values.update(kwargs)

    with pytest.raises(ValueError):
        strict_containment(**values)

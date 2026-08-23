from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from .schema import ContactLabels


@dataclass(frozen=True)
class SemanticContactMap:
    target_geoms: frozenset[str]
    goal_geoms: frozenset[str]
    source_geoms: frozenset[str]
    finger_geoms: Mapping[str, frozenset[str]]


def contact_labels_from_pairs(
    pairs: Iterable[tuple[str, str]], mapping: SemanticContactMap
) -> ContactLabels:
    normalized = {frozenset((str(left), str(right))) for left, right in pairs}

    def any_cross(left: frozenset[str], right: frozenset[str]) -> bool:
        return any(frozenset((one, two)) in normalized for one in left for two in right)

    finger_groups = tuple(
        sorted(
            name
            for name, geoms in mapping.finger_geoms.items()
            if any_cross(geoms, mapping.target_geoms)
        )
    )
    all_fingers = frozenset(
        geom for geoms in mapping.finger_geoms.values() for geom in geoms
    )
    return ContactLabels(
        gripper_target=any_cross(all_fingers, mapping.target_geoms),
        target_goal=any_cross(mapping.target_geoms, mapping.goal_geoms),
        target_source=any_cross(mapping.target_geoms, mapping.source_geoms),
        finger_groups=finger_groups,
    )

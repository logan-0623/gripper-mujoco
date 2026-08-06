from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import mujoco


@dataclass(frozen=True)
class ContainmentResult:
    local_center: np.ndarray
    projected_half_extents: np.ndarray
    containment_margin: np.ndarray
    fully_contained: bool


@dataclass(frozen=True)
class PlacementDiagnostics:
    local_center: np.ndarray
    projected_half_extents: np.ndarray
    containment_margin: np.ndarray
    fully_contained: bool
    base_contact: bool
    wall_contact: bool
    wall_only_contact: bool
    stable_frames: int
    strict_stable: bool


def strict_containment(
    *,
    local_center: np.ndarray,
    relative_rotation: np.ndarray,
    object_half_extents: np.ndarray,
    inner_half_extents: np.ndarray,
) -> ContainmentResult:
    center = np.asarray(local_center, dtype=np.float64)
    rotation = np.asarray(relative_rotation, dtype=np.float64)
    object_half = np.asarray(object_half_extents, dtype=np.float64)
    inner_half = np.asarray(inner_half_extents, dtype=np.float64)
    if center.shape != (3,) or rotation.shape != (3, 3):
        raise ValueError("placement pose must be a 3D centre and 3x3 rotation")
    if object_half.shape != (3,) or inner_half.shape != (2,):
        raise ValueError("placement extents must have shapes (3,) and (2,)")
    if not all(
        np.isfinite(value).all()
        for value in (center, rotation, object_half, inner_half)
    ):
        raise ValueError("placement geometry must be finite")
    if np.any(object_half <= 0.0) or np.any(inner_half <= 0.0):
        raise ValueError("placement extents must be positive")
    projected = np.abs(rotation[:2]) @ object_half
    margin = inner_half - np.abs(center[:2]) - projected
    return ContainmentResult(
        local_center=center.copy(),
        projected_half_extents=projected,
        containment_margin=margin,
        fully_contained=bool(np.all(margin >= 0.0)),
    )


def receptacle_inner_half_extents(model: mujoco.MjModel) -> np.ndarray:
    positive_x = model.geom("receptacle_wall_pos_x")
    negative_x = model.geom("receptacle_wall_neg_x")
    positive_y = model.geom("receptacle_wall_pos_y")
    negative_y = model.geom("receptacle_wall_neg_y")
    x_faces = (
        float(positive_x.pos[0] - positive_x.size[0]),
        float(-negative_x.pos[0] - negative_x.size[0]),
    )
    y_faces = (
        float(positive_y.pos[1] - positive_y.size[1]),
        float(-negative_y.pos[1] - negative_y.size[1]),
    )
    result = np.asarray((min(x_faces), min(y_faces)), dtype=np.float64)
    if not np.isfinite(result).all() or np.any(result <= 0.0):
        raise ValueError("receptacle inner wall faces must define positive extents")
    return result

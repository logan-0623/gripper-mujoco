"""Residual online adaptation for the representation study."""

from .core import ResidualActorCritic, generalized_advantage_estimate, normalized_curve_auc

__all__ = ["ResidualActorCritic", "generalized_advantage_estimate", "normalized_curve_auc"]
from .core import ResidualActorCritic

__all__ = ["ResidualActorCritic"]

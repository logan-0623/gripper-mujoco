"""LIBERO interaction-representation study.

This package is intentionally separate from the legacy Franka State Bank.  It
binds standard LeRobot observations to original LIBERO simulator trajectories.
"""

from .config import LIBERO_CONFIG_SCHEMA, LiberoStudyConfig, load_libero_study_config

__all__ = ["LIBERO_CONFIG_SCHEMA", "LiberoStudyConfig", "load_libero_study_config"]

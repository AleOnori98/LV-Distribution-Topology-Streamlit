from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent


class PathManager:
    """
    Simple centralised path helper.

    Usage:
        from config.settings import PathManager
        PathManager.ASSETS / "distribution_methodology.png"
    """

    ROOT = ROOT_DIR
    ASSETS = ROOT / "config" / "assets"
    EXAMPLES = ROOT / "examples"


# ---------------------------------------------------------------------
# App defaults
# ---------------------------------------------------------------------

# Target projected CRS for all calculations (UTM 33N as placeholder)
TARGET_CRS = 32633

# Cost defaults
DEFAULT_COST_PER_KM_LV = 3000.0      # USD/km
DEFAULT_FIXED_COSTS_LV = 0.0         # USD

# Heuristic defaults
DEFAULT_SAMPLING_DISTANCE_M = 40     # m between candidate poles
DEFAULT_USER_DISTANCE_M = 35         # max distance user–pole
DEFAULT_MAX_ASSOCIATIONS = 16        # max users per pole

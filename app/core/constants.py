# app/core/constants.py
# -*- coding: utf-8 -*-
"""
Astro v11 — Core constants & small helpers

Purpose
-------
Single source of truth for:
- planet sets & names
- aspect angles and default orbs (per module family)
- heuristic aspect weights (scoring)
- synastry/parallel/antiscia defaults
- time constants (year/months/naibod)
- house system aliases (stable canonical keys)
- tiny angle helpers (wrap/Δ/antiscia)

Design
------
- Pure-Python, no external dependencies.
- Safe to import from any core module.
- Functions are pure; constants are immutable by convention.

Notes
-----
- Ayanāṃśa presets are kept in `astronomy.py`; this module only stores
  numeric defaults consumed by callers (e.g., default parallel orb).
"""

from __future__ import annotations
from typing import Dict, Tuple
import math

__all__ = [
    # planet sets
    "MAJOR_BODIES", "ANGLE_POINTS", "SUPPORTED_BODIES",
    # aspects
    "ASPECT_ANGLES_DEG", "DEFAULT_ORBS_SYNASTRY", "DEFAULT_ORBS_PROGRESSIONS",
    "DEFAULT_ORBS_DIRECTIONS", "PARALLEL_DEFAULT_ARCMIN", "ANTISCIA_DEFAULT_ORB_DEG",
    "ASPECT_WEIGHTS",
    # time constants
    "TROPICAL_YEAR_D", "SOLAR_YEAR_D", "LUNAR_SYNODIC_D", "LUNAR_SIDEREAL_D",
    "LUNAR_DAY_D", "NAIBOD_DEG_PER_YEAR",
    # houses
    "HOUSE_SYSTEM_ALIASES",
    # helpers
    "wrap_deg", "delta_deg", "abs_sep_deg", "antiscia_of",
    # version tag
    "V11_CONSTANTS_VERSION",
]

# ── version tag ───────────────────────────────────────────────────────────────
V11_CONSTANTS_VERSION: str = "v11.0.0"

# ── canonical bodies ─────────────────────────────────────────────────────────
MAJOR_BODIES: Tuple[str, ...] = (
    "Sun", "Moon", "Mercury", "Venus", "Mars",
    "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto",
)

# Angle points routinely used across modules (houses/angles).
ANGLE_POINTS: Tuple[str, ...] = ("ASC", "MC", "IC", "DSC")

# Expanded set placeholder (keep ordering stable if extended)
SUPPORTED_BODIES: Tuple[str, ...] = MAJOR_BODIES  # extend here if you add nodes/asteroids

# ── aspect geometry ──────────────────────────────────────────────────────────
# Only include angles currently used by core modules to avoid accidental drift.
ASPECT_ANGLES_DEG: Dict[str, float] = {
    "conjunction": 0.0,
    "sextile": 60.0,
    "square": 90.0,
    "trine": 120.0,
    "opposition": 180.0,
    "quincunx": 150.0,  # included in synastry/progressions/directions
    # add more as needed (semisextile=30, semisquare=45, etc.) after aligning orbs/weights
}

# Synastry-style defaults: generous luminary orbs, tighter minors.
DEFAULT_ORBS_SYNASTRY: Dict[str, float] = {
    "conjunction": 8.0,
    "opposition": 6.0,
    "trine": 6.0,
    "square": 5.0,
    "sextile": 3.0,
    "quincunx": 2.0,
    "antiscia": 2.0,                 # for antiscia/contra-antiscia
    "parallel_arcmin": 40.0,         # declination parallels (±0°40′)
}

# Progressions reuse synastry defaults unless overridden by caller.
DEFAULT_ORBS_PROGRESSIONS: Dict[str, float] = dict(DEFAULT_ORBS_SYNASTRY)

# Directions (solar-arc) often use tighter orbs.
DEFAULT_ORBS_DIRECTIONS: Dict[str, float] = {
    "conjunction": 1.5,
    "opposition": 1.5,
    "trine": 1.0,
    "square": 1.0,
    "sextile": 0.75,
    "quincunx": 0.5,
    "antiscia": 0.5,
    # parallels usually not used in classic directions; add if desired.
}

PARALLEL_DEFAULT_ARCMIN: float = 40.0
ANTISCIA_DEFAULT_ORB_DEG: float = 2.0

# Heuristic weights for simple scoring/aggregation (dimensionless).
# Positive aspects > 0, challenging < 0; conjunction context-dependent so mildly positive here.
ASPECT_WEIGHTS: Dict[str, float] = {
    "conjunction": 0.9,
    "sextile": 0.7,
    "trine": 1.0,
    "square": -0.9,
    "opposition": -1.0,
    "quincunx": -0.4,
    # if you add minor aspects, define their weights explicitly.
}

# ── time constants ────────────────────────────────────────────────────────────
TROPICAL_YEAR_D: float = 365.242189  # mean tropical year
SOLAR_YEAR_D: float   = TROPICAL_YEAR_D  # alias used in returns
LUNAR_SYNODIC_D: float = 29.530588
LUNAR_SIDEREAL_D: float = 27.321582
LUNAR_DAY_D: float      = 1.03502      # ≈ 24h50m28s (mean)
NAIBOD_DEG_PER_YEAR: float = 360.0 / TROPICAL_YEAR_D  # ≈ 0.985647 deg/yr

# ── houses: canonical keys & common aliases ───────────────────────────────────
# Use these keys uniformly across the codebase; map user inputs to them.
HOUSE_SYSTEM_ALIASES: Dict[str, str] = {
    # canonical → self
    "placidus": "placidus",
    "regiomontanus": "regiomontanus",
    "koch": "koch",
    "campanus": "campanus",
    "equal": "equal",
    "whole-sign": "whole-sign",
    "porphyry": "porphyry",
    "alcabitius": "alcabitius",
    "topocentric": "topocentric",
    "morinus": "morinus",
    "axial-rotation": "axial-rotation",
    # aliases → canonical
    "pl": "placidus",
    "plac": "placidus",
    "regio": "regiomontanus",
    "rq": "regiomontanus",
    "koch-house": "koch",
    "equal-house": "equal",
    "wholesign": "whole-sign",
    "ws": "whole-sign",
    "porph": "porphyry",
    "alcab": "alcabitius",
    "topoc": "topocentric",
    "mor": "morinus",
    "axial": "axial-rotation",
}

# ── tiny angle helpers (no external imports) ──────────────────────────────────
def wrap_deg(x: float) -> float:
    """
    Wrap any angle to [0, 360).
    """
    x = math.fmod(float(x), 360.0)
    return x + 360.0 if x < 0.0 else x

def delta_deg(a: float, b: float) -> float:
    """
    Shortest signed difference b - a in degrees, range (-180, 180].
    Useful for aspect separations and root-finding residuals.
    """
    d = wrap_deg(b) - wrap_deg(a)
    if d > 180.0:
        d -= 360.0
    elif d <= -180.0:
        d += 360.0
    return d

def abs_sep_deg(a: float, b: float) -> float:
    """
    Absolute smallest separation between angles a and b (deg, 0..180].
    """
    return abs(delta_deg(a, b))

def antiscia_of(lon_deg: float) -> float:
    """
    Mirror a longitude across the solstitial axis (0° Cancer).
    antiscia(λ) = 180° − λ (wrapped to [0, 360)).
    """
    return wrap_deg(180.0 - wrap_deg(lon_deg))

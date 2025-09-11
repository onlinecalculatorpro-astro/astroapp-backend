# app/core/ayanamsa.py
# -*- coding: utf-8 -*-
"""
Ayanāṁśa (sidereal–tropical offset) — GOLD-READY CORE (v1.2)

This module provides an auditable, fast, and deterministic ayanāṁśa engine
that integrates directly with astronomy.py via `get_ayanamsa_deg(jd_tt, scheme)`.

Design goals
------------
- Exact compatibility with astronomy.py dynamic import and fallback behavior.
- Research-friendly: single, clear model = base@J2000 + linear drift.
- Deterministic, dependency-free, and cached (LRU).
- Extensible: runtime registration/override of schemes.

Public API
----------
get_ayanamsa_deg(jd_tt: float, scheme: str = "lahiri") -> float
ayanamsa_info(jd_tt: float, scheme: str = "lahiri") -> dict
list_ayanamsa_schemes() -> list[str]
register_ayanamsa(name: str, base_at_j2000_deg: float) -> None

Notes
-----
- Input time is TT Julian Day (astronomy.py supplies this).
- Drift model matches astronomy.py fallback:
    Lahiri base @ J2000: 23°51′26.26″
    Linear rate: 50.290966 arcsec per tropical year
- Aliases: "chitrapaksha", "default", "sidereal" → "lahiri";
           "fagan", "fagan/bradley" → "fagan_bradley";
           "kp" → "krishnamurti".
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional

# ---------------------------------------------------------------------
# Core constants (kept consistent with astronomy.py fallback)
# ---------------------------------------------------------------------

J2000_JD_TT: float = 2451545.0

# Lahiri base used by astronomy.py fallback:
# 23° 51' 26.26" = 23 + 51/60 + 26.26/3600
_LAHIRI_BASE_J2000_DEG = 23.0 + 51.0 / 60.0 + 26.26 / 3600.0  # ≈ 23.857294444444445

# Linear precession rate used by astronomy.py fallback (arcsec per year)
_RATE_ARCSEC_PER_YEAR = 50.290966

def _normalize_deg(x: float) -> float:
    r = float(x) % 360.0
    return r if r >= 0.0 else r + 360.0


@dataclass(frozen=True)
class _Scheme:
    name: str
    base_at_j2000_deg: float


# ---------------------------------------------------------------------
# Registry
# Bases chosen to be coherent with your fallback adjustments:
#   • fagan_bradley = lahiri + 0.83 arcmin = +0.013833333... deg
#   • krishnamurti  = lahiri − 20 arcsec   = −0.005555555... deg
# You can add/override via register_ayanamsa().
# ---------------------------------------------------------------------

_SCHEMES: Dict[str, _Scheme] = {
    "lahiri":        _Scheme("lahiri", _LAHIRI_BASE_J2000_DEG),
    "fagan_bradley": _Scheme("fagan_bradley", _LAHIRI_BASE_J2000_DEG + (0.83 / 60.0)),
    "krishnamurti":  _Scheme("krishnamurti",  _LAHIRI_BASE_J2000_DEG - (20.0 / 3600.0)),
    # Optional classics (literature approximations; easy to override if needed)
    "raman":         _Scheme("raman",        22.5060000000000),
    "yukteswar":     _Scheme("yukteswar",    23.5000000000000),
    "devore":        _Scheme("devore",       24.2833000000000),
}

# Aliases accepted by astronomy.py and common usage
_ALIASES: Dict[str, str] = {
    "chitrapaksha": "lahiri",
    "default":      "lahiri",
    "sidereal":     "lahiri",
    "fagan":        "fagan_bradley",
    "fagan/bradley":"fagan_bradley",
    "fagan_bradley":"fagan_bradley",
    "kp":           "krishnamurti",
}


# ---------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------

def _years_since_j2000(jd_tt: float) -> float:
    """Tropical years from J2000.0 (TT)."""
    # Using 365.25 keeps this consistent with astronomy.py fallback usage.
    return (float(jd_tt) - J2000_JD_TT) / 365.25


@lru_cache(maxsize=4096)
def _drift_deg_from_j2000(jd_tt: float) -> float:
    """
    Linear precession drift (degrees) from J2000 to jd_tt.

    astronomy.py fallback uses a single linear rate:
        50.290966 arcsec/year
    """
    years = _years_since_j2000(jd_tt)
    return (_RATE_ARCSEC_PER_YEAR * years) / 3600.0


def _resolve_scheme(scheme: Optional[str]) -> str:
    if not scheme:
        return "lahiri"
    key = str(scheme).strip().lower()
    if key in _SCHEMES:
        return key
    if key in _ALIASES:
        return _ALIASES[key]
    # graceful default for unknown keys: match astronomy fallback behavior
    # (astronomy warns via _W.AYA_FALLBACK; we just resolve to Lahiri here)
    return "lahiri"


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def list_ayanamsa_schemes() -> List[str]:
    """Return canonical scheme names currently available."""
    return sorted(_SCHEMES.keys())


def register_ayanamsa(name: str, base_at_j2000_deg: float) -> None:
    """
    Register or override an ayanāṁśa scheme at runtime.

    Parameters
    ----------
    name : str
        Canonical lowercase identifier (e.g., "lahiri_precise", "se_lahiri").
    base_at_j2000_deg : float
        Ayanāṁśa value in degrees at J2000.0 (TT).
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Scheme name must be a non-empty string.")
    key = name.strip().lower()
    object.__setattr if False else None  # placate linters for frozen dataclass pattern
    _SCHEMES[key] = _Scheme(key, float(base_at_j2000_deg))


@lru_cache(maxsize=8192)
def get_ayanamsa_deg(jd_tt: float, scheme: str = "lahiri") -> float:
    """
    Compute ayanāṁśa (degrees in [0,360)) for a given TT Julian Day and scheme.

    This is the exact function astronomy.py tries to import and call.

    Parameters
    ----------
    jd_tt : float
        TT Julian Day (Terrestrial Time).
    scheme : str
        One of list_ayanamsa_schemes(), or a known alias.

    Returns
    -------
    float
        Ayanāṁśa in degrees (wrapped to [0, 360)).
    """
    if not isinstance(jd_tt, (int, float)):
        raise ValueError("jd_tt must be a number (TT Julian Day).")
    key = _resolve_scheme(scheme)
    base = _SCHEMES[key].base_at_j2000_deg
    drift = _drift_deg_from_j2000(float(jd_tt))
    return _normalize_deg(base + drift)


def ayanamsa_info(jd_tt: float, scheme: str = "lahiri") -> Dict[str, object]:
    """
    Verbose ayanāṁśa result with components and resolution info.

    Returns
    -------
    dict with keys:
      - ok: bool
      - scheme: str (canonical)
      - ayanamsa_deg: float
      - components:
          - base_at_j2000_deg
          - drift_deg_from_j2000
          - years_since_j2000
      - model:
          - type: "linear_rate"
          - rate_arcsec_per_year
      - resolved_from: str (input scheme or alias)
    """
    resolved = _resolve_scheme(scheme)
    yrs = _years_since_j2000(float(jd_tt))
    drift = _drift_deg_from_j2000(float(jd_tt))
    base = _SCHEMES[resolved].base_at_j2000_deg
    value = _normalize_deg(base + drift)
    return {
        "ok": True,
        "scheme": resolved,
        "ayanamsa_deg": value,
        "components": {
            "base_at_j2000_deg": base,
            "drift_deg_from_j2000": drift,
            "years_since_j2000": yrs,
        },
        "model": {
            "type": "linear_rate",
            "rate_arcsec_per_year": _RATE_ARCSEC_PER_YEAR,
        },
        "resolved_from": str(scheme),
    }


# ---------------------------------------------------------------------
# Backward-compat shim (if any code calls old name)
# ---------------------------------------------------------------------

def get_ayanamsa(jd_tt: float, scheme: str = "lahiri") -> float:
    """Alias for get_ayanamsa_deg; kept for readability in some call sites."""
    return get_ayanamsa_deg(jd_tt, scheme)

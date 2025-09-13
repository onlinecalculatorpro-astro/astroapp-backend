# app/core/ayanamsa.py
# -*- coding: utf-8 -*-
"""
Ayanāṁśa (sidereal–tropical offset) — GOLD-READY CORE (v1.3)

Deterministic, dependency-free ayanāṁśa engine that integrates with
astronomy/compute_chart via `get_ayanamsa_deg(jd_tt, scheme)`.

Key features
------------
- Linear model: base@J2000 + constant drift (50.290966 arcsec / tropical year)
- TT input (Julian Day in Terrestrial Time)
- Output wrapped to [0, 360)
- LRU caching for speed
- Runtime registration/override of schemes
- Resolution helpers to detect alias/unknown (fallback) keys

Public API
----------
get_ayanamsa_deg(jd_tt: float, scheme: str = "lahiri") -> float
get_ayanamsa_with_resolution(jd_tt: float, scheme: Optional[str]) -> Tuple[float, str, bool, bool]
ayanamsa_info(jd_tt: float, scheme: str = "lahiri") -> dict
list_ayanamsa_schemes() -> list[str]
register_ayanamsa(name: str, base_at_j2000_deg: float) -> None
resolve_ayanamsa_scheme(scheme: Optional[str]) -> Tuple[str, bool, bool]

Notes
-----
For sidereal longitudes:
    sidereal_lon = (tropical_lon - ayanamsa_deg) % 360
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

__all__ = [
    "get_ayanamsa_deg",
    "get_ayanamsa_with_resolution",
    "ayanamsa_info",
    "list_ayanamsa_schemes",
    "register_ayanamsa",
    "resolve_ayanamsa_scheme",
    "get_ayanamsa",  # backward-compat alias
]

# ---------------------------------------------------------------------
# Core constants (kept consistent with astronomy fallback)
# ---------------------------------------------------------------------

J2000_JD_TT: float = 2451545.0

# Lahiri base used by astronomy fallback:
# 23° 51' 26.26" = 23 + 51/60 + 26.26/3600
_LAHIRI_BASE_J2000_DEG = 23.0 + 51.0 / 60.0 + 26.26 / 3600.0  # ≈ 23.857294444444445

# Linear precession rate used by astronomy fallback (arcsec per tropical year)
_RATE_ARCSEC_PER_YEAR = 50.290966


def _normalize_deg(x: float) -> float:
    """Wrap degrees to [0, 360)."""
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

# Aliases accepted by astronomy and common usage
_ALIASES: Dict[str, str] = {
    "chitrapaksha":  "lahiri",
    "default":       "lahiri",
    "sidereal":      "lahiri",
    "fagan":         "fagan_bradley",
    "fagan/bradley": "fagan_bradley",
    "fagan_bradley": "fagan_bradley",
    "kp":            "krishnamurti",
}


# ---------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------

def _years_since_j2000(jd_tt: float) -> float:
    """Tropical years from J2000.0 (TT). Uses 365.25 to match astronomy fallback."""
    return (float(jd_tt) - J2000_JD_TT) / 365.25


@lru_cache(maxsize=4096)
def _drift_deg_from_j2000(jd_tt: float) -> float:
    """
    Linear precession drift (degrees) from J2000 to jd_tt.

    astronomy fallback uses a single linear rate:
        50.290966 arcsec / year
    """
    years = _years_since_j2000(jd_tt)
    return (_RATE_ARCSEC_PER_YEAR * years) / 3600.0


def _resolve_scheme_core(scheme: Optional[str]) -> Tuple[str, bool, bool]:
    """
    Internal resolver that distinguishes:
      - direct canonical hit
      - alias hit
      - unknown -> default("lahiri")

    Returns
    -------
    (canonical, is_alias, is_unknown)
    """
    if not scheme:
        return ("lahiri", False, False)
    raw = str(scheme).strip().lower()
    if raw in _SCHEMES:
        return (raw, False, False)
    if raw in _ALIASES:
        return (_ALIASES[raw], True, False)
    # Unknown: default to Lahiri but flag as unknown so caller can warn.
    return ("lahiri", False, True)


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
    _SCHEMES[key] = _Scheme(key, float(base_at_j2000_deg))


def resolve_ayanamsa_scheme(scheme: Optional[str]) -> Tuple[str, bool, bool]:
    """
    Resolve a scheme/alias and tell the caller whether it was an alias or unknown.

    Returns
    -------
    (canonical, is_alias, is_unknown)
      canonical : str  - canonical scheme in _SCHEMES
      is_alias  : bool - True if `scheme` was a known alias (no fallback warning)
      is_unknown: bool - True if `scheme` was unknown and defaulted to Lahiri (warn!)
    """
    return _resolve_scheme_core(scheme)


@lru_cache(maxsize=8192)
def get_ayanamsa_deg(jd_tt: float, scheme: str = "lahiri") -> float:
    """
    Compute ayanāṁśa (degrees in [0,360)) for a given TT Julian Day and scheme.

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
    canonical, _, _ = _resolve_scheme_core(scheme)
    base = _SCHEMES[canonical].base_at_j2000_deg
    drift = _drift_deg_from_j2000(float(jd_tt))
    return _normalize_deg(base + drift)


def get_ayanamsa_with_resolution(jd_tt: float, scheme: Optional[str]) -> Tuple[float, str, bool, bool]:
    """
    Like get_ayanamsa_deg, but also returns resolution flags for warnings.

    Returns
    -------
    (ayanamsa_deg, canonical, is_alias, is_unknown)
    """
    canonical, is_alias, is_unknown = _resolve_scheme_core(scheme)
    value = get_ayanamsa_deg(jd_tt, canonical)
    return (value, canonical, is_alias, is_unknown)


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
      - resolved:
          - input: str | None
          - canonical: str
          - is_alias: bool
          - is_unknown: bool   # ← use this to trigger a fallback warning upstream
      - resolved_from: str (deprecated; kept for backward-compat)
    """
    canonical, is_alias, is_unknown = _resolve_scheme_core(scheme)
    yrs = _years_since_j2000(float(jd_tt))
    drift = _drift_deg_from_j2000(float(jd_tt))
    base = _SCHEMES[canonical].base_at_j2000_deg
    value = _normalize_deg(base + drift)
    return {
        "ok": True,
        "scheme": canonical,
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
        "resolved": {
            "input": None if scheme is None else str(scheme),
            "canonical": canonical,
            "is_alias": is_alias,
            "is_unknown": is_unknown,
        },
        # Back-compat key some code may still read:
        "resolved_from": str(scheme),
    }


# ---------------------------------------------------------------------
# Backward-compat shim (if any code calls old name)
# ---------------------------------------------------------------------

def get_ayanamsa(jd_tt: float, scheme: str = "lahiri") -> float:
    """Alias for get_ayanamsa_deg; kept for readability in some call sites."""
    return get_ayanamsa_deg(jd_tt, scheme)

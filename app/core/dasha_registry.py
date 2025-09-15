# app/core/dasha_registry.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, Callable, Optional
import math

__all__ = [
    "compute_dasha",
    "_moon_nirayana_deg_at",
    "available_schemes",
]

# ────────────────────────── optional deps / singletons ──────────────────────────
try:
    from app.core.ephemeris_adapter import EphemerisAdapter, Config as EphemConfig  # type: ignore
    from app.core.ephem_singleton import TS, PLANETS  # TS for TT/UTC bridge; PLANETS cache
    _EPH_OK = True
except Exception:
    EphemerisAdapter = object  # type: ignore
    EphemConfig = object       # type: ignore
    TS = None; PLANETS = None
    _EPH_OK = False

from app.core.ayanamsa import get_ayanamsa_deg

# Engines (import defensively; registry will include only those that import cleanly)
_VIM_OK = _ASHTO_OK = _YOGINI_OK = False
try:
    # Common export name used by your API route
    from app.core.vimshottari_dasha import compute_vimshottari as _compute_vim
    _VIM_OK = True
except Exception:
    pass

try:
    from app.core.ashtottari_dasha import compute_ashtottari as _compute_ashto
    _ASHTO_OK = True
except Exception:
    pass

try:
    from app.core.yogini_dasha import compute_yogini as _compute_yogini
    _YOGINI_OK = True
except Exception:
    pass

# ────────────────────────── small numerics ──────────────────────────
def _norm360(x: float) -> float:
    r = math.fmod(float(x), 360.0)
    return r + 360.0 if r < 0.0 else r

# ────────────────────────── robust ephemeris (cached) ──────────────────────────
_EPHEM_CACHED: Optional[Any] = None  # EphemerisAdapter instance or None

def _make_ephem_adapter() -> Any:
    """
    Construct an EphemerisAdapter that works across adapter versions:
      - Prefer passing TS to the adapter (not the Config).
      - Try multiple signatures; last resort: Config with timescale.
    Cache the successful adapter for reuse.
    """
    # Use cache if already constructed
    global _EPHEM_CACHED
    if _EPHEM_CACHED is not None:
        return _EPHEM_CACHED

    if not _EPH_OK:
        raise RuntimeError("EphemerisAdapter unavailable; enable app.core.ephemeris_adapter")

    # Prefer passing TS and PLANETS via Config where supported
    try:
        cfg = EphemConfig(frame="ecliptic-of-date", planets=PLANETS)  # type: ignore[arg-type]
    except TypeError:
        cfg = EphemConfig(frame="ecliptic-of-date")  # type: ignore

    # Try a few constructor shapes to be resilient to version changes
    try:
        ep = EphemerisAdapter(cfg, timescale=TS)  # type: ignore[arg-type]
    except TypeError:
        try:
            ep = EphemerisAdapter(cfg, TS)  # type: ignore[misc]
        except TypeError:
            try:
                cfg2 = EphemConfig(frame="ecliptic-of-date", planets=PLANETS, timescale=TS)  # type: ignore
                ep = EphemerisAdapter(cfg2)  # type: ignore
            except TypeError:
                ep = EphemerisAdapter(cfg)  # type: ignore

    _EPHEM_CACHED = ep
    return ep

def _moon_nirayana_deg_at(jd_tt: float, *, ayanamsa_key: str) -> float:
    """
    Compute Moon’s nirayana longitude at given TT JD using the robust, cached adapter
    and current ayanāṁśa key.
    """
    ephem = _make_ephem_adapter()
    rows = (ephem.ecliptic_longitudes(float(jd_tt), ["Moon"]) or {}).get("results", [])
    if not rows:
        raise RuntimeError("ephemeris returned no Moon longitude")
    moon_trop = float(rows[0]["longitude"])
    ay = float(get_ayanamsa_deg(float(jd_tt), ayanamsa_key))
    return _norm360(moon_trop - ay)

# ────────────────────────── scheme normalization / registry ──────────────────────────
def _normalize_scheme(name: Any) -> str:
    """
    Map user/system aliases to canonical keys:
      - 'vimśottarī', 'vimsottari', 'vimshottari_dasha' → 'vimshottari'
      - 'aṣṭottarī', 'ashtottari_dasha' → 'ashtottari'
      - 'yoginī', 'yogini_dasha' → 'yogini'
    Default: 'vimshottari' if unspecified.
    """
    if not name:
        return "vimshottari"
    s = str(name).strip().lower()

    # Remove common punctuation/diacritics marks for robust matching
    s = (s
         .replace("ś", "s").replace("ṣ", "s").replace("ī", "i")
         .replace("_", "").replace("-", "").replace(" ", "")
    )

    # direct hits
    if s in ("vimshottari", "vimsottari", "vimshottaridasha", "vimshottariy", "vimshottariyog"):
        return "vimshottari"
    if s in ("ashtottari", "ashtottaridasha", "astottari", "ashtottariy"):
        return "ashtottari"
    if s in ("yogini", "yoginidasha", "yoginii", "yoginiy"):
        return "yogini"

    # fallback to original if it matches exactly a known key
    if s in ("vimshottari", "ashtottari", "yogini"):
        return s

    # default
    return "vimshottari"

_DASHA_REGISTRY: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}
if _VIM_OK:
    _DASHA_REGISTRY["vimshottari"] = _compute_vim  # type: ignore[misc]
if _ASHTO_OK:
    _DASHA_REGISTRY["ashtottari"] = _compute_ashto  # type: ignore[misc]
if _YOGINI_OK:
    _DASHA_REGISTRY["yogini"] = _compute_yogini    # type: ignore[misc]

def available_schemes() -> Dict[str, bool]:
    """Expose which engines are currently wired in."""
    return {
        "vimshottari": _VIM_OK,
        "ashtottari": _ASHTO_OK,
        "yogini": _YOGINI_OK,
    }

# ────────────────────────── public dispatcher ──────────────────────────
def compute_dasha(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dispatch to a specific daśā engine based on payload.
    Accepts aliases in any of: 'scheme' | 'system' | 'dasha'.
    Falls back to 'vimshottari' if none provided.

    Each engine is responsible for:
      - interpreting jd_tt or {date,time,tz},
      - ayanāṁśa handling,
      - optional ephemeris short-circuit overrides, etc.

    Returns the engine's response dict (with 'ok': bool).
    """
    scheme = _normalize_scheme(
        payload.get("scheme") or payload.get("system") or payload.get("dasha")
    )
    fn = _DASHA_REGISTRY.get(scheme)
    if not fn:
        return {"ok": False, "error": f"unsupported_dasha:{scheme}", "available": available_schemes()}
    try:
        return fn(payload)
    except Exception as e:
        # Provide a consistent failure envelope from the dispatcher
        return {"ok": False, "error": f"dasha_dispatch_failed:{scheme}:{e}"}

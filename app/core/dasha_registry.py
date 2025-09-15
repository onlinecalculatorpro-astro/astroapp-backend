# app/core/dasha_registry.py
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Central dispatcher for Vedic daśā engines.

Changes in this version (Vimśottarī only):
- When scheme is "vimshottari", the registry now normalizes payload to your
  new engine contract:
    • method            : "sidereal" | "tropical"  (defaults to "sidereal")
    • coordinate_mode   : "geocentric" | "topocentric"
      (also accepts observer="geocentric|topocentric" or topocentric: bool)
    • ayanamsa          : normalized string; defaults to "lahiri"
  Backwards-compat keys (zodiac_mode, mode, ayanamsa_key) are accepted.

Other engines (ashtottari, yogini, chara, kalachakra) are passed through unchanged.
"""

from typing import Any, Dict, Callable, Optional
import math

__all__ = [
    "compute_dasha",
    "compute_dasha_for",
    "available_schemes",
    "_moon_nirayana_deg_at",
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
_VIM_OK = _ASHTO_OK = _YOGINI_OK = _CHARA_OK = _KCD_OK = False

try:
    # Vimśottarī
    from app.core.vimshottari_dasha import compute_vimshottari as _compute_vim
    _VIM_OK = True
except Exception:
    pass

try:
    # Aṣṭottarī
    from app.core.ashtottari_dasha import compute_ashtottari as _compute_ashto
    _ASHTO_OK = True
except Exception:
    pass

try:
    # Yoginī
    from app.core.yogini_dasha import compute_yogini as _compute_yogini
    _YOGINI_OK = True
except Exception:
    pass

try:
    # Jaimini Chara
    from app.core.chara_dasha import compute_chara_dasha as _compute_chara
    _CHARA_OK = True
except Exception:
    pass

try:
    # Kālachakra
    from app.core.kala_chakra_dasha import compute_kalachakra_dasha as _compute_kcd
    _KCD_OK = True
except Exception:
    pass

# ────────────────────────── small numerics / ephemeris ──────────────────────────
def _norm360(x: float) -> float:
    r = math.fmod(float(x), 360.0)
    return r + 360.0 if r < 0.0 else r

_EPHEM_CACHED: Optional[Any] = None  # EphemerisAdapter instance or None

def _make_ephem_adapter() -> Any:
    """
    Construct an EphemerisAdapter that works across adapter versions:
      - Prefer passing TS/PLANETS via Config.
      - Try multiple signatures; last resort: Config-only.
    Cache the successful adapter for reuse.
    """
    global _EPHEM_CACHED
    if _EPHEM_CACHED is not None:
        return _EPHEM_CACHED
    if not _EPH_OK:
        raise RuntimeError("EphemerisAdapter unavailable; enable app.core.ephemeris_adapter")

    try:
        cfg = EphemConfig(frame="ecliptic-of-date", planets=PLANETS)  # type: ignore[arg-type]
    except TypeError:
        cfg = EphemConfig(frame="ecliptic-of-date")  # type: ignore

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

def _moon_nirayana_deg_at(jd_tt: float, *, ayanamsa_key: str = "lahiri") -> float:
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

# ────────────────────────── alias normalization ──────────────────────────
def _normalize_scheme(name: Any) -> str:
    """
    Map user/system aliases to canonical keys:

    vimśottarī  → "vimshottari"
    aṣṭottarī   → "ashtottari"
    yoginī      → "yogini"
    jaimini/chara → "chara"
    kālachakra  → "kalachakra" (also "kcd")

    Default: "vimshottari" if unspecified.
    """
    if not name:
        return "vimshottari"
    s = str(name).strip().lower()

    # strip diacritics & punctuation for robust matching
    repl = {
        "ś": "s", "ṣ": "s", "ī": "i", "ā": "a", "ḍ": "d", "ṭ": "t",
        "_": "", "-": "", " ": ""
    }
    for k, v in repl.items():
        s = s.replace(k, v)

    # vimshottari
    if s in ("vimshottari", "vimsottari", "vimshottaridasha", "vimshottariy"):
        return "vimshottari"
    # ashtottari
    if s in ("ashtottari", "astottari", "ashtottaridasha"):
        return "ashtottari"
    # yogini
    if s in ("yogini", "yoginidasha"):
        return "yogini"
    # chara / jaimini
    if s in ("chara", "charadasha", "jaimini", "jaiminichara", "jaiminidasha", "charajaimini"):
        return "chara"
    # kalachakra
    if s in ("kalachakra", "kalacakra", "kcd", "kalachakradasha", "kalachakradasa"):
        return "kalachakra"

    # if someone already sent canonical key verbatim
    if s in ("vimshottari", "ashtottari", "yogini", "chara", "kalachakra"):
        return s

    return "vimshottari"

# ────────────────────────── Vimśottarī payload normalizer ──────────────────────────
def _norm_vim_payload(p: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make payload robust to legacy keys and ensure the new engine flags exist.
    - method: from method|mode|zodiac_mode → "sidereal"|"tropical" (default "sidereal")
    - coordinate_mode/topocentric: from coordinate_mode|observer|topocentric
    - ayanamsa: from ayanamsa|ayanamsa_key (default "lahiri")
    Leaves lat/lon/elevation, jd_tt/jd_ut1, date/time/tz, levels etc. untouched if present.
    """
    if not isinstance(p, dict):
        return {}

    out = dict(p)  # shallow copy

    # method
    method = (out.get("method") or out.get("mode") or out.get("zodiac_mode") or "sidereal")
    m = str(method).strip().lower()
    if m in ("sidereal","nirayana","nirāyaṇa","sid","s"):
        out["method"] = "sidereal"
    elif m in ("tropical","sayana","sāyana","trop","t"):
        out["method"] = "tropical"
    else:
        out["method"] = "sidereal"

    # coordinate_mode / topocentric
    if "coordinate_mode" in out and isinstance(out["coordinate_mode"], str):
        cm = out["coordinate_mode"].strip().lower()
    else:
        obs = str(out.get("observer","")).strip().lower()
        if obs in ("topocentric","apparent","obs"):
            cm = "topocentric"
        elif obs in ("geocentric","geo","center"):
            cm = "geocentric"
        else:
            cm = "topocentric" if bool(out.get("topocentric")) else "geocentric"
    out["coordinate_mode"] = "topocentric" if cm == "topocentric" else "geocentric"
    out["topocentric"] = (out["coordinate_mode"] == "topocentric")

    # ayanamsa
    ay = out.get("ayanamsa", out.get("ayanamsa_key", "lahiri"))
    ay = str(ay).strip().lower() if isinstance(ay, str) else (ay or "lahiri")
    out["ayanamsa"] = ay or "lahiri"
    out["ayanamsa_key"] = out["ayanamsa"]

    return out

# ────────────────────────── registry ──────────────────────────
_DASHA_REGISTRY: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}
if _VIM_OK:
    _DASHA_REGISTRY["vimshottari"] = _compute_vim  # type: ignore[misc]
if _ASHTO_OK:
    _DASHA_REGISTRY["ashtottari"] = _compute_ashto  # type: ignore[misc]
if _YOGINI_OK:
    _DASHA_REGISTRY["yogini"] = _compute_yogini    # type: ignore[misc]
if _CHARA_OK:
    _DASHA_REGISTRY["chara"] = _compute_chara      # type: ignore[misc]
if _KCD_OK:
    _DASHA_REGISTRY["kalachakra"] = _compute_kcd   # type: ignore[misc]
    _DASHA_REGISTRY["kcd"] = _compute_kcd          # alias

def available_schemes() -> Dict[str, bool]:
    """Expose which engines are currently wired in."""
    return {
        "vimshottari": _VIM_OK,
        "ashtottari": _ASHTO_OK,
        "yogini": _YOGINI_OK,
        "chara": _CHARA_OK,
        "kalachakra": _KCD_OK,
    }

# ───────────── dev helper: optional demo KCD table injection (off by default) ─────────────
def _demo_kcd_table() -> Dict[str, Any]:
    years = {1:7, 2:16, 3:9, 4:15, 5:19, 6:12, 7:7, 8:16, 9:9, 10:15, 11:19, 12:12}
    base = [1,2,3,4,5,6,7,8,9,10,11,12]
    seq = {}
    for idx in range(1, 109):
        sh = (idx - 1) % 12
        seq[idx] = base[sh:] + base[:sh]
    return {"name": "demo_dev_only", "sign_years": years, "pada_to_sequence": seq}

def _maybe_inject_demo_kcd(scheme_key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if scheme_key not in ("kalachakra", "kcd"):
        return payload
    if not isinstance(payload, dict):
        return {}
    if "kcd_table" in payload or "kcd_preset" in payload:
        return payload
    if bool(payload.get("use_demo_kcd_table", False)):
        payload = dict(payload)
        payload["kcd_table"] = _demo_kcd_table()
    return payload

# ────────────────────────── public dispatchers ──────────────────────────
def _dispatch(scheme_key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    fn = _DASHA_REGISTRY.get(scheme_key)
    if not fn:
        return {"ok": False, "error": f"unsupported_dasha:{scheme_key}", "available": available_schemes()}
    try:
        # Vimśottarī-only payload normalization (new engine flags & backwards-compat)
        if scheme_key == "vimshottari":
            payload = _norm_vim_payload(payload or {})

        # Dev helper (Kalachakra only)
        payload = _maybe_inject_demo_kcd(scheme_key, payload or {})

        out = fn(payload or {})
        if not isinstance(out, dict):
            return {"ok": False, "error": f"engine_returned_non_dict:{scheme_key}"}
        return out
    except Exception as e:
        return {"ok": False, "error": f"dasha_dispatch_failed:{scheme_key}:{e}"}

def compute_dasha(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dispatch to a specific daśā engine based on payload.
    Accepts aliases in any of: 'scheme' | 'system' | 'dasha'.
    Falls back to 'vimshottari' if none provided.
    """
    scheme_key = _normalize_scheme(
        payload.get("scheme") or payload.get("system") or payload.get("dasha")
    )
    return _dispatch(scheme_key, payload or {})

def compute_dasha_for(scheme: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Variant that takes the scheme explicitly (useful for routers/services).
    """
    scheme_key = _normalize_scheme(scheme)
    return _dispatch(scheme_key, payload or {})

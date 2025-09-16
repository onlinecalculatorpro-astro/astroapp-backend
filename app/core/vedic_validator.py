# app/core/vedic_validator.py
# -*- coding: utf-8 -*-
"""
Vedic API — Payload normalization & validation

Includes:
    - normalize_vim_payload(payload)  -> (norm, warnings, tz_normalized)
    - normalize_yoga_payload(payload) -> (norm, warnings, tz_normalized)

Design notes
------------
• Sidereal-first defaults (method="sidereal", observer="geocentric", ayanamsa="lahiri").
• NEVER returns or computes jd_utc. Only jd_tt / jd_ut1 (via ERFA-aligned timescales if available).
• Place resolution is optional; if present and resolver exists, we fill lat/lon/elevation/tz.
• Yoga normalizer accepts multiple input modes but does not compute houses/points itself;
  it shapes a request consumable by app.core.yogas.compute_yogas (which can compute when given
  lat/lon + timescales, and house_system).
• Request-scoped options for Yogas (house_system, catalog tag filters, varga-boost keys, etc.)
  are collected under "yoga_options" and passed through unchanged by the route/core layer.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Iterable
import re

# ── Optional timescales (ERFA-aligned); NO jd_utc here ──
try:
    # Expected signature: build_timescales(date_str, time_str, tz_name, dut1_seconds)
    from app.core.timescales import build_timescales  # type: ignore
    _TIMESCALES_OK = True
except Exception:
    build_timescales = None  # type: ignore
    _TIMESCALES_OK = False

# ── Optional place resolver (geocoding + tz + elevation) via astronomy.py ──
_resolve_place = None
try:
    import app.core.astronomy as _astro  # type: ignore
    for _fname in (
        "resolve_place",          # preferred
        "geocode_place",
        "place_to_site",
        "lookup_place",
        "city_to_coords",
        "resolve_site",
    ):
        _resolve_place = getattr(_astro, _fname, None)
        if callable(_resolve_place):
            break
except Exception:
    _astro = None  # type: ignore
    _resolve_place = None

# ── Optional varga key resolver (for key normalization only; no computation here) ──
try:
    from app.core import varga_charts as _varga  # type: ignore
    _VARGA_OK = True
except Exception:
    _varga = None  # type: ignore
    _VARGA_OK = False

# ── Optional yoga registry (for validating requested names) ──
try:
    # File name is yogas.py (registry lives here)
    from app.core.yogas import list_registered_yogas as _list_yoga_registry  # type: ignore
    _YOGA_REG_OK = True
except Exception:
    _list_yoga_registry = None  # type: ignore
    _YOGA_REG_OK = False

# ── Simple helpers ──
_NUM_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?$")


def _coerce_str(x: Any, default: str = "") -> str:
    if x is None:
        return default
    if isinstance(x, (bytes, bytearray)):
        try:
            return x.decode("utf-8", "ignore")
        except Exception:
            return default
    return str(x)


def _as_float(x: Any) -> Optional[float]:
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str) and _NUM_RE.match(x.strip()):
        try:
            return float(x.strip())
        except Exception:
            return None
    return None


def _pad_hms(t: str) -> str:
    """Ensure HH:MM:SS (append :00 if only HH:MM; default 12:00:00)."""
    t = (t or "").strip()
    if not t:
        return "12:00:00"
    parts = t.split(":")
    if len(parts) == 2:
        return f"{parts[0]}:{parts[1]}:00"
    if len(parts) == 1:
        hh = parts[0] or "12"
        return f"{hh}:00:00"
    return t


def _clamp_levels(v: Any, default: int = 5) -> int:
    try:
        depth = int(v)
    except Exception:
        if isinstance(v, list):
            depth = len(v)
        else:
            depth = default
    if depth < 1:
        depth = 1
    if depth > 5:
        depth = 5
    return depth


def _norm_method(v: Any, default: str = "sidereal") -> str:
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("sidereal", "nirayana", "nirāyaṇa", "sid", "s"):
            return "sidereal"
        if s in ("tropical", "sayana", "sāyana", "trop", "t"):
            return "tropical"
    return default


def _norm_observer(v: Any, default: str = "geocentric") -> str:
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("geocentric", "geo", "center"):
            return "geocentric"
        if s in ("topocentric", "apparent", "obs"):
            return "topocentric"
    return default


def _norm_ayanamsa(v: Any) -> str:
    # Default to LAHIRI (string key or numeric allowed)
    if v is None:
        return "lahiri"
    s = str(v).strip().lower()
    return s or "lahiri"


def _join_place(city: str, state: str, country: str) -> str:
    parts = [p.strip() for p in (city, state, country) if _coerce_str(p).strip()]
    return ", ".join(parts)


def _collect_list(payload: Dict[str, Any], keys: Iterable[str]) -> List[str]:
    """Accept list/tuple OR comma/space separated string for any of the given keys."""
    raw: List[str] = []
    for k in keys:
        v = payload.get(k)
        if not v:
            continue
        if isinstance(v, (list, tuple)):
            raw.extend([_coerce_str(x) for x in v])
        else:
            txt = _coerce_str(v)
            raw.extend([p for p in re.split(r"[,\s]+", txt) if p])
    # de-duplicate preserving order
    seen = set()
    out: List[str] = []
    for x in raw:
        xx = x.strip()
        if not xx:
            continue
        if xx not in seen:
            seen.add(xx)
            out.append(xx)
    return out


# ── Varga key collection & normalization (no computation here) ──
_VARGA_ALIAS_MIN = {
    "rasi": "D1", "d1": "D1",
    "hora": "D2", "d2": "D2",
    "drekkana": "D3", "d3": "D3",
    "chaturthamsa": "D4", "d4": "D4",
    "saptamsa": "D7", "d7": "D7",
    "navamsa": "D9", "navamsha": "D9", "d9": "D9",
    "dasamsa": "D10", "dashamsa": "D10", "d10": "D10",
    "dvadasamsa": "D12", "dvadashamsa": "D12", "d12": "D12",
    "shodasamsa": "D16", "sodasamsa": "D16", "d16": "D16",
    "vimsamsa": "D20", "d20": "D20",
    "chaturvimshamsa": "D24", "siddhamsa": "D24", "d24": "D24",
    "bhamsa": "D27", "nakshatramsa": "D27", "d27": "D27",
    "trimsamsa": "D30", "trimsamsha": "D30", "d30": "D30",
    "khavedamsa": "D40", "d40": "D40",
    "akshavedamsa": "D45", "d45": "D45",
    "shashtiamsa": "D60", "shastiamsa": "D60", "d60": "D60",
}


def _canon_varga_key(x: str) -> Optional[str]:
    if not x:
        return None
    k = x.strip()
    if not k:
        return None
    # Prefer project’s varga module resolver if available
    if _VARGA_OK:
        try:
            res = getattr(_varga, "_resolve_key", None)
            if callable(res):
                return str(res(k))
        except Exception:
            pass
    kl = k.lower()
    if kl in _VARGA_ALIAS_MIN:
        return _VARGA_ALIAS_MIN[kl]
    if kl.startswith("d") and kl[1:].isdigit():
        return "D" + kl[1:]
    return None


def _collect_vargas(payload: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """Return (canon_vargas, warnings). Accept list or comma strings across multiple keys."""
    warns: List[str] = []
    raw_keys: List[str] = []
    for key in ("vargas", "varga", "divisional", "divisional_charts"):
        v = payload.get(key)
        if not v:
            continue
        if isinstance(v, (list, tuple)):
            raw_keys.extend([_coerce_str(x) for x in v])
        else:
            txt = _coerce_str(v)
            raw_keys.extend([p for p in re.split(r"[,\s]+", txt) if p])
    canon: List[str] = []
    bad: List[str] = []
    for x in raw_keys:
        ck = _canon_varga_key(x)
        if ck:
            if ck not in canon:
                canon.append(ck)
        else:
            bad.append(x)
    if bad:
        warns.append("unknown_varga_keys:" + ",".join(bad))
    return canon, warns


# ─────────────────────────────────────────────────────────────────────────────
# Vimśottarī normalizer (existing behavior preserved)
# ─────────────────────────────────────────────────────────────────────────────

def normalize_vim_payload(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], str]:
    """
    Normalize inputs for Vimśottarī with basis (method) & observer switches,
    and optionally include a varga_request block (no varga computation here).

    Returns: (norm, warns, tz_norm)
    """
    warns: List[str] = []

    # Civic primitives
    date = _coerce_str(payload.get("date") or payload.get("birth_date") or "")
    time_in = _coerce_str(payload.get("time") or payload.get("birth_time") or "12:00")
    time_str = _pad_hms(time_in)

    # Place of Birth: combined or split
    pob_str = _coerce_str(
        payload.get("Place of Birth")
        or payload.get("place")
        or payload.get("birth_place")
        or payload.get("birthPlace")
        or ""
    ).strip()
    if not pob_str:
        pob_str = _join_place(
            _coerce_str(payload.get("place_city")),
            _coerce_str(payload.get("place_state")),
            _coerce_str(payload.get("place_country")),
        )

    # Basis/observer/ayanamsa
    method = _norm_method(payload.get("method", payload.get("mode", "sidereal")), default="sidereal")
    observer = _norm_observer(payload.get("observer", "geocentric"))
    ayanamsa = _norm_ayanamsa(payload.get("ayanamsa") or payload.get("ayanamsa_key"))

    # Optional pre-supplied site & tz
    lat = _as_float(payload.get("latitude") or payload.get("lat"))
    lon = _as_float(payload.get("longitude") or payload.get("lon"))
    elevation_m = _as_float(payload.get("elevation_m") or payload.get("elevation"))
    tz_norm = _coerce_str(payload.get("tz") or payload.get("place_tz") or "UTC").strip() or "UTC"

    # Depth
    depth = _clamp_levels(payload.get("levels", payload.get("depth", payload.get("max_levels", 5))), default=5)

    # Resolve place → site+tz
    if not pob_str:
        warns.append("missing_place_of_birth")
    else:
        if callable(_resolve_place):
            try:
                city = _coerce_str(payload.get("place_city")).strip() or None
                state = _coerce_str(payload.get("place_state")).strip() or None
                country = _coerce_str(payload.get("place_country")).strip() or None
                try:
                    pr = _resolve_place(pob_str, place_city=city, place_state=state, place_country=country)  # type: ignore[misc]
                except TypeError:
                    pr = _resolve_place(pob_str)  # type: ignore[misc]

                _lat = pr.get("lat") if isinstance(pr, dict) else getattr(pr, "lat", None)
                _lon = pr.get("lon") if isinstance(pr, dict) else getattr(pr, "lon", None)
                _tz = pr.get("tz") if isinstance(pr, dict) else getattr(pr, "tz", None)
                _elev = pr.get("elevation_m") if isinstance(pr, dict) else getattr(pr, "elevation_m", None)

                if _lat is not None and _lon is not None:
                    lat = _as_float(_lat)
                    lon = _as_float(_lon)
                if _elev is not None:
                    elevation_m = _as_float(_elev)
                if _tz:
                    tz_norm = _coerce_str(_tz).strip() or tz_norm
                else:
                    warns.append("place_resolved_without_tz:fallback_tz_applied")
            except Exception as e:
                warns.append(f"place_resolution_failed:{e!s}")
        else:
            warns.append("place_resolver_unavailable")

    # Observer normalization → coordinate_mode/topocentric flag
    coordinate_mode = observer  # "geocentric" | "topocentric"
    topocentric = (coordinate_mode == "topocentric")
    if topocentric and (lat is None or lon is None):
        coordinate_mode = "geocentric"
        topocentric = False
        warns.append("topocentric_requires_coordinates:fallback_geocentric")

    # Timescales → jd_tt / jd_ut1
    jd_tt: Optional[float] = _as_float(payload.get("jd_tt")) if payload.get("jd_tt") is not None else None
    jd_ut1: Optional[float] = _as_float(payload.get("jd_ut1")) if payload.get("jd_ut1") is not None else None

    if _TIMESCALES_OK and (jd_tt is None or jd_ut1 is None) and date:
        try:
            ts = build_timescales(date, time_str, tz_norm, 0.0)  # type: ignore[call-arg]
            if isinstance(ts, dict):
                if jd_tt is None and ts.get("jd_tt") is not None:
                    jd_tt = float(ts["jd_tt"])
                if jd_ut1 is None and ts.get("jd_ut1") is not None:
                    jd_ut1 = float(ts["jd_ut1"])
            else:
                if jd_tt is None:
                    jd_tt = float(getattr(ts, "jd_tt"))
                if jd_ut1 is None:
                    jd_ut1 = float(getattr(ts, "jd_ut1"))
        except Exception as e:
            warns.append(f"timescales_failed:{e!s}")

    # Varga request normalization (no computation)
    vargas, varga_warns = _collect_vargas(payload)
    warns.extend(varga_warns)
    include_vargas = bool(payload.get("include_vargas") or vargas)
    varga_mode = _norm_method(payload.get("varga_zodiac_mode"), default="sidereal")
    varga_ayan = payload.get("varga_ayanamsa", ayanamsa)

    varga_request = None
    if include_vargas and vargas:
        varga_request = {
            "enabled": True,
            "vargas": vargas,           # canonical keys like D9, D10, D30
            "zodiac_mode": varga_mode,  # "sidereal" (default) or "tropical"
            "ayanamsa": varga_ayan,     # string key or numeric degrees
        }

    # Canonical normalized dict
    norm: Dict[str, Any] = {
        "system": "vimshottari",
        "date": date,
        "time": time_str,
        "tz": tz_norm,

        "method": method,                    # "sidereal" | "tropical"
        "ayanamsa": ayanamsa,                # default "lahiri"

        "levels": depth,
        "max_levels": depth,

        "latitude": lat,
        "longitude": lon,
        "elevation_m": elevation_m,

        "coordinate_mode": coordinate_mode,  # "geocentric" | "topocentric"
        "topocentric": bool(topocentric),

        "place_name": pob_str or None,

        "jd_tt": jd_tt,
        "jd_ut1": jd_ut1,

        "raw": payload,
    }

    # Registry/common aliases
    norm.update({
        "tz_name": tz_norm,
        "ayanamsa_key": ayanamsa,
        "birth_date": date,
        "birth_time": time_str,
        "place_tz": tz_norm,
        "depth": depth,
    })

    if varga_request:
        norm["varga_request"] = varga_request

    # Final sanity notes
    if not date:
        warns.append("missing_date")
    if not time_str:
        warns.append("missing_time")

    return norm, warns, tz_norm


# ─────────────────────────────────────────────────────────────────────────────
# Yoga normalizer (new)
# ─────────────────────────────────────────────────────────────────────────────

_PLANET_KEY_MAP = {
    # Accept lower/loose keys and map to canonical names
    "sun": "Sun", "surya": "Sun",
    "moon": "Moon", "chandra": "Moon",
    "mars": "Mars", "mangal": "Mars", "kuja": "Mars",
    "mercury": "Mercury", "budha": "Mercury",
    "jupiter": "Jupiter", "guru": "Jupiter", "brihaspati": "Jupiter",
    "venus": "Venus", "shukra": "Venus",
    "saturn": "Saturn", "shani": "Saturn",
    "rahu": "Rahu", "northnode": "Rahu", "north_node": "Rahu",
    "ketu": "Ketu", "southnode": "Ketu", "south_node": "Ketu",
}

_CANON_PLANETS = ("Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn", "Rahu", "Ketu")


def _norm_points_deg(obj: Any, warns: List[str]) -> Optional[Dict[str, float]]:
    """Normalize points_deg mapping to canonical planet names with [0,360) degrees."""
    if not isinstance(obj, dict):
        return None
    out: Dict[str, float] = {}
    for k, v in obj.items():
        key = _coerce_str(k).strip().lower()
        if not key:
            continue
        canon = _PLANET_KEY_MAP.get(key)
        if not canon:
            # ignore unknown keys silently (no crash), note once
            warns.append(f"unknown_point_key:{k!s}")
            continue
        val = _as_float(v)
        if val is None:
            warns.append(f"bad_point_value:{k!s}")
            continue
        # wrap to [0,360)
        deg = float(val) % 360.0
        out[canon] = deg
    return out or None


def _norm_cusps_deg(seq: Any, warns: List[str]) -> Optional[List[float]]:
    """12 floats in [0,360)."""
    if not isinstance(seq, (list, tuple)):
        return None
    vals: List[float] = []
    for i, v in enumerate(seq):
        f = _as_float(v)
        if f is None:
            warns.append(f"bad_cusp_value:index={i}")
            return None
        vals.append(float(f) % 360.0)
    if len(vals) != 12:
        warns.append("cusps_require_12_values")
        return None
    return vals


def _validate_yoga_names(include: List[str]) -> Tuple[List[str], List[str]]:
    """If registry is available, filter to known names; return (ok_names, unknown_names)."""
    if not include:
        return [], []
    if not _YOGA_REG_OK or not callable(_list_yoga_registry):  # registry not loaded
        # Pass-through unchanged
        return include, []
    try:
        reg = _list_yoga_registry() or []
        names = {str(r.get("name", "")).strip().lower() for r in reg if r.get("name")}
        ok: List[str] = []
        bad: List[str] = []
        for n in include:
            nn = n.strip()
            if not nn:
                continue
            if nn.lower() in names:
                ok.append(nn)
            else:
                bad.append(nn)
        return ok, bad
    except Exception:
        return include, []


def normalize_yoga_payload(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], str]:
    """
    Normalize inputs for Yoga detection.

    Accepted shapes (request may include any of these):
      • Mode A (precomputed): points_deg + cusps_deg + method + ayanamsa
      • Mode B (points_deg + site/time → houses computed downstream)
      • Mode C (full compute): site/time only; downstream ephemeris/houses compute all

    Returns: (norm, warns, tz_norm)
    """
    warns: List[str] = []

    # --- Basis / observer / ayanamsa (sidereal-first) ---
    method = _norm_method(payload.get("method", payload.get("mode", "sidereal")), default="sidereal")
    observer = _norm_observer(payload.get("observer", "geocentric"))
    ayanamsa = _norm_ayanamsa(payload.get("ayanamsa") or payload.get("ayanamsa_key"))

    # House system (downstream default is "placidus")
    house_system = _coerce_str(
        payload.get("house_system") or payload.get("house") or payload.get("hs") or "placidus"
    ).strip() or "placidus"

    # Optional sign-lord variant
    sign_lord_variant = _coerce_str(
        payload.get("sign_lord_variant") or payload.get("sign_lords") or payload.get("lords_variant") or "classical"
    ).strip() or "classical"

    # Optional scoring/behavior flags
    def _as_bool(x: Any, default: bool = True) -> bool:
        if isinstance(x, bool):
            return x
        if isinstance(x, str):
            s = x.strip().lower()
            if s in ("1", "true", "yes", "y", "on"):
                return True
            if s in ("0", "false", "no", "n", "off"):
                return False
        return default

    chandra_mangala_by_sign = _as_bool(payload.get("chandra_mangala_by_sign"), True)
    gajakesari_include_same_house = _as_bool(payload.get("gajakesari_include_same_house"), True)
    include_mooltrikona_in_mahapurusha = _as_bool(payload.get("include_mooltrikona_in_mahapurusha"), True)
    use_vargas_for_scoring = _as_bool(payload.get("use_vargas_for_scoring"), True)
    include_arudha_notes = _as_bool(payload.get("include_arudha_notes"), True)

    # Orb clamping (0..15 typical)
    conj_orb_deg = _as_float(payload.get("conj_orb_deg") or payload.get("orb_deg")) or 6.0
    if conj_orb_deg < 0.0:
        conj_orb_deg = 0.0
        warns.append("conj_orb_clamped_to_min_0")
    if conj_orb_deg > 30.0:
        conj_orb_deg = 30.0
        warns.append("conj_orb_clamped_to_max_30")

    # Varga boost keys (default D9/D10)
    varga_boost_keys, _vw = _collect_vargas(payload)
    if not varga_boost_keys:
        # Also accept explicit tuple/list keys under varga_keys_for_boost
        varga_boost_keys = _collect_list(payload, ("varga_keys_for_boost", "varga_boost", "boost_vargas")) or ["D9", "D10"]

    # Catalog tag gating
    enable_tags = _collect_list(payload, ("enable_catalog_tags", "enable_tags", "include_tags"))
    disable_tags = _collect_list(payload, ("disable_catalog_tags", "disable_tags", "exclude_tags"))

    # Requested yoga names (optional)
    include_names = _collect_list(payload, ("include", "yogas", "only", "include_names"))
    include_names, unknown_names = _validate_yoga_names(include_names)
    if unknown_names:
        warns.append("unknown_yoga_names:" + ",".join(unknown_names))

    # --- Points & cusps (optional pass-through; downstream may compute anyway) ---
    points_deg = _norm_points_deg(payload.get("points_deg"), warns)
    cusps_deg = _norm_cusps_deg(payload.get("cusps_deg"), warns)

    # --- Site/time primitives (needed if downstream must compute houses/points) ---
    date = _coerce_str(payload.get("date") or payload.get("birth_date") or "")
    time_in = _coerce_str(payload.get("time") or payload.get("birth_time") or "12:00")
    time_str = _pad_hms(time_in)
    tz_norm = _coerce_str(payload.get("tz") or payload.get("place_tz") or "UTC").strip() or "UTC"

    # Coordinates (direct OR via place resolver)
    lat = _as_float(payload.get("latitude") or payload.get("lat"))
    lon = _as_float(payload.get("longitude") or payload.get("lon"))
    elevation_m = _as_float(payload.get("elevation_m") or payload.get("elevation"))

    place_str = _coerce_str(
        payload.get("Place of Birth")
        or payload.get("place")
        or payload.get("birth_place")
        or payload.get("birthPlace")
        or ""
    ).strip()
    if not place_str:
        place_str = _join_place(
            _coerce_str(payload.get("place_city")),
            _coerce_str(payload.get("place_state")),
            _coerce_str(payload.get("place_country")),
        )

    if place_str and callable(_resolve_place):
        try:
            city = _coerce_str(payload.get("place_city")).strip() or None
            state = _coerce_str(payload.get("place_state")).strip() or None
            country = _coerce_str(payload.get("place_country")).strip() or None
            try:
                pr = _resolve_place(place_str, place_city=city, place_state=state, place_country=country)  # type: ignore[misc]
            except TypeError:
                pr = _resolve_place(place_str)  # type: ignore[misc]

            _lat = pr.get("lat") if isinstance(pr, dict) else getattr(pr, "lat", None)
            _lon = pr.get("lon") if isinstance(pr, dict) else getattr(pr, "lon", None)
            _tz = pr.get("tz") if isinstance(pr, dict) else getattr(pr, "tz", None)
            _elev = pr.get("elevation_m") if isinstance(pr, dict) else getattr(pr, "elevation_m", None)

            if lat is None and _lat is not None:
                lat = _as_float(_lat)
            if lon is None and _lon is not None:
                lon = _as_float(_lon)
            if elevation_m is None and _elev is not None:
                elevation_m = _as_float(_elev)
            if _tz:
                tz_norm = _coerce_str(_tz).strip() or tz_norm
            else:
                warns.append("place_resolved_without_tz:fallback_tz_applied")
        except Exception as e:
            warns.append(f"place_resolution_failed:{e!s}")
    elif place_str and

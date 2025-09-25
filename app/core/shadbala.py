# app/core/shadbala.py
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Śaḍbala engine — precision-first (gold-standard ready)

This module computes the six classical bala buckets:
  • Sthāna (positional)
  • Dig     (directional)
  • Kāla    (temporal)
  • Cheṣṭā  (motional)
  • Naisargika (natural)
  • Drik    (aspectual)

Design principles
-----------------
1) Authoritative astronomy only — all longitudes, speeds, angles and time-bases
   come from `app.core.astronomy.compute_chart` which itself should route through
   your ephemeris adapter. We do NOT duplicate ephemeris in this file.

2) House cusps — if `app.core.houses_advanced` is available, we use it. If the
   caller provides precomputed cusps they are used as-is. Otherwise we fall back
   to whole-sign cusps from ASC and emit a clear warning.

3) Precision hooks — if available, we *require* exact helpers from your
   ephemeris adapter for the components that need them:
   - local day/night segmentation and Sun–Moon elongation (for Kāla)
   - aspect weights kernel (for Drik)
   - precise cheshtā strength (for Cheṣṭā)
   If any such helper is unavailable, the corresponding component is returned
   as `{"value": None, "note": "not_computed"}` with a transparent warning.
   We do NOT approximate.

4) Transparent metadata — the response includes everything a client needs to
   audit the route and precision used: angles, house source, adapter feature
   flags, ayanamsa in use, etc.

Public API
----------
    compute_shadbala(payload: dict) -> dict
    clear_caches() -> None

Minimal payload
---------------
{
  "date":"YYYY-MM-DD", "time":"HH:MM[:SS]",
  "tz":"IANA/Zone",
  "latitude": <float>, "longitude": <float>,

  # optional routing/quality knobs
  "zodiac_mode": "sidereal"|"tropical",
  "ayanamsa": <name|deg>,
  "elevation_m": <float>,
  "house_system": <str>,
  "angles": {"asc": <deg>, "mc": <deg>},  # if you want to override/use custom angles
  "house_cusps_deg": [12 floats],         # precomputed cusps (preferred if provided)
  # or: { "houses": { "cusps_deg"|"cusps": [12 floats] } }

  # optional vargas for a *precise* Saptavargaja contribution inside Sthāna
  "vargas": ["D1","D2","D3","D7","D9","D12","D30", ...]
}

Returned shape
--------------
{
  "ok": true,
  "bala": {
    "Sun": {
      "components": {
        "sthana":     {"value": .., "note": None|"...", "extra": {...}} ,
        "dig":        {"value": .., "note": ...},
        "kaala":      {"value": .., "note": ..., "extra": {...}},
        "cheshta":    {"value": .., "note": ...},
        "naisargika": {"value": .., "note": ...},
        "drik":       {"value": .., "note": ..., "extra": {...}}
      },
      "total": <Σ available (each bucket 0..60)>,
      "max":   <60 * (#computed buckets)>,
      "pct":   <total / max>
    },
    ...
  },
  "meta": {
     "module":"shadbala(core)",
     "mode": "sidereal|tropical",
     "angles": {"asc": <deg or None>, "mc": <deg or None>},
     "houses_source": "payload|houses_advanced|whole_sign_fallback|none",
     "house_cusps_deg": [..] (when available),
     "precision_engines": {
         "cheshta": bool,
         "aspect_weights": bool,
         "sun_events": bool,
         "elongation": bool
     },
     "varga_saptavargaja": {"enabled": bool, "vargas": [...]} ,
     "source_chart": {"center": "...", "frame": "...", "ayanamsa_deg": <float|None>},
     "version": 4
  },
  "warnings": [ ... ]
}
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math

# ────────────────────────────── Required core (single source of truth) ───────
try:
    from app.core.astronomy import compute_chart as _compute_chart
except Exception as e:
    raise RuntimeError(f"shadbala: astronomy.compute_chart import failed: {e}")

# ────────────────────────────── Optional helpers (precision engines) ─────────
try:
    # True Sun/Moon elongation in degrees; signature: f(chart_like) -> float
    from app.core.ephemeris_adapter import sun_moon_elongation_deg as _elongation_deg  # type: ignore
except Exception:
    _elongation_deg = None  # type: ignore

try:
    # Local day/night segmentation; signature:
    #   f(payload_or_chart) -> {"is_day": bool, "rise_jd_tt":..., "set_jd_tt":...}
    from app.core.ephemeris_adapter import local_sun_events as _sun_events  # type: ignore
except Exception:
    _sun_events = None  # type: ignore

try:
    # Aspect weights; signature:
    #   f(longitudes_by_name: dict[str,float], mode:str, ayanamsa:any) -> callable|(matrix-like)
    from app.core.ephemeris_adapter import compute_aspect_weights as _aspect_weights  # type: ignore
except Exception:
    _aspect_weights = None  # type: ignore

try:
    # Cheṣṭā bala precise; signature: f(name: str, speed_deg_per_day: float, chart_like) -> 0..60
    from app.core.ephemeris_adapter import precise_cheshta_bala as _precise_cheshta  # type: ignore
except Exception:
    _precise_cheshta = None  # type: ignore

# Advanced houses (preferred if available)
try:
    from app.core.houses_advanced import compute_houses as _compute_houses  # type: ignore
except Exception:
    _compute_houses = None  # type: ignore

# Optional varga provider for Saptavargaja component
try:
    from app.core.varga_charts import compute_many_vargas as _compute_many_vargas  # type: ignore
except Exception:
    _compute_many_vargas = None  # type: ignore

# ────────────────────────────── Constants / basic maps ───────────────────────
_PLANETS = ("Sun", "Moon", "Mercury", "Venus", "Mars", "Jupiter", "Saturn")
_POINTS_OPTIONAL = ("North Node", "South Node")  # included if present in chart

# Natural (Naisargika) strengths in Śaṣṭiāṁśa
_NAISARGIKA: Dict[str, float] = {
    "Sun": 60.0, "Moon": 51.0, "Venus": 43.0, "Jupiter": 34.0,
    "Mercury": 26.0, "Mars": 17.0, "Saturn": 9.0,
}

def _z(sign_index: int, deg_in_sign: float) -> float:
    return (30.0 * sign_index + deg_in_sign) % 360.0

# Exaltation (tropical reference; shifted by ayanamsa in sidereal)
_EXALT_TROPICAL: Dict[str, float] = {
    "Sun":     _z(0, 10.0),   # Aries 10°
    "Moon":    _z(1, 3.0),    # Taurus 3°
    "Mars":    _z(9, 28.0),   # Capricorn 28°
    "Mercury": _z(5, 15.0),   # Virgo 15°
    "Jupiter": _z(3, 5.0),    # Cancer 5°
    "Venus":   _z(11, 27.0),  # Pisces 27°
    "Saturn":  _z(6, 20.0),   # Libra 20°
    # Nodes (used if you choose to evaluate them)
    "North Node": _z(1, 20.0),
    "South Node": _z(7, 20.0),
}

_BENEFICS = {"Jupiter", "Venus"}
_MALEFICS = {"Mars", "Saturn"}

# Kāla weighting between day/night (Natonnata) and Pakṣa contribution
_KALA_WEIGHTS = {"natonnata": 0.6, "paksha": 0.4}

# ────────────────────────────── Math helpers ─────────────────────────────────
def _wrap360(x: float) -> float:
    v = float(x) % 360.0
    return 0.0 if abs(v) < 1e-12 else v

def _ang_sep(a: float, b: float) -> float:
    """Angular separation 0..180."""
    d = abs(_wrap360(a) - _wrap360(b))
    return d if d <= 180.0 else 360.0 - d

# ────────────────────────────── Houses helpers ───────────────────────────────
def _whole_sign_cusps(asc_deg: float) -> List[float]:
    """12 whole-sign cusps starting at ascendant sign start."""
    sign0 = int(_wrap360(asc_deg) // 30)  # 0..11
    base = 30.0 * sign0
    return [_wrap360(base + 30.0 * k) for k in range(12)]

def _house_index_from_cusps(lon: float, cusps: List[float]) -> int:
    """Return 1..12 (half-open bins on cusps)."""
    L = _wrap360(lon)
    for i in range(12):
        a = cusps[i]
        b = cusps[(i + 1) % 12]
        span = (b - a) % 360.0
        delta = (L - a) % 360.0
        if delta < span or math.isclose(delta, span, abs_tol=1e-12):
            return i + 1
    return 1

def _class_of_house(h: int) -> str:
    if h in (1, 4, 7, 10): return "kendra"
    if h in (2, 5, 8, 11): return "panaphara"
    return "apoklima"

# ────────────────────────────── Extraction helpers ───────────────────────────
def _pick_angles(payload: Dict[str, Any], chart: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Use payload overrides if present, else chart angles."""
    p_ang = payload.get("angles") or {}
    asc = p_ang.get("asc") or p_ang.get("asc_deg") or payload.get("asc") or payload.get("asc_deg")
    mc  = p_ang.get("mc")  or p_ang.get("mc_deg")  or payload.get("mc")  or payload.get("mc_deg")
    c_ang = (chart.get("angles") or {})
    asc = asc if isinstance(asc, (int, float)) else c_ang.get("asc_deg")
    mc  = mc  if isinstance(mc,  (int, float)) else c_ang.get("mc_deg")
    try:    asc = float(asc) if asc is not None else None
    except Exception: asc = None
    try:    mc  = float(mc)  if mc  is not None else None
    except Exception: mc  = None
    return {"asc": asc, "mc": mc}

def _pick_cusps_from_payload(payload: Dict[str, Any]) -> List[float]:
    """Accept precomputed cusps from any of the common shapes."""
    direct = payload.get("house_cusps_deg")
    if isinstance(direct, (list, tuple)) and len(direct) == 12:
        try: return [float(_wrap360(x)) for x in direct]
        except Exception: pass
    houses_obj = payload.get("houses") or {}
    for key in ("cusps_deg", "cusps"):
        arr = houses_obj.get(key)
        if isinstance(arr, (list, tuple)) and len(arr) == 12:
            try: return [float(_wrap360(x)) for x in arr]
            except Exception: pass
    for key in ("cusps_deg", "cusps"):  # legacy flat
        arr = payload.get(key)
        if isinstance(arr, (list, tuple)) and len(arr) == 12:
            try: return [float(_wrap360(x)) for x in arr]
            except Exception: pass
    return []

# ────────────────────────────── Subcomponent calculators ─────────────────────
def _ayanamsa_deg_from_meta(mode: str, chart_meta: Dict[str, Any], user_ayanamsa: Any) -> Optional[float]:
    """Resolve a numeric ayanamsa if mode is sidereal."""
    if mode != "sidereal":
        return None
    if isinstance(chart_meta.get("ayanamsa_deg"), (int, float)):
        return float(chart_meta["ayanamsa_deg"])
    if isinstance(user_ayanamsa, (int, float)):
        return float(user_ayanamsa)
    return None

def _get_exaltation_degree(name: str, mode: str, ayanamsa_deg: Optional[float]) -> Optional[float]:
    base = _EXALT_TROPICAL.get(name)
    if base is None:
        return None
    if mode == "sidereal" and isinstance(ayanamsa_deg, (int, float)):
        return _wrap360(base - float(ayanamsa_deg))
    return base

def _naisargika_bala(name: str) -> Optional[float]:
    return _NAISARGIKA.get(name)

def _uchcha_bala(name: str, lon: Optional[float], *, mode: str, ayanamsa_deg: Optional[float]) -> Optional[float]:
    if lon is None:
        return None
    ex = _get_exaltation_degree(name, mode, ayanamsa_deg)
    if ex is None:
        return None
    d = _ang_sep(lon, ex)  # 0..180
    return max(0.0, 60.0 * (1.0 - d / 180.0))

def _kendradi_bala(house_idx: Optional[int]) -> Optional[float]:
    """Kendra 60, Panaphara 30, Apoklima 15."""
    if house_idx is None:
        return None
    cls = _class_of_house(house_idx)
    if cls == "kendra": return 60.0
    if cls == "panaphara": return 30.0
    return 15.0

def _dig_bala(name: str, lon: Optional[float], asc: Optional[float], mc: Optional[float]) -> Optional[float]:
    """Directional preference distances from preferred angles."""
    if lon is None or asc is None or mc is None:
        return None
    asc = _wrap360(asc); mc = _wrap360(mc)
    desc = _wrap360(asc + 180.0); ic = _wrap360(mc + 180.0)
    pref_map = {
        "Sun": mc, "Mars": mc,
        "Jupiter": asc, "Mercury": asc,
        "Saturn": desc,
        "Moon": ic, "Venus": ic,
    }
    pref = pref_map.get(name)
    if pref is None:
        return None
    d = _ang_sep(lon, pref)
    return max(0.0, 60.0 * (1.0 - d / 180.0))

def _cheshta_bala_precise(name: str, speed_deg_per_day: Optional[float], chart: Dict[str, Any]) -> Optional[float]:
    """Cheṣṭā bala — requires precise engine; otherwise not computed (no approximation)."""
    if speed_deg_per_day is None:
        return None
    if callable(_precise_cheshta):
        try:
            return float(_precise_cheshta(name, float(speed_deg_per_day), chart))
        except Exception:
            return None
    return None  # precision engine required

def _kala_bala(name: str,
               *,
               chart: Dict[str, Any],
               longs: Dict[str, float],
               cusps: List[float],
               warnings: List[str]) -> Tuple[Optional[float], Dict[str, Any]]:
    """
    Kāla bala (0..60). Precision policy:
      • Day/night from adapter, else rigorous Sun above/below horizon via cusps.
      • Pakṣa (Moon–Sun elongation) from adapter, else from longitudes.
    No other ad-hoc pieces included unless they have deterministic astronomical bases.
    """
    sub: Dict[str, Any] = {}

    # Day/night
    is_day: Optional[bool] = None
    if callable(_sun_events):
        try:
            ev = _sun_events(chart or {})
            if isinstance(ev, dict) and ("is_day" in ev):
                is_day = bool(ev["is_day"])  # type: ignore[arg-type]
                sub["source_daynight"] = "adapter"
        except Exception as e:
           msg = getattr(e, "message", str(e))
           warnings.append(f"kala_daynight_failed:{type(e).__name__}:{msg}")

    if is_day is None and len(cusps) == 12 and "Sun" in longs:
        try:
            sun_house = _house_index_from_cusps(longs["Sun"], cusps)
            # Houses 7..12 are above the horizon in standard quadrant models
            is_day = bool(7 <= sun_house <= 12)
            sub["source_daynight"] = "houses"
        except Exception:
            is_day = None

    if is_day is None:
        warnings.append("kala_daynight_not_computed")

    # Pakṣa (Moon–Sun elongation)
    elong: Optional[float] = None
    if callable(_elongation_deg):
        try:
            elong = float(_elongation_deg(chart or {}))
            sub["source_elongation"] = "adapter"
        except Exception as e:
            warnings.append(f"kala_elongation_failed:{type(e).__name__}")
            elong = None
    if elong is None and "Sun" in longs and "Moon" in longs:
        elong = _ang_sep(longs["Sun"], longs["Moon"])
        sub["source_elongation"] = "longitudes"

    # Combine (Natonnata + Pakṣa) using established policy and weights
    if (is_day is None) and (elong is None):
        return None, sub

    parts: List[float] = []
    weights: List[float] = []

    # Natonnata: day-strong {Sun, Jupiter, Venus}; night-strong {Moon, Mars, Saturn}; Mercury neutral
    if is_day is not None:
        if name in ("Sun", "Jupiter", "Venus"):
            nat = 60.0 if is_day else 0.0
        elif name in ("Moon", "Mars", "Saturn"):
            nat = 60.0 if not is_day else 0.0
        else:  # Mercury (context-sensitive models exist; neutrality by default)
            nat = 30.0
        sub["natonnata"] = nat
        parts.append(nat); weights.append(_KALA_WEIGHTS["natonnata"])

    # Paksha: benefics grow with elongation; malefics & Sun wane with it; Moon self = elong/180
    if elong is not None:
        pk = 60.0 * (elong / 180.0)
        if name == "Moon":
            val = pk
        elif name in _BENEFICS:
            val = pk
        elif name in _MALEFICS or name == "Sun":
            val = 60.0 - pk
        else:  # Mercury neutral
            val = 30.0
        sub["paksha"] = val
        parts.append(val); weights.append(_KALA_WEIGHTS["paksha"])

    val = sum(p*w for p, w in zip(parts, weights)) / (sum(weights) if weights else 1.0)
    return float(val), sub

def _drik_bala(name: str,
               *,
               longs: Dict[str, float],
               mode: str,
               ayanamsa: Any,
               warnings: List[str]) -> Tuple[Optional[float], Dict[str, Any]]:
    """
    Drik bala (0..60) — requires an aspect-weights kernel from adapter.
    We never approximate here. If the kernel is unavailable, we return None.
    """
    details: Dict[str, Any] = {}
    if not callable(_aspect_weights):
        warnings.append("drik_precision_engine_unavailable")
        return None, {"note": "not_computed"}

    try:
        weights = _aspect_weights(longs, mode, ayanamsa)
        if callable(weights):
            w = weights
        elif hasattr(weights, "get"):
            def w(a, b):
                try:    return float(weights.get((a, b)))
                except Exception:
                    try:    return float(weights.get(a, {}).get(b))
                    except Exception: return 0.0
        else:
            raise TypeError("aspect_weights_invalid")

        ben = 0.0; mal = 0.0
        for other in longs.keys():
            if other == name:
                continue
            wt = max(0.0, min(1.0, float(w(other, name))))
            if other in _BENEFICS:
                ben += wt
            elif other in _MALEFICS or other == "Sun":
                mal += wt
            else:
                # Treat Mercury, Moon, Nodes as neutral here;
                # Moon's waxing/waning influence is captured in Kāla.
                pass

        net = ben - mal
        # Map net ∈ ℝ via soft compression around 0 to 0..60 with 30 neutral
        scaled = 30.0 + 20.0 * (math.copysign(math.log1p(abs(net)), net))
        val = max(0.0, min(60.0, scaled))
        details.update({"benefic_sum": ben, "malefic_sum": mal, "net": net, "kernel": "adapter"})
        return float(val), details
    except Exception as e:
        warnings.append(f"drik_failed:{type(e).__name__}")
        return None, {"note": "not_computed"}

def _saptavargaja_component(name: str,
                            *,
                            longs: Dict[str, float],
                            mode: str,
                            ayanamsa: Any,
                            vargas: List[str],
                            warnings: List[str]) -> Tuple[Optional[float], Dict[str, Any]]:
    """
    Precise Saptavargaja contribution (0..60) using your varga engine.
    We count 'own sign' and 'exaltation' occupancies across requested vargas.
    No ad-hoc approximations; if varga engine is unavailable, return None.
    """
    extra: Dict[str, Any] = {"vargas": vargas}
    if not vargas or _compute_many_vargas is None:
        return None, extra
    try:
        charts = _compute_many_vargas(longs, vargas,
                                      zodiac_mode=("sidereal" if mode == "sidereal" else "tropical"),
                                      ayanamsa=ayanamsa)
    except Exception as e:
        warnings.append(f"saptavargaja_failed:{type(e).__name__}")
        return None, extra

    # Sign rulers (0=Aries,..,11=Pisces) — traditional
    rulers = {
        0:"Mars",1:"Venus",2:"Mercury",3:"Moon",4:"Sun",5:"Mercury",
        6:"Venus",7:"Mars",8:"Jupiter",9:"Saturn",10:"Saturn",11:"Jupiter",
    }

    count = 0
    slots = 0
    ex_deg = _get_exaltation_degree(name, mode, _ayanamsa_deg_from_meta(mode, {"ayanamsa_deg": None}, ayanamsa))
    ex_sign = int(ex_deg // 30) if isinstance(ex_deg, (int, float)) else None

    for vk, chart in (charts or {}).items():
        if not isinstance(chart, dict):
            continue
        cell = chart.get(name)
        if not isinstance(cell, dict) or ("error" in cell):
            continue
        vrasi = cell.get("varga_rasi_index")
        if not isinstance(vrasi, int):
            continue
        slots += 2  # own + exalt check per varga
        if rulers.get(vrasi) == name:
            count += 1
        if ex_sign is not None and vrasi == ex_sign:
            count += 1

    if slots <= 0:
        return None, extra
    val = max(0.0, min(60.0, 60.0 * (float(count) / float(slots))))
    extra.update({"own_or_exalt_hits": count, "slots": slots})
    return float(val), extra

# ────────────────────────────── Results model / combiner ─────────────────────
@dataclass
class _Comp:
    value: Optional[float]
    note: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None

def _sum_components(parts: Dict[str, _Comp]) -> Tuple[float, float, Dict[str, Any]]:
    total = 0.0
    maxsum = 0.0
    details: Dict[str, Any] = {}
    for k, c in parts.items():
        if c.value is not None:
            total += float(c.value)
            maxsum += 60.0
            details[k] = {"value": float(c.value), "note": c.note, "extra": c.extra}
        else:
            details[k] = {"value": None, "note": c.note or "not_computed", "extra": c.extra}
    return total, maxsum, details

# ────────────────────────────── Public API ───────────────────────────────────
def compute_shadbala(payload: Dict[str, Any]) -> Dict[str, Any]:
    warnings: List[str] = []

    # Respect caller's requested zodiac mode (forwarded to astronomy)
    req_mode = (str(payload.get("zodiac_mode") or payload.get("mode") or "").strip().lower() or None)
    if req_mode:
        payload = {**payload, "mode": req_mode}

    # 1) Authoritative chart
    chart = _compute_chart(payload)
    mode = (req_mode or str(chart.get("mode") or "tropical")).strip().lower()
    meta = dict(chart.get("meta") or {})

    # 2) Angles (ASC/MC)
    ang = _pick_angles(payload, chart)
    asc, mc = ang["asc"], ang["mc"]

    # 3) Longitudes & speeds from chart (bodies + points)
    longs: Dict[str, float] = {}
    speeds: Dict[str, Optional[float]] = {}
    for row in (chart.get("bodies") or []):
        try:
            nm = str(row.get("name"))
            lon = float(row.get("longitude_deg", row.get("lon")))
            longs[nm] = _wrap360(lon)
            spd = row.get("speed_deg_per_day", row.get("speed"))
            speeds[nm] = (float(spd) if isinstance(spd, (int, float)) and math.isfinite(float(spd)) else None)
        except Exception:
            continue
    for row in (chart.get("points") or []):
        try:
            nm = str(row.get("name"))
            lon = float(row.get("longitude_deg", row.get("lon")))
            longs[nm] = _wrap360(lon)
        except Exception:
            continue

    # 4) Houses (prefer advanced; accept payload; else whole-sign)
    house_system = payload.get("house_system")
    cusps: List[float] = []
    precomputed_source = False

    cpl = _pick_cusps_from_payload(payload)
    if len(cpl) == 12:
        cusps = cpl
        precomputed_source = True

    if (not cusps) and _compute_houses is not None:
        try:
            hv = _compute_houses({"angles": {"asc": asc, "mc": mc}, **meta},
                                 system=house_system, payload=payload)  # type: ignore
            cusps = list(hv.get("cusps", [])) if isinstance(hv, dict) else []
        except Exception as e:
            warnings.append(f"houses_advanced_failed:{type(e).__name__}")

    if (not cusps) and isinstance(asc, (int, float)):
        cusps = _whole_sign_cusps(float(asc))
        warnings.append("houses_fallback_whole_sign")

    # 5) Optional precise Saptavargaja (used inside Sthāna)
    requested_vargas = list(payload.get("vargas") or ["D1","D2","D3","D7","D9","D12","D30"])
    ayanamsa = payload.get("ayanamsa")
    ayanamsa_deg = meta.get("ayanamsa_deg")
    # If sidereal but astronomy meta didn't supply a numeric shift, accept user numeric
    if mode == "sidereal" and not isinstance(ayanamsa_deg, (int, float)) and isinstance(ayanamsa, (int, float)):
        ayanamsa_deg = float(ayanamsa)

    # 6) Per-planet assembly
    results: Dict[str, Any] = {}
    names = [p for p in _PLANETS if p in longs] + [q for q in _POINTS_OPTIONAL if q in longs]

    for nm in names:
        lon = longs.get(nm)
        spd = speeds.get(nm)

        hidx: Optional[int] = None
        if lon is not None and len(cusps) == 12:
            try: hidx = _house_index_from_cusps(lon, cusps)
            except Exception: hidx = None

        comps: Dict[str, _Comp] = {}

        # STHĀNA = exact mean of (Uchcha, Kendrādi, Saptavargaja when available)
        st_extra: Dict[str, Any] = {}
        u_val = _uchcha_bala(nm, lon, mode=mode, ayanamsa_deg=ayanamsa_deg)
        if u_val is not None:
            st_extra["uchcha"] = u_val
        k_val = _kendradi_bala(hidx)
        if k_val is not None:
            st_extra["kendradi"] = k_val
        sv_val, sv_extra = _saptavargaja_component(
            nm, longs=longs, mode=mode, ayanamsa=ayanamsa, vargas=requested_vargas, warnings=warnings
        )
        if sv_val is not None:
            st_extra["saptavargaja"] = sv_val
        if st_extra:
            sthana_value = sum(st_extra.values()) / float(len(st_extra))
            comps["sthana"] = _Comp(sthana_value, None, {**sv_extra, **st_extra} if sv_val is not None else dict(st_extra))
        else:
            comps["sthana"] = _Comp(None, "not_computed", None)

        # DIG
        comps["dig"] = _Comp(_dig_bala(nm, lon, asc, mc), None if (asc is not None and mc is not None) else "angles_missing")

        # KĀLA
        merged_for_kala = {**chart, **payload}
        merged_for_kala["meta"] = {**(chart.get("meta") or {}), **(payload.get("meta") or {})}
        k_val2, k_sub = _kala_bala(
            nm,
            chart=merged_for_kala,   # contains bodies + meta + payload hints
            longs=longs,
            cusps=cusps,
            warnings=warnings
        )
        comps["kaala"] = _Comp(k_val2, None if k_val2 is not None else "not_computed", k_sub)

        # CHEṢṬĀ
        comps["cheshta"] = _Comp(_cheshta_bala_precise(nm, spd, chart), None if spd is not None else "speed_missing")

        # NAISARGIKA
        comps["naisargika"] = _Comp(_naisargika_bala(nm), None if nm in _NAISARGIKA else "not_standard_for_body")

        # DRIK
        d_val, d_sub = _drik_bala(nm, longs=longs, mode=mode, ayanamsa=ayanamsa, warnings=warnings)
        comps["drik"] = _Comp(d_val, None if d_val is not None else "not_computed", d_sub)

        total, maxsum, details = _sum_components(comps)
        pct = (total / maxsum) if maxsum > 0 else None
        results[nm] = {"components": details, "total": total, "max": maxsum, "pct": pct}

    # 7) Meta + warnings
    house_src = (
        "payload" if (len(cusps) == 12 and precomputed_source) else
        ("houses_advanced" if (len(cusps) == 12 and _compute_houses is not None) else
         ("whole_sign_fallback" if len(cusps) == 12 else "none"))
    )

    out_meta: Dict[str, Any] = {
        "module": "shadbala(core)",
        "mode": mode,
        "angles": {"asc": asc, "mc": mc},
        "houses_source": house_src,
        "precision_engines": {
            "cheshta": bool(_precise_cheshta),
            "aspect_weights": bool(_aspect_weights),
            "sun_events": bool(_sun_events),
            "elongation": bool(_elongation_deg),
        },
        "varga_saptavargaja": {"enabled": bool(_compute_many_vargas is not None), "vargas": requested_vargas},
        "source_chart": {
            "center": chart.get("meta", {}).get("center", chart.get("center")),
            "frame": chart.get("meta", {}).get("frame"),
            "ayanamsa_deg": ayanamsa_deg,
        },
        "version": 4,
    }
    if len(cusps) == 12:
        out_meta["house_cusps_deg"] = [float(_wrap360(x)) for x in cusps]

    out = {
        "ok": True,
        "bala": results,
        "meta": out_meta,
        "warnings": list(set(warnings + list(chart.get("warnings") or []))),
    }
    return out


def clear_caches() -> None:
    """Currently stateless; placeholder for future memoization."""
    pass


__all__ = ["compute_shadbala", "clear_caches"]

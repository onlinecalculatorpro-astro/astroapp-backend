# app/api/routes.py
from __future__ import annotations

import inspect
import json
import os
import logging
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional
from zoneinfo import ZoneInfo

from flask import Blueprint, jsonify, request

from app.version import VERSION
from app.utils.config import load_config
from app.utils.hc import flag_predictions
from app.utils.metrics import metrics
from app.utils.ratelimit import rate_limit
from app.core.validators import (
    ValidationError,
    parse_chart_payload,
    parse_prediction_payload,
    parse_rectification_payload,
)

# ───────────────────────────── numeric helpers ─────────────────────────────
def _wrap360(x: float) -> float:
    """Normalize degrees to [0, 360). Safe for any float-like."""
    try:
        v = float(x) % 360.0
        return 0.0 if abs(v) < 1e-12 else v  # collapse -0.0
    except Exception:
        return x  # leave non-numerics untouched


def _shortest_delta_deg(a2: float, a1: float) -> float:
    """Signed shortest angular delta in degrees a2 - a1 in (-180, 180]."""
    d = (float(a2) - float(a1) + 540.0) % 360.0 - 180.0
    return -180.0 if d == 180.0 else d


def _delta_arcsec(a: float, b: float) -> float:
    """Unsigned shortest difference (arcseconds) between two angles in deg."""
    return abs(_shortest_delta_deg(a, b)) * 3600.0


ARCSEC_TOL = float(os.getenv("ASTRO_ASC_TOL_ARCSEC", "3.6"))  # 0.001° default


log = logging.getLogger(__name__)
api = Blueprint("api", __name__)

DEBUG_VERBOSE = os.getenv("ASTRO_DEBUG_VERBOSE", "0").lower() in ("1", "true", "yes", "on")

# ───────────────────────────── Chart engine (primary + fallback) ─────────────────────────────
_compute_chart = None  # type: ignore
_CHART_ENGINE_NAME: Optional[str] = None

try:  # pragma: no cover
    from app.core.astronomy import compute_chart as _compute_chart  # type: ignore
    _CHART_ENGINE_NAME = "app.core.astronomy.compute_chart"
except Exception as e1:  # pragma: no cover
    try:
        from app.core.chart import compute_chart as _compute_chart  # type: ignore
        _CHART_ENGINE_NAME = "app.core.chart.compute_chart"
        log.warning("Primary astronomy.compute_chart missing; using fallback chart.compute_chart. err=%r", e1)
    except Exception as e2:
        _compute_chart = None  # type: ignore
        _CHART_ENGINE_NAME = None
        log.error("No compute_chart available: astronomy failed=%r, chart failed=%r", e1, e2)

# ───────────────────────────── Houses: prefer policy façade ─────────────────────
_HOUSES_KIND = "policy"
try:
    from app.core.house import compute_houses_with_policy as _houses_fn  # type: ignore
except Exception:
    from app.core.houses_advanced import compute_house_system as _houses_fn  # type: ignore
    _HOUSES_KIND = "legacy"

# Timescale kernel (required)
from app.core import time_kernel as _tk  # type: ignore

# Optional leap-seconds helper
try:  # pragma: no cover
    from app.core import leapseconds as _leaps  # type: ignore
except Exception:  # pragma: no cover
    _leaps = None  # type: ignore


# ───────────────────────────── ERFA-first angle helpers ─────────────────────────────
def _split_jd(jd: float) -> tuple[float, float]:
    d = int(jd // 1)
    return float(d), float(jd - d)


def _sind(a: float) -> float:  # degrees
    import math
    return math.sin(math.radians(a))


def _cosd(a: float) -> float:
    import math
    return math.cos(math.radians(a))


def _atan2d(y: float, x: float) -> float:
    import math
    if x == 0.0 and y == 0.0:
        raise ValueError("atan2(0,0) undefined")
    return _wrap360(math.degrees(math.atan2(y, x)))


def _gast_deg(jd_ut1: float, jd_tt: float) -> float:
    """
    Apparent sidereal time (GAST) in degrees, IAU 2006/2000A via ERFA when present;
    GMST-like fallback otherwise.
    """
    try:
        import erfa  # type: ignore
        d1u, d2u = _split_jd(jd_ut1)
        d1t, d2t = _split_jd(jd_tt)
        gst_rad = erfa.gst06a(d1u, d2u, d1t, d2t)
        import math
        return _wrap360(math.degrees(gst_rad))
    except Exception:
        # Fallback GMST approximation
        import math
        T = (float(jd_ut1) - 2451545.0) / 36525.0
        theta = (
            280.46061837
            + 360.98564736629 * (float(jd_ut1) - 2451545.0)
            + 0.000387933 * (T**2)
            - (T**3) / 38710000.0
        )
        return _wrap360(theta)


def _true_obliquity_deg(jd_tt: float) -> float:
    """
    True obliquity ε = mean(IAU 2006) + nutation(IAU 2000A) with ERFA;
    mean obliquity fallback if ERFA missing.
    """
    try:
        import erfa  # type: ignore
        d1, d2 = _split_jd(jd_tt)
        eps0 = erfa.obl06(d1, d2)
        _dpsi, deps = erfa.nut06a(d1, d2)
        import math
        return math.degrees(eps0 + deps)
    except Exception:
        # Meeus 2000 Eq. 22.3 mean obliquity (arcsec polynomial)
        import math
        T = (float(jd_tt) - 2451545.0) / 36525.0
        eps_arcsec = 84381.448 - 46.8150 * T - 0.00059 * (T**2) + 0.001813 * (T**3)
        return eps_arcsec / 3600.0


def _ramc_deg(jd_ut1: float, jd_tt: float, lon_east_deg: float) -> float:
    """Right Ascension of the MC = GAST + longitude (east-positive)."""
    return _wrap360(_gast_deg(jd_ut1, jd_tt) + float(lon_east_deg))


def _mc_from_ramc(ramc: float, eps: float) -> float:
    """λ_MC = atan2( sin(RAMC) * cos ε, cos(RAMC) )."""
    return _atan2d(_sind(ramc) * _cosd(eps), _cosd(ramc))


def _asc_from_phi_ramc(phi: float, ramc: float, eps: float) -> float:
    """
    Exact Ascendant using quadrant-safe arccot/atan2 form:
    ASC = arccot( - ( tan φ * sin ε + sin RAMC * cos ε ) / cos RAMC )
    """
    import math
    def _acotd(x: float) -> float:
        return _wrap360(math.degrees(math.atan2(1.0, x)))
    num = -((math.tan(math.radians(phi)) * _sind(eps)) + (_sind(ramc) * _cosd(eps)))
    den = _cosd(ramc)
    den = den if abs(den) > 1e-15 else math.copysign(1e-15, den if den != 0 else 1.0)
    return _acotd(num / den)


def _recompute_angles_exact(
    *,
    jd_ut1: float,
    jd_tt: float,
    latitude: Optional[float],
    longitude_east: Optional[float],
    mode: str,
    ayanamsa_deg: Optional[float],
) -> Optional[Dict[str, float]]:
    """Exact ASC/MC at ecliptic-of-date; sidereal = tropical − ayanāṁśa."""
    if latitude is None or longitude_east is None:
        return None
    eps = _true_obliquity_deg(jd_tt)
    ramc = _ramc_deg(jd_ut1, jd_tt, float(longitude_east))
    mc = _mc_from_ramc(ramc, eps)
    asc = _asc_from_phi_ramc(float(latitude), ramc, eps)
    if (mode or "tropical").lower() == "sidereal" and isinstance(ayanamsa_deg, (int, float)):
        asc = _wrap360(asc - float(ayanamsa_deg))
        mc = _wrap360(mc - float(ayanamsa_deg))
    return {"asc_deg": asc, "mc_deg": mc}


# ───────────────────────────── Timescales helpers ─────────────────────────────
def _datetime_to_jd_utc(dt_utc: datetime) -> float:
    """Return Julian Date (UTC). Prefer ERFA; fallback to Meeus form."""
    if dt_utc.tzinfo is None or dt_utc.tzinfo is not timezone.utc:
        raise ValueError("dt_utc must be timezone-aware UTC")
    try:
        import erfa  # PyERFA
        iy, im, id_ = dt_utc.year, dt_utc.month, dt_utc.day
        ih, iv = dt_utc.hour, dt_utc.minute
        sf = dt_utc.second + dt_utc.microsecond / 1e6
        d1, d2 = erfa.dtf2d("UTC", iy, im, id_, ih, iv, sf)
        return float(d1 + d2)
    except Exception:
        # Meeus fallback
        Y, M, D = dt_utc.year, dt_utc.month, dt_utc.day
        h, m = dt_utc.hour, dt_utc.minute
        s = dt_utc.second + dt_utc.microsecond / 1_000_000.0
        a = (14 - M) // 12
        y = Y + 4800 - a
        m_ = M + 12 * a - 3
        jdn = D + (153 * m_ + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045
        dayfrac = (h - 12) / 24.0 + m / 1440.0 + s / 86400.0
        return float(jdn + dayfrac)


def _find_kernel_callable() -> Callable[[float], Dict[str, Any]]:
    """
    Locate a JD_UTC -> timescales callable on app.core.time_kernel.
    Must return dict-like with jd_tt, jd_ut1, delta_t, delta_at, dut1, warnings?, policy?
    """
    candidates = (
        "jd_utc_to_timescales",
        "utc_jd_to_timescales",
        "timescales_from_jd_utc",
        "compute_from_jd_utc",
        "derive_timescales",
    )
    for name in candidates:
        fn = getattr(_tk, name, None)
        if callable(fn):
            def _wrap(jd_utc: float, fn=fn):
                out = fn(jd_utc)
                if is_dataclass(out):
                    return asdict(out)  # type: ignore
                if hasattr(out, "__dict__"):
                    return dict(out.__dict__)
                return dict(out)
            return _wrap

    TK = getattr(_tk, "TimeKernel", None)
    if TK is not None:
        inst = TK()  # type: ignore
        for name in ("from_jd_utc", "utc_jd_to_timescales"):
            fn = getattr(inst, name, None)
            if callable(fn):
                def _wrap2(jd_utc: float, fn=fn):
                    out = fn(jd_utc)
                    if is_dataclass(out):
                        return asdict(out)  # type: ignore
                    if hasattr(out, "__dict__"):
                        return dict(out.__dict__)
                    return dict(out)
                return _wrap2

    raise RuntimeError("time_kernel: no JD_UTC→timescales function found")


_JD_TO_TS = _find_kernel_callable()


def _compute_timescales_from_local(date_str: str, time_str: str, tz_name: str) -> Dict[str, Any]:
    """
    Convert local date/time/tz to JD(UTC), then expand to jd_tt/jd_ut1 with time_kernel.
    Returns a dict safe for JSON clients.
    """
    fmt = "%Y-%m-%d %H:%M" if time_str.count(":") == 1 else "%Y-%m-%d %H:%M:%S"

    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        raise ValidationError({"place_tz": "must be a valid IANA zone like 'Asia/Kolkata'"})

    dt_local = datetime.strptime(f"{date_str} {time_str}", fmt).replace(tzinfo=tz)
    dt_utc = dt_local.astimezone(timezone.utc)
    jd_utc = _datetime_to_jd_utc(dt_utc)

    ts = _JD_TO_TS(jd_utc)
    tz_offset_seconds = int(dt_local.utcoffset().total_seconds()) if dt_local.utcoffset() else 0

    out = {
        "jd_utc": float(jd_utc),
        "jd_tt": float(ts.get("jd_tt")),
        "jd_ut1": float(ts.get("jd_ut1")),
        "delta_t": float(ts.get("delta_t")),
        "delta_at": float(ts.get("delta_at")),
        "dut1": float(ts.get("dut1")),
        "timezone": tz_name,
        "tz_offset_seconds": tz_offset_seconds,
        "warnings": ts.get("warnings", []) or [],
    }
    if "policy" in ts:
        out["policy"] = ts["policy"]
    return out


# ─────────────────────── engine call adapters ───────────────────────
def _sig_accepts(fn, *names: str) -> Dict[str, bool]:
    try:
        params = inspect.signature(fn).parameters
    except (ValueError, TypeError):
        return {n: False for n in names}
    return {n: (n in params) for n in names}


def _call_compute_chart(payload: Dict[str, Any], ts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call compute_chart (primary or fallback), passing flexible inputs.
    Supports engines that accept a single 'payload' dict OR granular kwargs.
    Always supplies jd_ut/jd_tt/jd_ut1 from the SAME kernel to guarantee parity
    with the houses engine.
    """
    if _compute_chart is None:
        raise RuntimeError("chart_engine_unavailable")

    params = inspect.signature(_compute_chart).parameters

    # engines that accept a single 'payload' dict
    if "payload" in params:
        merged = dict(payload)
        merged["jd_ut"] = ts["jd_utc"]
        merged["jd_tt"] = ts["jd_tt"]
        merged["jd_ut1"] = ts["jd_ut1"]
        chart = _compute_chart(merged)
        chart.setdefault("meta", {})
        chart["meta"]["engine"] = _CHART_ENGINE_NAME or "unknown"
        chart["jd_ut"] = ts["jd_utc"]
        chart["jd_tt"] = ts["jd_tt"]
        chart["jd_ut1"] = ts["jd_ut1"]
        if "mode" not in chart and "mode" in payload:
            chart["mode"] = payload["mode"]
        return chart

    # granular-kwargs engines
    def pick(*cands: str) -> Optional[str]:
        for c in cands:
            if c in params:
                return c
        return None

    name_date = pick("date", "date_s", "date_str")
    name_time = pick("time", "time_s", "time_str")
    name_lat  = pick("latitude", "lat")
    name_lon  = pick("longitude", "lon")
    name_mode = pick("mode", "system")
    name_tz   = pick("place_tz", "timezone", "tz_name")
    name_bods = pick("bodies", "names", "planets")
    name_aya  = pick("ayanamsa", "ayanamsha", "aya")
    name_topo = pick("topocentric", "observer_topocentric")
    name_elev = pick("elevation_m", "elevation")

    kwargs: Dict[str, Any] = {}
    if name_date: kwargs[name_date] = payload["date"]
    if name_time: kwargs[name_time] = payload["time"]
    if name_lat:  kwargs[name_lat]  = payload["latitude"]
    if name_lon:  kwargs[name_lon]  = payload["longitude"]
    if name_mode: kwargs[name_mode] = payload["mode"]
    if name_tz:   kwargs[name_tz]   = payload.get("timezone") or payload.get("place_tz")

    if name_bods and "bodies" in payload: kwargs[name_bods] = payload["bodies"]
    if name_aya and "ayanamsa" in payload: kwargs[name_aya] = payload["ayanamsa"]
    if name_topo and "topocentric" in payload: kwargs[name_topo] = bool(payload["topocentric"])
    if name_elev and "elevation_m" in payload: kwargs[name_elev] = payload["elevation_m"]

    # Force identical timescales
    if "jd_ut" in params:  kwargs["jd_ut"]  = ts["jd_utc"]
    if "jd_tt" in params:  kwargs["jd_tt"]  = ts["jd_tt"]
    if "jd_ut1" in params: kwargs["jd_ut1"] = ts["jd_ut1"]
    if "timescales" in params: kwargs["timescales"] = ts

    chart = _compute_chart(**kwargs)

    chart["jd_ut"] = ts["jd_utc"]
    chart["jd_tt"] = ts["jd_tt"]
    chart["jd_ut1"] = ts["jd_ut1"]
    if "mode" not in chart and "mode" in payload:
        chart["mode"] = payload["mode"]

    chart.setdefault("meta", {})
    chart["meta"]["engine"] = _CHART_ENGINE_NAME or "unknown"
    return chart


def _call_compute_houses(payload: Dict[str, Any], ts: Dict[str, Any]) -> Any:
    """
    Prefer policy façade; pass jd_tt/jd_ut1 when accepted.
    Fallback to advanced compute_house_system signature if needed.
    """
    lat = float(payload["latitude"])
    lon = float(payload["longitude"])
    requested_system = (payload.get("house_system") or "").strip().lower() or None

    accepts = _sig_accepts(
        _houses_fn,
        "lat", "lon", "latitude", "longitude",
        "system", "requested_house_system", "house_system",
        "mode", "jd_ut", "jd_tt", "jd_ut1", "diagnostics", "validation",
    )

    kwargs: Dict[str, Any] = {}
    # lat/lon
    if accepts.get("lat"):
        kwargs["lat"] = lat
    elif accepts.get("latitude"):
        kwargs["latitude"] = lat
    else:
        kwargs["lat"] = lat
    if accepts.get("lon"):
        kwargs["lon"] = lon
    elif accepts.get("longitude"):
        kwargs["longitude"] = lon
    else:
        kwargs["lon"] = lon

    # requested system
    if requested_system:
        if accepts.get("system"):
            kwargs["system"] = requested_system
        elif accepts.get("requested_house_system"):
            kwargs["requested_house_system"] = requested_system
        elif accepts.get("house_system"):
            kwargs["house_system"] = requested_system

    # timescales (force same source as chart)
    if accepts.get("jd_tt"):
        kwargs["jd_tt"] = ts["jd_tt"]
    if accepts.get("jd_ut1"):
        kwargs["jd_ut1"] = ts["jd_ut1"]
    if accepts.get("jd_ut") and "jd_tt" not in kwargs and "jd_ut1" not in kwargs:
        kwargs["jd_ut"] = ts["jd_utc"]

    # optional flags passthrough
    if accepts.get("diagnostics") and "diagnostics" in payload:
        kwargs["diagnostics"] = bool(payload["diagnostics"])
    if accepts.get("validation") and "validation" in payload:
        kwargs["validation"] = bool(payload["validation"])

    return _houses_fn(**kwargs)


# ───────────────────────────── helpers: sidereal rotation ─────────────────────────────
def _rotate_sidereal_houses(h: Any, ay_deg: Optional[float]) -> Any:
    """
    Rotate a tropical houses payload to sidereal by subtracting ayanamsa degrees.
    Modifies angle keys and cusp arrays if present.
    """
    if ay_deg is None or not isinstance(h, dict):
        return h

    def rot(v: Optional[float]) -> Optional[float]:
        if v is None:
            return None
        x = (float(v) - float(ay_deg)) % 360.0
        return 0.0 if abs(x) < 1e-12 else x

    # rotate angles
    for k in ("asc", "asc_deg", "mc", "mc_deg", "vertex", "eastpoint"):
        if k in h and isinstance(h[k], (int, float)):
            h[k] = rot(h[k])

    # rotate cusps
    if "cusps" in h and isinstance(h["cusps"], list):
        h["cusps"] = [rot(c) for c in h["cusps"]]
    if "cusps_deg" in h and isinstance(h["cusps_deg"], list):
        h["cusps_deg"] = [rot(c) for c in h["cusps_deg"]]

    # tag system
    sys = h.get("system") or h.get("house_system")
    if isinstance(sys, str) and not sys.endswith("_sidereal"):
        sys = f"{sys}_sidereal"
        h["system"] = sys
        h["house_system"] = sys

    return h


def _extract_ayanamsa_from_chart(chart: Dict[str, Any]) -> Optional[float]:
    if not isinstance(chart, dict):
        return None
    meta = chart.get("meta") or {}
    ay = meta.get("ayanamsa_deg")
    if isinstance(ay, (int, float)):
        return float(ay)
    ay2 = chart.get("ayanamsa_deg")
    return float(ay2) if isinstance(ay2, (int, float)) else None


# ───────────────────────────── error helpers ─────────────────────────────
def _json_error(code: str, details: Any = None, http: int = 400):
    out: Dict[str, Any] = {"ok": False, "error": code}
    if details is not None:
        out["details"] = details
    return jsonify(out), http


# ───────────────────────────── houses normalization & parity ─────────────────────────────
def _normalize_houses_payload(h: Any) -> Any:
    """
    Normalize/standardize houses payload:
      • ensure 'cusps' and 'cusps_deg' mirror each other,
      • mirror 'system' and 'house_system',
      • ensure 'asc_deg'/'mc_deg' are present,
      • normalize all angles/cusps to [0, 360).
    """
    if not isinstance(h, dict):
        return h

    # Mirror fields
    if "cusps" not in h and "cusps_deg" in h:
        h["cusps"] = h["cusps_deg"]
    if "cusps_deg" not in h and "cusps" in h:
        h["cusps_deg"] = h["cusps"]
    if "house_system" not in h and "system" in h:
        h["house_system"] = h["system"]
    if "system" not in h and "house_system" in h:
        h["system"] = h["house_system"]
    if "asc_deg" not in h and "asc" in h:
        h["asc_deg"] = h["asc"]
    if "mc_deg" not in h and "mc" in h:
        h["mc_deg"] = h["mc"]

    # Normalize scalar angles
    for k in ("asc", "asc_deg", "mc", "mc_deg", "vertex", "eastpoint", "armc", "ramc"):
        if k in h and isinstance(h[k], (int, float)):
            h[k] = _wrap360(h[k])

    # Normalize cusp arrays
    for key in ("cusps", "cusps_deg"):
        if isinstance(h.get(key), list):
            h[key] = [_wrap360(c) if isinstance(c, (int, float)) else c for c in h[key]]

    return h


def _ensure_houses_angles_parity(
    h: Any, ts: Dict[str, Any], payload: Dict[str, Any], chart: Dict[str, Any]
) -> Any:
    """
    Recompute ASC/MC with ERFA-first math (same as astronomy engine).
    If houses ASC/MC differ beyond ARCSEC_TOL, overwrite with recomputed values
    and annotate a warning. This aligns coordinate conventions + timescales.
    """
    if not isinstance(h, dict):
        return h

    lat = payload.get("latitude")
    lon = payload.get("longitude")
    mode = (payload.get("mode") or "tropical").lower()
    ay = _extract_ayanamsa_from_chart(chart) if mode == "sidereal" else None

    recomputed = _recompute_angles_exact(
        jd_ut1=float(ts["jd_ut1"]),
        jd_tt=float(ts["jd_tt"]),
        latitude=float(lat) if isinstance(lat, (int, float)) else None,
        longitude_east=float(lon) if isinstance(lon, (int, float)) else None,
        mode=mode,
        ayanamsa_deg=ay,
    )

    if not recomputed:
        return h

    asc_new = recomputed["asc_deg"]
    mc_new = recomputed["mc_deg"]

    # current (if any)
    asc_old = h.get("asc_deg") if isinstance(h.get("asc_deg"), (int, float)) else h.get("asc")
    mc_old = h.get("mc_deg") if isinstance(h.get("mc_deg"), (int, float)) else h.get("mc")

    warn_list = h.get("warnings") or []
    changed = False

    if isinstance(asc_old, (int, float)):
        d_asc = _delta_arcsec(asc_new, float(asc_old))
        if d_asc > ARCSEC_TOL:
            h["asc_deg"] = _wrap360(asc_new); h["asc"] = h["asc_deg"]; changed = True
            warn_list.append(f"asc_corrected_for_parity({d_asc:.2f}arcsec)")
    else:
        h["asc_deg"] = _wrap360(asc_new); h["asc"] = h["asc_deg"]; changed = True

    if isinstance(mc_old, (int, float)):
        d_mc = _delta_arcsec(mc_new, float(mc_old))
        if d_mc > ARCSEC_TOL:
            h["mc_deg"] = _wrap360(mc_new); h["mc"] = h["mc_deg"]; changed = True
            warn_list.append(f"mc_corrected_for_parity({d_mc:.2f}arcsec)")
    else:
        h["mc_deg"] = _wrap360(mc_new); h["mc"] = h["mc_deg"]; changed = True

    if changed:
        h["warnings"] = warn_list

    return _normalize_houses_payload(h)


# NEW: normalize chart.bodies for predictors that expect a mapping
def _prepare_chart_for_predict(chart: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure `chart['bodies']` is a mapping {name: body_dict} for downstream
    predictors, while preserving the original list (if any) at `bodies_list`.
    Safe no-op if bodies is already a mapping.
    """
    ch = dict(chart)  # shallow copy to avoid mutating engine output
    bodies = ch.get("bodies")

    # If already a dict-like mapping, keep as-is.
    if isinstance(bodies, dict):
        return ch

    # If it's a list of dicts, build a name->dict map.
    if isinstance(bodies, list):
        name_map: Dict[str, Any] = {}
        for b in bodies:
            if isinstance(b, dict):
                nm = b.get("name")
                if isinstance(nm, str) and nm:
                    name_map[nm] = b
        if name_map:
            ch["bodies_list"] = bodies
            ch["bodies"] = name_map
    return ch


# Make horizon JSON-safe input hashable for caching/keys
def _freeze_horizon(h: Any) -> Any:
    if isinstance(h, dict):
        return tuple(sorted((k, _freeze_horizon(v)) for k, v in h.items()))
    if isinstance(h, (list, tuple)):
        return tuple(_freeze_horizon(v) for v in h)
    return h


# ───────────────────────────── endpoints ─────────────────────────────
@api.get("/api/health")
def health():
    return jsonify({"ok": True, "status": "up", "version": VERSION}), 200


@api.post("/api/calculate")
def calculate():
    try:
        body = request.get_json(force=True) or {}
        payload = parse_chart_payload(body)  # expects: date, time, place_tz, latitude, longitude, mode
        hs = str(body.get("house_system", "")).strip().lower()
        if hs:
            payload["house_system"] = hs
        # pass-through optional chart args if present
        for k in ("bodies", "ayanamsa", "topocentric", "elevation_m"):
            if k in body:
                payload[k] = body[k]
    except ValidationError as e:
        return _json_error("validation_error", e.errors(), 400)
    except Exception as e:
        return _json_error("bad_request", str(e) if DEBUG_VERBOSE else None, 400)

    tz_name = payload.get("place_tz") or payload.get("timezone") or "UTC"
    ts = _compute_timescales_from_local(payload["date"], payload["time"], tz_name)

    # chart is optional here — we won't fail the endpoint on chart errors
    chart: Dict[str, Any]
    chart_warnings: list[str] = []
    try:
        chart = _call_compute_chart(payload, ts)
    except Exception as e:
        chart = {
            "mode": payload.get("mode"),
            "jd_ut": ts["jd_utc"],
            "jd_tt": ts["jd_tt"],
            "jd_ut1": ts["jd_ut1"],
            "meta": {"engine": _CHART_ENGINE_NAME, "warnings": ["chart_failed"]},
        }
        chart_warnings.append("chart_failed")
        if DEBUG_VERBOSE:
            chart["error"] = str(e)

    # houses are required for this endpoint
    try:
        houses = _call_compute_houses(payload, ts)
        houses = _normalize_houses_payload(houses)
    except NotImplementedError as e:
        return _json_error("houses_not_implemented", str(e) if DEBUG_VERBOSE else None, 501)
    except ValueError as e:
        return _json_error("houses_error", str(e), 400)
    except Exception as e:
        return _json_error("houses_internal", str(e) if DEBUG_VERBOSE else "internal_error", 500)

    # If sidereal, rotate houses by the same ayanamsa used by the chart
    mode = (payload.get("mode") or "tropical").lower()
    if mode == "sidereal":
        ay = _extract_ayanamsa_from_chart(chart)
        if isinstance(ay, (int, float)):
            houses = _rotate_sidereal_houses(houses, ay)

    # ERFA parity fix for ASC/MC
    houses = _ensure_houses_angles_parity(houses, ts, payload, chart)

    if chart_warnings:
        chart["warnings"] = chart.get("warnings", []) + chart_warnings

    meta = {
        "timescales": ts,
        "timescales_locked": True,
        "chart_engine": _CHART_ENGINE_NAME,
        "houses_engine": _HOUSES_KIND,
    }
    return jsonify({"ok": True, "chart": chart, "houses": houses, "meta": meta}), 200


@api.post("/api/report")
def report():
    try:
        body = request.get_json(force=True) or {}
        payload = parse_chart_payload(body)
        hs = str(body.get("house_system", "")).strip().lower()
        if hs:
            payload["house_system"] = hs
        for k in ("bodies", "ayanamsa", "topocentric", "elevation_m"):
            if k in body:
                payload[k] = body[k]
    except ValidationError as e:
        return _json_error("validation_error", e.errors(), 400)

    tz_name = payload.get("place_tz") or payload.get("timezone") or "UTC"
    ts = _compute_timescales_from_local(payload["date"], payload["time"], tz_name)

    # chart optional (same behavior as /api/calculate)
    chart: Dict[str, Any]
    chart_warnings: list[str] = []
    try:
        chart = _call_compute_chart(payload, ts)
    except Exception as e:
        chart = {
            "mode": payload.get("mode"),
            "jd_ut": ts["jd_utc"],
            "jd_tt": ts["jd_tt"],
            "jd_ut1": ts["jd_ut1"],
            "meta": {"engine": _CHART_ENGINE_NAME, "warnings": ["chart_failed"]},
        }
        chart_warnings.append("chart_failed")
        if DEBUG_VERBOSE:
            chart["error"] = str(e)

    try:
        houses = _call_compute_houses(payload, ts)
        houses = _normalize_houses_payload(houses)
    except NotImplementedError as e:
        return _json_error("houses_not_implemented", str(e) if DEBUG_VERBOSE else None, 501)
    except ValueError as e:
        return _json_error("houses_error", str(e), 400)
    except Exception as e:
        return _json_error("houses_internal", str(e) if DEBUG_VERBOSE else "internal_error", 500)

    # If sidereal, rotate houses by the same ayanamsa used by the chart
    mode = (payload.get("mode") or "tropical").lower()
    if mode == "sidereal":
        ay = _extract_ayanamsa_from_chart(chart)
        if isinstance(ay, (int, float)):
            houses = _rotate_sidereal_houses(houses, ay)

    # ERFA parity fix for ASC/MC
    houses = _ensure_houses_angles_parity(houses, ts, payload, chart)

    if chart_warnings:
        chart["warnings"] = chart.get("warnings", []) + chart_warnings

    narrative = (
        "This is a placeholder narrative aligned to your mode and computed houses. "
        "Evidence will accompany predictions in /predictions."
    )

    meta = {
        "timescales": ts,
        "timescales_locked": True,
        "chart_engine": _CHART_ENGINE_NAME,
        "houses_engine": _HOUSES_KIND,
    }
    return jsonify({"ok": True, "chart": chart, "houses": houses, "narrative": narrative, "meta": meta}), 200


@api.post("/predictions")
def predictions_route():
    body = request.get_json(force=True) or {}
    try:
        payload, horizon = parse_prediction_payload(body)
        hs = str(body.get("house_system", "")).strip().lower()
        if hs:
            payload["house_system"] = hs
        for k in ("bodies", "ayanamsa", "topocentric", "elevation_m"):
            if k in body:
                payload[k] = body[k]
    except ValidationError as e:
        return _json_error("validation_error", e.errors(), 400)

    tz_name = payload.get("place_tz") or payload.get("timezone") or "UTC"
    ts = _compute_timescales_from_local(payload["date"], payload["time"], tz_name)

    # Chart is REQUIRED for predictions
    chart: Optional[Dict[str, Any]]
    chart_error: Optional[str] = None
    try:
        chart = _call_compute_chart(payload, ts)
    except Exception as e:
        chart = None
        chart_error = str(e) if DEBUG_VERBOSE else "chart_failed"

    try:
        houses = _call_compute_houses(payload, ts)
        houses = _normalize_houses_payload(houses)
    except NotImplementedError as e:
        return _json_error("houses_not_implemented", str(e) if DEBUG_VERBOSE else None, 501)
    except Exception as e:
        return _json_error("houses_internal", str(e) if DEBUG_VERBOSE else "internal_error", 500)

    if chart is None:
        return _json_error("chart_internal", chart_error or "internal_error", 500)

    # If sidereal, rotate houses by the same ayanamsa used by the chart
    mode = (payload.get("mode") or "tropical").lower()
    if mode == "sidereal":
        ay = _extract_ayanamsa_from_chart(chart)
        if isinstance(ay, (int, float)):
            houses = _rotate_sidereal_houses(houses, ay)

    # ERFA parity fix for ASC/MC
    houses = _ensure_houses_angles_parity(houses, ts, payload, chart)

    # Normalize chart for predictors that expect a mapping at chart["bodies"]
    chart_for_predict = _prepare_chart_for_predict(chart)

    # Run predictor
    try:
        from app.core.predict import predict  # local import to avoid import cycles
        preds_raw = predict(chart_for_predict, houses, horizon)
    except Exception as e:
        # Give more actionable debugging info during development
        return jsonify({"ok": False, "error": "internal_error", "message": str(e), "type": type(e).__name__}), 500

    # thresholds (with sensible defaults)
    th_path = os.environ.get("ASTRO_HC_THRESHOLDS", "config/hc_thresholds.json")
    try:
        with open(th_path, "r", encoding="utf-8") as f:
            hc = json.load(f) or {}
    except Exception:
        hc = {}

    defaults = hc.get("defaults", {}) or {}
    tau = float(defaults.get("tau", 0.88))
    floor = float(defaults.get("floor", 0.60))

    # request-body overrides (test only)
    overrides = body.get("hc_overrides") or {}
    if isinstance(overrides, dict):
        if "tau" in overrides:
            tau = float(overrides["tau"])
        if "floor" in overrides:
            floor = float(overrides["floor"])

    # env debug overrides
    env_over = os.environ.get("ASTRO_HC_DEBUG_OVERRIDES")
    if env_over:
        try:
            env_dict = json.loads(env_over)
            if isinstance(env_dict, dict):
                if "tau" in env_dict:
                    tau = float(env_dict["tau"])
                if "floor" in env_dict:
                    floor = float(env_dict["floor"])
        except Exception:
            pass

    preds = []
    for i, pr in enumerate(preds_raw):
        p = float(pr.get("probability", 0.0))
        abstained = p < floor
        hc_flag = (not abstained) and (p >= tau)
        preds.append(
            {
                "prediction_id": f"pred_{i}",
                "domain": pr.get("domain"),
                "horizon": horizon,
                "interval_start_utc": pr.get("interval", {}).get("start"),
                "interval_end_utc": pr.get("interval", {}).get("end"),
                "probability_calibrated": p,
                "hc_flag": hc_flag,
                "abstained": abstained,
                "evidence": pr.get("evidence"),
                "mode": chart.get("mode"),
                "ayanamsa_deg": chart.get("ayanamsa_deg") or (_extract_ayanamsa_from_chart(chart)),
                "notes": "QIA+calibrated placeholder; subject to M3 tuning "
                         + ("abstained" if abstained else "accepted"),
            }
        )

    # Final HC flagging (safe)
    if not overrides and not os.environ.get("ASTRO_HC_DEBUG_OVERRIDES"):
        try:
            preds = flag_predictions(preds, _freeze_horizon(horizon), th_path)
        except Exception as e:
            if DEBUG_VERBOSE:
                log.warning("flag_predictions failed: %r", e)

    meta = {
        "timescales": ts,
        "timescales_locked": True,
        "chart_engine": _CHART_ENGINE_NAME,
        "houses_engine": _HOUSES_KIND,
    }
    return jsonify({"ok": True, "predictions": preds, "meta": meta}), 200


@api.post("/rectification/quick")
def rect_quick():
    try:
        body = request.get_json(force=True) or {}
        payload, window_minutes = parse_rectification_payload(body)
    except ValidationError as e:
        return _json_error("validation_error", e.errors(), 400)

    from app.core.rectify import rectification_candidates  # local import
    result = rectification_candidates(payload, window_minutes)
    return jsonify({"ok": True, **result}), 200


@api.get("/api/openapi")
def openapi_spec():
    import yaml
    base = os.path.dirname(__file__)
    candidates = [
        os.path.join(base, "..", "openapi.yaml"),
        os.path.join(base, "..", "..", "openapi.yaml"),
    ]
    for p in candidates:
        try:
            with open(p, "r", encoding="utf-8") as f:
                spec = yaml.safe_load(f)
                return jsonify(spec), 200
        except Exception:
            continue
    return _json_error("openapi_not_found", None, 404)


@api.get("/system-validation")
def system_validation():
    cfg = load_config(os.environ.get("ASTRO_CONFIG", "config/defaults.yaml"))

    # leap status (best-effort)
    leap_status: Optional[Dict[str, Any]] = None
    if _leaps:
        for name in ("get_status", "status", "summary"):
            fn = getattr(_leaps, name, None)
            if callable(fn):
                try:
                    s = fn()
                    if is_dataclass(s):
                        leap_status = asdict(s)  # type: ignore
                    elif hasattr(s, "__dict__"):
                        leap_status = dict(s.__dict__)
                    elif isinstance(s, dict):
                        leap_status = s
                    break
                except Exception:
                    pass

    policy = {
        "houses_engine": _HOUSES_KIND,
        "polar": {
            "soft_fallback_lat_gt": float(os.getenv("ASTRO_POLAR_SOFT_LAT", "66.0")),
            "hard_reject_lat_ge": float(os.getenv("ASTRO_POLAR_HARD_LAT", "80.0")),
            "numeric_fallback": os.getenv("ASTRO_HOUSES_NUMERIC_FALLBACK", "1").lower() in ("1", "true", "yes", "on"),
        },
    }

    return jsonify(
        {
            "ok": True,
            "astronomy_accuracy": "ERFA-first timescales (JD_TT/JD_UT1), strict where required",
            "performance_slo": {"calculate_p95_ms": 800, "rect_quick_p95_s": 20},
            "mode_consistency": {
                "sidereal_default": cfg.mode == "sidereal",
                "ayanamsa": getattr(cfg, "ayanamsa", None),
            },
            "policy": policy,
            "leap_seconds": leap_status,
            "version": VERSION,
        }
    ), 200


@api.get("/metrics")
def metrics_export():
    from flask import Response
    return Response(metrics.export_prometheus(), mimetype="text/plain")


@api.get("/api/config")
@metrics.middleware("config")
@rate_limit(1)
def config_info():
    cfg_path = os.environ.get("ASTRO_CONFIG", "config/defaults.yaml")
    calib_path = os.environ.get("ASTRO_CALIBRATORS", "config/calibrators.json")
    th_path = os.environ.get("ASTRO_HC_THRESHOLDS", "config/hc_thresholds.json")

    cfg = load_config(cfg_path)
    calib_ver = None
    th_summary = None

    # timescale sample for "now" (debug aid)
    try:
        now_utc = datetime.now(timezone.utc)
        jd_now = _datetime_to_jd_utc(now_utc)
        ts_now = _JD_TO_TS(jd_now)
        ts_sample = {
            "jd_utc": float(jd_now),
            "jd_tt": float(ts_now.get("jd_tt")),
            "jd_ut1": float(ts_now.get("jd_ut1")),
            "delta_t": float(ts_now.get("delta_t")),
            "delta_at": float(ts_now.get("delta_at")),
            "dut1": float(ts_now.get("dut1")),
        }
    except Exception:
        ts_sample = None

    try:
        with open(calib_path, "r", encoding="utf-8") as f:
            calib_ver = (json.load(f) or {}).get("version")
    except Exception:
        pass

    try:
        with open(th_path, "r", encoding="utf-8") as f:
            th = json.load(f) or {}
            th_summary = {"entropy_H": th.get("entropy_H"), "defaults": th.get("defaults")}
    except Exception:
        pass

    return jsonify(
        {
            "ok": True,
            "mode": cfg.mode,
            "ayanamsa": getattr(cfg, "ayanamsa", None),
            "rate_limits_per_hour": getattr(cfg, "rate_limits_per_hour", None),
            "pro_features_enabled": getattr(cfg, "pro_features_enabled", None),
            "calibrators_version": calib_ver,
            "hc_thresholds_summary": th_summary,
            "timescale_sample": ts_sample,
            "version": VERSION,
        }
    ), 200


# ───────────────────────────── Dev endpoints ─────────────────────────────
@api.get("/api/dev/ephem")
def dev_ephem_status():
    """
    Ephemeris adapter status for debugging runtime env.
    """
    try:
        from app.core import ephemeris_adapter as ea
        eph, path = ea.load_kernel()  # returns (eph, path or None)
        return jsonify({
            "skyfield_available": ea._skyfield_available(),
            "kernel_loaded": bool(eph),
            "kernel_path": path,
            "chart_engine": _CHART_ENGINE_NAME,
        }), 200
    except Exception as e:
        return jsonify({"ok": False, "error": repr(e)}), 500


@api.post("/api/dev/echo_timescales")
def dev_echo_timescales():
    """
    Debug helper: given {date,time,place_tz} return JD_UTC/TT/UT1 etc.
    """
    body = request.get_json(force=True) or {}
    try:
        date = body.get("date") or "2000-01-01"
        time_ = body.get("time") or "12:00"
        tz = body.get("place_tz") or body.get("timezone") or "UTC"
        ts = _compute_timescales_from_local(date, time_, tz)
        return jsonify({"ok": True, "timescales": ts}), 200
    except Exception as e:
        return _json_error("timescales_error", str(e) if DEBUG_VERBOSE else None, 400)

# ───────────────────────────── Dev: fetch SPKs from Horizons (Type-2) ─────────────────────────────
import re, telnetlib, time, urllib.request, pathlib

def _horizons_spk_type2(cmd: str, start: str, stop: str, center: str = "500@10") -> Dict[str, Any]:
    """
    Ask Horizons (telnet) to build a Type-2 SPK for 'cmd' (e.g., '1','2','3','4','2060').
    Returns {"ok": bool, "message": str, "url": str|None, "saved_as": str|None}.
    """
    host, port = "horizons.jpl.nasa.gov", 6775
    tn = None
    out_txt = ""
    try:
        tn = telnetlib.Telnet(host, port, timeout=90)
        def rd(expect=b"Horizons>", t=60):
            nonlocal out_txt
            chunk = tn.read_until(expect, t)
            out_txt += chunk.decode("utf-8", "ignore")
            return chunk
        def wr(line: str):
            tn.write(line.encode("ascii") + b"\n")

        rd()                                # banner -> 'Horizons>'
        wr("PAGE=K")                        # no paging
        rd()

        # Batch block
        wr("!$$SOF")
        wr(f"COMMAND= '{cmd}'")
        wr(f"CENTER= '{center}'")
        wr("MAKE_EPHEM= 'NO'")
        wr("TABLE_TYPE= 'SPK'")
        wr(f"START_TIME= '{start}'")
        wr(f"STOP_TIME= '{stop}'")
        wr("SPK_TYPE= 2")
        wr("!$$EOF")

        # Horizons usually asks to confirm creation:
        # Look for a 'Create and deliver SPK' or similar prompt; send YES and prefer HTTP delivery.
        # We'll read a bit, then answer proactively.
        time.sleep(0.8)
        tn.write(b"Y\n")    # confirm
        time.sleep(0.4)
        tn.write(b"H\n")    # choose HTTP delivery when asked
        time.sleep(0.4)

        # Collect output for a while and close
        time.sleep(3.0)
        out_txt += tn.read_very_eager().decode("utf-8", "ignore")
    except Exception as e:
        return {"ok": False, "message": f"telnet error: {e}", "url": None, "saved_as": None}
    finally:
        try:
            if tn: tn.close()
        except Exception:
            pass

    # Try to find a URL in the transcript
    m = re.search(r"https?://\S+", out_txt)
    if not m:
        return {"ok": False, "message": "no HTTP URL found in Horizons response", "url": None, "saved_as": None}
    url = m.group(0)

    # Download to app/data/spk/<name>.bsp
    dst_dir = pathlib.Path("app/data/spk")
    dst_dir.mkdir(parents=True, exist_ok=True)
    name = {
        "1": "Ceres", "2": "Pallas", "3": "Juno", "4": "Vesta", "2060": "Chiron"
    }.get(str(cmd), f"spk_{cmd}")
    dst_path = dst_dir / f"{name}.bsp"
    try:
        urllib.request.urlretrieve(url, dst_path.as_posix())
    except Exception as e:
        return {"ok": False, "message": f"download error: {e}", "url": url, "saved_as": None}

    return {"ok": True, "message": "fetched", "url": url, "saved_as": dst_path.as_posix()}

@api.post("/api/dev/horizons_spk")
def dev_horizons_spk():
    """
    Server-side SPK fetcher: POST {targets:[ '1','2','3','4','2060' ],
    start:'2000-01-01', stop:'2050-01-01', center:'500@10'}.
    Saves files to app/data/spk/ and reports results.
    """
    body = request.get_json(force=True) or {}
    targets = [str(x) for x in (body.get("targets") or [])]
    start = body.get("start") or "2000-01-01"
    stop  = body.get("stop")  or "2050-01-01"
    center = body.get("center") or "500@10"
    if not targets:
        return _json_error("bad_request", "provide targets=['1','2','3','4','2060']", 400)

    results = []
    for cmd in targets:
        results.append({ "cmd": cmd, **_horizons_spk_type2(cmd, start, stop, center) })
    ok = all(r.get("ok") for r in results)
    return jsonify({"ok": ok, "results": results}), 200 if ok else 502


# ───────────────────────────── NEW: Ephemeris adapter endpoints ─────────────────────────────
@api.post("/api/ephemeris/diagnostics")
def ephemeris_diagnostics_endpoint():
    """Expose adapter diagnostics for the DevTools harness."""
    try:
        from app.core import ephemeris_adapter as ea
        return jsonify(ea.ephemeris_diagnostics()), 200
    except Exception as e:
        return _json_error("ephemeris_diag_error", str(e) if DEBUG_VERBOSE else None, 500)


@api.post("/api/ephemeris/longitudes")
def ephemeris_longitudes_endpoint():
    """
    Body:
      { jd_tt: float, names?: [str], frame?: "ecliptic-of-date"|"ecliptic-J2000",
        topocentric?: bool, latitude?: float, longitude?: float, elevation_m?: float }
    Returns:
      { name: longitude_deg, ... }  (nodes included if requested)
    """
    try:
        body = request.get_json(force=True) or {}
        jd_tt = float(body["jd_tt"])
        names = body.get("names")
        frame = str(body.get("frame") or "ecliptic-of-date")
        topocentric = bool(body.get("topocentric", False))
        latitude = body.get("latitude")
        longitude = body.get("longitude")
        elevation_m = body.get("elevation_m")
        from app.core import ephemeris_adapter as ea
        out = ea.ecliptic_longitudes(
            jd_tt,
            names=names,
            frame=frame,
            topocentric=topocentric,
            latitude=latitude,
            longitude=longitude,
            elevation_m=elevation_m,
        )
        return jsonify(out), 200
    except KeyError as e:
        return _json_error("validation_error", f"missing field: {e.args[0]}", 400)
    except Exception as e:
        return _json_error("ephemeris_longitudes_error", str(e) if DEBUG_VERBOSE else None, 500)


@api.post("/api/ephemeris/longitudes_and_velocities")
def ephemeris_lv_endpoint():
    """
    Body: same as /api/ephemeris/longitudes but 'names' is REQUIRED.
    Returns:
      { longitudes: {name:deg}, velocities: {name:deg_per_day (no nodes)} }
    """
    try:
        body = request.get_json(force=True) or {}
        jd_tt = float(body["jd_tt"])
        names = body["names"]
        frame = str(body.get("frame") or "ecliptic-of-date")
        topocentric = bool(body.get("topocentric", False))
        latitude = body.get("latitude")
        longitude = body.get("longitude")
        elevation_m = body.get("elevation_m")
        from app.core import ephemeris_adapter as ea
        out = ea.ecliptic_longitudes_and_velocities(
            jd_tt,
            names=names,
            frame=frame,
            topocentric=topocentric,
            latitude=latitude,
            longitude=longitude,
            elevation_m=elevation_m,
        )
        return jsonify(out), 200
    except KeyError as e:
        return _json_error("validation_error", f"missing field: {e.args[0]}", 400)
    except Exception as e:
        return _json_error("ephemeris_lv_error", str(e) if DEBUG_VERBOSE else None, 500)

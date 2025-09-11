# app/core/panchanga.py
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Vedic Panchāṅga — research-grade, sidereal-first, gold-standard numerics

Outputs (for a given civil moment or JD):
- Tithi (index 1..30, name, pakṣa) from exact nirayana (Moon − Sun)
- Nakṣatra (1..27, name, pada 1..4) from Moon’s nirayana longitude
- Yoga (1..27, name) from nirayana (Moon + Sun)
- Karaṇa (1..60, name) using classical fixed/chara sequence
- Vāra (weekday, local)
- Sunrise/Sunset at site (refraction standard −0.833°), with ERFA-based sidereal time
- Optional Muhūrta (30 equal divisions sunrise→next sunrise)

Design:
- Sidereal by subtraction of ayanāṁśa (app.core.ayanamsa.get_ayanamsa_deg)
- High-precision ecliptic-of-date longitudes (EphemerisAdapter)
- ERFA gst06a + true obliquity for alt/HA geometry
- Robust timescale handling via timescales/time_kernel if present; falls back carefully
- Root-finding (Brent-guarded) for horizon events; deterministic 1-second bucket de-dupe

Public API:
    compute_panchanga(payload: dict) -> dict
    panchanga_elements_at(jd_tt: float, *, ayanamsa_key: str = "lahiri") -> dict
    sunrise_sunset(date: str, time: str, tz: str, lat: float, lon: float, *, elevation_m: float | None = None) -> dict
    muhurta_windows(sunrise_jd_ut1: float, next_sunrise_jd_ut1: float, *, count: int = 30) -> list[tuple[float,float]]
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Iterable, Union, NamedTuple
import math

# ── Optional high-precision time helpers (same pattern as astronomy.py) ──
try:
    from app.core import time_kernel as _tk
except Exception:
    _tk = None
try:
    from app.core import timescales as _ts
except Exception:
    _ts = None

# ── ERFA: IAU SOFA routines (required for gold path; we still guard) ──
try:
    import erfa
except Exception:
    erfa = None  # we’ll degrade with clean warnings

# ── Ephemeris & ayanāṁśa ──
from app.core.ephem_singleton import TS, PLANETS  # singleton config as in western_predictive
try:
    from app.core.ephemeris_adapter import EphemerisAdapter, Config as EphemConfig
    _EPH_OK = True
except Exception:
    EphemerisAdapter = object  # type: ignore
    EphemConfig = object       # type: ignore
    _EPH_OK = False

from app.core.ayanamsa import get_ayanamsa_deg
from app.core.constants_vedic import NAKSHATRAS_27 as _NAK_27

# ────────────────────────────────────────────────────────────────────────
# Constants & naming
# ────────────────────────────────────────────────────────────────────────

_TAU = 2.0 * math.pi
_DEG = math.pi / 180.0

def _norm360(x: float) -> float:
    r = math.fmod(float(x), 360.0)
    return r + 360.0 if r < 0.0 else r

def _wrap180(x: float) -> float:
    return ((float(x) + 180.0) % 360.0) - 180.0

# 30 tithis (1..15 Śukla, 16..30 Kṛṣṇa)
_TITHI_NAMES = [
    "Pratipadā","Dvitīyā","Tṛtīyā","Caturthī","Pañcamī","Ṣaṣṭhī","Saptamī","Aṣṭamī","Navamī","Daśamī",
    "Ekādaśī","Dvādaśī","Trayodaśī","Caturdaśī","Paurṇimā/Amāvasyā seed",  # 15th label contextual
]  # We’ll expose pakṣa-aware labels below.

# 27 yogas
_YOGA_NAMES = [
    "Viṣkambha","Prīti","Āyuṣmān","Saubhāgya","Śobhana","Atigaṇḍa","Sukarmā","Dhṛti","Śūla",
    "Gaṇḍa","Vṛddhi","Dhruva","Vyāghāta","Harṣaṇa","Vajra","Siddhi","Vyatīpāta","Vāriyana",
    "Parigha","Śiva","Siddha","Sādhya","Śubha","Śukla","Brahmā","Indra","Vaidhṛti",
]

# 11 karaṇas: 7 cara + 4 sthira
_CHARA_KARANAS = ["Bava","Bālava","Kaulava","Taitila","Gara","Vaṇij","Viṣṭi (Bhadrā)"]
_STHIRA_KARANAS = ["Kiṁstughna","Śakuni","Catuṣpāda","Nāga"]

def _karana_name_from_number(n: int) -> str:
    """
    1..60 mapping:
      1           → Kiṁstughna
      2..57       → cycle of 7 cara starting at Bava
      58,59,60    → Śakuni, Catuṣpāda, Nāga
    """
    if n <= 1:
        return _STHIRA_KARANAS[0]
    if 2 <= n <= 57:
        return _CHARA_KARANAS[(n - 2) % 7]
    if n == 58:
        return _STHIRA_KARANAS[1]
    if n == 59:
        return _STHIRA_KARANAS[2]
    return _STHIRA_KARANAS[3]

# Weekday (vāra) names (Sunday=1 .. Saturday=7)
_VARA = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]

# ────────────────────────────────────────────────────────────────────────
# Low-level astro helpers (RA/Dec & obliquity, GAST)
# ────────────────────────────────────────────────────────────────────────

def _split_jd(jd: float) -> Tuple[float, float]:
    d = math.floor(jd)
    return d, jd - d

def _true_obliquity_deg(jd_tt: float) -> float:
    if erfa is None:
        # Meeus polynomial fallback
        T = (jd_tt - 2451545.0) / 36525.0
        eps_arcsec = 84381.448 - 46.8150*T - 0.00059*(T**2) + 0.001813*(T**3)
        return eps_arcsec / 3600.0
    d1, d2 = _split_jd(jd_tt)
    eps0 = erfa.obl06(d1, d2)
    _dpsi, deps = erfa.nut06a(d1, d2)
    return math.degrees(eps0 + deps)

def _gast_deg(jd_ut1: float, jd_tt: float) -> float:
    if erfa is None:
        T = (jd_ut1 - 2451545.0) / 36525.0
        theta = 280.46061837 + 360.98564736629 * (jd_ut1 - 2451545.0) + 0.000387933*(T**2) - (T**3)/38710000.0
        return _norm360(theta)
    d1u, d2u = _split_jd(jd_ut1)
    d1t, d2t = _split_jd(jd_tt)
    return _norm360(math.degrees(erfa.gst06a(d1u, d2u, d1t, d2t)))

def _ra_of_lambda_deg(lam: float, eps: float) -> float:
    # tan α = cos ε · tan λ
    s = math.sin(math.radians(lam)) * math.cos(math.radians(eps))
    c = math.cos(math.radians(lam))
    return _norm360(math.degrees(math.atan2(s, c)))

def _dec_of_lambda_deg(lam: float, eps: float) -> float:
    # sin δ = sin ε · sin λ
    s = math.sin(math.radians(eps)) * math.sin(math.radians(lam))
    s = max(-1.0, min(1.0, s))
    return math.degrees(math.asin(s))

# ────────────────────────────────────────────────────────────────────────
# Timescales
# ────────────────────────────────────────────────────────────────────────

class _TSOut(NamedTuple):
    jd_ut: float
    jd_tt: float
    jd_ut1: float
    warnings: List[str]

def _timescales_from_civil(date: str, time: str, tz: str, *, dut1_seconds: float = 0.0) -> _TSOut:
    warns: List[str] = []
    if _tk is not None:
        for fname in ("timescales_from_civil","compute_timescales","build_timescales","to_timescales","from_civil"):
            fn = getattr(_tk, fname, None)
            if callable(fn):
                try:
                    out = fn(date=date, time=time, tz=tz, dut1=dut1_seconds)
                except TypeError:
                    out = fn(date, time, tz, dut1_seconds)
                if isinstance(out, dict):
                    return _TSOut(float(out.get("jd_ut") or out.get("jd_utc")), float(out["jd_tt"]), float(out["jd_ut1"]), warns)
                if isinstance(out, (list, tuple)) and len(out) >= 3:
                    ju, jt, j1 = map(float, out[:3])
                    return _TSOut(ju, jt, j1, warns)
    if _ts is None:
        raise ValueError("timescales module not available")
    try:
        jd_ut = float(_ts.julian_day_utc(date, time, tz))
    except Exception as e:
        raise ValueError(f"Failed to compute JD_UTC from {date} {time} {tz}: {e}")
    try:
        y, m = map(int, date.split("-")[:2])
    except Exception:
        y, m = 2000, 1
    try:
        jd_tt = float(_ts.jd_tt_from_utc_jd(jd_ut, y, m))
    except Exception:
        # constant ΔT ~ 69s fallback
        jd_tt = jd_ut + 69.0/86400.0
        warns.append("deltaT_fallback_69s")
    jd_ut1 = jd_ut + float(dut1_seconds or 0.0)/86400.0
    return _TSOut(jd_ut, jd_tt, jd_ut1, warns)

# ────────────────────────────────────────────────────────────────────────
# Ephemeris wrappers (sidereal-first)
# ────────────────────────────────────────────────────────────────────────

class _EphemCtx:
    def __init__(self, frame: str = "ecliptic-of-date"):
        if not _EPH_OK:
            raise RuntimeError("EphemerisAdapter unavailable; enable app.core.ephemeris_adapter")
        self.ephem = EphemerisAdapter(EphemConfig(frame=frame, timescale=TS, planets=PLANETS))  # type: ignore

    def longitudes(self, jd_tt: float, names: List[str]) -> Dict[str, float]:
        rows = self.ephem.ecliptic_longitudes(float(jd_tt), names).get("results", [])
        if not rows:
            return {}
        return {str(r["name"]): float(r["longitude"]) for r in rows}

def _sun_moon_sidereal_longitudes(jd_tt: float, *, ayanamsa_key: str) -> Tuple[float, float, float]:
    """
    Returns (sun_trop, moon_trop, ay_deg) and caller can subtract ay_deg → nirayana.
    """
    ctx = _EphemCtx()
    got = ctx.longitudes(jd_tt, ["Sun","Moon"])
    if "Sun" not in got or "Moon" not in got:
        raise RuntimeError("ephemeris returned no Sun/Moon longitude")
    ay = float(get_ayanamsa_deg(jd_tt, ayanamsa_key))
    return float(got["Sun"]), float(got["Moon"]), ay

# ────────────────────────────────────────────────────────────────────────
# Panchanga primitives
# ────────────────────────────────────────────────────────────────────────

def _tithi_details(sun_nira: float, moon_nira: float) -> Dict[str, Any]:
    # Tithi = (Moon − Sun) / 12° in [0,30)
    diff = _norm360(moon_nira - sun_nira)
    tithi_float = diff / 12.0
    t_idx = int(math.floor(tithi_float)) + 1          # 1..30
    frac = tithi_float - (t_idx - 1)
    paksha = "Śukla" if t_idx <= 15 else "Kṛṣṇa"
    # Name
    n = (t_idx - 1) % 15
    base_name = _TITHI_NAMES[n if n < 14 else 14]
    if n == 14:
        # contextual 15th label
        base_name = "Paurṇimā" if paksha == "Śukla" else "Amāvasyā"
    return {
        "index": t_idx, "name": f"{paksha} {base_name}",
        "paksha": paksha, "progress": float(frac), "degrees_into_tithi": float(frac * 12.0),
        "longitude_diff_deg": float(diff),
    }

def _nakshatra_details(moon_nira: float) -> Dict[str, Any]:
    width = 360.0 / 27.0
    idx0 = int(math.floor(_norm360(moon_nira) / width))  # 0..26
    idx = idx0 + 1
    pos_in = _norm360(moon_nira) - idx0 * width
    pada = int(math.floor((pos_in / width) * 4.0)) + 1  # 1..4
    return {
        "index": idx, "name": _NAK_27[idx0],
        "pada": pada, "offset_deg": float(pos_in), "width_deg": float(width),
    }

def _yoga_details(sun_nira: float, moon_nira: float) -> Dict[str, Any]:
    width = 360.0 / 27.0
    s = _norm360(sun_nira + moon_nira)
    idx0 = int(math.floor(s / width))
    idx = idx0 + 1
    name = _YOGA_NAMES[idx0]
    return {"index": idx, "name": name, "sum_deg": float(s), "width_deg": float(width), "offset_deg": float(s - idx0*width)}

def _karana_details(tithi_index: int, tithi_progress: float) -> Dict[str, Any]:
    # Karana number 1..60 = floor(2 * tithi_float) + 1
    tithi_float = (tithi_index - 1) + tithi_progress
    k_num = int(math.floor(2.0 * tithi_float)) + 1
    k_name = _karana_name_from_number(k_num)
    half_in = (2.0 * tithi_float) - (k_num - 1)
    return {"number": k_num, "name": k_name, "half_progress": float(half_in)}

def _vara_name_local(date: str, tz: str) -> str:
    """
    ISO date, local tz → weekday name. We rely on timescales backend to avoid tz drift.
    """
    # minimal: use Python stdlib via timescales date normalization.
    import datetime as _dt
    try:
        from zoneinfo import ZoneInfo
        dt = _dt.datetime.fromisoformat(f"{date}T12:00:00").replace(tzinfo=ZoneInfo(tz))
        wd = dt.weekday()  # Monday=0..Sunday=6
        # Convert to Sunday=0..Saturday=6
        s0 = (wd + 1) % 7
        return _VARA[s0]
    except Exception:
        # Fallback: assume UTC
        dt = _dt.datetime.fromisoformat(f"{date}T12:00:00")
        wd = dt.weekday()
        s0 = (wd + 1) % 7
        return _VARA[s0]

# ────────────────────────────────────────────────────────────────────────
# Sunrise/Sunset — apparent center at alt = −0.833°
# ────────────────────────────────────────────────────────────────────────

@dataclass
class _Site:
    lat: float
    lon: float
    elev_m: Optional[float] = None

def _sun_altitude_deg(jd_tt: float, jd_ut1: float, *, lon_east_deg: float, lat_deg: float, ay_key: str) -> float:
    # Sun apparent ecliptic longitude (of date) → RA/Dec using true obliquity
    sun_trop, _moon_trop, ay = _sun_moon_sidereal_longitudes(jd_tt, ayanamsa_key=ay_key)
    eps = _true_obliquity_deg(jd_tt)
    ra = _ra_of_lambda_deg(sun_trop, eps)
    dec = _dec_of_lambda_deg(sun_trop, eps)
    lst = _norm360(_gast_deg(jd_ut1, jd_tt) + lon_east_deg)
    ha = _wrap180(lst - ra)
    # alt = asin( sin φ sin δ + cos φ cos δ cos H )
    sφ = math.sin(math.radians(lat_deg)); cφ = math.cos(math.radians(lat_deg))
    sδ = math.sin(math.radians(dec));     cδ = math.cos(math.radians(dec))
    cH = math.cos(math.radians(ha))
    alt = math.degrees(math.asin(sφ*sδ + cφ*cδ*cH))
    return alt

def _brent_zero(f, a: float, b: float, fa: float, fb: float, *, tol: float = 1e-6, max_iter: int = 48) -> float:
    if fa == 0.0: return a
    if fb == 0.0: return b
    if fa * fb > 0.0:
        # guarded bisection
        lo, hi = a, b
        flo, fhi = fa, fb
        for _ in range(max_iter):
            m = 0.5*(lo+hi)
            fm = f(m)
            if fm == 0.0 or abs(hi-lo) <= tol:
                return m
            if flo * fm <= 0:
                hi, fhi = m, fm
            else:
                lo, flo = m, fm
        return 0.5*(lo+hi)
    c, fc = a, fa
    d = e = b - a
    for _ in range(max_iter):
        if abs(fb) < abs(fa):
            a, b = b, a; fa, fb = fb, fa
        m = 0.5 * (a + b)
        if abs(b - a) <= tol:
            return b
        if fa != fc and fb != fc:
            s = (a*fb*fc)/((fa-fb)*(fa-fc)) + (b*fa*fc)/((fb-fa)*(fb-fc)) + (c*fa*fb)/((fc-fa)*(fc-fb))
        else:
            s = b - fb*(b-a)/(fb-fa)
        # safeguards
        cond = not ((3*a + b)/4 < s < b if a < b else b < s < (3*a + b)/4)
        cond |= (abs(s - b) >= abs(e)/2)
        cond |= (abs(e) < tol) or (abs(d) < tol)
        if cond:
            s = m
            d = e = b - a
        else:
            d, e = e, b - s
        fs = f(s)
        c, fc = a, fa
        if fa*fs < 0:
            b, fb = s, fs
        else:
            a, fa = s, fs
    return b

def sunrise_sunset_for_julian_day(
    *,
    date: str, tz: str, site: _Site, ayanamsa_key: str = "lahiri", solar_altitude_deg: float = -0.833
) -> Dict[str, Any]:
    """
    Compute sunrise/sunset for the civil date at site.
    Returns jd_ut1 for rise/set and next-day sunrise jd_ut1 (for muhūrta).
    """
    # Build midnight local → JD; scan the day in 5-min steps to bracket crossings
    # then refine with Brent on g(t) = alt(t) - alt0
    ts = _timescales_from_civil(date, "00:00:00", tz)
    jd0_utc = ts.jd_ut
    # local day span ≈ 1.2 days to be safe with high latitudes
    step = 5.0 / (24.0 * 60.0)
    alt0 = float(solar_altitude_deg)

    # utility to get alt on the fly
    def alt_at(jd_utc: float) -> Tuple[float,float,float]:
        # derive TT and UT1 for each UTC sample
        if _ts is not None:
            # quick ΔT per month
            y, m = _utc_ym(jd_utc)
            jd_tt = float(_ts.jd_tt_from_utc_jd(jd_utc, y, m)) if hasattr(_ts, "jd_tt_from_utc_jd") else jd_utc + 69.0/86400.0
        else:
            jd_tt = jd_utc + 69.0/86400.0
        jd_ut1 = jd_utc  # if DUT1 known, add here; typically small
        alt = _sun_altitude_deg(jd_tt, jd_ut1, lon_east_deg=site.lon, lat_deg=site.lat, ay_key=ayanamsa_key)
        return alt, jd_tt, jd_ut1

    def g(jd_utc: float) -> float:
        a, _, _ = alt_at(jd_utc)
        return a - alt0

    # Bracket list across local day
    samples: List[Tuple[float, float]] = []
    t = jd0_utc
    end = jd0_utc + 1.1
    prev = g(t)
    samples.append((t, prev))
    t += step
    rise_jd = None
    set_jd = None
    while t <= end + 1e-12:
        cur = g(t)
        samples.append((t, cur))
        if (prev == 0.0) or (cur == 0.0) or (prev * cur < 0.0):
            # refine
            a, fa = samples[-2]
            b, fb = samples[-1]
            jd_exact = _brent_zero(g, a, b, fa, fb, tol=5e-7)  # ~43 ms
            # classify by derivative sign: alt rising or falling
            pre = g(jd_exact - 2.0*step)
            post = g(jd_exact + 2.0*step)
            if pre < 0.0 and post > 0.0:
                if rise_jd is None:
                    rise_jd = jd_exact
            elif pre > 0.0 and post < 0.0:
                if set_jd is None:
                    set_jd = jd_exact
        prev = cur
        t += step

    # next-day sunrise (for muhūrta segmentation)
    next_ts = _timescales_from_civil(_iso_next(date), "00:00:00", tz)
    t = next_ts.jd_ut
    end = t + 1.1
    prev = g(t)
    t += step
    next_rise = None
    while t <= end + 1e-12 and next_rise is None:
        cur = g(t)
        if (prev == 0.0) or (cur == 0.0) or (prev * cur < 0.0):
            a, fa = t - step, prev
            b, fb = t, cur
            jd_exact = _brent_zero(g, a, b, fa, fb, tol=5e-7)
            pre = g(jd_exact - 2.0*step)
            post = g(jd_exact + 2.0*step)
            if pre < 0.0 and post > 0.0:
                next_rise = jd_exact
        prev = cur
        t += step

    return {
        "ok": (rise_jd is not None and set_jd is not None),
        "sunrise_jd_ut1": float(rise_jd) if rise_jd is not None else None,
        "sunset_jd_ut1": float(set_jd) if set_jd is not None else None,
        "next_sunrise_jd_ut1": float(next_rise) if next_rise is not None else None,
        "warnings": ts.warnings + ([] if erfa is not None else ["angles_fallback_meeus"]),
    }

def _utc_ym(jd_utc: float) -> Tuple[int, int]:
    # Inverse JD->UTC (month only) via simple algorithm; adequate for ΔT tables by month.
    # We keep it robust and minimal (Gregorian valid for modern dates).
    Z = int(jd_utc + 0.5)
    F = (jd_utc + 0.5) - Z
    if Z < 2299161:
        A = Z
    else:
        alpha = int((Z - 1867216.25)/36524.25)
        A = Z + 1 + alpha - int(alpha/4)
    B = A + 1524
    C = int((B - 122.1)/365.25)
    D = int(365.25 * C)
    E = int((B - D)/30.6001)
    day = B - D - int(30.6001 * E) + F
    month = E - 1 if E < 14 else E - 13
    year = C - 4716 if month > 2 else C - 4715
    return int(year), int(month)

def _iso_next(date: str) -> str:
    y, m, d = [int(x) for x in date.split("-")]
    import datetime as _dt
    return str((_dt.date(y, m, d) + _dt.timedelta(days=1)))

def muhurta_windows(sunrise_jd_ut1: float, next_sunrise_jd_ut1: float, *, count: int = 30) -> List[Tuple[float,float]]:
    if not (math.isfinite(sunrise_jd_ut1) and math.isfinite(next_sunrise_jd_ut1)):
        return []
    span = float(next_sunrise_jd_ut1 - sunrise_jd_ut1)
    if span <= 0.0:
        return []
    out: List[Tuple[float,float]] = []
    step = span / max(1, int(count))
    t = float(sunrise_jd_ut1)
    for _ in range(int(count)):
        a = t
        b = t + step
        out.append((a, b))
        t = b
    return out

# ────────────────────────────────────────────────────────────────────────
# Main element compute
# ────────────────────────────────────────────────────────────────────────

def panchanga_elements_at(jd_tt: float, *, ayanamsa_key: str = "lahiri") -> Dict[str, Any]:
    sun_trop, moon_trop, ay = _sun_moon_sidereal_longitudes(float(jd_tt), ayanamsa_key=ayanamsa_key)
    sun_nira = _norm360(sun_trop - ay)
    moon_nira = _norm360(moon_trop - ay)

    t = _tithi_details(sun_nira, moon_nira)
    n = _nakshatra_details(moon_nira)
    y = _yoga_details(sun_nira, moon_nira)
    k = _karana_details(t["index"], t["progress"])
    return {
        "ayanamsa_deg": float(ay),
        "sun_nirayana_deg": float(sun_nira),
        "moon_nirayana_deg": float(moon_nira),
        "tithi": t, "nakshatra": n, "yoga": y, "karana": k,
    }

# ────────────────────────────────────────────────────────────────────────
# Orchestrator
# ────────────────────────────────────────────────────────────────────────

def compute_panchanga(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Inputs (either provide jd_tt directly OR civil date/time/tz):
      - jd_tt (float) [preferred], optionally jd_ut and jd_ut1 for sunrise/sunset
      - OR date="YYYY-MM-DD", time="HH:MM[:SS]", tz="Area/City"
      - ayanamsa (str) default "lahiri"
      - site: latitude, longitude (degrees), elevation_m (optional) → if present, we compute sunrise/sunset and muhūrtas
      - options: include_muhurta: bool
    """
    if not _EPH_OK:
        return {"ok": False, "error": "ephemeris_unavailable", "meta": {}}

    ay_key = str(payload.get("ayanamsa", "lahiri")).strip().lower()
    warnings: List[str] = []

    jd_tt = payload.get("jd_tt")
    jd_ut = payload.get("jd_ut")
    jd_ut1 = payload.get("jd_ut1")
    if not isinstance(jd_tt, (int, float)):
        d = str(payload.get("date"))
        t = str(payload.get("time", "12:00:00"))
        tz = str(payload.get("tz") or payload.get("place_tz") or "UTC")
        ts = _timescales_from_civil(d, t, tz)
        jd_tt = ts.jd_tt
        jd_ut = ts.jd_ut
        jd_ut1 = ts.jd_ut1
        warnings.extend(ts.warnings)

    core = panchanga_elements_at(float(jd_tt), ayanamsa_key=ay_key)

    # Local weekday (vāra)
    if isinstance(payload.get("date"), str):
        tz = str(payload.get("tz") or payload.get("place_tz") or "UTC")
        vara = _vara_name_local(payload["date"], tz)
    else:
        vara = None

    out: Dict[str, Any] = {"ok": True, "elements": core, "vara": vara, "meta": {"ayanamsa": ay_key}}

    # Sunrise/Sunset + Muhūrta if site present
    lat = payload.get("latitude"); lon = payload.get("longitude")
    if isinstance(lat, (int, float)) and isinstance(lon, (int, float)) and isinstance(payload.get("date"), str):
        tz = str(payload.get("tz") or payload.get("place_tz") or "UTC")
        site = _Site(float(lat), float(lon), payload.get("elevation_m"))
        sr = sunrise_sunset_for_julian_day(date=payload["date"], tz=tz, site=site, ayanamsa_key=ay_key)
        out["sunrise_sunset"] = sr
        warnings.extend(sr.get("warnings", []))
        if bool(payload.get("include_muhurta", False)) and sr.get("sunrise_jd_ut1") and sr.get("next_sunrise_jd_ut1"):
            out["muhurta"] = muhurta_windows(sr["sunrise_jd_ut1"], sr["next_sunrise_jd_ut1"], count=30)

    if warnings:
        out["warnings"] = list(dict.fromkeys(warnings))
    return out

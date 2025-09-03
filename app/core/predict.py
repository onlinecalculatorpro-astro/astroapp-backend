# app/core/predict.py
from __future__ import annotations

"""
OCP Predictions Engine (safe scaffold)
-------------------------------------
Contract:
    predict(chart: dict, houses: dict, horizon: dict|str) -> dict

Return shape:
{
  "meta": { ... },
  "horizon": {"start": ISO, "end": ISO, "days": int, "level": "short|medium|long"},
  "predictions": [
      {
        "domain": "Career/Education",
        "probability": 0.612,
        "interval": {"start": ISO, "end": ISO},
        "evidence": {"dasha":..., "transit":..., "varga":..., "yoga":...,
                     "harmonic":..., "parts":..., "aspects":..., "stars":..., "lunar":...},
        "explain": ["short bullet about signal", "..."]
      },
      ...
  ],
  "notes": ["This is a scaffold output..."]
}

This module is resilient: if optional feature modules are missing,
it falls back to internal light-weight heuristics but preserves the
same output schema.
"""

from typing import Any, Dict, List, Tuple, Optional, Mapping
from datetime import datetime, timedelta, timezone
import json
import math
import os

# ----------------------- Optional imports with fallbacks ----------------------
def _try_import(path: str):
    mod = None
    try:
        parts = path.split('.')
        mod = __import__(path)
        for p in parts[1:]:
            mod = getattr(mod, p)
    except Exception:
        mod = None
    return mod

_ensemble = _try_import('app.core.ensemble')
_qia = _try_import('app.core.qia')
_calibration = _try_import('app.core.calibration')
_varga = _try_import('app.core.varga')
_transits = _try_import('app.core.transits')
_extras = _try_import('app.core.astro_extras')

# ------------------------------ Domain defaults -------------------------------
DOMAINS: Tuple[str, ...] = (
    "Career/Education",
    "Relationships",
    "Relocation",
    "Health",
    "Finance",
)

ENGINE_NAME = "OCP-Predict v0.3"
DEFAULT_CALIB_PATH = os.environ.get("ASTRO_CALIBRATORS", "config/calibrators.json")

# ------------------------------- Small utilities ------------------------------
def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def _norm_deg(d: float) -> float:
    d = d % 360.0
    return d if d >= 0 else d + 360.0

def _circ_delta(a: float, b: float) -> float:
    """Smallest absolute angular distance in degrees."""
    a, b = _norm_deg(a), _norm_deg(b)
    diff = abs(a - b) % 360.0
    return diff if diff <= 180.0 else 360.0 - diff

def _sigmoid(x: float) -> float:
    # numerically safe logistic
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)

# ------------------------------ Robust accessors ------------------------------
def _extract_angles(houses: Mapping[str, Any]) -> Tuple[float, float]:
    # Prefer nested "angles" dict if present
    ang = houses.get("angles") if isinstance(houses, Mapping) else None
    if isinstance(ang, Mapping):
        asc = _safe_float(ang.get("ASC"))
        mc = _safe_float(ang.get("MC"))
    else:
        asc = _safe_float(houses.get("asc_deg"))
        mc = _safe_float(houses.get("mc_deg"))
    return _norm_deg(asc), _norm_deg(mc)

def _extract_planets(chart: Mapping[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for b in chart.get("bodies", []) or []:
        try:
            name = b.get("name")
            if not name:
                continue
            lon = b.get("longitude_deg")
            out[str(name)] = _norm_deg(_safe_float(lon))
        except Exception:
            continue
    return out

# ------------------------------ Fallback engines ------------------------------
def _fb_combine(a: float, b: float, c: float, d: float) -> float:
    # Weighted mean pushed through a mild logistic to keep 0..1
    w = [0.35, 0.30, 0.20, 0.15]
    s = w[0]*a + w[1]*b + w[2]*c + w[3]*d
    return _clamp(0.08 + 0.84 * s)

def _fb_qia_adjust(p: float, ev: Mapping[str, float]) -> float:
    # Quality-Information Adjustment: gentle monotonic tweak using evidence entropy
    ks = ("harmonic","parts","aspects","stars","lunar")
    extra = sum(float(ev.get(k, 0.0)) for k in ks) / (len(ks) or 1)
    # Map extra in ~[0,0.2] to a gain in [-0.03, +0.05]
    gain = -0.03 + 0.08 * _clamp(extra / 0.2)
    q = p + gain * (0.5 - abs(p - 0.5)) * 2.0
    return _clamp(q, 0.05, 0.95)

def _fb_load_calibrators(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # Identity calibrators
        return {"domains": {d: {"type": "identity"} for d in DOMAINS}}

def _fb_apply_calibrator(domain: str, q: float, cal: Mapping[str, Any]) -> float:
    dom = (cal.get("domains") or {}).get(domain, {})
    t = (dom.get("type") or "identity").lower()
    if t == "isotonic":
        # simple 3-point isotonic-like piecewise
        pts = dom.get("points") or [[0.0,0.05],[0.5,0.5],[1.0,0.95]]
        # linear between nearest points
        x = _clamp(q)
        for i in range(1, len(pts)):
            x0,y0 = pts[i-1]
            x1,y1 = pts[i]
            if x <= x1:
                if x1 == x0:
                    return _clamp(y1)
                t = (x - x0) / (x1 - x0)
                return _clamp(y0 + t*(y1 - y0))
        return _clamp(pts[-1][1])
    elif t == "logistic":
        a = float(dom.get("a", 1.0))
        b = float(dom.get("b", 0.0))
        return _clamp(_sigmoid(a*(q - b)))
    # identity
    return _clamp(q)

def _fb_d9_navamsa_sign(lon: float) -> int:
    # 9 parts per sign; return 0..107
    segment = int(_norm_deg(lon) / (360.0/108.0))
    return segment

def _fb_d10_dashamsa_sign(lon: float) -> int:
    segment = int(_norm_deg(lon) / (360.0/120.0))
    return segment

def _fb_reinforcement_score(d9: int, d10: int) -> float:
    # Basic coherence measure
    return 0.05 + 0.45 * (1.0 - (abs(d9 - d10) % 12) / 12.0)

def _fb_detect_transits(fixed_points: Mapping[str, float], movers: Mapping[str, float]) -> float:
    # Reward closeness of slow movers to angles & luminaries
    targets = [k for k in ("Asc","MC","Sun","Moon") if k in fixed_points]
    movers_order = [k for k in ("Saturn","Jupiter","Mars") if k in movers]
    score = 0.0
    for m in movers_order:
        mlon = movers[m]
        best = min(_circ_delta(mlon, fixed_points[t]) for t in targets) if targets else 180.0
        # 0° => +0.25, 5° => ~0.18, 15° => ~0.08, 30° => ~0.02
        score += max(0.0, 0.25 * math.exp(- (best/12.0)**2))
    return _clamp(score, 0.0, 0.6)

def _fb_harmonic_longitudes(lons: Mapping[str, float], n: int) -> Dict[str, float]:
    return {k: _norm_deg(v * n) for k, v in lons.items()}

def _fb_part_of_fortune(asc: float, sun: float, moon: float, diurnal: bool = True) -> float:
    # day PoF = asc + moon - sun; night PoF = asc + sun - moon
    return _norm_deg(asc + (moon - sun if diurnal else sun - moon))

def _fb_find_aspects(lonsA: Mapping[str, float], lonsB: Mapping[str, float], orb_deg: float = 2.0) -> List[Dict[str, Any]]:
    aspects = []
    targets = [(0, "conjunction"), (60, "sextile"), (90, "square"), (120, "trine"), (180, "opposition")]
    for a, la in lonsA.items():
        for b, lb in lonsB.items():
            if a == b:
                continue
            d = _circ_delta(la, lb)
            for deg, name in targets:
                if abs(d - deg) <= orb_deg:
                    aspects.append({"a": a, "b": b, "angle": d, "aspect": name})
                    break
    return aspects

def _fb_fixed_star_ecliptics(jd_tt: float) -> Dict[str, float]:
    # Tiny fixed list (illustrative)
    return {"Regulus": 150.0, "Spica": 204.0, "Fomalhaut": 334.0, "Aldebaran": 70.0}

def _fb_star_conjunctions(planets: Mapping[str, float], stars: Mapping[str, float], orb_deg: float = 1.0) -> List[Tuple[str,str,float]]:
    out = []
    for p, plon in planets.items():
        for s, slon in stars.items():
            d = _circ_delta(plon, slon)
            if d <= orb_deg:
                out.append((p, s, d))
    return out

def _fb_lunar_phases(jd_tt: float) -> Dict[str, float]:
    # Dummy: map JD to a repeatable 0..1 illumination using a sine
    x = (jd_tt or 2451545.0) / 29.530588  # synodic approx cycles
    phase = x - math.floor(x)
    illum = 0.5 * (1 - math.cos(2 * math.pi * phase))
    return {"phase": float(phase), "illumination": float(illum)}

# Wire optionals or fallbacks
combine = _ensemble.combine if _ensemble and hasattr(_ensemble, "combine") else _fb_combine  # type: ignore
qia_adjust = _qia.qia_adjust if _qia and hasattr(_qia, "qia_adjust") else _fb_qia_adjust  # type: ignore
load_calibrators = _calibration.load_calibrators if _calibration and hasattr(_calibration, "load_calibrators") else _fb_load_calibrators  # type: ignore
apply_calibrator = _calibration.apply_calibrator if _calibration and hasattr(_calibration, "apply_calibrator") else _fb_apply_calibrator  # type: ignore

d9_navamsa_sign = _varga.d9_navamsa_sign if _varga and hasattr(_varga, "d9_navamsa_sign") else _fb_d9_navamsa_sign  # type: ignore
d10_dashamsa_sign = _varga.d10_dashamsa_sign if _varga and hasattr(_varga, "d10_dashamsa_sign") else _fb_d10_dashamsa_sign  # type: ignore
reinforcement_score = _varga.reinforcement_score if _varga and hasattr(_varga, "reinforcement_score") else _fb_reinforcement_score  # type: ignore

detect_transits = _transits.detect_transits if _transits and hasattr(_transits, "detect_transits") else _fb_detect_transits  # type: ignore

harmonic_longitudes = _extras.harmonic_longitudes if _extras and hasattr(_extras, "harmonic_longitudes") else _fb_harmonic_longitudes  # type: ignore
part_of_fortune = _extras.part_of_fortune if _extras and hasattr(_extras, "part_of_fortune") else _fb_part_of_fortune  # type: ignore
find_aspects = _extras.find_aspects if _extras and hasattr(_extras, "find_aspects") else _fb_find_aspects  # type: ignore
fixed_star_ecliptics = _extras.fixed_star_ecliptics if _extras and hasattr(_extras, "fixed_star_ecliptics") else _fb_fixed_star_ecliptics  # type: ignore
star_conjunctions = _extras.star_conjunctions if _extras and hasattr(_extras, "star_conjunctions") else _fb_star_conjunctions  # type: ignore
lunar_phases = _extras.lunar_phases if _extras and hasattr(_extras, "lunar_phases") else _fb_lunar_phases  # type: ignore

# -------------------------------- Horizon helpers -----------------------------
def _parse_horizon(h: Any) -> Dict[str, Any]:
    """
    Accepts:
      - dict with start/end (ISO date or datetime)
      - string "short"/"long"/"medium"
      - None -> defaults to 180 days from today (UTC)
    Returns dict with start, end (ISO date), days, and level.
    """
    now = datetime.now(timezone.utc).date()
    if isinstance(h, dict):
        start_raw = h.get("start")
        end_raw = h.get("end")
        def _parse_date(x):
            if x is None:
                return None
            try:
                # Allow date or datetime ISO
                return datetime.fromisoformat(str(x).replace("Z","")).date()
            except Exception:
                return None
        s = _parse_date(start_raw) or now
        e = _parse_date(end_raw) or (s + timedelta(days=180))
        if e < s:
            e = s
        days = (e - s).days
        level = "short" if days <= 90 else "medium" if days <= 240 else "long"
        return {"start": s.isoformat(), "end": e.isoformat(), "days": days, "level": level}
    else:
        level = str(h or "medium").lower()
        days = 90 if level == "short" else 240 if level == "long" else 180
        s = now
        e = now + timedelta(days=days)
        return {"start": s.isoformat(), "end": e.isoformat(), "days": days, "level": level}

def _make_intervals(hz: Dict[str, Any]) -> List[Dict[str, str]]:
    days = max(1, int(hz.get("days", 180)))
    level = hz.get("level", "medium")
    # choose bucket size
    bucket = 30 if level == "short" else 45 if level == "medium" else 60
    n = max(3, min(12, (days + bucket - 1) // bucket))
    s = datetime.fromisoformat(hz["start"]).date()
    out = []
    for i in range(n):
        bs = s + timedelta(days=i * bucket)
        be = min(s + timedelta(days=(i+1) * bucket), datetime.fromisoformat(hz["end"]).date())
        out.append({"start": bs.isoformat() + "T00:00:00Z", "end": be.isoformat() + "T00:00:00Z"})
        if be >= datetime.fromisoformat(hz["end"]).date():
            break
    return out

# ---------------------------- Evidence primitives -----------------------------
def natal_points(chart: Dict[str, Any], houses: Dict[str, Any]) -> Dict[str, float]:
    asc, mc = _extract_angles(houses)
    pts = {"Asc": asc, "MC": mc}
    for name, lon in _extract_planets(chart).items():
        if name in ("Sun","Moon","Jupiter","Saturn","Mars","Venus","Mercury"):
            pts[name] = lon
    return pts

def evidence_contributions(chart: Dict[str, Any], houses: Dict[str, Any]) -> Dict[str, float]:
    pts = natal_points(chart, houses)
    planets = _extract_planets(chart)

    moon = pts.get("Moon", 0.0)
    mc = pts.get("MC", 0.0)

    # Varga reinforcement
    d9_moon = d9_navamsa_sign(moon)
    d10_mc = d10_dashamsa_sign(mc)
    v_score = reinforcement_score(d9_moon, d10_mc)

    # Transit score
    movers = {k: v for k, v in pts.items() if k in ("Jupiter","Saturn","Mars")}
    fixed = {"Sun": pts.get("Sun",0.0), "Moon": moon, "Asc": pts.get("Asc",0.0), "MC": mc}
    t_score = detect_transits(fixed, movers)

    # Dasha proxy (smooth 0.25..0.75 band over Moon fractional position in sign)
    d_frac = (moon % 30.0) / 30.0
    d_score = 0.25 + 0.5 * abs(0.5 - d_frac)

    # Yoga proxy
    y_score = 0.06 if _circ_delta(pts.get("Jupiter",0.0), moon) < 15.0 else 0.02

    # Extras — all safe-guarded
    try:
        harmonic_longitudes(planets, 7)  # we don't need the values, just the presence
        harmonic_longitudes(planets, 9)
        harmonic_score = 0.05
    except Exception:
        harmonic_score = 0.05

    try:
        asc = pts.get("Asc", 0.0)
        pof = part_of_fortune(asc, planets.get("Sun",0.0), planets.get("Moon",0.0), True)
        parts_score = 0.1 if _circ_delta(pof, planets.get("Sun",0.0)) < 30.0 else 0.05
    except Exception:
        parts_score = 0.05

    try:
        aspects = find_aspects(planets, planets, orb_deg=2.0)
        ben = sum(1 for a in aspects if a.get("aspect") in ("trine","sextile"))
        chal = sum(1 for a in aspects if a.get("aspect") in ("square","opposition"))
        aspect_score = _clamp(0.1 + 0.05*ben - 0.02*chal, 0.01, 0.25)
    except Exception:
        aspect_score = 0.1

    try:
        stars = fixed_star_ecliptics(_safe_float(chart.get("jd_tt"), 2451545.0))
        conj = star_conjunctions(planets, stars, orb_deg=1.0)
        star_score = _clamp(0.02 + 0.03 * len(conj), 0.02, 0.15)
    except Exception:
        star_score = 0.02

    try:
        lunar_info = lunar_phases(_safe_float(chart.get("jd_tt"), 2451545.0))
        lunar_score = 0.05 + _clamp(float(lunar_info.get("illumination", 0.5)), 0.0, 1.0) * 0.1
    except Exception:
        lunar_score = 0.07

    return {
        "dasha": round(float(d_score), 3),
        "transit": round(float(t_score), 3),
        "varga": round(float(v_score), 3),
        "yoga": round(float(y_score), 3),
        "harmonic": round(float(harmonic_score), 3),
        "parts": round(float(parts_score), 3),
        "aspects": round(float(aspect_score), 3),
        "stars": round(float(star_score), 3),
        "lunar": round(float(lunar_score), 3)
    }

def calibrated_probability(ev: Dict[str, float], domain: str) -> float:
    base = combine(ev["dasha"], ev["transit"], ev["varga"], ev["yoga"])
    enhanced = (
        ev.get("harmonic",0.0) * 0.10 +
        ev.get("parts",0.0) * 0.15 +
        ev.get("aspects",0.0) * 0.20 +
        ev.get("stars",0.0) * 0.10 +
        ev.get("lunar",0.0) * 0.05
    )
    enhanced_base = _clamp(float(base) + float(enhanced), 0.05, 0.95)
    q = qia_adjust(enhanced_base, ev)
    calibrators = load_calibrators(DEFAULT_CALIB_PATH)
    result = apply_calibrator(domain, q, calibrators)
    return float(_clamp(result, 0.01, 0.99))

# --------------------------------- API surface --------------------------------
def predict(chart: Dict[str, Any], houses: Dict[str, Any], horizon: Any) -> Dict[str, Any]:
    """
    Public entrypoint called by /predictions.
    Accepts flexible 'horizon' and returns a structured payload (see module docstring).
    """
    hz = _parse_horizon(horizon)
    intervals = _make_intervals(hz)

    predictions: List[Dict[str, Any]] = []
    ev = evidence_contributions(chart, houses)

    for i, domain in enumerate(DOMAINS):
        p = calibrated_probability(ev, domain)
        interval = intervals[i % len(intervals)]
        explain = []
        if ev.get("transit", 0) > 0.2: explain.append("Significant transit proximity to angles/lights.")
        if ev.get("aspects", 0) >= 0.15: explain.append("Supportive aspect network.")
        if ev.get("parts", 0) >= 0.1: explain.append("Arabic Parts alignment boost.")
        if not explain: explain.append("Baseline signals with moderate confidence.")
        predictions.append({
            "domain": domain,
            "probability": round(p, 3),
            "interval": interval,
            "evidence": ev,
            "explain": explain,
        })

    meta = {
        "engine": ENGINE_NAME,
        "window_days": hz["days"],
        "domains": list(DOMAINS),
        "has_calibrators": os.path.exists(DEFAULT_CALIB_PATH),
        "has_optionals": {
            "ensemble": bool(_ensemble),
            "qia": bool(_qia),
            "calibration": bool(_calibration),
            "varga": bool(_varga),
            "transits": bool(_transits),
            "extras": bool(_extras),
        }
    }

    return {
        "meta": meta,
        "horizon": hz,
        "predictions": predictions,
        "notes": [
            "This is a resilient scaffold. When advanced engines are present, they are used automatically.",
            "Probabilities are calibrated if calibrators are provided; otherwise identity mapping is applied."
        ]
    }

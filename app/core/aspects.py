# app/core/aspects.py
from __future__ import annotations
"""
Aspects Engine — research-grade, deterministic, and testable.

Design goals
- Pure geometry: works on *any* set of longitudes you feed it (tropical or sidereal).
  (If you use sidereal, pass your ayanamsa-adjusted longitudes here; no hidden offsets.)
- Robust circular math (wraps, signed/absolute separations).
- Clear orb policy: base per-aspect caps + per-body moieties; combine via policy ('min' default).
- Applying/Separating flag using velocities when available (deg/day).
- Extensible aspect set (majors default; minors/quintiles/etc. available).
- Deterministic output with stable sort order and structured records.

Quickstart
----------
from app.core.aspects import compute_aspects, default_config

hits = compute_aspects(
    longitudes={"Sun": 123.4, "Moon": 98.7, "Mars": 310.2},
    velocities={"Sun": 0.9856, "Moon": 13.1764, "Mars": 0.5240},  # optional
    cfg=default_config()
)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Iterable, Literal, Any
import math
import os

# ───────────────────────────── angle helpers ─────────────────────────────

def _norm360(x: float) -> float:
    """Wrap angle to [0, 360)."""
    r = float(x) % 360.0
    return 0.0 if math.isclose(r, 0.0, abs_tol=1e-12) else r

def _signed_short_arc(a: float, b: float) -> float:
    """
    Signed shortest difference (b - a) in degrees on (-180, 180].
    Positive if b is ahead of a going forward through the zodiac.
    """
    return ((b - a + 540.0) % 360.0) - 180.0

def _abs_sep(a: float, b: float) -> float:
    """Absolute separation in [0, 180]."""
    return abs(_signed_short_arc(a, b))

# ───────────────────────────── aspect model ─────────────────────────────

@dataclass(frozen=True)
class AspectSpec:
    key: str                 # stable identifier, e.g., "conjunction"
    angle: float             # exact angle in degrees
    max_orb_deg: float       # absolute cap for this aspect
    enabled: bool = True
    weight: float = 1.0      # optional strength weight (used in scoring/ties)

# Classic major aspects (Lilly-like caps; tweak via env if desired)
_DEF_ORB_CONJ = float(os.getenv("ASPECT_ORB_CONJ", "8"))
_DEF_ORB_OPP  = float(os.getenv("ASPECT_ORB_OPP",  "8"))
_DEF_ORB_TRI  = float(os.getenv("ASPECT_ORB_TRI",  "7"))
_DEF_ORB_SQR  = float(os.getenv("ASPECT_ORB_SQR",  "6"))
_DEF_ORB_SEX  = float(os.getenv("ASPECT_ORB_SEX",  "5"))
_DEF_ORB_QCX  = float(os.getenv("ASPECT_ORB_QCX",  "3"))  # quincunx/inconjunct

MAJOR_ASPECTS: Dict[str, AspectSpec] = {
    "conjunction": AspectSpec("conjunction", 0.0,   _DEF_ORB_CONJ, weight=1.30),
    "opposition":  AspectSpec("opposition",  180.0, _DEF_ORB_OPP,  weight=1.20),
    "trine":       AspectSpec("trine",       120.0, _DEF_ORB_TRI,  weight=1.00),
    "square":      AspectSpec("square",       90.0, _DEF_ORB_SQR,  weight=0.95),
    "sextile":     AspectSpec("sextile",      60.0, _DEF_ORB_SEX,  weight=0.85),
    "quincunx":    AspectSpec("quincunx",    150.0, _DEF_ORB_QCX,  weight=0.60),
}

# Optional minors / harmonics (disabled by default)
MINOR_ASPECTS: Dict[str, AspectSpec] = {
    "semisextile":   AspectSpec("semisextile",   30.0, 2.0,  enabled=False, weight=0.40),
    "semisquare":    AspectSpec("semisquare",    45.0, 2.5,  enabled=False, weight=0.45),
    "sesquisquare":  AspectSpec("sesquisquare", 135.0, 2.5,  enabled=False, weight=0.45),
    "quintile":      AspectSpec("quintile",      72.0, 1.5,  enabled=False, weight=0.35),
    "biquintile":    AspectSpec("biquintile",   144.0, 1.5,  enabled=False, weight=0.35),
    "novile":        AspectSpec("novile",        40.0, 1.0,  enabled=False, weight=0.30),
    "septile":       AspectSpec("septile",     360.0/7.0, 1.0, enabled=False, weight=0.30),
    "biseptile":     AspectSpec("biseptile",  2*360.0/7.0, 0.8, enabled=False, weight=0.28),
    "triseptile":    AspectSpec("triseptile", 3*360.0/7.0, 0.7, enabled=False, weight=0.26),
}

# ───────────────────────────── moiety policy ─────────────────────────────

BodyClass = Literal["light", "inner", "social", "outer", "asteroid", "node", "angle", "point", "other"]

def _classify_body(name: str) -> BodyClass:
    n = (name or "").strip().lower()
    if n in ("sun", "moon"):
        return "light"
    if n in ("mercury", "venus", "mars"):
        return "inner"
    if n in ("jupiter", "saturn"):
        return "social"
    if n in ("uranus", "neptune", "pluto"):
        return "outer"
    if n in ("north node", "south node", "rahu", "ketu", "true node", "mean node"):
        return "node"
    if n in ("asc", "ascendant", "mc", "ic", "dsc", "midheaven", "imum coeli"):
        return "angle"
    if n.startswith("cusp") or n.startswith("house"):
        return "angle"
    if n in ("ceres","pallas","juno","vesta","chiron"):
        return "asteroid"
    return "point"

# Lilly-style moieties (half-orbs) by class; tweak via env if desired
MOIETIES_BY_CLASS: Dict[BodyClass, float] = {
    "light":   float(os.getenv("ASPECT_MOIETY_LIGHT",   "6.0")),  # Sun~8, Moon~6 traditional → moiety ≈6 for safety
    "inner":   float(os.getenv("ASPECT_MOIETY_INNER",   "3.0")),
    "social":  float(os.getenv("ASPECT_MOIETY_SOCIAL",  "3.0")),
    "outer":   float(os.getenv("ASPECT_MOIETY_OUTER",   "2.0")),
    "node":    float(os.getenv("ASPECT_MOIETY_NODE",    "1.5")),
    "angle":   float(os.getenv("ASPECT_MOIETY_ANGLE",   "4.0")),
    "asteroid":float(os.getenv("ASPECT_MOIETY_AST",     "1.0")),
    "point":   float(os.getenv("ASPECT_MOIETY_POINT",   "2.0")),
    "other":   float(os.getenv("ASPECT_MOIETY_OTHER",   "2.0")),
}

def _moiety_for(name: str) -> float:
    return MOIETIES_BY_CLASS.get(_classify_body(name), MOIETIES_BY_CLASS["other"])

OrbCombine = Literal["min", "sum_capped", "aspect_only", "moiety_only"]

@dataclass
class AspectConfig:
    aspects: Dict[str, AspectSpec] = field(default_factory=lambda: {**MAJOR_ASPECTS, **MINOR_ASPECTS})
    orb_combine: OrbCombine = "min"
    enable_minors: bool = False
    # Filter knobs
    max_orb_percent: float = 1.0       # keep hits with |delta| ≤ percent*allowed_orb (1.0 = full)
    min_score: float = 0.0             # keep hits with score ≥ this (score = weight * (1 - |delta|/allowed_orb))
    # Diagnostics / meta
    include_raw_values: bool = False   # include exact separations & deltas in result rows
    # Sorting
    sort_by: Literal["tightness", "score", "a_name", "angle"] = "tightness"

def default_config() -> AspectConfig:
    cfg = AspectConfig()
    # disable minors unless explicitly enabled
    if not cfg.enable_minors:
        for k, spec in list(cfg.aspects.items()):
            if k in MINOR_ASPECTS:
                cfg.aspects[k] = AspectSpec(spec.key, spec.angle, spec.max_orb_deg, enabled=False, weight=spec.weight)
    return cfg

# ───────────────────────────── core math ─────────────────────────────

def _allowed_orb(a_name: str, b_name: str, spec: AspectSpec, combine: OrbCombine) -> float:
    """
    Combine per-aspect cap and moieties per point into a working orb.
    - 'min'         : min( aspect_cap, moiety(a)+moiety(b) )   ← recommended
    - 'sum_capped'  : clamp( moiety(a)+moiety(b), 0, aspect_cap )
    - 'aspect_only' : aspect_cap
    - 'moiety_only' : moiety(a)+moiety(b)
    """
    cap = float(spec.max_orb_deg)
    msum = float(_moiety_for(a_name) + _moiety_for(b_name))
    if combine == "min":
        return min(cap, msum)
    if combine == "sum_capped":
        return min(cap, msum)
    if combine == "aspect_only":
        return cap
    if combine == "moiety_only":
        return msum
    return min(cap, msum)

def _nearest_enabled_aspect(sep: float, aspects: Dict[str, AspectSpec]) -> Tuple[Optional[AspectSpec], float]:
    """
    Given separation in [0,180], return (nearest enabled AspectSpec, delta = sep - angle).
    """
    best: Optional[AspectSpec] = None
    best_delta = 0.0
    best_abs = float("inf")
    for spec in aspects.values():
        if not spec.enabled:
            continue
        d = sep - spec.angle
        ad = abs(d)
        if ad < best_abs or (math.isclose(ad, best_abs, abs_tol=1e-12) and spec.weight > (best.weight if best else 0.0)):
            best, best_abs, best_delta = spec, ad, d
    return best, best_delta

# ───────────────────────────── output model ─────────────────────────────

@dataclass
class AspectHit:
    a: str
    b: str
    aspect: str
    angle: float
    orb: float                 # |sep - angle| (deg)
    orb_allowed: float         # working orb (deg)
    exact: bool
    applying: Optional[bool]   # None if velocities not provided
    score: float               # weight * (1 - orb/orb_allowed) in [0, weight]
    # optional extras for diagnostics
    a_lon: Optional[float] = None
    b_lon: Optional[float] = None
    a_vel: Optional[float] = None
    b_vel: Optional[float] = None
    sep: Optional[float] = None         # absolute separation [0..180]
    delta: Optional[float] = None       # signed (sep - angle)

# ───────────────────────────── applying/separating logic ─────────────────────────────

def _applying_flag(a_lon: float, b_lon: float, a_vel: float, b_vel: float, angle: float) -> Optional[bool]:
    """
    Determine if the pair is applying toward the given aspect angle.
    Let sep = |wrap_shortest(b-a)|. y = sep - angle. Applying if y is decreasing (y*dy/dt < 0).
    dy/dt = dsep/dt; with sep = |x|, x = wrap_shortest(b-a), dsep/dt = sign(x) * (b_vel - a_vel).
    """
    x = _signed_short_arc(a_lon, b_lon)          # (-180, 180]
    sep = abs(x)                                  # [0, 180]
    y = sep - angle
    # numerical noise guard near exactness
    if abs(y) < 1e-10:
        return None
    dsep_dt = (1.0 if x >= 0.0 else -1.0) * (b_vel - a_vel)
    if not (math.isfinite(dsep_dt) and abs(dsep_dt) < 1e6):
        return None
    return (y * dsep_dt) < 0.0

# ───────────────────────────── main API ─────────────────────────────

def _pairs(names: List[str]) -> Iterable[Tuple[str, str]]:
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            yield names[i], names[j]

def compute_aspects(
    longitudes: Dict[str, float],
    velocities: Optional[Dict[str, float]] = None,
    *,
    cfg: Optional[AspectConfig] = None,
) -> Dict[str, Any]:
    """
    Compute aspects for all unordered pairs in `longitudes`.
    Inputs:
      longitudes: {"Sun": λ_sun, "Moon": λ_moon, ...} (degrees 0..360, any zodiac)
      velocities: optional {"Sun": λdot_sun, ...} (deg/day); used for applying/separating.
      cfg: AspectConfig (see default_config()).

    Returns:
      {
        "hits": [AspectHit, ...] sorted per cfg.sort_by,
        "meta": { "count_pairs": N, "cfg": { ... } }
      }
    """
    if cfg is None:
        cfg = default_config()

    # Build the working aspect set (respect enable_minors)
    aspects = dict(cfg.aspects)
    if not cfg.enable_minors:
        for k in MINOR_ASPECTS.keys():
            if k in aspects:
                aspects[k] = AspectSpec(aspects[k].key, aspects[k].angle, aspects[k].max_orb_deg, enabled=False, weight=aspects[k].weight)

    names = [k for k in longitudes.keys()]
    hits: List[AspectHit] = []

    for A, B in _pairs(names):
        a_lon = _norm360(longitudes[A])
        b_lon = _norm360(longitudes[B])

        sep = _abs_sep(a_lon, b_lon)  # [0, 180]
        spec, delta = _nearest_enabled_aspect(sep, aspects)
        if spec is None:
            continue

        allowed = _allowed_orb(A, B, spec, cfg.orb_combine)
        orb = abs(delta)

        # filter by orb limits/percent
        if orb > allowed * float(cfg.max_orb_percent):
            continue

        # Applying/separating
        applying: Optional[bool] = None
        if velocities is not None and A in velocities and B in velocities:
            a_vel = float(velocities[A])
            b_vel = float(velocities[B])
            applying = _applying_flag(a_lon, b_lon, a_vel, b_vel, spec.angle)
        else:
            a_vel = b_vel = None  # type: ignore

        # score in [0, weight] (1 at exact, 0 at edge of allowed orb)
        tight = max(0.0, 1.0 - (orb / allowed)) if allowed > 0.0 else 0.0
        score = float(spec.weight * tight)

        if score < cfg.min_score:
            continue

        hit = AspectHit(
            a=A, b=B,
            aspect=spec.key,
            angle=spec.angle,
            orb=orb,
            orb_allowed=allowed,
            exact=(orb <= 1e-8),
            applying=applying,
            score=score,
            a_lon=a_lon if cfg.include_raw_values else None,
            b_lon=b_lon if cfg.include_raw_values else None,
            a_vel=(float(velocities.get(A)) if (velocities and A in velocities and cfg.include_raw_values) else None),  # type: ignore[union-attr]
            b_vel=(float(velocities.get(B)) if (velocities and B in velocities and cfg.include_raw_values) else None),  # type: ignore[union-attr]
            sep=(sep if cfg.include_raw_values else None),
            delta=(delta if cfg.include_raw_values else None),
        )
        hits.append(hit)

    # Sorting
    if cfg.sort_by == "score":
        hits.sort(key=lambda h: (-h.score, h.aspect, h.a, h.b))
    elif cfg.sort_by == "a_name":
        hits.sort(key=lambda h: (h.a, h.b, h.aspect, h.orb))
    elif cfg.sort_by == "angle":
        hits.sort(key=lambda h: (h.angle, h.orb, h.a, h.b))
    else:  # "tightness"
        hits.sort(key=lambda h: (h.orb / max(h.orb_allowed, 1e-12), -h.score, h.a, h.b))

    meta = {
        "count_pairs": len(names) * (len(names) - 1) // 2,
        "cfg": {
            "orb_combine": cfg.orb_combine,
            "enable_minors": cfg.enable_minors,
            "max_orb_percent": cfg.max_orb_percent,
            "min_score": cfg.min_score,
            "sort_by": cfg.sort_by,
        }
    }
    return {"hits": hits, "meta": meta}

# ───────────────────────────── convenience APIs ─────────────────────────────

def aspects_between(
    a_name: str, a_lon: float,
    b_name: str, b_lon: float,
    a_vel: Optional[float] = None, b_vel: Optional[float] = None,
    *, cfg: Optional[AspectConfig] = None
) -> Optional[AspectHit]:
    """
    Compute the single best (nearest) aspect between two points, or None if outside orb.
    """
    if cfg is None:
        cfg = default_config()
    a_lon = _norm360(a_lon); b_lon = _norm360(b_lon)
    sep = _abs_sep(a_lon, b_lon)
    aspects = dict(cfg.aspects)
    if not cfg.enable_minors:
        for k in MINOR_ASPECTS.keys():
            if k in aspects:
                aspects[k] = AspectSpec(aspects[k].key, aspects[k].angle, aspects[k].max_orb_deg, enabled=False, weight=aspects[k].weight)
    spec, delta = _nearest_enabled_aspect(sep, aspects)
    if spec is None:
        return None
    allowed = _allowed_orb(a_name, b_name, spec, cfg.orb_combine)
    orb = abs(delta)
    if orb > allowed * cfg.max_orb_percent:
        return None
    applying = None
    if a_vel is not None and b_vel is not None:
        applying = _applying_flag(a_lon, b_lon, float(a_vel), float(b_vel), spec.angle)
    tight = max(0.0, 1.0 - (orb / allowed)) if allowed > 0.0 else 0.0
    score = float(spec.weight * tight)
    return AspectHit(
        a=a_name, b=b_name, aspect=spec.key, angle=spec.angle,
        orb=orb, orb_allowed=allowed, exact=(orb <= 1e-8),
        applying=applying, score=score
    )

def enable_all_minors(cfg: AspectConfig) -> AspectConfig:
    """Utility to enable all minor aspects in an existing config."""
    for k in MINOR_ASPECTS:
        if k in cfg.aspects:
            spec = cfg.aspects[k]
            cfg.aspects[k] = AspectSpec(spec.key, spec.angle, spec.max_orb_deg, enabled=True, weight=spec.weight)
    cfg.enable_minors = True
    return cfg

# ───────────────────────────── JSON-friendly serialization ─────────────────────────────

def hits_to_dict(hits: List[AspectHit]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for h in hits:
        d = {
            "a": h.a, "b": h.b,
            "aspect": h.aspect,
            "angle": h.angle,
            "orb": h.orb,
            "orb_allowed": h.orb_allowed,
            "exact": h.exact,
            "applying": h.applying,
            "score": h.score,
        }
        if h.a_lon is not None: d["a_lon"] = h.a_lon
        if h.b_lon is not None: d["b_lon"] = h.b_lon
        if h.a_vel is not None: d["a_vel"] = h.a_vel
        if h.b_vel is not None: d["b_vel"] = h.b_vel
        if h.sep   is not None: d["sep"]   = h.sep
        if h.delta is not None: d["delta"] = h.delta
        out.append(d)
    return out

__all__ = [
    "AspectSpec",
    "AspectConfig",
    "default_config",
    "MAJOR_ASPECTS",
    "MINOR_ASPECTS",
    "compute_aspects",
    "aspects_between",
    "enable_all_minors",
    "hits_to_dict",
]

# app/core/aspects.py
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Literal, Iterable, Any
import itertools
import math
import random

__all__ = [
    "AspectConfig",
    "AspectHit",
    "ConflictPolicy",
    "FDRGate",
    "compute_aspects",   # PURE geometry function (list of hits)
    "analyze_aspects",   # pipeline (geometry → score → optional FDR → features)
    "run_aspects_api",   # HTTP adapter (no recursion)
]

# ─────────────────────────────────────────────────────────────────────────────
# Core math helpers
# ─────────────────────────────────────────────────────────────────────────────

def _norm360(x: float) -> float:
    v = float(x) % 360.0
    return 0.0 if math.isclose(v, 0.0, abs_tol=1e-12) else v

def _sep_small(a: float, b: float) -> float:
    """Smallest separation on circle in [0, 180]."""
    d = abs(_norm360(a - b))
    return d if d <= 180.0 else 360.0 - d

def _near(value: float, target: float) -> float:
    """Unsigned deviation |value - target| on [0, 180] (inputs already folded)."""
    return abs(value - target)

def _safe_get(d: Optional[Dict[str, float]], k: str, default: float) -> float:
    if not isinstance(d, dict):
        return default
    v = d.get(k, default)
    try:
        return float(v)
    except Exception:
        return default

# ─────────────────────────────────────────────────────────────────────────────
# Aspect catalog & default orbs
# ─────────────────────────────────────────────────────────────────────────────

MAJOR_ASPECTS: Dict[str, float] = {
    "conjunction": 0.0,
    "opposition": 180.0,
    "trine": 120.0,
    "square": 90.0,
    "sextile": 60.0,
}

MINOR_ASPECTS: Dict[str, float] = {
    "quincunx": 150.0,
    "semisextile": 30.0,
    "semisquare": 45.0,
    "sesquiquadrate": 135.0,
    "quintile": 72.0,
    "biquintile": 144.0,
}

DEFAULT_ORBS_DEG: Dict[str, float] = {
    # majors (conservative but practical)
    "conjunction": 8.0,
    "opposition": 8.0,
    "trine": 7.0,
    "square": 6.0,
    "sextile": 5.0,
    # minors
    "quincunx": 3.0,
    "semisextile": 2.0,
    "semisquare": 2.0,
    "sesquiquadrate": 2.0,
    "quintile": 2.0,
    "biquintile": 2.0,
    # antiscia / declination
    "antiscia": 1.5,
    "contra-antiscia": 1.5,
    "parallel": 0.5,         # declination degrees
    "contraparallel": 0.5,   # declination degrees
}

# Orb scaling heuristics per body (editable)
DEFAULT_BODY_WEIGHTS: Dict[str, float] = {
    "Sun": 1.00, "Moon": 1.00,
    "Mercury": 0.85, "Venus": 0.85, "Mars": 0.85,
    "Jupiter": 0.95, "Saturn": 0.95,
    "Uranus": 0.80, "Neptune": 0.80, "Pluto": 0.80,
    "North Node": 0.70, "South Node": 0.70,
}

# ─────────────────────────────────────────────────────────────────────────────
# Configuration & results
# ─────────────────────────────────────────────────────────────────────────────

AspectFamily = Literal["zodiacal", "antiscia", "declination"]

@dataclass
class AspectConfig:
    include_minors: bool = False
    include_antiscia: bool = True
    include_contra_antiscia: bool = True
    include_declination: bool = True
    # Orbs & weights
    orbs_deg: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_ORBS_DEG))
    body_orb_weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_BODY_WEIGHTS))
    # Minimum tightness (observed/orb) to emit a hit (0..1)
    min_tightness: float = 0.05
    # Optional white/allow list of pairs
    allowed_pairs: Optional[Iterable[Tuple[str, str]]] = None
    # NEW: restrict which aspect names are allowed (e.g., {"conjunction","square"})
    allowed_aspects: Optional[set[str]] = None

@dataclass
class AspectHit:
    a: str
    b: str
    family: AspectFamily
    aspect: str
    exact_deg: float               # target angle (or 0 for parallels)
    separation_deg: float          # measured separation (zodiacal) or |sum-180| etc.
    orb_deg: float                 # allowed orb used for this pair
    delta_deg: float               # |separation - exact| (zodiacal) OR absolute offset (others)
    tightness: float               # 1 - delta/orb, clipped [0,1]
    score: float                   # initial = tightness (policy may reweight)
    meta: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k in ("exact_deg","separation_deg","orb_deg","delta_deg","tightness","score"):
            d[k] = float(d[k])
        return d

# ─────────────────────────────────────────────────────────────────────────────
# Geometry — helpers per family
# ─────────────────────────────────────────────────────────────────────────────

def _pair_orb(a: str, b: str, base: float, weights: Dict[str, float]) -> float:
    wa = _safe_get(weights, a, 0.8)
    wb = _safe_get(weights, b, 0.8)
    return base * max(0.25, min(1.25, 0.5 * (wa + wb)))  # clamp extremes

def _zodiacal_hits_for_pair(a: str, la: float, b: str, lb: float, cfg: AspectConfig) -> List[AspectHit]:
    sep = _sep_small(_norm360(lb), _norm360(la))  # [0, 180]
    # Build catalog per config
    catalog = dict(MAJOR_ASPECTS)
    if cfg.include_minors:
        catalog.update(MINOR_ASPECTS)
    if cfg.allowed_aspects is not None:
        catalog = {k: v for k, v in catalog.items() if k in cfg.allowed_aspects}

    hits: List[AspectHit] = []
    for name, angle in catalog.items():
        base_orb = cfg.orbs_deg.get(name, DEFAULT_ORBS_DEG.get(name, 0.0))
        orb = _pair_orb(a, b, base_orb, cfg.body_orb_weights)
        delta = _near(sep, angle)
        if delta <= orb:
            tight = max(0.0, 1.0 - (delta / max(1e-12, orb)))
            if tight >= cfg.min_tightness:
                hits.append(AspectHit(
                    a=min(a,b), b=max(a,b),
                    family="zodiacal", aspect=name, exact_deg=angle,
                    separation_deg=sep, orb_deg=orb, delta_deg=delta,
                    tightness=tight, score=tight, meta={}
                ))
    return hits

def _antiscia_hits_for_pair(a: str, la: float, b: str, lb: float, cfg: AspectConfig) -> List[AspectHit]:
    la = _norm360(la); lb = _norm360(lb)
    s = (la + lb) % 360.0
    out: List[AspectHit] = []

    # antiscia
    want_antiscia = cfg.include_antiscia and (cfg.allowed_aspects is None or "antiscia" in cfg.allowed_aspects)
    if want_antiscia:
        target = 180.0
        base = cfg.orbs_deg.get("antiscia", DEFAULT_ORBS_DEG["antiscia"])
        orb = _pair_orb(a, b, base, cfg.body_orb_weights)
        delta = min(abs(s - 180.0), 360.0 - abs(s - 180.0))
        if delta <= orb:
            tight = max(0.0, 1.0 - (delta / max(1e-12, orb)))
            if tight >= cfg.min_tightness:
                out.append(AspectHit(
                    a=min(a,b), b=max(a,b),
                    family="antiscia", aspect="antiscia", exact_deg=target,
                    separation_deg=delta, orb_deg=orb, delta_deg=delta,
                    tightness=tight, score=tight, meta={}
                ))

    # contra-antiscia
    want_contra = cfg.include_contra_antiscia and (cfg.allowed_aspects is None or "contra-antiscia" in cfg.allowed_aspects)
    if want_contra:
        target = 0.0
        base = cfg.orbs_deg.get("contra-antiscia", DEFAULT_ORBS_DEG["contra-antiscia"])
        orb = _pair_orb(a, b, base, cfg.body_orb_weights)
        delta = min(s, 360.0 - s)  # distance to 0°/360°
        if delta <= orb:
            tight = max(0.0, 1.0 - (delta / max(1e-12, orb)))
            if tight >= cfg.min_tightness:
                out.append(AspectHit(
                    a=min(a,b), b=max(a,b),
                    family="antiscia", aspect="contra-antiscia", exact_deg=0.0,
                    separation_deg=delta, orb_deg=orb, delta_deg=delta,
                    tightness=tight, score=tight, meta={}
                ))
    return out

def _declination_hits_for_pair(a: str, da: Optional[float], b: str, db: Optional[float], cfg: AspectConfig) -> List[AspectHit]:
    if da is None or db is None:
        return []
    out: List[AspectHit] = []

    # Only consider declination if either not restricted or explicitly included
    want_decl = cfg.include_declination and (
        cfg.allowed_aspects is None or
        ("parallel" in cfg.allowed_aspects or "contraparallel" in cfg.allowed_aspects)
    )
    if not want_decl:
        return out

    base_p = cfg.orbs_deg.get("parallel", DEFAULT_ORBS_DEG["parallel"])
    base_c = cfg.orbs_deg.get("contraparallel", DEFAULT_ORBS_DEG["contraparallel"])
    orb_p = _pair_orb(a, b, base_p, cfg.body_orb_weights)
    orb_c = _pair_orb(a, b, base_c, cfg.body_orb_weights)

    same = (da >= 0) == (db >= 0)

    # parallel
    if same and (cfg.allowed_aspects is None or "parallel" in cfg.allowed_aspects):
        delta = abs(da - db)
        if delta <= orb_p:
            tight = max(0.0, 1.0 - (delta / max(1e-12, orb_p)))
            if tight >= cfg.min_tightness:
                out.append(AspectHit(
                    a=min(a,b), b=max(a,b),
                    family="declination", aspect="parallel", exact_deg=0.0,
                    separation_deg=delta, orb_deg=orb_p, delta_deg=delta,
                    tightness=tight, score=tight, meta={"dec_a": da, "dec_b": db}
                ))

    # contraparallel
    if (not same) and (cfg.allowed_aspects is None or "contraparallel" in cfg.allowed_aspects):
        delta = abs(da + db)  # distance to zero when mirrored
        if delta <= orb_c:
            tight = max(0.0, 1.0 - (delta / max(1e-12, orb_c)))
            if tight >= cfg.min_tightness:
                out.append(AspectHit(
                    a=min(a,b), b=max(a,b),
                    family="declination", aspect="contraparallel", exact_deg=0.0,
                    separation_deg=delta, orb_deg=orb_c, delta_deg=delta,
                    tightness=tight, score=tight, meta={"dec_a": da, "dec_b": db}
                ))
    return out

# ─────────────────────────────────────────────────────────────────────────────
# PURE Geometry API (NO side effects; NO recursion)
# ─────────────────────────────────────────────────────────────────────────────

def _pair_iter(names: List[str], whitelist: Optional[Iterable[Tuple[str, str]]]) -> Iterable[Tuple[str, str]]:
    if whitelist is None:
        yield from itertools.combinations(names, 2)
        return
    allow = {(min(x), max(x)) for x in whitelist}
    for a, b in itertools.combinations(names, 2):
        key = (min(a, b), max(a, b))
        if key in allow:
            yield a, b

def compute_aspects(
    points_deg: Dict[str, float],
    *,
    declinations_deg: Optional[Dict[str, float]] = None,
    config: Optional[AspectConfig] = None
) -> List[AspectHit]:
    """
    PURE geometry: compute hits for a single chart (or inter-chart if you union inputs).
    - points_deg: longitudes (deg) per body.
    - declinations_deg: optional declinations (deg) for parallels.
    - config: AspectConfig.
    Returns a flat list of AspectHit.
    """
    cfg = config or AspectConfig()
    names = sorted(points_deg.keys())
    out: List[AspectHit] = []
    for a, b in _pair_iter(names, cfg.allowed_pairs):
        la = _norm360(points_deg[a]); lb = _norm360(points_deg[b])
        out.extend(_zodiacal_hits_for_pair(a, la, b, lb, cfg))
        out.extend(_antiscia_hits_for_pair(a, la, b, lb, cfg))
        if cfg.include_declination:
            da = None if declinations_deg is None else declinations_deg.get(a)
            db = None if declinations_deg is None else declinations_deg.get(b)
            out.extend(_declination_hits_for_pair(a, da, b, db, cfg))
    return out

# ─────────────────────────────────────────────────────────────────────────────
# Policy Layer — weighting, conflict handling, features
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CategoryWeights:
    zodiacal: float = 1.0
    antiscia: float = 0.5
    declination: float = 0.6

@dataclass
class ConflictPolicy:
    strategy: Literal["zodiacal_dominant", "weighted_sum", "max_score"] = "zodiacal_dominant"
    cat_weights: CategoryWeights = field(default_factory=CategoryWeights)
    min_tightness: float = 0.2    # prefilter guard before any scoring

def _key_pair(hit: AspectHit) -> Tuple[str, str]:
    return (hit.a, hit.b)

def score_hits(hits: List[AspectHit], policy: Optional[ConflictPolicy] = None) -> List[AspectHit]:
    """
    Reweights hit.score by category weights; optionally down-selects by policy.
    """
    pol = policy or ConflictPolicy()
    # apply category weights
    for h in hits:
        if h.tightness < pol.min_tightness:
            h.score = 0.0
            continue
        w = getattr(pol.cat_weights, h.family)
        h.score = h.tightness * float(w)

    if pol.strategy == "weighted_sum":
        return hits

    # group by pair
    by_pair: Dict[Tuple[str,str], List[AspectHit]] = {}
    for h in hits:
        by_pair.setdefault(_key_pair(h), []).append(h)

    reweighted: List[AspectHit] = []
    for key, grp in by_pair.items():
        if pol.strategy == "max_score":
            best = max(grp, key=lambda x: (x.score, x.tightness), default=None)
            if best and best.score > 0.0:
                reweighted.append(best)
            continue

        # zodiacal_dominant:
        zods = [h for h in grp if h.family == "zodiacal" and h.score > 0.0]
        if zods:
            best_z = max(zods, key=lambda x: (x.score, x.tightness))
            modifiers = [h for h in grp if h.family != "zodiacal" and h.score > 0.0]
            boost = sum(m.score for m in modifiers)
            best = AspectHit(**{**best_z.as_dict(), "meta": {**best_z.meta, "modifiers": [m.as_dict() for m in modifiers]}})
            best.score = min(1.0, best.score + 0.5 * boost)
            reweighted.append(best)
        else:
            best = max(grp, key=lambda x: (x.score, x.tightness), default=None)
            if best and best.score > 0.0:
                reweighted.append(best)

    return reweighted

def build_pair_features(hits: List[AspectHit]) -> Dict[Tuple[str,str], Dict[str, float]]:
    """
    Aggregate per (a,b) pair:
      - max_zodiacal, max_antiscia, max_declination
      - combined_score (max of family maxima)
      - flags: count_* for quick heuristics
    """
    feats: Dict[Tuple[str,str], Dict[str, float]] = {}
    for h in hits:
        k = _key_pair(h)
        f = feats.setdefault(k, {
            "max_zodiacal": 0.0,
            "max_antiscia": 0.0,
            "max_declination": 0.0,
            "count_zodiacal": 0.0,
            "count_antiscia": 0.0,
            "count_declination": 0.0,
        })
        if h.family == "zodiacal":
            f["max_zodiacal"] = max(f["max_zodiacal"], h.score)
            f["count_zodiacal"] += 1.0
        elif h.family == "antiscia":
            f["max_antiscia"] = max(f["max_antiscia"], h.score)
            f["count_antiscia"] += 1.0
        else:
            f["max_declination"] = max(f["max_declination"], h.score)
            f["count_declination"] += 1.0
    for k, f in feats.items():
        f["combined_score"] = max(f["max_zodiacal"], f["max_antiscia"], f["max_declination"])
    return feats

# ─────────────────────────────────────────────────────────────────────────────
# Validation gate — permutation FDR (Benjamini–Hochberg)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FDRGate:
    num_permutations: int = 0      # 0 disables FDR
    q_level: float = 0.05
    seed: Optional[int] = 123
    preserve_decl_signs: bool = True

def _bh_threshold(pvals: List[float], q: float) -> float:
    m = len(pvals)
    if m == 0:
        return 0.0
    pairs = sorted((p, i+1) for i, p in enumerate(pvals))  # (p, rank)
    cutoff = 0.0
    for p, r in pairs:
        if p <= (r / m) * q:
            cutoff = max(cutoff, p)
    return cutoff

def _shuffle_longitudes(points: Dict[str, float], rng: random.Random) -> Dict[str, float]:
    # independent random rotations (break pair structure while preserving marginal)
    return {k: _norm360(rng.random() * 360.0) for k in points.keys()}

def _permute_declinations(decs: Optional[Dict[str, float]], rng: random.Random) -> Optional[Dict[str, float]]:
    if decs is None:
        return None
    names = list(decs.keys())
    vals = [decs[k] for k in names]
    rng.shuffle(vals)
    return {k: float(v) for k, v in zip(names, vals)}

def apply_fdr(
    points_deg: Dict[str, float],
    hits: List[AspectHit],
    *,
    declinations_deg: Optional[Dict[str, float]] = None,
    config: Optional[AspectConfig] = None,
    gate: Optional[FDRGate] = None
) -> Tuple[List[AspectHit], Dict[int, float], float]:
    """
    Permutation-based p-values on hit.score; BH FDR gate at q_level.
    Returns (kept_hits, pvals_by_index, bh_cutoff).
    """
    cfg = config or AspectConfig()
    g = gate or FDRGate()
    if g.num_permutations <= 0:
        # trivial pass-through with unit p-values
        return hits, {i: 1.0 for i in range(len(hits))}, 1.0

    rng = random.Random(g.seed)
    # observed scores
    obs_scores = [max(0.0, float(h.score)) for h in hits]
    null_scores: List[float] = []

    for _ in range(max(1, g.num_permutations)):
        p_shuf = _shuffle_longitudes(points_deg, rng)
        d_shuf = _permute_declinations(declinations_deg, rng) if g.preserve_decl_signs else None
        null_hits = compute_aspects(p_shuf, declinations_deg=d_shuf, config=cfg)
        null_hits = score_hits(null_hits, ConflictPolicy(strategy="weighted_sum"))  # same scoring
        null_scores.extend(max(0.0, float(h.score)) for h in null_hits)

    # conservative p-value: proportion of null >= observed (add-one to avoid zeros)
    N = max(1, len(null_scores))
    null_sorted = sorted(null_scores)
    pvals: Dict[int, float] = {}
    for i, s in enumerate(obs_scores):
        # binary search for first >= s
        lo, hi = 0, N
        while lo < hi:
            mid = (lo + hi) // 2
            if null_sorted[mid] >= s:
                hi = mid
            else:
                lo = mid + 1
        ge_count = N - lo
        p = (ge_count + 1) / (N + 1)   # +1 smoothing
        pvals[i] = p

    cutoff = _bh_threshold(list(pvals.values()), g.q_level)
    kept = [h for i, h in enumerate(hits) if pvals[i] <= cutoff] if cutoff > 0.0 else []
    return kept, pvals, cutoff

# ─────────────────────────────────────────────────────────────────────────────
# Convenience — one-shot pipeline
# ─────────────────────────────────────────────────────────────────────────────

def analyze_aspects(
    points_deg: Dict[str, float],
    *,
    declinations_deg: Optional[Dict[str, float]] = None,
    geometry_config: Optional[AspectConfig] = None,
    conflict_policy: Optional[ConflictPolicy] = None,
    fdr_gate: Optional[FDRGate] = None
) -> Dict[str, Any]:
    """
    End-to-end:
      1) geometry → hits
      2) policy weighting/selection → scored_hits
      3) optional FDR → filtered_hits
      4) features per pair
    """
    gcfg = geometry_config or AspectConfig()
    hits = compute_aspects(points_deg, declinations_deg=declinations_deg, config=gcfg)  # PURE call
    scored = score_hits(hits, conflict_policy or ConflictPolicy())
    kept, pvals, cutoff = apply_fdr(points_deg, scored, declinations_deg=declinations_deg,
                                    config=gcfg, gate=fdr_gate or FDRGate(num_permutations=0))
    features = build_pair_features(kept if (fdr_gate and fdr_gate.num_permutations > 0) else scored)

    return {
        "hits": [h.as_dict() for h in scored],
        "hits_after_fdr": [h.as_dict() for h in kept] if (fdr_gate and fdr_gate.num_permutations > 0) else None,
        "pvals": pvals if (fdr_gate and fdr_gate.num_permutations > 0) else None,
        "bh_cutoff": cutoff if (fdr_gate and fdr_gate.num_permutations > 0) else None,
        "features_by_pair": {f"{a}~{b}": feats for (a,b), feats in features.items()},
        "config": {
            "geometry": asdict(gcfg),
            "policy": asdict(conflict_policy or ConflictPolicy()),
            "fdr_gate": asdict(fdr_gate or FDRGate()),
        }
    }

# ─────────────────────────────────────────────────────────────────────────────
# HTTP-layer adapter (NO name collision with compute_aspects)
# ─────────────────────────────────────────────────────────────────────────────

def run_aspects_api(
    *,
    positions: Optional[Dict[str, float]] = None,
    orbs: Optional[Dict[str, float]] = None,
    aspects: Optional[List[str]] = None,
    declinations: Optional[Dict[str, float]] = None,
    mode: str = "tropical",
    **kwargs
) -> Dict[str, Any]:
    """
    Thin API adapter around the pure engine.
    - positions: Dict of body_name -> longitude_deg
    - declinations: Dict of body_name -> declination_deg (for parallel/contraparallel)
    - orbs: per-aspect orbs, e.g., {"conjunction":8, "square":6}
    - aspects: list of aspect names to include; if omitted, defaults apply
    - mode: "tropical" or "sidereal" (metadata only here)
    """
    if not positions:
        return {"aspects": [], "count": 0, "meta": {"error": "no_positions"}}

    config = AspectConfig()

    # Apply orb overrides (validate)
    if orbs:
        for k, v in orbs.items():
            if isinstance(v, (int, float)) and v >= 0:
                config.orbs_deg[str(k)] = float(v)

    # Apply aspect filtering / feature toggles
    if aspects:
        allowed = {str(a) for a in aspects}
        config.allowed_aspects = allowed
        # minors requested?
        config.include_minors = any(a in MINOR_ASPECTS for a in allowed)
        # antiscia families only if explicitly requested
        config.include_antiscia = "antiscia" in allowed
        config.include_contra_antiscia = "contra-antiscia" in allowed
        # declination families only if explicitly requested
        wants_decl = ("parallel" in allowed) or ("contraparallel" in allowed)
        config.include_declination = wants_decl

    result = analyze_aspects(
        positions,
        declinations_deg=declinations,
        geometry_config=config,
        conflict_policy=ConflictPolicy(strategy="zodiacal_dominant"),
        fdr_gate=FDRGate(num_permutations=0),  # disabled in API path
    )

    return {
        "aspects": result["hits"],
        "count": len(result["hits"]),
        "config": {
            "orbs_used": config.orbs_deg,
            "include_minors": config.include_minors,
            "include_antiscia": config.include_antiscia,
            "include_contra_antiscia": config.include_contra_antiscia,
            "include_declination": config.include_declination,
            "mode": mode,
            "allowed_aspects": sorted(list(config.allowed_aspects)) if config.allowed_aspects else None,
        },
        "meta": {
            "engine": "aspects.py",
            "features_computed": len(result["features_by_pair"]),
        },
    }

# ─────────────────────────────────────────────────────────────────────────────
# Minimal manual check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pts = {"A": 0.0, "B": 90.0, "C": 120.0}
    out = run_aspects_api(positions=pts, aspects=["square","trine"], orbs={"square": 6, "trine": 6})
    from pprint import pprint
    pprint(out)

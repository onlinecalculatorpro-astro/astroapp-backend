# app/core/common_predictive.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Callable, Literal, TYPE_CHECKING
import math, random
from datetime import datetime, date as _date

# Preloaded singletons (don’t reload kernels)
from app.core.ephem_singleton import TS, PLANETS

# Optional ephemeris adapter (type only at runtime if available)
try:
    from app.core.ephemeris_adapter import EphemerisAdapter, Config as EphemConfig
    _EPH_OK = True
except Exception:  # pragma: no cover - optional dep
    _EPH_OK = False
    EphemerisAdapter = object  # type: ignore
    EphemConfig = object       # type: ignore

__all__ = [
    "TS", "PLANETS", "_EPH_OK",
    "PROF",
    "TAU", "EPS",
    "norm360", "wrap180", "angdiff", "sign_index", "is_finite",
    "_to_date", "_jd_from_date",
    "timescales_from_civil", "_ts_resolve", "compute_houses",
    "FeatureFn", "pearson_corr", "permutation_pvalue_corr", "bh_fdr",
    "EvalResult", "evaluate_univariate", "holdout_replicate", "validate_predictions",
]

# Lightweight telemetry
PROF: Dict[str, int] = {"ephem_calls": 0, "refinements": 0}

# ── math helpers (shared) ────────────────────────────────────────────────────
TAU = 360.0
EPS = 1e-12

def norm360(x: float) -> float:
    """Normalize angle to [0, 360)."""
    r = float(x) % TAU
    return r + TAU if r < 0.0 else r

def wrap180(x: float) -> float:
    """Wrap angle to (-180, 180]."""
    v = ((float(x) + 180.0) % 360.0) - 180.0
    return v if v != -180.0 else 180.0

def angdiff(a: float, b: float) -> float:
    """Shortest signed separation a − b on the circle, in degrees, wrapped to (-180,180]."""
    return wrap180(float(a) - float(b))

def sign_index(lon_deg: float) -> int:
    """0..11 zodiac sign index with 0 = Aries."""
    return int(math.floor(norm360(lon_deg) / 30.0)) % 12

def is_finite(*xs: float) -> bool:
    return all(math.isfinite(float(x)) for x in xs)

# ── date/time helpers used by Western scans ──────────────────────────────────
def _to_date(s: Any) -> _date:
    """
    Best-effort coercion to date. Accepts:
    - date object
    - ISO datetime/date string (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS[.sss][Z|±HH:MM])
    Fallback: UTC today.
    """
    if isinstance(s, _date):
        return s
    if isinstance(s, str):
        try:
            if "T" in s or " " in s:
                return datetime.fromisoformat(s.replace("Z", "+00:00")).date()
            from datetime import datetime as _dt
            return _dt.strptime(s, "%Y-%m-%d").date()
        except Exception:
            pass
    return datetime.utcnow().date()

def _jd_from_date(d: _date, tz: str, ts_resolve: Callable[..., Dict[str, Any]] | None) -> float:
    """Compute JD_TT at local midnight of given date in given timezone using provided resolver."""
    if ts_resolve is None:
        raise RuntimeError("Timescale resolver unavailable; pass jd_tt/jd_ut1 directly.")
    ts = ts_resolve(d, "00:00:00", tz)
    return float(ts["jd_tt"])

# ── Houses & timescales (shared) ────────────────────────────────────────────
try:
    from app.core.validators import resolve_timescales_from_civil_erfa as _ts_resolve
    _TS_OK = True
    _TS_ERR: Optional[Exception] = None
except Exception as e:  # pragma: no cover - optional path
    _TS_OK = False
    _TS_ERR = e
    _ts_resolve = None  # type: ignore

# house engines (policy → fallback)
_HAS_POLICY = False
_HOUSES_OK = True
_HOUSES_ERR: Optional[Exception] = None
try:
    from app.core.house import compute_houses_with_policy as _compute_houses_policy
    _HAS_POLICY = True
except Exception:
    try:
        from app.core.houses import asc_mc_houses as _asc_mc_houses
    except Exception as _houses_err:  # pragma: no cover - optional path
        _HOUSES_OK = False
        _compute_houses_policy = None  # type: ignore
        _asc_mc_houses = None          # type: ignore
        _HOUSES_ERR = _houses_err

def compute_houses(
    *, latitude: float, longitude: float, jd_tt: float, jd_ut1: float, system: str = "placidus"
) -> Dict[str, Any]:
    """
    Return {'asc': float, 'mc': float, 'cusps': [12 floats]} using either:
    - policy engine (if present), or
    - asc/mc/houses fallback
    """
    if not _HOUSES_OK:
        raise RuntimeError(f"Houses unavailable: {_HOUSES_ERR}")
    if _HAS_POLICY:
        pay = _compute_houses_policy(  # type: ignore[misc]
            lat=latitude, lon=longitude, system=system, jd_tt=jd_tt, jd_ut1=jd_ut1, jd_ut=jd_ut1
        )
        return {"asc": float(pay["asc"]), "mc": float(pay["mc"]), "cusps": [float(x) for x in pay["cusps"]]}
    asc, mc, cusps = _asc_mc_houses(  # type: ignore[misc]
        system, latitude, longitude, jd_tt=jd_tt, jd_ut1=jd_ut1, jd_ut=jd_ut1
    )
    return {"asc": float(asc), "mc": float(mc), "cusps": [float(x) for x in cusps]}

def timescales_from_civil(date_yyyy_mm_dd: str, time_hh_mm_ss: str, place_tz: str) -> Dict[str, Any]:
    """Resolve civil date/time + TZ to timescales dict with jd_tt/jd_ut1."""
    if _ts_resolve is None:
        raise RuntimeError("Timescale resolver unavailable; pass jd_tt/jd_ut1 directly.")
    from datetime import datetime as _dt
    d = _dt.strptime(date_yyyy_mm_dd, "%Y-%m-%d").date()
    return _ts_resolve(d, time_hh_mm_ss, place_tz)

# ── Validation utilities (evaluate/holdout) ─────────────────────────────────
# FeatureFn takes (record, ephemeris) and returns {feature_name: float}
FeatureFn = Callable[[Dict[str, Any], EphemerisAdapter], Dict[str, float]]  # type: ignore[valid-type]

def _pearson_welford(x: List[float], y: List[float]) -> float:
    """
    Numerically stable Pearson via single-pass Welford/Chan algorithm.
    Returns NaN if variance is zero or n<2.
    """
    n = 0
    mean_x = mean_y = 0.0
    Sxx = Syy = Sxy = 0.0
    for xi, yi in zip(x, y):
        xi = float(xi); yi = float(yi)
        n += 1
        dx = xi - mean_x
        dy = yi - mean_y
        mean_x += dx / n
        mean_y += dy / n
        Sxx += dx * (xi - mean_x)
        Syy += dy * (yi - mean_y)
        Sxy += dx * (yi - mean_y)
    if n < 2 or Sxx <= 0.0 or Syy <= 0.0:
        return float("nan")
    return max(-1.0, min(1.0, Sxy / math.sqrt(Sxx * Syy)))

def pearson_corr(x: List[float], y: List[float]) -> float:
    """
    Public Pearson wrapper. For historical compatibility with prior code that expected numeric,
    we return 0.0 when correlation is undefined (n<2 or zero variance).
    """
    r = _pearson_welford(x, y)
    return 0.0 if not math.isfinite(r) else r

def _groups_from_ids(ids: List[Any]) -> List[List[int]]:
    buckets: Dict[Any, List[int]] = {}
    for i, gid in enumerate(ids):
        buckets.setdefault(gid, []).append(i)
    return list(buckets.values())

def permutation_pvalue_corr(
    x: List[float], y: List[int],
    *, n_perm: int = 2000, strata: Optional[List[Any]] = None,
    perm_mode: Literal["iid","within","circular"] = "iid",
    times: Optional[List[float]] = None, seed: Optional[int] = None
) -> Tuple[float, float]:
    """
    Permutation p-value for Pearson r between x and binary/int y.
    - perm_mode = "iid" (global shuffle), "within" (shuffle within strata), "circular" (cyclic shift within strata by time).
    Early stop when running p-value is clearly large (saves time).
    """
    rng = random.Random(seed)
    # normalize y to floats
    y = [float(int(v)) for v in y]
    r_obs = pearson_corr(x, y)
    if not math.isfinite(r_obs):
        return 0.0, 1.0
    if n_perm <= 0:
        return r_obs, 1.0

    indices = list(range(len(y)))
    groups = [indices] if strata is None else _groups_from_ids(strata)

    # for circular: pre-compute within-group order by time
    order_in_group: Dict[int, List[int]] = {}
    if perm_mode == "circular" and times is not None:
        pos = {i: t for i, t in enumerate(times)}
        for gi, g in enumerate(groups):
            order_in_group[gi] = sorted(g, key=lambda i: pos.get(i, 0.0))
    elif perm_mode == "circular":
        # fallback to within if no times
        perm_mode = "within"

    extreme = 0
    abs_obs = abs(r_obs)
    y_work = y[:]  # working copy
    k_last = 0

    # early-stop tuning
    upper_stop_threshold = 0.20  # if p-hat exceeds this after a warmup, abort
    warmup = min(500, max(100, n_perm // 10))

    for k in range(1, n_perm + 1):
        if perm_mode == "iid":
            if strata is None:
                rng.shuffle(y_work)
            else:
                # shuffle within each stratum
                for g in groups:
                    vals = [y_work[i] for i in g]
                    rng.shuffle(vals)
                    for i, v in zip(g, vals):
                        y_work[i] = v
        elif perm_mode == "within":
            for g in groups:
                vals = [y_work[i] for i in g]
                rng.shuffle(vals)
                for i, v in zip(g, vals):
                    y_work[i] = v
        else:  # circular
            for gi, g in enumerate(groups):
                ord_idx = order_in_group.get(gi, g)
                if not ord_idx:
                    continue
                s = rng.randrange(len(ord_idx))
                shifted = ord_idx[s:] + ord_idx[:s]
                vals = [y_work[i] for i in ord_idx]
                for i, v in zip(shifted, vals):
                    y_work[i] = v

        r_perm = pearson_corr(x, y_work)
        if abs(r_perm) >= abs_obs - 1e-15:
            extreme += 1

        # early stop if clearly big p
        if k >= warmup and (extreme + 1.0) / (k + 1.0) > upper_stop_threshold:
            k_last = k
            break
        k_last = k

    p = (extreme + 1.0) / (k_last + 1.0)
    return r_obs, p

def bh_fdr(pvals: List[float], alpha: float = 0.05) -> Tuple[List[float], List[bool]]:
    """
    Benjamini–Hochberg FDR control.
    Returns (q_values, rejected_flags).
    """
    m = len(pvals)
    if m == 0:
        return [], []
    order = sorted(range(m), key=lambda i: pvals[i])
    q = [0.0] * m
    min_q = 1.0
    # reverse pass
    for rank, _ in enumerate(reversed(order), start=1):
        j = order[-rank]
        pi = pvals[j]
        qj = (pi * m) / (m - rank + 1)
        if qj < min_q:
            min_q = qj
        q[j] = min_q
    rejected = [qv <= alpha for qv in q]
    return q, rejected

@dataclass
class EvalResult:
    feature: str
    n: int
    effect_r: float
    p_perm: float
    q_fdr: float
    accepted: bool

def evaluate_univariate(
    records: List[Dict[str, Any]],
    feature_fn: FeatureFn,
    *, ephem: Optional[EphemerisAdapter] = None,  # type: ignore[valid-type]
    n_perm: int = 2000, alpha: float = 0.05,
    stratify_by: Optional[str] = None, group_by: Optional[str] = None,
    perm_mode: Literal["iid","within","circular"] = "iid",
    use_time: bool = True, seed: Optional[int] = None
) -> List[EvalResult]:
    """
    Evaluate multiple scalar features via permutation p-value of Pearson r versus binary outcome.
    Returns sorted list of EvalResult with BH-FDR correction applied.
    """
    # Ephemeris is passed to feature_fn; keep lazy to avoid import side effects.
    ep = ephem or EphemerisAdapter(EphemConfig(frame="ecliptic-of-date", timescale=TS, planets=PLANETS))  # type: ignore

    y: List[int] = []
    strata: List[Any] = []
    rows: List[Dict[str, float]] = []
    times: List[float] = []

    for rec in records:
        y.append(int(rec.get("outcome", 0)))
        gid = rec.get(group_by) if group_by else (rec.get(stratify_by) if stratify_by else None)
        strata.append(gid)
        times.append(float(rec.get("jd_tt", 0.0)))
        rows.append(feature_fn(rec, ep))  # type: ignore

    # collect all feature names
    names: List[str] = sorted({k for r in rows for k in r.keys()})
    results: List[EvalResult] = []
    pvals: List[float] = []
    effects: List[float] = []
    ns: List[int] = []

    for name in names:
        x: List[float] = []
        yy: List[int] = []
        ss: List[Any] = []
        tt: List[float] = []

        for i, r in enumerate(rows):
            v = r.get(name, None)
            if v is None or not math.isfinite(v):  # ← fixed: no stray ')'
                continue
            x.append(float(v))
            yy.append(y[i])
            ss.append(strata[i])
            tt.append(times[i])

        if len(x) < 8 or len(set(yy)) < 2:
            effects.append(0.0); pvals.append(1.0); ns.append(len(x)); continue

        r_obs, p = permutation_pvalue_corr(
            x, yy, n_perm=n_perm,
            strata=ss if (perm_mode != "iid") else (ss if stratify_by else None),
            perm_mode=perm_mode,
            times=tt if (use_time and perm_mode == "circular") else None,
            seed=seed
        )
        effects.append(r_obs); pvals.append(p); ns.append(len(x))

    qvals, flags = bh_fdr(pvals, alpha=alpha) if names else ([], [])
    for name, n, r, p, q, ok in zip(names, ns, effects, pvals, qvals, flags):
        results.append(EvalResult(feature=name, n=n, effect_r=r, p_perm=p, q_fdr=q, accepted=ok))
    results.sort(key=lambda e: (e.q_fdr, e.p_perm, -abs(e.effect_r), e.feature))
    return results

def holdout_replicate(
    records: List[Dict[str, Any]], feature_fn: FeatureFn,
    *, train_frac: float = 0.7, alpha: float = 0.05,
    n_perm_train: int = 2000, n_perm_test: int = 4000,
    perm_mode: Literal["iid","within","circular"] = "iid",
    group_by: Optional[str] = None, use_time: bool = True, seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Simple train/test replication:
    - Shuffle records, split by train_frac
    - Select significant features on train (BH-FDR)
    - Re-test selected features on test with larger permutations
    """
    rnd = random.Random(seed)
    idx = list(range(len(records)))
    rnd.shuffle(idx)
    cut = max(1, int(len(idx) * train_frac))
    tr_idx = set(idx[:cut])
    train = [records[i] for i in range(len(records)) if i in tr_idx]
    test  = [records[i] for i in range(len(records)) if i not in tr_idx]

    train_res = evaluate_univariate(
        train, feature_fn, n_perm=n_perm_train, alpha=alpha,
        perm_mode=perm_mode, group_by=group_by, use_time=use_time, seed=seed
    )
    selected = [r.feature for r in train_res if r.accepted]

    # test pass (recompute with more perms)
    y_test: List[int] = []
    strata: List[Any] = []
    rows: List[Dict[str, float]] = []
    times: List[float] = []
    ep = EphemerisAdapter(EphemConfig(frame="ecliptic-of-date", timescale=TS, planets=PLANETS))  # type: ignore
    for rec in test:
        y_test.append(int(rec.get("outcome", 0)))
        strata.append(rec.get(group_by) if group_by else None)
        times.append(float(rec.get("jd_tt", 0.0)))
        rows.append(feature_fn(rec, ep))  # type: ignore

    detailed: List[Dict[str, Any]] = []
    replicated = 0
    for name in selected:
        x: List[float] = []
        yy: List[int] = []
        ss: List[Any] = []
        tt: List[float] = []

        for i, r in enumerate(rows):
            v = r.get(name, None)
            if v is None or not math.isfinite(v):
                continue
            x.append(float(v)); yy.append(y_test[i]); ss.append(strata[i]); tt.append(times[i])

        if len(x) < 8 or len(set(yy)) < 2:
            detailed.append({"feature": name, "n": len(x), "p_perm": 1.0, "effect_r": 0.0, "replicated": False})
            continue

        r_obs, p = permutation_pvalue_corr(
            x, yy, n_perm=n_perm_test,
            strata=ss if (perm_mode != "iid") else None,
            perm_mode=perm_mode,
            times=tt if (use_time and perm_mode == "circular") else None,
            seed=seed
        )
        ok = p <= alpha
        if ok:
            replicated += 1
        detailed.append({"feature": name, "n": len(x), "p_perm": p, "effect_r": r_obs, "replicated": ok})

    rate = (replicated / max(1, len(selected))) if selected else 0.0
    return {
        "train_results": [r.__dict__ for r in train_res],
        "selected_features": selected,
        "test_details": detailed,
        "replication_rate": rate,
        "n_train": len(train),
        "n_test": len(test),
    }

def validate_predictions(*_args, **_kwargs) -> Dict[str, Any]:
    """
    Stub maintained for backward compatibility with earlier routes.
    """
    return {"ok": True, "metrics": {"p_value": 1.0}, "warnings": []}

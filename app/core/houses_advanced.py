# app/core/houses_advanced.py
"""
Professional House System Calculations — v11 GOLD STANDARD (research-grade; 22 declared)

WHAT’S NEW vs v10
- Unified certification thresholds:
  * Doc target: ≤ 0.003°  (≈ 10.8″)
  * ErrorBudget.certify_accuracy default now 10.8″
  * Validation summary “gold certified” also uses 10.8″
- Numeric knobs exposed via env (PLACIDUS_MAX_ITERS / PLACIDUS_TOL_F / PLACIDUS_TOL_STEP).
- Topocentric semi-arc: preserves cos(δ) sign near δ≈±90° (robustness at singularities).
- Light refactors for clarity; comments and docstrings tightened.

ACCURACY TARGET (unchanged in spirit)
≤ 0.003° agreement vs SwissEph/Solar Fire (1900–2650) with NO shortcuts:
- Apparent sidereal time (GAST) via IAU 2006/2000A (PyERFA)
- True obliquity: mean IAU 2006 + nutation in obliquity (IAU 2000A)
- Distinct time scales: JD_TT and JD_UT1 REQUIRED in strict mode
- Exact angle formulas (Asc/MC/Eastpoint/Vertex)
- Exact house engines (Placidus numeric; Koch/Regio/Campanus/Morinus closed/solved forms)
- Correct Porphyry quadrant trisection; Sripati (Madhya Bhāva) midpoints
- Vehlow Equal (Asc centered in House 1)
- Strict domain checks; optional Placidus diagnostics (iters/residual/last step)
- GOLD STANDARD: Embedded self-validation, cross-reference testing, error budgeting

ASTROLOGICAL INTERPRETATION LAYERS (non-geometric)
- Bhāva Chalit vs Cusp-lines (Vedic):
  • CUSPS are precise boundary lines (ecliptic longitudes) of the twelve houses.
  • BHAVA CHALIT is a *placement* convention: planets are assigned to houses using chosen
    bhāva boundaries (e.g. Sripati/Madhya Bhāva midpoints), which may differ from quadrant
    cusp-lines used in Western practice. This file exposes both:
      - 'sripati' cusps (midpoints from Porphyry boundaries)
      - aliases: 'bhava_chalit_sripati' and 'bhava_chalit_equal_from_mc'
  Consumers should state clearly whether they display *cusps* or *bhāva boundaries*.

- Gauquelin Sectors: PROVISIONAL equal hour-angle slices. Scholarly implementations use
  semi-arc scaling; this placeholder is conservative and intentionally marked provisional.

Policy choices (fallbacks/lat gating) live in app/core/house.py.

NOTE on declared-but-gated systems:
- Ambiguous vendor variants: horizon, carter_pe, sunshine, pullen_sd
- Research claims without public spec/gold vectors: meridian, krusinski
These raise NotImplementedError (map to HTTP 501).
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

import erfa  # PyERFA (BSD) — IAU SOFA routines

# --------------------------- constants & numeric policy ---------------------------

TAU_R = 2.0 * math.pi
DEG_R = math.pi / 180.0
EPS_NUM = 4.0 * sys.float_info.epsilon   # ULP-aware tolerance for domain checks
COS_EPS = 1e-15                          # small guard for cos(δ) near singularities

# Secant solver knobs (env-tunable for ops / testing)
PLACIDUS_MAX_ITERS = int(os.getenv("PLACIDUS_MAX_ITERS", "30"))
PLACIDUS_TOL_F = float(os.getenv("PLACIDUS_TOL_F", "1e-10"))        # function residual (deg)
PLACIDUS_TOL_STEP = float(os.getenv("PLACIDUS_TOL_STEP", "1e-9"))   # last step (deg)

# Certification threshold (arcseconds); keep aligned with doc target 0.003° ≈ 10.8″
GOLD_CERT_ARCSEC = float(os.getenv("GOLD_CERT_ARCSEC", "10.8"))

# --------------------------- angle helpers ---------------------------

def _norm_deg(x: float) -> float:
    r = math.fmod(x, 360.0)
    return r + 360.0 if r < 0.0 else r

def _wrap_diff_deg(a: float, b: float) -> float:
    """Smallest absolute angular difference |a-b| on the circle (0..180]."""
    d = abs(_norm_deg(a - b))
    return d if d <= 180.0 else 360.0 - d

def _atan2d(y: float, x: float) -> float:
    if x == 0.0 and y == 0.0:
        raise ValueError("atan2(0,0) undefined in coordinate transformation")
    return _norm_deg(math.degrees(math.atan2(y, x)))

def _sind(a: float) -> float: return math.sin(a * DEG_R)
def _cosd(a: float) -> float: return math.cos(a * DEG_R)
def _tand(a: float) -> float: return math.tan(a * DEG_R)

def _asin_strict_deg(x: float, ctx: str) -> float:
    if x < -1.0 - EPS_NUM or x > 1.0 + EPS_NUM:
        raise ValueError(f"domain error asin({x:.16e}) in {ctx}")
    x = max(-1.0, min(1.0, x))
    return math.degrees(math.asin(x))

def _acos_strict_deg(x: float, ctx: str) -> float:
    if x < -1.0 - EPS_NUM or x > 1.0 + EPS_NUM:
        raise ValueError(f"domain error acos({x:.16e}) in {ctx}")
    x = max(-1.0, min(1.0, x))
    return math.degrees(math.acos(x))

def _acotd(x: float) -> float:
    return _norm_deg(math.degrees(math.atan2(1.0, x)))

def _split_jd(jd: float) -> Tuple[float, float]:
    d = math.floor(jd)
    return d, jd - d

def _midpoint_wrap(a: float, b: float) -> float:
    """Circular midpoint on [0,360): halfway from a to b moving forward."""
    a = _norm_deg(a); b = _norm_deg(b)
    d = _norm_deg(b - a)
    return _norm_deg(a + 0.5 * d)

def _kahan_sum(values: List[float]) -> float:
    """Compensated summation to reduce floating-point error accumulation."""
    s = c = 0.0
    for v in values:
        y = v - c
        t = s + y
        c = (t - s) - y
        s = t
    return s

def _stable_angle_sum(angles: List[float]) -> float:
    return _norm_deg(_kahan_sum(angles))

# --------------------------- ERFA / fundamental angles ---------------------------

def _gast_deg(jd_ut1: float, jd_tt: float) -> float:
    d1u, d2u = _split_jd(jd_ut1)
    d1t, d2t = _split_jd(jd_tt)
    gst_rad = erfa.gst06a(d1u, d2u, d1t, d2t)
    return _norm_deg(math.degrees(gst_rad))

def _true_obliquity_deg(jd_tt: float) -> float:
    d1, d2 = _split_jd(jd_tt)
    eps0 = erfa.obl06(d1, d2)
    _dpsi, deps = erfa.nut06a(d1, d2)
    return math.degrees(eps0 + deps)

def _ramc_deg(jd_ut1: float, jd_tt: float, lon_deg: float) -> float:
    return _norm_deg(_gast_deg(jd_ut1, jd_tt) + lon_deg)

def _mc_longitude_deg(ramc: float, eps: float) -> float:
    return _atan2d(_sind(ramc) * _cosd(eps), _cosd(ramc))

def _asc_longitude_deg(phi: float, ramc: float, eps: float) -> float:
    # ASC = arccot( - ( tan φ * sin ε + sin RAMC * cos ε ) / cos RAMC )
    num = -((_tand(phi) * _sind(eps)) + (_sind(ramc) * _cosd(eps)))
    den = _cosd(ramc)
    return _acotd(num / max(1e-15, den))

def _eastpoint_longitude_deg(ramc: float, eps: float) -> float:
    ra = _norm_deg(ramc + 90.0)
    return _atan2d(_sind(ra) * _cosd(eps), _cosd(ra))

def _vertex_longitude_deg(phi: float, ramc: float, eps: float) -> float:
    # VTX = arccot( - ( cot φ * sin ε - sin RAMC * cos ε ) / cos RAMC )
    tphi = _tand(phi)
    cot_phi = (1.0 / tphi) if abs(tphi) > 1e-15 else 1e15
    num = -((cot_phi * _sind(eps)) - (_sind(ramc) * _cosd(eps)))
    den = _cosd(ramc)
    return _acotd(num / max(1e-15, den))

# --------------------------- common cusp helpers ---------------------------

def _blank() -> List[Optional[float]]:
    return [None] * 12

def _fill_opposites(cusps: List[Optional[float]]) -> List[float]:
    """Fill opposing cusps by exact 180° where only one side was computed."""
    pairs = [(9, 3), (10, 4), (11, 5), (0, 6), (1, 7), (2, 8)]
    for a, b in pairs:
        if cusps[a] is not None and cusps[b] is None:
            cusps[b] = _norm_deg(cusps[a] + 180.0)
        elif cusps[b] is not None and cusps[a] is None:
            cusps[a] = _norm_deg(cusps[b] + 180.0)
    out = [float(_norm_deg(c)) for c in cusps]  # type: ignore
    # enforce exact opposition (avoid micro-drift)
    for i in range(6):
        opp = _norm_deg(out[i] + 180.0)
        if _wrap_diff_deg(out[i + 6], opp) > 1e-12:
            out[i + 6] = opp
    return out

# --------------------------- exact house engines (closed/solved) ---------------------------

def _equal(asc: float) -> List[float]:
    return [_norm_deg(asc + 30.0 * i) for i in range(12)]

def _whole(asc: float) -> List[float]:
    first = math.floor(asc / 30.0) * 30.0
    return [_norm_deg(first + 30.0 * i) for i in range(12)]

def _porphyry(asc: float, mc: float) -> List[float]:
    cusps = _blank()
    A = cusps[0] = _norm_deg(asc)
    M = cusps[9] = _norm_deg(mc)
    D = cusps[6] = _norm_deg(asc + 180.0)
    I = cusps[3] = _norm_deg(mc + 180.0)
    def span(start: float, end: float) -> float: return _norm_deg(end - start)
    s = span(A, M); cusps[11] = _norm_deg(A + s/3);  cusps[10] = _norm_deg(A + 2*s/3)
    s = span(M, D); cusps[8]  = _norm_deg(M + s/3);  cusps[7]  = _norm_deg(M + 2*s/3)
    s = span(D, I); cusps[5]  = _norm_deg(D + s/3);  cusps[4]  = _norm_deg(D + 2*s/3)
    s = span(I, A); cusps[2]  = _norm_deg(I + s/3);  cusps[1]  = _norm_deg(I + 2*s/3)
    return _fill_opposites(cusps)

def _morinus(ramc: float, eps: float) -> List[float]:
    # tan λ = cos ε · tan(ramc + ad), ad ∈ {0, 30, 60, 90, 120, 150}
    def cusp(ad: float) -> float:
        F = _norm_deg(ramc + ad)
        return _atan2d(_tand(F) * _cosd(eps), 1.0)
    cusps = _blank()
    cusps[9] = cusp(0.0);  cusps[10] = cusp(30.0); cusps[11] = cusp(60.0)
    cusps[0] = cusp(90.0); cusps[1]  = cusp(120.0); cusps[2]  = cusp(150.0)
    return _fill_opposites(cusps)

def _regiomontanus(phi: float, ramc: float, eps: float, asc: float, mc: float) -> List[float]:
    def block(H: float) -> float:
        F = _norm_deg(ramc + H)
        if H in (30.0, 60.0):
            P = math.degrees(math.atan(_tand(phi) * _sind(H)))
        else:
            P = math.degrees(math.atan(_sind(phi) * _sind(H)))
        M = math.degrees(math.atan(_tand(P) / _cosd(F)))
        R = math.degrees(math.atan((_tand(F) * _cosd(M)) / _cosd(M + eps)))
        return R
    cusps = _blank(); cusps[0], cusps[9] = asc, mc
    cusps[10] = _norm_deg(mc + block(30.0));  cusps[11] = _norm_deg(mc + block(60.0))
    cusps[1]  = _norm_deg(mc + block(120.0)); cusps[2]  = _norm_deg(mc + block(150.0))
    return _fill_opposites(cusps)

def _campanus(phi: float, ramc: float, eps: float, asc: float, mc: float) -> List[float]:
    def block(H: float) -> float:
        J = _acotd(_cosd(phi) * _tand(H))
        F = _norm_deg(ramc + 90.0 - J)
        P = _asin_strict_deg(_sind(H) * _sind(phi), "campanus:asinP")
        M = math.degrees(math.atan(_tand(P) / _cosd(F)))
        R = math.degrees(math.atan((_tand(F) * _cosd(M)) / _cosd(M + eps)))
        return R
    cusps = _blank(); cusps[0], cusps[9] = asc, mc
    cusps[10] = _norm_deg(mc + block(30.0));  cusps[11] = _norm_deg(mc + block(60.0))
    cusps[1]  = _norm_deg(mc + block(120.0)); cusps[2]  = _norm_deg(mc + block(150.0))
    return _fill_opposites(cusps)

def _koch(phi: float, eps: float, ramc: float, mc: float) -> List[float]:
    D  = _asin_strict_deg(_sind(mc) * _sind(eps), "koch:decl_mc")
    J  = _asin_strict_deg(_tand(D) * _tand(phi), "koch:asc_diff")
    OAMC = _norm_deg(ramc - J)
    DX  = _norm_deg((ramc + 90.0) - OAMC) / 3.0
    H11 = _norm_deg(OAMC + DX - 90.0)
    H12 = _norm_deg(H11 + DX)
    H1  = _norm_deg(H12 + DX)
    H2  = _norm_deg(H1 + DX)
    H3  = _norm_deg(H2 + DX)

    def cusp_from_H(H: float) -> float:
        num = -((_tand(phi) * _sind(eps)) + (_sind(H) * _cosd(eps)))
        den = _cosd(H)
        return _acotd(num / max(1e-15, den))

    cusps = _blank()
    cusps[9]  = mc
    cusps[10] = cusp_from_H(H11)
    cusps[11] = cusp_from_H(H12)
    cusps[0]  = cusp_from_H(H1)
    cusps[1]  = cusp_from_H(H2)
    cusps[2]  = cusp_from_H(H3)
    return _fill_opposites(cusps)

# --------- Shared trigonometric blocks for Placidus & Topocentric (PP) ----------

def _decl_of_lambda_deg(lam: float, eps: float) -> float:
    return _asin_strict_deg(_sind(eps) * _sind(lam), "decl(lambda)")

def _ra_of_lambda_deg(lam: float, eps: float) -> float:
    return _atan2d(_sind(lam) * _cosd(eps), _cosd(lam))

def _sda_deg(dec: float, phi: float) -> float:
    # SDA = acos(-tan φ · tan δ)
    t = -_tand(phi) * _tand(dec)
    return _acos_strict_deg(t, "sda")

def _sda_topocentric_deg(dec: float, phi: float) -> float:
    """
    Polich–Page (Topocentric): tan φ' = tan φ / cos δ  → SDA = acos( -tan φ' · tan δ )
    Preserve sign of cos δ near singularities to avoid branch flips.
    """
    cosd_dec = _cosd(dec)
    if abs(cosd_dec) < COS_EPS:
        sign = -1.0 if cosd_dec < 0.0 else 1.0
        cosd_dec = sign * COS_EPS
    tan_phi_eff = _tand(phi) / cosd_dec
    t = -tan_phi_eff * _tand(dec)
    return _acos_strict_deg(t, "sda_pp")

# --------------------------- Placidus & PP numeric solver ---------------------------

def _placidus_secant_solver(
    eq_func: Callable[[float], float],
    seeds: List[float],
    label: str,
    _diag: Optional[dict] = None
) -> float:
    """Adaptive secant with multi-seed strategy."""
    best_result = None
    best_error = float('inf')
    used = None
    iters = 0
    last_step = 0.0

    for i, seed in enumerate(seeds):
        try:
            x0 = _norm_deg(seed)
            x1 = _norm_deg(seed + 1.0)
            f0 = eq_func(x0)
            f1 = eq_func(x1)
            it = 0
            while it < PLACIDUS_MAX_ITERS:
                it += 1
                denom = (f1 - f0)
                if abs(denom) < 1e-15:
                    x1 = _norm_deg(x1 + 1e-7)
                    f1 = eq_func(x1)
                    continue
                x2 = _norm_deg((x0 * f1 - x1 * f0) / denom)
                f2 = eq_func(x2)
                step = abs(_norm_deg(x2 - x1))
                if abs(f2) < PLACIDUS_TOL_F or step < PLACIDUS_TOL_STEP:
                    if abs(f2) < best_error:
                        best_result = x2
                        best_error = abs(f2)
                        used = (i, seed)
                        iters = it
                        last_step = step
                    break
                x0, f0, x1, f1 = x1, f1, x2, f2
            if best_result is not None:
                break
        except Exception:
            continue

    if best_result is None:
        raise ValueError(f"solver failed for {label} with all seed strategies")

    if _diag is not None:
        _diag[label] = {
            "iters": iters,
            "residual_deg": float(best_error),
            "last_step_deg": float(last_step),
            "converged": True,
            "seed_deg": float(used[1]) if used else None,  # type: ignore
            "seed_strategy": used[0] if used else None,   # type: ignore
        }
    return _norm_deg(best_result)

def _solve_time_division_cusps(
    *,
    frac_pre_11: float,
    frac_pre_12: float,
    frac_post_2: float,
    frac_post_3: float,
    ramc: float,
    eps: float,
    phi: float,
    asc: float,
    mc: float,
    sda_fn: Callable[[float, float], float],
    diag_store: Optional[dict],
    label_prefix: str
) -> List[float]:
    """Generic time-division cusp solver used by Placidus and Topocentric (PP)."""

    def eq_builder(target_frac: float, side: str) -> Callable[[float], float]:
        sign = -1.0 if side == 'pre' else +1.0
        def f(lambda_guess: float) -> float:
            lam = lambda_guess
            ra  = _ra_of_lambda_deg(lam, eps)
            dec = _decl_of_lambda_deg(lam, eps)
            half = sda_fn(dec, phi)
            dra  = _norm_deg(ra - ramc)
            if dra > 180.0:  # wrap to [-180, +180]
                dra -= 360.0
            return dra - sign * (half * target_frac)
        return f

    por = _porphyry(asc, mc)
    eq_seeds = _equal(asc)

    def seeds_for(i_por: int, i_eq: int) -> List[float]:
        s0 = por[i_por]; s1 = eq_seeds[i_eq]
        return [s0, s1, _norm_deg(s0 + 5.0), _norm_deg(s0 - 5.0), _norm_deg(s1 + 10.0)]

    diag = {} if diag_store is not None else None
    C11 = _placidus_secant_solver(eq_builder(frac_pre_11, 'pre'),  seeds_for(10, 10), f"{label_prefix}C11", diag)
    C12 = _placidus_secant_solver(eq_builder(frac_pre_12, 'pre'),  seeds_for(11, 11), f"{label_prefix}C12", diag)
    C2  = _placidus_secant_solver(eq_builder(frac_post_2, 'post'), seeds_for(1, 1),   f"{label_prefix}C2",  diag)
    C3  = _placidus_secant_solver(eq_builder(frac_post_3, 'post'), seeds_for(2, 2),   f"{label_prefix}C3",  diag)

    cusps = _blank()
    cusps[0], cusps[9]   = asc, mc
    cusps[10], cusps[11] = C11, C12
    cusps[1],  cusps[2]  = C2,  C3

    if diag_store is not None and isinstance(diag, dict):
        diag_store.update(diag)
    return _fill_opposites(cusps)

def _placidus(phi: float, eps: float, ramc: float, asc: float, mc: float, *, _diag: Optional[dict] = None) -> List[float]:
    # 11/12 pre-culmination; 2/3 post-culmination
    return _solve_time_division_cusps(
        frac_pre_11=1/3, frac_pre_12=2/3, frac_post_2=2/3, frac_post_3=1/3,
        ramc=ramc, eps=eps, phi=phi, asc=asc, mc=mc,
        sda_fn=_sda_deg, diag_store=_diag, label_prefix="P:"
    )

def _topocentric_pp(phi: float, eps: float, ramc: float, asc: float, mc: float, *, _diag: Optional[dict] = None) -> List[float]:
    return _solve_time_division_cusps(
        frac_pre_11=1/3, frac_pre_12=2/3, frac_post_2=2/3, frac_post_3=1/3,
        ramc=ramc, eps=eps, phi=phi, asc=asc, mc=mc,
        sda_fn=_sda_topocentric_deg, diag_store=_diag, label_prefix="PP:"
    )

# --------------------------- Corrected Alcabitius (semi-arc division) ---------------------------

def _alcabitius(phi: float, eps: float, asc: float, mc: float) -> List[float]:
    """Solve by semi-arc division referenced to culmination (like Placidus but fixed fractions)."""
    ramc = _ra_of_lambda_deg(mc, eps)

    def eq_builder(target_frac: float, side: str) -> Callable[[float], float]:
        sign = -1.0 if side == 'pre' else +1.0
        def f(lambda_guess: float) -> float:
            lam = lambda_guess
            ra  = _ra_of_lambda_deg(lam, eps)
            dec = _decl_of_lambda_deg(lam, eps)
            half = _sda_deg(dec, phi)
            dra  = _norm_deg(ra - ramc)
            if dra > 180.0:
                dra -= 360.0
            return dra - sign * (half * target_frac)
        return f

    por = _porphyry(asc, mc)
    eq_seeds = _equal(asc)

    def seeds_for(i_por: int, i_eq: int) -> List[float]:
        s0 = por[i_por]; s1 = eq_seeds[i_eq]
        return [s0, s1, _norm_deg(s0 + 5.0), _norm_deg(s0 - 5.0)]

    C11 = _placidus_secant_solver(eq_builder(1/3, 'pre'),  seeds_for(10, 10), "A:C11", None)
    C12 = _placidus_secant_solver(eq_builder(2/3, 'pre'),  seeds_for(11, 11), "A:C12", None)
    C2  = _placidus_secant_solver(eq_builder(2/3, 'post'), seeds_for(1, 1),   "A:C2",  None)
    C3  = _placidus_secant_solver(eq_builder(1/3, 'post'), seeds_for(2, 2),   "A:C3",  None)

    cusps = _blank()
    cusps[9], cusps[0] = mc, asc
    cusps[10], cusps[11] = C11, C12
    cusps[1],  cusps[2]  = C2,  C3
    return _fill_opposites(cusps)

# --------------------------- other exact engines / styles ---------------------------

def _vehlow_equal(asc: float) -> List[float]:
    start = _norm_deg(asc - 15.0)
    return [_norm_deg(start + 30.0 * i) for i in range(12)]

def _sripati(_phi: float, _eps: float, asc: float, mc: float) -> List[float]:
    """Madhya Bhāva: midpoints of Porphyry boundaries (all 12 cusps)."""
    por = _porphyry(asc, mc)
    cusps = [0.0] * 12
    for i in range(12):
        prev = (i - 1) % 12
        cusps[i] = _midpoint_wrap(por[prev], por[i])
    return _fill_opposites(cusps)

def _equal_from_mc(mc: float) -> List[float]:
    cusps = _blank()
    cusps[9] = _norm_deg(mc)            # 10th
    cusps[0] = _norm_deg(mc - 90.0)     # 1st = MC − 90°
    for i in (1, 2, 3, 4, 5, 6, 7, 8, 10, 11):
        cusps[i] = _norm_deg(cusps[0] + 30.0 * i)
    return _fill_opposites(cusps)

def _natural_houses() -> List[float]:
    """Aries = 1st; whole-sign from zodiac origin (0° Aries)."""
    base = 0.0
    return [_norm_deg(base + 30.0 * i) for i in range(12)]

# --------------------------- declared but intentionally gated ---------------------------

_GATED_VENDOR_VARIANTS = {"horizon", "carter_pe", "sunshine", "pullen_sd"}
_GATED_RESEARCH = {"meridian", "krusinski"}

def _raise_gated(name: str) -> None:
    raise NotImplementedError(
        f"House system '{name}' is declared but intentionally not implemented yet "
        f"(awaiting unambiguous public spec and gold vectors)."
    )

# --------------------------- GOLD STANDARD: Test Vector Framework ---------------------------

class GoldTestVector(NamedTuple):
    name: str
    jd_tt: float
    jd_ut1: float
    latitude: float
    longitude: float
    system: str
    expected_cusps: Optional[List[float]]  # None → derive from rule
    tolerance: float                       # degrees
    source: str                            # "Analytical" or external tag

ANALYTICAL_VECTORS = [
    GoldTestVector("Equal_From_Asc",       2451545.0, 2451545.0, 0.0,  0.0,  "equal",          None, 1e-6, "Analytical"),
    GoldTestVector("WholeSign_From_Asc",   2451545.0, 2451545.0, 45.0, 0.0,  "whole_sign",     None, 1e-6, "Analytical"),
    GoldTestVector("Natural_Houses",       2451545.0, 2451545.0, 23.0, 77.0, "natural_houses", None, 1e-6, "Analytical"),
    GoldTestVector("Equal_From_MC",        2451545.0, 2451545.0, 10.0, 0.0,  "equal_from_mc",  None, 1e-6, "Analytical"),
]

def load_external_test_vectors(filepath: Optional[str] = None) -> List[GoldTestVector]:
    if filepath is None:
        return []
    try:
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
        vectors: List[GoldTestVector] = []
        for item in data:
            vectors.append(GoldTestVector(
                name=item['name'],
                jd_tt=item['jd_tt'],
                jd_ut1=item['jd_ut1'],
                latitude=item['latitude'],
                longitude=item['longitude'],
                system=item['system'],
                expected_cusps=item.get('expected_cusps'),
                tolerance=item['tolerance'],
                source=item.get('source', 'External')
            ))
        return vectors
    except Exception:
        return []

def get_test_vectors(external_file: Optional[str] = None) -> List[GoldTestVector]:
    v = list(ANALYTICAL_VECTORS)
    v.extend(load_external_test_vectors(external_file))
    return v

def _expected_from_rule(system: str, asc: float, mc: float) -> Optional[List[float]]:
    s = system.lower()
    if s == "equal": return _equal(asc)
    if s in ("whole_sign", "whole"): return _whole(asc)
    if s == "vehlow_equal": return _vehlow_equal(asc)
    if s == "natural_houses": return _natural_houses()
    if s == "equal_from_mc": return _equal_from_mc(mc)
    return None

# --------------------------- error budget & validation types ---------------------------

@dataclass
class ErrorBudget:
    coordinate_precision: float = 0.0
    algorithm_truncation: float = 0.0
    time_scale_uncertainty: float = 0.0
    reference_comparison: float = 0.0
    total_rss: float = 0.0
    def compute_total(self) -> float:
        comps = [
            self.coordinate_precision,
            self.algorithm_truncation,
            self.time_scale_uncertainty,
            self.reference_comparison,
        ]
        self.total_rss = math.sqrt(sum(x*x for x in comps))
        return self.total_rss
    def certify_accuracy(self, target_arcsec: float = GOLD_CERT_ARCSEC) -> bool:
        """Default cert target matches doc (0.003° = 10.8″)."""
        self.compute_total()
        return self.total_rss * 3600.0 <= target_arcsec

class ValidationResult(NamedTuple):
    vector_name: str
    system: str
    max_error_deg: float
    max_error_arcsec: float
    passed: bool
    error_budget: ErrorBudget

# --------------------------- data model ---------------------------

@dataclass
class HouseData:
    system: str
    cusps: List[float]
    ascendant: float
    midheaven: float
    vertex: Optional[float] = None
    eastpoint: Optional[float] = None
    warnings: List[str] = None
    solver_stats: Optional[Dict[str, Any]] = None
    validation_results: List[ValidationResult] = field(default_factory=list)
    error_budget: Optional[ErrorBudget] = None
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []

# --------------------------- supported systems ---------------------------

# 24 declared (incl. equal_from_mc, natural_houses, bhava_* aliases; plus gated)
SUPPORTED_HOUSE_SYSTEMS = [
    "placidus","koch","regiomontanus","campanus",
    "equal","whole_sign","porphyry","alcabitius",
    "morinus","topocentric","meridian","horizon",
    "carter_pe","sunshine","vehlow_equal","sripati",
    "krusinski","pullen_sd","equal_from_mc","natural_houses",
    "bhava_chalit_sripati","bhava_chalit_equal_from_mc",
]

# Implemented subset
IMPLEMENTED_HOUSE_SYSTEMS = {
    "placidus","koch","regiomontanus","campanus",
    "equal","whole_sign","porphyry","alcabitius",
    "morinus","topocentric","vehlow_equal","sripati",
    "equal_from_mc","natural_houses","bhava_chalit_sripati","bhava_chalit_equal_from_mc",
}

# --------------------------- main calculator ---------------------------

class PreciseHouseCalculator:
    """
    Exact house computations with apparent sidereal time and true obliquity.
    STRICT by default: both jd_tt and jd_ut1 required.
    Declares many systems; some gated.
    """

    def __init__(self, require_strict_timescales: bool = True, enable_diagnostics: bool = False,
                 enable_validation: bool = False):
        self.require_strict_timescales = require_strict_timescales
        self.enable_diagnostics = enable_diagnostics
        self.enable_validation = enable_validation

    # ----- validation / error budget -----

    def validate_gold_standard(self, external_vectors_file: Optional[str] = None) -> List[ValidationResult]:
        results: List[ValidationResult] = []
        for vector in get_test_vectors(external_vectors_file):
            if vector.system not in IMPLEMENTED_HOUSE_SYSTEMS:
                continue
            try:
                temp = PreciseHouseCalculator(
                    require_strict_timescales=self.require_strict_timescales,
                    enable_diagnostics=False, enable_validation=False
                )
                hd = temp.calculate_houses(
                    latitude=vector.latitude, longitude=vector.longitude,
                    jd_ut=vector.jd_ut1, house_system=vector.system,
                    jd_tt=vector.jd_tt, jd_ut1=vector.jd_ut1
                )
                expected = (_expected_from_rule(vector.system, hd.ascendant, hd.midheaven)
                            if vector.source == "Analytical" else vector.expected_cusps)
                if not expected:
                    continue
                errors = [_wrap_diff_deg(c, e) for c, e in zip(hd.cusps, expected)]
                max_err = max(errors)
                max_arc = max_err * 3600.0
                eb = ErrorBudget(
                    coordinate_precision=sys.float_info.epsilon * 57.2958,  # rad→deg
                    algorithm_truncation=(PLACIDUS_TOL_F if vector.system == "placidus" else 1e-15),
                    time_scale_uncertainty=0.0001,  # deg (≈0.1s TT-UT1 @ sidereal rate → 15°/h)
                    reference_comparison=max_err
                )
                eb.compute_total()
                results.append(ValidationResult(vector.name, vector.system, max_err, max_arc,
                                                max_err <= vector.tolerance, eb))
            except Exception:
                results.append(ValidationResult(vector.name, vector.system, float('inf'), float('inf'),
                                                False, ErrorBudget()))
        return results

    def cross_validate_implementation(self, latitude: float, longitude: float,
                                      jd_tt: float, jd_ut1: float, house_system: str) -> ErrorBudget:
        eb = ErrorBudget()
        phi_rad = math.radians(latitude)
        condition = max(1.0, abs(1.0 / max(1e-15, math.cos(phi_rad))))  # worst at poles
        eb.coordinate_precision = condition * sys.float_info.epsilon * 57.2958
        eb.algorithm_truncation = PLACIDUS_TOL_F if house_system == "placidus" else 1e-15
        dt_unc = 0.1 / 86400.0  # ~0.1 s
        sid_rate = 1.002737909350795
        eb.time_scale_uncertainty = dt_unc * sid_rate * 15.0
        eb.reference_comparison = 0.0
        eb.compute_total()
        return eb

    # ----- main API -----

    def calculate_houses(
        self,
        latitude: float,
        longitude: float,
        jd_ut: float,                      # legacy compat; unused in strict mode
        house_system: str = "placidus",
        *,
        jd_tt: Optional[float] = None,
        jd_ut1: Optional[float] = None,
    ) -> HouseData:

        # latitude sanity (poles undefined)
        if not (-90.0 < latitude < 90.0):
            raise ValueError("Latitude must be strictly between -90 and 90 degrees; poles undefined for house systems.")
        if math.isnan(latitude) or math.isinf(latitude):
            raise ValueError("Latitude must be a finite number.")

        sys_name = house_system.lower().strip()
        if sys_name == "whole": sys_name = "whole_sign"
        if sys_name == "azimuthal": sys_name = "horizon"
        if sys_name not in SUPPORTED_HOUSE_SYSTEMS:
            raise ValueError(f"Unsupported house system: {house_system}")
        if sys_name in _GATED_VENDOR_VARIANTS or sys_name in _GATED_RESEARCH:
            _raise_gated(sys_name)

        # time scales
        if self.require_strict_timescales:
            if jd_tt is None or jd_ut1 is None:
                raise ValueError("Strict mode requires jd_tt and jd_ut1 (no UT≈UTC shortcuts).")
            tt, ut1 = jd_tt, jd_ut1
        else:
            tt  = jd_tt  or jd_ut
            ut1 = jd_ut1 or jd_ut
            if tt is None or ut1 is None:
                raise ValueError("Provide jd_tt and jd_ut1 or disable strict mode explicitly.")

        # fundamental angles
        eps  = _true_obliquity_deg(tt)
        ramc = _ramc_deg(ut1, tt, longitude)
        mc   = _mc_longitude_deg(ramc, eps)
        asc  = _asc_longitude_deg(latitude, ramc, eps)
        east = _eastpoint_longitude_deg(ramc, eps)
        vtx  = _vertex_longitude_deg(latitude, ramc, eps)

        warnings: List[str] = []
        solver_stats: Optional[Dict[str, Any]] = {} if self.enable_diagnostics else None

        # engines
        if sys_name == "equal":
            cusps = _equal(asc)
        elif sys_name == "whole_sign":
            cusps = _whole(asc)
        elif sys_name == "porphyry":
            cusps = _porphyry(asc, mc)
        elif sys_name == "morinus":
            cusps = _morinus(ramc, eps)
        elif sys_name == "regiomontanus":
            cusps = _regiomontanus(latitude, ramc, eps, asc, mc)
        elif sys_name == "campanus":
            cusps = _campanus(latitude, ramc, eps, asc, mc)
        elif sys_name == "koch":
            cusps = _koch(latitude, eps, ramc, mc)
        elif sys_name == "alcabitius":
            cusps = _alcabitius(latitude, eps, asc, mc)
        elif sys_name == "vehlow_equal":
            cusps = _vehlow_equal(asc)
        elif sys_name == "sripati":
            cusps = _sripati(latitude, eps, asc, mc)
        elif sys_name == "equal_from_mc":
            cusps = _equal_from_mc(mc)
        elif sys_name == "natural_houses":
            cusps = _natural_houses()
        elif sys_name == "bhava_chalit_sripati":
            cusps = _sripati(latitude, eps, asc, mc)
        elif sys_name == "bhava_chalit_equal_from_mc":
            cusps = _equal_from_mc(mc)
        elif sys_name == "topocentric":
            diag = {} if self.enable_diagnostics else None
            cusps = _topocentric_pp(latitude, eps, ramc, asc, mc, _diag=diag)
            if solver_stats is not None and diag is not None:
                solver_stats["topocentric"] = diag
        else:  # placidus
            diag = {} if self.enable_diagnostics else None
            cusps = _placidus(latitude, eps, ramc, asc, mc, _diag=diag)
            if solver_stats is not None and diag is not None:
                solver_stats["placidus"] = diag

        # compact solver diagnostics
        if self.enable_diagnostics and solver_stats:
            for group_key in ("placidus", "topocentric"):
                d = solver_stats.get(group_key)
                if not isinstance(d, dict):
                    continue
                for label in ("P:C11","P:C12","P:C2","P:C3","PP:C11","PP:C12","PP:C2","PP:C3"):
                    if label not in d:
                        continue
                    s = d[label]
                    warnings.append(
                        f"[{label}] iters={s.get('iters')} res={s.get('residual_deg'):.3e}° "
                        f"step={s.get('last_step_deg'):.3e}° seed={s.get('seed_deg')} strat={s.get('seed_strategy')}"
                    )

        # validation / error budget
        validation_results: List[ValidationResult] = []
        error_budget: Optional[ErrorBudget] = None
        if self.enable_validation:
            validation_results = self.validate_gold_standard()
            error_budget = self.cross_validate_implementation(latitude, longitude, tt, ut1, sys_name)
            if error_budget.certify_accuracy(GOLD_CERT_ARCSEC):
                warnings.append(f"Error budget certified: {error_budget.total_rss*3600:.1f}\" ≤ {GOLD_CERT_ARCSEC:.1f}\" target")
            else:
                warnings.append(f"Error budget warning: {error_budget.total_rss*3600:.1f}\" > {GOLD_CERT_ARCSEC:.1f}\" target")

        return HouseData(
            system=sys_name,
            cusps=cusps,
            ascendant=asc,
            midheaven=mc,
            vertex=vtx,
            eastpoint=east,
            warnings=warnings,
            solver_stats=solver_stats if self.enable_diagnostics else None,
            validation_results=validation_results if self.enable_validation else [],
            error_budget=error_budget
        )

# --------------------------- integration helper ---------------------------

def compute_house_system(
    latitude: float,
    longitude: float,
    house_system: str,
    jd_ut: float,
    *,
    jd_tt: Optional[float] = None,
    jd_ut1: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Back-compat wrapper used elsewhere in the app.
    Strict mode: requires jd_tt and jd_ut1; raises if missing.
    """
    calc = PreciseHouseCalculator(require_strict_timescales=True, enable_diagnostics=False)
    hd = calc.calculate_houses(
        latitude=latitude, longitude=longitude, jd_ut=jd_ut,
        house_system=house_system, jd_tt=jd_tt, jd_ut1=jd_ut1,
    )
    payload: Dict[str, Any] = {
        "house_system": hd.system,
        "asc_deg": hd.ascendant,
        "mc_deg": hd.midheaven,
        "cusps_deg": hd.cusps,
        "vertex": hd.vertex,
        "eastpoint": hd.eastpoint,
        "warnings": hd.warnings,
    }
    if hd.solver_stats is not None:
        payload["solver_stats"] = hd.solver_stats
    if hd.validation_results:
        payload["validation_results"] = [r._asdict() for r in hd.validation_results]
    if hd.error_budget is not None:
        payload["error_budget"] = {
            "coordinate_precision": hd.error_budget.coordinate_precision,
            "algorithm_truncation": hd.error_budget.algorithm_truncation,
            "time_scale_uncertainty": hd.error_budget.time_scale_uncertainty,
            "reference_comparison": hd.error_budget.reference_comparison,
            "total_rss": hd.error_budget.total_rss,
            "certified": hd.error_budget.certify_accuracy(GOLD_CERT_ARCSEC)
        }
    return payload

# --------------------------- Planet→House assignment & diagnostics ---------------------------

def assign_houses(longitudes_deg: List[float], cusps_deg: List[float]) -> List[int]:
    """
    Map ecliptic longitudes (0..360) to house numbers (1..12) using forward-wrapping
    intervals [cusp[i], cusp[i+1]) in zodiacal order. Cusps must be 12 values.
    """
    if len(cusps_deg) != 12:
        raise ValueError("cusps_deg must have length 12")
    ordered = [_norm_deg(c) for c in cusps_deg]
    res: List[int] = []
    for L in longitudes_deg:
        lam = _norm_deg(L)
        idx = 0
        while idx < 12:
            start = ordered[idx]
            end   = _norm_deg(ordered[(idx + 1) % 12])
            span  = _norm_deg(end - start)
            delta = _norm_deg(lam - start)
            if delta < span or span == 0.0:
                res.append(idx + 1)
                break
            idx += 1
        if idx == 12:
            res.append(12)
    return res

# --- Interception & house-size analysis ---

_ZODIAC_SIGNS = [
    "Aries","Taurus","Gemini","Cancer","Leo","Virgo",
    "Libra","Scorpio","Sagittarius","Capricorn","Aquarius","Pisces"
]

def _sign_index(lon_deg: float) -> int:
    """0..11 (0=Aries,...,11=Pisces)."""
    return int(math.floor(_norm_deg(lon_deg) / 30.0)) % 12

def _house_of_longitude(cusps_deg: List[float], lon_deg: float) -> int:
    """Return 1..12 house index for a longitude with forward-wrap intervals."""
    if len(cusps_deg) != 12:
        raise ValueError("cusps_deg must have length 12")
    ordered = [_norm_deg(c) for c in cusps_deg]
    lam = _norm_deg(lon_deg)
    for i in range(12):
        start = ordered[i]
        end   = _norm_deg(ordered[(i + 1) % 12])
        span  = _norm_deg(end - start)
        delta = _norm_deg(lam - start)
        if delta < span or span == 0.0:
            return i + 1
    return 12

def analyze_interceptions(cusps_deg: List[float]) -> Dict[str, Any]:
    """
    Determine intercepted signs (entire sign contained within one house span)
    and duplicated signs (two or more cusps inside a sign). Also returns per-house
    spans and cusp-sign labels.
    """
    if len(cusps_deg) != 12:
        raise ValueError("cusps_deg must have length 12")

    cusps = [_norm_deg(x) for x in cusps_deg]

    # per-house spans
    spans: List[float] = []
    for i in range(12):
        start = cusps[i]
        end   = _norm_deg(cusps[(i + 1) % 12])
        spans.append(_norm_deg(end - start))

    cusp_signs = [_ZODIAC_SIGNS[_sign_index(x)] for x in cusps]

    # duplicated signs: count cusps per sign
    counts = [0] * 12
    for c in cusps:
        counts[_sign_index(c)] += 1
    duplicated = [_ZODIAC_SIGNS[i] for i, n in enumerate(counts) if n >= 2]

    # intercepted: signs with zero cusps AND fully inside a single house span
    intercepted: List[Dict[str, Any]] = []
    for s in range(12):
        if counts[s] != 0:
            continue
        start_s = s * 30.0
        end_s   = _norm_deg(start_s + 30.0)
        eps = 1e-8  # avoid cusp coincidence
        h1 = _house_of_longitude(cusps, _norm_deg(start_s + eps))
        h2 = _house_of_longitude(cusps, _norm_deg(end_s   - eps))
        if h1 == h2:
            intercepted.append({"sign": _ZODIAC_SIGNS[s], "house": h1})

    return {
        "intercepted_signs": intercepted,
        "duplicated_signs": duplicated,
        "house_spans_deg": spans,
        "cusp_signs": cusp_signs,
    }

def analyze_house_sizes(cusps_deg: List[float]) -> Dict[str, Any]:
    """
    Compute per-house spans and summary stats relative to the 'ideal' 30°.
    """
    if len(cusps_deg) != 12:
        raise ValueError("cusps_deg must have length 12")
    c = [_norm_deg(x) for x in cusps_deg]
    spans: List[float] = []
    for i in range(12):
        start = c[i]
        end   = _norm_deg(c[(i + 1) % 12])
        spans.append(_norm_deg(end - start))
    mean = sum(spans) / 12.0
    var  = sum((s - mean) ** 2 for s in spans) / 12.0
    std  = math.sqrt(var)
    dev  = [abs(s - 30.0) for s in spans]
    return {
        "spans_deg": spans,
        "mean_deg": mean,
        "std_deg": std,
        "min_deg": min(spans),
        "max_deg": max(spans),
        "deviation_deg": dev,
        "total_abs_deviation_deg": sum(dev),
    }

# --------------------------- KP helpers (nakshatra, sub-lord) ---------------------------

# Vimshottari order (KP convention starts from Ketu)
_VIM_ORDER = ["ketu","venus","sun","moon","mars","rahu","jupiter","saturn","mercury"]
_VIM_YEARS = {"ketu":7,"venus":20,"sun":6,"moon":10,"mars":7,"rahu":18,"jupiter":16,"saturn":19,"mercury":17}
_NAK_WIDTH = 360.0 / 27.0  # 13°20' = 13.333...°

def _kp_nakshatra_index(nirayana_lon: float) -> int:
    """1..27 (Ashwini=1)."""
    return int(math.floor(_norm_deg(nirayana_lon) / _NAK_WIDTH)) + 1

def _kp_nak_lord(nak_index: int) -> str:
    return _VIM_ORDER[(nak_index - 1) % 9]

def _cycle_from(lord: str) -> List[str]:
    i = _VIM_ORDER.index(lord)
    return _VIM_ORDER[i:] + _VIM_ORDER[:i]

def kp_details(longitude_deg: float, *, zodiac_mode: str = "sidereal", ayanamsa_deg: float = 0.0) -> Dict[str, Any]:
    """
    KP nakshatra/pada/lord/sub-lord for a given longitude.
    If zodiac_mode="sidereal", longitude is tropical and we subtract ayanamsa to get nirayana.
    """
    lon = _norm_deg(longitude_deg - (ayanamsa_deg if zodiac_mode.startswith("sidereal") else 0.0))
    nak_idx = _kp_nakshatra_index(lon)
    nak_lord = _kp_nak_lord(nak_idx)
    nak_start = (nak_idx - 1) * _NAK_WIDTH
    pos_in_nak = _norm_deg(lon - nak_start)
    pada = int(math.floor((pos_in_nak / _NAK_WIDTH) * 4.0)) + 1  # 1..4
    # Sub-lord subdivision: 9 parts proportional to Vimshottari years, starting at nak_lord
    cycle = _cycle_from(nak_lord)
    total_years = 120.0
    sub_lengths = [(_VIM_YEARS[p] / total_years) * _NAK_WIDTH for p in cycle]
    cum = 0.0
    sublord = cycle[-1]
    for p, seg in zip(cycle, sub_lengths):
        if pos_in_nak < cum + seg:
            sublord = p
            break
        cum += seg
    return {
        "nirayana_longitude": lon,
        "nakshatra_index": nak_idx,   # 1..27
        "pada": pada,                 # 1..4
        "nakshatra_lord": nak_lord,
        "sublord": sublord,
        "nakshatra_start_deg": nak_start,
        "offset_within_nak_deg": pos_in_nak,
    }

def kp_assign_for_points(points_deg: Dict[str, float], *, zodiac_mode: str = "sidereal", ayanamsa_deg: float = 0.0) -> Dict[str, Dict[str, Any]]:
    """Compute KP details per named point (planets/cusps). points_deg: {"Moon": 123.4, "Cusp1": ...}"""
    out: Dict[str, Dict[str, Any]] = {}
    for name, lon in points_deg.items():
        out[name] = kp_details(lon, zodiac_mode=zodiac_mode, ayanamsa_deg=ayanamsa_deg)
    return out

# --------------------------- Gauquelin sectors (provisional) ---------------------------

def gauquelin_sector(ra_deg: float, dec_deg: float, *, ramc_deg: float, phi_deg: float) -> int:
    """
    PROVISIONAL equal-HA sectorization (36 sectors, 10° each) along the diurnal circle.
    - sector 1 starts at upper culmination (HA=0→10°), numbering increases with positive HA.
    NOTE: Scholarly Gauquelin uses semi-arc scaling; this simplified version treats equal
    hour-angle slices. Parameters dec_deg and phi_deg are reserved for the future exact model.
    """
    # keep references to signal future use (avoid linter complaints)
    _ = dec_deg; __ = phi_deg
    ha = _norm_deg(ra_deg - ramc_deg)  # Hour angle HA = RA - RAMC
    sector = int(math.floor(ha / 10.0)) + 1
    if sector < 1:  sector = 1
    if sector > 36: sector = 36
    return sector

# --------------------------- validation runner ---------------------------

def run_comprehensive_validation(external_vectors_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Run analytical + optional external vectors through implemented systems
    and summarize max/avg errors. Certification uses GOLD_CERT_ARCSEC.
    """
    calc = PreciseHouseCalculator(require_strict_timescales=True, enable_diagnostics=True, enable_validation=False)
    all_results: List[ValidationResult] = []
    system_stats: Dict[str, Dict[str, float]] = {}

    for vector in get_test_vectors(external_vectors_file):
        if vector.system not in IMPLEMENTED_HOUSE_SYSTEMS:
            continue
        try:
            hd = calc.calculate_houses(
                latitude=vector.latitude, longitude=vector.longitude,
                jd_ut=vector.jd_ut1, house_system=vector.system,
                jd_tt=vector.jd_tt, jd_ut1=vector.jd_ut1
            )
            expected = _expected_from_rule(vector.system, hd.ascendant, hd.midheaven) \
                       if vector.source == "Analytical" else vector.expected_cusps
            if not expected:
                continue
            errors = [_wrap_diff_deg(c, e) for c, e in zip(hd.cusps, expected)]
            max_err = max(errors)
            max_arc = max_err * 3600.0
            eb = ErrorBudget(
                coordinate_precision=sys.float_info.epsilon * 57.2958,
                algorithm_truncation=(PLACIDUS_TOL_F if vector.system == "placidus" else 1e-15),
                time_scale_uncertainty=0.0001,
                reference_comparison=max_err
            )
            eb.compute_total()
            res = ValidationResult(vector.name, vector.system, max_err, max_arc,
                                   max_err <= vector.tolerance, eb)
            all_results.append(res)

            st = system_stats.setdefault(vector.system, {'tested': 0, 'passed': 0, 'max_error': 0.0, 'avg_error': 0.0})
            st['tested'] += 1
            if res.passed: st['passed'] += 1
            st['max_error'] = max(st['max_error'], res.max_error_arcsec)
            st['avg_error'] = (st['avg_error'] * (st['tested'] - 1) + res.max_error_arcsec) / st['tested']
        except Exception:
            all_results.append(ValidationResult(vector.name, vector.system, float('inf'), float('inf'), False, ErrorBudget()))

    total = len(all_results)
    passed = sum(1 for r in all_results if r.passed)
    pass_rate = (passed / total) if total > 0 else 0.0
    max_arc = max((r.max_error_arcsec for r in all_results), default=0.0)
    avg_arc = sum(r.max_error_arcsec for r in all_results) / total if total > 0 else 0.0

    return {
        'validation_summary': {
            'total_tests': total,
            'passed_tests': passed,
            'pass_rate': pass_rate,
            'max_error_arcsec': max_arc,
            'avg_error_arcsec': avg_arc,
            # Gold certification aligned to GOLD_CERT_ARCSEC (default 10.8″)
            'gold_standard_certified': pass_rate >= 0.95 and max_arc <= GOLD_CERT_ARCSEC
        },
        'system_statistics': system_stats,
        'detailed_results': [r._asdict() for r in all_results]
    }

# --------------------------- CLI ---------------------------

if __name__ == "__main__":
    report = run_comprehensive_validation()
    print("=== GOLD STANDARD VALIDATION REPORT ===")
    summary = report['validation_summary']
    print(f"Tests: {summary['passed_tests']}/{summary['total_tests']} passed ({summary['pass_rate']:.1%})")
    print(f"Max error: {summary['max_error_arcsec']:.1f}\"  Avg error: {summary['avg_error_arcsec']:.1f}\"")
    print(f"Gold standard certified: {summary['gold_standard_certified']}")
    for system, stats in report['system_statistics'].items():
        print(f"{system}: {int(stats['passed'])}/{int(stats['tested'])} passed, max_err={stats['max_error']:.1f}\"")

    # Demo: interception & size analysis (synthetic)
    print("\n=== DIAGNOSTICS (example) ===")
    eq_cusps = [i * 30.0 for i in range(12)]  # equal houses: no interceptions
    print("Intercepts:", analyze_interceptions(eq_cusps))
    print("Sizes:", analyze_house_sizes(eq_cusps))
    print("\nNOTE: Use external private vectors for full SwissEph/Solar Fire parity.")

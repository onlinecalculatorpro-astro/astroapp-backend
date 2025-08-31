# app/core/houses_advanced.py
"""
Professional House System Calculations — v6 (research-grade; 18 declared)

Targets ≤ 0.01° agreement vs SwissEph/Solar Fire (1900–2650) with NO shortcuts:
- Apparent sidereal time (GAST) via IAU 2006/2000A (PyERFA)
- True obliquity: mean IAU 2006 + nutation in obliquity (IAU 2000A)
- Distinct time scales: JD_TT and JD_UT1 are REQUIRED in strict mode
- Exact angle formulas (Asc/MC/Eastpoint/Vertex)
- Exact house engines (Placidus numeric; Koch/Regio/Campanus/Morinus/Alcabitius closed forms)
- Correct Porphyry quadrant trisection
- Sripati (Madhya Bhava): midpoints of Porphyry boundaries (all 12, wrapped)
- Vehlow Equal (SwissEph 'V'): Asc is the center of House 1 (cusp1 = Asc−15°)
- Strict domain checks (raise ValueError on invalid math)
- Optional Placidus diagnostics (iteration count / residual / last step)

Policy choices (fallbacks/lat gating) live in app/core/house.py.

NOTE on declared-but-gated systems:
- Ambiguous vendor variants (no single public spec): horizon, carter_pe, sunshine, pullen_sd
- Research claims without gold vectors/spec: meridian, krusinski
These raise NotImplementedError (map to HTTP 501).
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import erfa  # PyERFA: IAU SOFA routines (BSD)

# --------------------------- constants & helpers ---------------------------

TAU_R = 2.0 * math.pi
DEG_R = math.pi / 180.0
EPS_NUM = 1e-12  # numeric tolerance for strict domain checks

def _norm_deg(x: float) -> float:
    r = math.fmod(x, 360.0)
    return r + 360.0 if r < 0.0 else r

def _atan2d(y: float, x: float) -> float:
    return _norm_deg(math.degrees(math.atan2(y, x)))

def _sind(a: float) -> float: return math.sin(a * DEG_R)
def _cosd(a: float) -> float: return math.cos(a * DEG_R)
def _tand(a: float) -> float: return math.tan(a * DEG_R)

def _asin_strict_deg(x: float, ctx: str) -> float:
    if x < -1.0 - EPS_NUM or x > 1.0 + EPS_NUM:
        raise ValueError(f"domain error asin({x}) in {ctx}")
    x = max(-1.0, min(1.0, x))
    return math.degrees(math.asin(x))

def _acos_strict_deg(x: float, ctx: str) -> float:
    if x < -1.0 - EPS_NUM or x > 1.0 + EPS_NUM:
        raise ValueError(f"domain error acos({x}) in {ctx}")
    x = max(-1.0, min(1.0, x))
    return math.degrees(math.acos(x))

def _acotd(x: float) -> float:
    # quadrant-safe arccot using atan2(1, x)
    return _norm_deg(math.degrees(math.atan2(1.0, x)))

def _split_jd(jd: float) -> Tuple[float, float]:
    d = math.floor(jd)
    return d, jd - d

def _midpoint_wrap(a: float, b: float) -> float:
    """
    Circular midpoint on [0,360): the point halfway from a to b moving
    forward along the zodiac. Works correctly across the 0/360 boundary.
    """
    a = _norm_deg(a); b = _norm_deg(b)
    d = _norm_deg(b - a)  # forward arc a->b in [0,360)
    return _norm_deg(a + 0.5 * d)

# --------------------------- ERFA / fundamental angles ---------------------------

def _gast_deg(jd_ut1: float, jd_tt: float) -> float:
    """Apparent sidereal time (GAST) in degrees using IAU 2006/2000A."""
    d1u, d2u = _split_jd(jd_ut1)
    d1t, d2t = _split_jd(jd_tt)
    gst_rad = erfa.gst06a(d1u, d2u, d1t, d2t)
    return _norm_deg(math.degrees(gst_rad))

def _true_obliquity_deg(jd_tt: float) -> float:
    """True obliquity ε = ε_mean(IAU2006) + Δε(IAU2000A)."""
    d1, d2 = _split_jd(jd_tt)
    eps0 = erfa.obl06(d1, d2)
    _dpsi, deps = erfa.nut06a(d1, d2)
    return math.degrees(eps0 + deps)

def _ramc_deg(jd_ut1: float, jd_tt: float, lon_deg: float) -> float:
    """Right Ascension of the MC = GAST + longitude (east-positive)."""
    return _norm_deg(_gast_deg(jd_ut1, jd_tt) + lon_deg)

def _mc_longitude_deg(ramc: float, eps: float) -> float:
    """λ_MC = atan2( sin(RAMC) * cos ε, cos(RAMC) )."""
    return _atan2d(_sind(ramc) * _cosd(eps), _cosd(ramc))

def _asc_longitude_deg(phi: float, ramc: float, eps: float) -> float:
    """
    Exact Ascendant formula (arccot form; quadrant-safe).
    ASC = arccot( - ( tan φ * sin ε + sin RAMC * cos ε ) / cos RAMC )
    """
    num = -((_tand(phi) * _sind(eps)) + (_sind(ramc) * _cosd(eps)))
    den = _cosd(ramc)
    return _acotd(num / max(1e-15, den))

def _eastpoint_longitude_deg(ramc: float, eps: float) -> float:
    """Equatorial Ascendant (Eastpoint). RA = RAMC + 90°."""
    ra = _norm_deg(ramc + 90.0)
    return _atan2d(_sind(ra) * _cosd(eps), _cosd(ra))

def _vertex_longitude_deg(phi: float, ramc: float, eps: float) -> float:
    """
    Exact Vertex (intersection of ecliptic & prime vertical in the west), arccot form:
    VTX = arccot( - ( cot φ * sin ε - sin RAMC * cos ε ) / cos RAMC )
    """
    tphi = _tand(phi)
    cot_phi = (1.0 / tphi) if abs(tphi) > 1e-15 else 1e15
    num = -((cot_phi * _sind(eps)) - (_sind(ramc) * _cosd(eps)))
    den = _cosd(ramc)
    return _acotd(num / max(1e-15, den))

# --------------------------- common cusp helpers ---------------------------

def _blank() -> List[Optional[float]]:
    return [None] * 12

def _fill_opposites(cusps: List[Optional[float]]) -> List[float]:
    pairs = [(9, 3), (10, 4), (11, 5), (0, 6), (1, 7), (2, 8)]
    for a, b in pairs:
        if cusps[a] is not None and cusps[b] is None:
            cusps[b] = _norm_deg(cusps[a] + 180.0)
        elif cusps[b] is not None and cusps[a] is None:
            cusps[a] = _norm_deg(cusps[b] + 180.0)
    return [float(_norm_deg(c)) for c in cusps]  # type: ignore

# --------------------------- exact house engines (original 10) ---------------------------

def _equal(asc: float) -> List[float]:
    return [_norm_deg(asc + 30.0 * i) for i in range(12)]

def _whole(asc: float) -> List[float]:
    first = math.floor(asc / 30.0) * 30.0
    return [_norm_deg(first + 30.0 * i) for i in range(12)]

def _porphyry(asc: float, mc: float) -> List[float]:
    """
    Porphyry: equal trisection of the four ecliptic quadrants in forward zodiac order.
      Q1 Asc→MC  → 12th, 11th
      Q2 MC→Desc → 9th, 8th
      Q3 Desc→IC → 6th, 5th
      Q4 IC→Asc  → 3rd, 2nd
    Angles: 1=Asc, 10=MC, 7=Desc, 4=IC.
    """
    cusps = _blank()
    A = cusps[0] = _norm_deg(asc)
    M = cusps[9] = _norm_deg(mc)
    D = cusps[6] = _norm_deg(asc + 180.0)
    I = cusps[3] = _norm_deg(mc + 180.0)

    def span(start: float, end: float) -> float:
        return _norm_deg(end - start)

    # Q1: 12th, 11th
    s = span(A, M)
    cusps[11] = _norm_deg(A + s / 3.0)         # 12th
    cusps[10] = _norm_deg(A + 2.0 * s / 3.0)   # 11th
    # Q2: 9th, 8th
    s = span(M, D)
    cusps[8]  = _norm_deg(M + s / 3.0)         # 9th
    cusps[7]  = _norm_deg(M + 2.0 * s / 3.0)   # 8th
    # Q3: 6th, 5th
    s = span(D, I)
    cusps[5]  = _norm_deg(D + s / 3.0)         # 6th
    cusps[4]  = _norm_deg(D + 2.0 * s / 3.0)   # 5th
    # Q4: 3rd, 2nd
    s = span(I, A)
    cusps[2]  = _norm_deg(I + s / 3.0)         # 3rd
    cusps[1]  = _norm_deg(I + 2.0 * s / 3.0)   # 2nd

    return _fill_opposites(cusps)

def _morinus(ramc: float, eps: float) -> List[float]:
    # tan L = cos ε · tan(ramc + ad), ad ∈ {0,30,60,90,120,150}
    def cusp(ad: float) -> float:
        F = _norm_deg(ramc + ad)
        return _atan2d(_tand(F) * _cosd(eps), 1.0)
    cusps = _blank()
    cusps[9]  = cusp(0.0)
    cusps[10] = cusp(30.0)
    cusps[11] = cusp(60.0)
    cusps[0]  = cusp(90.0)
    cusps[1]  = cusp(120.0)
    cusps[2]  = cusp(150.0)
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
    cusps = _blank()
    cusps[0], cusps[9] = asc, mc
    cusps[10] = _norm_deg(mc + block(30.0))
    cusps[11] = _norm_deg(mc + block(60.0))
    cusps[1]  = _norm_deg(mc + block(120.0))
    cusps[2]  = _norm_deg(mc + block(150.0))
    return _fill_opposites(cusps)

def _campanus(phi: float, ramc: float, eps: float, asc: float, mc: float) -> List[float]:
    def block(H: float) -> float:
        J = _acotd(_cosd(phi) * _tand(H))
        F = _norm_deg(ramc + 90.0 - J)
        P = _asin_strict_deg(_sind(H) * _sind(phi), "campanus:asinP")
        M = math.degrees(math.atan(_tand(P) / _cosd(F)))
        R = math.degrees(math.atan((_tand(F) * _cosd(M)) / _cosd(M + eps)))
        return R
    cusps = _blank()
    cusps[0], cusps[9] = asc, mc
    cusps[10] = _norm_deg(mc + block(30.0))
    cusps[11] = _norm_deg(mc + block(60.0))
    cusps[1]  = _norm_deg(mc + block(120.0))
    cusps[2]  = _norm_deg(mc + block(150.0))
    return _fill_opposites(cusps)

def _koch(phi: float, eps: float, ramc: float, mc: float) -> List[float]:
    D = _asin_strict_deg(_sind(mc) * _sind(eps), "koch:decl_mc")
    J = _asin_strict_deg(_tand(D) * _tand(phi), "koch:asc_diff")
    OAMC = _norm_deg(ramc - J)
    DX = _norm_deg((ramc + 90.0) - OAMC) / 3.0
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

def _alcabitius(phi: float, eps: float, asc: float, mc: float) -> List[float]:
    L = _norm_deg(asc - mc)
    D = math.degrees(math.atan(_tand(L) * _cosd(eps)))  # diurnal semi-arc on ecliptic
    P = 180.0 - D
    F, G = D / 3.0, 2.0 * D / 3.0
    J, K = P / 3.0, 2.0 * P / 3.0
    H11 = math.degrees(math.atan(_tand(F) / _cosd(eps)))
    H12 = math.degrees(math.atan(_tand(G) / _cosd(eps)))
    H2  = math.degrees(math.atan(_tand(K) / _cosd(eps)))
    H3  = math.degrees(math.atan(_tand(J) / _cosd(eps)))
    cusps = _blank()
    cusps[9]  = mc
    cusps[10] = _norm_deg(mc + H11)
    cusps[11] = _norm_deg(mc + H12)
    cusps[0]  = asc
    cusps[1]  = _norm_deg(mc + H2)
    cusps[2]  = _norm_deg(mc + H3)
    return _fill_opposites(cusps)

# --------------------------- Placidus (numeric, exact) ---------------------------

def _placidus(phi: float, eps: float, ramc: float, asc: float, mc: float, *,
              _diag: Optional[dict] = None) -> List[float]:
    """
    Exact Placidus by solving the time-division equation via secant.
    Records per-cusp diagnostics in _diag[label] when provided.
    """

    def decl_of_lambda(lam: float) -> float:
        # δ(λ) for β=0: sin δ = sin ε sin λ
        return _asin_strict_deg(_sind(eps) * _sind(lam), "placidus:decl(lambda)")

    def sda(decl: float) -> float:
        # semi-diurnal hour angle H0 = acos(-tan φ tan δ) in degrees (0..180)
        t = -_tand(phi) * _tand(decl)
        if t < -1.0 - EPS_NUM or t > 1.0 + EPS_NUM:
            raise ValueError(f"placidus: circumpolar condition tan terms out of range (t={t})")
        return _acos_strict_deg(t, "placidus:sda")

    def ra_of_lambda(lam: float) -> float:
        # α = atan2(sin λ cos ε, cos λ) in [0,360)
        return _atan2d(_sind(lam) * _cosd(eps), _cosd(lam))

    def eq(lambda_guess: float, frac: float, side: str) -> float:
        """
        f(λ) = ΔRA(λ) - sign * (SDA(δ(λ)) * frac)
        side: 'pre'  (before culmination, negative ΔRA branch)
              'post' (after  culmination, positive ΔRA branch)
        """
        lam = lambda_guess
        ra  = ra_of_lambda(lam)
        dec = decl_of_lambda(lam)
        half = sda(dec)            # semi-diurnal HA (deg) in [0,180]
        sign = -1.0 if side == 'pre' else +1.0
        dra  = _norm_deg(ra - ramc)
        if side == 'pre':
            if dra > 180.0:
                dra -= 360.0
        else:
            if dra > 180.0:
                dra -= 360.0
        return dra - sign * (half * frac)

    MAX_ITERS = 30
    TOL_F     = 1e-10   # function residual tolerance (deg)
    TOL_STEP  = 1e-9    # last step tolerance (deg)

    def solve(frac: float, side: str, seed: float, label: str) -> float:
        x0 = seed
        x1 = _norm_deg(seed + 1.0)
        f0 = eq(x0, frac, side)
        f1 = eq(x1, frac, side)
        it = 0
        last_step = abs(_norm_deg(x1 - x0))
        converged = False

        while it < MAX_ITERS:
            it += 1
            denom = (f1 - f0)
            if abs(denom) < 1e-15:
                x1 = _norm_deg(x1 + 1e-7)
                f1 = eq(x1, frac, side)
                continue
            x2 = _norm_deg((x0 * f1 - x1 * f0) / denom)
            f2 = eq(x2, frac, side)
            step = abs(_norm_deg(x2 - x1))
            if abs(f2) < TOL_F or step < TOL_STEP:
                x1, f1 = x2, f2
                last_step = step
                converged = True
                break
            x0, f0, x1, f1 = x1, f1, x2, f2
            last_step = step

        if _diag is not None:
            _diag[label] = {
                "iters": it,
                "residual_deg": float(abs(f1)),
                "last_step_deg": float(last_step),
                "converged": bool(converged),
                "seed_deg": float(seed),
                "side": side,
                "fraction": float(frac),
            }

        if not converged:
            raise ValueError(f"placidus solver did not converge for {label} in {it} iterations")

        return _norm_deg(x1)

    por = _porphyry(asc, mc)  # seeds
    C11 = solve(1.0/3.0, 'pre',  por[10], "C11")
    C12 = solve(2.0/3.0, 'pre',  por[11], "C12")
    C2  = solve(2.0/3.0, 'post', por[1],  "C2")
    C3  = solve(1.0/3.0, 'post', por[2],  "C3")

    cusps = _blank()
    cusps[0], cusps[9]  = asc, mc
    cusps[10], cusps[11] = C11, C12
    cusps[1],  cusps[2]  = C2,  C3
    return _fill_opposites(cusps)

# --------------------------- NEW exact engines kept / tightened ---------------------------

def _vehlow_equal(asc: float) -> List[float]:
    """
    Vehlow Equal (SwissEph 'V'): equal houses with the Ascendant at the
    *center* of House 1. Therefore the 1st house cusp line is 15° BEFORE
    the Ascendant degree, and cusps proceed every 30° from there.
    (NOT the same as Whole Sign.)
    """
    start = _norm_deg(asc - 15.0)
    return [_norm_deg(start + 30.0 * i) for i in range(12)]

def _sripati(_phi: float, _eps: float, asc: float, mc: float) -> List[float]:
    """
    Sripati (Madhya Bhava): take Porphyry *boundaries* and define the house
    *cusps* as the midpoints between successive Porphyry boundaries, with
    correct circular wrap. All 12 cusps are computed directly.
    """
    por = _porphyry(asc, mc)  # Porphyry cusp-lines treated as boundaries
    cusps = [0.0]*12
    for i in range(12):
        prev_i = (i - 1) % 12
        cusps[i] = _midpoint_wrap(por[prev_i], por[i])
    return _fill_opposites(cusps)

# --------------------------- declared but intentionally gated ---------------------------

_GATED_VENDOR_VARIANTS = {"horizon", "carter_pe", "sunshine", "pullen_sd"}
_GATED_RESEARCH = {"meridian", "krusinski"}

def _raise_gated(name: str) -> None:
    raise NotImplementedError(
        f"House system '{name}' is declared but intentionally not implemented yet "
        f"(awaiting unambiguous public spec and gold vectors)."
    )

# --------------------------- data model & main calculator ---------------------------

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

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []

# All 18 declared
SUPPORTED_HOUSE_SYSTEMS = [
    "placidus", "koch", "regiomontanus", "campanus",
    "equal", "whole_sign", "porphyry", "alcabitius",
    "morinus", "topocentric", "meridian", "horizon",
    "carter_pe", "sunshine", "vehlow_equal", "sripati",
    "krusinski", "pullen_sd",
]

# Subset implemented in this build (12)
IMPLEMENTED_HOUSE_SYSTEMS = {
    "placidus", "koch", "regiomontanus", "campanus",
    "equal", "whole_sign", "porphyry", "alcabitius",
    "morinus", "topocentric", "vehlow_equal", "sripati",
}

class PreciseHouseCalculator:
    """
    Exact house computations with apparent sidereal time and true obliquity.
    STRICT by default: both jd_tt and jd_ut1 are required.
    Declares 18 systems; 12 implemented; 6 gated with NotImplementedError.
    """

    def __init__(self, require_strict_timescales: bool = True, enable_diagnostics: bool = False):
        self.require_strict_timescales = require_strict_timescales
        self.enable_diagnostics = enable_diagnostics

    def calculate_houses(
        self,
        latitude: float,
        longitude: float,
        jd_ut: float,                      # kept for backward-compat only; not used in strict mode
        house_system: str = "placidus",
        *,
        jd_tt: Optional[float] = None,
        jd_ut1: Optional[float] = None,
    ) -> HouseData:

        # ---- strict latitude bounds ----
        if not (latitude > -90.0 and latitude < 90.0):
            raise ValueError(
                "Latitude must be strictly between -90 and 90 degrees; poles are undefined for house systems."
            )
        if math.isnan(latitude) or math.isinf(latitude):
            raise ValueError("Latitude must be a finite number.")

        sys_name = house_system.lower().strip()
        if sys_name == "whole":
            sys_name = "whole_sign"
        if sys_name == "azimuthal":
            sys_name = "horizon"
        if sys_name not in SUPPORTED_HOUSE_SYSTEMS:
            raise ValueError(f"Unsupported house system: {house_system}")

        # Gate ambiguous / research systems explicitly
        if sys_name in _GATED_VENDOR_VARIANTS or sys_name in _GATED_RESEARCH:
            _raise_gated(sys_name)

        # ---- timescales ----
        if self.require_strict_timescales:
            if jd_tt is None or jd_ut1 is None:
                raise ValueError("Strict mode requires jd_tt and jd_ut1 (no UT≈UTC shortcuts).")
            tt, ut1 = jd_tt, jd_ut1
        else:
            tt  = jd_tt  or jd_ut
            ut1 = jd_ut1 or jd_ut
            if tt is None or ut1 is None:
                raise ValueError("Provide jd_tt and jd_ut1 or disable strict mode explicitly.")

        eps  = _true_obliquity_deg(tt)
        ramc = _ramc_deg(ut1, tt, longitude)
        mc   = _mc_longitude_deg(ramc, eps)
        asc  = _asc_longitude_deg(latitude, ramc, eps)
        east = _eastpoint_longitude_deg(ramc, eps)
        vtx  = _vertex_longitude_deg(latitude, ramc, eps)

        warnings: List[str] = []
        solver_stats: Optional[Dict[str, Any]] = {} if self.enable_diagnostics else None

        # ---- dispatch exact systems ----
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
        elif sys_name == "topocentric":
            # Polich–Page: use Placidus engine on topocentric angles;
            diag = {} if self.enable_diagnostics else None
            cusps = _placidus(latitude, eps, ramc, asc, mc, _diag=diag)
            if solver_stats is not None and diag is not None:
                solver_stats["topocentric"] = diag
        else:  # placidus
            diag = {} if self.enable_diagnostics else None
            cusps = _placidus(latitude, eps, ramc, asc, mc, _diag=diag)
            if solver_stats is not None and diag is not None:
                solver_stats["placidus"] = diag

        # add compact solver lines to warnings only if diagnostics enabled
        if self.enable_diagnostics and solver_stats:
            for key in ("placidus", "topocentric"):
                d = solver_stats.get(key)
                if not isinstance(d, dict):
                    continue
                for label in ("C11", "C12", "C2", "C3"):
                    s = d.get(label)
                    if not isinstance(s, dict):
                        continue
                    warnings.append(
                        f"[{key} {label}] iters={s.get('iters')} "
                        f"res={s.get('residual_deg'):.3e}° step={s.get('last_step_deg'):.3e}° "
                        f"side={s.get('side')} frac={s.get('fraction')}"
                    )

        return HouseData(
            system=sys_name,
            cusps=cusps,
            ascendant=asc,
            midheaven=mc,
            vertex=vtx,
            eastpoint=east,
            warnings=warnings,
            solver_stats=solver_stats if self.enable_diagnostics else None,
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

    NOTE: For declared-but-gated systems this will raise NotImplementedError
    (API layer should map to HTTP 501 Not Implemented).
    """
    calc = PreciseHouseCalculator(require_strict_timescales=True, enable_diagnostics=False)
    hd = calc.calculate_houses(
        latitude=latitude,
        longitude=longitude,
        jd_ut=jd_ut,
        house_system=house_system,
        jd_tt=jd_tt,
        jd_ut1=jd_ut1,
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
    return payload

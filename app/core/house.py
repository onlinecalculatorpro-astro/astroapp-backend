# app/core/house.py
from __future__ import annotations

"""
House-system policy façade.

What this module guarantees:
- Canonical, slug-based normalization of user-facing house-system names.
- Strict timescale requirements (JD_TT + JD_UT1) with explicit erroring if absent.
- A transparent polar-latitude policy (soft vs hard limits) with clear warnings.
- Deterministic fallback chains (env-augmentable), with a robust bias near poles.
- A stable result payload shape (asc/mc/cusps + meta) for routes/clients.
- Clean separation between public labels and engine labels.

Back-end:
    app.core.houses_advanced.PreciseHouseCalculator

POLAR POLICY (high level)
1) Validate latitude: must be strictly between -90 and 90, with an absolute pole guard.
2) If |lat| ≥ hard_limit and system ∈ HARD_REJECT → emit warning and try fallbacks.
3) If policy == "reject_above_66deg" and system ∈ RISKY and |lat| > soft_limit → raise error.
4) Build fallback chain:
       requested → env chain → default chain → robust tail
   If policy == "fallback_to_equal_above_66deg" and system is risky at |lat|>soft_limit:
       bias robust families first: equal/whole/equal_from_mc/natural_houses, then porphyry, then others.
5) Try in order (unless numeric fallbacks disabled); return the first success.
"""

from typing import Any, Dict, Final, List, Optional, Tuple
import difflib
import json
import math
import os

# ──────────────────────────────────────────────────────────────────────────────
# Engine import
# ──────────────────────────────────────────────────────────────────────────────
try:
    from app.core.houses_advanced import PreciseHouseCalculator, HouseData
except Exception as _e:  # pragma: no cover
    raise RuntimeError(
        "houses_advanced backend unavailable. Install numpy>=1.26 and pyerfa>=2.0.1. "
        f"Original import error: {type(_e).__name__}: {_e}"
    ) from _e


# ──────────────────────────────────────────────────────────────────────────────
# Optional metrics (best-effort; never crash)
# ──────────────────────────────────────────────────────────────────────────────
def _met_warn(kind: str) -> None:
    try:
        from app.main import MET_WARNINGS  # type: ignore
        MET_WARNINGS.labels(kind=kind).inc()
    except Exception:
        pass


def _met_fallback(requested: str, fallback: str) -> None:
    try:
        from app.main import MET_FALLBACKS  # type: ignore
        MET_FALLBACKS.labels(requested=requested, fallback=fallback).inc()
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Environment / policy knobs
# ──────────────────────────────────────────────────────────────────────────────
POLAR_SOFT_LIMIT_DEG: Final[float] = float(os.getenv("ASTRO_POLAR_SOFT_LAT", "66.0"))
POLAR_HARD_LIMIT_DEG: Final[float] = float(os.getenv("ASTRO_POLAR_HARD_LAT", "80.0"))
POLAR_ABSOLUTE_LIMIT_DEG: Final[float] = 89.999999  # guard near exact poles
POLAR_POLICY: Final[str] = os.getenv("ASTRO_POLAR_POLICY", "fallback_to_equal_above_66deg").strip()

# Time-division (and similar) systems that go numerically unstable at high lat
_HARD_REJECT_AT_POLAR: Final[frozenset[str]] = frozenset({"placidus", "koch", "topocentric", "alcabitius"})
_RISKY_AT_POLAR: Final[frozenset[str]] = frozenset({
    "placidus", "koch", "regiomontanus", "campanus", "topocentric", "alcabitius", "morinus"
})

# Numeric multi-fallback toggle
NUMERIC_FALLBACK_ENABLED: Final[bool] = os.getenv("ASTRO_HOUSES_NUMERIC_FALLBACK", "1").lower() in ("1", "true", "yes", "on")

# Optional JSON for custom fallback chains
#   ASTRO_HOUSES_FALLBACK_JSON='{"placidus":["koch","regiomontanus","campanus","porphyry","equal","whole"]}'
_FALLBACK_JSON: Optional[str] = os.getenv("ASTRO_HOUSES_FALLBACK_JSON")

# Engine passthrough defaults
DEFAULT_DIAGNOSTICS: Final[bool] = os.getenv("ASTRO_HOUSES_DEBUG", "").lower() in ("1", "true", "yes", "on")
DEFAULT_VALIDATION: Final[bool] = os.getenv("ASTRO_HOUSES_VALIDATE", "").lower() in ("1", "true", "yes", "on")

# Boot warnings (non-fatal) to surface once per process
_BOOT_WARNINGS: List[str] = []


# ──────────────────────────────────────────────────────────────────────────────
# Public systems (canonical labels)
# ──────────────────────────────────────────────────────────────────────────────
SUPPORTED_HOUSE_SYSTEMS: Final[Tuple[str, ...]] = (
    # core
    "placidus", "koch", "regiomontanus", "campanus",
    "porphyry", "alcabitius", "morinus", "topocentric",
    "equal", "whole",
    # extras
    "vehlow_equal", "sripati",
    # v10+ family
    "equal_from_mc", "natural_houses",
    "bhava_chalit_sripati", "bhava_chalit_equal_from_mc",
    # declared but gated (HTTP 501)
    "meridian", "horizon", "carter_pe", "sunshine", "krusinski", "pullen_sd",
)

# Systems declared but intentionally not implemented (HTTP 501 from routes)
GATED_NOT_IMPLEMENTED: Final[frozenset[str]] = frozenset({
    "horizon", "carter_pe", "sunshine", "pullen_sd",  # ambiguous vendor variants
    "meridian", "krusinski",                          # pending spec/gold vectors
})


# ──────────────────────────────────────────────────────────────────────────────
# Alias normalization (slug-based)
# ──────────────────────────────────────────────────────────────────────────────
def _slug(s: str | None) -> str:
    if not s:
        return ""
    return "".join(ch for ch in s.lower() if ch.isalnum())


def _assert_unique_slugs(names: List[str]) -> None:
    seen: Dict[str, str] = {}
    for n in names:
        sl = _slug(n)
        if sl in seen and seen[sl] != n:
            raise RuntimeError(
                f"Slug collision: '{n}' and '{seen[sl]}' both normalize to '{sl}'."
            )
        seen[sl] = n


# Guard at import to catch accidental collisions early
_assert_unique_slugs(list(SUPPORTED_HOUSE_SYSTEMS))

_CANON_FROM_SLUG: Final[Dict[str, str]] = {
    # implemented 1:1
    "placidus": "placidus",
    "koch": "koch",
    "regiomontanus": "regiomontanus",
    "campanus": "campanus",
    "porphyry": "porphyry",
    "alcabitius": "alcabitius",
    "morinus": "morinus",
    "topocentric": "topocentric",
    "equal": "equal",
    # whole-sign family
    "whole": "whole",
    "wholesign": "whole",
    # extras
    "vehlowequal": "vehlow_equal",
    "sripati": "sripati",
    "equalfrommc": "equal_from_mc",
    "naturalhouses": "natural_houses",
    "bhavachalitsripati": "bhava_chalit_sripati",
    "bhavachalitequalfrommc": "bhava_chalit_equal_from_mc",
    # gated & common aliases
    "meridian": "meridian",
    "horizon": "horizon",
    "carterpe": "carter_pe",
    "sunshine": "sunshine",
    "krusinski": "krusinski",
    "pullensd": "pullen_sd",
    # engine synonym
    "azimuthal": "horizon",
}


def _suggest_systems(name: str, n: int = 5) -> List[str]:
    slug = _slug(name)
    pool = list(SUPPORTED_HOUSE_SYSTEMS)
    # direct string similarity on canonical labels
    best = difflib.get_close_matches(name.lower(), pool, n=n, cutoff=0.6)
    # help if user typed a compact alias (slug form)
    slug_hits = difflib.get_close_matches(slug, list(_CANON_FROM_SLUG.keys()), n=n, cutoff=0.6)
    for sh in slug_hits:
        canon = _CANON_FROM_SLUG.get(sh)
        if canon and canon not in best:
            best.append(canon)
    return best[:n]


def _normalize_public_name(name: Optional[str]) -> str:
    if not name:
        return "placidus"
    canon = _CANON_FROM_SLUG.get(_slug(name))
    if not canon:
        suggestions = _suggest_systems(name, n=5)
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise ValueError(f"unsupported house system: '{name}'.{hint}")
    return canon


def canonicalize_system(name: str) -> str:
    """External helper: map user-facing input → canonical public label."""
    return _normalize_public_name(name)


def _public_to_engine(system_public: str) -> str:
    """Public label → engine label (math name)."""
    return "whole_sign" if system_public == "whole" else system_public


def _engine_to_public(system_engine: str) -> str:
    """Engine label → preferred public label."""
    return "whole" if system_engine == "whole_sign" else system_engine


def list_supported_house_systems() -> List[str]:
    return list(SUPPORTED_HOUSE_SYSTEMS)


# ──────────────────────────────────────────────────────────────────────────────
# Fallback chains
# ──────────────────────────────────────────────────────────────────────────────
def _dedupe(seq: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for s in seq:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _default_fallback_chain_for(requested_public: str) -> List[str]:
    chains: Dict[str, List[str]] = {
        "placidus":      ["koch", "regiomontanus", "campanus", "porphyry", "equal_from_mc", "natural_houses", "equal", "whole"],
        "koch":          ["placidus", "regiomontanus", "campanus", "porphyry", "equal_from_mc", "natural_houses", "equal", "whole"],
        "topocentric":   ["placidus", "koch", "regiomontanus", "campanus", "porphyry", "equal_from_mc", "natural_houses", "equal", "whole"],
        "alcabitius":    ["regiomontanus", "campanus", "porphyry", "equal_from_mc", "natural_houses", "equal", "whole"],
        "morinus":       ["regiomontanus", "campanus", "porphyry", "equal_from_mc", "natural_houses", "equal", "whole"],
        "regiomontanus": ["campanus", "porphyry", "equal_from_mc", "natural_houses", "equal", "whole"],
        "campanus":      ["regiomontanus", "porphyry", "equal_from_mc", "natural_houses", "equal", "whole"],
        "porphyry":      ["equal_from_mc", "natural_houses", "equal", "whole"],
        "equal":         ["equal_from_mc", "natural_houses", "porphyry", "whole"],
        "whole":         ["equal", "equal_from_mc", "natural_houses", "porphyry"],
        "vehlow_equal":  ["equal", "equal_from_mc", "natural_houses", "whole"],
        "sripati":       ["porphyry", "equal_from_mc", "natural_houses", "equal", "whole"],
        "equal_from_mc": ["natural_houses", "equal", "whole", "porphyry"],
        "natural_houses":["equal", "equal_from_mc", "porphyry", "whole"],
        "bhava_chalit_sripati":        ["sripati", "porphyry", "equal", "whole"],
        "bhava_chalit_equal_from_mc":  ["equal_from_mc", "natural_houses", "equal", "whole"],
    }
    return chains.get(requested_public, ["equal_from_mc", "natural_houses", "equal", "whole"])


def _fallback_chain(requested_public: str) -> List[str]:
    requested_public = _normalize_public_name(requested_public)
    default_chain = _default_fallback_chain_for(requested_public)

    env_chain: List[str] = []
    if _FALLBACK_JSON:
        try:
            mapping = json.loads(_FALLBACK_JSON)
            maybe = mapping.get(requested_public)
            if isinstance(maybe, list):
                env_chain = [_normalize_public_name(str(x)) for x in maybe]
        except Exception as e:
            _BOOT_WARNINGS.append(f"ASTRO_HOUSES_FALLBACK_JSON ignored: {type(e).__name__}: {e}")

    # robust tail always present
    tail = ["equal_from_mc", "natural_houses", "equal", "whole"]
    chain = [requested_public] + env_chain + default_chain + tail
    return _dedupe(chain)


# ──────────────────────────────────────────────────────────────────────────────
# Polar helpers
# ──────────────────────────────────────────────────────────────────────────────
def _needs_polar_fallback(public_system: str, latitude: float, limit: float) -> bool:
    return abs(latitude) > limit and public_system in _RISKY_AT_POLAR


def _allowed_at_lat(public_system: str, lat: float, hard_lim: float) -> bool:
    return not (abs(lat) >= hard_lim and public_system in _HARD_REJECT_AT_POLAR)


def _normalize_lon180(x: float) -> float:
    """Normalize longitude to (-180, 180], avoiding -180 exact."""
    v = ((float(x) + 180.0) % 360.0) - 180.0
    return 180.0 if math.isclose(v, -180.0, abs_tol=1e-12) else v


# ──────────────────────────────────────────────────────────────────────────────
# Policy façade
# ──────────────────────────────────────────────────────────────────────────────
def compute_houses_with_policy(
    *,
    # primary
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    system: Optional[str] = None,
    jd_tt: Optional[float] = None,
    jd_ut1: Optional[float] = None,
    jd_ut: Optional[float] = None,  # echoed only; not used by engine in strict mode
    diagnostics: Optional[bool] = None,
    validation: Optional[bool] = None,

    # legacy aliases (lenient)
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    requested_house_system: Optional[str] = None,
    enable_diagnostics: Optional[bool] = None,

    # policy overrides
    polar_policy: Optional[str] = None,
    polar_soft_limit: Optional[float] = None,
    polar_hard_limit: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute houses under the configured polar policy with fallbacks.

    Raises:
      ValueError           — bad inputs / strict-policy rejection
      NotImplementedError  — declared but gated systems (HTTP 501)
      RuntimeError         — engine failure after exhausting fallbacks
    """
    # ---- inputs
    latitude = lat if lat is not None else latitude
    longitude = lon if lon is not None else longitude
    if latitude is None or longitude is None:
        raise ValueError("lat/lon are required")

    lat_f = float(latitude)
    lon_f = _normalize_lon180(float(longitude))

    if not (-90.0 < lat_f < 90.0):
        raise ValueError("latitude must be strictly between -90 and 90 degrees")
    if abs(lat_f) >= POLAR_ABSOLUTE_LIMIT_DEG:
        raise ValueError(f"latitude {lat_f:.8f}° is at/near the pole; house systems are undefined")

    requested_public_raw = system if system is not None else requested_house_system
    requested_public = _normalize_public_name(requested_public_raw)

    if requested_public in GATED_NOT_IMPLEMENTED:
        raise NotImplementedError(
            f"house_system '{requested_public}' is declared but not implemented yet (pending spec/vectors)."
        )

    policy = (polar_policy or POLAR_POLICY).strip()
    soft_lim = float(polar_soft_limit if polar_soft_limit is not None else POLAR_SOFT_LIMIT_DEG)
    hard_lim = float(polar_hard_limit if polar_hard_limit is not None else POLAR_HARD_LIMIT_DEG)

    # strict timescales
    if jd_tt is None or jd_ut1 is None:
        raise ValueError(
            "houses: strict mode requires jd_tt and jd_ut1. "
            "Provide both (UTC→TT/UT1 via leap seconds & ΔT)."
        )

    # flags
    if diagnostics is None:
        diagnostics = enable_diagnostics
    if diagnostics is None:
        diagnostics = DEFAULT_DIAGNOSTICS
    if validation is None:
        validation = DEFAULT_VALIDATION

    warnings: List[str] = []
    warnings.extend(_BOOT_WARNINGS)  # surface boot-time notices (e.g., bad env JSON)

    # ---- polar rules (hard limit)
    if abs(lat_f) >= hard_lim and requested_public in _HARD_REJECT_AT_POLAR:
        warnings.append(
            f"Requested '{requested_public}' is unstable at latitude {lat_f:.2f}° "
            f"(≥ hard limit {hard_lim}°). Trying fallbacks."
        )
        _met_warn("polar_hard_limit")

    # ---- strict rejection policy (soft limit)
    if policy == "reject_above_66deg" and _needs_polar_fallback(requested_public, lat_f, soft_lim):
        _met_warn("polar_reject_strict")
        raise ValueError(f"house_system '{requested_public}' is unstable above |lat|>{soft_lim}°")

    # ---- build fallback chain and apply latitude gating
    chain_all = _fallback_chain(requested_public)
    chain = [s for s in chain_all if _allowed_at_lat(s, lat_f, hard_lim)]
    if not chain:
        raise ValueError("no allowed house systems at this latitude under current policy")

    # Bias toward robust systems if risky at soft polar latitudes
    if policy == "fallback_to_equal_above_66deg" and _needs_polar_fallback(chain[0], lat_f, soft_lim):
        def _robust_rank(s: str) -> int:
            if s in ("equal", "whole", "equal_from_mc", "natural_houses"):
                return 0
            if s == "porphyry":
                return 1
            return 2
        chain.sort(key=_robust_rank)
        warnings.append(
            f"Soft polar policy: '{requested_public}' at |lat|>{soft_lim}°. "
            f"Prioritizing robust families first."
        )
        _met_warn("polar_soft_fallback")

    # ---- compute with fallbacks
    calc = PreciseHouseCalculator(
        require_strict_timescales=True,
        enable_diagnostics=bool(diagnostics),
        enable_validation=bool(validation),
    )

    last_err: Optional[Exception] = None
    used_public: Optional[str] = None
    hd: Optional[HouseData] = None

    def _compute(sys_public: str) -> HouseData:
        return calc.calculate_houses(
            latitude=lat_f,
            longitude=lon_f,
            jd_ut=float(jd_ut) if jd_ut is not None else 0.0,  # ignored by strict mode; echoed for meta
            house_system=_public_to_engine(sys_public),
            jd_tt=float(jd_tt),
            jd_ut1=float(jd_ut1),
        )

    for sys_public in chain:
        # If numeric fallbacks disabled, only try the first choice
        if sys_public != chain[0] and not NUMERIC_FALLBACK_ENABLED:
            break
        try:
            hd = _compute(sys_public)
            used_public = _engine_to_public(hd.system)
            if sys_public != requested_public:
                warnings.append(f"Fallback applied: '{requested_public}' → '{sys_public}'.")
                _met_fallback(requested=requested_public, fallback=sys_public)
            break
        except Exception as e:
            last_err = e
            continue

    if hd is None or used_public is None:
        if last_err:
            raise RuntimeError(f"houses engine failed for all fallbacks (last: {type(last_err).__name__}: {last_err})")
        raise RuntimeError("houses engine failed and no fallback succeeded")

    # ---- payload (stable; keep legacy keys)
    all_warnings = list(warnings) + list(hd.warnings or [])

    payload: Dict[str, Any] = {
        # input vs used
        "requested_system": requested_public,
        "system": used_public,
        "engine_system": hd.system,
        # angles & cusps
        "asc": hd.ascendant,
        "mc": hd.midheaven,
        "asc_deg": hd.ascendant,   # legacy
        "mc_deg": hd.midheaven,    # legacy
        "cusps": hd.cusps,
        "cusps_deg": hd.cusps,     # legacy alias
        "angles": {"ASC": {"deg": hd.ascendant}, "MC": {"deg": hd.midheaven}},
        # auxiliary points
        "vertex": getattr(hd, "vertex", None),
        "eastpoint": getattr(hd, "eastpoint", None),
        # policy echo
        "policy": {
            "polar_policy": policy,
            "polar_soft_limit_deg": soft_lim,
            "polar_hard_limit_deg": hard_lim,
            "absolute_pole_guard_deg": POLAR_ABSOLUTE_LIMIT_DEG,
            "hard_reject_systems_at_polar": sorted(_HARD_REJECT_AT_POLAR),
            "risky_at_polar": sorted(_RISKY_AT_POLAR),
            "numeric_fallback_enabled": NUMERIC_FALLBACK_ENABLED,
            "fallback_chain_declared": chain_all,  # before gating/bias
            "fallback_chain_final": chain,         # after gating/bias
        },
        # meta echo
        "meta": {
            "timescales": {
                "jd_tt": float(jd_tt),
                "jd_ut1": float(jd_ut1),
                "jd_ut_legacy": float(jd_ut) if jd_ut is not None else None,
            },
            "input": {
                "requested_input_raw": requested_public_raw,
                "requested_input_slug": _slug(requested_public_raw) if requested_public_raw else None,
            },
            "diagnostics_enabled": bool(diagnostics),
            "validation_enabled": bool(validation),
        },
        # warnings (policy + engine)
        "warnings": all_warnings,
    }

    # Optional attachments from engine (if present)
    if getattr(hd, "solver_stats", None):
        payload["solver_stats"] = hd.solver_stats
    if getattr(hd, "validation_results", None):
        try:
            payload["validation_results"] = [r._asdict() for r in hd.validation_results]  # type: ignore[attr-defined]
        except Exception:
            payload["validation_results"] = hd.validation_results  # already serializable?
    if getattr(hd, "error_budget", None):
        eb = hd.error_budget
        payload["error_budget"] = {
            "coordinate_precision": getattr(eb, "coordinate_precision", None),
            "algorithm_truncation": getattr(eb, "algorithm_truncation", None),
            "time_scale_uncertainty": getattr(eb, "time_scale_uncertainty", None),
            "reference_comparison": getattr(eb, "reference_comparison", None),
            "total_rss": getattr(eb, "total_rss", None),
            "certified": eb.certify_accuracy() if hasattr(eb, "certify_accuracy") else None,
        }

    return payload


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────
__all__ = [
    "compute_houses_with_policy",
    "list_supported_house_systems",
    "SUPPORTED_HOUSE_SYSTEMS",
    "POLAR_POLICY",
    "POLAR_SOFT_LIMIT_DEG",
    "POLAR_HARD_LIMIT_DEG",
    "POLAR_ABSOLUTE_LIMIT_DEG",
    "GATED_NOT_IMPLEMENTED",
    "canonicalize_system",
]

# app/core/vedic_predictive.py
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Vedic predictive helpers — daśā, varga wrappers, Yoga detection,
and (NEW) gochar/ingress/station wrappers wired to app.core.vedic_gochar.

What’s here
-----------
- Windowed daśā periods across multiple systems:
    vimshottari (preferred engine), ashtottari, yogini, chara (Jaimini), kalachakra
- Legacy Vimśottarī generator kept for back-compat (used only if preferred engine missing).
- Vargas (divisional charts): thin wrappers that DELEGATE to `varga_charts`.
- Yoga detection:
    • Primary path delegates to `app.core.yoga.compute_yogas` (sidereal-first)
    • Legacy “basic” detectors are kept as a fallback for precomputed points/cusps.
- (NEW) Vedic Gochar:
    • graha dṛṣṭi transit hits (degree-true)
    • rāśi & nakṣatra ingresses
    • retrograde/direct stations
    • feature_drishti_proximity re-export
- (NEW) Śaḍbala and Aṣṭakavarga:
    • Thin, defensive wrappers delegating to app.core.shadbala / app.core.ashtakavarga
      with civil+tz+coords, sidereal-first defaults.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple, Literal, Optional, Set, Union
from datetime import datetime, timezone
import re
import math

from app.core.common_predictive import (
    norm360, sign_index, angdiff, compute_houses, timescales_from_civil
)
from app.core.ephem_singleton import TS, PLANETS  # TS is used for TT<->UTC conversions

# ───────────────────────── Gochar/Ingress/Stations (NEW) ─────────────────────
_GOCHAR_OK = False
try:
    from app.core.vedic_gochar import (  # type: ignore
        find_gochar_in_range as _find_gochar_in_range,
        find_rashi_ingresses_in_range as _find_rashi_ingresses_in_range,
        find_nakshatra_ingresses_in_range as _find_nakshatra_ingresses_in_range,
        find_stations_in_range as _find_stations_in_range,
        feature_drishti_proximity as _feature_drishti_proximity,
    )
    _GOCHAR_OK = True
except Exception:
    _GOCHAR_OK = False

# ───────────────────────── Preferred Vimśottarī engine ───────────────────────
try:
    from app.core.vimshottari_dasha import (
        generate_vimshottari_tree,
        flatten_periods,
    )
    _VIM_ENGINE_OK = True
except Exception:
    _VIM_ENGINE_OK = False

# ───────────────────────── Optional engines ──────────────────────────────────
try:
    from app.core.ashtottari_dasha import compute_ashtottari as _compute_ashtottari
    _ASHTO_OK = True
except Exception:
    _ASHTO_OK = False

try:
    from app.core.yogini_dasha import compute_yogini as _compute_yogini
    _YOGINI_OK = True
except Exception:
    _YOGINI_OK = False

try:
    from app.core.chara_dasha import compute_chara_dasha as _compute_chara
    _CHARA_OK = True
except Exception:
    _CHARA_OK = False
    _compute_chara = None  # type: ignore

try:
    from app.core.kala_chakra_dasha import compute_kalachakra_dasha as _compute_kcd
    _KCD_OK = True
except Exception:
    _KCD_OK = False
    _compute_kcd = None  # type: ignore

# ───────────────────────── Ephemeris access ──────────────────────────────────
try:
    from app.core.ephemeris_adapter import ecliptic_longitudes  # type: ignore
    _EPH_OK = True
except Exception:
    _EPH_OK = False
    ecliptic_longitudes = None  # type: ignore

# ───────────────────────── Varga engine (delegation) ─────────────────────────
try:
    from app.core.varga_charts import (
        varga_position as _varga_position,
        compute_varga_chart as _compute_varga_chart,
        compute_many_vargas as _compute_many_vargas,
    )
    _VARGA_OK = True
except Exception:
    _VARGA_OK = False
    _varga_position = None  # type: ignore
    _compute_varga_chart = None  # type: ignore
    _compute_many_vargas = None  # type: ignore

# ───────────────────────── Yoga core (primary) ───────────────────────────────
_YOGA_CORE_OK = False
try:
    # prefer singular module name
    from app.core.yoga import (
        compute_yogas as _compute_yogas_core,
        list_registered_yogas as _yoga_list,
        enable_yogas as _yoga_enable,
        disable_yogas as _yoga_disable,
    )
    _YOGA_CORE_OK = True
except Exception:
    try:
        # support alternate filename (# app/core/yogas.py)
        from app.core.yogas import (
            compute_yogas as _compute_yogas_core,
            list_registered_yogas as _yoga_list,
            enable_yogas as _yoga_enable,
            disable_yogas as _yoga_disable,
        )
        _YOGA_CORE_OK = True
    except Exception:
        _YOGA_CORE_OK = False
        _compute_yogas_core = None  # type: ignore
        _yoga_list = None  # type: ignore
        _yoga_enable = None  # type: ignore
        _yoga_disable = None  # type: ignore

# ───────────────────────── NEW: Śaḍbala / Aṣṭakavarga engines ────────────────
_SHADBALA_OK = False
try:
    # expected API: compute_shadbala(payload: dict, **opts) -> dict
    from app.core.shadbala import compute_shadbala as _compute_shadbala  # type: ignore
    _SHADBALA_OK = True
except Exception:
    _SHADBALA_OK = False
    _compute_shadbala = None  # type: ignore

_ASHTAKAVARGA_OK = False
try:
    # expected API: compute_ashtakavarga(payload: dict, **opts) -> dict
    from app.core.ashtakavarga import compute_ashtakavarga as _compute_ashtakavarga  # type: ignore
    _ASHTAKAVARGA_OK = True
except Exception:
    _ASHTAKAVARGA_OK = False
    _compute_ashtakavarga = None  # type: ignore


__all__ = [
    # Dasha
    "DashaPeriod", "vimsottari_dasha", "predict_dasha_periods",
    "feature_dasha_lords_onehot",
    # Vargas (wrappers and full-detail helpers)
    "compute_vargas_for_point", "compute_vargas",
    "compute_varga_for_point_full", "compute_varga_full",
    # Yogas (new wired API + legacy basic detectors retained)
    "yoga_detect", "list_yoga_catalog",
    "house_index_for_longitude",
    "detect_panch_mahapurusha", "detect_gajakesari", "detect_chandra_mangal",
    "detect_parivartana", "detect_adhi", "detect_vesi_vasi_ubhayachari",
    "detect_viparita_rajayoga_basic", "detect_neecha_bhanga_basic",
    "detect_kemadruma_basic", "detect_yogas",
    # Features
    "feature_yoga_flags",
    # NEW — Gochar/Ingress/Stations wrappers + feature
    "gochar_drishti", "ingresses_rashi", "ingresses_nakshatra",
    "stations_retro_direct", "feature_drishti_proximity",
    # NEW — Śaḍbala & Aṣṭakavarga
    "shadbala", "ashtakavarga",
    # NEW — Compatibility shim explicitly exported for routes
    "compute_ashtakavarga",
]

# =============================================================================
# VIMŚOTTARĪ DAŚĀ — legacy helpers (kept for back-compat)
# =============================================================================

_VIM_ORDER = [
    "ketu","venus","sun","moon","mars","rahu","jupiter","saturn","mercury"
]
_VIM_YEARS = {
    "ketu":7,"venus":20,"sun":6,"moon":10,"mars":7,"rahu":18,"jupiter":16,"saturn":19,"mercury":17
}
_TOTAL_YEARS = 120.0
_MEAN_YEAR_DAYS = 365.2425
_NAK_WIDTH = 360.0 / 27.0

def _nirayana(lon_tropical: float, ayanamsa_deg: float) -> float:
    return norm360(lon_tropical - ayanamsa_deg)

def _nak_index(nirayana_lon: float) -> int:
    x = nirayana_lon / _NAK_WIDTH
    i = int(math.floor(x)) % 27
    return i

def _nak_lord(idx: int) -> str:
    return _VIM_ORDER[idx % 9]

def _cycle_from(lord: str) -> List[str]:
    i = _VIM_ORDER.index(lord)
    base = _VIM_ORDER
    return base[i:] + base[:i]

def _years_to_days(years: float) -> float:
    # mean tropical year for API stability
    return float(years) * _MEAN_YEAR_DAYS

@dataclass(slots=True, frozen=True)
class DashaPeriod:
    start_jd_tt: float
    end_jd_tt: float
    level: int  # 1..5
    lord: str   # normalized lower-case planet key
    parent_chain: Tuple[str, ...]
    meta: Dict[str, Any]

def vimsottari_dasha(
    *,
    birth_jd_tt: float,
    moon_lon_tropical_deg: float,
    ayanamsa_deg: float = 0.0,
    levels: int = 3,            # supports 1..5
    span_years: float = 120.0,
) -> List[DashaPeriod]:
    if levels < 1: levels = 1
    if levels > 5: levels = 5

    moon_nir = _nirayana(moon_lon_tropical_deg, ayanamsa_deg)
    idx = _nak_index(moon_nir)
    lord0 = _nak_lord(idx)
    pos_in_nak = moon_nir - (idx * _NAK_WIDTH)
    rem_frac = (_NAK_WIDTH - pos_in_nak) / _NAK_WIDTH
    rem_frac = 0.0 if rem_frac < 0.0 else (1.0 if rem_frac > 1.0 else rem_frac)

    cycle = _cycle_from(lord0)
    t = birth_jd_tt
    max_days = _years_to_days(span_years)

    periods_lvl1: List[DashaPeriod] = []
    for i, lord in enumerate(cycle):
        full_years = float(_VIM_YEARS[lord])
        frac = rem_frac if i == 0 else 1.0
        dur_days = _years_to_days(full_years * frac)
        start = t
        end = start + dur_days
        periods_lvl1.append(DashaPeriod(start, end, 1, lord, (lord,), {"years": full_years, "frac": frac}))
        t = end
        if (end - birth_jd_tt) >= (max_days - 1e-9):
            break

    def expand(parent: DashaPeriod, level: int) -> List[DashaPeriod]:
        subs = _cycle_from(parent.lord)
        out: List[DashaPeriod] = []
        total_days = parent.end_jd_tt - parent.start_jd_tt
        if total_days <= 0.0:
            return out
        t0 = parent.start_jd_tt
        frac_unit = 1.0 / _TOTAL_YEARS
        for lord in subs:
            part = _VIM_YEARS[lord] * frac_unit
            dur = total_days * part
            chain = parent.parent_chain + (lord,)
            t1 = t0 + dur
            out.append(DashaPeriod(t0, t1, level, lord, chain, {"frac": part}))
            t0 = t1
        if out:
            last = out[-1]
            if abs(last.end_jd_tt - parent.end_jd_tt) > 1e-12:
                out[-1] = DashaPeriod(last.start_jd_tt, parent.end_jd_tt, last.level, last.lord, last.parent_chain, last.meta)
        return out

    result = list(periods_lvl1)
    if levels >= 2:
        lvl2: List[DashaPeriod] = []
        for p in periods_lvl1: lvl2.extend(expand(p, 2))
        result.extend(lvl2)
        if levels >= 3:
            lvl3: List[DashaPeriod] = []
            for p in lvl2: lvl3.extend(expand(p, 3))
            result.extend(lvl3)
            if levels >= 4:
                lvl4: List[DashaPeriod] = []
                for p in lvl3: lvl4.extend(expand(p, 4))
                result.extend(lvl4)
                if levels >= 5:
                    lvl5: List[DashaPeriod] = []
                    for p in lvl4: lvl5.extend(expand(p, 5))
                    result.extend(lvl5)

    if span_years < _TOTAL_YEARS:
        cut = birth_jd_tt + max_days + 1e-9
        keep: List[DashaPeriod] = []
        for p in result:
            if p.start_jd_tt >= cut:
                break
            end = p.end_jd_tt if p.end_jd_tt <= cut else cut
            keep.append(DashaPeriod(p.start_jd_tt, end, p.level, p.lord, p.parent_chain, p.meta))
        result = keep

    result.sort(key=lambda d: (d.start_jd_tt, d.level))
    return result

# ───────────────────────────── Time helpers (TT↔UTC) ──────────────────────────

def _datetime_to_jd_tt(dt: datetime) -> float:
    try:
        if dt.tzinfo is None:
            t = TS.utc(dt.replace(tzinfo=timezone.utc))
        else:
            t = TS.utc(dt.astimezone(timezone.utc))
        return float(t.tt)
    except Exception:
        epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        sec = (dt - epoch).total_seconds()
        return 2440587.5 + (sec / 86400.0)

def _jd_tt_to_iso_utc(j_tt: float) -> str:
    try:
        dt_utc = TS.tt_jd(float(j_tt)).utc_datetime()
        return dt_utc.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
    except Exception:
        unix = (float(j_tt) - 2440587.5) * 86400.0
        return datetime.utcfromtimestamp(unix).replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")

def _civil_window_to_tt(date_from: str, date_to: str, tz_name: str) -> Tuple[Optional[float], Optional[float]]:
    try:
        ts0 = timescales_from_civil(str(date_from), "00:00:00", str(tz_name))
        ts1 = timescales_from_civil(str(date_to), "23:59:59", str(tz_name))
        return float(ts0["jd_tt"]), float(ts1["jd_tt"])
    except Exception:
        return None, None

# ───────────────────────────── Helpers for multi-system dasha ──────────────────

def _normalize_system(name: str) -> str:
    n = (name or "").strip().lower()
    aliases = {
        "vimshottari": "vimshottari",
        "vimsottari": "vimshottari",
        "vimśottarī": "vimshottari",
        "vimshottari_dasha": "vimshottari",
        "ashtottari": "ashtottari",
        "aṣṭottarī": "ashtottari",
        "ashtottarī": "ashtottari",
        "yogini": "yogini",
        "yoginī": "yogini",
        "chara": "chara",
        "jaimini": "chara",
        "chara_dasha": "chara",
        "jaimini chara": "chara",
        "kalachakra": "kalachakra",
        "kcd": "kalachakra",
        "kalacakra": "kalachakra",
        "kalachakra_dasha": "kalachakra",
    }
    return aliases.get(n, n)

def _natal_to_payload(natal_chart: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    if "jd_tt" in natal_chart:
        payload["jd_tt"] = float(natal_chart["jd_tt"])
    else:
        payload["date"] = str(natal_chart.get("date") or "")
        payload["time"] = str(natal_chart.get("time") or "00:00:00")
        payload["tz"] = str(natal_chart.get("place_tz") or natal_chart.get("timezone") or "UTC")
    ay = natal_chart.get("ayanamsa")
    if ay is not None:
        payload["ayanamsa"] = ay
    for k in ("latitude", "longitude", "asc_sidereal_deg", "asc_tropical_deg"):
        if k in natal_chart:
            payload[k] = natal_chart[k]
    return payload

def _flatten_nested(nested: List[Dict[str, Any]], max_level: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    def _label_of(node: Dict[str, Any]) -> str:
        lbl = node.get("lord")
        if lbl:
            return str(lbl)
        sn = node.get("sign_name")
        if sn:
            return str(sn)
        si = node.get("sign_index")
        if isinstance(si, int):
            return f"Sign-{si}"
        return ""
    def walk(node: Dict[str, Any], path: List[str]):
        lvl = int(node.get("level", 0))
        if lvl < 1 or lvl > max_level:
            return
        label = _label_of(node)
        a = float(node.get("start_jd_tt"))
        b = float(node.get("end_jd_tt"))
        new_path = path + [label] if label else path + [""]
        out.append({
            "level": lvl,
            "lord": label,
            "start_jd_tt": a,
            "end_jd_tt": b,
            "path": tuple(new_path),
        })
        kids = node.get("children") or node.get("c")
        if isinstance(kids, list):
            for k in kids:
                walk(k, new_path)
    for root in (nested or []):
        walk(root, [])
    out.sort(key=lambda r: (r["start_jd_tt"], r["level"]))
    return out

def _clip_rows(rows: List[Dict[str, Any]], jd_a: float, jd_b: float) -> List[Dict[str, Any]]:
    keep: List[Dict[str, Any]] = []
    A, B = (jd_a, jd_b) if jd_b >= jd_a else (jd_b, jd_a)
    for r in rows:
        a = float(r["start_jd_tt"]); b = float(r["end_jd_tt"])
        if b <= A or a >= B:
            continue
        if a < A: a = A
        if b > B: b = B
        path = list(r.get("path") or ())
        chain = path if path else ([r.get("lord")] if r.get("lord") is not None else [])
        keep.append({
            "start_jd_tt": a,
            "end_jd_tt": b,
            "start_date": _jd_tt_to_iso_utc(a),
            "end_date": _jd_tt_to_iso_utc(b),
            "level": int(r["level"]),
            "mahadasha_lord": chain[0] if chain else r.get("lord"),
            "chain": chain,
            "lord": r.get("lord"),
            "meta": {},
        })
    keep.sort(key=lambda d: (d["start_jd_tt"], d["level"]))
    return keep

# ───────────────────────────── Public predictive API ──────────────────────────

def predict_dasha_periods(
    *,
    natal_chart: Dict[str, Any],
    start_date: datetime,
    end_date: datetime,
    dasha_system: str = "vimshottari",
    include_antardasha: bool = True,
    levels: Optional[int] = None,
) -> Dict[str, Any]:
    system = _normalize_system(dasha_system)

    # Resolve birth TT
    if "jd_tt" in natal_chart:
        birth_jd_tt = float(natal_chart["jd_tt"])
    else:
        d = str(natal_chart.get("date") or "").strip()
        t = str(natal_chart.get("time") or "00:00:00").strip()
        tz = str(natal_chart.get("place_tz") or natal_chart.get("timezone") or "UTC").strip()
        ts = timescales_from_civil(d, t, tz)
        birth_jd_tt = float(ts["jd_tt"])

    # Window bounds in TT (order-agnostic)
    jd0_tt = _datetime_to_jd_tt(start_date)
    jd1_tt = _datetime_to_jd_tt(end_date)
    if jd1_tt < jd0_tt:
        jd0_tt, jd1_tt = jd1_tt, jd0_tt

    # Levels
    L = int(levels) if isinstance(levels, int) else (3 if include_antardasha else 1)
    if L < 1: L = 1
    elif L > 5: L = 5

    # ---------------- Vimśottarī (preferred engine) ----------------
    if system == "vimshottari":
        if _VIM_ENGINE_OK:
            ay = natal_chart.get("ayanamsa")
            if ay is None:
                ay = natal_chart.get("ayanamsa_deg", "lahiri")

            tree = generate_vimshottari_tree(
                birth_jd_tt=float(birth_jd_tt),
                ayanamsa=ay,
                levels=L,
                end_jd_tt=float(jd1_tt),
            )
            periods = tree.get("periods", [])
            rows: List[Dict[str, Any]] = []
            clip_a = jd0_tt; clip_b = jd1_tt

            for depth in range(1, L + 1):
                flat = flatten_periods(periods, level=depth)
                for r in flat:
                    a = float(r["start_jd_tt"]); b = float(r["end_jd_tt"])
                    if b <= clip_a or a >= clip_b:
                        continue
                    path = r.get("path")
                    lord = r["lord"]
                    chain = list(path) if path else [lord]
                    rows.append({
                        "start_jd_tt": a,
                        "end_jd_tt": b,
                        "start_date": _jd_tt_to_iso_utc(a),
                        "end_date": _jd_tt_to_iso_utc(b),
                        "level": depth,
                        "mahadasha_lord": chain[0],
                        "chain": chain,
                        "lord": lord,
                        "meta": {},
                    })

            rows.sort(key=lambda d: (d["start_jd_tt"], d["level"]))
            return {"ok": True, "periods": rows, "system": "vimshottari", "levels": L}

        # Fallback to legacy path (only if the central engine is unavailable)
        if not _EPH_OK or ecliptic_longitudes is None:
            return {"ok": False, "error": "ephemeris_unavailable"}

        moon_rows = (ecliptic_longitudes(float(birth_jd_tt), names=["Moon"]) or {}).get("results", [])
        if not moon_rows:
            return {"ok": False, "error": "moon_longitude_unavailable"}
        moon_lon_trop = float(moon_rows[0]["longitude"])

        ay_deg = float(natal_chart.get("ayanamsa_deg", 0.0))

        all_periods = vimsottari_dasha(
            birth_jd_tt=birth_jd_tt,
            moon_lon_tropical_deg=moon_lon_trop,
            ayanamsa_deg=ay_deg,
            levels=L,
            span_years=_TOTAL_YEARS,
        )

        out: List[Dict[str, Any]] = []
        a0 = jd0_tt; b0 = jd1_tt
        for p in all_periods:
            a = p.start_jd_tt; b = p.end_jd_tt
            if b <= a0 or a >= b0:
                continue
            out.append({
                "start_jd_tt": a,
                "end_jd_tt": b,
                "start_date": _jd_tt_to_iso_utc(a),
                "end_date": _jd_tt_to_iso_utc(b),
                "level": p.level,
                "mahadasha_lord": p.parent_chain[0] if p.parent_chain else p.lord,
                "chain": list(p.parent_chain),
                "lord": p.lord,
                "meta": p.meta,
            })

        return {"ok": True, "periods": out, "system": "vimshottari", "levels": L}

    # ---------------- Aṣṭottarī ----------------
    if system == "ashtottari":
        if not _ASHTO_OK:
            return {"ok": False, "error": "ashtottari_engine_unavailable"}
        base = _natal_to_payload(natal_chart)
        base.update({
            "levels": L,
            "year_days": 365.24219,
            "limit_jd_tt": float(jd1_tt),
        })
        sched = _compute_ashtottari(base)
        if not sched.get("ok", False):
            return {"ok": False, "error": sched.get("error", "ashtottari_failed")}

        nested = sched.get("nested") or []
        flat = _flatten_nested(nested, max_level=L)
        rows = _clip_rows(flat, jd0_tt, jd1_tt)
        return {"ok": True, "periods": rows, "system": "ashtottari", "levels": L}

    # ---------------- Yoginī ----------------
    if system == "yogini":
        if not _YOGINI_OK:
            return {"ok": False, "error": "yogini_engine_unavailable"}
        base = _natal_to_payload(natal_chart)
        base.update({
            "levels": L,
            "year_days": 365.24219,
            "limit_jd_tt": float(jd1_tt),
        })
        sched = _compute_yogini(base)
        if not sched.get("ok", False):
            return {"ok": False, "error": sched.get("error", "yogini_failed")}

        nested = sched.get("nested") or []
        flat = _flatten_nested(nested, max_level=L)
        rows = _clip_rows(flat, jd0_tt, jd1_tt)
        return {"ok": True, "periods": rows, "system": "yogini", "levels": L}

    # ---------------- Chara (Jaimini) ----------------
    if system == "chara":
        if not _CHARA_OK or _compute_chara is None:
            return {"ok": False, "error": "chara_engine_unavailable"}

        base = _natal_to_payload(natal_chart)
        base.update({
            "levels": L,
            "year_days": 365.24219,
            "limit_jd_tt": float(jd1_tt),
            "latitude": natal_chart.get("latitude"),
            "longitude": natal_chart.get("longitude"),
            "asc_sidereal_deg": natal_chart.get("asc_sidereal_deg"),
            "asc_tropical_deg": natal_chart.get("asc_tropical_deg"),
            "start_from": natal_chart.get("chara_start_from", "lagna"),
            "direction_mode": natal_chart.get("chara_direction_mode", "rashi_nature"),
            "include_rahu_in_karakas": bool(natal_chart.get("include_rahu_in_karakas", False)),
        })

        sched = _compute_chara(base)
        if not sched.get("ok", False):
            return {"ok": False, "error": sched.get("error", "chara_failed")}

        nested = sched.get("nested") or []
        flat = _flatten_nested(nested, max_level=L)
        rows = _clip_rows(flat, jd0_tt, jd1_tt)
        return {"ok": True, "periods": rows, "system": "chara", "levels": L}

    # ---------------- Kālachakra ----------------
    if system == "kalachakra":
        if not _KCD_OK or _compute_kcd is None:
            return {"ok": False, "error": "kalachakra_engine_unavailable"}

        base = _natal_to_payload(natal_chart)
        if "kcd_table" in natal_chart:
            base["kcd_table"] = natal_chart["kcd_table"]
        if "kcd_preset" in natal_chart:
            base["kcd_preset"] = natal_chart["kcd_preset"]
        if "use_demo_kcd_table" in natal_chart:
            base["use_demo_kcd_table"] = bool(natal_chart["use_demo_kcd_table"])
        if "override_start_sign_index" in natal_chart:
            base["override_start_sign_index"] = int(natal_chart["override_start_sign_index"])

        base.update({
            "levels": L,
            "year_days": 365.24219,
            "limit_jd_tt": float(jd1_tt),
        })

        if "planet_longitudes_sidereal" in natal_chart:
            base["planet_longitudes_sidereal"] = natal_chart["planet_longitudes_sidereal"]

        sched = _compute_kcd(base)
        if not sched.get("ok", False):
            return {"ok": False, "error": sched.get("error", "kalachakra_failed")}

        nested = sched.get("nested") or []
        flat = _flatten_nested(nested, max_level=L)
        rows = _clip_rows(flat, jd0_tt, jd1_tt)
        return {"ok": True, "periods": rows, "system": "kalachakra", "levels": L}

    # Unknown system
    return {"ok": False, "error": "unsupported_dasha"}

# =============================================================================
# (NEW) GOCHAR / INGRESSES / STATIONS — thin wrappers over app.core.vedic_gochar
# =============================================================================

def gochar_drishti(
    *,
    natal_chart: Dict[str, Any],
    date_from: str,
    date_to: str,
    transiting_bodies: Optional[List[str]] = None,
    natal_targets: Optional[List[str]] = None,
    zodiac_mode: Literal["tropical","sidereal"] = "sidereal",
    ayanamsa: Union[str, float] = "lahiri",
    frame: str = "ecliptic-of-date",
    include_nodes: bool = False,
    treat_nodes_like_saturn: bool = False,
    orb_deg: float = 12.0,
    orb_map: Optional[Dict[str, float]] = None,
    step_minutes: Union[str, float, int] = "auto",
    prebatch_refinement: bool = False,
) -> Dict[str, Any]:
    if not _GOCHAR_OK:
        return {"ok": False, "error": "gochar_engine_unavailable"}

    tz = str(natal_chart.get("place_tz") or natal_chart.get("tz") or "UTC")
    jd0, jd1 = _civil_window_to_tt(date_from, date_to, tz)

    res = _find_gochar_in_range(
        natal_chart=natal_chart,
        start_jd_tt=jd0 if jd0 is not None else None,
        end_jd_tt=jd1 if jd1 is not None else None,
        time_range=[date_from, date_to] if (jd0 is None or jd1 is None) else None,
        transiting_bodies=transiting_bodies,
        natal_targets=natal_targets,
        frame=frame,
        zodiac_mode=zodiac_mode,
        ayanamsa_deg=float(ayanamsa) if isinstance(ayanamsa, (int, float)) else None,
        orb_deg=orb_deg,
        orb_map=orb_map,
        include_nodes=include_nodes,
        treat_nodes_like_saturn=treat_nodes_like_saturn,
        step_minutes=step_minutes,
        prebatch_refinement=prebatch_refinement,
        place_tz=tz, tz=tz,
    )
    return res


def ingresses_rashi(
    *,
    date_from: str,
    date_to: str,
    movers: Optional[List[str]] = None,
    zodiac_mode: Literal["tropical","sidereal"] = "sidereal",
    ayanamsa: Union[str,float] = "lahiri",
    frame: str = "ecliptic-of-date",
    observer: Literal["geocentric","topocentric"] = "geocentric",
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    elevation_m: Optional[float] = None,
    tz_name: str = "UTC",
    step_minutes: Union[str, float, int] = "auto",
) -> Dict[str, Any]:
    if not _GOCHAR_OK:
        return {"ok": False, "error": "gochar_engine_unavailable"}

    jd0, jd1 = _civil_window_to_tt(date_from, date_to, tz_name)
    if jd0 is None or jd1 is None:
        return {"ok": False, "error": "invalid_time_window"}

    return _find_rashi_ingresses_in_range(
        start_jd_tt=float(jd0),
        end_jd_tt=float(jd1),
        movers=movers,
        frame=frame,
        zodiac_mode=zodiac_mode,
        ayanamsa_deg=float(ayanamsa) if isinstance(ayanamsa, (int,float)) else 0.0,
        step_minutes=step_minutes,
        topocentric=(observer == "topocentric"),
        latitude=latitude,
        longitude=longitude,
        elevation_m=elevation_m,
    )


def ingresses_nakshatra(
    *,
    date_from: str,
    date_to: str,
    movers: Optional[List[str]] = None,
    zodiac_mode: Literal["tropical","sidereal"] = "sidereal",
    ayanamsa: Union[str,float] = "lahiri",
    frame: str = "ecliptic-of-date",
    observer: Literal["geocentric","topocentric"] = "geocentric",
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    elevation_m: Optional[float] = None,
    tz_name: str = "UTC",
    step_minutes: Union[str, float, int] = "auto",
) -> Dict[str, Any]:
    """Nakṣatra-boundary crossings for movers within [date_from, date_to]."""
    if not _GOCHAR_OK:
        return {"ok": False, "error": "gochar_engine_unavailable"}

    jd0, jd1 = _civil_window_to_tt(date_from, date_to, tz_name)
    if jd0 is None or jd1 is None:
        return {"ok": False, "error": "invalid_time_window"}

    return _find_nakshatra_ingresses_in_range(
        start_jd_tt=float(jd0),
        end_jd_tt=float(jd1),
        movers=movers,
        frame=frame,
        zodiac_mode=zodiac_mode,
        ayanamsa_deg=float(ayanamsa) if isinstance(ayanamsa, (int,float)) else 0.0,
        step_minutes=step_minutes,
        topocentric=(observer == "topocentric"),
        latitude=latitude,
        longitude=longitude,
        elevation_m=elevation_m,
    )


def stations_retro_direct(
    *,
    date_from: str,
    date_to: str,
    movers: Optional[List[str]] = None,   # defaults inside engine: Merc..Sat
    zodiac_mode: Literal["tropical","sidereal"] = "sidereal",
    ayanamsa: Union[str,float] = "lahiri",
    frame: str = "ecliptic-of-date",
    observer: Literal["geocentric","topocentric"] = "geocentric",
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    elevation_m: Optional[float] = None,
    tz_name: str = "UTC",
    step_minutes: Union[str, float, int] = "auto",
) -> Dict[str, Any]:
    if not _GOCHAR_OK:
        return {"ok": False, "error": "gochar_engine_unavailable"}

    jd0, jd1 = _civil_window_to_tt(date_from, date_to, tz_name)
    if jd0 is None or jd1 is None:
        return {"ok": False, "error": "invalid_time_window"}

    return _find_stations_in_range(
        start_jd_tt=float(jd0),
        end_jd_tt=float(jd1),
        movers=movers,
        frame=frame,
        zodiac_mode=zodiac_mode,
        ayanamsa_deg=float(ayanamsa) if isinstance(ayanamsa, (int,float)) else 0.0,
        step_minutes=step_minutes,
        topocentric=(observer == "topocentric"),
        latitude=latitude,
        longitude=longitude,
        elevation_m=elevation_m,
    )


def feature_drishti_proximity(*, hits: List[Dict[str, Any]], cap_deg: float = 12.0) -> List[float]:
    if not _GOCHAR_OK:
        return []
    return _feature_drishti_proximity(hits=hits, cap_deg=cap_deg)

# =============================================================================
# (NEW) ŚAḌBALA — thin wrapper over app.core.shadbala
# =============================================================================

def shadbala(
    *,
    natal_chart: Dict[str, Any],
    zodiac_mode: Literal["tropical","sidereal"] = "sidereal",
    ayanamsa: Union[str, float] = "lahiri",
    house_system: str = "placidus",
    include_velocity: bool = True,
    prefer_houses_advanced: bool = True,
    observer: Literal["geocentric","topocentric"] = "geocentric",
) -> Dict[str, Any]:
    """
    Compute Śaḍbala. Delegates to app.core.shadbala.compute_shadbala if available.

    Inputs: expects civil date/time/tz and coordinates present in natal_chart.
    Returns: engine result dict; if engine absent, returns {"ok": False, "error": "..."}.
    """
    if not _SHADBALA_OK or _compute_shadbala is None:
        return {"ok": False, "error": "shadbala_engine_unavailable"}

    payload: Dict[str, Any] = {
        "date": natal_chart.get("date"),
        "time": natal_chart.get("time") or "12:00:00",
        "tz": natal_chart.get("place_tz") or natal_chart.get("tz") or "UTC",
        "latitude": natal_chart.get("latitude"),
        "longitude": natal_chart.get("longitude"),
        "elevation_m": natal_chart.get("elevation_m"),
        "ayanamsa": ayanamsa,
        "zodiac_mode": zodiac_mode,
        "house_system": house_system,
        "observer": observer,
        "include_velocity": bool(include_velocity),
        "prefer_houses_advanced": bool(prefer_houses_advanced),
    }

    try:
        try:
            res = _compute_shadbala(payload,
                                    ayanamsa=ayanamsa,
                                    zodiac_mode=zodiac_mode,
                                    house_system=house_system,
                                    include_velocity=include_velocity,
                                    prefer_houses_advanced=prefer_houses_advanced)
        except TypeError:
            res = _compute_shadbala(payload)
    except Exception as e:
        return {"ok": False, "error": f"shadbala_engine_error:{e}"}

    if isinstance(res, dict):
        res.setdefault("ok", True)
        return res
    return {"ok": True, "result": res}

# =============================================================================
# (NEW) AṢṬAKAVARGA — thin wrapper over app.core.ashtakavarga
# =============================================================================

def ashtakavarga(
    *,
    natal_chart: Dict[str, Any],
    zodiac_mode: Literal["tropical", "sidereal"] = "sidereal",
    ayanamsa: Union[str, float] = "lahiri",
    house_system: str = "placidus",
    include_sav: bool = True,
    include_bav: bool = True,
    spec_path: Optional[str] = None,             # Optional override for rules/spec
    # NEW: expose custom rules so callers can bypass the validator if needed
    ruleset: Optional[str] = None,               # e.g. "parashari-bphs" | "custom"
    ruleset_map: Optional[Dict[str, Dict[str, List[int]]]] = None,  # when ruleset="custom"
) -> Dict[str, Any]:
    """
    Compute Aṣṭakavarga (BAV per planet and/or SAV).
    Delegates to app.core.ashtakavarga.compute_ashtakavarga if available.

    Notes:
    - Forwards civil date/time/tz and coordinates from `natal_chart`.
    - Passes through `angles` (asc/mc) if present to help bind Lagna without recomputation.
    - Supports optional custom rules via `ruleset` and `ruleset_map`.
    """
    if not _ASHTAKAVARGA_OK or _compute_ashtakavarga is None:
        return {"ok": False, "error": "ashtakavarga_engine_unavailable"}

    # Optional angles pass-through
    angles: Optional[Dict[str, float]] = None
    if isinstance(natal_chart.get("angles"), dict):
        ang_in = natal_chart["angles"]
        tmp: Dict[str, float] = {}
        try:
            if ang_in.get("asc") is not None:
                tmp["asc"] = float(ang_in.get("asc"))
        except Exception:
            pass
        try:
            if ang_in.get("ASC") is not None:
                tmp["asc"] = float(ang_in.get("ASC"))
        except Exception:
            pass
        try:
            if ang_in.get("Ascendant") is not None:
                tmp["asc"] = float(ang_in.get("Ascendant"))
        except Exception:
            pass
        try:
            if ang_in.get("mc") is not None:
                tmp["mc"] = float(ang_in.get("mc"))
        except Exception:
            pass
        try:
            if ang_in.get("MC") is not None:
                tmp["mc"] = float(ang_in.get("MC"))
        except Exception:
            pass
        try:
            if ang_in.get("Midheaven") is not None:
                tmp["mc"] = float(ang_in.get("Midheaven"))
        except Exception:
            pass
        angles = tmp or None

    payload: Dict[str, Any] = {
        "date": natal_chart.get("date"),
        "time": natal_chart.get("time") or "12:00:00",
        "tz": natal_chart.get("place_tz") or natal_chart.get("tz") or "UTC",
        "latitude": natal_chart.get("latitude"),
        "longitude": natal_chart.get("longitude"),
        "elevation_m": natal_chart.get("elevation_m"),
        "ayanamsa": ayanamsa,
        "zodiac_mode": zodiac_mode,
        "house_system": house_system,
    }
    if angles:
        payload["angles"] = angles
    if spec_path:
        payload["spec_path"] = spec_path
    if ruleset:
        payload["ruleset"] = ruleset
    if ruleset_map:
        payload["ruleset_map"] = ruleset_map

    try:
        try:
            # Many implementations accept extra kwargs and ignore unknowns.
            res = _compute_ashtakavarga(
                payload,
                ayanamsa=ayanamsa,
                zodiac_mode=zodiac_mode,
                house_system=house_system,
                include_sav=include_sav,
                include_bav=include_bav,
                spec_path=spec_path,
                ruleset=ruleset,
                ruleset_map=ruleset_map,
            )
        except TypeError:
            # Fallback to the strict single-arg signature.
            res = _compute_ashtakavarga(payload)
    except Exception as e:
        return {"ok": False, "error": f"ashtakavarga_engine_error:{e}"}

    if isinstance(res, dict):
        res.setdefault("ok", True)
        return res
    return {"ok": True, "result": res}


# ── NEW: Compatibility shim for routes expecting `compute_ashtakavarga(payload)` ──
def compute_ashtakavarga(payload: Dict[str, Any] = None, **opts) -> Dict[str, Any]:
    """
    Compatibility wrapper so routes can import:
        from app.core.vedic_predictive import compute_ashtakavarga
    Accepts a single payload dict (plus optional kwargs), forwards to
    app.core.ashtakavarga.compute_ashtakavarga with a permissive signature.
    """
    if not _ASHTAKAVARGA_OK or _compute_ashtakavarga is None:
        return {"ok": False, "error": "ashtakavarga_engine_unavailable"}

    merged: Dict[str, Any] = dict(payload or {})
    merged.update(opts or {})

    try:
        try:
            # Some cores accept a single dict; others prefer kwargs.
            res = _compute_ashtakavarga(merged)
        except TypeError:
            res = _compute_ashtakavarga(**merged)  # type: ignore[misc]
    except Exception as e:
        return {"ok": False, "error": f"ashtakavarga_engine_error:{e}"}

    if isinstance(res, dict):
        res.setdefault("ok", True)
        return res
    return {"ok": True, "result": res}

# =============================================================================
# VARGAS (DIVISIONAL CHARTS) — thin wrappers over app.core.varga_charts
# =============================================================================

def compute_vargas_for_point(
    *,
    lon_deg: float,
    zodiac_mode: Literal["tropical","sidereal"] = "sidereal",
    ayanamsa: Any = "lahiri",
    include: Iterable[str] = ("D1","D2","D3","D9","D10","D12"),
) -> Dict[str, int]:
    if not _VARGA_OK or _varga_position is None:
        return {}
    out: Dict[str, int] = {}
    for code in include:
        try:
            res = _varga_position(float(lon_deg), str(code), zodiac_mode=zodiac_mode, ayanamsa=ayanamsa)
            idx = int(res.get("varga_rasi_index"))
            out[str(res.get("varga", code))] = idx
        except Exception:
            continue
    return out


def compute_vargas(
    *,
    points_deg: Dict[str, float],
    zodiac_mode: Literal["tropical","sidereal"] = "sidereal",
    ayanamsa: Any = "lahiri",
    include: Iterable[str] = ("D1","D2","D3","D9","D10","D12"),
) -> Dict[str, Dict[str, int]]:
    if not _VARGA_OK or _compute_many_vargas is None:
        return {}
    varga_list = [str(v) for v in include]
    try:
        full = _compute_many_vargas(points_deg, varga_list, zodiac_mode=zodiac_mode, ayanamsa=ayanamsa)
    except Exception:
        return {}
    out: Dict[str, Dict[str, int]] = {name: {} for name in points_deg.keys()}
    for varga_key, per_point in (full or {}).items():
        if not isinstance(per_point, dict):
            continue
        for name, placed in per_point.items():
            try:
                idx = int(placed.get("varga_rasi_index"))
                out.setdefault(name, {})[varga_key] = idx
            except Exception:
                continue
    return out


def compute_varga_for_point_full(
    *,
    lon_deg: float,
    varga: str,
    zodiac_mode: Literal["tropical","sidereal"] = "sidereal",
    ayanamsa: Any = "lahiri",
) -> Dict[str, Any]:
    if not _VARGA_OK or _varga_position is None:
        return {"error": "varga_engine_unavailable"}
    return _varga_position(float(lon_deg), str(varga), zodiac_mode=zodiac_mode, ayanamsa=ayanamsa)


def compute_varga_full(
    *,
    points_deg: Dict[str, float],
    vargas: Iterable[str],
    zodiac_mode: Literal["tropical","sidereal"] = "sidereal",
    ayanamsa: Any = "lahiri",
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    if not _VARGA_OK or _compute_many_vargas is None:
        return {"error": "varga_engine_unavailable"}
    return _compute_many_vargas(points_deg, [str(v) for v in vargas], zodiac_mode=zodiac_mode, ayanamsa=ayanamsa)

# =============================================================================
# YOGAS — primary path via app.core.yoga, with legacy-basic fallback
# =============================================================================

def list_yoga_catalog() -> Dict[str, Any]:
    if not _YOGA_CORE_OK or _yoga_list is None:
        return {"ok": False, "error": "yoga_core_unavailable"}
    try:
        return {"ok": True, "catalog": _yoga_list()}
    except Exception as e:
        return {"ok": False, "error": f"yoga_catalog_error:{e}"}

_WS_RE = re.compile(r"\s+")

def _canon_name(s: str) -> str:
    t = (s or "").strip().lower()
    t = _WS_RE.sub(" ", t)
    t = t.replace("-", " ").replace("/", " ")
    t = "".join(ch for ch in t if ch.isalnum() or ch in (" ", "_"))
    t = t.replace(" ", "_")
    while "__" in t:
        t = t.replace("__", "_")
    return t.strip("_")

def _canon_set(seq: Iterable[str]) -> Set[str]:
    return { _canon_name(x) for x in seq if str(x).strip() }

def yoga_detect(norm: Dict[str, Any]) -> Dict[str, Any]:
    include_raw = list(norm.get("include") or ())
    include_names = _canon_set(include_raw)

    # ── Primary path: registry/core ──
    if _YOGA_CORE_OK and _compute_yogas_core is not None:
        payload = {
            "jd_tt": norm.get("jd_tt"),
            "jd_ut1": norm.get("jd_ut1"),
            "date": norm.get("date"),
            "time": norm.get("time"),
            "tz": norm.get("tz") or norm.get("place_tz") or norm.get("tz_name") or "UTC",
            "latitude": norm.get("latitude"),
            "longitude": norm.get("longitude"),
        }
        try:
            res = _compute_yogas_core(
                payload,
                ayanamsa=norm.get("ayanamsa", "lahiri"),
                house_system=str(norm.get("house_system") or "placidus"),
                enable_catalog_tags=tuple(norm.get("enable_catalog_tags") or ()),
                disable_catalog_tags=tuple(norm.get("disable_catalog_tags") or ()),
            )
        except Exception as e:
            return {"ok": False, "error": f"yoga_engine_error:{e}"}

        yogas_full = list(res.get("yogas") or [])
        if include_names:
            filtered: List[Dict[str, Any]] = []
            for y in yogas_full:
                nm = str(y.get("name") or "")
                if _canon_name(nm) in include_names:
                    filtered.append(y)
            yogas = filtered
        else:
            yogas = yogas_full

        present_list = [
            {"yoga": y.get("name"), "score": y.get("score"), "tags": y.get("tags", [])}
            for y in yogas if bool(y.get("present"))
        ]

        ok = bool(res.get("ok", False))
        out = {
            "ok": ok,
            "yogas": yogas,
            "context": res.get("context"),
            "warnings": list(res.get("warnings", [])),
            "present": present_list,
        }
        if not ok:
            out["error"] = res.get("error", "yoga_detect_failed")
        return out

    # ── Fallback: legacy-basic detectors; requires precomputed points + cusps ──
    pts = norm.get("points_deg") or {}
    cusps = norm.get("cusps_deg") or []
    if pts and isinstance(cusps, list) and len(cusps) == 12:
        default_inc = (
            "panch_mahapurusha","gajakesari","chandra_mangal","parivartana",
            "adhi","vesi_vasi_ubhayachari","viparita_rajayoga_basic",
            "neecha_bhanga_basic","kemadruma_basic"
        )
        inc = tuple(include_raw) if include_names else default_inc
        legacy_hits = detect_yogas(
            points_deg={k.lower(): v for k, v in pts.items()},
            cusps_deg=cusps,
            include=inc,
        )
        return {
            "ok": True,
            "yogas": legacy_hits,
            "context": {"mode": "legacy_basic"},
            "warnings": ["yoga_core_unavailable_fallback_basic"],
            "present": [{"yoga": y.get("yoga")} for y in legacy_hits],
        }

    return {"ok": False, "error": "yoga_engine_unavailable_or_insufficient_inputs"}

# ───────────────────────── Legacy-basic detectors (unchanged) ─────────────────

EXALT_SIGN = {"sun":0,"moon":1,"mars":9,"mercury":5,"jupiter":3,"venus":11,"saturn":6}
OWN_SIGNS = {
    "sun":[4],"moon":[3],"mars":[0,7],"mercury":[2,5],
    "jupiter":[8,11],"venus":[1,6],"saturn":[9,10]
}
DEBIL_SIGN = {"sun":6,"moon":7,"mars":3,"mercury":11,"jupiter":9,"venus":5,"saturn":0}

_OWNER_BY_SIGN: Dict[int, str] = {}
for _pl, _signs in OWN_SIGNS.items():
    for _s in _signs:
        _OWNER_BY_SIGN[_s] = _pl

def house_index_for_longitude(cusps_deg: List[float], lon_deg: float) -> int:
    if len(cusps_deg) != 12:
        raise ValueError("cusps_deg must be 12 values")
    c0 = [norm360(x) for x in cusps_deg]
    lam = norm360(lon_deg)
    for i in range(12):
        start = c0[i]
        end = norm360(c0[(i + 1) % 12])
        span = norm360(end - start)
        delta = norm360(lam - start)
        if span == 0.0 or delta < span:
            return i + 1
    return 12

def is_kendra(h: int) -> bool:
    return h in (1,4,7,10)

def in_own_or_exaltation(planet: str, sign_idx: int) -> bool:
    p = planet.lower()
    if EXALT_SIGN.get(p, -1) == sign_idx:
        return True
    return sign_idx in OWN_SIGNS.get(p, [])

def detect_panch_mahapurusha(points_deg: Dict[str, float], cusps_deg: List[float]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    add = out.append
    for p, name in (("mars","Ruchaka"), ("mercury","Bhadra"), ("jupiter","Hamsa"),
                    ("venus","Malavya"), ("saturn","Shasha")):
        lon = points_deg.get(p)
        if lon is None:
            continue
        s = sign_index(lon); h = house_index_for_longitude(cusps_deg, lon)
        if is_kendra(h) and in_own_or_exaltation(p, s):
            add({"yoga": name, "planet": p, "house": h, "sign_index": s})
    return out

def detect_gajakesari(points_deg: Dict[str, float], cusps_deg: List[float]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    moon = points_deg.get("moon"); jup = points_deg.get("jupiter")
    if moon is None or jup is None:
        return out
    fm = house_index_for_longitude(cusps_deg, moon)
    fj = house_index_for_longitude(cusps_deg, jup)
    diff = ((fj - fm) % 12) or 12
    if diff in (1,4,7,10):
        out.append({"yoga": "Gajakesari", "from": "moon", "to": "jupiter", "offset_houses": diff})
    return out

def detect_chandra_mangal(points_deg: Dict[str, float], max_orb_deg: float = 8.0) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    moon = points_deg.get("moon"); mars = points_deg.get("mars")
    if moon is None or mars is None:
        return out
    sep = abs(angdiff(moon, mars))
    if sep <= max_orb_deg:
        out.append({"yoga": "Chandra-Mangal", "orb_deg": float(sep)})
    return out

def detect_parivartana(points_deg: Dict[str, float]) -> List[Dict[str, Any]]:
    loc_owner: Dict[str, str] = {}
    setitem = loc_owner.__setitem__
    for pl, lon in points_deg.items():
        setitem(pl, _OWNER_BY_SIGN.get(sign_index(lon), ""))
    out: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str]] = set()
    add = out.append
    for a, lord_b in loc_owner.items():
        if not lord_b or lord_b == a:
            continue
        if (a, lord_b) in seen or (lord_b, a) in seen:
            continue
        if loc_owner.get(lord_b) == a:
            add({"yoga": "Parivartana", "pair": (a, lord_b)})
            seen.add((a, lord_b))
    return out

def detect_adhi(points_deg: Dict[str, float], cusps_deg: List[float]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    moon = points_deg.get("moon")
    if moon is None:
        return out
    moon_h = house_index_for_longitude(cusps_deg, moon)
    houses = {pl: house_index_for_longitude(cusps_deg, lon) for pl, lon in points_deg.items()}
    present: List[str] = []
    for pl in ("jupiter","venus","mercury"):
        h = houses.get(pl)
        if h is None:
            continue
        diff = ((h - moon_h) % 12) or 12
        if diff in (6,7,8):
            present.append(pl)
    if present:
        out.append({"yoga": "Adhi", "planets": sorted(present), "from_moon_house": moon_h})
    return out

def detect_vesi_vasi_ubhayachari(points_deg: Dict[str, float], sun_orb_block_deg: float = 12.0) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    sun = points_deg.get("sun")
    if sun is None:
        return out
    s = sign_index(sun)
    two = (s + 1) % 12
    twelve = (s - 1) % 12
    vesi: List[str] = []
    vasi: List[str] = []
    for pl, lon in points_deg.items():
        if pl == "sun":
            continue
        if abs(angdiff(sun, lon)) < sun_orb_block_deg:
            continue
        sp = sign_index(lon)
        if sp == two:
            vesi.append(pl)
        elif sp == twelve:
            vasi.append(pl)
    if vesi and vasi:
        out.append({"yoga": "Ubhayachari", "planets_2nd": sorted(vesi), "planets_12th": sorted(vasi)})
    elif vesi:
        out.append({"yoga": "Vesi", "planets": sorted(vesi)})
    elif vasi:
        out.append({"yoga": "Vasi", "planets": sorted(vasi)})
    return out

def detect_viparita_rajayoga_basic(points_deg: Dict[str, float], cusps_deg: List[float]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    houses = {pl: house_index_for_longitude(cusps_deg, lon) for pl, lon in points_deg.items()}
    add = out.append
    for pl, lon in points_deg.items():
        lord = _OWNER_BY_SIGN.get(sign_index(lon))
        if not lord:
            continue
        h = houses.get(pl)
        if h in (6,8,12):
            add({"yoga": "Viparita-Rajayoga (basic)", "planet": pl, "house": h, "owner": lord})
    return out

def detect_neecha_bhanga_basic(points_deg: Dict[str, float]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    add = out.append
    for pl, lon in points_deg.items():
        if DEBIL_SIGN.get(pl, -1) == sign_index(lon):
            add({"yoga": "Neecha (debilitation)", "planet": pl, "sign_index": sign_index(lon)})
    return out

def detect_kemadruma_basic(points_deg: Dict[str, float]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    moon = points_deg.get("moon")
    if moon is None:
        return out
    sm = sign_index(moon)
    second = (sm + 1) % 12
    twelfth = (sm - 1) % 12
    planets = [pl for pl in points_deg.keys() if pl != "sun"]
    ok_second = any(sign_index(points_deg[p]) == second for p in planets)
    ok_twelfth = any(sign_index(points_deg[p]) == twelfth for p in planets)
    if not ok_second and not ok_twelfth:
        out.append({"yoga": "Kemadruma (basic)"})
    return out

def detect_yogas(
    *,
    points_deg: Dict[str, float],
    cusps_deg: List[float],
    include: Iterable[str] = (
        "panch_mahapurusha","gajakesari","chandra_mangal","parivartana",
        "adhi","vesi_vasi_ubhayachari","viparita_rajayoga_basic","neecha_bhanga_basic","kemadruma_basic"
    ),
    orbs: Optional[Dict[str, float]] = None
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    inc = set(include)
    if "panch_mahapurusha" in inc: out.extend(detect_panch_mahapurusha(points_deg, cusps_deg))
    if "gajakesari" in inc: out.extend(detect_gajakesari(points_deg, cusps_deg))
    if "chandra_mangal" in inc:
        max_orb = (orbs or {}).get("chandra_mangal", 8.0)
        out.extend(detect_chandra_mangal(points_deg, max_orb_deg=max_orb))
    if "parivartana" in inc: out.extend(detect_parivartana(points_deg))
    if "adhi" in inc: out.extend(detect_adhi(points_deg, cusps_deg))
    if "vesi_vasi_ubhayachari" in inc: out.extend(detect_vesi_vasi_ubhayachari(points_deg))
    if "viparita_rajayoga_basic" in inc: out.extend(detect_viparita_rajayoga_basic(points_deg, cusps_deg))
    if "neecha_bhanga_basic" in inc: out.extend(detect_neecha_bhanga_basic(points_deg))
    if "kemadruma_basic" in inc: out.extend(detect_kemadruma_basic(points_deg))
    out.sort(key=lambda x: (x.get("yoga",""), x.get("planet",""), tuple(x.get("pair",())), tuple(x.get("planets",()))))
    return out

# =============================================================================
# FEATURE BUILDERS
# =============================================================================

def feature_dasha_lords_onehot(periods: List[Dict[str, Any]], *, levels: int = 2) -> List[List[int]]:
    L = levels if 1 <= int(levels) <= 5 else 2
    idx = {p:i for i,p in enumerate(_VIM_ORDER)}
    out: List[List[int]] = []
    for p in periods:
        chain = [str(x).lower() for x in (p.get("chain") or [])]
        n = 9 * L
        row = [0] * n
        for lev in range(min(L, len(chain))):
            j = idx.get(chain[lev])
            if j is not None:
                row[lev*9 + j] = 1
        out.append(row)
    return out

def feature_yoga_flags(yogas: List[Dict[str, Any]], *, include: Optional[Iterable[str]] = None) -> Dict[str, int]:
    inc = set(include) if include else None
    flags: Dict[str, int] = {}
    for y in yogas:
        name = str(y.get("yoga") or y.get("name") or "").strip()
        if not name:
            continue
        if inc and name not in inc:
            continue
        present = y.get("present")
        if present is None:
            flags[name] = 1
        else:
            flags[name] = 1 if bool(present) else 0
    return flags

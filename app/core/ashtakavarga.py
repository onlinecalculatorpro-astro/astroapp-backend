# app/core/ashtakavarga.py
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Aṣṭakavarga — research-grade, rule-driven engine (BAV + LAV + SAV)

What this module provides
-------------------------
- Pure, deterministic bindu arithmetic per BPHS-style tables (no orbs/heuristics).
- Pluggable rules via JSON/dict (no hardcoded vendor tables).
- Works directly from your chart pipeline:
    compute_ashtakavarga(payload[, spec]) -> dict
  where `payload` is the same dict you pass to astronomy.compute_chart.

Public API (stable)
-------------------
    AshtakavargaSpec
    load_spec(spec_dict_or_path: dict|str|None) -> AshtakavargaSpec
    rashi_index(lon_deg: float) -> int  # 1..12

    compute_bav(longitudes_sidereal: dict[str,float], spec) -> dict[str, list[int]]
    compute_lagna_av(lagna_lon_sidereal: float, spec) -> list[int]
    compute_sav(bav: dict[str, list[int]], lav: list[int], *, include_nodes: bool = False) -> list[int]

    from_sidereal_chart(chart: dict, spec) -> dict  # if you already have a sidereal chart
    quick_compute(lon_sid_map: dict[str,float], lagna_sid_deg: float, spec) -> dict

    compute_ashtakavarga(payload: dict, spec: AshtakavargaSpec|None = None) -> dict
        # High-level wrapper:
        #   1) astronomy.compute_chart(payload)
        #   2) extracts/derives SIDEREAL longitudes + Lagna
        #   3) builds BAV/LAV/SAV
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Tuple, Union, Any
import math
import os
import json

# ──────────────────────────── required core wiring ───────────────────────────
try:
    # single source of truth for positions/angles/mode/ayanamsa
    from app.core.astronomy import compute_chart as _compute_chart
except Exception as e:
    raise RuntimeError(f"ashtakavarga: astronomy.compute_chart import failed: {e}")

# optional: ayanāṁśa name → degrees resolver (used when chart is tropical)
try:
    from app.core.ayanamsa import get_ayanamsa_deg as _get_ayanamsa_deg  # type: ignore
except Exception:
    _get_ayanamsa_deg = None  # type: ignore

# ───────────────────────── helpers: angles & names ───────────────────────────

def _norm360(x: float) -> float:
    r = math.fmod(float(x), 360.0)
    return r + 360.0 if r < 0.0 else r

def rashi_index(lon_deg: float) -> int:
    """Return 1..12 for 0° Aries..330° Pisces (expects SIDEREAL longitude)."""
    return int(math.floor(_norm360(lon_deg) / 30.0)) + 1

_CANON_NAMES = {
    "sun":"Sun","surya":"Sun",
    "moon":"Moon","chandra":"Moon",
    "mars":"Mars","mangal":"Mars","kuja":"Mars",
    "mercury":"Mercury","budha":"Mercury",
    "jupiter":"Jupiter","guru":"Jupiter","brihaspati":"Jupiter",
    "venus":"Venus","shukra":"Venus",
    "saturn":"Saturn","shani":"Saturn",
    "rahu":"Rahu","north node":"Rahu","northnode":"Rahu",
    "ketu":"Ketu","south node":"Ketu","southnode":"Ketu",
    "asc":"Lagna","ascendant":"Lagna","lagna":"Lagna",
}

def _canon(name: str) -> str:
    return _CANON_NAMES.get(str(name).strip().lower(), str(name))

_PLANETS_7 = ("Sun","Moon","Mars","Mercury","Jupiter","Venus","Saturn")

# ───────────────────────── rules spec (pluggable) ────────────────────────────

@dataclass(frozen=True)
class AshtakavargaSpec:
    """
    Rule container:

    bav_offsets:
        dict[target][contributor] = tuple[int,...]
        - target, contributor ∈ {"Sun","Moon","Mars","Mercury","Jupiter","Venus","Saturn"}
        - Offsets are forward sign steps 1..12 from contributor’s sign.

    lav_offsets:
        tuple[int,...] of forward offsets from Lagna to mark 1s in LAV.

    expected_row_totals (optional):
        dict[target] = int total bindu count for that BAV row.

    expected_lav_total (optional): int total for LAV row.
    expected_sav_total (optional): int overall total of SAV across 12 signs.
    """
    bav_offsets: Dict[str, Dict[str, Tuple[int, ...]]]
    lav_offsets: Tuple[int, ...]
    expected_row_totals: Optional[Dict[str, int]] = None
    expected_lav_total: Optional[int] = None
    expected_sav_total: Optional[int] = None

# cache the default spec if loaded from env
_default_spec_cache: Optional[AshtakavargaSpec] = None

def load_spec(spec: Optional[Union[str, Mapping[str, object]]] = None) -> AshtakavargaSpec:
    """
    Load AshtakavargaSpec from:
      - a dict-like object (already parsed), or
      - a JSON file path, or
      - if None: look for env OCP_ASHTAKAVARGA_SPEC (JSON path).
    """
    global _default_spec_cache

    data: Mapping[str, object]
    if spec is None:
        if _default_spec_cache is not None:
            return _default_spec_cache
        path = os.getenv("OCP_ASHTAKAVARGA_SPEC", "").strip()
        if not path:
            raise RuntimeError("Ashtakavarga rules not provided. Set OCP_ASHTAKAVARGA_SPEC to a JSON file or pass a dict to load_spec().")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif isinstance(spec, str):
        with open(spec, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = spec

    def _tupled(x: Iterable[int]) -> Tuple[int, ...]:
        return tuple(int(v) for v in x)

    raw_bav = data.get("bav_offsets", {})  # type: ignore[assignment]
    if not isinstance(raw_bav, Mapping):
        raise ValueError("bav_offsets missing or not a mapping")
    bav: Dict[str, Dict[str, Tuple[int, ...]]] = {}
    for tgt, contribs in raw_bav.items():  # type: ignore[assignment]
        if not isinstance(contribs, Mapping):
            raise ValueError(f"bav_offsets[{tgt!r}] must be a mapping")
        row: Dict[str, Tuple[int, ...]] = {}
        for c, offsets in contribs.items():  # type: ignore[assignment]
            if not isinstance(offsets, (list, tuple)):
                raise ValueError(f"bav_offsets[{tgt!r}][{c!r}] must be a list/tuple of offsets")
            offs = []
            for k in offsets:
                kk = int(k)
                if kk < 0:
                    kk %= 12
                if kk == 0:
                    kk = 12  # 0 → contributor’s own sign becomes +12 forward
                offs.append(kk)
            row[_canon(c)] = _tupled(offs)
        bav[_canon(tgt)] = row

    raw_lav = data.get("lav_offsets", [])
    if not isinstance(raw_lav, (list, tuple)):
        raise ValueError("lav_offsets must be list/tuple")
    lav = []
    for k in raw_lav:
        kk = int(k)
        if kk < 0:
            kk %= 12
        if kk == 0:
            kk = 12
        lav.append(kk)

    exp_rows = data.get("expected_row_totals")
    exp_lav  = data.get("expected_lav_total")
    exp_sav  = data.get("expected_sav_total")

    spec_obj = AshtakavargaSpec(
        bav_offsets=bav,
        lav_offsets=tuple(lav),
        expected_row_totals={_canon(k): int(v) for k, v in (exp_rows or {}).items()} if isinstance(exp_rows, Mapping) else None,
        expected_lav_total=(int(exp_lav) if isinstance(exp_lav, (int, float)) else None),
        expected_sav_total=(int(exp_sav) if isinstance(exp_sav, (int, float)) else None),
    )
    if spec is None:
        _default_spec_cache = spec_obj
    return spec_obj

# ───────────────────────── core arithmetic (0/1 rows) ────────────────────────

def _forward_sign(base: int, step: int) -> int:
    """Return sign index (1..12) after stepping 'step' forward from base (1..12)."""
    b = int(base); s = int(step) % 12
    return ((b - 1 + s) % 12) + 1

def _bindu_row_from_offsets(contributor_sign: int, offsets: Tuple[int, ...]) -> List[int]:
    """Return a 12-bin 0/1 row with 1s at contributor_sign + offsets."""
    bins = [0] * 12
    for off in offsets:
        j = _forward_sign(contributor_sign, off)
        bins[j - 1] = 1
    return bins

def _sum_rows(rows: Iterable[List[int]]) -> List[int]:
    acc = [0] * 12
    for r in rows:
        for i in range(12):
            acc[i] += int(r[i])
    return acc

# ───────────────────────── public: BAV / LAV / SAV ───────────────────────────

def compute_bav(
    longitudes_sidereal: Mapping[str, float],
    spec: AshtakavargaSpec
) -> Dict[str, List[int]]:
    """
    Build BAV matrices for Sun..Saturn.

    longitudes_sidereal: {'Sun': lon, ..., 'Saturn': lon}
      → longitudes MUST be sidereal (ayanāṁśa already subtracted).

    Returns: {planet: [12 ints]} in Aries..Pisces order.
    """
    # contributor signs
    signs: Dict[str, int] = {}
    for nm, lon in longitudes_sidereal.items():
        nm2 = _canon(nm)
        if nm2 in _PLANETS_7:
            try:
                signs[nm2] = rashi_index(float(lon))
            except Exception:
                pass

    out: Dict[str, List[int]] = {}
    for target in _PLANETS_7:
        rule_row = spec.bav_offsets.get(target)
        if not rule_row:
            raise RuntimeError(f"BAV rules missing for target planet {target}")
        rows: List[List[int]] = []
        for contrib in _PLANETS_7:
            offs = rule_row.get(contrib)
            if not offs:
                raise RuntimeError(f"BAV rules missing for target={target}, contributor={contrib}")
            csign = signs.get(contrib)
            if csign is None:
                raise RuntimeError(f"Missing sidereal longitude for contributor {contrib}")
            rows.append(_bindu_row_from_offsets(csign, offs))
        out[target] = _sum_rows(rows)

        # optional row-total integrity
        if spec.expected_row_totals and target in spec.expected_row_totals:
            exp = int(spec.expected_row_totals[target])
            got = sum(out[target])
            if got != exp:
                raise AssertionError(f"Ashtakavarga integrity: BAV[{target}] total={got} != expected {exp}")

    return out

def compute_lagna_av(
    lagna_lon_sidereal: float,
    spec: AshtakavargaSpec
) -> List[int]:
    """Compute LAV (Lagna Aṣṭakavarga) from Lagna and lav_offsets."""
    lagna_sign = rashi_index(float(lagna_lon_sidereal))
    return _bindu_row_from_offsets(lagna_sign, spec.lav_offsets)

def compute_sav(
    bav: Mapping[str, List[int]],
    lav: List[int],
    *,
    include_nodes: bool = False   # kept for API symmetry; classic model ignores nodes
) -> List[int]:
    """Sum seven BAV rows + LAV → 12-bin SAV array."""
    rows = [bav[p] for p in _PLANETS_7 if p in bav]
    rows.append(lav)
    return _sum_rows(rows)

# ───────────────────────── adapters: extract from chart ──────────────────────

def _extract_sidereal_longitudes(chart: Mapping[str, Any]) -> Dict[str, float]:
    """
    Tries in order:
      1) chart['bodies_sidereal'] = [{'name','longitude_sidereal_deg'}, ...]
      2) chart['bodies'] with 'longitude_deg' plus chart/meta ayanāṁśa to siderealize
      3) chart['bodies'] with already-sidereal 'lon' (fallback)
    """
    out: Dict[str, float] = {}
    # 1) explicit sidereal
    bodies_s = chart.get("bodies_sidereal")
    if isinstance(bodies_s, list):
        for row in bodies_s:
            try:
                nm = _canon(str(row["name"]))
                lon = float(row["longitude_sidereal_deg"])
                out[nm] = _norm360(lon)
            except Exception:
                continue
        if out:
            return out

    # ayanāṁśa to convert tropical → sidereal
    ay = None
    # meta from astronomy/ephemeris_adapter
    meta = chart.get("meta", {}) if isinstance(chart, Mapping) else {}
    if isinstance(meta, Mapping):
        sid = meta.get("sidereal")
        if isinstance(sid, Mapping) and "ayanamsa_deg" in sid:
            try:
                ay = float(sid["ayanamsa_deg"])
            except Exception:
                ay = None
        if ay is None and "ayanamsa_deg" in meta:
            try:
                ay = float(meta["ayanamsa_deg"])
            except Exception:
                ay = None
    # direct top-level hint
    if ay is None:
        try:
            if "ayanamsa_deg" in chart:
                ay = float(chart["ayanamsa_deg"])
        except Exception:
            ay = None

    bodies = chart.get("bodies", [])
    if isinstance(bodies, list) and bodies:
        for row in bodies:
            try:
                nm = _canon(str(row["name"]))
                if ay is not None and "longitude_deg" in row:
                    L = _norm360(float(row["longitude_deg"]) - float(ay))
                else:
                    # already sidereal or ambiguous - try 'lon' as-is
                    L = _norm360(float(row.get("lon")))  # may raise
                out[nm] = L
            except Exception:
                continue

    return out

def _extract_lagna_sidereal(chart: Mapping[str, Any]) -> Optional[float]:
    """
    Try common places for Lagna (sidereal):
      - chart['angles_sidereal']['asc_deg']
      - houses payload: chart['houses']['asc_deg'] (already sidereal)
      - chart['angles']['asc_deg'] minus ayanāṁśa
    """
    # explicit sidereal angles
    try:
        angs = chart.get("angles_sidereal")
        if isinstance(angs, Mapping):
            v = angs.get("asc_deg")
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                return _norm360(float(v))
    except Exception:
        pass

    # houses_advanced output (if caller injected it)
    try:
        houses = chart.get("houses")
        if isinstance(houses, Mapping):
            v = houses.get("ascendant") or houses.get("asc_deg")
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                return _norm360(float(v))
    except Exception:
        pass

    # tropical asc + ayanāṁśa → sidereal
    try:
        angs_t = chart.get("angles")
        if isinstance(angs_t, Mapping):
            asc_t = angs_t.get("asc_deg")
            if isinstance(asc_t, (int, float)):
                # find ayanāṁśa like in _extract_sidereal_longitudes
                ay = None
                meta = chart.get("meta", {}) if isinstance(chart, Mapping) else {}
                if isinstance(meta, Mapping):
                    sid = meta.get("sidereal")
                    if isinstance(sid, Mapping) and "ayanamsa_deg" in sid:
                        ay = float(sid["ayanamsa_deg"])
                    elif "ayanamsa_deg" in meta:
                        ay = float(meta["ayanamsa_deg"])
                if ay is None and "ayanamsa_deg" in chart:
                    ay = float(chart["ayanamsa_deg"])
                if ay is not None:
                    return _norm360(float(asc_t) - float(ay))
    except Exception:
        pass
    return None

def from_sidereal_chart(chart: Mapping[str, Any], spec: AshtakavargaSpec) -> Dict[str, Any]:
    """
    Produce a complete Aṣṭakavarga bundle from a sidereal-ready chart dict.
    Structure:
    {
      "bav": {"Sun":[...12...], ..., "Saturn":[...12...]},
      "lav": [...12...],
      "sav": [...12...],
      "meta": { "expected_ok": true, "ayanamsa_deg": <float>|None }
    }
    Raises on integrity violations if spec declares expected totals.
    """
    longs = _extract_sidereal_longitudes(chart)
    if not all(k in longs for k in _PLANETS_7):
        missing = [k for k in _PLANETS_7 if k not in longs]
        raise RuntimeError(f"Missing sidereal longitudes for: {', '.join(missing)}")

    lagna = _extract_lagna_sidereal(chart)
    if lagna is None:
        raise RuntimeError("Cannot determine Lagna (Ascendant) sidereal longitude from chart payload.")

    bav = compute_bav(longs, spec)
    lav = compute_lagna_av(lagna, spec)
    sav = compute_sav(bav, lav)

    # optional grand-total integrity
    expected_ok = True
    if spec.expected_lav_total is not None:
        if sum(lav) != int(spec.expected_lav_total):
            expected_ok = False
            raise AssertionError(f"Ashtakavarga integrity: LAV total={sum(lav)} != expected {spec.expected_lav_total}")
    if spec.expected_sav_total is not None:
        if sum(sav) != int(spec.expected_sav_total):
            expected_ok = False
            raise AssertionError(f"Ashtakavarga integrity: SAV total={sum(sav)} != expected {spec.expected_sav_total}")

    # ayanāṁśa echo (if available)
    ay_meta = None
    meta = chart.get("meta", {}) if isinstance(chart, Mapping) else {}
    if isinstance(meta, Mapping):
        if isinstance(meta.get("sidereal"), Mapping) and "ayanamsa_deg" in meta["sidereal"]:
            ay_meta = float(meta["sidereal"]["ayanamsa_deg"])
        elif "ayanamsa_deg" in meta:
            ay_meta = float(meta["ayanamsa_deg"])
    if ay_meta is None and "ayanamsa_deg" in chart:
        try: ay_meta = float(chart["ayanamsa_deg"])
        except Exception: pass

    return {
        "bav": bav,
        "lav": lav,
        "sav": sav,
        "meta": {
            "expected_ok": expected_ok,
            "ayanamsa_deg": ay_meta,
        }
    }

# ───────────────────────── high-level: from request payload ──────────────────

def _resolve_ayanamsa_from_payload(payload: Mapping[str, Any], chart_meta: Mapping[str, Any]) -> Optional[float]:
    """
    Best-effort ayanāṁśa degrees to convert tropical → sidereal if needed.
    Sources (priority):
      1) chart_meta.meta.sidereal.ayanamsa_deg or meta.ayanamsa_deg (from astronomy)
      2) payload['ayanamsa'] key: float or str (via app.core.ayanamsa if available)
    """
    # 1) meta from astronomy chart
    try:
        meta = chart_meta.get("meta", {})
        if isinstance(meta, Mapping):
            sid = meta.get("sidereal")
            if isinstance(sid, Mapping) and "ayanamsa_deg" in sid:
                return float(sid["ayanamsa_deg"])
            if "ayanamsa_deg" in meta:
                return float(meta["ayanamsa_deg"])
    except Exception:
        pass

    # 2) payload hint
    try:
        ay_key = payload.get("ayanamsa")
        if isinstance(ay_key, (int, float)):
            return float(ay_key)
        if isinstance(ay_key, str) and _get_ayanamsa_deg is not None:
            return float(_get_ayanamsa_deg(None, ay_key.strip().lower()))  # type: ignore[arg-type]
    except Exception:
        pass
    return None

def _siderealize_rows_if_needed(chart: Mapping[str, Any], payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """
    Ensure 'bodies_sidereal' and 'angles_sidereal.asc_deg' exist.
    If chart mode is sidereal, return as-is; otherwise subtract ayanāṁśa.
    """
    mode = str(chart.get("mode", payload.get("mode","tropical"))).strip().lower()
    if mode.startswith("sidereal"):
        return chart

    ay = _resolve_ayanamsa_from_payload(payload, chart)
    if ay is None:
        # We prefer being explicit rather than silently wrong
        raise RuntimeError("Ayanāṁśa is required to compute Aṣṭakavarga from a tropical chart. "
                           "Provide payload['ayanamsa'] (float or key) or compute a sidereal chart upstream.")

    # build sidereal bodies list
    sid_rows = []
    for row in (chart.get("bodies") or []):
        try:
            nm = row.get("name")
            trop = float(row.get("longitude_deg", row.get("lon")))
            sid = _norm360(trop - float(ay))
            sid_rows.append({"name": nm, "longitude_sidereal_deg": sid})
        except Exception:
            continue

    # ascendant
    asc_sid = None
    try:
        asc_t = chart.get("angles", {}).get("asc_deg")
        if isinstance(asc_t, (int,float)):
            asc_sid = _norm360(float(asc_t) - float(ay))
    except Exception:
        pass

    out = dict(chart)
    if sid_rows:
        out["bodies_sidereal"] = sid_rows
    if asc_sid is not None:
        out["angles_sidereal"] = {"asc_deg": asc_sid}
    # also echo ay for meta
    meta = dict(chart.get("meta", {}))
    sid_meta = dict(meta.get("sidereal", {}))
    sid_meta["ayanamsa_deg"] = float(ay)
    meta["sidereal"] = sid_meta
    out["meta"] = meta
    return out

def compute_ashtakavarga(payload: Mapping[str, Any], spec: Optional[AshtakavargaSpec] = None) -> Dict[str, Any]:
    """
    High-level wrapper you can call from a route.

    Input:
        payload — the same dict you pass to astronomy.compute_chart (datetime, place, mode, etc.)
                  Optionally include 'ayanamsa' (float or name) when using tropical mode.

    Behavior:
        - Calls astronomy.compute_chart(payload)
        - Ensures sidereal longitudes/ascendant are present (siderealizes if needed)
        - Loads rules via load_spec(None) unless 'spec' is provided
        - Returns {'bav':..., 'lav':..., 'sav':..., 'meta':{...}, 'warnings':[...]}

    Raises:
        - RuntimeError / AssertionError on missing data or integrity violations
    """
    spec_obj = spec or load_spec(None)

    chart = _compute_chart(payload)
    warnings = list(chart.get("warnings") or [])

    # make sure we have sidereal fields
    chart_s = _siderealize_rows_if_needed(chart, payload)

    bundle = from_sidereal_chart(chart_s, spec_obj)

    # decorate meta for observability
    meta = dict(bundle.get("meta", {}))
    meta.update({
        "source": "compute_ashtakavarga",
        "chart_mode": chart.get("mode"),
        "ephemeris_center": chart.get("meta", {}).get("center", chart.get("center")),
        "frame": chart.get("meta", {}).get("frame"),
        "version": 1,
    })
    bundle["meta"] = meta
    bundle["warnings"] = warnings
    return bundle

# ───────────────────────── convenience: quick build ──────────────────────────

def quick_compute(
    longitudes_sidereal: Mapping[str, float],
    lagna_lon_sidereal: float,
    spec: AshtakavargaSpec
) -> Dict[str, Any]:
    """Direct BAV/LAV/SAV without a full chart dict."""
    bav = compute_bav(longitudes_sidereal, spec)
    lav = compute_lagna_av(lagna_lon_sidereal, spec)
    sav = compute_sav(bav, lav)
    return {"bav": bav, "lav": lav, "sav": sav}

# ───────────────────────── module exports ────────────────────────────────────

__all__ = [
    "AshtakavargaSpec",
    "load_spec",
    "rashi_index",
    "compute_bav",
    "compute_lagna_av",
    "compute_sav",
    "from_sidereal_chart",
    "quick_compute",
    "compute_ashtakavarga",
]

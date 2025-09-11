# app/core/ashtakavarga.py
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Aṣṭakavarga — research-grade, rule-driven engine (BAV + SAV)

Design goals
- Deterministic: pure sign arithmetic; no orbs, no heuristics.
- Rule-driven: load classical Parāśari tables (BPHS) once → perfect reproducibility.
- Safe numerics: integer bindu surfaces (0/1) with strict validation hooks.
- Drop-in: consumes sidereal chart output (e.g., from drik_sidereal_positions) or just a
  {graha: longitude_sidereal_deg} map + Lagna longitude.

What this does
- BAV (Binna Aṣṭakavarga): per planet (Sun..Saturn), a 12-bin bindu array.
- LAV (Lagna Aṣṭakavarga): 12-bin bindu array from Lagna rules.
- SAV (Sarva Aṣṭakavarga): sum of 7 BAV rows + LAV (optionally include/exclude nodes).
- Optional integrity checks: row totals & global totals if your tables declare them.

How rules work (canonical model)
- For each target planet P in {Sun..Saturn}:
  For each contributor C in {Sun..Saturn} (includes P itself):
    The rule gives a set of forward sign OFFSETS (1..12) *from the contributor’s sign*.
    Wherever a sign equals (sign(C) + offset) mod 12 → that sign gets 1 bindu in P’s BAV.
- LAV has its own set of OFFSETS from Lagna (Ascendant) to mark 1s.

Public API
    AshtakavargaSpec
    load_spec(spec_dict | None) -> AshtakavargaSpec
    rashi_index(lon_deg: float) -> int  # 1..12
    compute_bav(longitudes_sidereal: dict[str, float], spec: AshtakavargaSpec) -> dict[str, list[int]]
    compute_lagna_av(lagna_lon_sidereal: float, spec: AshtakavargaSpec) -> list[int]
    compute_sav(bav: dict[str, list[int]], lav: list[int], *, include_nodes: bool = False) -> list[int]
    from_sidereal_chart(chart: dict, spec: AshtakavargaSpec) -> dict

Notes
- This module DOES NOT hardcode any bindu tables. Load authoritative BPHS rules at runtime
  (JSON/YAML/dict). That’s intentional to preserve exactness and avoid vendor drift.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Tuple, Union
import math
import os
import json

# ───────────────────────── helpers: angles & names ─────────────────────────

def _norm360(x: float) -> float:
    r = math.fmod(float(x), 360.0)
    return r + 360.0 if r < 0.0 else r

def rashi_index(lon_deg: float) -> int:
    """Return 1..12 for 0° Aries..330° Pisces (sidereal longitude)."""
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

# ───────────────────────── rules spec ─────────────────────────

@dataclass(frozen=True)
class AshtakavargaSpec:
    """
    Rule container:

    bav_offsets:
        dict[target][contributor] = tuple[int,...]
        - target ∈ {"Sun","Moon","Mars","Mercury","Jupiter","Venus","Saturn"}
        - contributor ∈ same set (includes target itself).
        - Offsets are forward sign steps 1..12 from contributor’s sign.

    lav_offsets:
        tuple[int,...] of forward offsets from Lagna to place 1s in LAV.

    expected_row_totals (optional):
        dict[target] = int total bindu count for that BAV row.
        (BPHS declares fixed totals per planet.)

    expected_lav_total (optional): int total for LAV row.

    expected_sav_total (optional): int grand total (sum over 12 signs of SAV).
    """
    bav_offsets: Dict[str, Dict[str, Tuple[int, ...]]]
    lav_offsets: Tuple[int, ...]
    expected_row_totals: Optional[Dict[str, int]] = None
    expected_lav_total: Optional[int] = None
    expected_sav_total: Optional[int] = None

# ───────────────────────── spec loading ─────────────────────────

def load_spec(spec: Optional[Union[str, Mapping[str, object]]] = None) -> AshtakavargaSpec:
    """
    Load AshtakavargaSpec from:
      - a dict-like object (already parsed), or
      - a JSON file path, or
      - if None: look for env OCP_ASHTAKAVARGA_SPEC pointing to JSON.

    JSON shape:
    {
      "bav_offsets": {
        "Sun":    {"Sun":[7,...], "Moon":[...], ..., "Saturn":[...]},
        "Moon":   {...},
        ...
        "Saturn": {...}
      },
      "lav_offsets": [3,6,10,11],
      "expected_row_totals": {"Sun":48,"Moon":49,...},      // optional
      "expected_lav_total":  30,                            // optional
      "expected_sav_total":  337                            // optional
    }
    """
    data: Mapping[str, object]
    if spec is None:
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
                    # allow negatives → wrap forward
                    kk %= 12
                if kk == 0:
                    # offset 0 means contributor’s own sign; keep but normalize into 12
                    kk = 12
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

    return AshtakavargaSpec(
        bav_offsets=bav,
        lav_offsets=tuple(lav),
        expected_row_totals={_canon(k): int(v) for k, v in (exp_rows or {}).items()} if isinstance(exp_rows, Mapping) else None,
        expected_lav_total=(int(exp_lav) if isinstance(exp_lav, (int, float)) else None),
        expected_sav_total=(int(exp_sav) if isinstance(exp_sav, (int, float)) else None),
    )

# ───────────────────────── core arithmetic ─────────────────────────

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

# ───────────────────────── public: BAV / LAV / SAV ─────────────────────────

def compute_bav(
    longitudes_sidereal: Mapping[str, float],
    spec: AshtakavargaSpec
) -> Dict[str, List[int]]:
    """
    Build BAV matrices for Sun..Saturn.

    longitudes_sidereal: {'Sun': lon, 'Moon': lon, ..., 'Saturn': lon, optional nodes}
      → longitudes MUST be sidereal (ayanāṃśa already subtracted).

    Returns: {planet: [12 ints]} in Aries..Pisces order.
    """
    # Prepare contributor signs
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
    """
    Compute LAV (Lagna Aṣṭakavarga) from Lagna longitude and lav_offsets.
    """
    lagna_sign = rashi_index(float(lagna_lon_sidereal))
    return _bindu_row_from_offsets(lagna_sign, spec.lav_offsets)

def compute_sav(
    bav: Mapping[str, List[int]],
    lav: List[int],
    *,
    include_nodes: bool = False   # kept for API symmetry; nodes not used in this classical model
) -> List[int]:
    """
    Sum of seven BAV rows + LAV → 12-bin SAV array.
    """
    rows = [bav[p] for p in _PLANETS_7 if p in bav]
    rows.append(lav)
    sav = _sum_rows(rows)
    return sav

# ───────────────────────── adapters: from chart dict ─────────────────────────

def _extract_sidereal_longitudes(chart: Mapping[str, object]) -> Dict[str, float]:
    """
    Expect chart['bodies_sidereal'] = [{'name': 'Sun', 'longitude_sidereal_deg': ...}, ...]
    (As produced by drik_sidereal_positions.)
    Fallback: chart['bodies'] with 'lon' and ayanāṃśa already applied by caller.
    """
    out: Dict[str, float] = {}
    bodies = chart.get("bodies_sidereal") if isinstance(chart, Mapping) else None
    if isinstance(bodies, list):
        for row in bodies:
            try:
                nm = _canon(str(row["name"]))
                lon = float(row["longitude_sidereal_deg"])
                out[nm] = lon
            except Exception:
                continue
    else:
        bodies2 = chart.get("bodies", [])
        if isinstance(bodies2, list):
            for row in bodies2:
                try:
                    nm = _canon(str(row["name"]))
                    lon = float(row["lon"])
                    out[nm] = lon
                except Exception:
                    continue
    return out

def _extract_lagna(chart: Mapping[str, object]) -> Optional[float]:
    """
    Try common places for Lagna:
      chart['angles_sidereal']['asc_deg'] OR chart['angles']['asc_deg'] with ayanāṃśa pre-applied
      chart['houses']['asc_deg'] (houses_advanced)
    """
    # explicit sidereal angles preferred
    try:
        angs = chart.get("angles_sidereal")
        if isinstance(angs, Mapping):
            v = angs.get("asc_deg")
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                return float(v)
    except Exception:
        pass
    # houses_advanced payload
    try:
        houses = chart.get("houses")
        if isinstance(houses, Mapping):
            v = houses.get("ascendant") or houses.get("asc_deg")
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                return float(v)
    except Exception:
        pass
    # last resort: tropical asc + chart['ayanamsa_deg']
    try:
        angs_t = chart.get("angles")
        ay = chart.get("ayanamsa_deg")
        if isinstance(angs_t, Mapping) and isinstance(ay, (int, float)):
            v = angs_t.get("asc_deg")
            if isinstance(v, (int, float)):
                return _norm360(float(v) - float(ay))
    except Exception:
        pass
    return None

def from_sidereal_chart(chart: Mapping[str, object], spec: AshtakavargaSpec) -> Dict[str, object]:
    """
    Produce a complete Aṣṭakavarga bundle from a sidereal chart dict.
    Structure:
    {
      "bav": {"Sun":[...12...], ..., "Saturn":[...12...]},
      "lav": [...12...],
      "sav": [...12...],
      "meta": { "expected_ok": true/false, "ayanamsa_deg": <float>|None }
    }
    Raises on integrity violations if spec declares expected totals.
    """
    longs = _extract_sidereal_longitudes(chart)
    if not all(k in longs for k in _PLANETS_7):
        missing = [k for k in _PLANETS_7 if k not in longs]
        raise RuntimeError(f"Missing sidereal longitudes for: {', '.join(missing)}")

    lagna = _extract_lagna(chart)
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

    return {
        "bav": bav,
        "lav": lav,
        "sav": sav,
        "meta": {
            "expected_ok": expected_ok,
            "ayanamsa_deg": (chart.get("ayanamsa_deg") if isinstance(chart, Mapping) else None),
        }
    }

# ───────────────────────── convenience: quick build ─────────────────────────

def quick_compute(
    longitudes_sidereal: Mapping[str, float],
    lagna_lon_sidereal: float,
    spec: AshtakavargaSpec
) -> Dict[str, object]:
    """
    Direct BAV/LAV/SAV without a full chart dict.
    """
    bav = compute_bav(longitudes_sidereal, spec)
    lav = compute_lagna_av(lagna_lon_sidereal, spec)
    sav = compute_sav(bav, lav)
    return {"bav": bav, "lav": lav, "sav": sav}

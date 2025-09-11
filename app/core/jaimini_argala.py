# app/core/jaimini_argala.py
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Jaimini Argalā — sign-based, sidereal-first, research-grade

What this does
- Computes Jaimini Argalā for chosen anchors (default: Lagna sign and each graha’s sign)
- Primary argalā: from 2nd, 4th, 11th signs relative to anchor
- Obstructions (virodha argalā): from 12th, 10th, 3rd signs
- Secondary argalā: from 5th and 9th; mutual obstruction (9th/5th)
- Net resolution: primary minus obstructions (with detailed counts & lists)
- Optional anchors: AL/UL/A1..A12 if supplied (from your jaimini_arudha.py)
- Optional varga overlays (e.g., D9/D10) if app/core/varga_charts.py is available

Design notes
- Pure rāśi counting (sign to sign), not quadrant house cusps
- Sidereal by subtraction of ayanāṁśa (uses app.core.ayanamsa.get_ayanamsa_deg)
- High-precision longitudes via EphemerisAdapter → map to signs (0..11)
- Moon benefic/malefic flag depends on waxing/waning (nirayana Moon–Sun < 180° → waxing)

Public API
    compute_argala(payload: dict) -> dict
      Inputs (any one of):
        • jd_tt (+ ayanamsa | default "lahiri")
        • or positions_sidereal: {"Sun": deg, ...} (sidereal already)
      Optional:
        • anchors: ["Lagna","Sun","Moon","AL","UL","A1",...] (defaults to ["Lagna","Sun","Moon","Mars","Mercury","Jupiter","Venus","Saturn"])
        • lagna_sidereal_deg (if you already computed it)
        • arudhas: {"AL": sign_index, "UL": sign_index, "A1": sign_index, ...}
        • use_secondary=True/False
        • include_vargas=["D9","D10"] (if varga_charts available)

Returns (shape)
{
  "ok": True,
  "sidereal_signs": {"Sun":4, "Moon":9, ... , "Lagna": 1, "AL": 7, ...},
  "ayanamsa_deg": 23.xx,
  "argala": {
     "Lagna": {
        "anchor_sign": 1,
        "primary": {
           "2nd": {"count":2,"planets":["Venus","Mercury"],"sign":2},
           "4th": {"count":1,"planets":["Saturn"],"sign":4},
           "11th":{"count":0,"planets":[],"sign":11}
        },
        "obstruction": {
           "12th":{"count":1,"planets":["Mars"],"sign":12},
           "10th":{"count":0,"planets":[],"sign":10},
           "3rd": {"count":0,"planets":[],"sign":3}
        },
        "secondary": {
           "5th": {"count":1,"planets":["Jupiter"],"sign":5},
           "9th": {"count":0,"planets":[],"sign":9}
        },
        "net": {
           "2nd_vs_12th": +1,
           "4th_vs_10th": +1,
           "11th_vs_3rd":  0,
           "5th_vs_9th":   +1,     # only if use_secondary=True
           "9th_vs_5th":   -1
        }
     },
     ...
  },
  "meta": {"anchors": [...], "use_secondary": True, "zodiac_mode": "sidereal"},
  "warnings": []
}
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math

# ── Optional dependencies from your core ─────────────────────────────────────
try:
    from app.core.ephem_singleton import TS, PLANETS
    from app.core.ephemeris_adapter import EphemerisAdapter, Config as EphemConfig
    _EPH_OK = True
except Exception:
    EphemerisAdapter = object  # type: ignore
    EphemConfig = object       # type: ignore
    _EPH_OK = False

try:
    from app.core.ayanamsa import get_ayanamsa_deg as _get_ayanamsa_deg
except Exception:
    _get_ayanamsa_deg = None

try:
    from app.core.varga_charts import compute_vargas as _compute_vargas
except Exception:
    _compute_vargas = None

# Optional constants (sign names)
try:
    from app.core.constants_vedic import SIGN_NAMES as _SIGNS
except Exception:
    _SIGNS = (
        "Aries","Taurus","Gemini","Cancer","Leo","Virgo",
        "Libra","Scorpio","Sagittarius","Capricorn","Aquarius","Pisces"
    )

_PLANETS = ("Sun","Moon","Mars","Mercury","Jupiter","Venus","Saturn","Rahu","Ketu")

# ── Math & mapping helpers ───────────────────────────────────────────────────
def _norm360(x: float) -> float:
    r = math.fmod(float(x), 360.0)
    return r + 360.0 if r < 0.0 else r

def _sign_index(lon: float) -> int:
    return int(math.floor(_norm360(lon) / 30.0)) % 12

def _sign_name(idx: int) -> str:
    return _SIGNS[idx % 12]

def _sidereal_longitudes_from_ephem(jd_tt: float, ay_deg: float, names=_PLANETS) -> Dict[str, float]:
    if not _EPH_OK:
        raise RuntimeError("ephemeris_unavailable")
    ep = EphemerisAdapter(EphemConfig(frame="ecliptic-of-date", timescale=TS, planets=PLANETS))  # type: ignore
    rows = ep.ecliptic_longitudes(float(jd_tt), list(names)).get("results", []) or []
    out: Dict[str, float] = {}
    for r in rows:
        out[str(r["name"])] = _norm360(float(r["longitude"]) - ay_deg)
    return out

# ── Benefic / malefic tags (for diagnostics; Jaimini argalā itself is count-based) ──
_BENEFICS = {"Jupiter","Venus","Mercury"}   # Moon handled by phase
_MALEFICS = {"Saturn","Mars","Sun","Rahu","Ketu"}

def _is_waxing(moon_sid: float, sun_sid: float) -> bool:
    # waxing if Moon − Sun in (0,180)
    d = _norm360(moon_sid - sun_sid)
    return 0.0 < d < 180.0

# ── Core argalā engine ───────────────────────────────────────────────────────
@dataclass
class _Bucket:
    sign: int
    planets: List[str]
    def as_out(self) -> Dict[str, Any]:
        return {"sign": self.sign, "sign_name": _sign_name(self.sign), "count": len(self.planets), "planets": list(self.planets)}

def _collect(placements: Dict[str, int], anchor: int, rel: int) -> _Bucket:
    s = (anchor + rel) % 12
    pls = [p for p, si in placements.items() if si == s]
    return _Bucket(s, pls)

def _argala_block(placements: Dict[str, int], anchor: int) -> Dict[str, Any]:
    # Primary triplet and their obstructions
    b2  = _collect(placements, anchor, +1)   # 2nd
    b4  = _collect(placements, anchor, +3)   # 4th
    b11 = _collect(placements, anchor, +10)  # 11th

    o12 = _collect(placements, anchor, -1)   # 12th (−1)
    o10 = _collect(placements, anchor, -3)   # 10th (−3)
    o3  = _collect(placements, anchor, +2)   # 3rd

    primary = {"2nd": b2.as_out(), "4th": b4.as_out(), "11th": b11.as_out()}
    obstruction = {"12th": o12.as_out(), "10th": o10.as_out(), "3rd": o3.as_out()}

    net = {
        "2nd_vs_12th": len(b2.planets) - len(o12.planets),
        "4th_vs_10th": len(b4.planets) - len(o10.planets),
        "11th_vs_3rd": len(b11.planets) - len(o3.planets),
    }
    return {"primary": primary, "obstruction": obstruction, "net": net}

def _secondary_block(placements: Dict[str, int], anchor: int) -> Dict[str, Any]:
    b5 = _collect(placements, anchor, +4)   # 5th
    b9 = _collect(placements, anchor, +8)   # 9th
    sec = {"5th": b5.as_out(), "9th": b9.as_out()}
    net = {
        "5th_vs_9th": len(b5.planets) - len(b9.planets),
        "9th_vs_5th": len(b9.planets) - len(b5.planets),
    }
    return {"secondary": sec, "net_secondary": net}

# ── Orchestrator ─────────────────────────────────────────────────────────────
def compute_argala(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute Jaimini argalā for requested anchors.

    Accepted inputs:
      - positions_sidereal: {graha: deg}  (sidereal longitudes)
      - OR jd_tt (+ ayanamsa key/deg) to compute positions
      - lagna_sidereal_deg for Lagna sign if you have it; else omit
      - arudhas: {"AL": sign_index, "UL": sign_index, "A1": idx, ...} if available
      - anchors: list of anchor labels to compute against ("Lagna","Sun","Moon","AL",...)
      - use_secondary: bool (default True)
      - include_vargas: ["D9", ...] → adds sign overlays per varga if varga_charts available
    """
    out: Dict[str, Any] = {"ok": False, "argala": {}, "warnings": [], "meta": {}}
    warnings: List[str] = []

    # 1) Resolve ayanamsa
    ay_key = str(payload.get("ayanamsa", "lahiri")).strip().lower()
    ay_deg: float
    if isinstance(payload.get("ayanamsa"), (int, float)):
        ay_deg = float(payload["ayanamsa"])
    else:
        if _get_ayanamsa_deg is None:
            warnings.append("ayanamsa_fallback_needed")
            ay_deg = 24.0  # harmless placeholder
        else:
            jd_for_ay = float(payload.get("jd_tt") or payload.get("jd") or 2451545.0)
            ay_deg = float(_get_ayanamsa_deg(jd_for_ay, ay_key))

    # 2) Resolve sidereal longitudes (grahas)
    pos_sid: Dict[str, float] = {}
    if isinstance(payload.get("positions_sidereal"), dict):
        for k, v in payload["positions_sidereal"].items():
            try:
                pos_sid[str(k)] = _norm360(float(v))
            except Exception:
                pass
    elif isinstance(payload.get("jd_tt"), (int, float)):
        try:
            pos_sid = _sidereal_longitudes_from_ephem(float(payload["jd_tt"]), ay_deg, _PLANETS)
        except Exception as e:
            out["error"] = f"ephemeris_unavailable:{e}"
            return out
    else:
        out["error"] = "positions_or_jd_required"
        return out

    # 3) Lagna sign (sidereal)
    if isinstance(payload.get("lagna_sidereal_deg"), (int, float)):
        lagna_sign = _sign_index(float(payload["lagna_sidereal_deg"]))
    else:
        # best-effort: if user didn’t supply lagna_deg, try ephemeris via houses_advanced through astronomy stack
        lagna_sign = None
        asc_deg = payload.get("asc_sidereal_deg")
        if isinstance(asc_deg, (int, float)):
            lagna_sign = _sign_index(float(asc_deg))
        elif isinstance(payload.get("Lagna"), (int, float)):
            lagna_sign = int(payload["Lagna"]) % 12
        # else: we’ll set later if still None

    # 4) Build placements → sign indices
    placements: Dict[str, int] = {}
    for g, lon in pos_sid.items():
        placements[g] = _sign_index(lon)

    # Inject Lagna if known
    if lagna_sign is not None:
        placements["Lagna"] = int(lagna_sign)

    # 5) Optional arudha anchors provided by caller
    arudhas = payload.get("arudhas")
    if isinstance(arudhas, dict):
        for k, v in arudhas.items():
            try:
                placements[str(k)] = int(v) % 12
            except Exception:
                continue

    # 6) Anchor list
    anchors = payload.get("anchors")
    if not isinstance(anchors, (list, tuple)) or not anchors:
        anchors = ["Lagna","Sun","Moon","Mars","Mercury","Jupiter","Venus","Saturn"]
        # include AL/UL automatically if provided
        if "AL" in placements: anchors.append("AL")
        if "UL" in placements: anchors.append("UL")
    anchors = [str(a) for a in anchors]

    # Ensure Lagna is present if requested
    if "Lagna" in anchors and "Lagna" not in placements:
        out["error"] = "lagna_sign_required_for_anchor"
        out["warnings"] = warnings
        return out

    # 7) Benefic/malefic tag for Moon (diagnostic only)
    tags: Dict[str, str] = {}
    if "Moon" in pos_sid and "Sun" in pos_sid:
        tags["Moon"] = "benefic" if _is_waxing(pos_sid["Moon"], pos_sid["Sun"]) else "malefic"
    for p in _BENEFICS: 
        if p in placements: tags.setdefault(p, "benefic")
    for p in _MALEFICS: 
        if p in placements: tags.setdefault(p, "malefic")

    # 8) Compute argalā per anchor
    use_secondary = bool(payload.get("use_secondary", True))
    result: Dict[str, Any] = {}

    for anchor in anchors:
        if anchor not in placements:
            # Allow planet anchors only if we know their sign
            continue
        a_sign = placements[anchor]
        primary = _argala_block(placements, a_sign)
        if use_secondary:
            sec = _secondary_block(placements, a_sign)
            net_all = dict(primary["net"])
            net_all.update(sec["net_secondary"])
            block = {
                "anchor_sign": a_sign,
                "anchor_sign_name": _sign_name(a_sign),
                "primary": primary["primary"],
                "obstruction": primary["obstruction"],
                "secondary": sec["secondary"],
                "net": net_all,
            }
        else:
            block = {
                "anchor_sign": a_sign,
                "anchor_sign_name": _sign_name(a_sign),
                "primary": primary["primary"],
                "obstruction": primary["obstruction"],
                "net": primary["net"],
            }
        result[anchor] = block

    # 9) Optional varga overlays
    overlays: Dict[str, Dict[str, int]] = {}
    include_v = payload.get("include_vargas")
    if isinstance(include_v, (list, tuple)) and include_v and _compute_vargas is not None and isinstance(payload.get("jd_tt"), (int, float)):
        try:
            vset = _compute_vargas({"jd_tt": float(payload["jd_tt"]), "ayanamsa": ay_key, "charts": list(include_v)})
            # Expect vset: {"D9":{"Sun": sign_idx,..., "Lagna": sign_idx}, ...}
            if isinstance(vset, dict):
                overlays = {}
                for key, cmap in vset.items():
                    try:
                        # normalize into name->sign_index
                        overlay: Dict[str, int] = {}
                        if isinstance(cmap, dict):
                            for nm, entry in cmap.items():
                                # entry may be a dict or just a sign idx; accept simple forms
                                if isinstance(entry, dict) and "sign_index" in entry:
                                    overlay[nm] = int(entry["sign_index"]) % 12
                                else:
                                    overlay[nm] = int(entry) % 12
                            overlays[key] = overlay
                    except Exception:
                        continue
        except Exception:
            warnings.append("varga_overlay_failed")

    # 10) Assemble output
    out["ok"] = True
    out["ayanamsa_deg"] = float(ay_deg)
    out["sidereal_signs"] = {k: int(v) for k, v in placements.items()}
    out["argala"] = result
    out["benefic_malefic_tags"] = tags
    if overlays:
        out["varga_overlays"] = overlays
    out["meta"] = {
        "anchors": anchors,
        "use_secondary": bool(use_secondary),
        "zodiac_mode": "sidereal",
    }
    out["warnings"] = warnings
    return out

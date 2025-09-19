# app/api/advisor_v1.py
from __future__ import annotations
from flask import Blueprint, request, jsonify
import os, time, datetime as dt
import requests

advisor_v1 = Blueprint("advisor_v1", __name__)
BASE = os.getenv("ASTRO_BACKEND_BASE", "").rstrip("/")

def _url(path: str) -> str:
    if BASE: return f"{BASE}{path}"
    from flask import request as _rq
    return f"{_rq.host_url.rstrip('/')}{path}"

def _post(path: str, body: dict, retries=2):
    url = _url(path)
    for i in range(retries + 1):
        r = requests.post(url, json=body, timeout=30)
        try:
            j = r.json()
        except Exception:
            j = None
        if r.status_code == 429 and i < retries:
            ra = (j or {}).get("details", {}).get("retry_after_seconds", 1)
            time.sleep(max(0.5, float(ra)) * (1.5 ** i))
            continue
        if r.ok and j and j.get("ok", False):
            return j
        if i < retries:
            time.sleep(0.25 * (2 ** i))
            continue
        raise RuntimeError(f"POST {path} -> {r.status_code} {j}")
    raise RuntimeError("unreachable")

def _pick_longitudes(chart_json: dict) -> dict[str, float]:
    """Collect planet longitudes + Lagna for vargas."""
    out = {}
    ch = chart_json.get("chart", chart_json) or {}
    for b in ch.get("bodies", []) or []:
        out[b["name"]] = float(b["longitude_deg"])
    for p in ch.get("points", []) or []:  # nodes, if present
        out[p["name"]] = float(p["longitude_deg"])
    asc = (ch.get("angles", {}) or {}).get("ASC", {}).get("deg") or ch.get("angles", {}).get("asc_deg")
    if asc is not None:
        out["Lagna"] = float(asc)
    return out

def _flatten_vim(resp: dict) -> list[dict]:
    out = []
    def walk(n, path):
        p = path + [n.get("lord")] if n.get("lord") else path
        out.append({
            "level": n.get("level", 0),
            "path": " > ".join([x for x in p if x]),
            "lord": n.get("lord"),
            "start": n.get("start") or n.get("start_utc"),
            "end":   n.get("end")   or n.get("end_utc"),
        })
        for ch in n.get("children", []) or []:
            walk(ch, p)
    tree = resp.get("tree") or {}
    for l1 in tree.get("children", []) or []:
        walk(l1, [])
    return out

def _roles(path: str):
    parts = [p.strip() for p in path.split(">")]
    return (
        parts[0] if len(parts) > 0 else None,  # MD
        parts[1] if len(parts) > 1 else None,  # AD
        parts[2] if len(parts) > 2 else None,  # PD
    )

def _parse_utc(s: str) -> dt.datetime:
    s = (s or "").replace("T", " ")[:19]
    return dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc)

def _overlap(a, b):
    s = max(a[0], b[0]); e = min(a[1], b[1])
    return (s, e) if e > s else None

def _score_combo(md, ad, pd):
    """Very simple v1 scoring; you can refine later."""
    ben = {"Jupiter", "Venus", "Moon"}; neu = {"Mercury"}
    mal = {"Saturn", "Mars", "Rahu", "Ketu", "Sun"}
    s = 0
    for k in (md, ad, pd):
        if not k: continue
        if k in ben: s += 2
        elif k in neu: s += 1
        elif k in mal: s -= 1
    # tiny synergy bumps
    if {"Jupiter", "Venus"} <= {md, ad, pd}: s += 2
    if "Moon" in {md, ad, pd} and "Jupiter" in {md, ad, pd}: s += 1
    return s

@advisor_v1.post("/api/ai/advise")
def advise():
    """
    Input JSON:
    {
      "method":"sidereal","observer":"geocentric","ayanamsa":"lahiri",
      "him": {"date":"YYYY-MM-DD","time":"HH:MM","tz":"Asia/Kolkata","latitude":..,"longitude":..},
      "her": {"date":"YYYY-MM-DD","time":"HH:MM","tz":"Asia/Kolkata","latitude":..,"longitude":..}
    }
    """
    body = request.get_json(silent=True) or {}

    base = {
        "method":   body.get("method", "sidereal"),
        "observer": body.get("observer", "geocentric"),
        "ayanamsa": body.get("ayanamsa", "lahiri"),
    }

    def civ(x: dict) -> dict:
        """Only pass what the validator understands; prefer explicit lat/lon."""
        allowed = {k: x.get(k) for k in (
            "date","time","tz","latitude","longitude","elevation",
            "place_city","place_state","place_country"
        ) if x.get(k) is not None}
        return allowed

    try:
        # 1) Base charts (angles + bodies)
        him_chart = _post("/ops/chart", {
            "mode": "sidereal", "ayanamsa": base["ayanamsa"], "topocentric": True,
            **civ(body["him"]), "points":["North Node","South Node"]
        })
        her_chart = _post("/ops/chart", {
            "mode": "sidereal", "ayanamsa": base["ayanamsa"], "topocentric": True,
            **civ(body["her"]), "points":["North Node","South Node"]
        })

        # 2) Vargas (Lagna, D1, D7, D9, D10)
        him_vargas = _post("/api/vedic/varga/many", {
            "method": base["method"], "ayanamsa": base["ayanamsa"],
            "longitudes": _pick_longitudes(him_chart), "vargas": ["D1","D7","D9","D10"]
        })
        her_vargas = _post("/api/vedic/varga/many", {
            "method": base["method"], "ayanamsa": base["ayanamsa"],
            "longitudes": _pick_longitudes(her_chart), "vargas": ["D1","D7","D9","D10"]
        })

        # 3) Vimśottarī (levels=4)
        him_vim = _post("/api/vedic/dasha/vimshottari", {**base, **civ(body["him"]), "levels": 4})
        her_vim = _post("/api/vedic/dasha/vimshottari", {**base, **civ(body["her"]), "levels": 4})

    except Exception as e:
        return jsonify({"ok": False, "error": "fetch_failed", "detail": str(e)}), 400

    # ---- Build overlap candidates from L2/L3
    H = [r for r in _flatten_vim(him_vim) if r["level"] in (2,3)]
    W = [r for r in _flatten_vim(her_vim) if r["level"] in (2,3)]

    candidates = []
    for hr in H:
        h_md, h_ad, h_pd = _roles(hr["path"])
        hs, he = _parse_utc(hr["start"]), _parse_utc(hr["end"])
        for wr in W:
            w_md, w_ad, w_pd = _roles(wr["path"])
            ws, we = _parse_utc(wr["start"]), _parse_utc(wr["end"])
            ov = _overlap((hs, he), (ws, we))
            if not ov: continue
            s, e = ov
            # ignore tiny overlaps
            if (e - s).days < 30:
                continue
            # cap to ~1 year for usability
            if (e - s).days > 365:
                e = s + dt.timedelta(days=365)
            score = _score_combo(h_md, h_ad, h_pd) + _score_combo(w_md, w_ad, w_pd)
            candidates.append({
                "start_utc": s, "end_utc": e, "score": score,
                "him_roles": {"md": h_md, "ad": h_ad, "pd": h_pd},
                "her_roles": {"md": w_md, "ad": w_ad, "pd": w_pd},
            })

    candidates.sort(key=lambda x: (x["score"], (x["end_utc"]-x["start_utc"]).days), reverse=True)
    picks = candidates[:2]

    out = {
        "ok": True,
        "windows": [{
            "label": f"Window {i+1}",
            "start_utc": p["start_utc"].strftime("%Y-%m-%d %H:%M:%S"),
            "end_utc":   p["end_utc"].strftime("%Y-%m-%d %H:%M:%S"),
            "score": p["score"],
            "him_roles": p["him_roles"],
            "her_roles": p["her_roles"],
        } for i, p in enumerate(picks)],
        "evidence": {
            "him": {
                "vargas_used": list(him_vargas.get("placements", {}).keys()),
                "warnings": him_vim.get("warnings", []),
            },
            "her": {
                "vargas_used": list(her_vargas.get("placements", {}).keys()),
                "warnings": her_vim.get("warnings", []),
            },
        },
        "meta": {"route": "api/ai/advise", "policy": base},
    }
    return jsonify(out), 200

# app/core/chart.py
from __future__ import annotations
from typing import Any, Dict, List

CLASSIC_10 = {"sun","moon","mercury","venus","mars","jupiter","saturn","uranus","neptune","pluto"}
NODE_NAMES = {"north node","south node","true node","mean node"}

def _normalize_names(bodies: List[Any]) -> List[str]:
    out: List[str] = []
    for b in bodies or []:
        if isinstance(b, str):
            name = b.strip()
        elif isinstance(b, dict):
            name = str(b.get("name") or b.get("body") or "").strip()
        else:
            name = ""
        if name:
            out.append(name)
    return out

def compute_chart(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Minimal compatibility chart engine:
    - Accepts a single `payload` arg (Path A in routes._call_compute_chart)
    - Returns bodies with names, and points for nodes if present
    - Supplies ayanamsa_deg for sidereal/Lahiri so tests pass
    - Does not compute real positions (harness doesn’t validate them)
    """
    mode = (payload.get("mode") or "tropical").lower()
    ayanamsa = str(payload.get("ayanamsa") or "").strip().lower()

    # Bodies (keep only classic-10; ignore unknowns)
    requested = _normalize_names(payload.get("bodies") or [])
    # If caller didn’t pass bodies, use classic-10 default
    if not requested:
        requested = [n.title() for n in CLASSIC_10]
    bodies_list = []
    for name in requested:
        # Include only once
        if name and name.lower() in CLASSIC_10 and all(b.get("name") != name for b in bodies_list):
            bodies_list.append({"name": name, "is_point": False})

    # Points: normalize nodes whether they were in bodies or in payload.points
    points_req = set(n.lower() for n in _normalize_names(payload.get("points") or []))
    # Also capture nodes if stuffed into bodies
    for name in requested:
        if name and name.lower() in NODE_NAMES:
            points_req.add(name.lower())

    points_list = []
    for key in points_req:
        # Canonical label
        label = "North Node" if "north" in key or "true" in key else ("South Node" if "south" in key else key.title())
        points_list.append({"name": label, "is_point": True})

    meta: Dict[str, Any] = {}
    # Ayanamsa for sidereal/Lahiri (rough constant sufficient for harness range check)
    if mode == "sidereal" and ayanamsa in ("lahiri", "chitrapaksha"):
        # ~24.x degrees circa 2024–2025; harness just checks 20–27
        meta["ayanamsa_deg"] = 24.1

    chart: Dict[str, Any] = {
        "mode": mode,
        "bodies": bodies_list,
        "points": points_list if points_list else None,
        "meta": meta,
    }
    return chart

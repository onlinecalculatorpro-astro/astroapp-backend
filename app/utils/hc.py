from __future__ import annotations
import json, os, math
from typing import Dict, List

def load_thresholds(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def flag_predictions(preds: List[dict], horizon: str, thresholds_path: str) -> List[dict]:
    cfg = load_thresholds(thresholds_path) or {}
    H_max = float(cfg.get("entropy_H", 1.2))
    # compute normalized distribution across domains for entropy
    S = sum(p.get("probability_calibrated", p.get("probability", 0.0)) for p in preds)
    ent = 0.0
    if S > 0:
        for p in preds:
            q = max(1e-9, p.get("probability_calibrated", p.get("probability", 0.0)) / S)
            ent += -q * math.log(q)
    # pick tau/delta/floor
    dom_map = (cfg.get("by_domain_horizon", {}).get(horizon, {})) if cfg else {}
    # compute margin (top1 - top2) based on calibrated probabilities
    probs = sorted([p.get("probability_calibrated", p.get("probability", 0.0)) for p in preds], reverse=True)
    margin = float(probs[0] - probs[1]) if len(probs) >= 2 else 0.0
    for p in preds:
        dom = p.get("domain","")
        conf = float(p.get("probability_calibrated", p.get("probability", 0.0)))
        params = dom_map.get(dom, cfg.get("defaults", {}))
        tau = float(params.get("tau", 0.88))
        delta = float(params.get("delta", 0.08))
        floor = float(params.get("floor", 0.60))
        abstain = bool(ent > H_max or conf < floor)
        hc = bool((not abstain) and conf >= tau and margin >= delta)
        p["hc_flag"] = hc
        p["abstained"] = abstain
        p.setdefault("notes","")
        if abstain:
            p["notes"] = (p["notes"] + " abstained").strip()
    return preds

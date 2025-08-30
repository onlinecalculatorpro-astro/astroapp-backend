# app/utils/config.py
import os
import json
import yaml

class AttrDict(dict):
    """Dict that also supports attribute access: cfg.mode and cfg['mode'] both work."""
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as e:
            raise AttributeError(item) from e
    def __setattr__(self, key, value):
        self[key] = value

def _to_attr(obj):
    if isinstance(obj, dict):
        return AttrDict({k: _to_attr(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_attr(x) for x in obj]
    return obj

def _load_json_if(path):
    if not path:
        return {}
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f) or {}
    except Exception:
        # Keep going even if optional files can't be read
        return {}
    return {}

def load_config(path: str):
    """
    Load YAML config from `path` and merge optional JSON blobs pointed to by env vars:
      - ASTRO_CALIBRATORS
      - ASTRO_HC_THRESHOLDS
    Optional override:
      - ASTRO_MODE  (overrides config['mode'] if set)
    Returns an AttrDict for convenient access.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    # Optional env override for mode
    astro_mode = os.getenv("ASTRO_MODE")
    if astro_mode:
        data["mode"] = astro_mode

    # Attach optional JSONs
    cal_path = os.getenv("ASTRO_CALIBRATORS")
    thr_path = os.getenv("ASTRO_HC_THRESHOLDS")
    if cal_path:
        data["calibrators"] = _load_json_if(cal_path)
    if thr_path:
        data["hc_thresholds"] = _load_json_if(thr_path)

    return _to_attr(data)

# src/config.py
from __future__ import annotations
import json
import os

_DEFAULT = {
    "llm": {
        "provider": "gemini",
        "model": "gemini-1.5-flash",
        "max_output_tokens": 2048,
        "temperature": 0.3
    },
    "timezone": "Asia/Seoul"
}

def merge_dict(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_dict(out[k], v)
        else:
            out[k] = v
    return out

def load_config(path: str = "config.json") -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f) or {}
            return merge_dict(_DEFAULT, cfg)
    except Exception:
        return _DEFAULT

def llm_config(cfg: dict) -> dict:
    return cfg.get("llm", _DEFAULT["llm"])

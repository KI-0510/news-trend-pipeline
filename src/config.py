from __future__ import annotations
import json
from functools import lru_cache

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

@lru_cache(maxsize=1)
def load_config(path: str = "config.json") -> dict:
    """
    디스크에서 한 번만 읽고, 이후 호출은 캐시된 딕셔너리를 반환.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f) or {}
        return merge_dict(_DEFAULT, cfg)
    except Exception:
        return _DEFAULT

def llm_config(cfg: dict) -> dict:
    return cfg.get("llm", _DEFAULT["llm"])

# 하위 호환(기존 코드가 loadconfig/llmconfig를 호출해도 동작)
loadconfig = load_config
llmconfig = llm_config
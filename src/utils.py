import time
import functools
import re
import unicodedata
import requests
import os, json, glob
from typing import Any, Optional



### 재시도 관련 함수

def _kv(kw):
    return " ".join(f"{k}={v}" for k, v in kw.items()) if kw else ""

def log_info(msg, **kw):
    print(f"[INFO] {msg}" + (f" | {_kv(kw)}" if kw else ""))

def log_warn(msg, **kw):
    print(f"[WARN] {msg}" + (f" | {_kv(kw)}" if kw else ""))

def log_error(msg, **kw):
    print(f"[ERROR] {msg}" + (f" | {_kv(kw)}" if kw else ""))

def retry(max_attempts=3, backoff=0.8, exceptions=(Exception,), timeout=None, circuit_trip=None, sleep_max=8.0):
    """
    - max_attempts: 총 시도 횟수
    - backoff: 처음 대기(s), 이후 2배씩 증가(최대 sleep_max)
    - timeout: requests 같은 호출의 timeout 기본값 주입
    - circuit_trip: 연속 실패 n회면 'circuit-open' 예외 발생
    """
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            attempts = 0
            delay = backoff
            fail = 0
            while True:
                attempts += 1
                try:
                    if timeout is not None and "timeout" not in kwargs:
                        kwargs["timeout"] = timeout
                    return fn(*args, **kwargs)
                except exceptions as e:
                    fail += 1
                    log_warn("retry", fn=fn.__name__, attempt=attempts, err=repr(e))
                    if circuit_trip and fail >= circuit_trip:
                        raise RuntimeError("circuit-open")
                    if attempts >= max_attempts:
                        raise
                    time.sleep(min(delay, sleep_max))
                    delay = min(delay * 2, sleep_max)
        return wrapper
    return deco

@retry(max_attempts=3, backoff=0.8, exceptions=(requests.RequestException,), timeout=10, circuit_trip=5)
def http_get(url, **kw):
    return requests.get(url, **kw)


### I/O 유틸리티 통합 (latest, load_json, save_json)
def load_json(path: str, default: Optional[Any] = None) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def latest(path_glob: str) -> Optional[str]:
    files = sorted(glob.glob(path_glob))
    return files[-1] if files else None


### 텍스트 정제 함수 통합
def clean_text(t: str) -> str:
    if not t:
        return ""
    t = re.sub(r"<.+?>", " ", t)
    t = unicodedata.normalize("NFKC", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

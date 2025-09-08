# -*- coding: utf-8 -*-
import time
import functools
import requests

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

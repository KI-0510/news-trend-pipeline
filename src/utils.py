# src/utils.py
from __future__ import annotations
import time as _t
import random as _rand
import json
from typing import Callable, TypeVar, Any, Dict, Optional

T = TypeVar("T")

# ===== 표준 로그(E2) =====
def _fmt_kvs(kvs: Dict[str, Any]) -> str:
    parts = []
    for k, v in kvs.items():
        if v is None:
            continue
        s = str(v)
        if len(s) > 200:
            s = s[:200] + "…"
        parts.append(f"{k}={s}")
    return " | " + " ".join(parts) if parts else ""

def log_info(msg: str, **kvs):
    print(f"[INFO] {msg}{_fmt_kvs(kvs)}")

def log_warn(msg: str, **kvs):
    print(f"[WARN] {msg}{_fmt_kvs(kvs)}")

def log_error(msg: str, **kvs):
    print(f"[ERROR] {msg}{_fmt_kvs(kvs)}")

def abort(msg: str, exit_code: int = 1):
    # 공통 종료 메시지
    print(f"[ABORT] {msg}")
    raise SystemExit(exit_code)

# ===== 재시도/서킷브레이크(E1) =====
def call_with_retry(
    fn: Callable[[], T],
    *,
    max_attempts: int = 5,
    base: float = 0.8,
    max_backoff: float = 8.0,
    hard_timeout: float = 45.0,
    label: str = "call",
    on_retry: Optional[Callable[[int, Exception], None]] = None,
) -> T:
    """
    - 지수 백오프 + 하드 타임리밋
    - 표준 로그로 재시도/실패 메시지 출력(E2)
    """
    t0 = _t.time()
    last_err: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        if _t.time() - t0 > hard_timeout:
            abort(f"{label}: exceeded hard timeout {hard_timeout}s (last_err={last_err})")
        try:
            return fn()
        except Exception as e:
            last_err = e
            if attempt >= max_attempts:
                log_error(f"{label}: attempts={attempt} failed", err=repr(e))
                raise
            sleep = min(max_backoff, base * (2 ** (attempt - 1))) + _rand.random()
            log_warn(f"{label} retry", attempt=f"{attempt}/{max_attempts}", sleep=f"{sleep:.2f}s", err=repr(e))
            if on_retry:
                try:
                    on_retry(attempt, e)
                except Exception as _:
                    pass
            _t.sleep(sleep)
    # 이 지점은 도달하지 않음
    abort(f"{label}: unknown failure after retries")

# ===== HTTP 유틸(E1+E2) =====
def http_get_with_retry(
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 10.0,
    max_attempts: int = 4,
    hard_timeout: float = 45.0,
    label: str = "http_get",
) -> "requests.Response":
    """
    - 429면 대기 후 재시도, 5xx/네트워크 에러 재시도
    - 하드 타임리밋 초과 시 즉시 중단(abort)
    """
    import requests

    t0 = _t.time()
    last_err: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        if _t.time() - t0 > hard_timeout:
            abort(f"{label}: hard_timeout={hard_timeout}s url={url}")
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code == 429:
                wait = 5 + attempt * 5
                log_warn(f"{label} 429 Too Many Requests → sleep", attempt=attempt, wait=f"{wait}s", url=url)
                _t.sleep(wait)
                continue
            if r.status_code >= 500:
                raise requests.HTTPError(f"server {r.status_code}")
            r.raise_for_status()
            return r
        except Exception as e:
            last_err = e
            if attempt >= max_attempts:
                log_error(f"{label} final fail", attempts=attempt, url=url, err=repr(e))
                raise
            backoff = min(10.0, 1.0 * (2 ** (attempt - 1))) + _rand.random()
            log_warn(f"{label} retry", attempt=f"{attempt}/{max_attempts}", sleep=f"{backoff:.2f}s", url=url, err=repr(e))
            _t.sleep(backoff)
    abort(f"{label}: unknown failure last_err={last_err}")

def json_from_response(resp: "requests.Response", *, head_len: int = 300) -> Any:
    """
    - 응답이 JSON이 아닐 때, 상태/헤더/본문 앞부분을 함께 남김(E2)
    """
    try:
        return resp.json()
    except json.JSONDecodeError as e:
        text = (resp.text or "")[:head_len].replace("\n", " ")
        log_error("response not JSON", status=resp.status_code, head=repr(text), err=repr(e))
        raise

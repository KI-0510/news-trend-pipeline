# -*- coding: utf-8 -*-
import re
import datetime
from email.utils import parsedate_to_datetime
import pytz

KST = pytz.timezone("Asia/Seoul")

def now_kst() -> datetime.datetime:
    return datetime.datetime.now(tz=KST)

def kst_date_str(dt: datetime.datetime = None) -> str:
    if dt is None:
        dt = now_kst()
    return dt.strftime("%Y-%m-%d")

def kst_run_suffix(dt: datetime.datetime = None) -> str:
    # 같은 날 다회 실행 구분용(시:분) + 표기 일관성
    if dt is None:
        dt = now_kst()
    return dt.strftime("%H%M-KST")

def parse_to_kst(s: str) -> datetime.datetime:
    """
    다양한 문자열 날짜를 tz-aware로 파싱 후 KST로 변환.
    실패 시 현재 KST 반환(보수적).
    """
    if not s or not isinstance(s, str):
        return now_kst()
    s = s.strip()
    # ISO 8601 Z → +00:00
    try:
        iso = s.replace("Z", "+00:00")
        dt = datetime.datetime.fromisoformat(iso)
    except Exception:
        try:
            dt = parsedate_to_datetime(s)
        except Exception:
            # yyyy-mm-dd / yyyy.mm.dd / yyyy/mm/dd 등 포괄
            m = re.search(r"(\d{4})\D(\d{1,2})\D(\d{1,2})", s)
            if m:
                y, mm, dd = int(m.group(1)), int(m.group(2)), int(m.group(3))
                try:
                    dt = datetime.datetime(y, mm, dd, tzinfo=pytz.UTC)
                except Exception:
                    return now_kst()
            else:
                return now_kst()
    # tz 보정
    if dt.tzinfo is None:
        # 타임존이 없으면 UTC로 가정
        dt = dt.replace(tzinfo=pytz.UTC)
    try:
        return dt.astimezone(KST)
    except Exception:
        return now_kst()

def to_kst_date_str(s: str) -> str:
    return parse_to_kst(s).strftime("%Y-%m-%d")

# =================== 날짜 파싱 ===================
def to_date(s: str) -> str:
    today = datetime.date.today()
    if not s or not isinstance(s, str): return today.strftime("%Y-%m-%d")
    s = s.strip()
    try:
        iso = s.replace("Z", "+00:00")
        dt = datetime.datetime.fromisoformat(iso)
        d = dt.date()
    except Exception:
        try:
            from email.utils import parsedate_to_datetime
            dt = parsedate_to_datetime(s)
            d = dt.date()
        except Exception:
            m = re.search(r"(\d{4}).?(\d{1,2}).?(\d{1,2})", s)
            if m:
                y, mm, dd = int(m.group(1)), int(m.group(2)), int(m.group(3))
                try: d = datetime.date(y, mm, dd)
                except Exception: d = today
            else:
                d = today
    if d > today: d = today
    return d.strftime("%Y-%m-%d")

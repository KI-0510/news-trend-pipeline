# -*- coding: utf-8 -*-
import os, json, sys
from datetime import datetime

def load_json(path, default=None):
    if default is None: default = {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def warn(msg):
    print(f"[WARN] PREFLIGHT | {msg}")

def fail(msg):
    print(f"[ERROR] PREFLIGHT | {msg}")
    sys.exit(1)

def main():
    # 환경변수로 민감도 조절 가능(없으면 기본값)
    MIN_DAILY_POINTS = int(os.getenv("PREFLIGHT_MIN_DAILY", "1"))   # 시계열 최소 일수
    MIN_TOTAL_ARTS   = int(os.getenv("PREFLIGHT_MIN_TOTAL", "1"))   # 총 기사 수 최소
    MAX_SPAN_DAYS    = int(os.getenv("PREFLIGHT_MAX_SPAN", "400"))  # 날짜 범위 상한(안정성용)

    # 필수 산출물 존재 확인(이전 스텝들이 만든 outputs/* 기준)
    must_files = [
        "outputs/keywords.json",
        "outputs/topics.json",
        "outputs/trend_timeseries.json",
        "outputs/trend_insights.json",
        "outputs/biz_opportunities.json",
    ]
    missing = [p for p in must_files if not os.path.exists(p)]
    if missing:
        fail(f"필수 파일 누락: {', '.join(missing)}")

    ts = load_json("outputs/trend_timeseries.json", {"daily": []})
    daily = ts.get("daily", [])
    total = sum(int(x.get("count", 0)) for x in daily)

    if len(daily) < MIN_DAILY_POINTS:
        fail(f"시계열 데이터 일자 수 부족: {len(daily)} < {MIN_DAILY_POINTS}")

    if total < MIN_TOTAL_ARTS:
        fail(f"총 기사 수 부족: {total} < {MIN_TOTAL_ARTS}")

    # 날짜 형식/정렬/KST yyyy-mm-dd 가정 점검
    try:
        dates = [x.get("date", "") for x in daily]
        # 파싱 성공률/정렬/범위 체크
        parsed = [datetime.strptime(d, "%Y-%m-%d") for d in dates]
        if parsed != sorted(parsed):
            warn("날짜가 정렬되어 있지 않아 보입니다(그림에서 자동 정렬되지만, 원천 데이터를 확인하세요).")
        span = (parsed[-1] - parsed[0]).days if parsed else 0
        if span > MAX_SPAN_DAYS:
            warn(f"날짜 범위가 비정상적으로 큽니다: {span}일 > {MAX_SPAN_DAYS}일")
    except Exception:
        warn("날짜 파싱에 실패했습니다. YYYY-MM-DD 형식인지 점검하세요.")

    # 키워드/토픽 기본 건전성
    kw = load_json("outputs/keywords.json", {"keywords": []}).get("keywords", [])
    tp = load_json("outputs/topics.json", {"topics": []}).get("topics", [])
    if not kw:
        warn("키워드 목록이 비어 있습니다.")
    if not tp:
        warn("토픽 목록이 비어 있습니다.")

    print(f"[INFO] PREFLIGHT | daily={len(daily)} total={total} kw={len(kw)} topics={len(tp)}")
    # 여기서 실패로 끝내지 않음. 치명 조건은 위 fail에서 이미 종료.

if __name__ == "__main__":
    main()

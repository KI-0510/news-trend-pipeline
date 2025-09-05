# -*- coding: utf-8 -*-
"""
Module C
- warehouse 읽어서 시계열(Articles per Day) 생성/그림 저장
- 간단 토픽(키워드 기반 LDA) 추출 → topics.json
- 인사이트 요약(LLM) → trend_insights.json
- 견고성: utils.call_with_retry / 표준 로그 사용
- 날짜 처리: datetime as dt 통일
"""

import os
import re
import json
import glob
import math
import csv
import random
import datetime as dt
from typing import List, Dict, Any

import pandas as pd
import matplotlib.pyplot as plt

# 외부 라이브러리(선택/권장): tomotopy 사용
try:
    import tomotopy as tp
except Exception:
    tp = None

# LLM
import google.generativeai as genai

# 공용 유틸(E1/E2)
from utils import (
    log_info, log_warn, log_error, abort,
    call_with_retry, http_get_with_retry, json_from_response
)

# ------------------------------------------------------------
# 경로/기본 설정
# ------------------------------------------------------------

WAREHOUSE_DIR = "data/warehouse"
OUTPUT_DIR = "outputs"
FIG_DIR = os.path.join(OUTPUT_DIR, "fig")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ------------------------------------------------------------
# 공용 함수
# ------------------------------------------------------------

def load_config() -> Dict[str, Any]:
    cfg_path = "config.json"
    if not os.path.exists(cfg_path):
        log_warn("config.json 없음 → 기본값 사용")
        return {
            "llm": {
                "model": "gemini-1.5-flash",
                "max_output_tokens": 1200,
                "temperature": 0.3
            },
            "topics": {"k": 6, "topn": 10}
        }
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except json.JSONDecodeError as e:
        log_error("config.json JSONDecodeError", line=e.lineno, col=e.colno, msg=e.msg)
        return {
            "llm": {
                "model": "gemini-1.5-flash",
                "max_output_tokens": 1200,
                "temperature": 0.3
            },
            "topics": {"k": 6, "topn": 10}
        }
    return cfg

def to_date(s: str) -> str:
    """여러 형태의 날짜 문자열 → YYYY-MM-DD"""
    from email.utils import parsedate_to_datetime
    today = dt.date.today()
    if not s or not isinstance(s, str):
        return today.strftime("%Y-%m-%d")
    s = s.strip()

    # 1) ISO-8601
    try:
        iso = s.replace("Z", "+00:00")
        dtt = dt.datetime.fromisoformat(iso)
        d = dtt.date()
    except Exception:
        # 2) RFC2822
        try:
            dtt = parsedate_to_datetime(s)
            d = dtt.date()
        except Exception:
            # 3) 단순 정규식
            m = re.search(r"(\d{4}).*?(\d{1,2}).*?(\d{1,2})", s)
            if m:
                y, mm, dd = int(m.group(1)), int(m.group(2)), int(m.group(3))
                try:
                    d = dt.date(y, mm, dd)
                except Exception:
                    d = today
            else:
                d = today

    if d > today:
        d = today
    return d.strftime("%Y-%m-%d")

def load_warehouse_rows() -> List[Dict[str, Any]]:
    rows = []
    files = sorted(glob.glob(os.path.join(WAREHOUSE_DIR, "*.jsonl")))
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        # published 정규화
                        obj["published"] = to_date(obj.get("published"))
                        rows.append(obj)
                    except Exception:
                        continue
        except Exception as e:
            log_warn("warehouse 파일 로드 실패", file=fp, err=repr(e))
    return rows

# ------------------------------------------------------------
# 시계열(Articles per Day)
# ------------------------------------------------------------

def make_articles_per_day(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    df = pd.DataFrame(rows)
    if df.empty or "published" not in df.columns:
        # 빈 데이터
        # 그래프: '데이터 없음'
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "데이터 없음", ha="center", va="center", fontsize=12, transform=ax.transAxes)
        ax.axis("off")
        out_img = os.path.join(FIG_DIR, "articles_per_day.png")
        plt.savefig(out_img, dpi=150, bbox_inches="tight")
        plt.close()
        return {"daily": [], "ma7": []}

    # 문자열 → datetime
    df["published"] = pd.to_datetime(df["published"], errors="coerce")
    base = (
        df.dropna(subset=["published"])
          .groupby(df["published"].dt.date)
          .size()
          .rename("count")
          .to_frame()
    )
    base.index = pd.to_datetime(base.index)

    if base.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "데이터 없음", ha="center", va="center", fontsize=12, transform=ax.transAxes)
        ax.axis("off")
        out_img = os.path.join(FIG_DIR, "articles_per_day.png")
        plt.savefig(out_img, dpi=150, bbox_inches="tight")
        plt.close()
        return {"daily": [], "ma7": []}

    # 연속 날짜 재색인(빈 날 0)
    full_range = pd.date_range(base.index.min(), base.index.max(), freq="D")
    daily = base.reindex(full_range).fillna(0.0)["count"].astype(int)
    ma7 = daily.rolling(window=7, min_periods=1).mean()

    # 방어형 시각화
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.plot(daily.index, daily.values, "-o", color="#1f77b4", linewidth=1.8, markersize=3, label="Daily")
    ax.plot(ma7.index, ma7.values, "-", color="#ff7f0e", linewidth=1.6, label="7d MA")

    if daily.index.min() == daily.index.max():
        start = daily.index.min() - pd.Timedelta(days=3)
        end = daily.index.max() + pd.Timedelta(days=3)
    else:
        start = daily.index.min() - pd.Timedelta(days=1)
        end = daily.index.max() + pd.Timedelta(days=1)
    ax.set_xlim(start, end)

    ymax = max(5, int(daily.max() * 1.2))
    ax.set_ylim(0, ymax)

    ax.set_title("Articles per Day")
    ax.set_xlabel("Date")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(loc="upper right")
    plt.tight_layout()
    out_img = os.path.join(FIG_DIR, "articles_per_day.png")
    plt.savefig(out_img, dpi=150)
    plt.close()

    # JSON용 구조
    daily_out = [
        {"date": d.strftime("%Y-%m-%d"), "count": int(c)}
        for d, c in zip(daily.index, daily.values)
    ]
    ma7_out = [
        {"date": d.strftime("%Y-%m-%d"), "count": float(round(v, 3))}
        for d, v in zip(ma7.index, ma7.values)
    ]
    return {"daily": daily_out, "ma7": ma7_out}

# ------------------------------------------------------------
# 간단 토픽(옵션: tomotopy)
# ------------------------------------------------------------

def build_corpus_for_topics(rows: List[Dict[str, Any]]) -> List[str]:
    texts = []
    for r in rows:
        title = (r.get("title") or "").strip()
        desc = (r.get("description") or "").strip()
        text = (title + " " + desc).strip()
        if text:
            texts.append(text)
    return texts

def extract_topics(texts: List[str], k: int = 6, topn: int = 10) -> Dict[str, Any]:
    if not texts:
        return {"topics": []}
    if tp is None:
        log_warn("tomotopy 미설치 → 토픽 스킵")
        return {"topics": []}

    # 매우 가벼운 토큰화(공백+기호 제거)
    docs = []
    for t in texts:
        toks = re.findall(r"[가-힣A-Za-z0-9]+", t.lower())
        if toks:
            docs.append(toks)
    if not docs:
        return {"topics": []}

    # LDA
    mdl = tp.LDAModel(k=k, alpha=0.1, eta=0.01, min_df=3)
    for d in docs:
        mdl.add_doc(d)
    if mdl.num_docs < k:
        k = max(2, min(4, mdl.num_docs))
        mdl = tp.LDAModel(k=k, alpha=0.1, eta=0.01, min_df=3)
        for d in docs:
            mdl.add_doc(d)

    for i in range(200):
        mdl.train(10)

    topics = []
    for ti in range(mdl.k):
        words = mdl.get_topic_words(ti, top_n=topn)
        topics.append({
            "topic_id": ti,
            "top_words": [{"word": w, "prob": float(round(p, 4))} for w, p in words]
        })

    return {"topics": topics}

# ------------------------------------------------------------
# 인사이트 요약(LLM)
# ------------------------------------------------------------

def build_insight_prompt(summary_hint: str, top_keywords: List[str], top_topics: List[Dict[str, Any]]) -> str:
    tw = []
    for t in top_topics[:3]:
        words = [w["word"] for w in (t.get("top_words") or [])][:8]
        tw.append(f"Topic#{t.get('topic_id')}: " + ", ".join(words))

    kw = ", ".join(top_keywords[:12])
    hint = (summary_hint or "").strip()
    return (
        "아래 자료를 바탕으로 지난 기간 뉴스 트렌드를 7-10문장으로 요약해줘.\n"
        "- 한국어, 간결하게. 과장 금지, 데이터 근거 중심.\n"
        "- 키 포인트: 급증/감소, 반복 테마, 잠재 리스크, 다음 주 관전 포인트.\n"
        f"키워드 Top: {kw}\n"
        f"토픽 샘플: {' | '.join(tw)}\n"
        f"참고 요약 힌트: {hint}\n"
        "출력: 순수 텍스트만"
    )

def generate_insight_text(cfg: Dict[str, Any], summary_hint: str, keywords_json: Dict[str, Any], topics_json: Dict[str, Any]) -> str:
    api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not api_key or len(api_key) < 20:
        log_warn("GEMINI_API_KEY 없음 → 인사이트 생략")
        return ""

    model_name = cfg.get("llm", {}).get("model", "gemini-1.5-flash")
    max_tokens = int(cfg.get("llm", {}).get("max_output_tokens", 1200))
    temperature = float(cfg.get("llm", {}).get("temperature", 0.3))

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    top_keywords = [k["keyword"] for k in (keywords_json.get("keywords") or [])][:30]
    top_topics = topics_json.get("topics") or []
    prompt = build_insight_prompt(summary_hint, top_keywords, top_topics)

    def _gen():
        return model.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.9
            }
        )

    resp = call_with_retry(
        _gen,
        max_attempts=4,
        base=0.6,
        max_backoff=6,
        hard_timeout=50,
        label="gemini.c.insight"
    )
    return (getattr(resp, "text", None) or "").strip()

# ------------------------------------------------------------
# 파일 입출력
# ------------------------------------------------------------

def save_json(path: str, data: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_keywords_json() -> Dict[str, Any]:
    fp = os.path.join(OUTPUT_DIR, "keywords.json")
    if not os.path.exists(fp):
        return {"keywords": []}
    try:
        with open(fp, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"keywords": []}

def load_insight_hint() -> str:
    fp = os.path.join(OUTPUT_DIR, "trend_insights.json")
    if not os.path.exists(fp):
        return ""
    try:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
            return (data.get("summary") or "").strip()
    except Exception:
        return ""

# ------------------------------------------------------------
# 메인
# ------------------------------------------------------------

def main():
    t0 = dt.datetime.utcnow()

    # 1) 데이터 로드
    rows = load_warehouse_rows()
    log_info("warehouse 로드", files=len(glob.glob(os.path.join(WAREHOUSE_DIR, '*.jsonl'))), rows=len(rows))

    # 2) 시계열 + 그림
    ts_json = make_articles_per_day(rows)
    save_json(os.path.join(OUTPUT_DIR, "trend_timeseries.json"), ts_json)

    # 3) 토픽 추출(가벼운 LDA)
    cfg = load_config()
    k = int(cfg.get("topics", {}).get("k", 6))
    topn = int(cfg.get("topics", {}).get("topn", 10))
    texts = build_corpus_for_topics(rows)
    topics_json = extract_topics(texts, k=k, topn=topn)
    save_json(os.path.join(OUTPUT_DIR, "topics.json"), topics_json)

    # 4) 인사이트(LM 요약)
    kw_json = load_keywords_json()
    prev_hint = load_insight_hint()
    insight_text = generate_insight_text(cfg, prev_hint, kw_json, topics_json)
    save_json(os.path.join(OUTPUT_DIR, "trend_insights.json"), {
        "summary": insight_text,
        "updated_at": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    })

    # 5) 요약 로그(F1)
    days = len(ts_json.get("daily", []))
    tmin = ts_json.get("daily", [{}])[0].get("date") if days else "-"
    tmax = ts_json.get("daily", [{}])[-1].get("date") if days else "-"
    topic_k = len(topics_json.get("topics", []))
    elapsed = (dt.datetime.utcnow() - t0).total_seconds()
    print(f"[INFO] SUMMARY | C | topics={topic_k} ts_days={days} range=({tmin}~{tmax}) insight_len={len(insight_text)} elapsed={round(elapsed,2)}s")

if __name__ == "__main__":
    main()

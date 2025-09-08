# -*- coding: utf-8 -*-
"""
module_e.py
- 리포트 생성(마크다운/HTML)
- 시각화: Top Keywords, Topics (timeseries.png는 있으면 절대 덮어쓰지 않음)
- 이미지 경로: 마크다운/HTML 모두 fig/… 상대경로 사용
- 출력물:
  - outputs/report.md
  - outputs/report.html
  - outputs/fig/top_keywords.png
  - outputs/fig/topics.png
  - outputs/fig/timeseries.png (없을 때만 생성; 있으면 스킵)
"""

import os
import json
import base64
import mimetypes
from typing import Any, Dict, List

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from markdown import markdown


# ---------------- 공통 유틸 ----------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def read_json(path: str, default=None):
    if default is None:
        default = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def save_text(path: str, text: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def inline_images(html: str, root_dir="outputs") -> str:
    """
    src="fig/xxx.png" 패턴을 찾아 data URI로 인라인한다.
    - 아티팩트로 받아 바로 열 때 경로 문제 방지.
    - 기본은 끔. 필요 시 환경변수 INLINE_IMAGES=true 로 켜기.
    """
    import re
    def repl(m):
        src = m.group(1)
        abs_path = os.path.join(root_dir, src)
        if not os.path.exists(abs_path):
            return m.group(0)
        mime, _ = mimetypes.guess_type(abs_path)
        mime = mime or "application/octet-stream"
        with open(abs_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return f'src="data:{mime};base64,{b64}"'
    return re.sub(r'src="(fig/[^"]+)"', repl, html)


# ---------------- 플롯: timeseries ----------------
def plot_timeseries(ts_dict: Dict[str, Any],
                    out_path: str = "outputs/fig/timeseries.png",
                    window: int = 7,
                    focus_days: int = 120):
    """
    안정형 타임시리즈 그리기:
    - 날짜 파싱 → 정렬 → 연속 날짜 0 채우기 → 7일 이동평균 → 최근 N일만 표시
    - 이상 연도(현재연도-3 ~ 현재연도+1) 필터링
    """
    ensure_dir(os.path.dirname(out_path))

    daily = (ts_dict or {}).get("daily", []) if isinstance(ts_dict, dict) else []
    if not daily:
        plt.figure(figsize=(14, 4.5))
        plt.title("Articles per Day (no data)")
        plt.xlabel("Date")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close()
        return

    df = pd.DataFrame(daily)
    # 날짜 파싱(유연) + 정렬
    df["date"] = pd.to_datetime(df["date"], errors="coerce", infer_datetime_format=True, utc=False)
    df = df.dropna(subset=["date"]).sort_values("date")

    from datetime import datetime
    now_year = datetime.now().year
    y_min, y_max = now_year - 3, now_year + 1
    df = df[(df["date"].dt.year >= y_min) & (df["date"].dt.year <= y_max)]

    df["count"] = pd.to_numeric(df.get("count", 0), errors="coerce").fillna(0).astype(int)

    if df.empty:
        plt.figure(figsize=(14, 4.5))
        plt.title("Articles per Day (empty after filtering)")
        plt.xlabel("Date")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close()
        return

    # 연속 날짜 인덱스 + 빈 날 0 채우기
    df = df.set_index("date")
    full_idx = pd.date_range(df.index.min().normalize(), df.index.max().normalize(), freq="D")
    df = df.reindex(full_idx).fillna(0)
    df.index.name = "date"
    df["count"] = df["count"].astype(int)

    # 이동평균
    df["ma"] = df["count"].rolling(window=window, min_periods=1).mean()

    # 최근 N일만
    if focus_days and len(df) > focus_days:
        df = df.iloc[-focus_days:]

    # 플롯
    plt.figure(figsize=(14, 4.5))
    plt.plot(df.index, df["count"], label="Daily", color="tab:blue", linewidth=1.6)
    plt.plot(df.index, df["ma"], label=f"{window}d MA", color="tab:orange", linewidth=1.6)

    ax = plt.gca()
    locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    dmin = df.index.min().strftime("%Y-%m-%d")
    dmax = df.index.max().strftime("%Y-%m-%d")
    plt.title(f"Articles per Day ({dmin} ~ {dmax})")
    plt.xlabel("Date")
    plt.ylabel("Count")
    plt.legend(loc="upper right")
    plt.grid(alpha=0.25, linestyle="--", linewidth=0.6)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()

    # 디버그 프리뷰 남기기
    try:
        ensure_dir("outputs/debug")
        df.reset_index().rename(columns={"index": "date"}) \
          .to_csv("outputs/debug/timeseries_preview.csv", index=False, encoding="utf-8")
        print("[INFO] timeseries preview saved: outputs/debug/timeseries_preview.csv")
    except Exception:
        pass


def ensure_timeseries_png(ts_dict: Dict[str, Any], out_path="outputs/fig/timeseries.png"):
    # 이미 있으면 절대 재생성하지 않음(덮어쓰기 금지)
    if os.path.exists(out_path):
        print("[INFO] timeseries.png exists -> skip regenerate")
        return
    plot_timeseries(ts_dict, out_path=out_path)


# ---------------- 플롯: 키워드/토픽 ----------------
def plot_top_keywords(keywords_obj: Dict[str, Any], out_path="outputs/fig/top_keywords.png", topn=20):
    ensure_dir(os.path.dirname(out_path))
    items = (keywords_obj or {}).get("keywords") or []
    if not items:
        # 빈 그래프라도 저장
        plt.figure(figsize=(10, 6))
        plt.title("Top Keywords (no data)")
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close()
        return

    df = pd.DataFrame(items)
    if "keyword" not in df.columns or "score" not in df.columns:
        df = df.rename(columns={"word": "keyword", "weight": "score"})
    df = df.dropna(subset=["keyword"]).copy()
    df["score"] = pd.to_numeric(df.get("score", 0), errors="coerce").fillna(0)
    df = df.sort_values("score", ascending=False).head(topn)

    plt.figure(figsize=(10, 6))
    plt.barh(df["keyword"][::-1], df["score"][::-1], color="tab:blue")
    plt.title("Top Keywords")
    plt.xlabel("Score")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()


def plot_topics(topics_obj: Dict[str, Any], out_path="outputs/fig/topics.png", max_topics=6, words_per_topic=6):
    ensure_dir(os.path.dirname(out_path))
    topics = (topics_obj or {}).get("topics") or []
    if not topics:
        plt.figure(figsize=(10, 6))
        plt.title("Topics (no data)")
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close()
        return

    # 간단히: 각 토픽 상위 단어들을 하나의 막대그래프로 나열
    rows = []
    for t in topics[:max_topics]:
        tid = t.get("topic_id")
        for w in (t.get("top_words") or [])[:words_per_topic]:
            rows.append({"topic": f"T{tid}", "word": w.get("word", ""), "prob": float(w.get("prob", 0.0))})
    if not rows:
        plt.figure(figsize=(10, 6))
        plt.title("Topics (no data)")
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close()
        return

    df = pd.DataFrame(rows)
    # 토픽별로 그룹 막대
    pivot = df.pivot_table(index="word", columns="topic", values="prob", fill_value=0)
    # 상위 단어 선별(전체 기여도 기준)
    pivot["sum"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("sum", ascending=True).drop(columns=["sum"]).tail(max_topics * 4)

    plt.figure(figsize=(12, 7))
    pivot.plot(kind="barh", ax=plt.gca(), stacked=False, alpha=0.85)
    plt.title("Topics · Top Words")
    plt.xlabel("Probability")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()


# ---------------- 마크다운/HTML ----------------
def build_markdown(keywords_obj: Dict[str, Any],
                   topics_obj: Dict[str, Any],
                   ts_obj: Dict[str, Any],
                   insights_obj: Dict[str, Any],
                   opp_obj: Dict[str, Any]) -> str:
    """
    섹션 순서(기준선):
    Weekly/New Biz Report → Executive Summary → Key Metrics → Top Keywords → Topics → Trend → Insights → Opportunities (Top 5) → Appendix
    이미지 경로는 fig/… 상대경로로 고정
    """
    # Key Metrics
    daily = (ts_obj or {}).get("daily", [])
    n_days = len(daily)
    total_cnt = sum(int(x.get("count", 0)) for x in daily)
    date_range = "-"
    if n_days > 0:
        date_range = f"{daily[0].get('date','?')} ~ {daily[-1].get('date','?')}"

    # Top Keywords (표)
    kw_items = (keywords_obj or {}).get("keywords") or []
    kw_rows = []
    for i, it in enumerate(sorted(kw_items, key=lambda x: x.get("score", 0), reverse=True), 1):
        if i > 20:
            break
        kw_rows.append(f"| {i} | {it.get('keyword','')} | {round(float(it.get('score', 0)), 4)} |")
    if not kw_rows:
        kw_rows.append("| - | - | - |")

    # Opportunities (표)
    ideas = (opp_obj or {}).get("ideas") or []
    opp_rows = []
    for it in ideas[:5]:
        opp_rows.append(f"| {it.get('title') or it.get('idea') or ''} | {it.get('target_customer','')} | {it.get('value_prop','')} | {it.get('score') or it.get('priority_score') or 0} |")
    if not opp_rows:
        opp_rows.append("| (데이터 없음) | | | |")

    # Insights
    summary = (insights_obj or {}).get("summary") or "_요약 없음_"

    md = []
    md.append("# Weekly/New Biz Report\n")
    md.append("## Executive Summary\n")
    md.append(summary + "\n")

    md.append("## Key Metrics\n")
    md.append(f"- 기간: {date_range}\n- 총 기사 수: {total_cnt}\n- 집계 일수: {n_days}\n")

    md.append("## Top Keywords\n")
    md.append("![Top Keywords](fig/top_keywords.png)\n\n")
    md.append("| Rank | Keyword | Score |\n|---:|---|---:|\n")
    md.append("\n".join(kw_rows) + "\n")

    md.append("## Topics\n")
    md.append("![Topics](fig/topics.png)\n\n")

    md.append("## Trend\n")
    md.append("![Articles per Day](fig/timeseries.png)\n\n")

    md.append("## Insights\n")
    md.append(summary + "\n")

    md.append("## Opportunities (Top 5)\n")
    md.append("| Title | Target | Value Prop | Score |\n|---|---|---|---:|\n")
    md.append("\n".join(opp_rows) + "\n")

    md.append("## Appendix\n")
    md.append("- 데이터: keywords.json, topics.json, trend_timeseries.json, trend_insights.json, biz_opportunities.json\n")

    return "\n".join(md)


def md_to_html(md_text: str) -> str:
    # 기본 마크다운 → HTML 변환
    body = markdown(md_text, output_format="html5", extensions=["tables"])
    # 심플 템플릿
    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>Weekly/New Biz Report</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans KR", "Apple SD Gothic Neo", "맑은 고딕", Arial, sans-serif; line-height: 1.6; padding: 24px; color: #222; }}
  h1, h2, h3 {{ margin-top: 1.4em; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ddd; padding: 6px 8px; }}
  th {{ background: #f5f5f7; }}
  img {{ max-width: 100%; height: auto; }}
  code {{ background: #f6f8fa; padding: 2px 4px; border-radius: 4px; }}
</style>
</head>
<body>
{body}
</body>
</html>"""
    return html


# ---------------- 메인 ----------------
def main():
    out_dir = "outputs"
    fig_dir = os.path.join(out_dir, "fig")
    ensure_dir(fig_dir)

    # 데이터 로드
    keywords = read_json(os.path.join(out_dir, "keywords.json"), default={"keywords": []})
    topics = read_json(os.path.join(out_dir, "topics.json"), default={"topics": []})
    ts = read_json(os.path.join(out_dir, "trend_timeseries.json"), default={"daily": []})
    insights = read_json(os.path.join(out_dir, "trend_insights.json"), default={"summary": ""})
    opps = read_json(os.path.join(out_dir, "biz_opportunities.json"), default={"ideas": []})

    # 시각화 생성
    plot_top_keywords(keywords, out_path=os.path.join(fig_dir, "top_keywords.png"))
    plot_topics(topics, out_path=os.path.join(fig_dir, "topics.png"))

    # timeseries.png는 있으면 스킵, 없으면 생성
    ensure_timeseries_png(ts, out_path=os.path.join(fig_dir, "timeseries.png"))

    # 마크다운/HTML 생성
    md_text = build_markdown(keywords, topics, ts, insights, opps)
    save_text(os.path.join(out_dir, "report.md"), md_text)

    html = md_to_html(md_text)
    # 필요 시 이미지 인라인(기본 꺼짐)
    if os.getenv("INLINE_IMAGES", "").lower() in ("1", "true", "yes", "y"):
        html = inline_images(html, root_dir=out_dir)
    save_text(os.path.join(out_dir, "report.html"), html)

    # 요약 로그
    img_ok = all(os.path.exists(os.path.join(fig_dir, p)) for p in ["top_keywords.png", "topics.png", "timeseries.png"])
    print(f"[INFO] SUMMARY | E | report={'ok' if os.path.exists(os.path.join(out_dir,'report.html')) else 'fail'} images={img_ok} elapsed=?s")


if __name__ == "__main__":
    main()

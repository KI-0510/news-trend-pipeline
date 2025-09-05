# -*- coding: utf-8 -*-

import os
import json
import glob
import re
import shutil
import datetime as dt
from typing import List, Dict, Any

from utils import (
    log_info, log_warn, log_error, abort,
    call_with_retry, http_get_with_retry, json_from_response
)

# ---------- 공통 로더 ----------
def load_json(path: str, default=None):
    if default is None:
        default = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def latest(globpat: str):
    files = sorted(glob.glob(globpat))
    return files[-1] if files else None

# ---------- 데이터 준비 ----------
def load_data():
    keywords = load_json("outputs/keywords.json", {"keywords": [], "stats": {}})
    topics   = load_json("outputs/topics.json", {"topics": []})
    ts       = load_json("outputs/trend_timeseries.json", {"daily": []})
    insights = load_json("outputs/trend_insights.json", {"summary": "", "top_topics": [], "evidence": []})
    opps     = load_json("outputs/biz_opportunities.json", {"ideas": []})

    meta_path = latest("data/news_meta_*.json")
    meta_items = load_json(meta_path, []) if meta_path else []

    return keywords, topics, ts, insights, opps, meta_items

# ---------- 폰트/플레이스홀더 ----------
def ensure_fonts():
    import matplotlib
    from matplotlib import font_manager
    candidates = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]
    font_path = next((p for p in candidates if os.path.exists(p)), None)
    if font_path:
        font_manager.fontManager.addfont(font_path)
        font_name = font_manager.FontProperties(fname=font_path).get_name()
    else:
        font_name = "NanumGothic"

    matplotlib.rcParams["font.family"] = font_name
    matplotlib.rcParams["font.sans-serif"] = [font_name, "NanumGothic", "Noto Sans KR", "Malgun Gothic", "AppleGothic", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    try:
        from matplotlib import font_manager as fm
        fm._rebuild()
    except Exception:
        pass
    return font_name

def ensure_placeholder_image(path: str, text: str):
    if os.path.exists(path):
        return
    import matplotlib.pyplot as plt
    ensure_fonts()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.text(0.5, 0.5, text, ha="center", va="center")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log_warn("플레이스홀더 이미지 생성", path=path)

# ---------- 플롯(키워드/토픽) ----------
def plot_top_keywords(keywords: Dict[str, Any], out_path="outputs/fig/top_keywords.png", topn=15):
    import matplotlib.pyplot as plt
    import seaborn as sns
    ensure_fonts()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    data = (keywords.get("keywords") or [])[:topn]
    if not data:
        plt.figure(figsize=(8, 5))
        plt.text(0.5, 0.5, "키워드 데이터 없음", ha="center")
        plt.axis("off")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        return
    labels = [d.get("keyword", "") for d in data][::-1]
    scores = [d.get("score", 0) for d in data][::-1]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=scores, y=labels, color="#3b82f6")
    plt.title("Top Keywords")
    plt.xlabel("Score")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_topics(topics: Dict[str, Any], out_path="outputs/fig/topics.png", topn_words=6):
    import math
    import matplotlib.pyplot as plt
    import seaborn as sns
    ensure_fonts()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tps = topics.get("topics") or []
    if not tps:
        plt.figure(figsize=(8, 5))
        plt.text(0.5, 0.5, "토픽 데이터 없음", ha="center")
        plt.axis("off")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        return
    k = len(tps)
    cols = 2
    rows = math.ceil(k / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axes = axes.flatten() if k > 1 else [axes]
    for i, t in enumerate(tps):
        ax = axes[i]
        words = (t.get("top_words") or [])[:topn_words]
        labels = [w.get("word", "") for w in words][::-1]
        probs  = [w.get("prob", 0.0) for w in words][::-1]
        sns.barplot(x=probs, y=labels, ax=ax, color="#10b981")
        ax.set_title(f"Topic #{t.get('topic_id')}")
        ax.set_xlabel("Prob.")
        ax.set_ylabel("")
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_timeseries(ts: Dict[str, Any], out_path="outputs/fig/timeseries.png", overwrite=False):
    """
    기본값(overwrite=False): module_c 산출물 보존
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    ensure_fonts()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path) and not overwrite:
        log_info("timeseries.png 이미 존재 → 덮어쓰기 생략", path=out_path)
        return
    daily = ts.get("daily") or []
    if not daily:
        plt.figure(figsize=(10, 4))
        plt.text(0.5, 0.5, "데이터 없음", ha="center")
        plt.axis("off")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        return
    idx = pd.to_datetime([d.get("date") for d in daily])
    vals = [int(d.get("count", 0)) for d in daily]
    s = pd.Series(vals, index=idx).sort_index()
    ma7 = s.rolling(window=7, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.plot(s.index, s.values, "-o", color="#1f77b4", linewidth=1.8, markersize=3, label="Daily")
    ax.plot(ma7.index, ma7.values, "-", color="#ff7f0e", linewidth=1.6, label="7d MA")
    if s.index.min() == s.index.max():
        start = s.index.min() - pd.Timedelta(days=3)
        end   = s.index.max() + pd.Timedelta(days=3)
    else:
        start = s.index.min() - pd.Timedelta(days=1)
        end   = s.index.max() + pd.Timedelta(days=1)
    ax.set_xlim(start, end)
    ymax = max(5, int(s.max() * 1.2))
    ax.set_ylim(0, ymax)
    ax.set_title("Articles per Day")
    ax.set_xlabel("Date")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    log_info("timeseries 렌더링 완료(모듈E)", path=out_path, overwrite=overwrite)

# ---------- 요구 이미지 확보 ----------
def ensure_required_images():
    # keyword_network.png 없으면 플레이스홀더 생성
    ensure_placeholder_image("outputs/fig/keyword_network.png", "keyword_network.png")
    # Trend 섹션은 fig/timeseries.png 를 사용(경로만 'fig/'로 쓸 거라, 실제 파일은 outputs/fig/timeseries.png 그대로 두면 됨)

# ---------- Key Metrics ----------
def make_key_metrics(ts: Dict[str, Any],
                     keywords: Dict[str, Any],
                     topics: Dict[str, Any],
                     opps: Dict[str, Any]) -> List[str]:
    daily = ts.get("daily") or []
    if daily:
        dates = [d.get("date") for d in daily if d.get("date")]
        counts = [int(d.get("count", 0)) for d in daily]
        date_min = dates[0]
        date_max = dates[-1]
        days = len(dates)
        total = sum(counts)
        avg = round(total / max(1, days), 2)
        maxv = max(counts) if counts else 0
        minv = min(counts) if counts else 0
    else:
        date_min = date_max = "-"
        days = 0
        total = avg = maxv = minv = 0

    tp_cnt = len(topics.get("topics") or [])
    kw_cnt = len(keywords.get("keywords") or [])
    idea_cnt = len(opps.get("ideas") or [])

    # 샘플 HTML 톤에 맞춘 간단 지표 4~6개
    return [
        f"문서 수: {total}",
        f"키워드 수(상위): {min(15, kw_cnt)}",
        f"토픽 수: {tp_cnt}",
        f"시계열 데이터 일자 수: {days}",
    ]

# ---------- 제목 날짜 ----------
def pick_report_date(ts: Dict[str, Any]) -> str:
    daily = ts.get("daily") or []
    if daily:
        return daily[-1].get("date") or dt.datetime.utcnow().strftime("%Y-%m-%d")
    return dt.datetime.utcnow().strftime("%Y-%m-%d")

# ---------- 리포트 생성 ----------
def build_report_md(keywords: Dict[str, Any],
                    topics: Dict[str, Any],
                    ts: Dict[str, Any],
                    insights: Dict[str, Any],
                    opps: Dict[str, Any],
                    out_md: str = "outputs/report.md"):
    os.makedirs(os.path.dirname(out_md), exist_ok=True)

    report_date = pick_report_date(ts)
    daily = ts.get("daily", [])
    # Top Keywords 테이블용 데이터
    kw_list = (keywords.get("keywords") or [])[:15]
    # Topics 리스트용 단어 샘플
    topic_items = topics.get("topics") or []

    lines: List[str] = []
    # Title
    lines.append(f"# Weekly/New Biz Report ({report_date})")
    lines.append("")

    # Executive Summary
    lines.append("## Executive Summary")
    summary_txt = (insights.get("summary") or "").strip()
    if not summary_txt:
        summary_txt = "_요약 없음_"
    # 한 줄 안내(샘플과 톤 맞춤)
    lines.append("- 이번 기간 핵심 토픽과 키워드, 주요 시사점을 요약합니다.")
    lines.append("")
    lines.append(summary_txt)
    lines.append("")

    # Key Metrics
    lines.append("## Key Metrics")
    for m in make_key_metrics(ts, keywords, topics, opps):
        lines.append(f"- {m}")
    lines.append("")

    # Top Keywords (표 + 이미지 2개)
    lines.append("## Top Keywords")
    lines.append("")
    lines.append("| Rank | Keyword | Score |")
    lines.append("|---:|---|---:|")
    if not kw_list:
        lines.append("| 1 | (없음) | 0 |")
    else:
        for i, it in enumerate(kw_list, 1):
            k = it.get("keyword", "")
            s = it.get("score", 0)
            lines.append(f"| {i} | {k} | {s:.3f} |")
    lines.append("")
    # 이미지 경로는 fig/… (report.html이 outputs/에 있으므로 상대경로)
    lines.append("![Top Keywords](fig/top_keywords.png)")
    lines.append("![Keyword Network](fig/keyword_network.png)")
    lines.append("")

    # Topics (리스트 + 이미지)
    lines.append("## Topics")
    if not topic_items:
        lines.append("- (토픽 없음)")
    else:
        for t in topic_items:
            tid = t.get("topic_id")
            words = [w.get("word", "") for w in (t.get("top_words") or [])][:6]
            lines.append(f"- Topic #{tid}: " + ", ".join(words))
    lines.append("")
    lines.append("![Topics](fig/topics.png)")
    lines.append("")

    # Trend (timeseries.png)
    lines.append("## Trend")
    lines.append("- 최근 14~30일 기사 수 추세와 7일 이동평균선을 제공합니다.")
    lines.append("")
    lines.append("![Timeseries](fig/timeseries.png)")
    lines.append("")

    # Insights (evidence만 간단 노출)
    lines.append("## Insights")
    ev = insights.get("evidence") or []
    if isinstance(ev, list) and ev:
        for e in ev[:6]:
            lines.append(f"- {e}")
    else:
        lines.append("- _추가 인사이트 없음_")
    lines.append("")

    # Opportunities (Top 5) — 표 헤더 일치
    lines.append("## Opportunities (Top 5)")
    lines.append("")
    lines.append("| Idea | Target | Value Prop | Score |")
    lines.append("|---|---|---|---:|")
    ideas = (opps.get("ideas") or [])[:5]
    if not ideas:
        lines.append("| (없음) | - | - | 0 |")
    else:
        for it in ideas:
            idea = (it.get("title") or it.get("idea") or "").replace("|", "-")
            target = (it.get("target") or "").replace("|", "-")
            value = (it.get("desc") or it.get("description") or "").replace("\n", " ").replace("|", "-")
            score = it.get("score", 0)
            lines.append(f"| {idea} | {target} | {value} | {score} |")
    lines.append("")

    # Appendix
    lines.append("## Appendix")
    nowz = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    lines.append(f"- 생성시각: {nowz}")
    lines.append("")

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log_info("report.md 생성 완료", path=out_md)

def build_report_html(md_path="outputs/report.md", out_html="outputs/report.html"):
    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    try:
        with open(md_path, "r", encoding="utf-8") as f:
            md = f.read()
    except Exception:
        md = "# 리포트 없음\n\nreport.md 파일을 찾을 수 없습니다."

    def md_to_html(text: str) -> str:
        html_lines = []
        for line in text.splitlines():
            if line.startswith("# "):
                html_lines.append(f"<h1>{line[2:].strip()}</h1>")
            elif line.startswith("## "):
                html_lines.append(f"<h2>{line[3:].strip()}</h2>")
            elif line.startswith("### "):
                html_lines.append(f"<h3>{line[4:].strip()}</h3>")
            elif line.startswith("![") and "](" in line and line.endswith(")"):
                try:
                    alt = line.split("![", 1)[1].split("]", 1)[0]
                    src = line.split("](", 1)[1][:-1]
                    # report.html과 동일 폴더(outputs) 기준 상대경로 fig/… 유지
                    html_lines.append(f'<div><img src="{src}" alt="{alt}" style="max-width:100%;height:auto;"></div>')
                except Exception:
                    html_lines.append(f"<p>{line}</p>")
            elif line.startswith("|"):
                html_lines.append(f"<p>{line}</p>")
            elif line.startswith("- "):
                html_lines.append(f"<p>{line}</p>")
            else:
                if line.strip() == "":
                    html_lines.append("<br>")
                else:
                    html_lines.append(f"<p>{line}</p>")
        return "\n".join(html_lines)

    body = md_to_html(md)
    html = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>Auto Report</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Noto Sans KR', sans-serif; line-height:1.6; padding:24px; }}
img {{ max-width:100%; height:auto; }}
table {{ border-collapse: collapse; width:100%; }}
th,td {{ border:1px solid #ddd; padding:8px; }}
th {{ background:#f7f7f7; }}
code {{ background:#f1f5f9; padding:2px 4px; border-radius:4px; }}
</style>
</head>
<body>
{body}
</body>
</html>"""
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    log_info("report.html 생성 완료", path=out_html)

# ---------- 엔트리 ----------
if __name__ == "__main__":
    keywords, topics, ts, insights, opps, meta_items = load_data()

    os.makedirs("outputs/fig", exist_ok=True)

    # 키워드/토픽 그림(없으면 생성)
    if not os.path.exists("outputs/fig/top_keywords.png"):
        plot_top_keywords(keywords)
    if not os.path.exists("outputs/fig/topics.png"):
        plot_topics(topics)

    # timeseries는 module C 산출물 유지
    from pathlib import Path
    if not Path("outputs/fig/timeseries.png").exists():
        # 없으면 생성만 시도
        plot_timeseries(ts, overwrite=True)
    else:
        plot_timeseries(ts, overwrite=False)

    # 요구 이미지 확보(플레이스홀더 포함)
    ensure_required_images()

    # 리포트 생성
    build_report_md(keywords, topics, ts, insights, opps, out_md="outputs/report.md")
    build_report_html(md_path="outputs/report.md", out_html="outputs/report.html")

    log_info("Module E 완료")

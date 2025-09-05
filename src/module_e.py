# -*- coding: utf-8 -*-

import os
import json
import glob
import re
import datetime as dt
from pathlib import Path
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
    # evidence는 리스트가 맞음
    insights = load_json("outputs/trend_insights.json", {"summary": "", "top_topics": [], "evidence": []})
    opps     = load_json("outputs/biz_opportunities.json", {"ideas": []})

    meta_path = latest("data/news_meta_*.json")
    meta_items = load_json(meta_path, []) if meta_path else []

    return keywords, topics, ts, insights, opps, meta_items

# ---------- 간단 토크나이저 ----------
def simple_tokenize_ko(text: str):
    toks = re.findall(r"[가-힣A-Za-z0-9]+", text or "")
    toks = [t.lower() for t in toks if len(t) >= 2]
    return toks

def build_docs_from_meta(meta_items):
    docs = []
    for it in meta_items:
        title = (it.get("title") or it.get("title_og") or "").strip()
        desc  = (it.get("description") or it.get("description_og") or "").strip()
        doc = (title + " " + desc).strip()
        if doc:
            docs.append(doc)
    return docs

# ---------- 시각화 ----------
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
    matplotlib.rcParams["font.sans-serif"] = [font_name, "NanumGothic", "Noto Sans CJK KR", "Malgun Gothic", "AppleGothic", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    try:
        from matplotlib import font_manager as fm
        fm._rebuild()
    except Exception:
        pass

    return font_name

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
    기본값(overwrite=False): module_c가 생성한 timeseries.png를 보존.
    새로 그리고 싶으면 overwrite=True 또는 out_path를 다른 이름으로 지정.
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

# ---------- 리포트 생성 ----------
def build_report_md(keywords: Dict[str, Any],
                    topics: Dict[str, Any],
                    ts: Dict[str, Any],
                    insights: Dict[str, Any],
                    opps: Dict[str, Any],
                    out_md: str = "outputs/report.md"):
    os.makedirs(os.path.dirname(out_md), exist_ok=True)

    daily = ts.get("daily", [])
    date_min = daily[0]["date"] if daily else "-"
    date_max = daily[-1]["date"] if daily else "-"
    total_days = len(daily)

    ideas = opps.get("ideas") or []
    top_kw = [k.get("keyword", "") for k in (keywords.get("keywords") or [])][:10]
    top_topics = []
    for t in (topics.get("topics") or [])[:3]:
        words = [w.get("word", "") for w in (t.get("top_words") or [])][:6]
        top_topics.append(f"Topic#{t.get('topic_id')}: " + ", ".join(words))

    lines: List[str] = []
    lines.append("# 뉴스 트렌드 리포트")
    lines.append("")
    lines.append(f"- 기간: {date_min} ~ {date_max} (총 {total_days}일)")
    lines.append(f"- 작성시각: {dt.datetime.utcnow().isoformat(timespec='seconds')}Z")
    lines.append("")
    lines.append("## Articles per Day")
    lines.append("![Articles per Day](outputs/fig/timeseries.png)")
    lines.append("")
    lines.append("## Top Keywords")
    lines.append("![Top Keywords](outputs/fig/top_keywords.png)")
    lines.append("")
    lines.append("## Topics")
    lines.append("![Topics](outputs/fig/topics.png)")
    lines.append("")
    lines.append("## Insight Summary")
    lines.append((insights.get("summary") or "").strip() or "_요약 없음_")
    lines.append("")

    ev = insights.get("evidence") or []
    if isinstance(ev, list) and ev:
        lines.append("### Evidence")
        for e in ev[:6]:
            lines.append(f"- {e}")
        lines.append("")

    if ideas:
        lines.append("## Biz Opportunities")
        for i, it in enumerate(ideas[:10], 1):
            title = it.get("title") or f"Idea {i}"
            desc  = (it.get("desc") or it.get("description") or "").strip()
            lines.append(f"- {i}. {title}")
            if desc:
                lines.append(f"  - {desc}")
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

    # 초간단 마크다운 → HTML 변환
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
                    html_lines.append(f'<div><img src="{src}" alt="{alt}" style="max-width: 100%; height: auto;"></div>')
                except Exception:
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
<title>뉴스 트렌드 리포트</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body{{font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans KR", Arial, "Apple SD Gothic Neo", "Malgun Gothic", "Noto Sans CJK KR", sans-serif; line-height:1.5; max-width: 960px; margin: 24px auto; padding: 0 16px;}}
img{{border:1px solid #eee; padding:6px; background:#fff;}}
h1,h2,h3{{margin-top: 1.2em;}}
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

    # 플롯 생성
    plot_top_keywords(keywords)
    plot_topics(topics)
    # timeseries는 기본적으로 덮어쓰지 않음 → module_c 결과 보존
    plot_timeseries(ts, overwrite=False)

    # 리포트 생성
    build_report_md(keywords, topics, ts, insights, opps, out_md="outputs/report.md")
    build_report_html(md_path="outputs/report.md", out_html="outputs/report.html")

    log_info("Module E 완료")

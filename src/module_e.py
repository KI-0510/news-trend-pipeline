# -*- coding: utf-8 -*-

import os
import json
import glob
import re
import shutil
import datetime as dt
from typing import List, Dict, Any

from utils import (
    log_info, log_warn
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
    # insights: evidence는 리스트로 가정
    insights = load_json("outputs/trend_insights.json", {"summary": "", "top_topics": [], "evidence": []})
    opps     = load_json("outputs/biz_opportunities.json", {"ideas": []})

    meta_path = latest("data/news_meta_*.json")
    meta_items = load_json(meta_path, []) if meta_path else []

    return keywords, topics, ts, insights, opps, meta_items

# ---------- 간단 토크나이저/메타 문서 ----------
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

# ---------- 폰트 ----------
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

# ---------- 플롯 ----------
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
    기본값(overwrite=False): module_c가 만든 timeseries.png 보존
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

# ---------- 이미지/필드 유틸 ----------
def file_exists(path: str) -> bool:
    try:
        return os.path.exists(path)
    except Exception:
        return False

def md_image_if_exists(rel_path: str, alt: str) -> str:
    """
    report.html이 outputs/ 경로에 생성되므로,
    마크다운에는 fig/xxx.png처럼 상대경로를 넣고,
    실제 존재 여부는 outputs/fig/xxx.png로 확인한다.
    """
    abs_path = os.path.join("outputs", rel_path)
    if file_exists(abs_path):
        return f"![{alt}]({rel_path})"
    else:
        log_warn("이미지 없음 → 삽입 생략", path=abs_path)
        return ""

def normalize_idea_row(it: Dict[str, Any]) -> Dict[str, Any]:
    """
    다양한 스키마(title/idea/name, desc/description/value/value_prop, target/segment/...)
    를 너그럽게 수용해 표가 비지 않도록 정규화
    """
    title = it.get("title") or it.get("idea") or it.get("name") or ""
    target = it.get("target") or it.get("target_customer") or it.get("audience") or it.get("segment") or ""
    desc = it.get("desc") or it.get("description") or it.get("value") or it.get("value_prop") or ""
    try:
        score = float(it.get("score") or it.get("priority_score") or 0)
    except Exception:
        score = 0.0

    title = str(title).replace("|", "-").strip()
    target = str(target).replace("|", "-").strip()
    desc = str(desc).replace("\n", " ").replace("|", "-").strip()

    return {"title": title, "target": target, "desc": desc, "score": score}

# ---------- Key Metrics ----------
def make_key_metrics(ts: Dict[str, Any],
                     keywords: Dict[str, Any],
                     topics: Dict[str, Any],
                     opps: Dict[str, Any]) -> List[str]:
    daily = ts.get("daily") or []
    if daily:
        dates = [d.get("date") for d in daily if d.get("date")]
        counts = [int(d.get("count", 0)) for d in daily]
        days = len(dates)
        total = sum(counts)
    else:
        days = 0
        total = 0

    tp_cnt = len(topics.get("topics") or [])
    kw_cnt = len(keywords.get("keywords") or [])

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
    kw_list = (keywords.get("keywords") or [])[:15]
    topic_items = topics.get("topics") or []

    lines: List[str] = []
    # Title
    lines.append(f"# Weekly/New Biz Report ({report_date})")
    lines.append("")

    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("- 이번 기간 핵심 토픽과 키워드, 주요 시사점을 요약합니다.")
    lines.append("")
    summary_txt = (insights.get("summary") or "").strip() or "_요약 없음_"
    lines.append(summary_txt)
    lines.append("")

    # Key Metrics
    lines.append("## Key Metrics")
    for m in make_key_metrics(ts, keywords, topics, opps):
        lines.append(f"- {m}")
    lines.append("")

    # Top Keywords (표 + 이미지)
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
            try:
                s = float(s)
            except Exception:
                s = 0.0
            lines.append(f"| {i} | {k} | {s:.3f} |")
    lines.append("")
    img_kw = md_image_if_exists("fig/top_keywords.png", "Top Keywords")
    if img_kw: lines.append(img_kw)
    img_kw_net = md_image_if_exists("fig/keyword_network.png", "Keyword Network")
    if img_kw_net: lines.append(img_kw_net)
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
    img_topics = md_image_if_exists("fig/topics.png", "Topics")
    if img_topics: lines.append(img_topics)
    lines.append("")

    # Trend
    lines.append("## Trend")
    lines.append("- 최근 14~30일 기사 수 추세와 7일 이동평균선을 제공합니다.")
    lines.append("")
    img_ts = md_image_if_exists("fig/timeseries.png", "Timeseries")
    if img_ts:
        lines.append(img_ts)
    else:
        lines.append("_timeseries 이미지 없음_")
    lines.append("")

    # Insights (요약 → Evidence)
    lines.append("## Insights")
    insight_body = (insights.get("summary") or "").strip()
    lines.append(insight_body if insight_body else "_요약 없음_")
    ev = insights.get("evidence") or []
    if isinstance(ev, list) and ev:
        lines.append("")
        for e in ev[:6]:
            lines.append(f"- {e}")
    lines.append("")

    # Opportunities (Top 5)
    lines.append("## Opportunities (Top 5)")
    lines.append("")
    lines.append("| Idea | Target | Value Prop | Score |")
    lines.append("|---|---|---|---:|")
    ideas_raw = opps.get("ideas") or []
    ideas_norm = [normalize_idea_row(it) for it in ideas_raw]
    ideas_norm.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    top5 = ideas_norm[:5]
    if not top5:
        lines.append("| (없음) | - | - | 0 |")
    else:
        for it in top5:
            lines.append(f"| {it['title']} | {it['target']} | {it['desc']} | {it['score']:.1f} |")
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

    # 타임시리즈는 모듈 C 산출물 유지(없으면 한 번 생성)
    if not os.path.exists("outputs/fig/timeseries.png"):
        plot_timeseries(ts, overwrite=True)
    else:
        plot_timeseries(ts, overwrite=False)

    # 리포트 생성
    build_report_md(keywords, topics, ts, insights, opps, out_md="outputs/report.md")
    build_report_html(md_path="outputs/report.md", out_html="outputs/report.html")

    log_info("Module E 완료")

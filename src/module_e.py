import os
import json
import glob
import re
import datetime as dt
from pathlib import Path
from utils import (
    log_info, log_warn, log_error, abort,
    call_with_retry, http_get_with_retry, json_from_response
)

# ---------- 공통 로더 ----------
def load_json(path, default=None):
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

def plot_top_keywords(keywords, out_path="outputs/fig/top_keywords.png", topn=15):
    import matplotlib.pyplot as plt
    import seaborn as sns
    ensure_fonts()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    data = keywords.get("keywords", [])[:topn]
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

def plot_topics(topics, out_path="outputs/fig/topics.png", topn_words=6):
    import math
    import matplotlib.pyplot as plt
    import seaborn as sns
    ensure_fonts()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    tps = topics.get("topics", [])
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
        import seaborn as sns
        sns.barplot(x=probs, y=labels, ax=ax, color="#10b981")
        ax.set_title(f"Topic #{t.get('topic_id')}")
        ax.set_xlabel("Prob.")
        ax.set_ylabel("")

    # 남는 축 비우기
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_timeseries(ts, out_path="outputs/fig/timeseries.png", overwrite=False):
    """
    - 기본적으로 module_c가 생성한 timeseries.png를 덮어쓰지 않기 위해 overwrite=False.
    - 미리보기용으로 새로 그리고 싶으면 overwrite=True 또는 out_path를 다른 파일명으로 넘겨.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    ensure_fonts()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # 이미 파일이 있고, 덮어쓰지 않도록 설정되면 안전하게 스킵
    if os.path.exists(out_path) and not overwrite:
        log_info("timeseries.png 이미 존재 → 덮어쓰기 생략", path=out_path)
        return

    daily = ts.get("daily", [])
    if not daily:
        plt.figure(figsize=(10, 4))
        plt.text(0.5, 0.5, "데이터 없음", ha="center")
        plt.axis("off")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        return

    # module_c와 동일한 스타일로 렌더링
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

# ---------- 사용 예(선택) ----------
if __name__ == "__main__":
    keywords, topics, ts, insights, opps, meta_items = load_data()
    # 키워드/토픽 그림은 생성
    plot_top_keywords(keywords)
    plot_topics(topics)
    # timeseries는 기본적으로 덮어쓰지 않음(모듈 C 산출물 보존)
    plot_timeseries(ts, overwrite=False)
    log_info("Module E plots done")

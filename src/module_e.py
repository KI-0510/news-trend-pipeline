import os
import json
import glob
import re
import datetime
from pathlib import Path

# ---------- 공통 로더 ----------
def load_json(path, default=None):
    if default is None:
        default ={}
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
    keywords = load_json("outputs/keywords.json", {"keywords": [], "stats":{}})
    topics   = load_json("outputs/topics.json", {"topics":[]})
    ts       = load_json("outputs/trend_timeseries.json", {"daily":[]})
    insights = load_json("outputs/trend_insights.json", {"summary": "", "top_topics": [], "evidence":{}})
    opps     = load_json("outputs/biz_opportunities.json", {"ideas":[]})

    meta_path = latest("data/news_meta_*.json")
    meta_items = load_json(meta_path, []) if meta_path else[]

    return keywords, topics, ts, insights, opps, meta_items

# ---------- 간단 토크나이저 ----------
def simple_tokenize_ko(text: str):
    toks = re.findall(r"[가-힣A-Za-z0-9]+", text or "")
    toks = [t.lower() for t in toks if len(t) >= 2]
    return toks

def build_docs_from_meta(meta_items):
    docs =[]
    for it in meta_items:
        title = (it.get("title") or it.get("title_og") or "").strip()
        desc  = (it.get("description") or it.get("description_og") or "").strip()
        doc = (title + " " + desc).strip()
        if doc:
            docs.append(doc)
    return docs

# ---------- 시각화 ----------
def ensure_fonts():
    import os
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
        # 폰트 파일 경로를 못 찾았을 때의 안전값
        font_name = "NanumGothic"
    
    # rcParams에 확실히 반영
    matplotlib.rcParams["font.family"] = font_name
    matplotlib.rcParams["font.sans-serif"] = [font_name, "NanumGothic", "Noto Sans CJK KR", "Malgun Gothic", "AppleGothic", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    
    # 캐시 재생성(환경 바뀐 경우)
    try:
        from matplotlib import font_manager as fm
        fm._rebuild()
    except Exception:
        pass
    
    return font_name

def plot_wordcloud_from_keywords(keywords_obj, out_path="outputs/fig/wordcloud.png"):
    import os
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt

    ensure_fonts()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    items = (keywords_obj or {}).get("keywords") or []
    if not items:
        # 데이터 없으면 안내 이미지
        plt.figure(figsize=(8, 5))
        plt.text(0.5, 0.5, "워드클라우드 데이터 없음", ha="center", va="center")
        plt.axis("off")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        return

    # 단어:가중치 딕셔너리 만들기
    freqs = {}
    for it in items[:200]:  # 상위 200개까지만
        w = (it.get("keyword") or "").strip()
        s = float(it.get("score", 0) or 0)
        if w:
            freqs[w] = freqs.get(w, 0.0) + max(s, 0.0)

    # 한글 폰트 경로(ensure_fonts에서 등록한 후보)
    candidates = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]
    font_path = next((p for p in candidates if os.path.exists(p)), None)

    wc = WordCloud(
        width=1200, height=600, background_color="white",
        font_path=font_path, colormap="tab20",
        prefer_horizontal=0.9, min_font_size=8, max_words=200,
        normalize_plurals=False
    ).generate_from_frequencies(freqs)

    wc.to_file(out_path)

def plot_top_keywords(keywords, out_path="outputs/fig/top_keywords.png", topn=15):
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    ensure_fonts()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    data = keywords.get("keywords", [])[:topn]
    if not data:
        plt.figure(figsize=(8,5))
        plt.text(0.5,0.5,"키워드 데이터 없음", ha="center")
        plt.axis("off")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        return

    labels = [d["keyword"] for d in data][::-1]
    scores = [d["score"] for d in data][::-1]

    plt.figure(figsize=(10,6))
    sns.barplot(x=scores, y=labels, color="#3b82f6")
    plt.title("Top Keywords")
    plt.xlabel("Score")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_topics(topics, out_path="outputs/fig/topics.png", topn_words=6):
    import os
    import math
    import matplotlib.pyplot as plt
    import seaborn as sns
    ensure_fonts()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    tps = topics.get("topics", [])
    if not tps:
        plt.figure(figsize=(8,5))
        plt.text(0.5,0.5,"토픽 데이터 없음", ha="center")
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
        words = t.get("top_words", [])[:topn_words]
        labels = [w["word"] for w in words][::-1]
        probs = [w["prob"] for w in words][::-1]
        sns.barplot(x=probs, y=labels, ax=ax, color="#10b981")
        ax.set_title(f"Topic #{t.get('topic_id')}")
        ax.set_xlabel("Prob.")
        ax.set_ylabel("")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_timeseries(ts, out_path="outputs/fig/timeseries.png"):
    # 폰트/경로는 기존 방식을 그대로 사용
    import os
    import matplotlib.pyplot as plt
    import pandas as pd
    import matplotlib.dates as mdates
    from datetime import datetime, timedelta

    ensure_fonts()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    daily = ts.get("daily", [])
    if not daily:
        plt.figure(figsize=(10, 5))
        plt.title("Articles per Day (no data)")
        plt.xlabel("Date")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        return

    # 1) DataFrame + 날짜 파싱(유연) + 정렬
    df = pd.DataFrame(daily).copy()
    # YYYY-MM-DD가 대부분이지만 혹시 혼재해도 흡수
    df["date"] = pd.to_datetime(df["date"], errors="coerce", infer_datetime_format=True, utc=False)
    df["count"] = pd.to_numeric(df.get("count", 0), errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=["date"]).sort_values("date")

    # 2) 이상치 연도 제거(현재연도-3 ~ 현재연도+1)
    now_year = datetime.now().year
    y_min, y_max = now_year - 3, now_year + 1
    df = df[(df["date"].dt.year >= y_min) & (df["date"].dt.year <= y_max)]

    if df.empty:
        plt.figure(figsize=(10, 5))
        plt.title("Articles per Day (empty after filtering)")
        plt.xlabel("Date")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        return

    # 3) 연속 날짜 인덱스 만들고 빈 날짜는 0으로 채우기
    df = df.set_index("date")
    full_idx = pd.date_range(df.index.min().normalize(), df.index.max().normalize(), freq="D")
    df = df.reindex(full_idx).fillna(0)
    df.index.name = "date"
    df["count"] = df["count"].astype(int)

    # 1일치만 있는 경우 전용 처리(축 여백 고정 + 값 라벨)
    if len(df) == 1:
        d0 = df.index[0]
        y = float(df["count"].iloc[0])

        plt.figure(figsize=(12, 4.5))
        plt.xlim(d0 - timedelta(days=1), d0 + timedelta(days=1))
        # y축 여백(값에 비례), 0부터 시작
        ypad = max(1, y * 0.15)
        plt.ylim(0, y + ypad)

        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

        plt.plot([d0], [y], marker="o", color="#6366f1", label="Daily")
        # 값 라벨
        plt.annotate(
            f"{int(y)}",
            (d0, y),
            textcoords="offset points",
            xytext=(0, -14),
            ha="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7)
        )

        plt.title(f"Articles per Day ({d0.strftime('%Y-%m-%d')})")
        plt.xlabel("Date")
        plt.ylabel("Count")
        plt.legend(loc="upper right")
        plt.grid(alpha=0.25, linestyle="--", linewidth=0.6)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()

        try:
            os.makedirs("outputs/debug", exist_ok=True)
            df.reset_index().rename(columns={"index": "date"}) \
              .to_csv("outputs/debug/timeseries_preview.csv", index=False, encoding="utf-8")
        except Exception:
            pass
        return

    # 4) 이동평균(7일)
    df["ma7"] = df["count"].rolling(window=7, min_periods=1).mean()

    # 5) 최근 90~120일만 포커스(축 과확장 방지)
    focus_days = 120
    if len(df) > focus_days:
        df = df.iloc[-focus_days:]

    # 6) 플롯
    plt.figure(figsize=(12, 4.5))
    plt.plot(df.index, df["count"], label="Daily", color="#6366f1", linewidth=1.6)
    plt.plot(df.index, df["ma7"], label="7d MA", color="#f59e0b", linewidth=1.6)

    # 날짜 축 포맷러(가독성)
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
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    # 디버그 프리뷰(선택): x축 범위/E2 체크용
    try:
        os.makedirs("outputs/debug", exist_ok=True)
        df.reset_index().rename(columns={"index": "date"}) \
          .to_csv("outputs/debug/timeseries_preview.csv", index=False, encoding="utf-8")
        print("[INFO] timeseries preview saved: outputs/debug/timeseries_preview.csv")
    except Exception:
        pass
    
    # 4) 이동평균(7일)
    df["ma7"] = df["count"].rolling(window=7, min_periods=1).mean()
  
    # 5) 최근 90~120일만 포커스(축 과확장 방지)
    focus_days = 120
    if len(df) > focus_days:
        df = df.iloc[-focus_days:]

    # 6) 플롯
    plt.figure(figsize=(12, 4.5))
    plt.plot(df.index, df["count"], label="Daily", color="#6366f1", linewidth=1.6)
    plt.plot(df.index, df["ma7"], label="7d MA", color="#f59e0b", linewidth=1.6)

    # 날짜 축 포맷러(가독성)
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
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    # 디버그 프리뷰(선택): x축 범위/E2 체크용
    try:
        os.makedirs("outputs/debug", exist_ok=True)
        df.reset_index().rename(columns={"index": "date"}) \
          .to_csv("outputs/debug/timeseries_preview.csv", index=False, encoding="utf-8")
        print("[INFO] timeseries preview saved: outputs/debug/timeseries_preview.csv")
    except Exception:
        pass


def plot_keyword_network(keywords, docs, out_path="outputs/fig/keyword_network.png",
                         topn=50, min_cooccur=2, max_edges=200, label_top=None):
    import os, math, re
    import matplotlib.pyplot as plt
    import seaborn as sns
    import networkx as nx

    # 폰트 적용 + 폰트명 반환
    font_name = ensure_fonts()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # 상위 키워드 집합
    kw = [k["keyword"].lower() for k in (keywords.get("keywords", [])[:topn])]
    vocab = set(kw)

    # 데이터 부족 처리
    if not docs or not vocab:
        plt.figure(figsize=(8, 5))
        plt.text(0.5, 0.5, "키워드 네트워크 생성 불가(데이터 부족)", ha="center", va="center")
        plt.axis("off")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        return {"nodes": 0, "edges": 0}

    # 문서 토큰화(문서 내 중복 토큰 제거)
    def _tok(x: str):
        return re.findall(r"[가-힣A-Za-z0-9]+", x or "")

    token_docs = [[t.lower() for t in set(_tok(d)) if len(t) >= 2] for d in docs]

    # 동시출현/빈도 계산
    from collections import Counter
    co = Counter()
    freq = Counter()
    for toks in token_docs:
        toks_in = [t for t in toks if t in vocab]
        for w in toks_in:
            freq[w] += 1
        for i in range(len(toks_in)):
            for j in range(i + 1, len(toks_in)):
                a, b = sorted((toks_in[i], toks_in[j]))
                co[(a, b)] += 1

    edges = [(a, b, c) for (a, b), c in co.items() if c >= min_cooccur]
    edges = sorted(edges, key=lambda x: x[2], reverse=True)[:max_edges]

    # 그래프 구성
    G = nx.Graph()
    for w, f in freq.items():
        if f > 0 and w in vocab:  # vocab에 포함된 키워드만 노드
            G.add_node(w, freq=f)
    for a, b, c in edges:
        if a in G.nodes and b in G.nodes:
            G.add_edge(a, b, weight=c)

    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        plt.figure(figsize=(8, 5))
        plt.text(0.5, 0.5, "키워드 네트워크 생성 불가(엣지 없음)", ha="center", va="center")
        plt.axis("off")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        return {"nodes": 0, "edges": 0}

    # 커뮤니티 색상
    try:
        comms = list(nx.algorithms.community.greedy_modularity_communities(G))
        comm_map = {}
        for ci, com in enumerate(comms):
            for n in com:
                comm_map[n] = ci
        num_comm = max(comm_map.values()) + 1 if comm_map else 1
    except Exception:
        comm_map = {n: 0 for n in G.nodes()}
        num_comm = 1

    # 레이아웃(간격 조금 넉넉히)
    pos = nx.spring_layout(G, seed=42, k=0.9)

    # 스타일
    palette = sns.color_palette("tab10", n_colors=max(10, num_comm))
    node_sizes = [300 + 50 * math.sqrt(G.nodes[n].get("freq", 1)) for n in G.nodes()]
    node_colors = [palette[comm_map.get(n, 0)] for n in G.nodes()]
    edge_widths = [0.5 + 0.6 * G[u][v]["weight"] for u, v in G.edges()]

    plt.figure(figsize=(10, 7))
    nx.draw_networkx_edges(G, pos, alpha=0.25, width=edge_widths, edge_color="#666")
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                           alpha=0.9, linewidths=0.5, edgecolors="#333")

    # 라벨: label_top=None이면 모든 노드에 라벨
    if label_top is None:
        label_nodes = list(G.nodes())
    else:
        label_nodes = [w for w, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:label_top]]
        label_nodes = [n for n in label_nodes if n in G.nodes()]

    labels = {n: n for n in label_nodes}
    nx.draw_networkx_labels(
        G, pos, labels=labels,
        font_size=8,
        font_family=font_name,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7)
    )

    plt.title("Keyword Co-occurrence Network")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    return {"nodes": G.number_of_nodes(), "edges": G.number_of_edges()}
                             

def _fmt_int(x):
    try:
        return f"{int(x):,}"
    except Exception:
        return str(x)

def _fmt_score(x, nd=3):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)

def _truncate(s, n=80):
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= n else s[:n-1] + "…"
    
# ---------- 리포트 생성 ----------
def build_markdown(keywords, topics, ts, insights, opps, fig_dir="fig", out_md="outputs/report.md"):
    klist = keywords.get("keywords", [])[:15]
    tlist = topics.get("topics", [])
    daily = ts.get("daily", [])
    summary = (insights.get("summary", "") or "").strip()

    # 기간/총 기사 수 계산(표시는 깔끔하게)
    n_days = len(daily)
    total_cnt = sum(int(x.get("count", 0)) for x in daily)
    if n_days > 0:
        date_range = f"{daily[0].get('date','?')} ~ {daily[-1].get('date','?')}"
    else:
        date_range = "-"

    # 오늘 날짜
    today = datetime.datetime.utcnow().strftime("%Y-%m-%d")
    lines = []

    # 타이틀/요약
    lines.append(f"# Weekly/New Biz Report ({today})\n")
    lines.append("## Executive Summary\n")
    lines.append("- 이번 기간 핵심 토픽과 키워드, 주요 시사점을 요약합니다.\n")
    if summary:
        lines.append(summary + "\n")

    # Key Metrics(숫자 포맷 보강)
    lines.append("## Key Metrics\n")
    num_docs = keywords.get("stats", {}).get("num_docs", "N/A")
    num_docs_disp = _fmt_int(num_docs) if isinstance(num_docs, (int, float)) or str(num_docs).isdigit() else str(num_docs)
    lines.append(f"- 기간: {date_range}")
    lines.append(f"- 총 기사 수: {_fmt_int(total_cnt)}")
    lines.append(f"- 문서 수: {num_docs_disp}")
    lines.append(f"- 키워드 수(상위): {len(klist)}")
    lines.append(f"- 토픽 수: {len(tlist)}")
    lines.append(f"- 시계열 데이터 일자 수: {n_days}\n")

    # Top Keywords(정렬 + 점수 자릿수 + 파이프 이스케이프)
    lines.append("## Top Keywords\n")
    lines.append(f"![Word Cloud]({fig_dir}/wordcloud.png)\n")
    if klist:
        # 원본 전체를 점수 기준 정렬 후 표는 상위 15개로 표시
        kw_all = sorted((keywords.get("keywords") or []), key=lambda x: x.get("score", 0), reverse=True)
        lines.append("| Rank | Keyword | Score |")
        lines.append("|---:|---|---:|")
        for i, k in enumerate(kw_all[:15], 1):
            kw = (k.get("keyword", "") or "").replace("|", r"\|")
            sc = _fmt_score(k.get("score", 0), nd=3)
            lines.append(f"| {i} | {kw} | {sc} |")
    else:
        lines.append("- (데이터 없음)")
    lines.append(f"\n![Top Keywords]({fig_dir}/top_keywords.png)\n")
    lines.append(f"![Keyword Network]({fig_dir}/keyword_network.png)\n")

    # Topics(그대로, 안전 접근만)
    lines.append("## Topics\n")
    if tlist:
        for t in tlist:
            words = ", ".join([w.get("word", "") for w in t.get("top_words", [])[:6]])
            lines.append(f"- Topic #{t.get('topic_id')}: {words}")
    else:
        lines.append("- (데이터 없음)")
    lines.append(f"\n![Topics]({fig_dir}/topics.png)\n")

    # Trend(그대로)
    lines.append("## Trend\n")
    lines.append("- 최근 14~30일 기사 수 추세와 7일 이동평균선을 제공합니다.")
    lines.append(f"\n![Timeseries]({fig_dir}/timeseries.png)\n")

    # Insights(그대로)
    lines.append("## Insights\n")
    if summary:
        lines.append(summary + "\n")
    else:
        lines.append("- (요약 없음)\n")

    # Opportunities(점수 정렬 + 텍스트 길이 제한 + 포맷)
    lines.append("## Opportunities (Top 5)\n")
    ideas_all = (opps.get("ideas", []) or [])
    if ideas_all:
        ideas_sorted = sorted(
            ideas_all,
            key=lambda it: float(it.get("priority_score", it.get("score", 0)) or 0),
            reverse=True
        )[:5]
        lines.append("| Idea | Target | Value Prop | Score |")
        lines.append("|---|---|---|---:|")
        for it in ideas_sorted:
            idea = _truncate((it.get('idea', '') or it.get('title', '') or ''), 80).replace("|", r"\|")
            tgt  = _truncate(it.get('target_customer', ''), 40).replace("|", r"\|")
            vp   = _truncate((it.get('value_prop', '') or '').replace("\n", " "), 100).replace("|", r"\|")
            sc_raw = it.get('priority_score', it.get('score', ''))
            if isinstance(sc_raw, (int, float)) or (isinstance(sc_raw, str) and sc_raw.replace('.', '', 1).isdigit()):
                sc = _fmt_score(sc_raw, nd=2)
            else:
                sc = str(sc_raw)
            lines.append(f"| {idea} | {tgt} | {vp} | {sc} |")
    else:
        lines.append("- (아이디어 없음)")

    # Appendix(그대로)
    lines.append("\n## Appendix\n")
    lines.append("- 데이터: keywords.json, topics.json, trend_timeseries.json, trend_insights.json, biz_opportunities.json")

    Path(out_md).parent.mkdir(parents=True, exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def build_html_from_md(md_path="outputs/report.md", out_html="outputs/report.html"):
    try:
        import markdown
        with open(md_path, "r", encoding="utf-8") as f:
            md = f.read()
        html = markdown.markdown(md, extensions=["extra", "tables", "toc"])
        html_tpl = f"""<!doctype html><html lang="ko"><head> <meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"> <title>Auto Report</title> <link rel="preconnect" href="https://fonts.gstatic.com"><style> body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Noto Sans KR', sans-serif; line-height:1.6; padding:24px; }} img {{ max-width:100%; height:auto; }} table {{ border-collapse: collapse; width:100%; }} th,td {{ border:1px solid #ddd; padding:8px; }} th {{ background:#f7f7f7; }} code {{ background:#f1f5f9; padding:2px 4px; border-radius:4px; }} </style></head><body>{html}</body></html>"""
        with open(out_html, "w", encoding="utf-8") as f:
            f.write(html_tpl)
    except Exception as e:
        print("[WARN] HTML 변환 실패:", e)

def export_csvs(ts_obj, keywords_obj, topics_obj, out_dir="outputs/export"):
    import pandas as pd
    import os

    os.makedirs(out_dir, exist_ok=True)

    # timeseries
    daily = (ts_obj or {}).get("daily", [])
    if daily:
        df_ts = pd.DataFrame(daily)
        # 컬럼 보정
        if "date" in df_ts.columns and "count" in df_ts.columns:
            df_ts.to_csv(
                os.path.join(out_dir, "timeseries_daily.csv"),
                index=False,
                encoding="utf-8"
            )

    # keywords top20
    kws = (keywords_obj or {}).get("keywords", [])[:20]
    if kws:
        pd.DataFrame(kws).to_csv(
            os.path.join(out_dir, "keywords_top20.csv"),
            index=False,
            encoding="utf-8"
        )

    # topics top words (토픽별 상위 단어 펼쳐 저장)
    topics = (topics_obj or {}).get("topics", [])
    rows = []
    for t in topics:
        tid = t.get("topic_id")
        for w in (t.get("top_words") or [])[:10]:
            rows.append({
                "topic_id": tid,
                "word": w.get("word", ""),
                "prob": w.get("prob", 0)
            })

    if rows:
        pd.DataFrame(rows).to_csv(
            os.path.join(out_dir, "topics_top_words.csv"),
            index=False,
            encoding="utf-8"
        )



def main():
    keywords, topics, ts, insights, opps, meta_items = load_data()
    os.makedirs("outputs/fig", exist_ok=True)

    try:
        plot_top_keywords(keywords)
    except Exception as e:
        print("[WARN] top_keywords 그림 실패:", e)

    try:
        plot_topics(topics)
    except Exception as e:
        print("[WARN] topics 그림 실패:", e)

    try:
        plot_wordcloud_from_keywords(keywords)
    except Exception as e:
        print("[WARN] wordcloud 생성 실패:", e)
        
    try:
        plot_timeseries(ts)
    except Exception as e:
        print("[WARN] timeseries 그림 실패:", e)

# 네트워크 생성 (module_e.py main 내부)
    try:
        docs = build_docs_from_meta(meta_items)
        kw_list = keywords.get("keywords", [])
        n_kw = len(kw_list)
        label_cap = 25  # 노드 수가 적으면 전부, 많으면 최대 25개 라벨
        plot_keyword_network(
            keywords, docs,
            out_path="outputs/fig/keyword_network.png",
            topn=n_kw,                 # 상위 키워드 수만큼 노드
            min_cooccur=1,             # 데이터 적은 날 안정성↑
            max_edges=200,
            label_top=(None if n_kw <= label_cap else label_cap)
        )
    except Exception as e:
        print("[WARN] 키워드 네트워크 실패:", e)
    
    try:
        export_csvs(ts, keywords, topics)
    except Exception as e:
        print("[WARN] CSV 내보내기 실패:", e)
        
    try:
        build_markdown(keywords, topics, ts, insights, opps)
        build_html_from_md()
    except Exception as e:
        print("[WARN] 리포트 생성 실패:", e)

    print("[INFO] Module E 완료 | report.md, report.html 생성")

if __name__ == "__main__":
    main()

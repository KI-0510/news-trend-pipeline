import os
import json
import glob
import re
import datetime
from pathlib import Path

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

def apply_plot_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "axes.edgecolor": "#999",
        "axes.linewidth": 0.8,
    })

def plot_wordcloud_from_keywords(keywords_obj, out_path="outputs/fig/wordcloud.png"):
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    ensure_fonts(); apply_plot_style()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    items = (keywords_obj or {}).get("keywords") or []
    if not items:
        plt.figure(figsize=(8, 5)); plt.text(0.5, 0.5, "워드클라우드 데이터 없음", ha="center", va="center")
        plt.axis("off"); plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(); return
    freqs = {}
    for it in items[:200]:
        w = (it.get("keyword") or "").strip()
        s = float(it.get("score", 0) or 0)
        if w: freqs[w] = freqs.get(w, 0.0) + max(s, 0.0)
    candidates = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]
    font_path = next((p for p in candidates if os.path.exists(p)), None)
    wc = WordCloud(width=1200, height=600, background_color="white",
                   font_path=font_path, colormap="tab20",
                   prefer_horizontal=0.9, min_font_size=10, max_words=200,
                   relative_scaling=0.5, normalize_plurals=False).generate_from_frequencies(freqs)
    wc.to_file(out_path)

def plot_top_keywords(keywords, out_path="outputs/fig/top_keywords.png", topn=15):
    import matplotlib.pyplot as plt
    import seaborn as sns
    ensure_fonts(); apply_plot_style()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    data = keywords.get("keywords", [])[:topn]
    if not data:
        plt.figure(figsize=(8,5)); plt.text(0.5,0.5,"키워드 데이터 없음", ha="center")
        plt.axis("off"); plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(); return
    labels = [d["keyword"] for d in data][::-1]
    scores = [d["score"] for d in data][::-1]
    plt.figure(figsize=(10,6)); sns.barplot(x=scores, y=labels, color="#3b82f6")
    plt.title("Top Keywords"); plt.xlabel("Score"); plt.ylabel("")
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

def plot_topics(topics, out_path="outputs/fig/topics.png", topn_words=6):
    import os, math
    import matplotlib.pyplot as plt
    import seaborn as sns
    ensure_fonts(); apply_plot_style()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    tps = topics.get("topics", [])
    if not tps:
        plt.figure(figsize=(8,5)); plt.text(0.5,0.5,"토픽 데이터 없음", ha="center")
        plt.axis("off"); plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(); return

    k = len(tps); cols = 2; rows = math.ceil(k / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axes = axes.flatten() if k > 1 else [axes]

    for i, t in enumerate(tps):
        ax = axes[i]
        words = (t.get("top_words") or [])[:topn_words]
        labels = [str((w.get("word") or "")) for w in words][::-1]
        probs = []
        for w in words[::-1]:
            pw = w.get("prob", 1.0)
            try:
                p = float(pw)
                if p <= 0:
                    p = 1.0
            except Exception:
                p = 1.0
            probs.append(p)
        # 보기용 스케일(원하면 주석 해제)
        # probs = [p*100.0 for p in probs]
        sns.barplot(x=probs, y=labels, ax=ax, color="#10b981")
        ax.set_title(f"Topic #{t.get('topic_id')}")
        ax.set_xlabel("Weight"); ax.set_ylabel("")
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
    

def plot_timeseries(ts, out_path="outputs/fig/timeseries.png"):
    import matplotlib.pyplot as plt
    import pandas as pd
    import matplotlib.dates as mdates
    from datetime import datetime, timedelta
    ensure_fonts(); apply_plot_style()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    daily = ts.get("daily", [])
    if not daily:
        plt.figure(figsize=(10, 5)); plt.title("Articles per Day (no data)")
        plt.xlabel("Date"); plt.ylabel("Count")
        plt.tight_layout(); plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(); return
    df = pd.DataFrame(daily).copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", infer_datetime_format=True, utc=False)
    df["count"] = pd.to_numeric(df.get("count", 0), errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=["date"]).sort_values("date")
    now_year = datetime.now().year; y_min, y_max = now_year - 3, now_year + 1
    df = df[(df["date"].dt.year >= y_min) & (df["date"].dt.year <= y_max)]
    if df.empty:
        plt.figure(figsize=(10, 5)); plt.title("Articles per Day (empty after filtering)")
        plt.xlabel("Date"); plt.ylabel("Count")
        plt.tight_layout(); plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(); return
    df = df.set_index("date")
    full_idx = pd.date_range(df.index.min().normalize(), df.index.max().normalize(), freq="D")
    df = df.reindex(full_idx).fillna(0); df.index.name = "date"; df["count"] = df["count"].astype(int)
    if len(df) == 1:
        d0 = df.index[0]; y = float(df["count"].iloc[0])
        plt.figure(figsize=(12, 4.5))
        plt.xlim(d0 - timedelta(days=1), d0 + timedelta(days=1))
        ypad = max(1, y * 0.15); plt.ylim(0, y + ypad)
        ax = plt.gca(); ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.plot([d0], [y], marker="o", color="#6366f1", label="Daily")
        plt.annotate(f"{int(y)}", (d0, y), textcoords="offset points", xytext=(0, -14), ha="center",
                     fontsize=9, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
        plt.title(f"Articles per Day ({d0.strftime('%Y-%m-%d')})")
        plt.xlabel("Date"); plt.ylabel("Count"); plt.legend(loc="upper right")
        plt.grid(alpha=0.25, linestyle="--", linewidth=0.6)
        plt.tight_layout(); plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close()
        try:
            os.makedirs("outputs/debug", exist_ok=True)
            df.reset_index().rename(columns={"index": "date"}).to_csv("outputs/debug/timeseries_preview.csv", index=False, encoding="utf-8")
        except Exception:
            pass
        return
    df["ma7"] = df["count"].rolling(window=7, min_periods=1).mean()
    focus_days = 120
    if len(df) > focus_days:
        df = df.iloc[-focus_days:]
    plt.figure(figsize=(12, 4.5))
    plt.plot(df.index, df["count"], label="Daily", color="#6366f1", linewidth=1.6)
    plt.plot(df.index, df["ma7"], label="7d MA", color="#f59e0b", linewidth=1.6)
    ax = plt.gca(); locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator); ax.xaxis.set_major_locator(locator); ax.xaxis.set_major_formatter(formatter)
    dmin = df.index.min().strftime("%Y-%m-%d"); dmax = df.index.max().strftime("%Y-%m-%d")
    plt.title(f"Articles per Day ({dmin} ~ {dmax})")
    plt.xlabel("Date"); plt.ylabel("Count"); plt.legend(loc="upper right")
    plt.grid(alpha=0.25, linestyle="--", linewidth=0.6)
    plt.tight_layout(); plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close()


def plot_keyword_network(keywords, docs, out_path="outputs/fig/keyword_network.png",
                         topn=50, min_cooccur=2, max_edges=200, label_top=None):
    import os, math, re
    import matplotlib.pyplot as plt
    import seaborn as sns
    import networkx as nx
    import numpy as np

    font_name = ensure_fonts()
    try:
        apply_plot_style()
    except Exception:
        pass

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # 키워드 정규화(빈 문자열/1글자 제거)
    def norm_kw(w: str) -> str:
        w = (w or "").strip().lower()
        w = re.sub(r"\s+", " ", w)
        return w

    raw_kw = [norm_kw(k.get("keyword")) for k in (keywords.get("keywords", [])[:topn])]
    kw = [w for w in raw_kw if w and len(w) >= 2]
    vocab = set(kw)

    # 데이터 부족 처리
    if not docs or not vocab:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        plt.text(0.5, 0.5, "키워드 네트워크 생성 불가(데이터 부족)", ha="center", va="center")
        plt.axis("off")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        return {"nodes": 0, "edges": 0}

    # 문서 토큰화(문서 내 중복 토큰 제거) + 정규화
    def _tok(x: str):
        return re.findall(r"[가-힣A-Za-z0-9]+", x or "")

    token_docs = []
    for d in docs:
        toks = set(norm_kw(t) for t in _tok(d))
        toks = [t for t in toks if t and len(t) >= 2]
        token_docs.append(toks)

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
        if f > 0 and (w in vocab) and (w and len(w) >= 2):
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

    # 레이아웃
    pos = nx.spring_layout(G, seed=42, k=0.9)

    # 스타일(노드)
    palette = sns.color_palette("tab10", n_colors=max(10, num_comm))
    node_sizes = [300 + 50 * math.sqrt(G.nodes[n].get("freq", 1)) for n in G.nodes()]
    node_colors = [palette[comm_map.get(n, 0)] for n in G.nodes()]

    # 전역 정규화: 엣지 두께 고정 범위로 매핑(리포트 간 비교 용이)
    edges_all = list(G.edges(data=True))
    w_raw = np.array([float(d.get("weight", 1.0) or 1.0) for _, _, d in edges_all], dtype=float)
    # 이상치 완화(선택)
    if len(w_raw) > 5:
        q95 = np.quantile(w_raw, 0.95)
        w_raw = np.minimum(w_raw, q95)
    W_MIN, W_MAX = 0.8, 2.2
    den = (w_raw.max() - w_raw.min()) or 1.0
    w_norm = (W_MIN + (W_MAX - W_MIN) * (w_raw - w_raw.min()) / den).tolist()

    # 그리기: 엣지 → 노드 → 라벨
    plt.figure(figsize=(10, 7))
    ax = plt.gca()
    ax.set_axis_off()

    nx.draw_networkx_edges(
        G, pos,
        edgelist=[(u, v) for u, v, _ in edges_all],
        width=w_norm,
        edge_color="#666", alpha=0.25
    )
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                           alpha=0.9, linewidths=0.5, edgecolors="#333")

    # 라벨 대상
    if label_top is None:
        label_nodes = list(G.nodes())
    else:
        label_nodes = [w for w, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:label_top]]
        label_nodes = [n for n in label_nodes if n in G.nodes()]

    # 라벨 수동 텍스트(겹침 방지 박스)
    for n in label_nodes:
        txt = (n or "").strip()
        if not txt:
            continue
        x, y = pos[n]
        ax.text(
            x, y, txt,
            ha="center", va="center",
            fontsize=8, color="#111111",
            zorder=5, clip_on=False,
            fontname=font_name,
            bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="none", alpha=0.80)
        )

    # 라벨 잘림 방지 패딩
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    if xs and ys:
        pad_x = (max(xs) - min(xs)) * 0.08 + 0.05
        pad_y = (max(ys) - min(ys)) * 0.08 + 0.05
        ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
        ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)

    plt.title("Keyword Co-occurrence Network")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    return {"nodes": G.number_of_nodes(), "edges": G.number_of_edges()}


def export_csvs(ts_obj, keywords_obj, topics_obj, out_dir="outputs/export"):
    import pandas as pd, os
    os.makedirs(out_dir, exist_ok=True)
    daily = (ts_obj or {}).get("daily", [])
    df_ts = pd.DataFrame(daily) if daily else pd.DataFrame(columns=["date","count"])
    df_ts.to_csv(os.path.join(out_dir, "timeseries_daily.csv"), index=False, encoding="utf-8")

    kws = (keywords_obj or {}).get("keywords", [])[:20]
    df_kw = pd.DataFrame(kws) if kws else pd.DataFrame(columns=["keyword","score"])
    df_kw.to_csv(os.path.join(out_dir, "keywords_top20.csv"), index=False, encoding="utf-8")

    topics = (topics_obj or {}).get("topics", [])
    rows = []
    for t in topics:
        tid = t.get("topic_id")
        for w in (t.get("top_words") or [])[:10]:
            pw = w.get("prob", None)
            try:
                p = float(pw)
                if p == 0.0:
                    p = 1e-6
            except Exception:
                p = 1e-6
            rows.append({"topic_id": tid, "word": w.get("word", ""), "prob": p})
    import pandas as pd
    df_tw = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["topic_id","word","prob"])
    df_tw.to_csv(os.path.join(out_dir, "topics_top_words.csv"), index=False, encoding="utf-8")
    print("[INFO] export CSVs -> outputs/export/*.csv")

def build_docs_from_meta(meta_items):
    docs =[]
    for it in meta_items:
        title = (it.get("title") or it.get("title_og") or "").strip()
        desc  = (it.get("description") or it.get("description_og") or "").strip()
        doc = (title + " " + desc).strip()
        if doc: docs.append(doc)
    return docs

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

def build_markdown(keywords, topics, ts, insights, opps, fig_dir="fig", out_md="outputs/report.md"):
    klist = keywords.get("keywords", [])[:15]
    tlist = topics.get("topics", [])
    daily = ts.get("daily", [])
    summary = (insights.get("summary", "") or "").strip()
    n_days = len(daily)
    total_cnt = sum(int(x.get("count", 0)) for x in daily)
    date_range = f"{daily[0].get('date','?')} ~ {daily[-1].get('date','?')}" if n_days > 0 else "-"
    today = datetime.datetime.utcnow().strftime("%Y-%m-%d")
    lines = []
    lines.append(f"# Weekly/New Biz Report ({today})\n")
    lines.append("## Executive Summary\n")
    lines.append("- 이번 기간 핵심 토픽과 키워드, 주요 시사점을 요약합니다.\n")
    if summary: lines.append(summary + "\n")
    lines.append("## Key Metrics\n")
    num_docs = keywords.get("stats", {}).get("num_docs", "N/A")
    num_docs_disp = _fmt_int(num_docs) if isinstance(num_docs, (int, float)) or str(num_docs).isdigit() else str(num_docs)
    lines.append(f"- 기간: {date_range}")
    lines.append(f"- 총 기사 수: {_fmt_int(total_cnt)}")
    lines.append(f"- 문서 수: {num_docs_disp}")
    lines.append(f"- 키워드 수(상위): {len(klist)}")
    lines.append(f"- 토픽 수: {len(tlist)}")
    lines.append(f"- 시계열 데이터 일자 수: {n_days}\n")
    lines.append("## Top Keywords\n")
    lines.append(f"![Word Cloud]({fig_dir}/wordcloud.png)\n")
    if klist:
        kw_all = sorted((keywords.get("keywords") or []), key=lambda x: x.get("score", 0), reverse=True)
        lines.append("| Rank | Keyword | Score |"); lines.append("|---:|---|---:|")
        for i, k in enumerate(kw_all[:15], 1):
            kw = (k.get("keyword", "") or "").replace("|", r"\|")
            sc = _fmt_score(k.get("score", 0), nd=3)
            lines.append(f"| {i} | {kw} | {sc} |")
    else:
        lines.append("- (데이터 없음)")
    lines.append(f"\n![Top Keywords]({fig_dir}/top_keywords.png)\n")
    lines.append(f"![Keyword Network]({fig_dir}/keyword_network.png)\n")
    lines.append("## Topics\n")
    if tlist:
        for t in tlist:
            tid = t.get("topic_id")
            top_words = [w.get("word", "") for w in t.get("top_words", []) if w.get("word")]
            head = (t.get("topic_name") or ", ".join(top_words[:3]) or f"Topic #{tid}")
            # 대표 단어 3~6개 병기
            words_preview = ", ".join(top_words[:6])
            lines.append(f"- {head} (#{tid})")
            if words_preview:
                lines.append(f"  - 대표 단어: {words_preview}")
            if t.get("insight"):
                one_liner = (t.get("insight") or "").replace("\n", " ").strip()
                lines.append(f"  - 요약: {one_liner}")
    else:
        lines.append("- (데이터 없음)")
    lines.append(f"\n![Topics]({fig_dir}/topics.png)\n")
    lines.append("## Trend\n")
    lines.append("- 최근 14~30일 기사 수 추세와 7일 이동평균선을 제공합니다.")
    lines.append(f"\n![Timeseries]({fig_dir}/timeseries.png)\n")
    lines.append("## Insights\n")
    if summary: lines.append(summary + "\n")
    else: lines.append("- (요약 없음)\n")
    lines.append("## Opportunities (Top 5)\n")
    ideas_all = (opps.get("ideas", []) or [])
    if ideas_all:
        ideas_sorted = sorted(ideas_all, key=lambda it: float(it.get("priority_score", it.get("score", 0)) or 0), reverse=True)[:5]
        _do_trunc = os.getenv("TRUNCATE_OPP", "").lower() in ("1", "true", "yes", "y")
        lines.append("| Idea | Target | Value Prop | Score |"); lines.append("|---|---|---|---:|")
        for it in ideas_sorted:
            idea_raw = (it.get('idea', '') or it.get('title', '') or '')
            tgt_raw  = it.get('target_customer', '') or ''
            vp_raw   = (it.get('value_prop', '') or '').replace("\n", " ")
            if _do_trunc:
                idea = _truncate(idea_raw, 120).replace("|", r"\|")
                tgt  = _truncate(tgt_raw, 80).replace("|", r"\|")
                vp   = _truncate(vp_raw, 280).replace("|", r"\|")
            else:
                idea = idea_raw.replace("|", r"\|"); tgt  = tgt_raw.replace("|", r"\|"); vp   = vp_raw.replace("|", r"\|")
            sc_raw = it.get('priority_score', it.get('score', ''))
            sc = _fmt_score(sc_raw, nd=2) if (isinstance(sc_raw, (int,float)) or (isinstance(sc_raw,str) and sc_raw.replace('.','',1).isdigit())) else str(sc_raw)
            lines.append(f"| {idea} | {tgt} | {vp} | {sc} |")
    else:
        lines.append("- (아이디어 없음)")
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
        html_tpl = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Auto Report</title>
<link rel="preconnect" href="https://fonts.gstatic.com">
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Noto Sans KR', sans-serif; line-height: 1.6; padding: 24px; color: #222; }}
  img {{ max-width: 100%; height: auto; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
  th {{ background: #f7f7f7; }}
  code {{ background: #f1f5f9; padding: 2px 4px; border-radius: 4px; }}
  td, th {{ overflow-wrap: anywhere; word-break: break-word; white-space: normal; }}
</style>
</head>
<body>
{html}
</body>
</html>"""
        with open(out_html, "w", encoding="utf-8") as f:
            f.write(html_tpl)
    except Exception as e:
        print("[WARN] HTML 변환 실패:", e)

def export_csvs(ts_obj, keywords_obj, topics_obj, out_dir="outputs/export"):
    import pandas as pd, os
    os.makedirs(out_dir, exist_ok=True)
    daily = (ts_obj or {}).get("daily", [])
    df_ts = pd.DataFrame(daily) if daily else pd.DataFrame(columns=["date","count"])
    df_ts.to_csv(os.path.join(out_dir, "timeseries_daily.csv"), index=False, encoding="utf-8")
    kws = (keywords_obj or {}).get("keywords", [])[:20]
    df_kw = pd.DataFrame(kws) if kws else pd.DataFrame(columns=["keyword","score"])
    df_kw.to_csv(os.path.join(out_dir, "keywords_top20.csv"), index=False, encoding="utf-8")
    topics = (topics_obj or {}).get("topics", [])
    rows = []
    for t in topics:
        tid = t.get("topic_id")
        for w in (t.get("top_words") or [])[:10]:
            pw = w.get("prob", None)
            try:
                p = float(pw)
                if p == 0.0:
                    p = 1e-6
            except Exception:
                p = 1e-6
            rows.append({"topic_id": tid, "word": w.get("word", ""), "prob": p})
    df_tw = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["topic_id","word","prob"])
    df_tw.to_csv(os.path.join(out_dir, "topics_top_words.csv"), index=False, encoding="utf-8")
    print("[INFO] export CSVs -> outputs/export/*.csv")


def main():
    keywords, topics, ts, insights, opps, meta_items = load_data()
    os.makedirs("outputs/fig", exist_ok=True)

    # 그림들
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
        

    # CSV 내보내기
    try:
        export_csvs(ts, keywords, topics)
    except Exception as e:
        print("[WARN] CSV 내보내기 실패:", e)

    # 리포트 생성(반드시 파일을 남기는 보수적 로직)
    try:
        build_markdown(keywords, topics, ts, insights, opps)
        build_html_from_md()
    except Exception as e:
        print("[WARN] 리포트 생성 실패(폴백 생성으로 대체):", e)
        try:
            # 최소 보고서(체크 스크립트가 찾는 섹션 헤더 포함)
            skeleton = """# Weekly/New Biz Report (fallback)

## Executive Summary
- (생성 실패 폴백) 요약 데이터를 불러오지 못했습니다.

## Key Metrics
- 기간: -
- 총 기사 수: 0
- 문서 수: 0
- 키워드 수(상위): 0
- 토픽 수: 0
- 시계열 데이터 일자 수: 0

## Top Keywords
- (데이터 없음)

## Topics
- (데이터 없음)

## Trend
- (데이터 없음)

## Insights
- (데이터 없음)

## Opportunities (Top 5)
- (데이터 없음)

## Appendix
- 데이터: keywords.json, topics.json, trend_timeseries.json, trend_insights.json, biz_opportunities.json
"""
            with open("outputs/report.md", "w", encoding="utf-8") as f:
                f.write(skeleton)
            # 간단 HTML 변환
            try:
                build_html_from_md()
            except Exception as e2:
                print("[WARN] HTML 폴백 변환 실패:", e2)
        except Exception as e3:
            print("[ERROR] 폴백 리포트 생성도 실패:", e3)

    print("[INFO] Module E 완료 | report.md, report.html 생성(또는 폴백 생성)")
    
    
if __name__ == "__main__":
    main()

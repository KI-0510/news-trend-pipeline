import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.dates as mdates
import seaborn as sns
import os
import json
import math
import networkx as nx
import re
from pathlib import Path
from wordcloud import WordCloud
import numpy as np
from adjustText import adjust_text
from src.utils import load_json, save_json, latest
from datetime import datetime, timedelta
from pathlib import Path

### 추가 이미지 (timeseries_spikes.png, strong_signals_topbar.png, topics_bubble.png)
FIG_DIR = "outputs/fig"
EXPORT_DIR = "outputs/export"

def _ensure_dirs():
    Path(FIG_DIR).mkdir(parents=True, exist_ok=True)
    Path(EXPORT_DIR).mkdir(parents=True, exist_ok=True)

def _safe_read_csv(path, **kwargs):
    try:
        if os.path.exists(path):
            return pd.read_csv(path, **kwargs)
    except Exception:
        pass
    return pd.DataFrame()

def _parse_date(s):
    try:
        return datetime.strptime(str(s), "%Y-%m-%d")
    except Exception:
        try:
            return pd.to_datetime(s)
        except Exception:
            return None

def _savefig(path, tight=True, dpi=160, facecolor="white"):
    if tight:
        plt.tight_layout()
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=dpi, facecolor=facecolor)
    plt.close()

# 1) timeseries_spikes.png 생성
def gen_timeseries_spikes(
    daily_json_path="outputs/trend_timeseries.json",
    out_timeseries=f"{FIG_DIR}/timeseries.png",
    out_spikes=f"{FIG_DIR}/timeseries_spikes.png",
    out_spike_csv=f"{EXPORT_DIR}/timeseries_spikes.csv",
    ma_window=7, z_thresh=2.5
):
    _ensure_dirs()
    data = {}
    try:
        with open(daily_json_path, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
    except Exception:
        data = {}

    daily = data.get("daily", [])
    if not daily:
        return

    df = pd.DataFrame(daily)
    if "date" not in df.columns or "count" not in df.columns:
        return
    df["date"] = df["date"].apply(_parse_date)
    df = df.dropna(subset=["date"]).sort_values("date")
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0)

    # 기본 시계열
    plt.figure(figsize=(10, 4))
    plt.plot(df["date"], df["count"], label="Daily", color="#2c7be5", alpha=0.85)
    if len(df) >= ma_window:
        df["ma"] = df["count"].rolling(ma_window).mean()
        plt.plot(df["date"], df["ma"], label=f"{ma_window}-day MA", color="#495057")
    plt.title("Daily Articles")
    plt.xlabel("Date"); plt.ylabel("Count")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.legend(loc="upper left")
    _savefig(out_timeseries)

    # 스파이크 탐지
    if df["count"].std() > 0:
        z = (df["count"] - df["count"].mean()) / df["count"].std()
        df["z"] = z
        spikes = df.loc[df["z"] >= z_thresh].copy()
    else:
        spikes = pd.DataFrame()

    # 스파이크 플롯
    plt.figure(figsize=(10, 4))
    plt.plot(df["date"], df["count"], color="#2c7be5", alpha=0.6)
    if not spikes.empty:
        plt.scatter(spikes["date"], spikes["count"], color="#d92550", s=40, label=f"Spikes (z>={z_thresh:.1f})")
        for _, r in spikes.iterrows():
            plt.annotate(r["date"].strftime("%m-%d"), (r["date"], r["count"]), textcoords="offset points", xytext=(0,8), ha="center", fontsize=8)
        plt.legend()
    plt.title("Spikes")
    plt.xlabel("Date"); plt.ylabel("Count")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    _savefig(out_spikes)

    # CSV 저장
    if not spikes.empty:
        out = spikes[["date","count","z"]].copy()
        out["date"] = out["date"].dt.strftime("%Y-%m-%d")
        out.to_csv(out_spike_csv, index=False, encoding="utf-8")

# 2) strong_signals_topbar.png 생성
def gen_strong_signals_bar(
    csv_path=f"{EXPORT_DIR}/trend_strength.csv",
    out_path=f"{FIG_DIR}/strong_signals_topbar.png",
    topn=15
):
    _ensure_dirs()
    df = _safe_read_csv(csv_path)
    if df.empty:
        return
    term_col = "term" if "term" in df.columns else df.columns[0]
    val_col = "cur" if "cur" in df.columns else df.columns[1]
    df = df.sort_values(val_col, ascending=False).head(topn)

    plt.figure(figsize=(10, 6))
    plt.barh(df[term_col][::-1], df[val_col][::-1], color="#2c7be5")
    plt.title("Strong Signals (Top)")
    plt.xlabel(val_col)
    _savefig(out_path)

# 3) topics_bubble.png 생성
def gen_topics_bubble(topics_json="outputs/topics.json",
                      out_path=f"{FIG_DIR}/topics_bubble.png",
                      min_bubble=40, jitter=0.01):
    _ensure_dirs()
    try:
        with open(topics_json, "r", encoding="utf-8") as f:
            topics = json.load(f) or {}
    except Exception:
        topics = {}
    tlist = topics.get("topics") or []
    if not tlist:
        return

    xs, ys, ss, labels = [], [], [], []
    for t in tlist:
        # 관대하게 대체
        interest = t.get("interest", t.get("score", 0))
        positivity = t.get("positive", t.get("sentiment", 0.5))
        activity = t.get("activity", len(t.get("top_words", [])) * 5)

        try:
            x = float(interest or 0)
        except Exception:
            x = 0.0
        try:
            y = float(positivity if positivity is not None else 0.5)
        except Exception:
            y = 0.5
        try:
            s = float(activity or 1) * 10
        except Exception:
            s = 10.0

        # 최소 버블 크기 보장 + 겹침 완화 지터
        s = max(min_bubble, s)
        x += np.random.uniform(-jitter, jitter)
        y += np.random.uniform(-jitter, jitter)

        xs.append(x); ys.append(y); ss.append(s)
        labels.append(t.get("topic_name") or f"topic_{t.get('topic_id')}")

    plt.figure(figsize=(10, 6))
    sc = plt.scatter(xs, ys, s=ss, alpha=0.35, c=ys, cmap="coolwarm", edgecolors="#999999", linewidths=0.5)
    # 라벨은 상위 20개만
    for i, lab in enumerate(labels[:20]):
        plt.annotate(lab, (xs[i], ys[i]), fontsize=8, alpha=0.85)
    plt.colorbar(sc, label="Positivity")
    plt.xlabel("Interest"); plt.ylabel("Positivity")
    plt.title("Topics Bubble Map")
    _savefig(out_path)


### 기본 설정
def load_data():
    keywords = load_json("outputs/keywords.json", {"keywords": [], "stats": {}})
    topics = load_json("outputs/topics.json", {"topics": []})
    ts = load_json("outputs/trend_timeseries.json", {"daily": []})
    insights = load_json("outputs/trend_insights.json", {"summary": "", "top_topics": [], "evidence": {}})
    opps = load_json("outputs/biz_opportunities.json", {"ideas": []})
    meta_path = latest("data/news_meta_*.json")
    meta_items = load_json(meta_path, []) if meta_path else []
    return keywords, topics, ts, insights, opps, meta_items

def simple_tokenize_ko(text: str):
    toks = re.findall(r"[가-힣A-Za-z0-9]+", text or "")
    toks = [t.lower() for t in toks if len(t) >= 2]
    return toks

def build_docs_from_meta(meta_items):
    docs = []
    for it in meta_items:
        title = (it.get("title") or it.get("title_og") or "").strip()
        desc = (it.get("description") or it.get("description_og") or "").strip()
        doc = (title + " " + desc).strip()
        if doc:
            docs.append(doc)
    return docs

# --- Matplotlib 한글 폰트 설정 ---
def ensure_fonts():
    
    # 시스템에 설치된 Nanum 폰트 또는 Noto Sans CJK 폰트 경로 탐색
    font_paths = fm.findSystemFonts(fontpaths=None, fontext='ttf')
    nanum_gothic = next((path for path in font_paths if 'NanumGothic' in path), None)
    noto_sans_cjk = next((path for path in font_paths if 'NotoSansKR' in path or 'NotoSansCJK' in path), None)

    font_path = nanum_gothic or noto_sans_cjk
    
    if font_path:
        fm.fontManager.addfont(font_path)
        font_name = fm.FontProperties(fname=font_path).get_name()
        plt.rcParams['font.family'] = font_name
    else:
        # 적절한 폰트가 없는 경우 기본 폰트로 설정 (경고 메시지 출력)
        print("[WARN] NanumGothic or NotoSansKR font not found. Please install it for proper Korean display.")
        plt.rcParams['font.family'] = 'sans-serif'
        
    plt.rcParams['axes.unicode_minus'] = False
    print(f"[INFO] Matplotlib font set to: {plt.rcParams['font.family']}")

def apply_plot_style():
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
    # --- 폰트 설정 ---
    font_paths = fm.findSystemFonts(fontpaths=None, fontext='ttf')
    nanum_gothic = next((p for p in font_paths if 'NanumGothic' in p), None)
    noto_sans_cjk = next((p for p in font_paths if 'NotoSansKR' in p or 'NotoSansCJK' in p), None)
    font_path = nanum_gothic or noto_sans_cjk
    if not font_path:
        print("[WARN] 한글 폰트 없음. 영어만 표시될 수 있습니다.")

    # --- 데이터 준비 ---
    items = (keywords_obj or {}).get("keywords", [])
    if not items:
        plt.figure(figsize=(8, 5))
        plt.text(0.5, 0.5, "워드클라우드 데이터 없음", ha="center", va="center")
        plt.axis("off")
        plt.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close()
        return

    freqs = {it["keyword"]: max(float(it.get("score", 0)), 0) for it in items[:200] if it.get("keyword")}

    # --- 워드클라우드 생성 ---
    wc = WordCloud(
        width=1600,                   # 이미지 크기 확대 → 고해상도
        height=900,
        background_color="white",
        colormap="tab20c",            # 더 부드러운 컬러맵
        font_path=font_path,
        prefer_horizontal=0.8,        # 세로/가로 80% 비율
        relative_scaling=0.4,         # 단어 크기 비율 완화 → 시각적 균형
        min_font_size=12,
        max_font_size=None,           # 자동 최적화
        contour_color="#dddddd",      # 윤곽선 추가
        contour_width=1.0,
        random_state=42,
        margin=2,                     # 단어 간격 최소화
        collocations=False,           # 중복 단어 결합 방지
        normalize_plurals=False,
        scale=3                       # 고밀도 렌더링
    ).generate_from_frequencies(freqs)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    wc.to_file(out_path)
    print("[INFO] Saved styled wordcloud:", out_path)



def plot_top_keywords(keywords, out_path="outputs/fig/top_keywords.png", topn=15):
    ensure_fonts()
    apply_plot_style()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    data = keywords.get("keywords", [])[:topn]
    if not data:
        plt.figure(figsize=(8, 5))
        plt.text(0.5, 0.5, "키워드 데이터 없음", ha="center")
        plt.axis("off")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        return
    labels = [d["keyword"] for d in data]
    scores = [d["score"] for d in data] 
    plt.figure(figsize=(10, 6))
    sns.barplot(x=scores, y=labels, color="#3b82f6")
    plt.title("Top Keywords")
    plt.xlabel("Score")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_topics(topics, out_path="outputs/fig/topics.png", topn_words=6):
    ensure_fonts()
    apply_plot_style()
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

        sns.barplot(x=probs, y=labels, ax=ax, color="#10b981")

        # ✅ 제목 업데이트
        topic_id = t.get("topic_id")
        topic_name = t.get("topic_name", f"topic_{topic_id}")
        ax.set_title(f"Topic #{topic_id} ({topic_name})")

        ax.set_xlabel("Weight")
        ax.set_ylabel("")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_timeseries(ts, out_path="outputs/fig/timeseries.png"):
    
    ensure_fonts()
    apply_plot_style()
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
    df = pd.DataFrame(daily).copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=False)
    df["count"] = pd.to_numeric(df.get("count", 0), errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=["date"]).sort_values("date")
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
    df = df.set_index("date")
    full_idx = pd.date_range(df.index.min().normalize(), df.index.max().normalize(), freq="D")
    df = df.reindex(full_idx).fillna(0)
    df.index.name = "date"
    df["count"] = df["count"].astype(int)
    if len(df) == 1:
        d0 = df.index[0]
        y = float(df["count"].iloc[0])
        plt.figure(figsize=(12, 4.5))
        plt.xlim(d0 - timedelta(days=1), d0 + timedelta(days=1))
        ypad = max(1, y * 0.15)
        plt.ylim(0, y + ypad)
        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.plot([d0], [y], marker="o", color="#6366f1", label="Daily")
        plt.annotate(f"{int(y)}", (d0, y), textcoords="offset points", xytext=(0, -14), ha="center",
                     fontsize=9, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
        plt.title(f"Articles per Day ({d0.strftime('%Y-%m-%d')})")
        plt.xlabel("Date")
        plt.ylabel("Count")
        plt.legend(loc="upper right")
        plt.grid(alpha=0.25, linestyle="--")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        return
    plt.figure(figsize=(12, 4.5))
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.plot(df.index, df["count"], marker="o", markersize=3, linewidth=1, color="#6366f1", label="Daily")
    if len(df) >= 7:
        df["ma7"] = df["count"].rolling(window=7, min_periods=1).mean()
        plt.plot(df.index, df["ma7"], linestyle="--", linewidth=2, color="#f43f5e", label="7-day MA")
    plt.title("Articles per Day")
    plt.xlabel("Date")
    plt.ylabel("Count")
    plt.legend(loc="upper right")
    plt.grid(alpha=0.25, linestyle="--")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

def plot_keyword_network(keywords, docs, out_path="outputs/fig/keyword_network.png",
                         topn=20, min_cooccur=2, max_edges=50, label_top=20):
    """ 키워드 네트워크 생성 (PageRank 기반 라벨링 + 커뮤니티 탐지 시각화 버전) """
    from networkx.algorithms import community
    from matplotlib.patches import Patch

    ensure_fonts()
    apply_plot_style()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # 1. 키워드 점수 가져오기
    keyword_scores = {
        it.get("keyword"): float(it.get("score", 0))
        for it in keywords.get("keywords", [])[:topn]
        if it.get("keyword")
    }

    if not keyword_scores or not docs:
        return {"nodes": 0, "edges": 0}

    # 2. 공동 출현 계산
    cooccur = {}
    for doc in docs:
        words_in_doc = set(simple_tokenize_ko(doc)).intersection(keyword_scores.keys())
        words_in_doc = sorted(words_in_doc)
        for i in range(len(words_in_doc)):
            for j in range(i + 1, len(words_in_doc)):
                pair = tuple(sorted((words_in_doc[i], words_in_doc[j])))
                cooccur[pair] = cooccur.get(pair, 0) + 1

    # 3. 상위 엣지 선택
    edges = sorted(
        [(u, v, w) for (u, v), w in cooccur.items() if w >= min_cooccur],
        key=lambda x: x[2],
        reverse=True
    )[:max_edges]

    # 4. 그래프 구성
    G = nx.Graph()
    for word, score in keyword_scores.items():
        G.add_node(word, score=score)
    G.add_weighted_edges_from(edges)

    # 고립 노드 제거
    G.remove_nodes_from(list(nx.isolates(G)))
    if G.number_of_nodes() == 0:
        return {"nodes": 0, "edges": 0}

    # 5. 커뮤니티 탐지
    communities = list(community.greedy_modularity_communities(G))
    for i, comm in enumerate(communities):
        for node in comm:
            G.nodes[node]['community'] = i

    # 6. PageRank 계산
    pagerank = nx.pagerank(G, weight='weight')

    # 7. 커뮤니티 대표 라벨 설정
    community_labels = {}
    for i, comm in enumerate(communities):
        top_node = max(comm, key=lambda node: pagerank.get(node, 0))
        community_labels[i] = top_node

    # 8. 시각화
    fig, ax = plt.subplots(figsize=(14, 10))
    pos = nx.spring_layout(G, k=3.0, iterations=50, seed=42)

    node_sizes = [100 + 2500 * G.nodes[n]['score'] for n in G.nodes()]
    community_ids = [G.nodes[n]['community'] for n in G.nodes()]
    node_colors = [plt.cm.tab20(c % 20) for c in community_ids]

    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors='#000000',   # 더 진한 외곽선
        linewidths=1.0,
        alpha=0.9,
        ax=ax
    )


    edge_weights = [d['weight'] for _, _, d in G.edges(data=True)]
    w_max = max(edge_weights, default=1)
    edge_widths = [0.3 + 2.0 * (w / w_max) for w in edge_weights]
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='gray', alpha=0.5, ax=ax)

    # 9. 상위 PageRank 노드 라벨링
    top_nodes = sorted(pagerank, key=pagerank.get, reverse=True)[:label_top]
    texts = [
        ax.text(pos[n][0], pos[n][1], n,
                fontsize=12, ha='center', va='center',
                fontweight='bold', color="#222222")
        for n in top_nodes if n in pos
    ]

    if texts:
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    # 10. 범례
    legend_elements = [
        Patch(facecolor=plt.cm.tab20(i % 20), edgecolor='black', label=f'Group: {comm_label}')
        for i, comm_label in community_labels.items()
    ]
    ax.legend(
        handles=legend_elements,
        title="Groups",
        loc='lower left',           # ← 내부로 이동
        bbox_to_anchor=(0.01, 0.01),  # ← 좌측 하단 살짝 안쪽
        frameon=True,
        framealpha=0.9,
        edgecolor='gray',
        fontsize=10,
        title_fontsize=12
    )

    ax.set_title("키워드 네트워크 및 주제 그룹 분석", fontsize=16)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

    print("[INFO] Saved keyword_network.png")
    return {"nodes": G.number_of_nodes(), "edges": G.number_of_edges()}

def plot_heatmap(df, topics_map, out_path='outputs/fig/matrix_heatmap.png',
                 top_org_k=10, drop_all_zero_cols=True):
    """
    - x축: topics_map에 존재하는 topic_id만 표시 (개수 가변)
    - y축: 기업이 많으면 합계 기준 상위 top_org_k만 표시(기본 10)
    - 옵션: 전부 0인 토픽 열 제거
    """
    try:
        if df is None or df.empty:
            print("[INFO] 입력 데이터 없음. 히트맵 스킵")
            return

        # 헬퍼: topic 값을 정수 topic_id로 정규화
        def _normalize_topic_to_int(s):
            s = str(s).strip()
            s = re.sub(r"^topic_\s*", "", s)
            s = re.sub(r"\.0$", "", s)
            v = pd.to_numeric(s, errors="coerce")
            return None if pd.isna(v) else int(v)

        # topics_map -> 동적 화이트리스트/라벨 맵
        # topics_map 예: {"topic_0": "AI 로봇 기술 사업", "topic_1": "삼성 vs 애플, 배터리 & 게이밍 시장"}
        keep_topics = []
        topic_name_map = {}
        for k, name in (topics_map or {}).items():
            if isinstance(k, str) and k.startswith("topic_"):
                try:
                    tid = int(k.split("_", 1)[1])
                    keep_topics.append(tid)
                    topic_name_map[tid] = name
                except Exception:
                    continue

        if not keep_topics:
            print("[INFO] 토픽 목록이 비어 있음. 히트맵 스킵")
            return

        # topic 정규화
        df = df.copy()
        df["topic"] = df["topic"].apply(_normalize_topic_to_int)

        # 피벗
        heatmap_data = (
            df.pivot_table(index="org", columns="topic", values="hybrid_score", aggfunc="sum")
              .fillna(0)
        )
        if heatmap_data.empty:
            print("[INFO] 피벗 결과 비어 있음")
            return

        # x축: 지정된 topic_id만 남기고 없던 것도 0으로 채워 정렬 유지
        heatmap_data = heatmap_data.reindex(columns=sorted(keep_topics), fill_value=0)

        # 전부 0인 열 제거(옵션)
        if drop_all_zero_cols:
            non_zero_mask = (heatmap_data != 0).any(axis=0)
            heatmap_data = heatmap_data.loc[:, non_zero_mask]
            if heatmap_data.shape[1] == 0:
                print("[INFO] 모든 토픽 열이 0이라 그릴 게 없음")
                return

        # y축: 기업 상위 top_org_k만 남기기(합계 기준)
        if heatmap_data.shape[0] > top_org_k:
            top_orgs = heatmap_data.sum(axis=1).nlargest(top_org_k).index
            heatmap_data = heatmap_data.loc[top_orgs]

        if heatmap_data.empty:
            print("[INFO] 필터링 후 데이터 없음")
            return

        # x축 라벨
        cols = list(heatmap_data.columns)
        xlabels = [topic_name_map.get(int(c), f"topic_{int(c)}") for c in cols]

        # 시각화
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.figure(figsize=(16, 10))
        sns.heatmap(heatmap_data, cmap="viridis", linewidths=.5)

        plt.xticks(
            ticks=range(len(cols)),
            labels=xlabels,
            rotation=0, ha='left',
            fontsize=10, fontweight='bold'
        )
        plt.title('기업별 토픽 집중도 (Hybrid Score)', fontsize=16)
        plt.xlabel('토픽', fontsize=12)
        plt.ylabel('기업', fontsize=12)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        print("[INFO] Saved matrix_heatmap.png")
    except Exception as e:
        print(f"[ERROR] Failed to generate heatmap: {e}")

def plot_topic_share(df, topics_map, top_n_topics=None, out_dir="outputs/fig", debug=False):
    """
    - 호출: plot_topic_share(df, topics_map)
    - 제목 포맷: "topic_id #<id>, <topic_name>"
    - topic_name은 outputs/topics.json에서 우선 읽고, 없으면 topics_map으로 보완
    - topic_id 화이트리스트: topics_map의 키("topic_0" 형태)에서 추출
    - top_n_topics: None이면 topic_id 개수(len(keep_topics))로 자동
    - topic_share 없으면 hybrid_score 비율로 자동 계산
    - 각 토픽별 상위 5개 기업 + Others 파이 차트 저장
    """
    try:
        os.makedirs(out_dir, exist_ok=True)

        if df is None or df.empty:
            print("[INFO] 입력 df 비어 있음. 스킵")
            return
        if not isinstance(topics_map, dict) or len(topics_map) == 0:
            print("[INFO] topics_map 비어 있음. 스킵")
            return

        # 1) topic 정규화
        def _normalize_topic_to_int(val):
            s = str(val).strip()
            s = re.sub(r"^topic_\\s*", "", s)
            s = re.sub(r"\\.0$", "", s)
            v = pd.to_numeric(s, errors="coerce")
            return np.nan if pd.isna(v) else int(v)

        # 2) topics.json에서 topic_name 우선 로딩
        topic_name_from_file = {}
        topics_path = os.path.join("outputs", "topics.json")
        topics_list = []
        if os.path.exists(topics_path):
            try:
                with open(topics_path, "r", encoding="utf-8") as f:
                    topics_data = json.load(f)
                topics_list = topics_data.get("topics", []) if isinstance(topics_data, dict) else []
                for t in topics_list:
                    tid = t.get("topic_id")
                    if tid is not None:
                        topic_name_from_file[int(tid)] = t.get("topic_name", None)
            except Exception:
                pass

        # 3) topics_map → keep_topics, topic_name_map(파일 우선)
        keep_topics = []
        topic_name_map = {}
        for k, name in topics_map.items():
            if isinstance(k, str) and k.startswith("topic_"):
                try:
                    tid = int(k.split("_", 1)[1])
                    keep_topics.append(tid)
                    # 파일 이름이 있으면 우선, 없으면 topics_map 사용
                    topic_name_map[tid] = topic_name_from_file.get(tid, name)
                except Exception:
                    continue
        # 파일에만 있는 토픽이 있을 수도 있으니 보완
        for tid, nm in topic_name_from_file.items():
            if tid not in topic_name_map:
                topic_name_map[tid] = nm if nm else f"Topic {tid}"

        keep_topics = sorted(set(keep_topics))
        if debug:
            print("[DEBUG] keep_topics:", keep_topics)
        if not keep_topics:
            print("[INFO] 추출된 topic_id 없음. 스킵")
            return

        # 4) top_n_topics 자동(토픽 수 = len(keep_topics))
        if top_n_topics is None:
            top_n_topics = len(keep_topics)
        else:
            top_n_topics = min(top_n_topics, len(keep_topics))
        if debug:
            print("[DEBUG] effective top_n_topics:", top_n_topics)

        # 5) df 정리
        work = df.copy()
        for col in ["topic", "org", "hybrid_score"]:
            if col not in work.columns:
                print(f"[WARN] 필요한 컬럼 누락: {col}. 스킵")
                return
        work["topic"] = work["topic"].apply(_normalize_topic_to_int)
        work["org"] = work["org"].astype(str).str.strip()

        # 6) topic_share 없으면 계산
        if "topic_share" not in work.columns:
            grp = work.groupby(["topic", "org"], as_index=False)["hybrid_score"].sum()
            tot = grp.groupby("topic", as_index=False)["hybrid_score"].sum().rename(columns={"hybrid_score": "topic_total"})
            merged = grp.merge(tot, on="topic", how="left")
            merged["topic_share"] = np.where(
                merged["topic_total"] > 0,
                merged["hybrid_score"] / merged["topic_total"],
                0.0
            )
        else:
            merged = work.groupby(["topic", "org"], as_index=False)[["hybrid_score", "topic_share"]].sum()

        # 7) topic_id 화이트리스트 적용
        merged = merged[merged["topic"].isin(keep_topics)].copy()
        if merged.empty:
            print("[INFO] 화이트리스트 적용 후 데이터 없음. 스킵")
            return

        # 8) 상위 토픽 선정(합계 기준) → 개수는 top_n_topics로
        topic_order = (merged.groupby("topic")["hybrid_score"].sum()
                              .sort_values(ascending=False).index.tolist())
        topic_order = topic_order[:top_n_topics]

        # 9) 색상
        base_colors = plt.get_cmap("tab10").colors
        others_color = (0.6, 0.6, 0.6)

        # 10) 토픽별 파이 차트 + 제목 포맷 적용
        for topic in topic_order:
            topic_df = merged[merged["topic"] == topic].copy()
            if topic_df.empty:
                continue

            # 상위 5 + Others
            top_orgs = topic_df.sort_values("topic_share", ascending=False).head(5).reset_index(drop=True)
            if len(topic_df) > 5:
                others_share = topic_df[~topic_df["org"].isin(top_orgs["org"])]["topic_share"].sum()
                if others_share > 0:
                    top_orgs = pd.concat(
                        [top_orgs, pd.DataFrame([{"org": "Others", "topic_share": others_share}])],
                        ignore_index=True
                    )

            # 제목: "topic_id #<id>, <topic_name>"
            topic_id_int = int(topic)
            topic_name = topic_name_map.get(topic_id_int, f"Topic {topic_id_int}")
            title_text = f"[topic_id #{topic_id_int} : {topic_name}]"

            # 키워드 박스(선택)
            box_text = "topic_word (prob)\nN/A"
            if topics_list:
                tw = []
                for t in topics_list:
                    if t.get("topic_id") == topic_id_int:
                        tw = t.get("top_words", [])[:5]
                        break
                if tw:
                    lines = [f"{w.get('word')} ({w.get('prob', 0):.2f})" for w in tw if w.get("word") is not None]
                    box_text = "topic_word (prob)\n" + ("\n".join(lines))

            # 색상
            colors = list(base_colors[:max(0, len(top_orgs) - 1)])
            if len(top_orgs) - 1 > len(colors):
                extra = plt.get_cmap("tab20").colors
                need = (len(top_orgs) - 1) - len(colors)
                colors += list(extra[:need])
            if (top_orgs["org"] == "Others").any():
                colors.append(others_color)

            # 플롯(파이차트)
            plt.figure(figsize=(10, 8))
            wedges, texts, autotexts = plt.pie(
                top_orgs["topic_share"],
                labels=top_orgs["org"],
                autopct=lambda p: f"{p:.1f}%" if p >= 1 else "",
                startangle=140,
                pctdistance=0.8,
                colors=colors
            )
            for txt in texts:
                txt.set_fontsize(11)
            for at in autotexts:
                at.set_fontsize(10)
            # 제목 적용
            plt.title(title_text, fontsize=16)

            # 키워드 박스(우측 하단)
            plt.gcf().text(
                0.85, 0.15, box_text,
                fontsize=10, fontweight='bold',
                va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", lw=0.8, alpha=0.9)
            )

            plt.axis('equal')
            plt.tight_layout()
            out_path = os.path.join(out_dir, f"topic_share_{topic_id_int}.png")
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[INFO] Saved {os.path.basename(out_path)}")

    except Exception as e:
        print(f"[ERROR] Failed to generate pie charts: {e}")

def plot_company_focus(df, top_n_orgs=3, out_dir="outputs/fig", debug=False):
    """
    상위 기업별 집중도 바 차트
    - x축: topics.json에 존재하는 topic_id 전부를 항상 표시
    - 기업별 company_focus가 없는 토픽은 0으로 채워서 막대 0 높이로 표시
    - x축 라벨: "Topic #id\nname" 형식
    """

    try:
        # 방어
        if df is None or df.empty:
            print("[INFO] 입력 df 비어 있음. 스킵")
            return
        need_cols = {"org", "topic", "company_focus", "hybrid_score"}
        missing = need_cols - set(df.columns)
        if missing:
            print(f"[WARN] 필요한 컬럼 누락: {missing}. 스킵")
            return

        os.makedirs(out_dir, exist_ok=True)

        # 1) topics.json 로드 및 라벨/화이트리스트 생성
        topics_json_path = os.path.join("outputs", "topics.json")
        with open(topics_json_path, "r", encoding="utf-8") as f:
            topics_data = json.load(f)

        topics_list = topics_data.get("topics", []) if isinstance(topics_data, dict) else []
        keep_topics = [int(t["topic_id"]) for t in topics_list if "topic_id" in t]
        keep_topics = sorted(set(keep_topics))

        # 라벨 맵: id -> "Topic #id\nname"
        topic_label_map = {
            int(t["topic_id"]): f"Topic #{int(t['topic_id'])}\n{t.get('topic_name', '')}"
            for t in topics_list if "topic_id" in t
        }

        if debug:
            print("[DEBUG] keep_topics:", keep_topics)

        if not keep_topics:
            print("[INFO] 토픽 id가 없음. 스킵")
            return

        # 2) topic 정규화: "topic_3", " 3", "3.0", 3 → 3
        def _normalize_topic_to_int(val):
            s = str(val).strip()
            s = re.sub(r"^topic_\\s*", "", s)
            s = re.sub(r"\\.0$", "", s)
            v = pd.to_numeric(s, errors="coerce")
            return np.nan if pd.isna(v) else int(v)

        work = df.copy()
        work["topic"] = work["topic"].apply(_normalize_topic_to_int)
        work["org"] = work["org"].astype(str).str.strip()

        # 3) 상위 기업 선정(하이브리드 합 기준)
        top_orgs = (work.groupby("org")["hybrid_score"]
                         .sum().nlargest(top_n_orgs).index.tolist())
        if debug:
            print("[DEBUG] top_orgs:", top_orgs)

        # 4) 각 기업별로 x축을 topic_id 전부로 고정
        for org in top_orgs:
            org_df_raw = work[work["org"] == org].copy()
            if org_df_raw.empty:
                continue

            # 기업-토픽별 company_focus 집계(혹시 중복 행이 있을 경우 합산)
            org_agg = (org_df_raw.groupby("topic", as_index=False)["company_focus"].sum())

            # x축을 keep_topics로 강제(reindex), 없는 토픽은 0으로 채움
            org_agg = org_agg.set_index("topic").reindex(keep_topics, fill_value=0).reset_index()
            org_agg.rename(columns={"index": "topic"}, inplace=True)

            # x축 라벨 치환
            org_agg["topic_label"] = org_agg["topic"].map(topic_label_map)
            # 라벨이 없을 경우 대비
            org_agg["topic_label"] = org_agg["topic_label"].fillna(
                org_agg["topic"].apply(lambda x: f"Topic #{int(x)}")
            )

            # 5) 시각화: x축은 topic_id 순서 유지
            plt.figure(figsize=(14, 8))
            sns.barplot(
                data=org_agg,
                x="topic_label",
                y="company_focus",
                palette="coolwarm",
                order=org_agg["topic_label"]  # 지정 순서 유지
            )
            plt.title(f"'{org}'의 토픽별 집중도", fontsize=16)
            plt.xlabel("토픽", fontsize=12)
            plt.ylabel("집중도 점수", fontsize=12)

            # xtick 라벨 스타일: 줄바꿈 고려
            plt.xticks(rotation=45, ha="center")

            plt.tight_layout()
            out_path = os.path.join(out_dir, f"company_focus_{org}.png")
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[INFO] Saved {os.path.basename(out_path)}")

    except Exception as e:
        print(f"[ERROR] Failed to generate bar charts: {e}")


def plot_idea_score_distribution(ideas: list, output_path: str = 'outputs/fig/idea_score_distribution.png'):
    """ 아이디어별 점수 분포 바 차트 생성 (Market, Urgency, Feasibility, Risk) """

    if not ideas:
        print("[WARN] No ideas provided for score chart.")
        return

    # 아이디어 이름은 최대 15자까지만 표시
    labels = [idea.get("idea", "")[:15] + "…" if len(idea.get("idea", "")) > 15 else idea.get("idea", "") for idea in ideas]
    market = [idea["score_breakdown"]["market"] for idea in ideas]
    urgency = [idea["score_breakdown"]["urgency"] for idea in ideas]
    feasibility = [idea["score_breakdown"]["feasibility"] for idea in ideas]
    risk = [idea["score_breakdown"]["risk"] for idea in ideas]

    x = np.arange(len(ideas))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - 1.5*width, market, width, label='Market')
    bars2 = ax.bar(x - 0.5*width, urgency, width, label='Urgency')
    bars3 = ax.bar(x + 0.5*width, feasibility, width, label='Feasibility')
    bars4 = ax.bar(x + 1.5*width, risk, width, label='Risk')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_ylabel("Score (0.0 ~ 1.0)")
    ax.set_title("아이디어별 점수 분포", fontsize=16)
    ax.legend()

    # ✅ 막대 위에 값 표시
    for bars in [bars1, bars2, bars3, bars4]:
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=9)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved idea_score_distribution.png")


def plot_company_network_from_json(json_path="outputs/company_network.json",
                                   output_path="outputs/fig/company_network.png",
                                   top_edges=30, top_nodes=10):

    if not os.path.exists(json_path):
        print("[WARN] company_network.json not found")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    edges_all = data.get("edges", [])
    central = data.get("centrality", []) or []
    if not edges_all:
        print("[WARN] No edges in company_network.json")
        return

    # 1) 상위 엣지 선별
    edges_sorted = sorted(edges_all, key=lambda e: e.get("weight", 0), reverse=True)[:top_edges]

    # 2) 그래프 구성 (rel_type 유지)
    G = nx.Graph()
    for e in edges_sorted:
        u, v = e.get("source"), e.get("target")
        w = float(e.get("weight", 1.0))
        r = e.get("rel_type", "neutral")
        if not u or not v:
            continue
        G.add_edge(u, v, weight=w, rel_type=r)

    if G.number_of_nodes() == 0:
        print("[WARN] Graph empty")
        return

    # 3) 강조 노드 기준: JSON 중심성 상위 우선, 없으면 현재 그래프 기준
    if central:
        top_nodes = {c.get("org") for c in central[:top_nodes] if c.get("org")}
    else:
        deg = nx.degree_centrality(G)
        top_nodes = {n for n, _ in sorted(deg.items(), key=lambda x: x[1], reverse=True)[:top_nodes]}

    # 4) 폰트 안전 설정
    try:
        # 프로젝트 공통 한글 폰트 설정을 재사용
        font_name = plt.rcParams['font.family'][0]
    except Exception:
        font_name = "sans-serif"

    # 5) 레이아웃: 가중치 반영(Spring)
    pos = nx.spring_layout(G, weight="weight", seed=42)

    # 6) 엣지 스타일: rel_type별 색상
    edge_colors = []
    weights = []
    for u, v, d in G.edges(data=True):
        weights.append(float(d.get("weight", 1.0)))
        rt = d.get("rel_type", "neutral")
        if rt == "rivalry":
            edge_colors.append("#e74c3c")   # red
        elif rt == "partnership":
            edge_colors.append("#27ae60")   # green
        else:
            edge_colors.append("#7a7a7a")   # gray

    w_arr = np.array(weights, dtype=float)
    if w_arr.size == 0:
        print("[WARN] No edge weights")
        return
    q95 = np.quantile(w_arr, 0.95)
    w_arr = np.minimum(w_arr, q95)
    w_norm = (0.6 + 1.8 * (w_arr - w_arr.min()) / (w_arr.max() - w_arr.min() + 1e-6)).tolist()

    plt.figure(figsize=(11, 8))
    nx.draw_networkx_edges(G, pos, width=w_norm, edge_color=edge_colors, alpha=0.35)

    # 7) 노드 스타일
    node_colors = ["#e74c3c" if n in top_nodes else "#86b6f6" for n in G.nodes()]
    node_sizes = [1200 if n in top_nodes else 600 for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                           edgecolors="#333", linewidths=0.6, alpha=0.95)
    nx.draw_networkx_labels(G, pos, font_size=9, font_color="#222", font_family=font_name)

    # 8) 범례(간단 표기)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#e74c3c", lw=2, label="경쟁"),
        Line2D([0], [0], color="#27ae60", lw=2, label="협력"),
        Line2D([0], [0], color="#7a7a7a", lw=2, label="중립"),
        Line2D([0], [0], marker='o', color='w', label='허브(강조)',
               markerfacecolor="#e74c3c", markeredgecolor="#333", markersize=10)
    ]

    # 범례 추가 (그래프 내부 빈 공간에 위치, 테두리 포함)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#e74c3c", lw=2, label="경쟁"),       # 빨간 선
        Line2D([0], [0], color="#27ae60", lw=2, label="협력"),       # 초록 선
        Line2D([0], [0], color="#7a7a7a", lw=2, label="중립"),       # 회색 선
        Line2D([0], [0], marker='o', color='w', label='허브 기업',   # 빨간 노드
               markerfacecolor="#e74c3c", markeredgecolor="#333", markersize=10),
        Line2D([0], [0], marker='o', color='w', label='일반 기업',    # 파란 노드
               markerfacecolor="#86b6f6", markeredgecolor="#333", markersize=8)
    ]

    # 범례 추가 (그래프 안쪽 좌하단 + 테두리 추가)
    legend = plt.legend(handles=legend_elements,
                        loc="lower left",
                        frameon=True,
                        framealpha=1,
                        edgecolor="#333",
                        fontsize=9)
    legend.get_frame().set_linewidth(0.8)

    # 그래프 전체 테두리 추가
    ax = plt.gca()
    ax.add_patch(plt.Rectangle(
        (0, 0), 1, 1, transform=ax.transAxes,
        fill=False, edgecolor="#555", linewidth=1.2
    ))


    plt.title("기업 경쟁/협력 네트워크 (핵심 관계망)", fontsize=14, fontname=font_name)
    plt.axis("off")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved simplified company_network.png with {len(G.nodes())} nodes and {len(G.edges())} edges")


def plot_tech_maturity_map(maturity_data):
    """ 4. 기술 성숙도 맵 버블 차트 생성 (범례를 차트 안에 표시) """
    if not maturity_data.get("results"):
        return

    records = []
    for item in maturity_data["results"]:
        tech = item.get("technology")
        metrics = item.get("metrics", {})
        analysis = item.get("analysis", {})
        records.append({
            "technology": tech,
            "frequency": metrics.get("frequency", 0),
            "sentiment": metrics.get("sentiment", 0.0),
            "events": sum(metrics.get("events", {}).values()),
            "stage": analysis.get("stage", "N-A")
        })

    df = pd.DataFrame(records)
    if df.empty:
        return

    # ✅ 성숙도 단계별 색상 지정
    stage_palette = {
        "Emerging": "#9CA3AF",   # 회색
        "Growth": "#10B981",     # 녹색
        "Maturity": "#3B82F6",   # 파란색
        "N-A": "#D1D5DB"         # 데이터 없음 → 연회색
    }

    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    sns.scatterplot(
        data=df, x="frequency", y="sentiment", size="events",
        hue="stage", sizes=(200, 2000), alpha=0.7,
        palette=stage_palette, ax=ax  # ✅ 고정된 색상 사용
    )

    # 기술 라벨 표시
    texts = []
    for i in range(df.shape[0]):
        texts.append(ax.text(
            x=df.frequency[i], y=df.sentiment[i], s=df.technology[i],
            fontdict=dict(color='black', size=11, weight='bold')  # ✅ 글자 진하게
        ))

    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    plt.title('기술 성숙도 맵 (Technology Maturity Map)', fontsize=16)
    plt.xlabel('시장 관심도 (뉴스 빈도)', fontsize=12)
    plt.ylabel('시장 긍정성 (감성 점수)', fontsize=12)

    # 범례 설정
    handles, labels = ax.get_legend_handles_labels()
    num_stages = df['stage'].nunique()
    stage_handles = handles[1:num_stages+1]
    stage_labels = labels[1:num_stages+1]

    legend = ax.legend(stage_handles, stage_labels, title='성숙도 단계', loc='best', frameon=True, framealpha=0.8)
    if legend:
        legend.get_title().set_fontsize('14')
        for text in legend.get_texts():
            text.set_fontsize('12')

    plt.grid(True)
    plt.tight_layout()
    plt.savefig('outputs/fig/tech_maturity_map.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[INFO] Saved tech_maturity_map.png")


def plot_weak_signal_radar(weak_signals_df):
    """ 5. 약한 신호 레이더 차트 생성 (라벨 수정 버전) """
    if weak_signals_df.empty:
        return

    plt.figure(figsize=(12, 8))
    ax = plt.gca() # 축 객체 가져오기
    sns.scatterplot(
        data=weak_signals_df, x="total", y="z_like", size="cur",
        sizes=(100, 1000), alpha=0.7, color="red", ax=ax, legend=False
    )

    # --- ✨ 시인성 향상된 라벨 추가 ✨ ---
    texts = []
    for i in range(weak_signals_df.shape[0]):
        label = weak_signals_df.term[i]
        x = weak_signals_df.total[i]
        y = weak_signals_df.z_like[i]
        texts.append(ax.text(
            x, y, label,
            fontsize=11,
            color='black',  # 라벨 색: 대비를 위해 검정
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", lw=0.5, alpha=0.8)  # 흰 배경 박스
        ))

    adjust_text(
        texts, ax=ax,
        arrowprops=dict(arrowstyle='->', color='gray', lw=0.5)  # 화살표는 흐린 회색
    )


    plt.title('약한 신호 레이더 (Weak Signal Radar)', fontsize=16)
    plt.xlabel('익숙함 (총 누적 언급량)', fontsize=12)
    plt.ylabel('임팩트 (통계적 급등 수준)', fontsize=12)

    if not weak_signals_df.empty:
        plt.axhline(0.8, color='gray', linestyle='--', linewidth=0.8)
        plt.text(weak_signals_df['total'].max(), 0.8, '  주목 기준선', color='gray', va='bottom')

    plt.grid(True)
    plt.tight_layout()
    plt.savefig('outputs/fig/weak_signal_radar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[INFO] Saved weak_signal_radar.png")

def plot_strong_signals(strong_signals_df):
    """ 6. 강한 신호 임팩트 순위 바 차트 생성 (좌우 대칭 및 값 표시 버전) """
    if strong_signals_df.empty:
        return

    rising = strong_signals_df[strong_signals_df['z_like'] > 0].head(5)
    rising['trend'] = '상승(Rising)'

    falling = strong_signals_df[strong_signals_df['z_like'] < 0].tail(5)
    falling['trend'] = '하강(Falling)'

    combined = pd.concat([rising, falling]).sort_values('z_like', ascending=False)
    
    if combined.empty:
        print("[INFO] No significant rising or falling signals to plot.")
        return

    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    
    sns.barplot(
        data=combined,
        y="term",
        x="z_like",
        hue="trend",
        palette={"상승(Rising)": "#3b82f6", "하강(Falling)": "#ef4444"},
        dodge=False,
        ax=ax
    )

    # --- ✨✨✨ 1. 좌우 대칭을 위한 X축 범위 설정 ✨✨✨ ---
    # z_like 값의 절대값 중 가장 큰 값을 기준으로 좌우 대칭 설정
    max_abs_z = combined['z_like'].abs().max()
    limit = max_abs_z * 1.2  # 약간의 여백 추가
    ax.set_xlim(-limit, limit)
    # --- ✨✨✨ 여기까지 ---

    # --- ✨✨✨ 2. 각 막대에 값(z_like 점수) 표시 ✨✨✨ ---
    for p in ax.patches:
        width = p.get_width()
        # 값의 위치를 막대 끝에서 약간 떨어지게 설정
        x_pos = width + (limit * 0.02) if width > 0 else width - (limit * 0.02)
        
        # 텍스트 정렬 설정
        ha = 'left' if width > 0 else 'right'
        
        ax.text(x=x_pos, 
                y=p.get_y() + p.get_height() / 2, 
                s=f'{width:.2f}', # 소수점 둘째 자리까지 표시
                va='center', 
                ha=ha,
                fontsize=10)
    # --- ✨✨✨ 여기까지 ---

    plt.title('주요 신호 시계열 변화 (상승/하강)', fontsize=16)
    plt.xlabel('임팩트 (z_like)', fontsize=12)
    plt.ylabel('키워드', fontsize=12)
    ax.axvline(0, color='grey', linewidth=0.8)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('outputs/fig/strong_signals_barchart.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[INFO] Saved strong_signals_barchart.png")



def main():
    """ 메인 실행 함수 """
    # 폰트 설정
    ensure_fonts()

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
    try:
        docs = build_docs_from_meta(meta_items)
        kw_list = keywords.get("keywords", [])
        n_kw = len(kw_list)
        label_cap = 25
        plot_keyword_network(
            keywords, docs,
            out_path="outputs/fig/keyword_network.png",
            topn=n_kw,
            # min_cooccur=1,
            # max_edges=200,
            label_top=(None if n_kw <= label_cap else label_cap)
        )
    except Exception as e:
        print("[WARN] 키워드 네트워크 실패:", e)

    # 데이터 로드
    try:
        df = pd.read_csv('outputs/export/company_topic_matrix_long.csv')
    except FileNotFoundError:
        print("[ERROR] company_topic_matrix_long.csv not found. Please run module_d.py first.")
        return

    # 토픽 키워드 맵 로드
    try:
        with open('outputs/topics.json', 'r', encoding='utf-8') as f:
            topics_data = json.load(f)
        topics_map = {f"topic_{t['topic_id']}": ", ".join(w['word'] for w in t['top_words'][:2]) for t in topics_data['topics']}
    except Exception:
        topics_map = {}
        print("[WARN] topics.json not found or failed to parse. Topic IDs will be used as labels.")

    #plot_idea_score_distribution
    try:
        with open('outputs/biz_opportunities.json', 'r', encoding='utf-8') as f:
            ideas_data = json.load(f)
        top_ideas = sorted(ideas_data["ideas"], key=lambda it: it.get("score", 0), reverse=True)[:5]
        plot_idea_score_distribution(top_ideas)
    except Exception as e:
        import traceback
        print(f"[ERROR] Failed to generate idea score chart: {e}")
        traceback.print_exc()
    
    # 기업 네트워크 시각화
    try:
        plot_company_network_from_json("outputs/company_network.json", "outputs/fig/company_network.png")
    except Exception as e:
        print(f"[WARN] company network visualization failed: {repr(e)}")

    # 기술 성숙도 레이더 시각화
    try:
        tech_maturity_data = load_json('outputs/tech_maturity.json')
    except Exception:
        tech_maturity_data = {"results": []}
        print("[WARN] tech_maturity.json not found.")

    try:
        strong_signals_df = pd.read_csv('outputs/export/trend_strength.csv')
    except FileNotFoundError:
        strong_signals_df = pd.DataFrame()
        print("[WARN] trend_strength.csv not found.")

    try:
        weak_signals_df = pd.read_csv('outputs/export/weak_signals.csv')
    except FileNotFoundError:
        weak_signals_df = pd.DataFrame()
        print("[WARN] weak_signals.csv not found.")


    # 시각화 함수 호출
    os.makedirs('outputs/fig', exist_ok=True)
    plot_heatmap(df, topics_map)
    plot_topic_share(df, topics_map)
    plot_company_focus(df)
    plot_tech_maturity_map(tech_maturity_data)
    plot_weak_signal_radar(weak_signals_df)
    plot_strong_signals(strong_signals_df)

    # 추가 이미지
    gen_timeseries_spikes()
    gen_strong_signals_bar()
    gen_topics_bubble()
    
    print("\n[SUCCESS] All visualizations have been generated in 'outputs/fig/'")

if __name__ == '__main__':
    main()
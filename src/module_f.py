# -*- coding: utf-8 -*-
import os
import re
import glob
import json
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd

# 외부 의존 유틸 (기존 코드와 호환)
from src.utils import load_json, save_json, latest

# -----------------------------
# 상수/경로
# -----------------------------
FIG_DIR = "outputs/fig"
EXPORT_DIR = "outputs/export"
OUT_MD = "outputs/report.md"
OUT_HTML = "outputs/report.html"


# -----------------------------
# 안전 유틸
# -----------------------------
def _fmt_int(x):
    try:
        return f"{int(x):,}"
    except Exception:
        try:
            return f"{float(x):.0f}"
        except Exception:
            return str(x) if x is not None else "-"

def _fmt_float(x, nd=2):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "-"

def _truncate(s, n=80):
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= n else s[:n-1] + "…"

def _exists(path):
    return path and os.path.exists(path)

def _safe_read_csv(path, **kwargs):
    try:
        if _exists(path):
            return pd.read_csv(path, **kwargs)
    except Exception:
        pass
    return pd.DataFrame()

def _to_markdown_table(df: pd.DataFrame, max_rows=50):
    if df is None or df.empty:
        return "- (데이터 없음)\n"
    use = df.head(max_rows).copy()
    return use.to_markdown(index=False) + ("\n" if len(use) else "\n")

def _load_data():
    keywords = load_json("outputs/keywords.json", {"keywords": [], "stats": {}})
    topics = load_json("outputs/topics.json", {"topics": []})
    ts = load_json("outputs/trend_timeseries.json", {"daily": []})
    insights = load_json("outputs/trend_insights.json", {"summary": "", "top_topics": [], "evidence": {}})
    opps = load_json("outputs/biz_opportunities.json", {"ideas": []})
    tech_maturity = load_json("outputs/tech_maturity.json", {"results": []})
    weak_insights = load_json("outputs/weak_signal_insights.json", {"results": []})
    # 최신 메타(기사 원문 메타)
    meta_path = latest("data/news_meta_*.json")
    meta_items = load_json(meta_path, []) if meta_path else []
    return {
        "keywords": keywords,
        "topics": topics,
        "ts": ts,
        "insights": insights,
        "opps": opps,
        "tech_maturity": tech_maturity,
        "weak_insights": weak_insights,
        "meta_items": meta_items
    }

def _export_csvs(ts_obj, keywords_obj, topics_obj, out_dir=EXPORT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    # 기존 로직 유지 + 방어코드 강화
    daily = (ts_obj or {}).get("daily", [])
    df_ts = pd.DataFrame(daily) if daily else pd.DataFrame(columns=["date", "count"])
    df_ts.to_csv(os.path.join(out_dir, "timeseries_daily.csv"), index=False, encoding="utf-8")

    kws = (keywords_obj or {}).get("keywords", [])[:20]
    df_kw = pd.DataFrame(kws) if kws else pd.DataFrame(columns=["keyword", "score"])
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
    df_tw = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["topic_id", "word", "prob"])
    df_tw.to_csv(os.path.join(out_dir, "topics_top_words.csv"), index=False, encoding="utf-8")
    print("[INFO] export CSVs -> outputs/export/*.csv")


# -----------------------------
# 마크다운 빌더 헬퍼(섹션 템플릿)
# -----------------------------
def _section_header(title):
    return f"\n## {title}\n"

def _section_summary(bullets):
    if not bullets:
        return ""
    lines = ["**핵심 요약**"]
    for b in bullets:
        lines.append(f"- {b}")
    return "\n".join(lines) + "\n"

def _insert_images(image_paths, captions=None):
    lines = []
    if not isinstance(image_paths, (list, tuple)):
        image_paths = [image_paths]
    captions = captions or []
    for i, p in enumerate(image_paths):
        rel = p
        if _exists(p):
            # outputs/ 접두 제거
            if p.startswith("outputs/"):
                rel = p.replace("outputs/", "")
            else:
                rel = p
            # 윈도우 역슬래시 → 슬래시 정규화
            rel = rel.replace("\\", "/")
            cap = captions[i] if i < len(captions) else ""
            lines.append(f"![{cap or 'Figure'}]({rel})")
    return ("\n".join(lines) + "\n") if lines else ""

def _guide_block(tips=None, so_what=None, next_step=None):
    lines = []
    if tips:
        lines.append(f"> 해석 가이드: {tips}")
    if so_what:
        lines.append(f"> 그래서: {so_what}")
    if next_step:
        lines.append(f"> 다음 액션: {next_step}")
    return ("\n".join(lines) + "\n") if lines else ""

# -----------------------------
# 섹션 1) 커버/원페이지 대시보드
# -----------------------------
def _section_dashboard(data):
    keywords = data.get("keywords", {})
    topics = data.get("topics", {})
    ts = data.get("ts", {})
    insights = data.get("insights", {})
    today = datetime.now().strftime("%Y-%m-%d")

    daily = ts.get("daily", [])
    n_days = len(daily)
    total_cnt = sum(int(x.get("count", 0)) for x in daily)
    date_range = f"{daily[0].get('date', '?')} ~ {daily[-1].get('date', '?')}" if n_days > 0 else "-"

    klist = (keywords.get("keywords") or [])[:20]
    tlist = (topics.get("topics") or [])

    bullets = [
        f"기간: {date_range}",
        f"기사 수: {_fmt_int(total_cnt)}",
        f"토픽 수: {_fmt_int(len(tlist))} | 상위 키워드: {_fmt_int(len(klist))}"
    ]

    lines = []
    lines.append(_section_header(f"Intellegence Report ({today})"))
    lines.append(_section_summary(bullets))

    # 미니 차트/요약 이미지 삽입 (있을 경우)
    imgs = [
        f"{FIG_DIR}/topic_share_mini.png",
        f"{FIG_DIR}/strong_signals_topbar.png",
    ]
    lines.append(_insert_images(imgs, captions=["상위 토픽 점유율", "강한 신호 상위"]))

    # 한 줄 요약 3개
    summary = (insights.get("summary") or "").strip()
    if summary:
        one_liners = [s.strip() for s in re.split(r"[•\-\n]", summary) if s.strip()]
        lines.append("**이번 주 하이라이트**")
        for s in one_liners:
            lines.append(f"- {s}")
        lines.append("")
    else:
        lines.append("- (요약 없음)\n")

    return "\n".join(lines)


# -----------------------------
# 섹션 2) 모니터링 스냅샷(이번 주 하이라이트)
# -----------------------------
def _section_monitoring_snapshot(data):
    meta_items = data.get("meta_items", [])
    lines = []
    lines.append(_section_header("모니터링 스냅샷"))

    # 카드형 요약 이미지를 하나로 합성해 놓았다고 가정
    card_img = f"{FIG_DIR}/weekly_highlights_cards.png"
    lines.append(_insert_images(card_img, captions=["이번 주 하이라이트"]))

    # 근거 기사 상위 5개 (있다면) -> latest news_meta_*.json에서 상위 5개 기사 로드
    rows = []
    for it in meta_items[:5]:
        rows.append({
            "제목": _truncate(it.get("title", ""), 100),
            "링크": it.get("url", "")
        })
    df_news = pd.DataFrame(rows)
    if not df_news.empty:
        lines.append("### 기사 하이라이트(상위 5)")
        lines.append(_to_markdown_table(df_news, max_rows=5))
        lines.append("")

    return "\n".join(lines)


# -----------------------------
# 섹션 3) 핵심 지표 보드(Key Metrics Board)
# -----------------------------
def _section_key_metrics(data):
    keywords = data.get("keywords", {})
    topics = data.get("topics", {})
    ts = data.get("ts", {})

    daily = ts.get("daily", [])
    n_days = len(daily)
    total_cnt = sum(int(x.get("count", 0)) for x in daily)
    date_range = f"{daily[0].get('date', '?')} ~ {daily[-1].get('date', '?')}" if n_days > 0 else "-"

    klist = (keywords.get("keywords") or [])[:20]
    tlist = (topics.get("topics") or [])

    df_kpi = pd.DataFrame([{
        "기간": date_range,
        "총 기사 수": total_cnt,
        "문서 수": keywords.get("stats", {}).get("num_docs", 0),
        "상위 키워드 수": len(klist),
        "토픽 수": len(tlist),
        "시계열 일수": n_days
    }])

    lines = []
    lines.append(_section_header("핵심 지표 보드"))
    lines.append(_insert_images(f"{FIG_DIR}/kpi_board.png", captions=["주요 KPI 보드"]))
    lines.append(_to_markdown_table(df_kpi, max_rows=10))

    lines.append(_guide_block(
        tips="전주 대비 증감률을 함께 보아야 변화를 정확히 해석할 수 있습니다.",
        so_what="문서/기사 수가 급증하면 키워드/토픽 점유 구조가 바뀔 가능성이 큽니다.",
        next_step="이상값이 보이면 타임라인 섹션의 변곡점 주석과 함께 원인을 추적하세요."
    ))
    return "\n".join(lines)


# -----------------------------
# 섹션 4) 키워드 인텔리전스(Top Keywords & Clusters)
# -----------------------------
def _section_keywords_intel(data):
    keywords = data.get("keywords", {})
    klist = (keywords.get("keywords") or [])
    klist_sorted = sorted(klist, key=lambda x: x.get("score", 0), reverse=True)

    # 키워드 표(상위 20)
    rows = []
    for i, k in enumerate(klist_sorted[:20], 1):
        rows.append({
            "순위": i,
            "키워드": k.get("keyword", ""),
            "점수": _fmt_float(k.get("score", 0), 3),
            "변화(참고)": k.get("delta", "")
        })
    df_kw = pd.DataFrame(rows)

    lines = []
    lines.append(_section_header("키워드 인텔리전스"))
    lines.append(_section_summary([
        "상위 키워드 분포와 클러스터를 통해 테마 맥락을 파악합니다.",
        "점수 상승과 긍정 감성 결합은 유망 신호입니다."
    ]))

    # 이미지: 워드클라우드, 키워드 네트워크
    imgs = [f"{FIG_DIR}/wordcloud.png", f"{FIG_DIR}/keyword_network.png"]
    # top_keywords 막대차트
    top_kw_bar = os.path.join(FIG_DIR, "top_keywords.png")
    if _exists(top_kw_bar):
        imgs.append(top_kw_bar)
    lines.append(_insert_images(imgs, captions=["워드클라우드", "키워드 네트워크", "상위 키워드 바차트"]))

    # 표: 상위 20 키워드
    lines.append("### 상위 키워드 Top 20")
    lines.append(_to_markdown_table(df_kw, max_rows=20))

    # 클러스터 표(있다면)
    cluster_csv = os.path.join(EXPORT_DIR, "keyword_clusters.csv")
    df_cluster = _safe_read_csv(cluster_csv)
    if not df_cluster.empty:
        lines.append("### 키워드 클러스터")
        lines.append(_to_markdown_table(df_cluster, max_rows=10))

    lines.append(_guide_block(
        tips="클러스터는 규제/정책, 공급망, 수요/고객 등 테마로 해석하면 빠릅니다.",
        so_what="상위 키워드가 네트워크 허브와 겹치면 시장 영향력이 높은 신호일 가능성이 큽니다.",
        next_step="해당 키워드를 포함한 토픽과 기업 교차점(매트릭스 섹션)을 확인하세요."
    ))
    return "\n".join(lines)


# -----------------------------
# 섹션 5) 토픽 레이더(Topics & Momentum)
# -----------------------------
def _section_topics_radar(data):
    topics = data.get("topics", {})
    tlist = (topics.get("topics") or [])

    # 토픽 표: topic_id + topic_summary만 남기기
    rows = []
    for t in tlist:
        tid = t.get("topic_id")
        # topic_summary가 없을 때 insight/summary를 백업으로 사용
        summ = (t.get("topic_summary")
                or t.get("summary")
                or t.get("insight")
                or "")
        summ = (summ or "").replace("\n", " ").strip()
        rows.append({
            "topic_id": tid,
            "topic_summary": _truncate(summ, 120)
        })
    df_topics = pd.DataFrame(rows)

    lines = []
    lines.append(_section_header("토픽 레이더"))
    lines.append(_section_summary([
        "관심도(X), 긍정성(Y), 성장률(색상)로 토픽 지형을 한눈에 파악합니다.",
        "상위 토픽의 미니 시계열로 모멘텀을 점검합니다."
    ]))

    # 이미지: 버블차트, 미니 시계열
    imgs = [f"{FIG_DIR}/topics_bubble.png", f"{FIG_DIR}/topics_mini_trends.png", f"{FIG_DIR}/topics.png"]
    lines.append(_insert_images(imgs, captions=["토픽 버블 지도", "상위 토픽 미니 트렌드", "토픽 요약"]))

    # 표: 토픽 개요(두 컬럼만)
    lines.append("### 토픽 개요")
    if not df_topics.empty:
        # 혹시 컬럼 누락 대비
        cols = [c for c in ["topic_id", "topic_summary"] if c in df_topics.columns]
        lines.append(_to_markdown_table(df_topics[cols], max_rows=50))
    else:
        lines.append("- (토픽 데이터 없음)")

    # 토픽 상승/하락 표(있다면)
    growth_csv = os.path.join(EXPORT_DIR, "topic_growth.csv")
    df_growth = _safe_read_csv(growth_csv)
    if not df_growth.empty:
        lines.append("### 토픽 상승/하락")
        lines.append(_to_markdown_table(df_growth, max_rows=15))

    lines.append(_guide_block(
        tips="성장률이 높고 긍정성이 유지되는 토픽은 우선 탐색 대상입니다.",
        so_what="토픽 성장과 기업 진입 변화가 동시에 보이면 시장 전환점일 가능성이 있습니다.",
        next_step="상위 토픽을 기업×토픽 매트릭스에서 교차 확인하고, 기회/리스크로 분기하세요."
    ))
    return "\n".join(lines)

# -----------------------------
# 섹션 6) 기업×토픽 매트릭스/히트맵(경쟁 구도)
# -----------------------------
def _section_company_topic_matrix(data):
    topics_obj = data.get("topics", {}) or {}
    topics_map = {
        f"topic_{t.get('topic_id')}": ", ".join([w.get('word', '') for w in (t.get('top_words') or [])[:2]])
        for t in (topics_obj.get('topics') or [])
    }

    csv_path = os.path.join(EXPORT_DIR, "company_topic_matrix_wide.csv")
    df = _safe_read_csv(csv_path, encoding="utf-8-sig")
    lines = []
    lines.append(_section_header("기업×토픽 매트릭스/히트맵"))

    # 요약 계산
    summary_bullets = []
    top_competitive_topic = "N/A"
    top_focused_org = "N/A"
    rising_star = "N/A"

    if not df.empty:
        topic_cols = [c for c in df.columns if c.startswith("topic_")]
        # 숫자 변환
        df_numeric = df.copy()
        for col in topic_cols:
            # "0.12 (34%)" 같은 포맷 방어
            df_numeric[col] = (
                df[col].astype(str).str.split(" ").str[0].replace({"": "0", "nan": "0"}).astype(float)
            )
        # 경쟁 치열 토픽(>0인 기업 수 최대)
        competitive_scores = df_numeric[topic_cols].gt(0).sum()
        if not competitive_scores.empty and competitive_scores.max() > 0:
            top_competitive_topic = competitive_scores.idxmax()
            top_competitive_topic = topics_map.get(top_competitive_topic, top_competitive_topic)

        # 가장 집중 높은 기업(전체 합)
        df_numeric["total_score"] = df_numeric[topic_cols].sum(axis=1)
        if df_numeric["total_score"].max() > 0:
            top_focused_org = df_numeric.loc[df_numeric["total_score"].idxmax(), "org"]

        # 최고 단일 조합
        max_val = 0
        for col in topic_cols:
            col_max = df_numeric[col].max()
            if col_max > max_val:
                max_val = col_max
                org_name = df_numeric.loc[df_numeric[col].idxmax(), "org"]
                rising_star = f"{org_name} × {topics_map.get(col, col)}"

        summary_bullets = [
            f"가장 경쟁 치열 토픽: {top_competitive_topic}",
            f"가장 집중도 높은 기업: {top_focused_org}",
            f"최고 단일 조합: {rising_star}"
        ]
    else:
        summary_bullets = ["- (분석할 유효 데이터가 없습니다.)"]

    lines.append(_section_summary(summary_bullets))

    # 이미지: 히트맵, 토픽 점유율 파이, 기업 집중 바
    imgs = [
        f"{FIG_DIR}/matrix_heatmap.png",
    ]
    # 토픽별/기업별 추가 이미지 자동 삽입
    imgs += sorted(glob.glob(os.path.join(FIG_DIR, "topic_share_*.png")))
    imgs += sorted(glob.glob(os.path.join(FIG_DIR, "company_focus_*.png")))

    lines.append(_insert_images(imgs, captions=[]))

    # 표: 원본 매트릭스 상위 N행
    if not df.empty:
        lines.append("### 매트릭스 표(미리보기)")
        lines.append(_to_markdown_table(df, max_rows=20))

    lines.append(_guide_block(
        tips="히트맵에서 진한 교차지점은 전략 초점을 의미합니다. 점수는 소스 커버리지에 민감합니다.",
        so_what="경쟁 치열 토픽에서의 점유율 변화는 공격/수비 전략 신호일 수 있습니다.",
        next_step="상위 기업과 토픽 교차를 기회 섹션으로 연결하고, 파트너/경쟁사 움직임을 추적하세요."
    ))
    return "\n".join(lines)


# -----------------------------
# 섹션 7) 관계·경쟁 네트워크(허브/브로커/커뮤니티)
# -----------------------------
def _section_relationship_network(data):
    net_path = "outputs/company_network.json"
    lines = []
    lines.append(_section_header("관계·경쟁 네트워크"))

    net = {}
    try:
        if _exists(net_path):
            with open(net_path, "r", encoding="utf-8") as f:
                net = json.load(f) or {}
    except Exception:
        pass

    if not net:
        lines.append("- (네트워크 데이터가 없어 본 섹션을 생략합니다.)\n")
        return "\n".join(lines)

    edges = net.get("edges", [])
    nodes = net.get("nodes", [])
    top_pairs = net.get("top_pairs", [])
    central = net.get("centrality", [])
    betw = net.get("betweenness", [])
    comms = net.get("communities", [])

    # 핵심 요약
    lines.append("**핵심 요약**")
    lines.append(f"- 관계망 규모: 노드 {len(nodes)}개 / 엣지 {len(edges)}개")
    if top_pairs:
        tp = top_pairs[0]
        lines.append(f"- 가장 강한 관계: {tp.get('source')} ↔ {tp.get('target')} (가중치 {tp.get('weight')}, 유형 {tp.get('rel_type')})")
    if central:
        lines.append(f"- 허브 후보: {central[0].get('org')} (Degree {central[0].get('degree_centrality')})")
    if betw:
        lines.append(f"- 브로커 후보: {betw[0].get('org')} (Betweenness {betw[0].get('betweenness')})")
    lines.append("")

    # 상위 관계쌍
    if top_pairs:
        df_pairs = pd.DataFrame(top_pairs)[["source","target","weight","rel_type"]]
        lines.append("### 상위 관계쌍(Edge)")
        lines.append("동일 문서/문장에서 함께 언급된 기업 쌍이며, 가중치는 동시출현 빈도입니다. 유형은 규칙 기반 추정입니다.\n")
        lines.append(df_pairs.rename(columns={
            "source": "Source", "target": "Target", "weight": "Weight", "rel_type": "Type"
        }).to_markdown(index=False))
        lines.append("")

    # 중심성 상위
    if central:
        df_c = pd.DataFrame(central)[["org","degree_centrality"]]
        lines.append("### 중심성 상위(연결 허브)")
        lines.append("Degree 중심성은 연결된 상대 수의 비율입니다. 높을수록 허브 성격.\n")
        lines.append(df_c.rename(columns={"org": "Org", "degree_centrality": "DegreeCentrality"}).to_markdown(index=False))
        lines.append("")
    if betw:
        df_b = pd.DataFrame(betw)[["org","betweenness"]]
        lines.append("### 매개 중심성 상위(정보 브로커)")
        lines.append("Betweenness는 집단 간 ‘다리’ 역할 정도입니다. 높을수록 중개자 성격.\n")
        lines.append(df_b.rename(columns={"org": "Org", "betweenness": "Betweenness"}).to_markdown(index=False))
        lines.append("")

    # 커뮤니티
    if comms:
        lines.append("### 커뮤니티(관계 클러스터)")
        lines.append("> 모듈러리티 기반 자동 추출 집단. 같은 집단 내 기업은 유사 주제/밸류체인 공유 가능성.")
        preview = []
        for c in comms[:5]:
            members = (c.get("members", []) or [])[:6]
            theme = (c.get("interpretation", "") or "").strip()
            if not theme:
                theme = f"{members[0]} 중심의 연관 클러스터" if members else "주요 기업 연관 클러스터"
            preview.append(f"- C{c.get('community_id')}: {', '.join(members)} | 해석: {theme}")
        lines.extend(preview)
        lines.append("")

    # 이미지
    img_rel = os.path.join(FIG_DIR, "company_network.png")
    lines.append(_insert_images(img_rel, captions=["기업 관계 네트워크"]))

    lines.append(_guide_block(
        tips="동시출현이 높은 쌍은 경쟁/협력 가능성을 시사합니다.",
        so_what="허브/브로커 포지션은 영향력/협상력을 가리킵니다.",
        next_step="커뮤니티별 주요 토픽/키워드와 교차하여 전략 단위를 정의하세요."
    ))
    return "\n".join(lines)


# -----------------------------
# 섹션 8) 트렌드 타임라인(Time Series)
# -----------------------------
def _section_time_series(data):
    ts = data.get("ts", {})
    daily = ts.get("daily", [])
    n_days = len(daily)
    total_cnt = sum(int(x.get("count", 0)) for x in daily)
    date_range = f"{daily[0].get('date', '?')} ~ {daily[-1].get('date', '?')}" if n_days > 0 else "-"

    lines = []
    lines.append(_section_header("트렌드 타임라인"))
    lines.append(_section_summary([
        f"기간: {date_range}",
        f"총 기사 수: {_fmt_int(total_cnt)}",
        "일별 기사 수와 7일 이동평균, 변곡점을 주석으로 표시합니다."
    ]))

    # 이미지
    lines.append(_insert_images(os.path.join(FIG_DIR, "timeseries.png"), captions=["일별 기사 수 추이"]))
    # 이벤트 마커/변곡점 이미지가 추가 생성되어 있으면 자동 삽입
    spike_img = os.path.join(FIG_DIR, "timeseries_spikes.png")
    lines.append(_insert_images(spike_img, captions=["이상치/스파이크 마커"]))

    # 표: 스파이크 리스트 (있다면)
    spikes_csv = os.path.join(EXPORT_DIR, "timeseries_spikes.csv")
    df_spikes = _safe_read_csv(spikes_csv)
    if not df_spikes.empty:
        lines.append("### 스파이크/이벤트 목록")
        lines.append(_to_markdown_table(df_spikes, max_rows=20))

    lines.append(_guide_block(
        tips="단기 급등은 이벤트성 가능성이 있으니 2주 추세를 함께 보세요.",
        so_what="스파이크 이후 토픽/신호 방향을 교차 확인하면 원인 파악이 빨라집니다.",
        next_step="스파이크 날짜의 주요 기사와 키워드 변동을 키워드/토픽 섹션에서 재확인하세요."
    ))
    return "\n".join(lines)


# -----------------------------
# 섹션 9) 시그널 보드(강한/약한 신호)
# -----------------------------
def _section_signals_board(data):
    lines = []
    try:
        lines.append(_section_header("시그널 보드(강한/약한 신호)"))

        # -----------------------------
        # 강한 신호
        # -----------------------------
        strong_csv = os.path.join(EXPORT_DIR, "trend_strength.csv")
        df_strong = _safe_read_csv(strong_csv)

        if not df_strong.empty:
            lines.append("### 강한 신호(Strong Signals)")

            # 1) 기본 상위 바 차트(있으면)
            img_topbar = os.path.join(FIG_DIR, "strong_signals_topbar.png")
            if _exists(img_topbar):
                lines.append(_insert_images(img_topbar, captions=["강한 신호 상위 바"]))

            # 2) strong_signals_barchart.png (시계열/임팩트 뷰)
            img_barchart = os.path.join(FIG_DIR, "strong_signals_barchart.png")
            if _exists(img_barchart):
                lines.append(_insert_images(img_barchart, captions=["강한 신호 시계열/임팩트 뷰"]))

            # 표: 주요 컬럼만 안전 선택
            strong_cols_pref = ["term", "cur", "wow", "z_like", "sentiment", "comment", "prev", "diff", "ma7", "total"]
            cols = [c for c in strong_cols_pref if c in df_strong.columns]
            if cols:
                lines.append(_to_markdown_table(df_strong[cols], max_rows=20))
            lines.append("")

        # -----------------------------
        # 약한 신호
        # -----------------------------
        weak_csv = os.path.join(EXPORT_DIR, "weak_signals.csv")
        df_weak = _safe_read_csv(weak_csv)
        weak_insights = data.get("weak_insights", {"results": []}) or {"results": []}
        insights_map = {it.get("signal"): it.get("interpretation") for it in (weak_insights.get("results") or [])}

        if not df_weak.empty:
            lines.append("### 약한 신호(Weak/Emerging Signals)")

            # 레이더 차트(파일 존재 시 삽입)
            img_radar = os.path.join(FIG_DIR, "weak_signal_radar.png")
            if _exists(img_radar):
                lines.append(_insert_images(img_radar, captions=["약한 신호 레이더(누적 언급량 vs 급등도)"]))

            # 표 요약(핵심 칼럼만)
            rep_rows = []
            for _, row in df_weak.head(30).iterrows():
                term = row.get("term") or row.get("signal") or ""
                cur = row.get("cur", "")
                z_like = row.get("z_like", "")
                rep_rows.append({
                    "약한 신호": term,
                    "지표(cur / z_like)": f"{cur} / {z_like}",
                    "해석(1줄)": insights_map.get(term, "-")
                })
            lines.append(_to_markdown_table(pd.DataFrame(rep_rows), max_rows=30))
            lines.append("")

        if (df_strong.empty if isinstance(df_strong, pd.DataFrame) else True) and \
           (df_weak.empty if isinstance(df_weak, pd.DataFrame) else True):
            lines.append("- (강/약한 신호 데이터 없음)\n")

        lines.append(_guide_block(
            tips="약한 신호는 가속도·이례성이 핵심이고, 강한 신호는 현재 주목도의 절대값입니다.",
            so_what="약한→강한 신호로 전환하는 구간이 가장 좋은 기회가 됩니다.",
            next_step="상위 신호 5개를 2주간 추적하고, 토픽/기업 매트릭스와 교차 검증하세요."
        ))
        return "\n".join(lines)

    except Exception as e:
        return f"\n## 시그널 보드(강한/약한 신호)\n- (섹션 생성 중 오류: {e})\n"


# -----------------------------
# 섹션 10) 기술 성숙도 맵(Technology Maturity)
# -----------------------------
def _section_tech_maturity(data):
    tech_maturity = data.get("tech_maturity", {"results": []}) or {"results": []}
    items = tech_maturity.get("results") or []

    lines = []
    lines.append(_section_header("기술 성숙도 맵"))
    lines.append(_section_summary([
        "관심도(X), 긍정성(Y), 버블(사업 활발도)로 기술의 위치를 표현합니다.",
        "단계: Seed/Early/Growth/Mature/Legacy"
    ]))

    # 이미지
    lines.append(_insert_images(os.path.join(FIG_DIR, "tech_maturity_map.png"), captions=["기술 성숙도 지도"]))

    # 표
    rows = []
    for it in items:
        tech = it.get("technology", "N/A")
        analysis = it.get("analysis", {}) or {}
        stage = analysis.get("stage", "N/A")
        reason = analysis.get("reason", "-")
        rows.append({"기술": tech, "단계": stage, "판단 근거": _truncate(reason, 200)})
    df_tm = pd.DataFrame(rows)
    if not df_tm.empty:
        lines.append(_to_markdown_table(df_tm, max_rows=30))
    else:
        lines.append("- (분석된 기술 성숙도 데이터가 없습니다.)\n")

    lines.append(_guide_block(
        tips="Growth 구간이면서 강한 신호 결합 시 유망합니다.",
        so_what="Early라도 약한 신호가 누적되면 전환 후보입니다.",
        next_step="상위 기술 3개를 기회 섹션과 연결하고 파일럿 과제를 정의하세요."
    ))
    return "\n".join(lines)

# -----------------------------
# 섹션 11) 비즈니스 기회(Top 5 + 점수 분포)
# -----------------------------
def _section_opportunities(data):
    opps = data.get("opps", {"ideas": []}) or {"ideas": []}
    ideas_all = opps.get("ideas") or []

    lines = []
    lines.append(_section_header("비즈니스 기회(Top 5)"))

    # 점수 분포 이미지
    lines.append(_insert_images(os.path.join(FIG_DIR, "idea_score_distribution.png"), captions=["아이디어 점수 분포"]))

    if ideas_all:
        ideas_sorted = sorted(
            ideas_all,
            key=lambda it: float(it.get("score", it.get("priority_score", 0)) or 0),
            reverse=True
        )[:5]

        # 표
        rows = []
        for it in ideas_sorted:
            idea_raw = (it.get("idea", "") or it.get("title", "") or "")
            tgt_raw = it.get("target_customer", "") or ""
            vp_raw = (it.get("value_prop", "") or "").replace("\n", " ")

            bd = it.get("score_breakdown", {}) or {}
            score_val = it.get("score", it.get("priority_score", ""))
            rows.append({
                "아이디어": _truncate(idea_raw, 120),
                "타깃": _truncate(tgt_raw, 80),
                "가치제안": _truncate(vp_raw, 180),
                "점수(시장/긴급/실행/리스크)": f"{score_val} ({bd.get('market','')}/{bd.get('urgency','')}/{bd.get('feasibility','')}/{bd.get('risk','')})"
            })
        df_opp = pd.DataFrame(rows)
        lines.append(_to_markdown_table(df_opp, max_rows=5))
    else:
        lines.append("- (아이디어 없음)\n")

    lines.append(_guide_block(
        tips="점수는 내부 가중치 기반(예: Market 0.35, Urgency 0.25, Feasibility 0.25, Risk -0.15).",
        so_what="상위안은 단기 임팩트/실행 가능성이 높은 편입니다.",
        next_step="각 아이디어에 대해 2주 파일럿(가설·대상·KPI)을 정의하고 담당 오너를 배정하세요."
    ))
    return "\n".join(lines)


# -----------------------------
# 섹션 12) 리스크/이슈 관측소(Risk Watch)
# -----------------------------
def _section_risk_watch(data):
    lines = []
    lines.append(_section_header("리스크/이슈 관측소"))

    # 이미지: 부정 감성 급등, 리스크 키워드 네트워크(있으면)
    imgs = [
        os.path.join(FIG_DIR, "risk_negative_spikes.png"),
        os.path.join(FIG_DIR, "risk_keyword_network.png"),
    ]
    lines.append(_insert_images(imgs, captions=["부정 감성 급등 토픽", "리스크 키워드 네트워크"]))

    # 표: 이슈 목록(있다면)
    risk_csv = os.path.join(EXPORT_DIR, "risk_issues.csv")
    df_risk = _safe_read_csv(risk_csv)
    if not df_risk.empty:
        cols = [c for c in ["date", "topic", "impact_range", "summary", "mitigation"] if c in df_risk.columns]
        if cols:
            lines.append("### 이슈 목록")
            lines.append(_to_markdown_table(df_risk[cols], max_rows=20))
    else:
        lines.append("- (리스크/이슈 데이터 없음)\n")

    lines.append(_guide_block(
        tips="부정 감성 급등은 PR·컴플라이언스·조달과 연계가 필요합니다.",
        so_what="공급망 키워드와 동반 급등 시 실제 운영 리스크로 이어질 수 있습니다.",
        next_step="영향범위가 큰 이슈는 즉시 리스크 레지스터에 등록하고 완화 액션을 실행하세요."
    ))
    return "\n".join(lines)


# -----------------------------
# 섹션 13) 결론: So What & Next Steps
# -----------------------------
def _section_conclusion(data):
    lines = []
    lines.append(_section_header("결론: So What & Next Steps"))

    # 이미지: 임팩트×긴급도 우선순위 매트릭스(있다면)
    lines.append(_insert_images(os.path.join(FIG_DIR, "priority_matrix.png"), captions=["우선순위 매트릭스"]))

    # 표: 2주 실행안(있다면)
    plan_csv = os.path.join(EXPORT_DIR, "two_week_plan.csv")
    df_plan = _safe_read_csv(plan_csv)
    if not df_plan.empty:
        cols = [c for c in ["hypothesis", "target", "kpi", "owner", "due"] if c in df_plan.columns]
        if cols:
            lines.append("### 다음 2주 실행안")
            lines.append(_to_markdown_table(df_plan[cols], max_rows=20))
    else:
        # 샘플 템플릿
        df_plan = pd.DataFrame([
            {"hypothesis":"토픽 A 관심도↑는 고객 니즈 증가", "target":"SMB 고객 20명 인터뷰",
             "kpi":"응답률 30%/신규 리드 10건", "owner":"사업개발팀", "due":(datetime.now()+timedelta(days=14)).strftime("%Y-%m-%d")}
        ])
        lines.append("### 다음 2주 실행안(템플릿)")
        lines.append(_to_markdown_table(df_plan, max_rows=10))

    lines.append(_guide_block(
        tips="데이터-인사이트-액션의 연결을 표준 템플릿으로 기록하세요.",
        so_what="우선순위 상위 안건부터 빠르게 실험해야 기회를 놓치지 않습니다.",
        next_step="오너/기한/KPI가 명확한 항목부터 킥오프하세요."
    ))
    return "\n".join(lines)


# -----------------------------
# 섹션 14) 부록(Appendix)
# -----------------------------
def _section_appendix(data):
    lines = []
    lines.append(_section_header("부록(Appendix)"))
    lines.append("분석의 투명성을 확보하고, 다음 단계 분석을 위한 원천 자료를 제공합니다.\n")

    # 표: 데이터 소스/버전/생성일/파일 해시 등(가능한 항목만)
    rows = []
    files = [
        "outputs/keywords.json",
        "outputs/topics.json",
        "outputs/trend_timeseries.json",
        "outputs/trend_insights.json",
        "outputs/biz_opportunities.json",
        "outputs/tech_maturity.json",
        "outputs/company_network.json",
    ]
    for fp in files:
        try:
            mtime = datetime.fromtimestamp(os.path.getmtime(fp)).strftime("%Y-%m-%d %H:%M:%S") if _exists(fp) else "-"
            size = os.path.getsize(fp) if _exists(fp) else 0
            rows.append({"파일": fp, "수정시각": mtime, "크기(bytes)": size})
        except Exception:
            rows.append({"파일": fp, "수정시각": "-", "크기(bytes)": "-"})
    df_meta = pd.DataFrame(rows)
    lines.append("### 데이터 메타")
    lines.append(_to_markdown_table(df_meta, max_rows=50))

    # 표: 주요 파라미터(있다면)
    params_csv = os.path.join(EXPORT_DIR, "params_summary.csv")
    df_params = _safe_read_csv(params_csv)
    if not df_params.empty:
        lines.append("### 주요 파라미터")
        lines.append(_to_markdown_table(df_params, max_rows=50))

    lines.append(_guide_block(
        tips="재현성과 감사추적을 위해 생성일, 버전, 파라미터를 함께 보관하세요.",
        so_what="데이터 출처와 가공 과정을 명확히 남기면 신뢰성이 높아집니다.",
        next_step="주요 이미지/표 생성 실패 로그도 추적 테이블로 남겨두세요."
    ))
    return "\n".join(lines)


# -----------------------------
# 리포트 조립: 마크다운 빌드
# -----------------------------
def build_markdown_new():
    data = _load_data()
    # 필요한 CSV/엑스포트 보장
    _export_csvs(data.get("ts", {}), data.get("keywords", {}), data.get("topics", {}))

    lines = []
    # 1) 대시보드
    lines.append(_section_dashboard(data))
    # 2) 스냅샷
    lines.append(_section_monitoring_snapshot(data))
    # 3) 핵심 지표
    lines.append(_section_key_metrics(data))
    # 4) 키워드 인텔리전스
    lines.append(_section_keywords_intel(data))
    # 5) 토픽 레이더
    lines.append(_section_topics_radar(data))
    # 6) 기업×토픽 매트릭스
    lines.append(_section_company_topic_matrix(data))
    # 7) 관계·경쟁 네트워크
    lines.append(_section_relationship_network(data))
    # 8) 타임라인
    lines.append(_section_time_series(data))
    # 9) 시그널 보드
    lines.append(_section_signals_board(data))
    # 10) 기술 성숙도
    lines.append(_section_tech_maturity(data))
    # 11) 비즈니스 기회
    lines.append(_section_opportunities(data))
    # 12) 리스크/이슈 관측소
    lines.append(_section_risk_watch(data))
    # 13) 결론
    lines.append(_section_conclusion(data))
    # 14) 부록
    lines.append(_section_appendix(data))

    Path(OUT_MD).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return OUT_MD


# -----------------------------
# MD → HTML 변환
# -----------------------------
def build_html_from_md_new(md_path=OUT_MD, out_html=OUT_HTML):
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
<title>Weekly/New Biz Report</title>
<link rel="preconnect" href="https://fonts.gstatic.com">
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Noto Sans KR', sans-serif; line-height: 1.6; padding: 24px; color: #222; }}
  img {{ max-width: 100%; height: auto; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
  th {{ background: #f7f7f7; }}
  code {{ background: #f1f5f9; padding: 2px 4px; border-radius: 4px; }}
  td, th {{ overflow-wrap: anywhere; word-break: break-word; white-space: normal; }}
  h2 {{ margin-top: 28px; }}
  .toc {{ margin-bottom: 16px; }}
</style>
</head>
<body>
{html}
</body>
</html>"""
        Path(out_html).parent.mkdir(parents=True, exist_ok=True)
        with open(out_html, "w", encoding="utf-8") as f:
            f.write(html_tpl)
    except Exception as e:
        print("[WARN] HTML 변환 실패:", e)


# -----------------------------
# main
# -----------------------------
def main():
    try:
        md_path = build_markdown_new()
        build_html_from_md_new(md_path, OUT_HTML)
        print("[INFO] New report generated:", md_path, OUT_HTML)
    except Exception as e:
        print("[ERROR] New report generation failed:", e)
        # 폴백: 기존 함수들 호출 가능
        try:
            # 기존 빌더가 있다면 호출
            from __main__ import build_markdown, build_html_from_md
            keywords, topics, ts, insights, opps, meta_items = _load_data().values()
            build_markdown(keywords, topics, ts, insights, opps)
            build_html_from_md()
        except Exception as e2:
            print("[ERROR] Fallback also failed:", e2)

if __name__ == "__main__":
    main()
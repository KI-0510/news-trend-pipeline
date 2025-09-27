# -*- coding: utf-8 -*-
import os
import json
import re
import glob
import unicodedata
import datetime
from typing import List, Dict, Any, Tuple, Optional
from email.utils import parsedate_to_datetime
from collections import Counter, defaultdict
import csv

# ================= 공용 스위치/로그 =================
def load_config():
    try:
        with open("config.json","r",encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}
CFG = load_config()

def llm_config(cfg: dict) -> dict:
    return cfg.get("llm", {}) or {}
LLM = llm_config(CFG)

def use_pro_mode() -> bool:
    v = os.getenv("USE_PRO", "").lower()
    if v in ("1","true","yes","y"):
        return True
    if v in ("0","false","no","n"):
        return False
    return bool(CFG.get("use_pro", False))

def _log_mode(prefix="Module C"):
    try:
        is_pro = use_pro_mode()
    except Exception:
        is_pro = False
    mode = "PRO" if is_pro else "LITE"
    print(f"[INFO] USE_PRO={str(is_pro).lower()} → {prefix} ({mode}) 시작")

# ================= KST 날짜 유틸 =================
from datetime import datetime, timedelta, timezone
KST = timezone(timedelta(hours=9))

def kst_today_str() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d")

def kst_tomorrow_str() -> str:
    return (datetime.now(KST) + timedelta(days=1)).strftime("%Y-%m-%d")

# ================= 공용 유틸 =================
def latest(globpat: str) -> Optional[str]:
    files = sorted(glob.glob(globpat))
    return files[-1] if files else None

def clean_text(t: str) -> str:
    if not t: return ""
    t = re.sub(r"<.+?>", " ", t)
    t = unicodedata.normalize("NFKC", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def to_date(s: str) -> str:
    today = datetime.now(KST).date()
    if not s or not isinstance(s, str): return today.strftime("%Y-%m-%d")
    s = s.strip()
    try:
        iso = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(iso)
        d = dt.date()
    except Exception:
        try:
            dt = parsedate_to_datetime(s); d = dt.date()
        except Exception:
            m = re.search(r"(\d{4}).*?(\d{1,2}).*?(\d{1,2})", s)
            if m:
                y, mm, dd = int(m.group(1)), int(m.group(2)), int(m.group(3))
                try: d = datetime(y, mm, dd).date()
                except Exception: d = today
            else:
                d = today
    if d > today: d = today
    return d.strftime("%Y-%m-%d")

# ================= 토큰화/불용어 =================
EN_STOP = {
    "the","and","to","of","in","for","on","with","at","by","from","as","is","are","be","it",
    "that","this","an","a","or","if","we","you","they","he","she","was","were","been","than",
    "into","about","over","under","per","via"
}
KO_FUNC = {
    "하다","있다","되다","통해","이번","대한","것으로","밝혔다","다양한","함께","현재",
    "기자","대표","회장","주요","기준","위해","위한","지원","전략","정책","협력","확대",
    "말했다","강조했다","대상","대상으로","최근","지난해","생활","시장","스마트","디지털","글로벌",
    "그는","그녀는","이어","한편","또한","이날","이라며","이라고","모델을","성과를","받았다","서울","기반으로",
    "있는","있으며","있다는","이후","설명했다","전했다","계획이다","관계자는","따르면",
    "올해","내년","최대","신규","기존","국제","국내","세계","오전","오후",
    "등을","따라","있도록","지난","특히","대비","아니라","만에","의원은","라고",
    "있습니다","관련","한다","진행한다","예정이다","가능하다","있었다",
    "이상","넘어","제공한다","같은","했다","많은","그리고","같다","우리","하고",
    "때문에","이렇게","이런","등이","각각"
}
STOP_EXT = set(EN_STOP) | set(KO_FUNC)

def norm_tok(x: str) -> str:
    return x.lower()

def tokenize(t: str) -> List[str]:
    toks = re.findall(r"[가-힣A-Za-z0-9]{2,}", t or "")
    toks = [norm_tok(x) for x in toks if x and x.lower() not in STOP_EXT]
    return toks

# ================= 데이터 로더(하루 1파일) =================
def select_latest_files_per_day(glob_pattern: str, days: int) -> List[str]:
    """
    오늘(KST) 제외. 날짜별 최신 1개만 선택. 최근 N일 반환.
    """
    all_files = sorted(glob.glob(glob_pattern))
    daily_files = defaultdict(list)
    for f in all_files:
        date_key = os.path.basename(f)[:10]
        daily_files[date_key].append(f)

    latest_daily_files = []
    for date_key in sorted(daily_files.keys()):
        latest_file_for_day = sorted(daily_files[date_key])[-1]
        latest_daily_files.append(latest_file_for_day)

    today_kst = kst_today_str()
    past_files = [f for f in latest_daily_files if os.path.basename(f)[:10] < today_kst]
    return past_files[-days:]

def load_today_meta() -> Tuple[List[str], List[str]]:
    meta_path = latest("data/news_meta_*.json")
    if not meta_path: return [], []
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            items = json.load(f) or []
    except Exception:
        return [], []
    docs, dates = [], []
    for it in items:
        title = clean_text((it.get("title") or it.get("title_og") or "").strip())
        desc  = clean_text((it.get("body") or it.get("description") or it.get("description_og") or "").strip())
        if not title and not desc: continue
        doc = (title + " " + desc).strip()
        if not doc: continue
        docs.append(doc)
        d_raw = it.get("published_time") or it.get("pubDate_raw") or ""
        dates.append(to_date(d_raw))
    return docs, dates

# ================= 타임시리즈(발행일 기준, d/d+1 최신 파일만) =================
def _group_by_day_latest(glob_pattern: str) -> Dict[str, str]:
    files = sorted(glob.glob(glob_pattern))
    by_day = defaultdict(list)
    for f in files:
        d = os.path.basename(f)[:10]
        by_day[d].append(f)
    latest = {}
    for d, fps in by_day.items():
        latest[d] = sorted(fps)[-1]
    return latest

def _iter_jsonl(fp: str):
    try:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = (line or "").strip()
                if not line: continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue
    except Exception:
        return

def _std_date_from_published(published: str, fallback: str) -> str:
    if published:
        try:
            iso = published.replace("Z", "+00:00")
            dt = datetime.fromisoformat(iso)
            return dt.date().strftime("%Y-%m-%d")
        except Exception:
            try:
                dt = parsedate_to_datetime(published)
                return dt.date().strftime("%Y-%m-%d")
            except Exception:
                pass
    return fallback

def build_timeseries_published_based(days:int=30) -> dict:
    """
    정책:
    - 집계 대상 날짜 = 과거 N일 (오늘 제외)
    - 각 날짜 d에 대해 d, d+1의 최신 jsonl만 스캔
    - 두 파일에서 published==d 인 레코드만 카운트 (url/title로 중복 제거)
    """
    latest_map = _group_by_day_latest("data/warehouse/*.jsonl")
    if not latest_map:
        return {"daily": []}

    today = kst_today_str()
    all_days_sorted = sorted([d for d in latest_map.keys() if d < today])
    target_days = all_days_sorted[-max(1, days):]

    out = []
    for d in target_days:
        d_plus_1 = (datetime.strptime(d, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        fps = []
        if d in latest_map: fps.append((d, latest_map[d]))
        if d_plus_1 in latest_map: fps.append((d_plus_1, latest_map[d_plus_1]))

        cnt = 0
        seen = set()
        for file_day, fp in fps:
            for obj in _iter_jsonl(fp):
                title = obj.get("title") or ""
                if not title: continue
                pub_day = _std_date_from_published(obj.get("published") or obj.get("created_at") or "", fallback=file_day)
                if pub_day == d:
                    key = obj.get("url") or title[:200]
                    if key in seen: continue
                    seen.add(key)
                    cnt += 1
        out.append({"date": d, "count": cnt})
    return {"daily": out}

# ================= 토픽(Lite/PRO) =================
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def is_bad_token(w: str) -> bool:
    base = w.split()[0] if " " in w else w
    return base.lower() in STOP_EXT

def build_topics_lite(docs: List[str], max_features=8000, topn=10) -> Dict[str, Any]:
    if not docs:
        return {"topics": []}
    vec = CountVectorizer(max_features=max_features, stop_words=list(EN_STOP))
    X = vec.fit_transform(docs)
    lda = LatentDirichletAllocation(n_components=min(8, max(2, len(docs)//5)), random_state=42)
    W = lda.fit_transform(X)
    H = lda.components_
    terms = vec.get_feature_names_out()
    topics = []
    for tid, row in enumerate(H):
        idx = row.argsort()[::-1]
        payload = []
        for i in idx[:max(40, topn)]:
            w = terms[i]
            if is_bad_token(w): continue
            payload.append({"word": w, "prob": float(row[i])})
        if not payload:
            payload = [{"word": terms[i], "prob": float(row[i])} for i in idx[:max(40, topn)]]
        topics.append({"topic_id": int(tid), "top_words": payload[:topn]})
    return {"topics": topics}

try:
    from bertopic import BERTopic  # optional
    def pro_build_topics_bertopic(docs: List[str], topn=10) -> Dict[str, Any]:
        if not docs:
            return {"topics": []}
        model = BERTopic(language="multilingual", calculate_probabilities=True, verbose=False)
        topics, _ = model.fit_transform(docs)
        topics_obj = {"topics": []}
        topic_info = model.get_topics()
        for tid, items in topic_info.items():
            if tid == -1: continue
            head = items[:max(40, topn)]
            kept = []
            for w, s in head:
                base = w.split()[0] if " " in w else w
                if is_bad_token(base): continue
                kept.append((w, float(s or 0.0)))
            if not kept:
                kept = [(w, float(s or 0.0)) for w, s in head]
            scores = [max(float(s or 0.0), 0.0) for _, s in kept]
            maxv = max(scores) if scores else 0.0
            payload = []
            if maxv > 0 and (max(scores) - min(scores)) > 1e-12:
                for (w, s) in kept[:topn]:
                    prob = max(float(s or 0.0), 0.0) / maxv
                    payload.append({"word": w, "prob": prob})
            else:
                decay = 0.95
                for rank, (w, _s) in enumerate(kept[:topn], start=0):
                    prob = max(0.2, decay**rank)
                    payload.append({"word": w, "prob": prob})
            topics_obj["topics"].append({"topic_id": int(tid), "top_words": payload[:topn]})
        return topics_obj
except Exception:
    def pro_build_topics_bertopic(docs: List[str], topn=10) -> Dict[str, Any]:
        return build_topics_lite(docs, max_features=8000, topn=topn)

def _ensure_prob_payload(topics_obj: Dict[str, Any], topn=10, decay=0.95, floor=0.2) -> Dict[str, Any]:
    out = {"topics": []}
    for t in topics_obj.get("topics", []):
        words = t.get("top_words") or []
        scores = [float(w.get("prob", 0.0)) for w in words]
        maxv = max(scores) if scores else 0.0
        payload = []
        if maxv > 0 and (max(scores) - min(scores)) > 1e-12:
            for w in words[:topn]:
                prob = max(float(w.get("prob", 0.0)), 0.0) / maxv
                payload.append({"word": w.get("word",""), "prob": prob})
        else:
            for rank, w in enumerate(words[:topn], start=0):
                prob = max(floor, decay**rank)
                payload.append({"word": w.get("word",""), "prob": prob})
        out["topics"].append({"topic_id": int(t.get("topic_id", 0)), "top_words": payload[:topn]})
    return out

# ================= 시그널 계산(강/약) =================
def daily_counts(rows):
    # rows: list of (date, tokens)
    by_day = defaultdict(Counter)
    for d, toks in rows:
        for t in toks:
            by_day[d][t] += 1
    return dict(sorted(by_day.items()))

def moving_avg(vals, w=7):
    out = []
    for i in range(len(vals)):
        s = max(0, i - w + 1)
        seg = vals[s:i+1]
        out.append(sum(seg) / max(1, len(seg)))
    return out

def z_like(vals, ma):
    z = []
    for v, m in zip(vals, ma):
        z.append((v - m) / ((m**0.5) + 1.0))
    return z

def to_rows(dc):
    terms = set()
    for _, c in dc.items():
        terms.update(c.keys())
    dates = sorted(dc.keys())
    rows = []
    for t in sorted(terms):
        counts = [dc[d].get(t, 0) for d in dates]
        ma7 = moving_avg(counts, 7)
        z = z_like(counts, ma7)
        cur = counts[-1] if counts else 0
        prev = counts[-2] if len(counts) >= 2 else 0
        diff = cur - prev
        rows.append({
            "term": t,
            "dates": dates,
            "counts": counts,
            "cur": cur, "prev": prev, "diff": diff,
            "ma7": ma7[-1] if ma7 else 0.0,
            "z_like": z[-1] if z else 0.0,
            "total": sum(counts)
        })
    return rows

def export_trend_strength(rows):
    os.makedirs("outputs/export", exist_ok=True)
    with open("outputs/export/trend_strength.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["term", "cur", "prev", "diff", "ma7", "z_like", "total"])
        for r in sorted(rows, key=lambda x: (x["z_like"], x["diff"], x["cur"]), reverse=True)[:300]:
            w.writerow([r["term"], r["cur"], r["prev"], r["diff"], round(r["ma7"],3), round(r["z_like"],3), r["total"]])

def export_weak_signals(rows):
    cand = []
    for r in rows:
        if r["total"] <= 15 and r["cur"] >= 2 and r["z_like"] > 0.8:
            cand.append(r)
    with open("outputs/export/weak_signals.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["term", "cur", "prev", "diff", "ma7", "z_like", "total"])
        for r in sorted(cand, key=lambda x: (x["z_like"], x["cur"]), reverse=True)[:200]:
            w.writerow([r["term"], r["cur"], r["prev"], r["diff"], round(r["ma7"],3), round(r["z_like"],3), r["total"]])

# ===== 이벤트 추출(메타 기반) =====
EVENT_MAP = {
    "LAUNCH":      [r"출시", r"론칭", r"발표", r"선보이", r"공개"],
    "PARTNERSHIP": [r"제휴", r"파트너십", r"업무협약", r"\bMOU\b", r"맞손"],
    "INVEST":      [r"투자", r"유치", r"라운드", r"시리즈 [ABCD]"],
    "ORDER":       [r"수주", r"계약 체결", r"납품 계약", r"공급 계약", r"수의 계약"],
    "CERT":        [r"인증", r"허가", r"승인", r"적합성 평가", r"CE ?인증", r"FDA ?승인"],
    "REGUL":       [r"규제", r"가이드라인", r"행정예고", r"고시", r"지침", r"제정", r"개정"],
}

def _detect_events_from_items(items: list) -> list:
    rows = []
    for it in items:
        title = (it.get("title") or it.get("title_og") or "").strip()
        body  = (it.get("body") or it.get("description") or it.get("description_og") or "").strip()
        text  = f"{title}\n{body}"
        date  = (it.get("date") or it.get("pubDate") or "")[:10]
        url   = it.get("url") or ""
        src   = it.get("source") or it.get("press") or ""
        for etype, pats in EVENT_MAP.items():
            for pat in pats:
                if re.search(pat, text, flags=re.IGNORECASE):
                    rows.append({"date": date or "", "type": etype, "title": title[:300], "url": url, "source": src})
                    break
    return rows

def _dedup_events(rows: list) -> list:
    seen, out = set(), []
    for r in rows:
        key = (r.get("date",""), r.get("type",""), r.get("title",""))
        if key in seen: continue
        seen.add(key); out.append(r)
    return out

def export_events(out_path="outputs/export/events.csv"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    meta_path = latest("outputs/debug/news_meta_latest.json") or latest("data/news_meta_*.json")
    if not meta_path:
        print("[INFO] events.csv skipped (no meta)")
        return
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            items = json.load(f) or []
    except Exception as e:
        print("[WARN] events: meta load failed:", repr(e))
        items = []
    rows = _detect_events_from_items(items)
    rows = _dedup_events(rows)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date","type","title","url","source"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[INFO] events.csv exported | rows={len(rows)}")

# ================= LLM 인사이트(요약 목업) =================
def gemini_insight(api_key: str, model: str, context: dict, max_tokens: int = 2048, temperature: float = 0.3) -> str:
    topics = context.get("topics", [])
    ts = context.get("timeseries", [])
    kws = context.get("keywords", [])
    return f"최근 {len(ts)}일 트렌드와 상위 {len(topics)}개 토픽, {len(kws)}개 키워드를 바탕으로 요약했습니다."

# ================= 메인 =================
def main():
    _log_mode("Module C")
    os.makedirs("outputs", exist_ok=True)

    # 1) 오늘 메타 → 토픽 전용
    docs_today, _ = load_today_meta()

    # 2) 발행일 기반 타임시리즈 (오늘 제외, d와 d+1 최신 파일만 사용)
    ts_obj = build_timeseries_published_based(days=30)
    with open("outputs/trend_timeseries.json", "w", encoding="utf-8") as f:
        json.dump(ts_obj, f, ensure_ascii=False, indent=2)

    # 3) 토픽 생성
    try:
        if use_pro_mode():
            topics_obj = pro_build_topics_bertopic(docs_today or [], topn=10)
        else:
            topics_obj = build_topics_lite(docs_today or [], max_features=8000, topn=10)
    except Exception as e:
        print(f"[WARN] Pro 토픽 실패, Lite로 폴백: {e}")
        topics_obj = build_topics_lite(docs_today or [], max_features=8000, topn=10)

    topics_obj = _ensure_prob_payload(topics_obj, topn=10, decay=0.95, floor=0.2)
    with open("outputs/topics.json", "w", encoding="utf-8") as f:
        json.dump(topics_obj, f, ensure_ascii=False, indent=2)

    # 4) 인사이트(시계열 tail 14일)
    try:
        with open("outputs/keywords.json", "r", encoding="utf-8") as f:
            keywords_obj = json.load(f)
    except Exception:
        keywords_obj = {"keywords": []}
    top_keywords = [k.get("keyword") for k in keywords_obj.get("keywords", [])[:10]]

    tail_14 = ts_obj.get("daily", [])[-14:] if isinstance(ts_obj.get("daily", []), list) else []
    api_key = os.getenv("GEMINI_API_KEY", "")
    model_name = str(LLM.get("model", "gemini-2.0-flash"))
    summary = gemini_insight(
        api_key=api_key,
        model=model_name,
        context={"topics": topics_obj.get("topics", []), "timeseries": tail_14, "keywords": top_keywords},
        max_tokens=int(LLM.get("max_output_tokens", 2048)),
        temperature=float(LLM.get("temperature", 0.3)),
    )
    top_topics = []
    for t in topics_obj.get("topics", []):
        words = [w.get("word", "") for w in (t.get("top_words") or [])][:5]
        top_topics.append({"topic_id": t.get("topic_id"), "words": words})
    insights_obj = {"summary": summary, "top_topics": top_topics, "evidence": {"timeseries": tail_14}}
    with open("outputs/trend_insights.json", "w", encoding="utf-8") as f:
        json.dump(insights_obj, f, ensure_ascii=False, indent=2)

    # 5) 신호(강/약) 산출은 여기선 텍스트 소스가 필요하므로, 기존 파이프라인과 동일하게 signal_export.py에서 처리 추천.
    #    필요 시 이 파일에서 rows(date,tokens)를 구성해 계산해도 되지만, 정책 일관성을 위해 signal_export에 통합 권장.
    export_events()  # 이벤트 CSV 생성

    print(f"[INFO] C done | topics={len(topics_obj.get('topics',[]))} ts_days={len(ts_obj.get('daily',[]))}")

if __name__ == "__main__":
    main()

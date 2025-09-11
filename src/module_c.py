# -*- coding: utf-8 -*-
import os
import json
import re
import glob
import unicodedata
import datetime
from typing import List, Dict, Any, Tuple, Optional
from email.utils import parsedate_to_datetime
from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# ---------------- 공통 설정 ----------------
try:
    from config import load_config, llm_config
except Exception:
    def load_config() -> dict:
        return {}
    def llm_config(cfg: dict) -> dict:
        llm = cfg.get("llm") or {}
        return {
            "model": llm.get("model", "gemini-1.5-flash"),
            "max_output_tokens": int(llm.get("max_output_tokens", 2048)),
            "temperature": float(llm.get("temperature", 0.3)),
        }

CFG = load_config()
LLM = llm_config(CFG)

# ---------------- 파일/텍스트 유틸 ----------------
def latest(globpat: str) -> Optional[str]:
    files = sorted(glob.glob(globpat))
    return files[-1] if files else None

def clean_text(t: str) -> str:
    if not t:
        return ""
    t = re.sub(r"<.+?>", " ", t)
    t = unicodedata.normalize("NFKC", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def to_date(s: str) -> str:
    today = datetime.date.today()
    if not s or not isinstance(s, str):
        return today.strftime("%Y-%m-%d")
    s = s.strip()
    try:
        iso = s.replace("Z", "+00:00")
        dt = datetime.datetime.fromisoformat(iso)
        d = dt.date()
    except Exception:
        try:
            dt = parsedate_to_datetime(s)
            d = dt.date()
        except Exception:
            m = re.search(r"(\d{4}).*?(\d{1,2}).*?(\d{1,2})", s)
            if m:
                y, mm, dd = int(m.group(1)), int(m.group(2)), int(m.group(3))
                try:
                    d = datetime.date(y, mm, dd)
                except Exception:
                    d = today
            else:
                d = today
    if d > today:
        d = today
    return d.strftime("%Y-%m-%d")

# ---------------- 데이터 로더 ----------------
def load_today_meta() -> Tuple[List[str], List[str]]:
    """
    최신 data/news_meta_*.json에서 문서(docs)와 날짜(dates)만 반환
    """
    meta_path = latest("data/news_meta_*.json")
    if not meta_path:
        return [], []
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            items = json.load(f) or []
    except Exception:
        return [], []

    docs, dates = [], []
    for it in items:
        title = clean_text((it.get("title") or it.get("title_og") or "").strip())
        desc  = clean_text((it.get("body") or it.get("description") or it.get("description_og") or "").strip())
        if not title and not desc:
            continue
        doc = (title + " " + desc).strip()
        if not doc:
            continue
        docs.append(doc)

        d_raw = it.get("published_time") or it.get("pubDate_raw") or ""
        dates.append(to_date(d_raw))
    return docs, dates

def load_warehouse(days: int = 30) -> Tuple[List[str], List[str]]:
    """
    최근 N일 data/warehouse/*.jsonl에서 날짜만 수집(시계열용)
    반환: (docs, dates) — docs는 사용하지 않으므로 빈 리스트
    """
    files = sorted(glob.glob("data/warehouse/*.jsonl"))[-days:]
    docs, dates = [], []
    for fp in files:
        try:
            file_day = os.path.basename(fp)[:10]  # 'YYYY-MM-DD'
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    line = (line or "").strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    d_raw = obj.get("published") or obj.get("created_at") or file_day
                    try:
                        d_std = to_date(d_raw)
                    except Exception:
                        d_std = file_day
                    dates.append(d_std)
        except Exception:
            continue
    return docs, dates

# ---------------- 시계열 집계 ----------------
def timeseries_by_date(dates: List[str]) -> Dict[str, Any]:
    cnt = Counter(dates or [])
    daily = [{"date": d, "count": int(cnt[d])} for d in sorted(cnt.keys())]
    return {"daily": daily}

# ---------------- 토픽 모델링(품질 강화) ----------------
EN_STOP = {
    "the","and","to","of","in","for","on","with","at","by","from","as","is","are","be","it",
    "that","this","an","a","or","if","we","you","they","he","she","was","were","been","than",
    "into","about","over","under","per","via"
}
KO_FUNC = {
    "하다","있다","되다","통해","이번","대한","것으로","밝혔다","다양한","함께","현재",
    "기자","대표","회장","주요","기준","위해","위한","지원","전략","정책","협력","확대",
    "말했다","강조했다","대상","대상으로","최근","지난해","생활","시장","스마트","디지털","글로벌",
    "그는","그녀는","이어","한편","또한"
}

def build_topics(docs: List[str],
                 k_candidates=(7,8,9,10,11),
                 max_features=8000,
                 min_df=5,
                 topn=10) -> Dict[str, Any]:
    if not docs:
        return {"topics": []}

    vec = CountVectorizer(
        ngram_range=(1,2),
        max_features=max_features,
        min_df=min_df,
        token_pattern=r"[가-힣A-Za-z0-9_]{2,}",
        stop_words=list(EN_STOP)  # 영어 일반어 컷
    )
    X = vec.fit_transform(docs)
    vocab = vec.get_feature_names_out()
    if X.shape[1] == 0:
        return {"topics": []}

    def topic_words(lda, n_top=topn):
        comps = lda.components_
        topics = []
        for tid, comp in enumerate(comps):
            idx = comp.argsort()[-n_top:][::-1]
            words = [vocab[i] for i in idx]
            topics.append((tid, words))
        return topics

    def is_bad_topic(words):
        # 기능어/일반어 비율이 높으면 배드(임계 25%)
        bad = 0
        for w in words:
            base = w.split()[0] if " " in w else w
            if base in KO_FUNC or base.lower() in EN_STOP:
                bad += 1
            elif re.fullmatch(r"\d+$", base):
                bad += 1
        return (bad / max(1, len(words))) >= 0.25

    best_topics = None
    best_score = -1.0

    for k in k_candidates:
        lda = LatentDirichletAllocation(
            n_components=k, learning_method="batch", random_state=42, max_iter=15
        )
        _ = lda.fit_transform(X)
        ts = topic_words(lda, n_top=topn)
        good = sum(1 for _, ws in ts if not is_bad_topic(ws))
        score = good / float(k)
        if score > best_score:
            best_score = score
            best_topics = ts

    topics_obj = {"topics": []}
    if not best_topics:
        return topics_obj

    for tid, words in best_topics:
        filtered = []
        for w in words:
            base = w.split()[0] if " " in w else w
            if base in KO_FUNC or base.lower() in EN_STOP:
                continue
            if re.fullmatch(r"\d+$", base):
                continue
            filtered.append(w)
        if not filtered:
            filtered = words
        topics_obj["topics"].append({
            "topic_id": int(tid),
            "top_words": [{"word": w} for w in filtered[:topn]]
        })
    return topics_obj

# ---------------- 인사이트 요약(Gemini 폴백 포함) ----------------
def gemini_insight(api_key: str, model: str, context: Dict[str, Any],
                   max_tokens: int = 2048, temperature: float = 0.3) -> str:
    """
    Gemini로 토픽/시계열 요약 생성. 키 없거나 실패해도 폴백 요약 반환.
    """
    prompt = (
        "아래는 한국어 뉴스에서 추출한 토픽과 날짜별 기사 수 요약입니다.\n"
        "요청:\n"
        "1) 상위 토픽을 3~5개 주제로 묶어 핵심 맥락 설명(2~3문장)\n"
        "2) 최근 변화/스파이크가 있으면 2문장으로 짚기\n"
        "3) 실무 인사이트 3가지 bullet(구체적 액션)\n"
        "주의: 문장 중간에 끊지 말고 완결된 문장으로 끝내세요.\n"
        f"데이터: {json.dumps(context, ensure_ascii=False)}"
    )

    if not api_key:
        daily = context.get("timeseries", []) if isinstance(context, dict) else []
        total_days = len(daily)
        diff = 0
        if total_days >= 2:
            try:
                diff = int(daily[-1]["count"]) - int(daily[-2]["count"])
            except Exception:
                diff = 0
        return (
            f"(로컬 요약) 최근 {total_days}일 흐름 기준 간단 요약. "
            f"마지막 일자 증감 {diff}건. 상위 토픽은 산업·제품·정책 축으로 분포. "
            f"액션: 1) 상위 토픽 사례 수집 2) 급증 원인 파악 3) 파트너십/조달 검토."
        )

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        gmodel = genai.GenerativeModel(model or "gemini-1.5-flash")
        resp = gmodel.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": max_tokens or 2048,
                "temperature": temperature if temperature is not None else 0.3,
                "top_p": 0.9,
            }
        )
        text = (getattr(resp, "text", None) or "").strip()
        if not text:
            raise RuntimeError("빈 응답")
        if not re.search(r"[\.!?]$|[다요]$", text):
            try:
                resp2 = gmodel.generate_content(
                    text + "\n\n위 요약을 한 문장으로 자연스럽게 마무리해 주세요.",
                    generation_config={"max_output_tokens": 256, "temperature": temperature or 0.3, "top_p": 0.9}
                )
                add = (getattr(resp2, "text", None) or "").strip()
                if add:
                    text = (text + " " + add).strip()
            except Exception:
                pass
        return text
    except Exception as e:
        return f"(요약 생성 실패: {e}) 최근 흐름과 상위 토픽 기준으로 우선 과제를 정리하세요."

# ---------------- 메인 파이프라인 ----------------
def main():
    os.makedirs("outputs", exist_ok=True)

    # 1) 오늘 메타 문서/날짜
    res = load_today_meta()
    docs_today = res[0] if isinstance(res, tuple) and len(res) >= 1 else []
    dates_today = res[1] if isinstance(res, tuple) and len(res) >= 2 else []

    # 2) 과거 N일(warehouse) 날짜
    _, wh_dates = load_warehouse(days=30)

    # 3) 집계 대상
    docs = docs_today or []
    dates = (dates_today or []) + (wh_dates or [])

    # 4) 시계열은 '먼저' 저장(체크 C 보장)
    ts_obj = timeseries_by_date(dates)
    with open("outputs/trend_timeseries.json", "w", encoding="utf-8") as f:
        json.dump(ts_obj, f, ensure_ascii=False, indent=2)

    # 5) 토픽 모델링(문서 없어도 빈 구조 저장)
    topics_obj = build_topics(docs, k_candidates=(7,8,9,10,11), max_features=8000, min_df=5, topn=10)
    with open("outputs/topics.json", "w", encoding="utf-8") as f:
        json.dump(topics_obj, f, ensure_ascii=False, indent=2)

    # 6) 인사이트 요약(Gemini 키 없어도 안전)
    api_key = os.getenv("GEMINI_API_KEY", "")
    model_name = str(LLM.get("model", "gemini-1.5-flash"))
    summary = gemini_insight(
        api_key=api_key,
        model=model_name,
        context={
            "topics": topics_obj.get("topics", []),
            "timeseries": ts_obj.get("daily", []),
        },
        max_tokens=int(LLM.get("max_output_tokens", 2048)),
        temperature=float(LLM.get("temperature", 0.3)),
    )

    # 7) top_topics 구성(상위 단어 5개)
    top_topics = []
    for t in topics_obj.get("topics", []):
        words = [w.get("word", "") for w in (t.get("top_words") or [])][:5]
        top_topics.append({"topic_id": t.get("topic_id"), "words": words})

    # 8) 인사이트 저장(증거: 최근 14일 시계열)
    tail_14 = ts_obj.get("daily", [])[-14:] if isinstance(ts_obj.get("daily", []), list) else []
    insights_obj = {
        "summary": summary,
        "top_topics": top_topics,
        "evidence": {"timeseries": tail_14},
    }
    with open("outputs/trend_insights.json", "w", encoding="utf-8") as f:
        json.dump(insights_obj, f, ensure_ascii=False, indent=2)

    print(
        "[INFO] Module C done | topics=%d | ts_days=%d | model=%s"
        % (len(topics_obj.get("topics", [])), len(ts_obj.get("daily", [])), model_name)
    )

if __name__ == "__main__":
    main()

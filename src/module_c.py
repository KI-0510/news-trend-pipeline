# -*- coding: utf-8 -*-
import os
import json
import re
import glob
import unicodedata
import time
import datetime
from typing import List, Dict, Any, Tuple, Optional
from email.utils import parsedate_to_datetime
from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# 설정/LLM
try:
    from config import load_config, llm_config
except Exception:
    def load_config():
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

# 선택 로그 유틸(없으면 print 대체)
try:
    from utils import log_info, log_warn, log_error
except Exception:
    def log_info(*args, **kwargs): print("[INFO]", *args, kwargs if kwargs else "")
    def log_warn(*args, **kwargs): print("[WARN]", *args, kwargs if kwargs else "")
    def log_error(*args, **kwargs): print("[ERROR]", *args, kwargs if kwargs else "")

# 날짜 보정 유틸(프로젝트 공용)
try:
    from timeutil import to_kst_date_str
except Exception:
    def to_kst_date_str(s: str) -> str:
        # 간단 폴백(UTC 기준)
        return s or datetime.date.today().strftime("%Y-%m-%d")

# ---------------- 파일 유틸 ----------------
def latest(globpat: str):
    files = sorted(glob.glob(globpat))
    return files[-1] if files else None

# ---------------- 날짜/텍스트 처리 ----------------
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

def clean_text(t: str) -> str:
    if not t:
        return ""
    t = re.sub(r"<.+?>", " ", t)
    t = unicodedata.normalize("NFKC", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# ---------------- 데이터 로더 ----------------
def load_warehouse(days: int = 30) -> Tuple[List[str], List[str]]:
    """
    과거 N일 data/warehouse/*.jsonl에서 제목/날짜를 읽어 날짜 리스트만 활용
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
                        title = clean_text((obj.get("title") or "").strip())
                        if not title:
                            continue
                        d_raw = obj.get("published") or obj.get("created_at") or file_day
                        try:
                            d_std = to_kst_date_str(d_raw)
                        except Exception:
                            d_std = to_date(d_raw)
                        # 토픽은 오늘 데이터로만, 여기선 날짜만
                        dates.append(d_std)
                    except Exception:
                        continue
        except Exception:
            continue
    return docs, dates  # docs는 사용 안 함(의도)

def load_today_meta() -> Tuple[List[str], List[str]]:
    """
    최신 data/news_meta_*.json에서 title + description/body를 합쳐 오늘 문서와 날짜를 생성
    """
    meta_path = latest("data/news_meta_*.json")
    if not meta_path:
        return [], []
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            items = json.load(f)
    except Exception:
        return [], []

    docs, dates = [], []
    for it in (items or []):
        title = clean_text((it.get("title") or it.get("title_og") or "").strip())
        desc = clean_text((it.get("body") or it.get("description") or it.get("description_og") or "").strip())
        if not title and not desc:
            continue
        doc = (title + " " + desc).strip()
        if not doc:
            continue
        docs.append(doc)
        d_raw = it.get("published_time") or it.get("pubDate_raw") or ""
        try:
            d_std = to_kst_date_str(d_raw)
        except Exception:
            d_std = to_date(d_raw)
        dates.append(d_std)
    return docs, dates

# ---------------- 토픽 모델링(품질 강화) ----------------
EN_STOP = {
    "the","and","to","of","in","for","on","with","at","by","from","as","is","are","be","it",
    "that","this","an","a","or","if","we","you","they","he","she","was","were","been","than",
    "into","about","over","under","per","via"
}
KO_FUNC = {"하다","있다","되다","통해","이번","대한","것으로","밝혔다","다양한","함께","현재"}

def build_topics(docs: List[str], k_candidates=(8,9,10), max_features=8000, min_df=3, topn=10) -> Dict[str, Any]:
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
        bad = 0
        for w in words:
            base = w.split()[0] if " " in w else w
            if base in KO_FUNC or base.lower() in EN_STOP:
                bad += 1
        return (bad / max(1, len(words))) >= 0.4

    best = None
    best_score = -1.0
    best_topics = None

    for k in k_candidates:
        lda = LatentDirichletAllocation(n_components=k, learning_method="batch", random_state=42, max_iter=15)
        _ = lda.fit_transform(X)
        ts = topic_words(lda, n_top=topn)
        good = sum(1 for _, ws in ts if not is_bad_topic(ws))
        score = good / float(k)
        if score > best_score:
            best_score = score
            best = lda
            best_topics = ts

    topics_obj = {"topics": []}
    if not best_topics:
        return topics_obj

    for tid, words in best_topics:
        filtered = [w for w in words if (w.split()[0] if " " in w else w) not in KO_FUNC and w.lower() not in EN_STOP]
        if not filtered:
            filtered = words
        topics_obj["topics"].append({
            "topic_id": int(tid),
            "top_words": [{"word": w} for w in filtered[:topn]]
        })
    return topics_obj

# ---------------- 시계열 집계 ----------------
def timeseries_by_date(dates: List[str]) -> Dict[str, Any]:
    if not dates:
        return {"daily": []}
    cnt = Counter(dates)
    daily = [{"date": d, "count": int(cnt[d])} for d in sorted(cnt.keys())]
    return {"daily": daily}

# ---------------- Gemini 인사이트 요약 ----------------
def strip_code_fence(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"^```[\t ]*\w*[\t ]*\n", "", t, flags=re.M)
    t = re.sub(r"\n```[\t ]*$", "", t, flags=re.M)
    return t.strip()

def clean_json_text(t: str) -> str:
    t = strip_code_fence(t or "")
    t = t.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    t = re.sub(r"[\u0000-\u0008\u000B\u000C\u000E-\u001F\u2028\u2029]", "", t)
    t = re.sub(r",\s*(\}|\])", r"\1", t)
    t = t.lstrip("\ufeff")
    return t.strip()

def gemini_insight(api_key: str, model: str, context: Dict[str, Any], max_tokens: int = 2048, temperature: float = 0.3) -> str:
    # prompt는 기존 포맷 유지
    import google.generativeai as genai
    if not api_key:
        log_warn("gemini_insight.no_api_key")
        return "(요약 생성 실패: API 키 없음)"

    genai.configure(api_key=api_key)
    gmodel = genai.GenerativeModel(model)

    if max_tokens is None:
        max_tokens = int(LLM.get("max_output_tokens", 2048))
    if temperature is None:
        temperature = float(LLM.get("temperature", 0.3))

    prompt = (
        "아래는 한국어 뉴스에서 추출한 토픽과 날짜별 기사 수 요약입니다.\n"
        "요청:\n"
        "1) 상위 토픽을 3~5개 주제로 묶어 핵심 맥락 설명(2~3문장)\n"
        "2) 최근 변화/스파이크가 있으면 2문장으로 짚기\n"
        "3) 실무 인사이트 3가지 bullet(구체적 액션)\n"
        "주의: 문장 중간에 끊지 말고 완결된 문장으로 끝내세요.\n"
        f"데이터: {json.dumps(context, ensure_ascii=False)}"
    )

    try:
        resp = gmodel.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.9,
            }
        )
        text = (getattr(resp, "text", None) or "").strip()
    except Exception as e:
        log_error("gemini_insight.fail", err=repr(e))
        return f"(요약 생성 실패: {e})"

    # 종결성 보완
    if text and not re.search(r"[\.!?]$|[다요]$", text):
        follow = (
            "위 요약을 마무리 문장 1~2문장으로 자연스럽게 이어서 완결해 주세요. "
            "중복 없이 핵심만 덧붙여 마무리 문장으로 끝내세요."
        )
        try:
            resp2 = gmodel.generate_content(
                f"{text}\n\n{follow}",
                generation_config={
                    "max_output_tokens": max_tokens // 4,
                    "temperature": temperature,
                    "top_p": 0.9,
                }
            )
            add = (getattr(resp2, "text", None) or "").strip()
            if add:
                text = (text + " " + add).strip()
        except Exception:
            pass

    return text or "(요약 없음)"

# ---------------- 메인 파이프라인 ----------------
def main():
    os.makedirs("outputs", exist_ok=True)

    # 1) 오늘 메타 기반 문서/날짜 로드
    docs_today, dates_today = load_today_meta()

    # 2) 과거 N일(warehouse): 날짜만 사용
    _, wh_dates = load_warehouse(days=30)

    # 3) 병합: 토픽/요약은 오늘만, 시계열은 오늘+과거
    docs = docs_today
    dates = (dates_today or []) + (wh_dates or [])

    # 4) 토픽 모델링(품질 강화 버전)
    topics_obj = build_topics(docs, k_candidates=(8,9,10), max_features=8000, min_df=3, topn=10)

    # 5) 시계열 집계
    ts_obj = timeseries_by_date(dates)

    # 6) 인사이트 요약(Gemini)
    api_key = os.getenv("GEMINI_API_KEY", "")
    model_name = str(LLM.get("model", "gemini-1.5-flash"))
    context = {
        "topics": topics_obj.get("topics", []),
        "timeseries": ts_obj.get("daily", []),
    }
    summary = gemini_insight(
        api_key=api_key,
        model=model_name,
        context=context,
        max_tokens=int(LLM.get("max_output_tokens", 2048)),
        temperature=float(LLM.get("temperature", 0.3)),
    )

    # 7) top_topics 구성(상위 단어 5개)
    top_topics = []
    for t in topics_obj.get("topics", []):
        words = [w.get("word", "") for w in (t.get("top_words") or [])][:5]
        top_topics.append({"topic_id": t.get("topic_id"), "words": words})

    # 8) evidence 저장(최근 14일)
    daily = ts_obj.get("daily", [])
    tail_14 = daily[-14:] if len(daily) > 14 else daily

    insights_obj = {
        "summary": summary,
        "top_topics": top_topics,
        "evidence": {"timeseries": tail_14},
    }

    # 9) 저장
    with open("outputs/topics.json", "w", encoding="utf-8") as f:
        json.dump(topics_obj, f, ensure_ascii=False, indent=2)

    with open("outputs/trend_timeseries.json", "w", encoding="utf-8") as f:
        json.dump(ts_obj, f, ensure_ascii=False, indent=2)

    with open("outputs/trend_insights.json", "w", encoding="utf-8") as f:
        json.dump(insights_obj, f, ensure_ascii=False, indent=2)

    print(
        "[INFO] SUMMARY | C | topics_k=%d docs=%d ts_days=%d model=%s"
        % (
            len(topics_obj.get("topics", [])),
            len(docs or []),
            len(ts_obj.get("daily", [])),
            model_name,
        )
    )

if __name__ == "__main__":
    main()

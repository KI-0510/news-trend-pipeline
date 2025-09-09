# -*- coding: utf-8 -*-
import os
import json
import re
import glob
import unicodedata
import time
import datetime
from typing import List, Dict, Any, Tuple
from email.utils import parsedate_to_datetime

import tomotopy as tp
import google.generativeai as genai

from config import load_config, llm_config
CFG = load_config()
LLM = llm_config(CFG)

from utils import log_info, log_warn, log_error, retry
from timeutil import to_kst_date_str

# ------------- 파일 유틸 -------------
def latest(globpat: str):
    files = sorted(glob.glob(globpat))
    return files[-1] if files else None

# ------------- 날짜 처리 -------------
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

# ------------- 텍스트 전처리 -------------
def clean_text(t: str) -> str:
    if not t:
        return ""
    t = re.sub(r"<.+?>", " ", t)
    t = unicodedata.normalize("NFKC", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# ------------- 데이터 로더 -------------
from typing import Tuple, List
import glob
import os
import json

def load_warehouse(days: int = 30) -> Tuple[List[str], List[str]]:
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
                        
                        # 날짜 원천: published > created_at > 파일명 날짜
                        d_raw = obj.get("published") or obj.get("created_at") or file_day
                        try:
                            d_std = to_kst_date_str(d_raw)
                        except Exception:
                            d_std = to_date(d_raw)  # 안전 폴백
                        
                        docs.append(title)  # 토픽에 쓸 수도 있으니 유지
                        dates.append(d_std)
                    
                    except Exception:
                        continue
        
        except Exception:
            continue
    
    return docs, dates

def load_today_meta() -> Tuple[List[str], List[str]]:
    """
    data/news_meta_*.json에서 title+description 합쳐 문서 생성.
    반환: (docs, dates_str_YYYY-MM-DD)
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
    for it in items:
        title = clean_text(it.get("title") or it.get("title_og"))
        desc  = clean_text(it.get("description") or it.get("description_og"))
        doc = (title + " " + desc).strip()
        if doc:
            docs.append(doc)
            d_raw = it.get("published_time") or it.get("pubDate_raw") or ""
            dates.append(to_kst_date_str(d_raw))
    return docs, dates

# ------------- 토크나이저 -------------
def simple_tokenize_ko(text: str):
    toks = re.findall(r"[가-힣A-Za-z0-9]+", text or "")
    toks = [t.lower() for t in toks if len(t) >= 2]
    return toks

# ------------- LDA 토픽 모델링 -------------
def lda_topics(docs: List[str],
               k: int = 6,
               topn: int = 8,
               min_cf: int = 2,
               iters: int = 150) -> Dict[str, Any]:
    mdl = tp.LDAModel(k=k, alpha=0.1, eta=0.01, min_cf=min_cf)
    for d in docs:
        mdl.add_doc(simple_tokenize_ko(d))

    if len(mdl.docs) == 0:
        return {"topics": [], "doc_topics": []}

    mdl.burn_in = 50
    for _ in range(iters):
        mdl.train(10)

    topics = []
    for ti in range(mdl.k):
        words = mdl.get_topic_words(ti, top_n=topn)
        topics.append({
            "topic_id": ti,
            "top_words": [{"word": w, "prob": float(p)} for w, p in words]
        })

    doc_topics = []
    for di in range(len(mdl.docs)):
        dist = mdl.docs[di].get_topics(top_n=3)
        doc_topics.append([{"topic_id": tid, "prob": float(prob)} for tid, prob in dist])

    return {"topics": topics, "doc_topics": doc_topics}

# ------------- 시계열 집계 -------------
def timeseries_by_date(dates):
    counts = {}
    for d in dates:
        if not d:
            continue
        # d가 원시 문자열(ISO/헤더)일 수 있으니 KST yyyy-mm-dd로 표준화
        try:
            kst_d = to_kst_date_str(d)
        except Exception:
            # 이미 표준 yyyy-mm-dd면 그대로
            kst_d = d
        counts[kst_d] = counts.get(kst_d, 0) + 1
    daily = [{"date": k, "count": v} for k, v in sorted(counts.items())]
    return {"daily": daily}

# ------------- 인사이트 생성(Gemini) -------------
# gemini_insight 내부에 아래 래퍼 추가
@retry(max_attempts=3, backoff=0.8, exceptions=(Exception,), circuit_trip=4)
def _gen_content(model, prompt, max_tokens, temperature):
    return model.generate_content(
        prompt,
        generation_config={
            "max_output_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
        }
    )
    
def gemini_insight(api_key: str,
                   model: str,
                   context: Dict[str, Any],
                   max_tokens: int = None,
                   temperature: float = None) -> str:
    """
    LLM 설정(D1)을 그대로 반영해서 요약 생성.
    문장 종결성이 약하면 한 번 더 이어쓰기 요청으로 보완.
    """
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

    # 기존 호출 부분 교체
    try:
        resp = _gen_content(gmodel, prompt, max_tokens, temperature)
        text = (getattr(resp, "text", None) or "").strip()
    except Exception as e:
        log_error("gemini_insight.fail", err=repr(e))
        return f"(요약 생성 실패: {e})"
        
    # 종결성 보완: 문장 부호로 끝나지 않으면 이어쓰기 한 번
    if text and not re.search(r"[\.!?]$|[다요요요]$|[다]$", text):
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

# ------------- 메인 파이프라인 -------------
def main():
    os.makedirs("outputs", exist_ok=True)
    # 오늘 데이터 로드
    docs_today, dates_today = load_today_meta()
    docs_wh, dates_wh = load_warehouse(days=30)
    
    # 리스트로 변환 (필요 시)
    dates_today = list(dates_today) if dates_today else[]
    dates_wh = list(dates_wh) if dates_wh else[]
    
    # 날짜 리스트 병합
    dates = dates_today + dates_wh
    
    # 3) 토픽 모델링
    topics_obj = lda_topics(docs, k=6, topn=8, min_cf=2, iters=150)

    # 4) 시계열 집계
    ts_obj = timeseries_by_date(dates)

    # 5) 인사이트 요약
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

    # 6) top_topics 간단 구성(각 토픽 상위 단어 5개)
    top_topics = []
    for t in topics_obj.get("topics", []):
        words = [w.get("word", "") for w in (t.get("top_words") or [])][:5]
        top_topics.append({
            "topic_id": t.get("topic_id"),
            "words": words
        })

    # 7) evidence를 dict 형태로 저장(롤백 기준)
    # 최근 14일치만 잘라 넣기
    daily = ts_obj.get("daily", [])
    tail_14 = daily[-14:] if len(daily) > 14 else daily
    insights_obj = {
        "summary": summary,
        "top_topics": top_topics,
        "evidence": {
            "timeseries": tail_14
        }
    }

    # 8) 저장
    with open("outputs/topics.json", "w", encoding="utf-8") as f:
        json.dump(topics_obj, f, ensure_ascii=False, indent=2)

    with open("outputs/trend_timeseries.json", "w", encoding="utf-8") as f:
        json.dump(ts_obj, f, ensure_ascii=False, indent=2)

    with open("outputs/trend_insights.json", "w", encoding="utf-8") as f:
        json.dump(insights_obj, f, ensure_ascii=False, indent=2)

    print("[INFO] SUMMARY | C | topics_k=%d docs=%d ts_days=%d model=%s"
          % (len(topics_obj.get("topics", [])),
             len(docs),
             len(ts_obj.get("daily", [])),
             model_name))

if __name__ == "__main__":
    main()

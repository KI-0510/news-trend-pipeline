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

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# ================= 공용 스위치/로그 =================
def use_pro_mode() -> bool:
    v = os.getenv("USE_PRO", "").lower()
    if v in ("1","true","yes","y"):
        return True
    if v in ("0","false","no","n"):
        return False
    try:
        with open("config.json","r",encoding="utf-8") as f:
            cfg = json.load(f) or {}
            return bool(cfg.get("use_pro", False))
    except Exception:
        return False

def _log_mode(prefix="Module C"):
    try:
        is_pro = use_pro_mode()
    except Exception:
        is_pro = False
    mode = "PRO" if is_pro else "LITE"
    print(f"[INFO] USE_PRO={str(is_pro).lower()} → {prefix} ({mode}) 시작")

# ================= 설정 로드 =================
from src.config import load_config, llm_config
CFG = load_config()
LLM = llm_config(CFG)

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
    today = datetime.date.today()
    if not s or not isinstance(s, str): return today.strftime("%Y-%m-%d")
    s = s.strip()
    try:
        iso = s.replace("Z", "+00:00")
        dt = datetime.datetime.fromisoformat(iso)
        d = dt.date()
    except Exception:
        try:
            dt = parsedate_to_datetime(s); d = dt.date()
        except Exception:
            m = re.search(r"(\d{4}).*?(\d{1,2}).*?(\d{1,2})", s)
            if m:
                y, mm, dd = int(m.group(1)), int(m.group(2)), int(m.group(3))
                try: d = datetime.date(y, mm, dd)
                except Exception: d = today
            else:
                d = today
    if d > today: d = today
    return d.strftime("%Y-%m-%d")

# ================= 데이터 로더 =================
# src/module_c.py

# ================= 데이터 로더 =================
def select_latest_files_per_day(glob_pattern: str, days: int) -> List[str]:
    all_files = sorted(glob.glob(glob_pattern))
    daily_files = defaultdict(list)
    for f in all_files:
        date_key = os.path.basename(f)[:10]
        daily_files[date_key].append(f)
    latest_daily_files = []
    for date_key in sorted(daily_files.keys()):
        latest_file_for_day = sorted(daily_files[date_key])[-1]
        latest_daily_files.append(latest_file_for_day)
    return latest_daily_files

def load_today_meta() -> Tuple[List[str], List[str]]:
    meta_path = latest("data/news_meta_*.json")
    if not meta_path: return [], []
    docs, dates = [], []
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            items = json.load(f) or []
    except Exception:
        return [], []
    for it in items:
        title = clean_text((it.get("title") or it.get("title_og") or "").strip())
        desc  = clean_text((it.get("body") or it.get("description") or it.get("description_og") or "").strip())
        doc = (title + " " + desc).strip()
        if not doc: continue
        d_raw = it.get("published_time") or it.get("pubDate_raw") or ""
        docs.append(doc)
        dates.append(to_date(d_raw))
    return docs, dates

# load_warehouse 함수는 이제 파일 경로 목록만 반환하도록 단순화합니다.
def load_warehouse_paths(days: int = 30) -> List[str]:
    # 하루치 여유분을 포함하여 D+1일자 파일을 가져올 수 있도록 합니다.
    return select_latest_files_per_day("data/warehouse/*.jsonl", days=(days + 1))

# ================= 시계열 =================
def calculate_stable_timeseries(warehouse_files: List[str]) -> Dict[str, Any]:
    """
    D일자와 D+1일자 파일을 기반으로 D일의 최종 기사 수를 계산하는 안정적인 시계열 분석 함수
    """
    file_map = {os.path.basename(f)[:10]: f for f in warehouse_files}
    if not file_map:
        return {"daily": []}
    
    sorted_dates = sorted(file_map.keys())
    start_date = datetime.datetime.strptime(sorted_dates[0], "%Y-%m-%d").date()
    # 마지막 날짜는 D+1일이 없으므로, 그 전날까지만 계산합니다.
    end_date = datetime.datetime.strptime(sorted_dates[-1], "%Y-%m-%d").date() - datetime.timedelta(days=1)
    
    daily_counts = []
    current_date = start_date

    while current_date <= end_date:
        date_str_d = current_date.strftime("%Y-%m-%d")
        date_str_d_plus_1 = (current_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        
        count = 0
        
        files_to_check = []
        if date_str_d in file_map:
            files_to_check.append(file_map[date_str_d])
        if date_str_d_plus_1 in file_map:
            files_to_check.append(file_map[date_str_d_plus_1])
            
        for fp in files_to_check:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            obj = json.loads(line)
                            d_raw = obj.get("published") or obj.get("created_at") or os.path.basename(fp)[:10]
                            published_date = to_date(d_raw)
                            
                            if published_date == date_str_d:
                                count += 1
                        except Exception:
                            continue
            except Exception:
                continue
                
        daily_counts.append({"date": date_str_d, "count": count})
        current_date += datetime.timedelta(days=1)
        
    return {"daily": daily_counts}

# ================= 불용/컷(토픽 공통) =================
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
    "등을", "따라", "있도록", "지난", "특히", "대비", "아니라", "만에", "의원은", "라고",
    "있습니다", "관련", "한다", "진행한다", "예정이다", "가능하다", "있었다",
    "이상", "넘어", "제공한다", "같은", "했다", "많은", "그리고", "같다", "우리", "하고",
    "때문에", "이렇게", "이런", "등이", "각각"
}

def is_bad_token(base: str) -> bool:
    if base in KO_FUNC or base.lower() in EN_STOP: return True
    if re.fullmatch(r"\d+$", base): return True
    if re.fullmatch(r"\d{1,2}$", base): return True
    if re.fullmatch(r"\d{1,2}월$", base): return True
    if re.fullmatch(r"\d{1,2}일$", base): return True
    if re.search(r"(억|조|달러|원)$", base): return True
    return False

# ================= 내부 헬퍼: prob 세이프가드 =================
def _ensure_prob_payload(obj: dict, topn: int = 10, decay: float = 0.95, floor: float = 0.2) -> dict:
    """
    - 저장 직전에 호출하여 모든 토픽 단어에 prob를 강제 주입.
    - 문자열 리스트면 dict 리스트로 변환 후 주입.
    - 이미 prob가 있더라도 0/음수/NaN은 최소값 보정.
    """
    topics = obj.get("topics") or []
    for t in topics:
        ws = t.get("top_words") or []
        # 문자열 리스트 → dict 변환
        if ws and isinstance(ws[0], str):
            ws = [{"word": w} for w in ws if w]
            t["top_words"] = ws
        # 주입/보정
        if not ws:
            continue
        all_have = all((isinstance(w, dict) and ("prob" in w)) for w in ws)
        if all_have:
            for w in ws:
                try:
                    p = float(w.get("prob", 0))
                    if not (p > 0):
                        w["prob"] = 1e-6
                except Exception:
                    w["prob"] = 1e-6
        else:
            for rank, w in enumerate(ws[:topn], start=0):
                if not isinstance(w, dict):
                    continue
                prob = max(floor, decay ** rank)
                w["prob"] = float(prob)
    return obj

# ================= Lite 토픽(LDA) — prob 포함 =================
def build_topics_lite(docs: List[str],
                      max_features=8000,
                      topn=10) -> Dict[str, Any]:
    # --- 제안 내용 반영 시작 ---
    # 1. config.json에서 파라미터 읽어오기
    k_candidates = CFG.get("topic_k_candidates", [8, 10, 12, 14])
    min_df_val = int(CFG.get("topic_min_df", 7))
    max_df_val = float(CFG.get("topic_max_df", 0.85))

    # 2. config.json 및 사전의 불용어 목록 통합
    phrase_stop_cfg = set(CFG.get("phrase_stop", []) or [])
    stopwords_cfg = set(CFG.get("stopwords", []) or [])
    
    # phrase_stop은 텍스트에서 먼저 제거 (코드는 함수 후반부에 이미 존재)
    
    # CountVectorizer에 적용할 최종 불용어 목록
    final_stopwords = list(set(EN_STOP) | set(KO_FUNC) | stopwords_cfg)
    
    print(f"[DEBUG][C] LITE builder 진입 | k_candidates={k_candidates} min_df={min_df_val} max_df={max_df_val}")
    
    # 3. 문서 리스트에서 phrase_stop 먼저 제거
    processed_docs = []
    if docs:
        for doc in docs:
            temp_doc = doc
            for phrase in phrase_stop_cfg:
                temp_doc = temp_doc.replace(phrase, " ")
            processed_docs.append(temp_doc)
    else:
        return {"topics": []}

    vec = CountVectorizer(
        ngram_range=(1, 3), # 3-gram(tri-gram) 포함
        max_features=max_features,
        min_df=min_df_val, # 상향된 min_df 적용
        max_df=max_df_val,
        token_pattern=r"[가-힣A-Za-z0-9_]{2,}",
        stop_words=final_stopwords # 통합된 불용어 목록 적용
    )
    # --- 제안 내용 반영 끝 ---

    X = vec.fit_transform(processed_docs)
    vocab = vec.get_feature_names_out()
    if X.shape[1] == 0:
        return {"topics": []}

    # 이하 로직은 동일
    def topic_pairs(lda, n_top=topn):
        comps = lda.components_
        topics = []
        for tid, comp in enumerate(comps):
            idx = comp.argsort()[-max(n_top, 30):][::-1]
            pairs = [(vocab[i], float(comp[i])) for i in idx]
            topics.append((tid, pairs))
        return topics

    def bad_ratio_from_pairs(pairs):
        bad = 0
        for w, _s in pairs:
            base = w.split()[0] if " " in w else w
            if is_bad_token(base): bad += 1
        return bad / max(1, len(pairs))

    best_topics = None; best_score = -1.0
    for k in k_candidates:
        lda = LatentDirichletAllocation(n_components=k, learning_method="batch", random_state=42, max_iter=15)
        _ = lda.fit_transform(X)
        ts = topic_pairs(lda, n_top=topn)
        good = sum(1 for _, pairs in ts if bad_ratio_from_pairs(pairs) < 0.20)
        score = good / float(k)
        if score > best_score:
            best_score = score; best_topics = ts

    topics_obj = {"topics": []}
    if not best_topics:
        print("[DEBUG][C] LITE 생성 완료 | topics=0")
        return topics_obj

    for tid, pairs in best_topics:
        kept = []
        for w, s in pairs:
            base = w.split()[0] if " " in w else w
            if is_bad_token(base):
                continue
            kept.append((w, s))
        if not kept:
            kept = pairs[:]

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

        topics_obj["topics"].append({"topic_id": int(tid), "top_words": payload})
    print(f"[DEBUG][C] LITE 생성 완료 | topics={len(topics_obj.get('topics', []))}")
    return topics_obj


# ================= Pro 토픽(BERTopic) — prob 포함 =================
def pro_build_topics_bertopic(docs, topn=10):
    print("[DEBUG][C] PRO builder 진입")
    try:
        from bertopic import BERTopic
        from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
        from sentence_transformers import SentenceTransformer
        from sklearn.feature_extraction.text import CountVectorizer
        import numpy as np
    except Exception as e:
        raise RuntimeError(f"Pro 토픽 모드 준비 실패(패키지 없음): {e}")

    # --- 주제 필터링 로직 시작 ---
    core_keywords = set(CFG.get("pro_topic_core_keywords", []))
    min_core_keyword_match = 2

    if not docs or not core_keywords:
        print("[DEBUG][C] PRO 생성 완료 | topics=0 (docs or core_keywords empty)")
        return {"topics": []}

    filtered_docs = []
    for doc in docs:
        doc_lower = doc.lower()
        match_count = sum(1 for keyword in core_keywords if keyword in doc_lower)
        if match_count >= min_core_keyword_match:
            filtered_docs.append(doc)
    
    print(f"[DEBUG][C] Core Keyword Filtering: {len(docs)} -> {len(filtered_docs)} docs")

    if len(filtered_docs) < 10:
        print(f"[WARN] Pro 토픽 분석을 위한 문서 수가 부족하여({len(filtered_docs)}개), Lite로 폴백합니다.")
        return build_topics_lite(docs, topn=topn)
    # --- 주제 필터링 로직 끝 ---

    emb = SentenceTransformer("jhgan/ko-sroberta-multitask")
    vectorizer_model = CountVectorizer(
        ngram_range=(1,3),
        min_df=2,
        token_pattern=r"[가-힣A-Za-z0-9_]{2,}",
        stop_words=list(set(EN_STOP)|set(KO_FUNC))
    )
    rep = [KeyBERTInspired(top_n_words=15), MaximalMarginalRelevance(diversity=0.5)]
    
    min_topic_size_pro = int(CFG.get("pro_topic_min_size", 5))
    nr_topics_pro = CFG.get("pro_nr_topics", None)

    model = BERTopic(
        embedding_model=emb,
        vectorizer_model=vectorizer_model,
        representation_model=rep,
        min_topic_size=min_topic_size_pro,
        nr_topics=nr_topics_pro,
        calculate_probabilities=False,
        verbose=False
    )
    topics, probs = model.fit_transform(filtered_docs)
    
    try:
        # 아래 docs를 filtered_docs로 수정
        model.reduce_outliers(filtered_docs, topics, probabilities=probs, strategy="c-tf-idf", threshold=0.08)
    except Exception:
        pass
    try:
        # 아래 docs를 filtered_docs로 수정
        model.merge_topics(filtered_docs, topics, threshold=0.88)
    except Exception:
        pass
        
    topics_obj = {"topics": []}
    try:
        ctfidf = model.c_tf_idf_
        terms = model.vectorizer_model.get_feature_names_out()
        for tid in sorted(set(topics)):
            if tid == -1:
                continue
            vec = ctfidf[tid].toarray().ravel()
            idx = vec.argsort()[::-1][:max(40, topn)]
            pairs = [(terms[i], float(vec[i])) for i in idx]

            kept = []
            for w, s in pairs:
                base = w.split()[0] if " " in w else w
                if is_bad_token(base):
                    continue
                kept.append((w, s))
            if not kept:
                kept = pairs[:]

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
        print(f"[DEBUG][C] PRO 생성 완료 | topics={len(topics_obj.get('topics', []))}")
        return topics_obj

    except Exception:
        # 폴백: get_topics 사용(여기도 prob 보장)
        topic_info = model.get_topics()
        for tid, items in topic_info.items():
            if tid == -1:
                continue
            head = items[:max(40, topn)]
            kept = []
            for w, s in head:
                base = w.split()[0] if " " in w else w
                if is_bad_token(base):
                    continue
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
        print(f"[DEBUG][C] PRO 생성 완료(폴백) | topics={len(topics_obj.get('topics', []))}")
        return topics_obj

# ================= 인사이트 요약 =================
def gemini_insight(api_key: str, model: str, context: Dict[str, Any],
                   max_tokens: int = 2048, temperature: float = 0.3) -> str:
    # prompt 부분을 수정하여 keywords 정보를 활용하도록 합니다.
    prompt = (
        "아래는 한국어 기술/시장 뉴스에서 추출한 데이터입니다: (1) 주요 토픽, (2) 날짜별 기사 수, (3) 최상위 키워드.\n"
        "당신은 디스플레이 산업의 사업 전략 전문가입니다. 이 데이터를 종합하여 '데일리 인텔리전스 브리핑'을 생성해주세요.\n"
        "요청:\n"
        "1. 핵심 맥락: 토픽과 키워드를 연결하여, 시장의 가장 중요한 흐름 2~3가지를 설명하세요.\n"
        "2. 최근 변화/스파이크: 시계열 데이터의 급증 지점을 언급하고, 그 원인을 키워드와 토픽을 근거로 추론하세요.\n"
        "3. 실무 인사이트: 분석 결과를 바탕으로 사업 개발, 기술 기획 담당자가 실행할 수 있는 구체적인 액션 아이템 3가지를 제안하세요.\n"
        "주의: 문장 중간에 끊지 말고 완결된 문장으로 간결하게 작성하세요.\n\n"
        f"데이터: {json.dumps(context, ensure_ascii=False)}"
    )
    if not api_key:
        daily = context.get("timeseries", []) if isinstance(context, dict) else []
        total_days = len(daily)
        diff = 0
        if total_days >= 2:
            try: diff = int(daily[-1]["count"]) - int(daily[-2]["count"])
            except Exception: diff = 0
        return (f"(로컬 요약) 최근 {total_days}일 흐름 기준 간단 요약. 마지막 일자 증감 {diff}건. "
                f"상위 토픽은 산업·제품·정책 축으로 분포. 액션: 1) 상위 토픽 사례 수집 2) 급증 원인 파악 3) 파트너십/조달 검토.")
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        gmodel = genai.GenerativeModel(model or "gemini-1.5-flash")
        resp = gmodel.generate_content(
            prompt,
            generation_config={"max_output_tokens": max_tokens or 2048, "temperature": temperature if temperature is not None else 0.3, "top_p": 0.9}
        )
        text = (getattr(resp, "text", None) or "").strip()
        if not text: raise RuntimeError("빈 응답")
        if not re.search(r"[\.!?]$|[다요]$", text):
            try:
                resp2 = gmodel.generate_content(
                    text + "\n\n위 요약을 한 문장으로 자연스럽게 마무리해 주세요.",
                    generation_config={"max_output_tokens": 256, "temperature": temperature or 0.3, "top_p": 0.9}
                )
                add = (getattr(resp2, "text", None) or "").strip()
                if add: text = (text + " " + add).strip()
            except Exception:
                pass
        return text
    except Exception as e:
        return f"(요약 생성 실패: {e}) 최근 흐름과 상위 토픽 기준으로 우선 과제를 정리하세요."

# ================= 강/약 신호(기울기 반영) =================
def _term_counts_in_docs(docs: list, terms: list) -> dict:
    res = {}
    for t in terms:
        tl = (t or "").lower()
        if not tl: continue
        c = 0
        for d in docs:
            if tl in (d or "").lower():
                c += 1
        res[t] = c
    return res

def _z_like(cur: float, mean: float, sd: float) -> float:
    if sd <= 0: return 0.0
    return (cur - mean) / sd

def export_trend_and_weak_signals(docs: list, dates: list, keywords_obj: dict):
    import csv
    os.makedirs("outputs/export", exist_ok=True)
    terms = [k.get("keyword","") for k in (keywords_obj.get("keywords") or [])[:80]]
    terms = [t for t in terms if t and len(t) >= 2]

    days = sorted(set(dates or []))[-28:]
    day_docs = {d: [] for d in days}
    for d, doc in zip(dates, docs):
        if d in day_docs:
            day_docs[d].append(doc)

    def count_on_day(term, day):
        tl = term.lower()
        return sum(1 for doc in day_docs.get(day, []) if tl in (doc or "").lower())

    def series_for_term(term):
        return [count_on_day(term, d) for d in days]

    def rolling_avg(arr, k):
        if len(arr) < k: return 0.0
        return sum(arr[-k:]) / float(k)

    totals_all = _term_counts_in_docs(docs, terms)
    vals = list(totals_all.values()) or [0]
    mean_all = sum(vals)/max(1,len(vals))
    sd_all = (sum((x-mean_all)**2 for x in vals)/max(1,len(vals)))**0.5

    rows = []
    for t in terms:
        s = series_for_term(t)
        cur = s[-1] if s else 0
        prev = s[-2] if len(s) >= 2 else 0
        diff = cur - prev
        ma7 = rolling_avg(s, 7)
        total = sum(s)
        z = _z_like(cur, mean_all, sd_all) if sd_all > 0 else 0.0
        if len(s) >= 14:
            front7 = sum(s[-14:-7]) / 7.0
            back7  = sum(s[-7:]) / 7.0
        else:
            half = max(1, len(s)//2)
            front7 = sum(s[:half]) / float(half)
            back7  = sum(s[half:]) / float(max(1,len(s)-half))
        slope = back7 - front7
        rows.append({"term": t, "cur": cur, "prev": prev, "diff": diff, "ma7": round(ma7,3),
                     "z_like": round(z,3), "total": total, "slope": round(slope,3)})

    cur_sorted = sorted([r["cur"] for r in rows], reverse=True)
    z_sorted = sorted([r["z_like"] for r in rows], reverse=True)
    total_sorted = sorted([r["total"] for r in rows])

    def quantile(vs, p):
        if not vs: return 0
        i = max(0, min(len(vs)-1, int(len(vs)*p)))
        return vs[i]

    cur_q90   = quantile(cur_sorted, 0.1)
    z_q80     = quantile(z_sorted, 0.2)
    total_q70 = quantile(total_sorted, 0.3)
    total_q50 = quantile(total_sorted, 0.5)
    z_q60     = quantile(z_sorted, 0.4)

    trend = [r for r in rows if r["cur"] >= cur_q90 and r["z_like"] >= z_q80 and r["total"] >= total_q70]
    weak  = [r for r in rows if r["total"] <= total_q50 and r["z_like"] >= z_q60 and r["cur"] < cur_q90 and r["slope"] > 0]

    trend_terms = {r["term"] for r in trend}
    weak = [r for r in weak if r["term"] not in trend_terms]

    with open("outputs/export/trend_strength.csv","w",encoding="utf-8",newline="") as f:
        w = csv.DictWriter(f, fieldnames=["term","cur","prev","diff","ma7","z_like","total","slope"])
        w.writeheader(); [w.writerow(r) for r in trend]
    with open("outputs/export/weak_signals.csv","w",encoding="utf-8",newline="") as f:
        w = csv.DictWriter(f, fieldnames=["term","cur","prev","diff","ma7","z_like","total","slope"])
        w.writeheader(); [w.writerow(r) for r in weak]

# ================= 메인 =================
def main():
    _log_mode("Module C")
    os.makedirs("outputs", exist_ok=True)

    # 1. 데이터 로드 (변경된 함수 이름으로 호출)
    docs_today, _ = load_today_meta()
    warehouse_paths = load_warehouse_paths(days=30) 
    
    # 2. 새로운 방식으로 시계열 계산
    ts_obj = calculate_stable_timeseries(warehouse_paths)
    with open("outputs/trend_timeseries.json", "w", encoding="utf-8") as f:
        json.dump(ts_obj, f, ensure_ascii=False, indent=2)

    # 3. 토픽 분석은 오늘 수집한 데이터만 사용
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

    # 4. 인사이트 생성
    try:
        with open("outputs/keywords.json", "r", encoding="utf-8") as f:
            keywords_obj = json.load(f)
    except Exception:
        keywords_obj = {"keywords": []}
    top_keywords = [k.get("keyword") for k in keywords_obj.get("keywords", [])[:10]]

    api_key = os.getenv("GEMINI_API_KEY", "")
    model_name = str(LLM.get("model", "gemini-2.0-flash"))
    summary = gemini_insight(
        api_key=api_key,
        model=model_name,
        context={"topics": topics_obj.get("topics", []), "timeseries": ts_obj.get("daily", []), "keywords": top_keywords},
        max_tokens=int(LLM.get("max_output_tokens", 2048)),
        temperature=float(LLM.get("temperature", 0.3)),
    )

    top_topics = []
    for t in topics_obj.get("topics", []):
        words = [w.get("word", "") for w in (t.get("top_words") or [])][:5]
        top_topics.append({"topic_id": t.get("topic_id"), "words": words})
    tail_14 = ts_obj.get("daily", [])[-14:] if isinstance(ts_obj.get("daily", []), list) else []
    insights_obj = {"summary": summary, "top_topics": top_topics, "evidence": {"timeseries": tail_14}}
    with open("outputs/trend_insights.json", "w", encoding="utf-8") as f:
        json.dump(insights_obj, f, ensure_ascii=False, indent=2)
    
    # 5. 강/약 신호 분석은 이제 signal_export.py에서 독립적으로 수행되므로 여기서 호출하지 않습니다.

    import datetime
    meta = {"module": "C", "mode": "PRO" if use_pro_mode() else "LITE", "time_utc": datetime.datetime.utcnow().isoformat() + "Z"}
    with open("outputs/run_meta_c.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[INFO] Module C done | topics=%d | ts_days=%d | model=%s" % (len(topics_obj.get("topics", [])), len(ts_obj.get("daily", [])), model_name))
    

if __name__ == "__main__":
    main()

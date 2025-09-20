# -*- coding: utf-8 -*-
# Module C – Topics, Timeseries, Insights
# - 전처리: module_b와 동일 철학(불용어/정규식/조사·어미 컷 + alias 정규화)
# - Lite=LDA(튜닝: k 후보 + 확률 정규화), Pro=BERTopic(min_topic_size + 확률 정규화)
# - 토픽 후처리: 불용어/지명/숫자·날짜·단위 컷 + 어워드 과다 토픽 억제
# - 인사이트: 도메인 겹침/노이즈 겹침 가중 점수로 상위 토픽 선정
# - LLM(Gemini)로 topic_name/2~3문장 요약(insight) 선택적 생성

import os, re, glob, json, datetime, unicodedata, random
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter
import numpy as np

# Lite
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Pro (optional)
try:
    from bertopic import BERTopic
    from umap import UMAP
    from hdbscan import HDBSCAN
except Exception:
    BERTopic, UMAP, HDBSCAN = None, None, None

# Optional LLM (Gemini)
def _maybe_import_gemini():
    try:
        import google.generativeai as genai
        return genai
    except Exception:
        return None

# -------------------------
# Config loader
# -------------------------
def load_config(path: str = "config.json") -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}

def llm_config(cfg: dict) -> dict:
    llm = cfg.get("llm") or {}
    return {
        "provider": (llm.get("provider") or "").lower(),
        "model": llm.get("model", "gemini-1.5-flash"),
        "max_output_tokens": int(llm.get("max_output_tokens", 1024)),
        "temperature": float(llm.get("temperature", 0.3)),
    }

CFG = load_config()
LLM = llm_config(CFG)

# -------------------------
# Mode / logging
# -------------------------
def use_pro_mode() -> bool:
    v = os.getenv("USE_PRO", "").lower()
    if v in ("1","true","yes","y"): return True
    if v in ("0","false","no","n"): return False
    return bool(CFG.get("use_pro", False))

def _log_mode():
    print(f"[INFO] USE_PRO={str(use_pro_mode()).lower()} → Module C 시작")

# -------------------------
# IO helpers
# -------------------------
def latest(globpat: str) -> Optional[str]:
    files = sorted(glob.glob(globpat))
    return files[-1] if files else None

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def load_json(path, default=None):
    if default is None: default = {}
    try:
        with open(path,"r",encoding="utf-8") as f: return json.load(f)
    except Exception: return default

def save_json(path, obj):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path,"w",encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_lines(path) -> List[str]:
    try:
        with open(path, encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]
    except Exception:
        return []

# -------------------------
# Patterns / stopwords (module_b와 동일 자원 사용)
# -------------------------
def compile_patterns(cfg: dict):
    rp = cfg.get("regex_patterns", {}) or {}
    def _c(key, default):
        try:
            return re.compile(rp.get(key, default))
        except Exception:
            return re.compile(default)
    return {
        "NUMERIC_ONLY": _c("NUMERIC_ONLY", r"^\d+$"),
        "DATE_PAT": _c("DATE_PAT", r"^\d{1,2}일$|^\d{1,2}월$|^\d{4}년$|^\d{4}$"),
        "CURRENCY_PAT": _c("CURRENCY_PAT", r"^[0-9,\.]+(원|달러|유로|엔|위안|억원|조원)$"),
        "PERSON_NAME_PAT": _c("PERSON_NAME_PAT", r"^[가-힣]{2,4}$"),
        "UNIT_TOKEN_PAT": _c("UNIT_TOKEN_PAT", r"^\d+(?:[.,]\d+)?(%|배|건|개|명|곳|회|대|종|분기|세대|nm|나노|인치)$"),
    }

_LOCATION_CORE = {
    "서울","부산","대구","인천","광주","대전","울산","세종",
    "경기","경기도","강원","강원도","충북","충남","전북","전남","경북","경남",
    "제주","제주도","수원","용인","성남","고양","화성","부천","안산","안양","남양주",
}
_LOCATION_SUFFIX = {"도","시","군","구","읍","면","동","리"}

def is_location_token(tok: str) -> bool:
    if not tok: return False
    if tok in _LOCATION_CORE: return True
    if len(tok) >= 2 and tok[-1] in _LOCATION_SUFFIX: return True
    return False

def merged_stopwords(cfg: dict) -> Tuple[List[str], List[str]]:
    defaults = cfg.get("keyword_extraction_defaults", {}) or {}
    phrase = sorted(
        set(cfg.get("phrase_stop", []) or [])
        | set(load_lines("data/dictionaries/phrase_stopwords.txt"))
    )
    stop = sorted(
        set(cfg.get("stopwords", []) or [])
        | set(load_lines("data/dictionaries/stopwords_ext.txt"))
        | set(defaults.get("MORE_STOP", []))
        | set(defaults.get("EN_STOP", []))
    )
    return phrase, stop

def to_stopword_list(sw) -> List[str]:
    items = []
    seen = set()
    for s in (sw or []):
        t = str(s).strip()
        if not t or t in seen:
            continue
        seen.add(t)
        items.append(t)
    return items

# -------------------------
# Alias normalization
# -------------------------
ALIAS_MAP: Dict[str,str] = (CFG.get("alias") or {})

def normalize_alias_token(tok: str) -> str:
    if not tok: return tok
    if tok in ALIAS_MAP:
        return ALIAS_MAP[tok]
    low = tok.lower()
    if low in ALIAS_MAP:
        return ALIAS_MAP[low]
    return tok

def apply_alias(tokens: List[str]) -> List[str]:
    return [normalize_alias_token(t) for t in tokens]

# -------------------------
# Text normalize / tokenize (module_b와 동일 철학)
# -------------------------
_HAN_ALNUM = re.compile(r"[가-힣A-Za-z0-9]+")

_JOSA = ("은","는","이","가","을","를","에","에서","으로","로","과","와","에게","한테","께","이나","나","든지","까지","부터","라도","마저","밖에","뿐")
_EOMI = ("했다","하였다","한다","했다가","하며","해서","하는","되다","되면","되었","된다","되니","되어","됐다","됐다가","했다며",
         "이라","이라며","이라서")

def basic_normalize(txt: str) -> str:
    t = (txt or "").strip()
    t = unicodedata.normalize("NFKC", t)
    toks = _HAN_ALNUM.findall(t)
    return " ".join(toks)

def nounish_strip(sentence: str) -> str:
    toks, out = sentence.split(), []
    for tk in toks:
        t = tk
        for suf in _JOSA:
            if t.endswith(suf) and len(t) >= len(suf) + 2:
                t = t[:-len(suf)]; break
        for suf in _EOMI:
            if t.endswith(suf) and len(t) >= len(suf) + 2:
                t = t[:-len(suf)]; break
        if len(t) >= 2: out.append(t)
    return " ".join(out)

def preprocess_docs(raw_docs: List[str], phrase_stop: List[str], stopwords: List[str], P: dict) -> List[str]:
    ps, sw = set(phrase_stop or []), set(stopwords or [])
    out = []
    for d in raw_docs:
        if not d: continue
        t = basic_normalize(d)
        for ph in ps:
            if ph: t = t.replace(ph, " ")
        t = nounish_strip(t)
        toks = []
        for w in t.split():
            if w in sw: continue
            if w in {"은","는","이","가","을","를","에","에서","으로","로","과","와","에게","한테","께","이나","나","든지","까지","부터","라도","마저","밖에","뿐"}:
                continue
            for suf in ("이라고","라고","며","면서","지만","다는","라고도","다고","고"):
                if w.endswith(suf) and len(w) >= len(suf)+2:
                    w = w[:-len(suf)]; break
            if w in sw: continue
            if P["NUMERIC_ONLY"].match(w) or P["DATE_PAT"].match(w) or P["CURRENCY_PAT"].match(w) or P["UNIT_TOKEN_PAT"].match(w):
                continue
            if len(w) < 2: continue
            toks.append(w)
        # alias 정규화
        toks = apply_alias(toks)
        t2 = " ".join(toks)
        if t2: out.append(t2)
    return out

# -------------------------
# Corpus / timeseries
# -------------------------
def build_docs_from_meta(items: List[dict]) -> List[str]:
    docs = []
    for it in items:
        title = (it.get("title") or it.get("title_og") or "").strip()
        desc  = (it.get("description_short") or it.get("description") or it.get("description_og") or "").strip()
        txt = (title + " " + desc).strip()
        if txt: docs.append(txt)
    return docs

def to_date(s: str) -> str:
    today = datetime.date.today()
    if not s or not isinstance(s, str): return today.strftime("%Y-%m-%d")
    s = s.strip()
    try:
        iso = s.replace("Z", "+00:00")
        dt = datetime.datetime.fromisoformat(iso)
        d = dt.date()
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

def make_timeseries(items: List[dict]) -> Dict[str, Any]:
    daily = Counter()
    for it in items:
        d_raw = it.get("published_time") or it.get("pubDate_raw") or ""
        d = to_date(d_raw)
        daily[d] += 1
    rows = [{"date": d, "count": int(c)} for d, c in sorted(daily.items())]
    return {"daily": rows}

# -------------------------
# Topic modeling – Lite (LDA) with tuning and proper probs
# -------------------------
def topic_diversity(components: np.ndarray, terms: np.ndarray, topn: int = 10) -> float:
    top_words = []
    for k in range(components.shape[0]):
        idx = components[k].argsort()[::-1][:topn]
        top_words.extend(terms[idx])
    uniq = len(set(top_words))
    total = len(top_words) if top_words else 1
    return uniq / total

def fit_best_lda(X, terms, k_list, random_state=42, max_iter=20) -> Tuple[LatentDirichletAllocation, np.ndarray]:
    best = None
    best_score = -1e18
    for k in k_list:
        lda = LatentDirichletAllocation(
            n_components=max(2, int(k)),
            learning_method="batch",
            random_state=random_state,
            max_iter=max_iter
        )
        lda.fit(X)
        ll = float(lda.score(X))       # log-likelihood (higher better)
        div = topic_diversity(lda.components_, terms, topn=10)  # 0~1
        score = ll + 5.0 * div
        if score > best_score:
            best, best_score = lda, score
    return best, best.components_

def topics_lite(corpus: List[str], stopwords: List[str], cfg: dict, topn=10, random_state=42) -> Dict[str, Any]:
    if not corpus:
        return {"topics": []}
    min_df = int(cfg.get("topic_min_df", 3))
    max_df = float(cfg.get("topic_max_df", 0.95))
    k_cands = cfg.get("lda_k_candidates", [7,8,9,10,11])

    cv = CountVectorizer(
        ngram_range=(1,3), min_df=min_df, max_df=max_df,
        token_pattern=r"[가-힣A-Za-z0-9_]{2,}",
        stop_words=to_stopword_list(stopwords)
    )
    X = cv.fit_transform(corpus)
    if X.shape[1] == 0:
        return {"topics": []}
    terms = cv.get_feature_names_out()

    if isinstance(k_cands, list) and len(k_cands) >= 2:
        lda, comps = fit_best_lda(X, terms, k_cands, random_state=random_state, max_iter=20)
    else:
        k = int(cfg.get("n_topics", 8))
        lda = LatentDirichletAllocation(
            n_components=max(2, k),
            learning_method="batch",
            random_state=random_state,
            max_iter=20
        )
        lda.fit(X)
        comps = lda.components_

    topics = []
    for tid in range(comps.shape[0]):
        idx_sorted = comps[tid].argsort()[::-1]
        idx_top = idx_sorted[:max(10, topn)]
        vals = comps[tid][idx_top].astype(float)

        s = float(vals.sum())
        if s > 0:
            probs = (vals / s).tolist()
        else:
            probs = [0.0] * len(vals)

        if (max(probs) - min(probs)) < 1e-9:
            decay = 0.95
            probs = [max(0.2, decay**r) for r in range(len(vals))]
            s2 = sum(probs) or 1.0
            probs = [p / s2 for p in probs]

        words = []
        for j, i_term in enumerate(idx_top[:topn]):
            w = terms[i_term]
            p = float(probs[j])
            words.append({"word": w, "prob": p})
        topics.append({"topic_id": int(tid), "top_words": words})
    return {"topics": topics}

# -------------------------
# Topic modeling – Pro (BERTopic) with proper probs
# -------------------------
def topics_pro(corpus: List[str], cfg: dict, topn=10, random_state=42) -> Dict[str, Any]:
    if BERTopic is None or not corpus:
        return {"topics": []}
    n_neighbors = int(cfg.get("umap_neighbors", 15))
    min_cluster_size = int(cfg.get("min_cluster_size", 12))
    min_topic_size = int(cfg.get("min_topic_size", 15))

    umap_model = UMAP(n_neighbors=n_neighbors, n_components=10, metric="cosine", low_memory=True, random_state=random_state)
    hdb_model = HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
    tm = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdb_model,
        calculate_probabilities=False,
        verbose=False,
        min_topic_size=min_topic_size,
        random_state=random_state
    )
    topics, _ = tm.fit_transform(corpus)
    info = tm.get_topic_info()

    out = []
    for t in info["Topic"].tolist():
        if int(t) < 0:
            continue
        top_pairs = tm.get_topic(int(t))[:max(10, topn)]  # [(word, ctfidf), ...]
        terms = [w for (w, _p) in top_pairs]
        vals = np.array([float(_p or 0.0) for (_w, _p) in top_pairs], dtype=float)

        s = float(vals.sum())
        if s > 0:
            probs = (vals / s).tolist()
        else:
            probs = [0.0] * len(vals)

        if (max(probs) - min(probs)) < 1e-9:
            decay = 0.95
            probs = [max(0.2, decay**r) for r in range(len(vals))]
            s2 = sum(probs) or 1.0
            probs = [p / s2 for p in probs]

        words = [{"word": terms[i], "prob": float(probs[i])} for i in range(min(len(terms), topn))]
        out.append({"topic_id": int(t), "top_words": words})
    return {"topics": out}

# -------------------------
# Topic post-filter + award-heavy drop
# -------------------------
AWARD_HINTS = {"수상","본상","금상","은상","대상","공모전","어워드","레드닷","red dot","if","idea","디자인"}

def post_filter_topics(topics_obj: Dict[str, Any], stopwords: List[str], P: dict) -> Dict[str, Any]:
    sw = set(stopwords or [])
    filtered = []
    for t in topics_obj.get("topics", []):
        kept = []
        for w in (t.get("top_words") or []):
            ww = (w.get("word") or "").strip()
            if not ww: continue
            if ww in sw: continue
            if is_location_token(ww): continue
            if P["NUMERIC_ONLY"].match(ww) or P["DATE_PAT"].match(ww) or P["CURRENCY_PAT"].match(ww) or P["UNIT_TOKEN_PAT"].match(ww):
                continue
            kept.append({"word": ww, "prob": float(w.get("prob", 0.0))})
        if not kept:
            continue
        # 어워드 과다 토픽 드롭(상위 10 단어 중 5개 이상 어워드 힌트 포함)
        top10 = [w["word"] for w in kept[:10]]
        cnt_aw = sum(1 for x in top10 if any(h in x.lower() for h in AWARD_HINTS))
        if cnt_aw >= 5:
            continue
        filtered.append({"topic_id": int(t.get("topic_id", 0)), "top_words": kept})
    return {"topics": filtered}

# -------------------------
# LLM-based topic naming (optional; Gemini)
# -------------------------
def name_topics_with_llm(topics_obj: Dict[str, Any], cfg: dict, llm: dict) -> Dict[str, Any]:
    provider = (llm.get("provider") or "").lower()
    if provider != "gemini":
        return topics_obj
    api_key = os.getenv("GEMINI_API_KEY", "")
    genai = _maybe_import_gemini()
    if not api_key or genai is None:
        return topics_obj
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(llm.get("model", "gemini-1.5-flash"))
    except Exception:
        return topics_obj

    for t in topics_obj.get("topics", []):
        words = [w.get("word","") for w in (t.get("top_words") or [])][:8]
        if not words:
            continue
        prompt = (
            "다음 한국어 키워드 목록을 2~4단어의 간결한 토픽 이름으로 요약해 주세요.\n"
            f"키워드: {', '.join(words)}\n"
            "출력은 따옴표 없이 한국어 토픽 이름만 한 줄로 주세요."
        )
        try:
            resp = model.generate_content(prompt)
            name = (getattr(resp, "text", None) or "").strip()
            if name:
                t["topic_name"] = name
        except Exception:
            continue
    return topics_obj

# -------------------------
# LLM-based topic insight summary (optional; Gemini)
# -------------------------
def summarize_topics_with_llm(topics_obj: Dict[str, Any], cfg: dict, llm: dict) -> Dict[str, Any]:
    provider = (llm.get("provider") or "").lower()
    if provider != "gemini":
        return topics_obj
    api_key = os.getenv("GEMINI_API_KEY", "")
    genai = _maybe_import_gemini()
    if not api_key or genai is None:
        return topics_obj
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(llm.get("model", "gemini-1.5-flash"))
    except Exception:
        return topics_obj

    for t in topics_obj.get("topics", []):
        words = [w.get("word","") for w in (t.get("top_words") or []) if w.get("word")][:10]
        if not words:
            continue
        prompt = (
            f"토픽 키워드: {', '.join(words)}\n"
            "디스플레이/반도체 도메인 관점에서 2~3문장으로 간결 요약해 주세요.\n"
            "- 한국어로 작성\n"
            "- 추측·미확인 정보 금지\n"
            "- 무엇(기술/제품/공정)과 왜 중요한지 중심"
        )
        try:
            resp = model.generate_content(prompt)
            text = (getattr(resp, "text", None) or "").strip()
            if text:
                t["insight"] = text
        except Exception:
            continue
    return topics_obj

# -------------------------
# Insights (도메인/노이즈 가중 기반)
# -------------------------
DOMAIN_HINTS = set([(x or "").strip().lower() for x in CFG.get("domain_hints", [])])
NOISE_HINTS  = set([(x or "").strip().lower() for x in CFG.get("common_debuff", [])])

def domain_overlap(words: List[str]) -> float:
    w = [w.lower() for w in words]
    # 부분 포함 허용(힌트가 복합어일 수 있음)
    return sum(1 for x in w if any(h in x for h in DOMAIN_HINTS)) / max(1, len(w))

def noise_overlap(words: List[str]) -> float:
    w = [w.lower() for w in words]
    return sum(1 for x in w if x in NOISE_HINTS) / max(1, len(w))

def build_insights(topics_obj: Dict[str, Any], ts_obj: Dict[str, Any], topk=5) -> Dict[str, Any]:
    topics = topics_obj.get("topics", [])
    scored = []
    for t in topics:
        words = [w.get("word","") for w in (t.get("top_words") or [])][:10]
        base = sum(float(w.get("prob",0.0)) for w in (t.get("top_words") or []))
        ds = domain_overlap(words)
        ns = noise_overlap(words)
        s = base + 0.6*ds - 0.3*ns  # 운영하며 튜닝 가능
        scored.append((t, s, ds))
    scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
    top = [{
        "topic_id": int(t["topic_id"]),
        "label": (t.get("topic_name") or (t["top_words"][0]["word"] if t["top_words"] else f"Topic {t['topic_id']}")),
        "insight": t.get("insight", "")
    } for t,_,_ in scored[:topk]]

    daily = ts_obj.get("daily", [])
    summary = f"주요 {len(top)}개 토픽이 도출되었고, 최근 {len(daily)}일 시계열을 기반으로 트렌드가 산출되었습니다."
    evidence = [{"topic_id": it["topic_id"], "words": [w["word"] for w in (topics[it["topic_id"]]["top_words"] if it["topic_id"] < len(topics) else [])[:5]]} for it in top]
    return {"summary": summary, "top_topics": top, "evidence": evidence}

# -------------------------
# Main
# -------------------------
def main():
    _log_mode()
    cfg = CFG
    llm = LLM
    use_pro = use_pro_mode()
    random.seed(42)

    meta_path = latest("data/news_meta_*.json")
    if not meta_path:
        print("[ERROR] meta 파일이 없습니다. Module A부터 실행하세요.")
        return 1

    with open(meta_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    # 1) 시계열
    ts_obj = make_timeseries(items)

    # 2) 코퍼스 전처리
    raw_docs = build_docs_from_meta(items)
    phrase_stop, stopwords = merged_stopwords(cfg)
    P = compile_patterns(cfg)
    corpus = preprocess_docs(raw_docs, phrase_stop, stopwords, P)

    # 3) 토픽
    topn = int(cfg.get("topic_topn_words", 10))
    if use_pro and BERTopic is not None:
        topics_obj = topics_pro(corpus, cfg=cfg, topn=topn, random_state=42)
    else:
        topics_obj = topics_lite(corpus, stopwords=stopwords, cfg=cfg, topn=topn, random_state=42)

    # 4) 토픽 후처리
    topics_obj = post_filter_topics(topics_obj, stopwords=stopwords, P=P)

    # 5) (선택) LLM 토픽 이름/요약
    topics_obj = name_topics_with_llm(topics_obj, cfg=cfg, llm=llm)
    topics_obj = summarize_topics_with_llm(topics_obj, cfg=cfg, llm=llm)

    # 6) 인사이트(상위 토픽 선정)
    insights_obj = build_insights(topics_obj, ts_obj, topk=min(5, max(1, len(topics_obj.get("topics",[])))))

    # 7) 저장
    ensure_dir("outputs")
    save_json("outputs/topics.json", topics_obj)
    save_json("outputs/trend_timeseries.json", ts_obj)
    save_json("outputs/trend_insights.json", insights_obj)

    # 8) 디버그 메타
    save_json("outputs/debug/run_meta_c.json", {
        "use_pro": use_pro,
        "docs": len(corpus),
        "topics": len(topics_obj.get("topics", [])),
        "ts_days": len(ts_obj.get("daily", [])),
        "llm_naming": bool(os.getenv("GEMINI_API_KEY", "")) and (LLM.get("provider") == "gemini"),
        "llm_summary": bool(os.getenv("GEMINI_API_KEY", "")) and (LLM.get("provider") == "gemini")
    })

    print(f"[INFO][C] done | topics={len(topics_obj.get('topics', []))} days={len(ts_obj.get('daily', []))}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

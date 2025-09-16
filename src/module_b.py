# -*- coding: utf-8 -*-
"""
Module B - Keywords (improved)
- 기존 파이프라인을 보존하면서 뉴스 특화 전처리, 후보 확장(ngram up to 3),
  간단한 NER 주입, 위치 기반 가중치(title/lead/first paragraph) 등을 추가합니다.
- 외부 리소스: data/entities_org.txt, data/brands.txt, data/product_alias.json
"""
import os
import json
import glob
import re
import unicodedata
import time
import string
import csv
import datetime
from collections import defaultdict
from typing import List, Dict, Any, Tuple

from soynlp.normalizer import normalize, repeat_normalize, emoticon_normalize
from krwordrank.word import KRWordRank
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------ 설정/유틸 ------------------

def norm_kw_light(s: str) -> str:
    if not s: return ""
    s = unicodedata.normalize("NFKC", s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s.replace("‐", "-").replace("-", "-")

def _load_lines(p: str) -> List[str]:
    try:
        with open(p, encoding="utf-8") as f:
            return [x.strip() for x in f if x.strip()]
    except Exception:
        return []

def load_config(path: str = "config.json") -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f) or {}
    except FileNotFoundError:
        cfg = {}
    cfg.setdefault("top_n_keywords", 50)
    cfg.setdefault("min_docfreq", 6)
    cfg.setdefault("min_docfreq_autotune", True)
    cfg.setdefault("stopwords", [])
    cfg.setdefault("phrase_stop", [])
    cfg.setdefault("alias", {})
    cfg.setdefault("common_debuff", [])
    cfg.setdefault("domain_hints", [])
    cfg.setdefault("weights", {})
    cfg.setdefault("keyword_extraction_defaults", {})
    cfg.setdefault("regex_patterns", {})
    cfg.setdefault("news_byline_patterns", [])
    return cfg

CFG = load_config()
STOPWORDS = None  # lazy load
PHRASE_STOPWORDS = None
UNIFIED_ALIAS_MAP = None
REGEX = None

def load_stopwords(cfg: Dict[str, Any]) -> set:
    defaults = cfg.get("keyword_extraction_defaults", {})
    from_config = set(cfg.get("stopwords", []))
    from_file = set(_load_lines(os.path.join("data/dictionaries", "stopwords_ext.txt")))
    from_more = set(defaults.get("MORE_STOP", []))
    en_stop = set(defaults.get("EN_STOP", []))
    all_stops = from_config | from_file | from_more | en_stop
    return {norm_kw_light(s) for s in all_stops if s}

def load_phrase_stopwords(cfg: Dict[str, Any]) -> set:
    from_config = set(cfg.get("phrase_stop", []))
    from_file = set(_load_lines(os.path.join("data/dictionaries", "phrase_stopwords.txt")))
    all_stops = from_config | from_file
    return {norm_kw_light(s) for s in all_stops if s}

def load_unified_alias_map(cfg: Dict[str, Any]) -> Dict[str, str]:
    defaults = cfg.get("keyword_extraction_defaults", {})
    merged = {}
    for src in (defaults.get("FIX_MAP", {}), cfg.get("alias", {})):
        for k, v in (src or {}).items():
            merged[norm_kw_light(k)] = v
    return merged

def compile_regex_patterns(cfg: Dict[str, Any]) -> Dict[str, re.Pattern]:
    compiled = {}
    for name, pat in (cfg.get("regex_patterns") or {}).items():
        try:
            compiled[name] = re.compile(pat)
        except Exception as e:
            print(f"[WARN] regex compile 실패: {name} -> {e}")
    # news byline patterns (list of strings)
    for i, pat in enumerate(cfg.get("news_byline_patterns", [])):
        try:
            compiled[f"BYLINE_{i}"] = re.compile(pat)
        except Exception as e:
            print(f"[WARN] byline regex compile 실패: {pat} -> {e}")
    return compiled

# lazy init
def _init_globals():
    global STOPWORDS, PHRASE_STOPWORDS, UNIFIED_ALIAS_MAP, REGEX
    if STOPWORDS is None:
        STOPWORDS = load_stopwords(CFG)
    if PHRASE_STOPWORDS is None:
        PHRASE_STOPWORDS = load_phrase_stopwords(CFG)
    if UNIFIED_ALIAS_MAP is None:
        UNIFIED_ALIAS_MAP = load_unified_alias_map(CFG)
    if REGEX is None:
        REGEX = compile_regex_patterns(CFG)

# 폴백 패턴(설정 누락 시 사용)
NUMERIC_ONLY_FB = re.compile(r"^\d+$")
DATE_PAT_FB = re.compile(r"^\d{1,2}(일|월)$|^\d{4}(년)?$")
CURRENCY_PAT_FB = re.compile(r"^[0-9,\.]+(원|달러|유로|엔|위안|억원|조원)$")
BROKEN_KO_FB = re.compile(r"^[ㄱ-ㅎㅏ-ㅣ]+$")
TAIL_BAD_FB = re.compile(r"(하기\s?위|위해|위한|하며|하고|하는|으로|로|에|에서|부터|까지)$")
BAD_ENDING_FB = re.compile(
    r"(?:"
    r".*(?:을|를)\s*활용한$|"
    r".*(?:하|되)겠다$|"
    r".*(?:했|혔)다$|"
    r".*(?:이|라)며$|"
    r".*(?:하|되)는\s*(?:핵심|기술|방안)$|"
    r".*(?:을|를)\s*통(?:해)?$|"
    r".*(?:다|고|라)는$"
    r")"
)

def _log_mode(prefix: str = "Module B"):
    is_pro = os.getenv("USE_PRO", str(CFG.get("use_pro", False))).lower() in ("1", "true", "yes", "y")
    mode = "PRO" if is_pro else "LITE"
    print(f"[INFO] USE_PRO={str(is_pro).lower()} → {prefix} ({mode}) 시작")

# ------------------ 텍스트 유틸 ------------------

def _has_jongseong(ch: str) -> bool:
    if not '가' <= ch <= '힣': return False
    return (ord(ch) - 0xAC00) % 28 != 0

def strip_korean_particle(word: str) -> str:
    if not word or len(word) < 2: return word
    last, prev = word[-1], word[-2]
    rules = {"가": False, "은": True, "는": False, "을": True, "를": False, "과": True, "와": False}
    if last in rules and _has_jongseong(prev) == rules[last]:
        return word[:-1]
    return word

def strip_verb_ending(word: str) -> str:
    return re.sub(r"(하다|하게|하고|하며|하면|하는|해요?|했다|합니다|된다|되는|될|됐다|있다|있음|또한)$", "", word)

def unify_keyword(w: str) -> str:
    _init_globals()
    return UNIFIED_ALIAS_MAP.get(norm_kw_light(w), w)

def normalize_keyword(w: str) -> str:
    if not w: return ""
    w = unify_keyword(w)
    w = re.sub(r"^[\'\"‘’“”]+|[\'\"‘’“”]+$", "", w.strip())
    w = w.strip(string.punctuation + "·…")
    w = re.sub(r"\s+", " ", w)
    if re.fullmatch(r"[A-Za-z0-9 \-_/]+", w): w = w.lower()
    return w

def clean_text(t: str) -> str:
    if not t: return ""
    t = re.sub(r"<.+?>", " ", t)
    t = unicodedata.normalize("NFKC", t)
    t = normalize(t)
    t = emoticon_normalize(t, num_repeats=2)
    t = repeat_normalize(t, num_repeats=2)
    return t.strip()

def is_meaningful_token(tok: str) -> bool:
    if not tok or len(tok) < 2: return False
    _init_globals()
    tl = norm_kw_light(tok)
    if tl in STOPWORDS: return False

    num_pat = REGEX.get("NUMERIC_ONLY", NUMERIC_ONLY_FB)
    date_pat = REGEX.get("DATE_PAT", DATE_PAT_FB)
    cur_pat  = REGEX.get("CURRENCY_PAT", CURRENCY_PAT_FB)
    brk_pat  = REGEX.get("BROKEN_KO", BROKEN_KO_FB)
    tail_pat = REGEX.get("TAIL_BAD_RE", TAIL_BAD_FB)
    bad_end  = REGEX.get("BAD_ENDING_PAT", BAD_ENDING_FB)

    if num_pat.fullmatch(tok): return False
    if date_pat.fullmatch(tok): return False
    if cur_pat.fullmatch(tok): return False
    if brk_pat.fullmatch(tok): return False
    if len(tok) <= 2 and tok.endswith("스"): return False
    if tail_pat.search(tok): return False
    if len(tok) >= 4 and bad_end.search(tok): return False
    if re.search(r"\s$", tok): return False
    return True

def is_valid_phrase(phrase: str) -> bool:
    _init_globals()
    pl = norm_kw_light(phrase)
    if pl in PHRASE_STOPWORDS: return False
    tail_pat = REGEX.get("TAIL_BAD_RE", TAIL_BAD_FB)
    bad_end  = REGEX.get("BAD_ENDING_PAT", BAD_ENDING_FB)
    if tail_pat.search(phrase): return False
    if len(phrase) >= 4 and bad_end.search(phrase): return False
    if re.search(r"\s$", phrase): return False
    return True

# ------------------ 도큐먼트 빌드 ------------------

def latest(globpat: str):
    files = sorted(glob.glob(globpat))
    return files[-1] if files else None

def build_docs(meta_items: List[Dict[str, Any]]) -> List[str]:
    """
    기존 API(다른 스크립트 의존성 유지): title + body 를 합쳐 문자열 목록 반환
    """
    docs = []
    for it in meta_items:
        title = clean_text(it.get("title") or it.get("title_og") or "")
        body = clean_text(it.get("body") or it.get("description") or it.get("description_og") or "")
        doc = (title + " " + body).strip()
        if doc: docs.append(doc)
    return docs

def build_docs_meta(meta_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    더 풍부한 메타를 가진 문서 리스트 반환(키워드 위치 정보 포함).
    각 항목: {"title","body","lead","text","lead_offset","first_para"}
    """
    out = []
    for it in meta_items:
        title = clean_text(it.get("title") or it.get("title_og") or "")
        body = clean_text(it.get("body") or it.get("description") or it.get("description_og") or "")
        text = (title + " " + body).strip()
        # lead: 첫 250자(문장 단위로 자르기)
        lead = ""
        if body:
            # split into sentences roughly by . ? ! or newline or '다.' endings
            sents = re.split(r"(?<=[\.\?\!\n]|다\.)\s*", body)
            lead = " ".join(sents[:2]).strip()
            first_para = sents[0].strip() if sents else body[:250]
        else:
            first_para = ""
        out.append({
            "title": title,
            "body": body,
            "lead": lead,
            "text": text,
            "first_para": first_para
        })
    return out

# ------------------ 후보 생성 ------------------

def dedup_docs_by_cosine(docs: List[str], threshold: float = 0.90) -> List[str]:
    if len(docs) <= 1: return docs
    vec = TfidfVectorizer(max_features=7000, ngram_range=(1, 2))
    X = vec.fit_transform(docs)
    sim = cosine_similarity(X, dense_output=False)
    keep_indices = []
    removed = set()
    for i in range(len(docs)):
        if i in removed: continue
        keep_indices.append(i)
        for j in range(i + 1, len(docs)):
            if j in removed: continue
            if sim[i, j] >= threshold:
                removed.add(j)
    return [docs[i] for i in keep_indices]

def extract_keywords_krwordrank(docs: List[str], topk: int = 30) -> List[Dict[str, Any]]:
    n = len(docs)
    min_count, max_iter = (1, 5) if n < 20 else (2, 8) if n < 50 else (5, 12)
    kwr = KRWordRank(min_count=min_count, max_length=10, verbose=False)
    keywords, _, _ = kwr.extract(docs, max_iter=max_iter)

    results = []
    for w, score in sorted(keywords.items(), key=lambda x: x[1], reverse=True):
        if len(results) >= topk * 2: break
        w_norm = normalize_keyword(w)
        w_norm = strip_korean_particle(w_norm)
        w_norm = strip_verb_ending(w_norm)
        if is_meaningful_token(w_norm):
            results.append({"keyword": w_norm, "score": float(score)})
    return results

def pro_extract_keywords_keybert(docs: List[str], topk: int = 50) -> List[Dict[str, Any]]:
    try:
        from keybert import KeyBERT
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError(f"Pro 모드 준비 실패(패키지 없음): {e}")
    if not docs: return []
    model = SentenceTransformer("jhgan/ko-sroberta-multitask")
    kb = KeyBERT(model=model)
    sample_docs = docs[:2000]
    joined_text = " ".join(sample_docs)
    pairs = kb.extract_keywords(
        joined_text,
        keyphrase_ngram_range=(2, 3),
        stop_words=None,
        use_mmr=True,
        diversity=0.8,
        use_maxsum=True,
        nr_candidates=max(topk * 12, 400),
        top_n=max(topk * 4, 200)
    )
    return [{"keyword": normalize_keyword(p), "score": float(s)} for p, s in pairs if p]

def top_ngrams_by_tfidf(docs: List[str], topn: int = 70, min_df: int = 6, max_ngram: int = 3) -> List[str]:
    vec = TfidfVectorizer(ngram_range=(1, max_ngram), min_df=min_df, max_features=7000,
                          token_pattern=r"[가-힣A-Za-z0-9_]{2,}")
    X = vec.fit_transform(docs)
    if X.shape[1] == 0: return []
    tfidf_sum = X.sum(axis=0).A1
    terms = vec.get_feature_names_out()
    pairs = sorted(zip(terms, tfidf_sum), key=lambda x: x[1], reverse=True)
    out = []
    for t, _ in pairs[:topn]:
        nt = normalize_keyword(t)
        if is_meaningful_token(nt) and is_valid_phrase(nt):
            out.append(nt)
    return out

def load_entities_weight() -> Tuple[set, set]:
    orgs, prods = set(), set()
    try:
        with open("outputs/export/entities.csv", "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                e = (r.get("entity") or "").strip()
                typ = (r.get("type") or "").strip().upper()
                if not e: continue
                if typ == "ORG": orgs.add(e)
                elif typ == "PRODUCT": prods.add(e)
    except Exception:
        pass
    return orgs, prods

def load_entity_lists() -> Tuple[set, set, Dict[str, List[str]]]:
    """
    data/entities_org.txt, data/brands.txt, data/product_alias.json
    """
    orgs = set(_load_lines("data/entities_org.txt"))
    brands = set(_load_lines("data/brands.txt"))
    alias_map = {}
    try:
        with open("data/product_alias.json", encoding="utf-8") as f:
            alias_map = json.load(f)
    except Exception:
        alias_map = {}
    # normalize simple
    orgs = {x.strip() for x in orgs if x}
    brands = {x.strip() for x in brands if x}
    return orgs, brands, alias_map

# ------------------ 후처리, 가중치, 다양화 ------------------

def postprocess_keywords(docs: List[str], keywords: List[Dict[str, Any]], min_docfreq: int) -> List[Dict[str, Any]]:
    num_docs = len(docs)
    if CFG.get("min_docfreq_autotune", True):
        min_docfreq = 3 if num_docs < 40 else 5 if num_docs < 100 else 7

    df_map = defaultdict(int)
    for d in docs:
        tokens = set(norm_kw_light(t) for t in re.findall(r"[가-힣]+|[A-Za-z0-9_]+", d))
        for t in tokens: df_map[t] += 1

    merged = {}
    for k in keywords:
        w = normalize_keyword(k["keyword"])
        if not is_meaningful_token(w) or not is_valid_phrase(w):
            continue
        wl = norm_kw_light(w)
        approx_df = max((df_map.get(t, 0) for t in df_map if wl in t or t in wl), default=0)
        if approx_df < min_docfreq:
            continue
        if w not in merged or merged[w]["score"] < k["score"]:
            merged[w] = {"keyword": w, "score": float(k["score"])}
    return sorted(merged.values(), key=lambda x: x["score"], reverse=True)

def mmr_diversify(candidates: List[Dict[str, Any]], topn: int = 50, diversity: float = 0.7) -> List[Dict[str, Any]]:
    if not candidates or topn <= 0: return []
    cands = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)
    selected: List[Dict[str, Any]] = []
    vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 5))

    def included_by_any(term: str, sel_terms: List[str]) -> bool:
        tl = norm_kw_light(term)
        if len(tl) < 4: return False
        for s in sel_terms:
            sl = norm_kw_light(s)
            if len(sl) < 4: continue
            if tl in sl:
                return True
        return False

    while cands and len(selected) < topn:
        if not selected:
            selected.append(cands.pop(0))
            continue
        sel_terms = [x["keyword"] for x in selected]
        rem_terms = [x["keyword"] for x in cands]
        M = vec.fit_transform(sel_terms + rem_terms)
        sim = cosine_similarity(M)
        S = len(sel_terms)
        best, best_i = -1e9, None
        for i, cand in enumerate(cands):
            if included_by_any(cand["keyword"], sel_terms):
                continue
            max_sim = sim[S+i, :S].max() if S > 0 else 0.0
            score = (1 - diversity) * float(cand.get("score", 0.0)) - diversity * float(max_sim)
            if score > best:
                best, best_i = score, i
        if best_i is None:
            break
        selected.append(cands.pop(best_i))

    return selected

def build_tfidf(docs: List[str]):
    vec = TfidfVectorizer(max_features=7000, ngram_range=(1, 2))
    X = vec.fit_transform(docs)
    return vec, X

# ------------------ 메인 파이프라인 ------------------

def main():
    _log_mode("Module B")
    t0 = time.time()
    _init_globals()

    topk = int(CFG.get("top_n_keywords", 50))
    base_min_df = int(CFG.get("min_docfreq", 6))
    weights = CFG.get("weights", {}) or {}
    # 기존 가중치
    entity_boost = float(weights.get("entity_boost", 1.35))
    common_debuff_w = float(weights.get("common_debuff", 0.55))
    person_name_debuff = float(weights.get("person_name_debuff", 0.8))
    domain_hint_boost = float(weights.get("domain_hint_boost", 1.2))
    bigram_top_boost = float(weights.get("bigram_top30_boost", 1.35))
    mmr_diversity = float(weights.get("mmr_diversity", 0.7))
    # 위치 기반 보정 가중치 (config에서 추가 가능)
    title_boost = float(weights.get("title_boost", 1.6))
    lead_boost = float(weights.get("lead_boost", 1.3))
    firstpara_boost = float(weights.get("firstpara_boost", 1.2))
    trigram_top_boost = float(weights.get("trigram_top_boost", 1.25))
    entity_exact_boost = float(weights.get("entity_exact_boost", 1.5))

    meta_path = latest("data/news_meta_*.json")
    if not meta_path:
        raise SystemExit("[ERROR] data/news_meta_*.json 없음.")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_items = json.load(f)

    docs_meta = build_docs_meta(meta_items)
    docs = [d["text"] for d in docs_meta if d.get("text")]
    if not docs:
        raise SystemExit("[ERROR] 문서가 비어 있음")
    docs = dedup_docs_by_cosine(docs, threshold=0.90)
    print(f"[INFO] 문서 수 (중복 제거 후): {len(docs)}")

    # 후보 추출: Lite/Pro 선택
    is_pro = os.getenv("USE_PRO", str(CFG.get("use_pro", False))).lower() in ("1", "true", "yes", "y")
    try:
        if is_pro:
            base_candidates = pro_extract_keywords_keybert(docs, topk)
        else:
            base_candidates = extract_keywords_krwordrank(docs, topk)
    except Exception as e:
        print(f"[WARN] 키워드 추출 실패, Lite로 폴백: {e}")
        base_candidates = extract_keywords_krwordrank(docs, topk)

    # ngram 보강 (bigram + trigram)
    try:
        num_docs = len(docs)
        if CFG.get("min_docfreq_autotune", True):
            md = 3 if num_docs < 40 else 5 if num_docs < 100 else 7
        else:
            md = base_min_df

        ngrams = top_ngrams_by_tfidf(docs, topn=120, min_df=md, max_ngram=3)
        if ngrams:
            avg_score = (sum(k["score"] for k in base_candidates) / max(1, len(base_candidates))) if base_candidates else 1.0
            seen = {norm_kw_light(k["keyword"]) for k in base_candidates}
            cutoff = max(1, int(len(ngrams) * 0.35))
            for i, ng in enumerate(ngrams):
                ngl = norm_kw_light(ng)
                if ngl in seen: continue
                score = avg_score * (trigram_top_boost if i < cutoff and len(ng.split()) >= 3 else (bigram_top_boost if i < cutoff else 1.0))
                base_candidates.append({"keyword": ng, "score": float(score)})
                seen.add(ngl)
    except Exception as e:
        print(f"[WARN] ngram 보강 실패: {e}")

    # 간단한 NER/엔티티 주입: 로컬 리스트 사용
    try:
        orgs_list, brands_list, product_alias = load_entity_lists()
        # orgs, brands가 발견되면 후보로 추가(높은 점수)
        seen = {norm_kw_light(k["keyword"]) for k in base_candidates}
        avg_score = (sum(k["score"] for k in base_candidates) / max(1, len(base_candidates))) if base_candidates else 1.0
        boost_score = avg_score * entity_exact_boost
        for e in sorted(orgs_list | brands_list):
            if not e: continue
            ne = normalize_keyword(e)
            if norm_kw_light(ne) in seen: continue
            if is_meaningful_token(ne):
                base_candidates.append({"keyword": ne, "score": float(boost_score)})
                seen.add(norm_kw_light(ne))
        # product_alias map: inject aliases keys
        for canonical, variants in (product_alias or {}).items():
            can = normalize_keyword(canonical)
            if norm_kw_light(can) in seen: continue
            base_candidates.append({"keyword": can, "score": float(boost_score)})
            seen.add(norm_kw_light(can))
    except Exception as e:
        print(f"[WARN] 엔티티 주입 실패: {e}")

    # 후처리(필터/DF)
    keywords = postprocess_keywords(docs, base_candidates, base_min_df)

    # 가중치 적용: 엔티티/도메인 힌트/일반 디버프/인명 디버프 + 위치 기반 보정
    orgs, prods = load_entities_weight()
    domain_hints_set = set(CFG.get("domain_hints", []))
    common_debuff_set = set(CFG.get("common_debuff", []))
    person_name_pat = REGEX.get("PERSON_NAME_PAT")

    # 위치 기반 인덱스: 작은 최적화를 위해 텍스트에서 위치 인덱스를 계산
    # (title/lead/first_para 포함 여부)
    title_texts = [d["title"] for d in docs_meta]
    lead_texts = [d["lead"] for d in docs_meta]
    firstpara_texts = [d["first_para"] for d in docs_meta]

    boosted = []
    for k in keywords:
        kw, score = k["keyword"], float(k["score"])
        lk = norm_kw_light(kw)

        # entity exact
        if kw in orgs or kw in prods:
            score *= entity_boost

        # common debuff
        if lk in common_debuff_set:
            score *= common_debuff_w

        # 인명 디버프(보수적)
        if person_name_pat and person_name_pat.fullmatch(kw):
            if kw not in orgs and kw not in prods and len(kw) <= 3:
                score *= person_name_debuff

        # domain hints
        if any(h in kw for h in domain_hints_set):
            score *= domain_hint_boost

        # 위치 기반 보정: title/lead/first paragraph에서 자주 등장하면 가중
        # (단, 너무 짧은 키워드는 제외)
        if len(lk) >= 2:
            # check any document: if appears in title anywhere -> boost multiplicatively
            appears_in_title = any((kw in t) for t in title_texts if t)
            appears_in_lead = any((kw in t) for t in lead_texts if t)
            appears_in_first = any((kw in t) for t in firstpara_texts if t)
            if appears_in_title: score *= title_boost
            elif appears_in_lead: score *= lead_boost
            elif appears_in_first: score *= firstpara_boost

        boosted.append({"keyword": kw, "score": score})

    boosted.sort(key=lambda x: x["score"], reverse=True)

    # 다양화 + 포함관계 중복 제거
    diversified = mmr_diversify(boosted, topn=topk, diversity=mmr_diversity)

    # 저장
    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/keywords.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"stats": {"num_docs": len(docs)}, "keywords": diversified}, f, ensure_ascii=False, indent=2)

    meta = {"module": "B", "mode": "PRO" if is_pro else "LITE", "time_utc": datetime.datetime.utcnow().isoformat() + "Z"}
    with open("outputs/run_meta_b.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[INFO] 모듈 B 완료 | 상위 키워드={len(diversified)} | 출력={out_path} | 경과(초)={round(time.time() - t0, 2)}")

if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
# Module B - Keywords (Integrated & Quality-boosted)
# - config.json + data/dictionaries 리소스 병합
# - KR-WordRank + TF-IDF 기반 (라이트), 문서별 KeyBERT MMR 재랭킹 및 BERTopic 보정(프로)
# - 숫자/날짜/통화/단위 필터 + 행정지명/인명/일반어 디버프 + 하드 드롭
# - 값 정렬은 itemgetter(1) 공용 헬퍼로 통일(오타 방지)

import os
import re
import glob
import json
import math
import logging
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import numpy as np
from operator import itemgetter  # 값 정렬 키

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# -------------------------
# Optional deps (graceful fallback)
# -------------------------
try:
    from krwordrank.word import KRWordRank
    from krwordrank.hangle import normalize as kr_normalize
except Exception:
    KRWordRank = None
    def kr_normalize(x, english=False, number=True): return (x or "").strip()

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    TfidfVectorizer, cosine_similarity = None, None

try:
    from keybert import KeyBERT
except Exception:
    KeyBERT = None

try:
    from bertopic import BERTopic
    from umap import UMAP
    from hdbscan import HDBSCAN
except Exception:
    BERTopic, UMAP, HDBSCAN = None, None, None

# NEW: 안전 임포트 (n-gram 후보 생성용)
try:
    from sklearn.feature_extraction.text import CountVectorizer
except Exception:
    CountVectorizer = None

# -------------------------
# Utilities / IO
# -------------------------
def load_json(path: str, default=None):
    if default is None:
        default = {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def load_lines(path: str) -> List[str]:
    try:
        with open(path, encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]
    except Exception:
        return []

def latest_file(pattern: str) -> Optional[str]:
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None

# MMR 다양성 재랭킹 (Lite/Pro 공용)
def mmr_diversify_terms(sorted_pairs: List[Tuple[str, float]], topn: int = 50, diversity: float = 0.55) -> List[Tuple[str, float]]:
    def char_bigrams(s: str):
        s = (s or "").strip()
        return {s[i:i+2] for i in range(len(s) - 1)} if len(s) >= 2 else {s}
    cand = [(k, float(s)) for k, s in sorted_pairs]
    rep = {k: char_bigrams(k) for k, _ in cand}
    selected: List[Tuple[str, float]] = []
    while cand and len(selected) < topn:
        best_i, best_score = 0, -1e9
        for i, (k, sc) in enumerate(cand):
            if not selected:
                mmr = sc
            else:
                max_sim = 0.0
                for ks, _ in selected:
                    a, b = rep[k], rep[ks]
                    inter = len(a & b)
                    union = len(a | b) or 1
                    sim = inter / union
                    if sim > max_sim:
                        max_sim = sim
                mmr = diversity * sc - (1.0 - diversity) * max_sim
            if mmr > best_score:
                best_score, best_i = mmr, i
        selected.append(cand.pop(best_i))
    return selected

# 공용: dict를 값(value) 기준 내림차순 정렬
def sort_items_by_value_desc(d: Dict[str, float]):
    return sorted(d.items(), key=itemgetter(1), reverse=True)  # (key,value) 2-튜플의 인덱스 1이 값

# -------------------------
# Build docs from meta
# -------------------------
def build_docs(items: List[dict]) -> List[str]:
    docs = []
    for it in items:
        title = (it.get("title") or it.get("title_og") or "").strip()
        body  = (it.get("body") or it.get("description") or it.get("description_og") or "").strip()
        txt = (title + " " + body).strip()
        if txt:
            docs.append(txt)
    return docs

def dedup_docs_by_cosine(docs: List[str], threshold: float = 0.90) -> List[str]:
    if TfidfVectorizer is None or cosine_similarity is None or len(docs) <= 1:
        return docs
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

# -------------------------
# Normalization / noun-ish cleanup
# -------------------------
_HANGUL_ALNUM = re.compile(r"[가-힣A-Za-z0-9]+")
def basic_normalize(txt: str) -> str:
    t = kr_normalize((txt or "").strip(), english=False, number=True)
    toks = _HANGUL_ALNUM.findall(t)
    return " ".join(toks)

_JOSA = ("은","는","이","가","을","를","에","에서","으로","로","과","와","에게","한테","께","이나","나","든지","까지","부터","라도","마저","밖에","뿐")
_EOMI = ("했다","하였다","한다","했다가","하며","해서","하는","되다","되면","되었","된다","되니","되어","됐다","됐다가","했다며","이라","이라며","이라서")

def nounish_strip(sentence: str) -> str:
    toks = sentence.split()
    out = []
    for tk in toks:
        t = tk
        for suf in _JOSA:
            if t.endswith(suf) and len(t) >= len(suf) + 2:
                t = t[:-len(suf)]
                break
        for suf in _EOMI:
            if t.endswith(suf) and len(t) >= len(suf) + 2:
                t = t[:-len(suf)]
                break
        if len(t) >= 2:
            out.append(t)
    return " ".join(out)

# -------------------------
# Patterns / Locations
# -------------------------
def compile_patterns(cfg: dict):
    rp = cfg.get("regex_patterns", {}) or {}
    def _comp(key, default):
        try:
            return re.compile(rp.get(key, default))
        except Exception:
            return re.compile(default)
    pats = {
        "NUMERIC_ONLY": _comp("NUMERIC_ONLY", r"^\d+$"),
        "DATE_PAT": _comp("DATE_PAT", r"^\d{1,2}일$|^\d{1,2}월$|^\d{4}년$|^\d{4}$"),
        "CURRENCY_PAT": _comp("CURRENCY_PAT", r"^[0-9,\.]+(원|달러|유로|엔|위안|억원|조원)$"),
        "PERSON_NAME_PAT": _comp("PERSON_NAME_PAT", r"^[가-힣]{2,4}$"),
        # 숫자+단위(확장: 세대/nm/나노/인치 포함)
        "UNIT_TOKEN_PAT": _comp("UNIT_TOKEN_PAT", r"^\d+(?:[.,]\d+)?(%|배|건|개|명|곳|회|대|종|분기|세대|nm|나노|인치)$"),
    }
    return pats

_LOCATION_CORE = {
    "서울","부산","대구","인천","광주","대전","울산","세종",
    "경기","경기도","강원","강원도","충북","충남","전북","전남","경북","경남",
    "제주","제주도","수원","용인","성남","고양","화성","부천","안산","안양","남양주",
}
_LOCATION_SUFFIX = {"도","시","군","구","읍","면","동","리"}

def is_location_token(tok: str) -> bool:
    if not tok: return False
    if tok in _LOCATION_CORE:
        return True
    if len(tok) >= 2 and tok[-1] in _LOCATION_SUFFIX:
        return True
    return False

# -------------------------
# Preprocess with strict filters
# -------------------------
def preprocess_docs(docs: List[str], phrase_stop: List[str], stopwords: List[str],
                    use_nounish: bool=True, patterns: Optional[dict]=None) -> List[str]:
    ps = set(phrase_stop or [])
    sw = set(stopwords or [])
    P = patterns or {}

    out = []
    for d in docs:
        if not d:
            continue
        t = basic_normalize(d)
        for ph in ps:
            if ph:
                t = t.replace(ph, " ")
        if use_nounish:
            t = nounish_strip(t)

        toks = []
        for w in t.split():
            # 1) 조사 자체 토큰 스킵
            if w in {"은","는","이","가","을","를","에","에서","으로","로","과","와","에게","한테","께","이나","나","든지","까지","부터","라도","마저","밖에","뿐"}:
                continue

            # 2) 간단 어미/인용형 컷 적용
            for suf in ("이라고","라고","며","면서","지만","다는","라고도","다고","고"):
                if w.endswith(suf) and len(w) >= len(suf) + 2:
                    w = w[:-len(suf)]
                    break

            # 3) 불용어(사전+ext) 매칭
            if w in sw:
                continue

            # 4) 숫자/날짜/통화/단위 제거
            if P.get("NUMERIC_ONLY") and P["NUMERIC_ONLY"].match(w):
                continue
            if P.get("DATE_PAT") and P["DATE_PAT"].match(w):
                continue
            if P.get("CURRENCY_PAT") and P["CURRENCY_PAT"].match(w):
                continue
            if P.get("UNIT_TOKEN_PAT") and P["UNIT_TOKEN_PAT"].match(w):
                continue

            if len(w) < 2:
                continue
            toks.append(w)

        t2 = " ".join(toks)
        if t2:
            out.append(t2)
    return out

# -------------------------
# Alias / brand-entity resources
# -------------------------
def build_alias_map(config_alias: Dict[str, str], product_alias_path: str) -> Dict[str, str]:
    alias = dict(config_alias or {})
    pa = load_json(product_alias_path, {})
    for can, variants in pa.items():
        for v in variants:
            alias[v] = can
    return alias

def normalize_alias(token: str, alias_map: Dict[str, str]) -> str:
    if token in alias_map:
        return alias_map[token]
    low = token.lower()
    for k, v in alias_map.items():
        if k.lower() == low:
            return v
    return token

def load_brand_entity_lists() -> Tuple[set, set]:
    brands = set(load_lines("data/dictionaries/brands.txt"))
    entities = set()
    for p in ("data/dictionaries/entities.txt", "data/dictionaries/entities_org.txt", "data/dictionaries/entites.txt"):
        entities.update(load_lines(p))
    brands = {b.strip() for b in brands if b.strip()}
    entities = {e.strip() for e in entities if e.strip()}
    return brands, entities

# -------------------------
# Docfreq filter / sampler / n-gram candidates / domain relevance
# -------------------------
def docfreq_map(candidates: List[str], docs: List[str]) -> Dict[str, int]:
    df = {c: 0 for c in candidates}
    for d in docs:
        for c in candidates:
            if " " in c:
                if c in d:
                    df[c] += 1
            else:
                if re.search(rf"(?<!\S){re.escape(c)}(?!\S)", d):
                    df[c] += 1
    return df

def filter_by_docfreq(scores: Dict[str, float], docs: List[str], min_df: int = 3, autotune: bool = True) -> Dict[str, float]:
    if not scores:
        return {}
    n_docs = len(docs)
    if autotune:
        if n_docs >= 60:  min_df = max(min_df, 4)
        if n_docs >= 120: min_df = max(min_df, 5)
    df = docfreq_map(list(scores.keys()), docs)
    return {k: v for k, v in scores.items() if df.get(k, 0) >= max(1, int(min_df))}

def even_sample(lst: List[str], k: int) -> List[str]:
    if k <= 0 or not lst:
        return []
    if len(lst) <= k:
        return lst[:]
    step = max(1, len(lst) // k)
    return lst[::step][:k]

def build_ngram_candidates(docs: List[str], stopwords: List[str], ngram_range=(2,3), min_df=2, max_df=0.95, topk=300) -> List[str]:
    if CountVectorizer is None or not docs:
        return []
    try:
        cv = CountVectorizer(ngram_range=ngram_range, min_df=min_df, max_df=max_df,
                             token_pattern=r"[가-힣A-Za-z0-9_]{2,}")
        X = cv.fit_transform(docs)
        terms = cv.get_feature_names_out()
        freqs = np.asarray(X.sum(axis=0)).ravel()
        pairs = [(t, int(freqs[i])) for i, t in enumerate(terms)]
        pairs.sort(key=itemgetter(1), reverse=True)
        sw = set(stopwords or [])
        out = []
        for t, _ in pairs:
            toks = t.split()
            if any(tok in sw for tok in toks):
                continue
            out.append(t)
            if len(out) >= topk:
                break
        return out
    except Exception:
        return []

def adjust_by_domain_relevance(scores: Dict[str, float], docs: List[str], domain_hints: List[str], debuff: float = 0.4) -> Dict[str, float]:
    # 간단 공출현율 기반: 도메인 힌트가 등장하는 문서에서의 등장 비율이 낮으면 약화
    if not scores or not docs:
        return scores
    hints = [h.lower() for h in (domain_hints or []) if h]
    if not hints:
        return scores
    df = docfreq_map(list(scores.keys()), docs)
    hint_docs = [d for d in docs if any(h in d.lower() for h in hints)]
    if not hint_docs:
        return scores
    df_hint = docfreq_map(list(scores.keys()), hint_docs)
    out = {}
    for k, v in scores.items():
        total = max(1, df.get(k, 0))
        in_hint = df_hint.get(k, 0)
        rate = in_hint / total
        s = v * (1.0 if rate >= 0.5 else (0.7 if rate >= 0.3 else debuff))
        out[k] = s
    return out

# -------------------------
# Domain weighting with debuffs (UPDATED)
# -------------------------
def apply_domain_weights(scores: Dict[str, float],
                         domain_hints: List[str],
                         common_debuff: List[str],
                         alias_map: Dict[str, str],
                         weight_cfg: Dict[str, float],
                         brands: Optional[set]=None,
                         entities: Optional[set]=None,
                         patterns: Optional[dict]=None) -> Dict[str, float]:
    if not scores:
        return {}
    P = patterns or {}
    boosted = {}
    db = float(weight_cfg.get("domain_hint_boost", 1.0))
    cd = float(weight_cfg.get("common_debuff", 1.0))
    entity_boost = float(weight_cfg.get("entity_boost", 1.35))
    brand_boost  = float(weight_cfg.get("brand_boost", 1.2))
    person_debuf = float(weight_cfg.get("person_name_debuff", 0.8))
    loc_debuf    = float(weight_cfg.get("location_debuff", 0.6))
    num_debuf    = float(weight_cfg.get("number_debuff", 0.5))
    out_domain_debuff = float(weight_cfg.get("out_of_domain_debuff", 0.4))

    # 외부 사전(도메인 외) 로드
    out_of_domain = set(load_lines("data/dictionaries/out_of_domain.txt"))

    dh = set(domain_hints or [])
    cm = set(common_debuff or [])
    brands = brands or set()
    entities = entities or set()

    for k, v in scores.items():
        k2 = normalize_alias(k, alias_map)
        s = v

        if any(h.lower() in k2.lower() for h in dh):
            s *= db
        if k2 in cm or any(c.lower() == k2.lower() for c in cm):
            s *= cd

        if k2 in entities:
            s *= entity_boost
        if k2 in brands:
            s *= brand_boost

        if k2 in out_of_domain:
            s *= out_domain_debuff

        # 인명(2~4자 한글) 약화(엔티티/브랜드 제외)
        if P.get("PERSON_NAME_PAT") and P["PERSON_NAME_PAT"].fullmatch(k2):
            if k2 not in entities and k2 not in brands:
                s *= person_debuf

        # 행정지명 약화
        if is_location_token(k2):
            s *= loc_debuf

        # 숫자/날짜/통화/단위 토큰 약화(전처리 누락 대비)
        if (P.get("NUMERIC_ONLY") and P["NUMERIC_ONLY"].match(k2)) \
           or (P.get("DATE_PAT") and P["DATE_PAT"].match(k2)) \
           or (P.get("CURRENCY_PAT") and P["CURRENCY_PAT"].match(k2)) \
           or (P.get("UNIT_TOKEN_PAT") and P["UNIT_TOKEN_PAT"].match(k2)):
            s *= num_debuf

        if s <= 0:
            continue
        boosted[k2] = max(boosted.get(k2, 0.0), s)
    return boosted

# -------------------------
# Stats / autotune
# -------------------------
def compute_doc_stats(docs: List[str]) -> Tuple[int, float]:
    n = len(docs)
    avg = np.mean([len(x) for x in docs]) if docs else 0.0
    return n, float(avg)

def autotune_kr(n_docs: int, avg_len: float, min_count_base: int=3) -> Tuple[int,int]:
    mc = max(min_count_base, int(round(math.log1p(n_docs) + avg_len/800)))
    mc = min(max(3, mc), 12)
    max_len = 12 if avg_len < 400 else 15
    return mc, max_len

# -------------------------
# KR-WordRank + TF-IDF (Light)
# -------------------------
def extract_krwordrank(docs: List[str], beta: float=0.85, max_iter: int=20, min_count: Optional[int]=None,
                       max_length: Optional[int]=None, topk: int=200) -> Dict[str, float]:
    if KRWordRank is None:
        return {}
    n_docs, avg_len = compute_doc_stats(docs)
    if min_count is None or max_length is None:
        mc, ml = autotune_kr(n_docs, avg_len)
        if min_count is None: min_count = mc
        if max_length is None: max_length = ml
    extractor = KRWordRank(min_count=min_count, max_length=max_length, verbose=False)
    keywords, rank, _ = extractor.extract(docs, beta=beta, max_iter=max_iter)
    sorted_items = sort_items_by_value_desc(keywords)  # 값 기준 정렬
    return dict(sorted_items[: max(1, int(topk or 1))])

def tfidf_weights(docs: List[str], vocab: List[str]) -> Dict[str, float]:
    if TfidfVectorizer is None or not docs:
        return {v: 1.0 for v in vocab}
    vec = TfidfVectorizer(ngram_range=(1,3), min_df=2, max_df=0.9)
    X = vec.fit_transform(docs)
    idf = dict(zip(vec.get_feature_names_out(), vec.idf_))
    return {v: float(idf.get(v, 1.0)) for v in vocab}

def hybrid_rank(docs: List[str], beta: float=0.85, max_iter: int=20, topk: int=200,
                w_kr: float=0.7, w_tfidf: float=0.3) -> Dict[str, float]:
    kr = extract_krwordrank(docs, beta=beta, max_iter=max_iter, topk=topk)
    if not kr:
        return {}
    vocab = list(kr.keys())
    idf = tfidf_weights(docs, vocab)
    def norm(d):
        vals = list(d.values()) if d else [0.0]
        mn, mx = min(vals), max(vals)
        if mx - mn < 1e-9:
            return {k: 1.0 for k in d}
        return {k: (v - mn) / (mx - mn) for k, v in d.items()}
    kr_n = norm(kr)
    idf_n = norm(idf)
    blended = {k: w_kr*kr_n.get(k,0.0) + w_tfidf*idf_n.get(k,0.0) for k in vocab}
    sorted_items = sort_items_by_value_desc(blended)  # 값 기준 정렬
    return dict(sorted_items[: max(1, int(topk or 1))])

def tfidf_only(docs: List[str], topk: int=200) -> Dict[str, float]:
    if TfidfVectorizer is None or not docs:
        return {}
    vec = TfidfVectorizer(ngram_range=(1,3), min_df=2, max_df=0.9)
    X = vec.fit_transform(docs)
    terms = vec.get_feature_names_out()
    avg = np.asarray(X.mean(axis=0)).ravel()
    pairs = list(zip(terms, avg))
    pairs.sort(key=itemgetter(1), reverse=True)  # 값 기준 정렬
    return dict(pairs[: max(1, int(topk or 1))])

# -------------------------
# KeyBERT MMR reranking (Pro, per-document)
# -------------------------
def keybert_rerank_doc(doc: str, candidates: List[str], model_name: str, topn: int,
                       use_mmr: bool=True, diversity: float=0.5, ngram_range: Tuple[int,int]=(1,3)) -> Dict[str,float]:
    if KeyBERT is None or not doc or not candidates:
        return {}
    try:
        kb = KeyBERT(model=model_name)
        extracted = kb.extract_keywords(
            doc,
            keyphrase_ngram_range=ngram_range,
            stop_words=None,  # 전처리에서 불용어 제거
            use_mmr=use_mmr,
            diversity=diversity,
            top_n=max(topn, len(candidates))
        )
        cand_set = set(candidates)
        rer = [(p, s) for (p, s) in extracted if p in cand_set]
        return dict(rer[:topn]) if rer else {}
    except Exception:
        return {}

# -------------------------
# BERTopic topic context (optional)
# -------------------------
def topic_context_keywords(docs: List[str], model_name: str, umap_neighbors: int=15,
                           min_cluster_size: int=12, topn_per_topic: int=10) -> Dict[int, List[str]]:
    if BERTopic is None:
        return {}
    try:
        umap_model = UMAP(n_neighbors=umap_neighbors, n_components=10, metric="cosine", low_memory=True, random_state=42)
        hdb_model = HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean", prediction_data=True)
        tm = BERTopic(umap_model=umap_model, hdbscan_model=hdb_model,
                      embedding_model=model_name, calculate_probabilities=False, verbose=False)
        topics, _ = tm.fit_transform(docs)
        info = tm.get_topic_info()
        out = {}
        for t in info["Topic"].tolist():
            if int(t) < 0:
                continue
            words = [w for w, _ in tm.get_topic(int(t))[:topn_per_topic]]
            out[int(t)] = words
        return out
    except Exception:
        return {}

# -------------------------
# Main
# -------------------------
def main():
    cfg = load_json("config.json", {})
    weights = cfg.get("weights", {}) or {}

    # Merge config + data/dictionaries resources
    defaults = cfg.get("keyword_extraction_defaults", {}) or {}

    phrase_stop = sorted(
        set(cfg.get("phrase_stop", []) or [])
        | set(load_lines("data/dictionaries/phrase_stopwords.txt"))
    )
    stopwords = sorted(
        set(cfg.get("stopwords", []) or [])
        | set(load_lines("data/dictionaries/stopwords_ext.txt"))
        | set(defaults.get("MORE_STOP", []))
        | set(defaults.get("EN_STOP", []))
    )

    alias_seed = {}
    alias_seed.update(defaults.get("FIX_MAP", {}) or {})
    alias_seed.update(cfg.get("alias", {}) or {})
    alias_map = build_alias_map(alias_seed, product_alias_path="data/dictionaries/product_alias.json")

    brands, entities = load_brand_entity_lists()

    # Patterns
    patterns = compile_patterns(cfg)

    topn_keywords = int(cfg.get("top_n_keywords", 50))
    use_pro = os.environ.get("USE_PRO", "").lower() in ("1","true","yes","y") or bool(cfg.get("use_pro", False))

    meta_path = latest_file("data/news_meta_*.json")
    if not meta_path:
        raise SystemExit("no data/news_meta_*.json found")
    with open(meta_path, encoding="utf-8") as f:
        items = json.load(f)

    raw_docs = build_docs(items)
    if not raw_docs:
        raise SystemExit("no documents")

    # Dedup (optional)
    raw_docs = dedup_docs_by_cosine(raw_docs, threshold=0.90)

    # Preprocess with strict filters
    pre_docs = preprocess_docs(raw_docs, phrase_stop=phrase_stop, stopwords=stopwords,
                               use_nounish=True, patterns=patterns)
    if not pre_docs:
        raise SystemExit("no valid docs after preprocessing")

    beta = float(weights.get("beta", 0.85))
    max_iter = int(weights.get("max_iter", 20))

    # Base extraction
    if KRWordRank is not None:
        base_scores = hybrid_rank(pre_docs, beta=beta, max_iter=max_iter, topk=max(200, topn_keywords))
    else:
        base_scores = tfidf_only(pre_docs, topk=max(200, topn_keywords))

    combined = base_scores.copy()

    # Pro: per-document KeyBERT MMR reranking and aggregation (UPDATED)
    if use_pro and KeyBERT is not None and combined:
        # 1) 후보군 확장: 2~3그램 복합어 추가 (조사 포함 후보 차단)
        _JOSA_SET = set(["은","는","이","가","을","를","에","에서","으로","로","과","와","에게","한테","께","이나","나","든지","까지","부터","라도","마저","밖에","뿐"])
        ngram_cands = build_ngram_candidates(
            pre_docs,
            stopwords=sorted(set(stopwords) | _JOSA_SET),
            ngram_range=(2,3),
            min_df=2, max_df=0.95, topk=300
        )
        cand = list(set(list(combined.keys()) + ngram_cands))

        model_name = cfg.get("keybert_model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        diversity = float(cfg.get("mmr_diversity", 0.55))
        max_docs_rerank = int(cfg.get("max_docs_rerank", 80))

        # 2) 균등 샘플링으로 치우침 방지
        sel_docs = even_sample(pre_docs, max_docs_rerank)

        agg, cnt = defaultdict(float), defaultdict(int)
        for d in sel_docs:
            rer = keybert_rerank_doc(d, candidates=cand, model_name=model_name,
                                     topn=min(len(cand), topn_keywords),
                                     use_mmr=True, diversity=diversity, ngram_range=(1,3))
            if not rer:
                continue
            vals = list(rer.values())
            mn, mx = min(vals), max(vals)
            for k, v in rer.items():
                nv = (v - mn) / (mx - mn + 1e-12)
                agg[k] += nv
                cnt[k] += 1

        if agg:
            all_keys = list(set(list(combined.keys()) + list(agg.keys())))
            def norm(d):
                vals = [d.get(k,0.0) for k in all_keys]
                mn, mx = min(vals), max(vals)
                return {k: (d.get(k,0.0)-mn)/(mx-mn+1e-12) for k in all_keys}
            base_n = norm(combined)
            rer_n  = {k: (agg[k]/max(1,cnt[k])) for k in all_keys}
            vals = list(rer_n.values())
            mn, mx = (min(vals), max(vals)) if vals else (0.0, 1.0)
            rer_n = {k: (rer_n.get(k,0.0)-mn)/(mx-mn+1e-12) for k in all_keys}
            combined = {k: 0.4*base_n.get(k,0.0) + 0.6*rer_n.get(k,0.0) for k in all_keys}

    # Optional: topic context boost (Pro)
    if use_pro and BERTopic is not None and len(pre_docs) >= 20:
        model_name = cfg.get("keybert_model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        umap_neighbors = int(cfg.get("umap_neighbors", 15))
        min_cluster_size = int(cfg.get("min_cluster_size", 12))
        topic_kw = topic_context_keywords(pre_docs, model_name=model_name,
                                          umap_neighbors=umap_neighbors,
                                          min_cluster_size=min_cluster_size,
                                          topn_per_topic=topn_keywords)
        topic_set = set([w for lst in topic_kw.values() for w in lst])
        for k in list(combined.keys()):
            if k in topic_set:
                combined[k] *= 1.05

    # NEW: 1차 정제(문서빈도) + 도메인 관련도 보정(공출현율)
    min_df_cfg = int(cfg.get("min_docfreq", 3))
    auto_df = bool(cfg.get("min_docfreq_autotune", True))
    combined = filter_by_docfreq(combined, pre_docs, min_df=min_df_cfg, autotune=auto_df)
    combined = adjust_by_domain_relevance(combined, pre_docs, cfg.get("domain_hints", []),
                                          debuff=float(weights.get("out_of_domain_debuff", 0.4)))

    # Domain/alias/brand/entity weights + debuffs (UPDATED)
    combined = apply_domain_weights(
        combined,
        domain_hints=cfg.get("domain_hints", []),
        common_debuff=cfg.get("common_debuff", []),
        alias_map=alias_map,
        weight_cfg=weights,
        brands=brands,
        entities=entities,
        patterns=patterns
    )

    # Hard drop: 숫자/날짜/통화/단위 최종 제거
    def _hard_drop(tok: str) -> bool:
        return (patterns["NUMERIC_ONLY"].match(tok)
                or patterns["DATE_PAT"].match(tok)
                or patterns["CURRENCY_PAT"].match(tok)
                or patterns["UNIT_TOKEN_PAT"].match(tok))
    combined = {k: v for k, v in combined.items() if not _hard_drop(k)}

    # Final hard drop: stopwords + phrase + optional regex
    stop_final = set(stopwords) | set(cfg.get("stopwords", []))  # 사전+config 병합

    # 선택: config에 regex_stopwords가 있을 경우 컴파일
    regex_list = []
    for pat in (cfg.get("regex_stopwords", []) or []):
        try:
            regex_list.append(re.compile(pat))
        except Exception:
            pass

    def _is_regex_stop(s: str) -> bool:
        return any(r.search(s) for r in regex_list)

    # 불용어/정규식 최종 제거
    combined = {k: v for k, v in combined.items()
                if (k not in stop_final) and (not _is_regex_stop(k))}

    # Output with Lite MMR diversification
    top_pairs = sort_items_by_value_desc(combined)[: max(100, topn_keywords)]
    top_pairs = mmr_diversify_terms(top_pairs, topn=topn_keywords, diversity=float(cfg.get("mmr_diversity", 0.55)))
    top_items = top_pairs

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/keywords.json", "w", encoding="utf-8") as f:
        json.dump({"keywords": [{"keyword": k, "score": float(s)} for k, s in top_items]}, f, ensure_ascii=False, indent=2)

    os.makedirs("outputs/debug", exist_ok=True)
    with open("outputs/debug/run_meta_b.json", "w", encoding="utf-8") as f:
        json.dump({
            "use_pro": use_pro,
            "docs": len(pre_docs),
            "deps": {"krwordrank": KRWordRank is not None, "keybert": KeyBERT is not None, "bertopic": BERTopic is not None},
            "resources_dir": "data/dictionaries"
        }, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()

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
from src.config import load_config

import numpy as np
from operator import itemgetter  # 값 정렬 키
from src.utils import load_json, save_json, latest


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


# -------------------------
# Utilities / IO
# -------------------------
def load_lines(path: str) -> List[str]:
    try:
        with open(path, encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]
    except Exception:
        return []

def latest(pattern: str) -> Optional[str]:
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None

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


_JOSA = ("은","는","이","가","을","를","에","에서","으로","로","과","와","에게","한테","께","이나","나","든지","까지","부터","라도","마저","밖에","뿐","의","처럼","만큼","보다")
_EOMI = ("했다","하였다","한다","했다가","하며","해서","하는","되다","되면","되었","된다","되니","되어","됐다","됐다가","했다며","이다","였다","이다가","이며","이어서","인","일","입니다","습니다","으니까","으니까요","는데요","고요","구요","네요","군요","시오","십시오")

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
def compile_patterns(CFG: dict):
    rp = CFG.get("regex_patterns", {}) or {}
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
        # 숫자+단위(%, 배, 건, 개, 명, 곳, 회, 대, 종, 분기 등)
        "UNIT_TOKEN_PAT": re.compile(r"^\d+(?:[.,]\d+)?(%|배|건|개|명|곳|회|대|종|분기)$")
    }
    return pats

_LOCATION_CORE = {
    "서울","부산","대구","인천","광주","대전","울산","세종",
    "경기","경기도","강원","강원도","충북","충남","전북","전남","경북","경남",
    "제주","제주도","수원","용인","성남","고양","화성","부천","안산","안양","남양주"
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
            # 사전 불용어 제거
            if w in sw:
                continue
            # 숫자/날짜/통화/단위 제거
            if P.get("NUMERIC_ONLY") and P["NUMERIC_ONLY"].match(w):  # 숫자만
                continue
            if P.get("DATE_PAT") and P["DATE_PAT"].match(w):          # 날짜/연도
                continue
            if P.get("CURRENCY_PAT") and P["CURRENCY_PAT"].match(w):  # 통화
                continue
            if P.get("UNIT_TOKEN_PAT") and P["UNIT_TOKEN_PAT"].match(w):  # 숫자+단위
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
# Domain weighting with debuffs
# -------------------------
def apply_domain_weights(scores: Dict[str, float],
                         domain_hints: List[str],
                         common_debuff: List[str],
                         alias_map: Dict[str, str],
                         weight_CFG: Dict[str, float],
                         brands: Optional[set]=None,
                         entities: Optional[set]=None,
                         patterns: Optional[dict]=None) -> Dict[str, float]:
    if not scores:
        return {}
    P = patterns or {}
    boosted = {}
    db = float(weight_CFG.get("domain_hint_boost", 1.0))
    cd = float(weight_CFG.get("common_debuff", 1.0))
    entity_boost = float(weight_CFG.get("entity_boost", 1.35))
    brand_boost  = float(weight_CFG.get("brand_boost", 1.2))
    person_debuf = float(weight_CFG.get("person_name_debuff", 0.8))
    loc_debuf    = float(weight_CFG.get("location_debuff", 0.6))
    num_debuf    = float(weight_CFG.get("number_debuff", 0.5))

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

        # 인명(2~4자 한글) 약화(엔티티 제외)
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
    vec = TfidfVectorizer(ngram_range=(1,3), min_df=3, max_df=0.9)
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
    vec = TfidfVectorizer(ngram_range=(1,3), min_df=3, max_df=0.9)
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
                       use_mmr: bool=True, diversity: float=0.5, ngram_range: Tuple[int,int]=(1,3), stopwords: Optional[set]=None) -> Dict[str,float]:
    if KeyBERT is None or not doc or not candidates:
        return {}
    try:
        kb = KeyBERT(model=model_name)
        extracted = kb.extract_keywords(
            doc,
            keyphrase_ngram_range=ngram_range,
            stop_words=list(stopwords) if stopwords else None,
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
    CFG = load_config()
    weights = CFG.get("weights", {}) or {}

    # Merge config + data/dictionaries resources
    defaults = CFG.get("keyword_extraction_defaults", {}) or {}

    phrase_stop = sorted(
        set(CFG.get("phrase_stop", []) or [])
        | set(load_lines("data/dictionaries/phrase_stopwords.txt"))
    )

    stopwords = sorted(
        set(CFG.get("stopwords", []) or [])
        | set(load_lines("data/dictionaries/stopwords_ext.txt"))
        | set(defaults.get("MORE_STOP", []))
        | set(defaults.get("EN_STOP", []))
    )

    alias_seed = {}
    alias_seed.update(defaults.get("FIX_MAP", {}) or {})
    alias_seed.update(CFG.get("alias", {}) or {})
    alias_map = build_alias_map(alias_seed, product_alias_path="data/dictionaries/product_alias.json")

    brands, entities = load_brand_entity_lists()

    # Patterns
    patterns = compile_patterns(CFG)

    topn_keywords = int(CFG.get("top_n_keywords", 50))
    use_pro = os.environ.get("USE_PRO", "").lower() in ("1","true","yes","y") or bool(CFG.get("use_pro", False))

    meta_path = latest("data/news_meta_*.json")
    if not meta_path:
        raise SystemExit("no data/news_meta_*.json found")
    with open(meta_path, encoding="utf-8") as f:
        items = json.load(f)

    raw_docs = build_docs(items)
    if not raw_docs:
        raise SystemExit("no documents")

    # Dedup (optional)
    raw_docs = dedup_docs_by_cosine(raw_docs, threshold=0.93)

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

    # Pro: per-document KeyBERT MMR reranking and aggregation
    if use_pro and KeyBERT is not None and combined:
        cand = list(combined.keys())
        model_name = CFG.get("keybert_model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        diversity = float(CFG.get("mmr_diversity", 0.5))
        max_docs_rerank = int(CFG.get("max_docs_rerank", 80))
        sel_docs = pre_docs[:max_docs_rerank]

        agg, cnt = defaultdict(float), defaultdict(int)
        for d in sel_docs:
            rer = keybert_rerank_doc(d, candidates=cand, model_name=model_name,
                                     topn=min(len(cand), topn_keywords),
                                     use_mmr=True, diversity=diversity, ngram_range=(1,3), stopwords=set(stopwords))
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
            rer_n = {k: (agg[k]/max(1,cnt[k])) for k in all_keys}
            vals = list(rer_n.values())
            mn, mx = (min(vals), max(vals)) if vals else (0.0, 1.0)
            rer_n = {k: (rer_n.get(k,0.0)-mn)/(mx-mn+1e-12) for k in all_keys}
            combined = {k: 0.4*base_n.get(k,0.0) + 0.6*rer_n.get(k,0.0) for k in all_keys}

    # Optional: topic context boost (Pro)
    if use_pro and BERTopic is not None and len(pre_docs) >= 20:
        model_name = CFG.get("keybert_model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        umap_neighbors = int(CFG.get("umap_neighbors", 15))
        min_cluster_size = int(CFG.get("min_cluster_size", 12))
        topic_kw = topic_context_keywords(pre_docs, model_name=model_name,
                                          umap_neighbors=umap_neighbors,
                                          min_cluster_size=min_cluster_size,
                                          topn_per_topic=topn_keywords)
        topic_set = set([w for lst in topic_kw.values() for w in lst])
        for k in list(combined.keys()):
            if k in topic_set:
                combined[k] *= 1.05

    # Domain/alias/brand/entity weights + debuffs
    combined = apply_domain_weights(
        combined,
        domain_hints=CFG.get("domain_hints", []),
        common_debuff=CFG.get("common_debuff", []),
        alias_map=alias_map,
        weight_CFG=weights,
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

    # Output
    top_items = sort_items_by_value_desc(combined)[: topn_keywords]

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

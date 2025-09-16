# -*- coding: utf-8 -*-
# Module B - Keywords (Integrated: config+data/dictionaries, KR-WordRank+TF-IDF, optional KeyBERT MMR, optional BERTopic boost)

import os
import re
import glob
import json
import math
import logging
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import numpy as np

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
except Exception:
    TfidfVectorizer = None

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
# IO helpers / config
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


# -------------------------
# Build docs from meta
# -------------------------
def build_docs(items: List[dict]) -> List[str]:
    docs = []
    for it in items:
        title = (it.get("title") or it.get("title_og") or "").strip()
        body = (it.get("body") or it.get("description") or it.get("description_og") or "").strip()
        txt = (title + " " + body).strip()
        if txt:
            docs.append(txt)
    return docs


# -------------------------
# Normalization / noun-ish cleanup
# -------------------------
_HANGUL_ALNUM = re.compile(r"[가-힣A-Za-z0-9]+")

def basic_normalize(txt: str) -> str:
    t = kr_normalize((txt or "").strip(), english=False, number=True)
    toks = _HANGUL_ALNUM.findall(t)
    return " ".join(toks)

_JOSA = ("은","는","이","가","을","를","에","에서","으로","로","과","와","에게","한테","께","이나","나","든지","까지","부터","라도","마저","밖에","뿐")
_EOMI = ("했다","하였다","한다","했다가","하며","해서","하는","되다","되면","되었","된다","되니","되어","됐다","됐다가","했다며")

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

def preprocess_docs(docs: List[str], phrase_stop: List[str], stopwords: List[str], use_nounish: bool=True) -> List[str]:
    ps = set(phrase_stop or [])
    sw = set(stopwords or [])
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
        toks = [w for w in t.split() if w not in sw]
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
    # Support multiple common filenames for entities
    entities = set()
    for p in ("data/dictionaries/entities.txt", "data/dictionaries/entities_org.txt", "data/dictionaries/entites.txt"):
        entities.update(load_lines(p))
    brands = {b.strip() for b in brands if b.strip()}
    entities = {e.strip() for e in entities if e.strip()}
    return brands, entities


# -------------------------
# Domain weighting
# -------------------------
def apply_domain_weights(scores: Dict[str, float],
                         domain_hints: List[str],
                         common_debuff: List[str],
                         alias_map: Dict[str, str],
                         weight_cfg: Dict[str, float],
                         brands: Optional[set]=None,
                         entities: Optional[set]=None) -> Dict[str, float]:
    if not scores:
        return {}
    boosted = {}
    db = float(weight_cfg.get("domain_hint_boost", 1.0))
    cd = float(weight_cfg.get("common_debuff", 1.0))
    entity_boost = float(weight_cfg.get("entity_boost", 1.35))
    brand_boost = float(weight_cfg.get("brand_boost", 1.2))
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
    return dict(sorted(keywords.items(), key=lambda x: x[21], reverse=True)[:topk])

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
    return dict(sorted(blended.items(), key=lambda x: x[21], reverse=True)[:topk])

def tfidf_only(docs: List[str], topk: int=200) -> Dict[str, float]:
    if TfidfVectorizer is None or not docs:
        return {}
    vec = TfidfVectorizer(ngram_range=(1,3), min_df=2, max_df=0.9)
    X = vec.fit_transform(docs)
    terms = vec.get_feature_names_out()
    avg = np.asarray(X.mean(axis=0)).ravel()
    pairs = list(zip(terms, avg))
    pairs.sort(key=lambda x: x[21], reverse=True)
    return dict(pairs[:topk])


# -------------------------
# KeyBERT MMR reranking (Pro)
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
            stop_words=None,
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

    pre_docs = preprocess_docs(raw_docs, phrase_stop=phrase_stop, stopwords=stopwords, use_nounish=True)
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
        model_name = cfg.get("keybert_model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        diversity = float(cfg.get("mmr_diversity", 0.5))
        max_docs_rerank = int(cfg.get("max_docs_rerank", 80))
        sel_docs = pre_docs[:max_docs_rerank]

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
            rer_n = {k: (agg[k]/max(1,cnt[k])) for k in all_keys}
            vals = list(rer_n.values())
            mn, mx = (min(vals), max(vals)) if vals else (0.0, 1.0)
            rer_n = {k: (rer_n.get(k,0.0)-mn)/(mx-mn+1e-12) for k in all_keys}
            combined = {k: 0.4*base_n.get(k,0.0) + 0.6*rer_n.get(k,0.0) for k in all_keys}

    # Optional: topic context boost
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

    # Domain/alias/brand/entity weights
    combined = apply_domain_weights(
        combined,
        domain_hints=cfg.get("domain_hints", []),
        common_debuff=cfg.get("common_debuff", []),
        alias_map=alias_map,
        weight_cfg=weights,
        brands=brands,
        entities=entities
    )

    # Output
    top_items = sorted(combined.items(), key=lambda x: x[21], reverse=True)[:topn_keywords]

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

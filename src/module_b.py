# src/module_b.py
# -*- coding: utf-8 -*-
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
    def kr_normalize(x, english=False, number=True): return x

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
# IO helpers
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

def latest_meta_path(pattern: str = "data/news_meta_*.json") -> Optional[str]:
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


# -------------------------
# Build docs
# -------------------------
def build_docs(items: List[dict]) -> List[str]:
    docs = []
    for it in items:
        body = (it.get("body") or "").strip()
        desc = (it.get("description") or "").strip()
        title = (it.get("title") or "").strip()
        txt = body or (title + " " + desc)
        if txt:
            docs.append(txt)
    return docs


# -------------------------
# Text normalization
# -------------------------
_HANGUL_RE = re.compile(r"[가-힣A-Za-z0-9]+")

def basic_normalize(txt: str) -> str:
    t = kr_normalize((txt or "").strip(), english=False, number=True)
    tokens = _HANGUL_RE.findall(t)
    return " ".join(tokens)

_JOSA_SUFFIX = ("은","는","이","가","을","를","에","에서","으로","로","과","와",
                "에게","한테","께","이나","나","든지","까지","부터","라도","마저","밖에","뿐")
_EOMI_SUFFIX = ("했다","하였다","한다","했다가","하며","해서","하는",
                "되다","되면","되었","된다","되니","되어","됐다","됐다가","했다며")

def nounish_strip(sentence: str) -> str:
    toks = sentence.split()
    out = []
    for tk in toks:
        t = tk
        for suf in _JOSA_SUFFIX:
            if t.endswith(suf) and len(t) >= len(suf) + 2:
                t = t[: -len(suf)]
                break
        for suf in _EOMI_SUFFIX:
            if t.endswith(suf) and len(t) >= len(suf) + 2:
                t = t[: -len(suf)]
                break
        if len(t) >= 2:
            out.append(t)
    return " ".join(out)

def preprocess_docs(docs: List[str],
                    phrase_stop: List[str],
                    stopwords: List[str],
                    use_nounish: bool = True) -> List[str]:
    out = []
    ps = set(phrase_stop or [])
    sw = set(stopwords or [])
    for d in docs:
        if not d:
            continue
        t = basic_normalize(d)
        for ph in ps:
            if ph:
                t = t.replace(ph, " ")
        if use_nounish:
            t = nounish_strip(t)
        tokens = [w for w in t.split() if w not in sw]
        t2 = " ".join(tokens)
        if t2:
            out.append(t2)
    return out


# -------------------------
# KR-WordRank + TF-IDF hybrid (Lite 기본)
# -------------------------
def extract_krwordrank(docs: List[str],
                       beta: float = 0.85,
                       max_iter: int = 20,
                       topk: int = 200) -> Dict[str, float]:
    if KRWordRank is None:
        logger.warning("KRWordRank not available, skip.")
        return {}
    n_docs = len(docs)
    avg_len = np.mean([len(x) for x in docs]) if docs else 0.0
    min_count = max(3, int(round(math.log1p(n_docs) + avg_len / 800)))
    max_length = 12 if avg_len < 400 else 15
    extractor = KRWordRank(min_count=min_count, max_length=max_length, verbose=False)
    keywords, _, _ = extractor.extract(docs, beta=beta, max_iter=max_iter)
    items = sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:topk]
    return dict(items)

def tfidf_only(docs: List[str], topk: int = 200) -> Dict[str, float]:
    if TfidfVectorizer is None or not docs:
        return {}
    vec = TfidfVectorizer(ngram_range=(1,3), min_df=2, max_df=0.9)
    X = vec.fit_transform(docs)
    feat = vec.get_feature_names_out()
    avg = np.asarray(X.mean(axis=0)).ravel()
    items = list(zip(feat, avg))
    items.sort(key=lambda x: x[1], reverse=True)
    return dict(items[:topk])


# -------------------------
# KeyBERT rerank (Pro 전용)
# -------------------------
def keybert_rerank_docs(docs: List[str],
                        candidates: List[str],
                        model_name: str,
                        topn: int,
                        diversity: float = 0.5) -> Dict[str, float]:
    if KeyBERT is None:
        return {}
    kb = KeyBERT(model=model_name)
    agg, count = defaultdict(float), defaultdict(int)
    for d in docs:
        extracted = kb.extract_keywords(
            d, keyphrase_ngram_range=(1,3),
            use_mmr=True, diversity=diversity,
            top_n=min(topn, len(candidates))
        )
        cand_set = set(candidates)
        rer = [(p, s) for (p, s) in extracted if p in cand_set]
        if rer:
            vals = [s for _, s in rer]
            mn, mx = min(vals), max(vals)
            for k, v in rer:
                nv = (v - mn) / (mx - mn + 1e-12)
                agg[k] += nv
                count[k] += 1
    return {k: agg[k]/max(1,count[k]) for k in agg}


# -------------------------
# BERTopic boost (Pro 전용)
# -------------------------
def topic_context_boost(docs: List[str], model_name: str, topn: int) -> set:
    if BERTopic is None:
        return set()
    umap_model = UMAP(n_neighbors=15, n_components=10, metric="cosine", low_memory=True)
    hdb_model = HDBSCAN(min_cluster_size=12, metric="euclidean", prediction_data=True)
    topic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdb_model,
                           embedding_model=model_name, calculate_probabilities=False, verbose=False)
    topics, _ = topic_model.fit_transform(docs)
    rep = topic_model.get_topic_info()
    topic_kw = []
    for t in rep["Topic"].tolist():
        if int(t) < 0:
            continue
        words = [w for w, _ in topic_model.get_topic(int(t))[:topn]]
        topic_kw.extend(words)
    return set(topic_kw)


# -------------------------
# Orchestrator
# -------------------------
def main():
    cfg = load_json("config.json", {})
    weights = cfg.get("weights", {}) or {}
    phrase_stop = cfg.get("phrase_stop", []) or []
    sw = (cfg.get("stopwords", []) or []) + load_lines("stopwords_ext.txt") + load_lines("phrase_stopwords.txt")
    topn_keywords = int(cfg.get("top_n_keywords", 50))
    use_pro = os.environ.get("USE_PRO", "").lower() == "true" or bool(cfg.get("use_pro", False))

    meta_path = latest_meta_path()
    if not meta_path:
        raise SystemExit("no data/news_meta_*.json found")
    with open(meta_path, encoding="utf-8") as f:
        items = json.load(f)
    docs_raw = build_docs(items)
    docs = preprocess_docs(docs_raw, phrase_stop=phrase_stop, stopwords=sw, use_nounish=True)

    if not docs:
        raise SystemExit("no valid docs after preprocessing")

    # ---------------- Lite 모드 ----------------
    if not use_pro:
        logger.info("[Module B] Running in LITE mode")
        if KRWordRank is not None:
            scores = extract_krwordrank(docs, beta=0.85, max_iter=20, topk=max(200, topn_keywords))
        else:
            scores = tfidf_only(docs, topk=max(200, topn_keywords))
        top_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topn_keywords]

    # ---------------- Pro 모드 ----------------
    else:
        logger.info("[Module B] Running in PRO mode")
        base = extract_krwordrank(docs, beta=0.85, max_iter=20, topk=max(200, topn_keywords))
        if not base:
            base = tfidf_only(docs, topk=max(200, topn_keywords))

        combined = base.copy()

        # KeyBERT rerank
        if KeyBERT is not None:
            reranked = keybert_rerank_docs(docs[:80], list(base.keys()),
                                           model_name=cfg.get("keybert_model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
                                           topn=topn_keywords,
                                           diversity=float(cfg.get("mmr_diversity", 0.5)))
            for k,v in reranked.items():
                combined[k] = 0.4*combined.get(k,0.0) + 0.6*v

        # BERTopic boost
        if BERTopic is not None and len(docs) >= 20:
            topic_kw = topic_context_boost(docs, model_name=cfg.get("keybert_model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"), topn=topn_keywords)
            for k in combined.keys():
                if k in topic_kw:
                    combined[k] *= 1.05

        top_items = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:topn_keywords]

    # ---------------- Save ----------------
    os.makedirs("outputs/debug", exist_ok=True)
    with open("outputs/keywords.json", "w", encoding="utf-8") as f:
        json.dump({"keywords": [{"keyword": k, "score": float(s)} for k, s in top_items]}, f, ensure_ascii=False, indent=2)

    with open("outputs/debug/run_meta_b.json", "w", encoding="utf-8") as f:
        json.dump({"use_pro": use_pro, "docs": len(docs)}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
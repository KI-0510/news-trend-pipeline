# -*- coding: utf-8 -*-
import os
import json
import glob
import re
import unicodedata
import time
import string
import csv
from collections import defaultdict
from typing import List, Tuple
from soynlp.normalizer import normalize, repeat_normalize, emoticon_normalize
from krwordrank.word import KRWordRank
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ===== 조사/어미/정규화 =====
def _has_jongseong(ch: str) -> bool:
    code = ord(ch)
    if 0xAC00 <= code <= 0xD7A3:
        return ((code - 0xAC00) % 28) != 0
    return False

def strip_korean_particle(word: str) -> str:
    if not word or len(word) < 2:
        return word
    last = word[-1]
    prev = word[-2]
    if last in ("이", "의"):
        return word
    rules = {"가": False, "은": True, "는": False, "을": True, "를": False, "과": True, "와": False}
    if last in rules and _has_jongseong(prev) == rules[last]:
        return word[:-1]
    return word

def strip_verb_ending(word: str) -> str:
    return re.sub(r"(하다|하게|하고|하며|하면|하는|해요?|했다|합니다|된다|되는|될|됐다|있다|있음|또한)$", "", word)

def normalize_keyword(w: str) -> str:
    if not w:
        return ""
    w = re.sub(r"^[\'\"‘’“”]+|[\'\"‘’“”]+$", "", w.strip())
    w = w.strip(string.punctuation + "·…")
    w = re.sub(r"\s+", " ", w)
    if re.fullmatch(r"[A-Za-z0-9 \-_/]+", w):
        w = w.lower()
    return w

# ===== 공통 유틸 =====
def latest(globpat: str):
    files = sorted(glob.glob(globpat))
    return files[-1] if files else None

def load_config():
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"top_n_keywords": 50, "stopwords": [], "dedup_threshold": 0.90, "min_docfreq": 5}

def clean_text(t: str) -> str:
    if not t:
        return ""
    t = re.sub(r"<.+?>", " ", t)
    t = unicodedata.normalize("NFKC", t)
    t = normalize(t)
    t = emoticon_normalize(t, num_repeats=2)
    t = repeat_normalize(t, num_repeats=2)
    return t.strip()

def dedup_docs_by_cosine(docs, threshold=0.90):
    if len(docs) <= 1:
        return docs
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vec.fit_transform(docs)
    sim = cosine_similarity(X, dense_output=False)
    keep = []
    removed = set()
    for i in range(len(docs)):
        if i in removed:
            continue
        keep.append(i)
        for j in range(i + 1, len(docs)):
            if sim[i, j] >= threshold:
                removed.add(j)
    return [docs[i] for i in keep]

def build_docs(meta_items):
    docs = []
    for it in meta_items:
        title = clean_text(it.get("title") or it.get("title_og"))
        body = clean_text(it.get("body") or it.get("description") or it.get("description_og"))
        doc = (title + " " + body).strip()
        if doc:
            docs.append(doc)
    return docs

# ===== stopwords 합집합 =====
def _load_lines(p):
    try:
        with open(p, encoding="utf-8") as f:
            return [x.strip() for x in f if x.strip()]
    except Exception:
        return []

def _load_json(p, default=None):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default if default is not None else {}

def norm_kw_light(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("‐", "-").replace("‑", "-")
    return s

CFG = load_config()
DICT_DIR = "data/dictionaries"
STOP_CFG = set([norm_kw_light(x) for x in (CFG.get("stopwords") or [])])
STOP_EXT = set([norm_kw_light(x) for x in _load_lines(os.path.join(DICT_DIR, "stopwords_ext.txt"))])
PHRASE_STOP = set([norm_kw_light(x) for x in _load_lines(os.path.join(DICT_DIR, "phrase_stopwords.txt"))])
STOPWORDS = set(x for x in (STOP_CFG | STOP_EXT | PHRASE_STOP) if x)

EN_STOP = {
    "the","and","to","of","in","for","on","with","at","by","from","as","is","are","be","it",
    "that","this","an","a","or","if","we","you","they","he","she","was","were","been","than",
    "into","about","over","under","per","via"
}
STOPWORDS |= set(EN_STOP)

# ===== KRWordRank 키워드 =====
def extract_keywords_krwordrank(docs, topk=30):
    n = len(docs)
    if n < 20:
        min_count, max_iter = 1, 5
    elif n < 50:
        min_count, max_iter = 2, 8
    else:
        min_count, max_iter = 5, 12
    kwr = KRWordRank(min_count=min_count, max_length=10)
    keywords, _, _ = kwr.extract(docs, max_iter=max_iter)
    results = []
    for w, score in sorted(keywords.items(), key=lambda x: x[1], reverse=True):
        if len(results) >= topk:
            break
        w_norm = normalize_keyword(w)
        w_norm = strip_korean_particle(w_norm)
        w_norm = strip_verb_ending(w_norm)
        if len(w_norm) < 2:
            continue
        if norm_kw_light(w_norm) in STOPWORDS:
            continue
        if re.fullmatch(r"[0-9\W_]+", w_norm):
            continue
        results.append({"keyword": w_norm, "score": float(score)})
    return results

# ===== 빅람 상위 구문(2-gram) TF-IDF =====
def top_bigrams_by_tfidf(docs, topn=50, min_df=5):
    vec = TfidfVectorizer(ngram_range=(2,2), min_df=min_df, max_features=5000,
                          token_pattern=r"[가-힣A-Za-z0-9_]{2,}")
    X = vec.fit_transform(docs)
    if X.shape[1] == 0:
        return []
    tfidf_sum = X.sum(axis=0).A1
    terms = vec.get_feature_names_out()
    pairs = list(zip(terms, tfidf_sum))
    pairs.sort(key=lambda x: x[1], reverse=True)
    pairs = [(t, s) for t, s in pairs if re.search(r"[가-힣A-Za-z0-9]", t) and norm_kw_light(t) not in STOPWORDS]
    return [t for t, _ in pairs[:topn]]

# ===== 키워드 후처리 =====
def postprocess_keywords(docs, keywords, min_docfreq=1):
    df = defaultdict(int)
    for d in docs:
        tokens = set(re.findall(r"[가-힣]+|[A-Za-z0-9_]+", d))
        for t in tokens:
            df[t] += 1
    merged = {}
    for k in keywords:
        w = normalize_keyword(k["keyword"])
        if not w or len(w) < 1:
            continue
        if re.fullmatch(r"[0-9\W_]+", w):
            continue
        if norm_kw_light(w) in STOPWORDS:
            continue
        # 숫자/연도 단독 컷
        if re.fullmatch(r"\d{1,4}$", w):
            continue
        exact_df = df.get(w, 0)
        approx_df = max((df[t] for t in df if w in t or t in w), default=0)
        if max(exact_df, approx_df) < min_docfreq:
            continue
        if w not in merged or merged[w]["score"] < k["score"]:
            merged[w] = {"keyword": w, "score": float(k["score"])}
    return sorted(merged.values(), key=lambda x: x["score"], reverse=True)

# ===== 엔터티 가중치 + MMR 다양화 =====
def load_entities_weight():
    ent_path = "outputs/export/entities.csv"
    orgs = set(); prods = set()
    try:
        with open(ent_path, "r", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                e = (r.get("entity") or "").strip()
                typ = (r.get("type") or "").strip().upper()
                if not e:
                    continue
                if typ == "ORG":
                    orgs.add(e)
                elif typ == "PRODUCT":
                    prods.add(e)
    except Exception:
        pass
    return orgs, prods

def mmr_diversify(candidates, topn=50, diversity=0.7):
    terms = [c["keyword"] for c in candidates]
    if not terms:
        return candidates[:topn]
    vec = TfidfVectorizer(analyzer="char", ngram_range=(3,5))
    M = vec.fit_transform(terms)
    sim = cosine_similarity(M)
    selected = []
    used = set()
    for i, _ in enumerate(candidates):
        if len(selected) >= topn:
            break
        ok = True
        for j in selected:
            if sim[i, j] >= diversity:
                ok = False
                break
        if ok:
            selected.append(i); used.add(i)
    if len(selected) < topn:
        for i in range(len(candidates)):
            if i in used:
                continue
            selected.append(i)
            if len(selected) >= topn:
                break
    return [candidates[i] for i in selected]

def build_tfidf(docs):
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vec.fit_transform(docs)
    return vec, X

def main():
    t0 = time.time()
    cfg = CFG
    topk = int(cfg.get("top_n_keywords", 50))
    min_docfreq = int(cfg.get("min_docfreq", 5))

    meta_path = latest("data/news_meta_*.json")
    if not meta_path:
        print("[ERROR] data/news_meta_*.json 없음. 모듈 A부터 실행 필요")
        raise SystemExit(1)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_items = json.load(f)

    docs = build_docs(meta_items)
    if not docs:
        print("[ERROR] 문서가 비어 있음")
        raise SystemExit(1)

    pre_n = len(docs)
    docs = dedup_docs_by_cosine(docs, threshold=0.90)
    post_n = len(docs)
    print(f"[INFO] 문서 중복 제거: {pre_n} -> {post_n}")

    # 1) KRWordRank
    keywords = extract_keywords_krwordrank(docs, topk=topk)

    # 2) 빅람 합류
    try:
        bigrams = top_bigrams_by_tfidf(docs, topn=50, min_df=min_docfreq)
        if bigrams:
            avg_score = (sum(k["score"] for k in keywords) / max(1, len(keywords))) if keywords else 1.0
            seen = {k["keyword"] for k in keywords}
            for bg in bigrams:
                if norm_kw_light(bg) in STOPWORDS:
                    continue
                if bg not in seen:
                    keywords.append({"keyword": bg, "score": float(avg_score)})
                    seen.add(bg)
    except Exception:
        pass

    # 3) 후처리(빈도/불용어/숫자컷)
    keywords = postprocess_keywords(docs, keywords, min_docfreq=min_docfreq)

    # 4) 엔터티 가중치
    orgs, prods = load_entities_weight()
    boosted = []
    COMMON_DEBUFF = {"스마트","디지털","시장","글로벌","생활","기술","최근","지난해"}
    for k in keywords:
        kw = k["keyword"]; score = k["score"]
        if kw in orgs or kw in prods:
            score *= 1.2
        if kw in COMMON_DEBUFF:
            score *= 0.8
        boosted.append({"keyword": kw, "score": score})
    boosted.sort(key=lambda x: x["score"], reverse=True)

    # 5) MMR 다양화
    diversified = mmr_diversify(boosted, topn=topk, diversity=0.7)

    _vec, _X = build_tfidf(docs)

    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/keywords.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "stats": {"num_docs": len(docs)},
            "keywords": diversified
        }, f, ensure_ascii=False, indent=2)

    print(f"[INFO] 모듈 B 완료 | 문서 수={len(docs)} | 상위 키워드={len(diversified)} | 출력={out_path} | 경과(초)={round(time.time() - t0, 2)}")

if __name__ == "__main__":
    main()

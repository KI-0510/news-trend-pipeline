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

def _log_mode(prefix="Module B"):
    try:
        is_pro = use_pro_mode()
    except Exception:
        is_pro = False
    mode = "PRO" if is_pro else "LITE"
    print(f"[INFO] USE_PRO={str(is_pro).lower()} → {prefix} ({mode}) 시작")

# ================= 텍스트 유틸/클린업 =================
def _has_jongseong(ch: str) -> bool:
    code = ord(ch)
    if 0xAC00 <= code <= 0xD7A3:
        return ((code - 0xAC00) % 28) != 0
    return False

def strip_korean_particle(word: str) -> str:
    if not word or len(word) < 2: return word
    last = word[-1]; prev = word[-2]
    if last in ("이", "의"): return word
    rules = {"가": False, "은": True, "는": False, "을": True, "를": False, "과": True, "와": False}
    if last in rules and _has_jongseong(prev) == rules[last]:
        return word[:-1]
    return word

def strip_verb_ending(word: str) -> str:
    return re.sub(r"(하다|하게|하고|하며|하면|하는|해요?|했다|합니다|된다|되는|될|됐다|있다|있음|또한)$", "", word)

def normalize_keyword(w: str) -> str:
    if not w: return ""
    w = re.sub(r"^[\'\"‘’“”]+|[\'\"‘’“”]+$", "", w.strip())
    w = w.strip(string.punctuation + "·…")
    w = re.sub(r"\s+", " ", w)
    if re.fullmatch(r"[A-Za-z0-9 \-_/]+", w): w = w.lower()
    return w

def latest(globpat: str):
    files = sorted(glob.glob(globpat))
    return files[-1] if files else None

def load_config():
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"top_n_keywords": 50, "stopwords": [], "dedup_threshold": 0.90, "min_docfreq": 6}

def clean_text(t: str) -> str:
    if not t: return ""
    t = re.sub(r"<.+?>", " ", t)
    t = unicodedata.normalize("NFKC", t)
    t = normalize(t)
    t = emoticon_normalize(t, num_repeats=2)
    t = repeat_normalize(t, num_repeats=2)
    return t.strip()

def dedup_docs_by_cosine(docs, threshold=0.90):
    if len(docs) <= 1: return docs
    vec = TfidfVectorizer(max_features=7000, ngram_range=(1, 2))
    X = vec.fit_transform(docs)
    sim = cosine_similarity(X, dense_output=False)
    keep = []
    removed = set()
    for i in range(len(docs)):
        if i in removed: continue
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
        if doc: docs.append(doc)
    return docs

# ================= 사전/불용/컷 =================
def _load_lines(p):
    try:
        with open(p, encoding="utf-8") as f:
            return [x.strip() for x in f if x.strip()]
    except Exception:
        return []

def norm_kw_light(s: str) -> str:
    if not s: return ""
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

# 보도문/형식어 추가 컷(항상 보강 가능)
MORE_STOP = {
    "이날","11","11일","이라고","이라며","대비","가장","특히","세계","지난","따르면","모든","적극",
    "디스","프로","기술","미래","혁신","글로벌","전문","모델을","성과를","받았다","밝혔다","강조했다",
    "국내","대한민국","서울","있는","새로운","여러","플랫폼","사업","핵심"
}
STOPWORDS |= set([norm_kw_light(x) for x in MORE_STOP])

CURRENCY_PAT = re.compile(r"^[0-9,\.]+(원|달러|유로|엔|위안|억원|조원)$")
DATE_PAT = re.compile(r"^\d{1,2}일$|^\d{4}년$|^\d{4}$")
NUMERIC_ONLY = re.compile(r"^\d+$")
BROKEN_KO = re.compile(r"^[ㄱ-ㅎㅏ-ㅣ]+$")

def is_meaningful_token(tok: str) -> bool:
    if not tok: return False
    t = normalize_keyword(tok)
    if len(t) < 2: return False
    if norm_kw_light(t) in STOPWORDS: return False
    if NUMERIC_ONLY.fullmatch(t): return False
    if DATE_PAT.fullmatch(t): return False
    if CURRENCY_PAT.fullmatch(t): return False
    if BROKEN_KO.fullmatch(t): return False
    if len(t) <= 2 and t.endswith("스"): return False
    return True

# ================= Lite 키워드(KRWordRank) =================
def extract_keywords_krwordrank(docs, topk=30):
    n = len(docs)
    if n < 20: min_count, max_iter = 1, 5
    elif n < 50: min_count, max_iter = 2, 8
    else: min_count, max_iter = 5, 12
    kwr = KRWordRank(min_count=min_count, max_length=10)
    keywords, _, _ = kwr.extract(docs, max_iter=max_iter)
    results = []
    for w, score in sorted(keywords.items(), key=lambda x: x[1], reverse=True):
        if len(results) >= topk: break
        w_norm = normalize_keyword(w)
        w_norm = strip_korean_particle(w_norm)
        w_norm = strip_verb_ending(w_norm)
        if not is_meaningful_token(w_norm): continue
        results.append({"keyword": w_norm, "score": float(score)})
    return results

# ================= Pro 키워드(KeyBERT) =================
def pro_extract_keywords_keybert(docs, topk=50):
    try:
        from keybert import KeyBERT
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError(f"Pro 키워드 모드 준비 실패(패키지 없음): {e}")

    if not docs:
        return []

    # 임베딩 모델
    model = SentenceTransformer("jhgan/ko-sroberta-multitask")
    kb = KeyBERT(model=model)

    # 너무 많을 때 과부하 방지 샘플링
    sample_docs = docs[:2000]
    # 한 덩어리로 뽑은 뒤 후처리로 정리
    joined = [" ".join(sample_docs)]

    pairs = kb.extract_keywords(
        joined,
        keyphrase_ngram_range=(2,3),      # 구 중심
        stop_words=None,                  # 사전/후처리에서 컷
        use_mmr=True,
        diversity=0.8,                    # 다양성↑
        use_maxsum=True,                  # 후보군 넓히기
        nr_candidates=max(topk*8, 300),
        top_n=max(150, topk*3)
    )
    out = []
    for p, s in pairs:
        p = (p or "").strip()
        if not p:
            continue
        out.append({"keyword": p, "score": float(s)})
    return out

# ================= 빅람/후처리/가중/MMR =================
def top_bigrams_by_tfidf(docs, topn=70, min_df=6):
    vec = TfidfVectorizer(ngram_range=(2,2), min_df=min_df, max_features=7000,
                          token_pattern=r"[가-힣A-Za-z0-9_]{2,}")
    X = vec.fit_transform(docs)
    if X.shape[1] == 0: return []
    tfidf_sum = X.sum(axis=0).A1
    terms = vec.get_feature_names_out()
    pairs = list(zip(terms, tfidf_sum))
    pairs.sort(key=lambda x: x[1], reverse=True)
    out = []
    for t, _ in pairs[:topn]:
        if not is_meaningful_token(t): continue
        out.append(t)
    return out

def postprocess_keywords(docs, keywords, min_docfreq=6):
    df = defaultdict(int)
    for d in docs:
        tokens = set(re.findall(r"[가-힣]+|[A-Za-z0-9_]+", d))
        for t in tokens: df[t] += 1
    merged = {}
    for k in keywords:
        w = normalize_keyword(k["keyword"])
        if not is_meaningful_token(w): continue
        exact_df = df.get(w, 0)
        approx_df = max((df[t] for t in df if w in t or t in w), default=0)
        if max(exact_df, approx_df) < min_docfreq: continue
        if w not in merged or merged[w]["score"] < k["score"]:
            merged[w] = {"keyword": w, "score": float(k["score"])}
    return sorted(merged.values(), key=lambda x: x["score"], reverse=True)

def load_entities_weight():
    ent_path = "outputs/export/entities.csv"
    orgs = set(); prods = set()
    try:
        with open(ent_path, "r", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                e = (r.get("entity") or "").strip()
                typ = (r.get("type") or "").strip().upper()
                if not e: continue
                if typ == "ORG": orgs.add(e)
                elif typ == "PRODUCT": prods.add(e)
    except Exception:
        pass
    return orgs, prods

def mmr_diversify(candidates, topn=50, diversity=0.6):
    terms = [c["keyword"] for c in candidates]
    if not terms: return candidates[:topn]
    vec = TfidfVectorizer(analyzer="char", ngram_range=(3,5))
    M = vec.fit_transform(terms)
    sim = cosine_similarity(M)
    selected = []
    used = set()
    for i, _ in enumerate(candidates):
        if len(selected) >= topn: break
        ok = True
        for j in selected:
            if sim[i, j] >= diversity:
                ok = False; break
        if ok:
            selected.append(i); used.add(i)
    if len(selected) < topn:
        for i in range(len(candidates)):
            if i in used: continue
            selected.append(i)
            if len(selected) >= topn: break
    return [candidates[i] for i in selected]

def build_tfidf(docs):
    vec = TfidfVectorizer(max_features=7000, ngram_range=(1, 2))
    X = vec.fit_transform(docs)
    return vec, X

# ================= 메인 =================
def main():
    _log_mode("Module B")
    t0 = time.time()
    cfg = CFG
    topk = int(cfg.get("top_n_keywords", 50))
    min_docfreq = int(cfg.get("min_docfreq", 6))

    meta_path = latest("data/news_meta_*.json")
    if not meta_path:
        print("[ERROR] data/news_meta_*.json 없음. 모듈 A부터 실행 필요")
        raise SystemExit(1)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_items = json.load(f)

    docs = build_docs(meta_items)
    if not docs:
        print("[ERROR] 문서가 비어 있음"); raise SystemExit(1)

    pre_n = len(docs)
    docs = dedup_docs_by_cosine(docs, threshold=0.90)
    post_n = len(docs)
    print(f"[INFO] 문서 중복 제거: {pre_n} -> {post_n}")

    # 1) Pro/Lite 분기로 후보 뽑기
    try:
        if use_pro_mode():
            base_candidates = pro_extract_keywords_keybert(docs, topk=topk)
        else:
            base_candidates = extract_keywords_krwordrank(docs, topk=topk)
    except Exception as e:
        print(f"[WARN] Pro 키워드 실패, Lite로 폴백: {e}")
        base_candidates = extract_keywords_krwordrank(docs, topk=topk)

    # 2) 빅람 보강(+상위 30% 가중)
    try:
        bigrams = top_bigrams_by_tfidf(docs, topn=70, min_df=min_docfreq)
        if bigrams:
            avg_score = (sum(k["score"] for k in base_candidates) / max(1, len(base_candidates))) if base_candidates else 1.0
            seen = {k["keyword"] for k in base_candidates}
            cutoff = max(1, int(len(bigrams) * 0.3))
            for idx, bg in enumerate(bigrams):
                if not is_meaningful_token(bg): continue
                if bg in seen: continue
                score = avg_score * (1.2 if idx < cutoff else 1.0)
                base_candidates.append({"keyword": bg, "score": float(score)})
                seen.add(bg)
    except Exception:
        pass

    # 3) 후처리(불용/숫자/날짜/화폐/깨진 토큰 컷 + 문서빈도)
    keywords = postprocess_keywords(docs, base_candidates, min_docfreq=min_docfreq)

    # 4) 엔터티 가중치 + 일반어 디버프
    orgs, prods = load_entities_weight()
    boosted = []
    COMMON_DEBUFF = {"스마트","디지털","시장","글로벌","생활","기술","최근","지난해","세계","사업","플랫폼"}
    for k in keywords:
        kw = k["keyword"]; score = k["score"]
        if kw in orgs or kw in prods: score *= 1.2
        if kw in COMMON_DEBUFF: score *= 0.7
        boosted.append({"keyword": kw, "score": score})
    boosted.sort(key=lambda x: x["score"], reverse=True)

    # 5) MMR 다양화
    diversified = mmr_diversify(boosted, topn=topk, diversity=0.6)

    _vec, _X = build_tfidf(docs)

    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/keywords.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"stats": {"num_docs": len(docs)}, "keywords": diversified}, f, ensure_ascii=False, indent=2)

    # 6) 실행 메타 기록
    import datetime
    meta = {
        "module": "B",
        "mode": "PRO" if use_pro_mode() else "LITE",
        "time_utc": datetime.datetime.utcnow().isoformat() + "Z"
    }
    with open("outputs/run_meta_b.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[INFO] 모듈 B 완료 | 문서 수={len(docs)} | 상위 키워드={len(diversified)} | 출력={out_path} | 경과(초)={round(time.time() - t0, 2)}")

if __name__ == "__main__":
    main()

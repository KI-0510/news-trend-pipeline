import os
import json
import glob
import re
import time
from soynlp.normalizer import normalize
from krwordrank.word import KRWordRank
from sklearn.feature_extraction.text import TfidfVectorizer


def latest(globpat: str):
    files = sorted(glob.glob(globpat))
    return files[-1] if files else None


def load_config():
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"top_n_keywords": 30, "stopwords": []}


def clean_text(t: str) -> str:
    if not t:
        return ""
    t = re.sub(r"<.+?>", " ", t)  # 태그 제거(혹시 남아있다면)
    t = normalize(t, unicode_norm=True)
    return t.strip()


def build_docs(meta_items):
    docs = []
    for it in meta_items:
        title = clean_text(it.get("title") or it.get("title_og"))
        desc = clean_text(it.get("description") or it.get("description_og"))
        doc = (title + " " + desc).strip()
        if doc:
            docs.append(doc)
    return docs


def extract_keywords_krwordrank(docs, topk=30, stopwords=None):
    stopwords = set(stopwords or [])
    kwr = KRWordRank(min_count=3, max_length=10, beta=0.85, verbose=False)
    keywords, _, _ = kwr.extract(docs, max_iter=10)
    results = []
    for w, score in sorted(keywords.items(), key=lambda x: x[1], reverse=True):
        if len(results) >= topk:
            break
        if len(w) < 2:
            continue
        if w in stopwords:
            continue
        results.append({"keyword": w, "score": float(score)})
    return results


def build_tfidf(docs):
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vec.fit_transform(docs)
    return vec, X


def main():
    t0 = time.time()
    cfg = load_config()
    topk = int(cfg.get("top_n_keywords", 30))
    stopwords = cfg.get("stopwords", [])

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

    # KRWordRank 키워드
    keywords = extract_keywords_krwordrank(docs, topk=topk, stopwords=stopwords)

    # TF-IDF(추후 유사도/클러스터링에 활용)
    _vec, _X = build_tfidf(docs)

    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/keywords.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "stats": {"num_docs": len(docs)},
            "keywords": keywords
        }, f, ensure_ascii=False, indent=2)

    print(f"[INFO] 모듈 B 완료 | 문서 수={len(docs)} | 상위 키워드={len(keywords)} | 출력={out_path} | 경과(초)={round(time.time() - t0, 2)}")


if __name__ == "__main__":
    main()

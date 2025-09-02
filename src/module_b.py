import os
import json
import glob
import re
import unicodedata
import time
from soynlp.normalizer import normalize, repeat_normalize, emoticon_normalize
from krwordrank.word import KRWordRank
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# [추가] 조사 제거 함수
def strip_korean_particle(word: str) -> str:
    """단어 끝의 흔한 조사를 1회 제거합니다."""
    return re.sub(r"(은|는|이|가|을|를|과|와|의|에|에서|으로|로|도|만|보다|부터|까지)\$", "", word)

def latest(globpat: str):
    """주어진 패턴에 맞는 최신 파일을 반환합니다."""
    files = sorted(glob.glob(globpat))
    return files[-1] if files else None

def load_config():
    """설정 파일을 로드합니다. 없을 경우 기본값을 반환합니다."""
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"top_n_keywords": 30, "stopwords": []}

def clean_text(t: str) -> str:
    """텍스트 정규화: 태그 제거, 유니코드 정규화, 이모티콘/반복 문자 처리"""
    if not t:
        return ""
    t = re.sub(r"<.+?>", " ", t) # HTML 태그 제거
    t = unicodedata.normalize("NFKC", t) # 유니코드 정규화(전/반각 등)
    t = normalize(t) # soynlp 기본 정규화
    t = emoticon_normalize(t, num_repeats=2) # 이모티콘 축약 (예: ㅋㅋㅋㅋ → ㅋㅋ)
    t = repeat_normalize(t, num_repeats=2) # 반복 문자 축약
    return t.strip()

def dedup_docs_by_cosine(docs, threshold=0.90):
    """코사인 유사도 기반 문서 중복 제거"""
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
        # i와 매우 유사한 문서(제거) 식별
        for j in range(i + 1, len(docs)):
            if sim[i, j] >= threshold:
                removed.add(j)

    return [docs[i] for i in keep]

def build_docs(meta_items):
    """메타데이터에서 문서 생성"""
    docs = []
    for it in meta_items:
        title = clean_text(it.get("title") or it.get("title_og"))
        desc = clean_text(it.get("description") or it.get("description_og"))
        doc = (title + " " + desc).strip()
        if doc:
            docs.append(doc)
    return docs

def extract_keywords_krwordrank(docs, topk=30, stopwords=None):
    """데이터 양에 따라 KRWordRank 파라미터 자동 조정"""
    stopwords = set(stopwords or [])
    n = len(docs)

    # 문서 수에 따른 파라미터 스케일 조정
    if n < 50:
        min_count, max_iter = 2, 8
    elif n < 200:
        min_count, max_iter = 3, 10
    else:
        min_count, max_iter = 5, 12

    kwr = KRWordRank(min_count=min_count, max_length=10)
    keywords, _, _ = kwr.extract(docs, max_iter=max_iter)

    results = []
    for w, score in sorted(keywords.items(), key=lambda x: x[1], reverse=True):
        if len(results) >= topk:
            break
        
        # [적용] 조사 제거 및 키워드 필터링
        w_norm = strip_korean_particle(w)
        if len(w_norm) < 2:
            continue
        if w_norm in stopwords:
            continue
        # 숫자/기호만 있는 단어 필터링
        if re.fullmatch(r"[0-9\W_]+", w_norm):
            continue
        results.append({"keyword": w_norm, "score": float(score)})
    return results

def build_tfidf(docs):
    """TF-IDF 벡터라이저 생성"""
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

    # 문서 생성 → URL 기반 중복 제거(모듈 A) 이미 적용된 상태
    docs = build_docs(meta_items)
    if not docs:
        print("[ERROR] 문서가 비어 있음")
        raise SystemExit(1)

    # 코사인 유사도 기반 중복 제거 추가 (콘텐츠 유사도 기준)
    pre_n = len(docs)  # [추가] 중복 제거 전 문서 수
    docs = dedup_docs_by_cosine(docs, threshold=0.90)
    post_n = len(docs)  # [추가] 중복 제거 후 문서 수
    print(f"[INFO] 문서 중복 제거: {pre_n} -> {post_n}")  # [수정] 가독성 향상

    # KRWordRank 키워드 추출 (자동 튜닝 적용)
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

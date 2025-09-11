import os
import json
import glob
import re
import unicodedata
import time
import string
from collections import defaultdict
from typing import List, Tuple
from soynlp.normalizer import normalize, repeat_normalize, emoticon_normalize
from krwordrank.word import KRWordRank
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# [추가] 조사 제거 함수
def _has_jongseong(ch: str) -> bool:
    """한글 음절에 받침이 있는지 여부"""
    code = ord(ch)
    if 0xAC00 <= code <= 0xD7A3:
        return ((code - 0xAC00) % 28) != 0
    return False

def strip_korean_particle(word: str) -> str:
    """
    조사 제거를 보수적으로 수행.
    - '이'는 절대 제거하지 않음(디스플레이 보호)
    - '의'도 제거하지 않음(협의/정의 등 보호)
    - '가, 은, 는, 을, 를, 과, 와'만 받침 규칙이 맞을 때 1글자 제거
    - 그 외(에, 에서, 으로/로, 도, 만, 보다, 부터, 까지)는 건드리지 않음
    """
    if not word or len(word) < 2:
        return word

    last = word[-1]
    prev = word[-2]

    if last in ("이", "의"):
        return word

    rules = {
        "가": False,  # 앞 음절에 받침이 없어야 '가'
        "은": True,   # 받침 있으면 '은'
        "는": False,  # 받침 없으면 '는'
        "을": True,   # 받침 있으면 '을'
        "를": False,  # 받침 없으면 '를'
        "과": True,   # 받침 있으면 '과'
        "와": False,  # 받침 없으면 '와'
    }

    if last in rules and _has_jongseong(prev) == rules[last]:
        return word[:-1]
    return word

# [추가] 용언 어미 제거 함수
def strip_verb_ending(word: str) -> str:
    """흔한 동사/형용사 어말 처리 간단 컷(한 번만)"""
    return re.sub(r"(하다|하게|하고|하며|하면|하는|해요?|했다|합니다|된다|되는|될|됐다|있다|있음|또한)$", "", word)

# [추가] 키워드 정규화 함수
def normalize_keyword(w: str) -> str:
    """따옴표/양쪽 공백/양끝 기호 제거, 영문 소문자화, 의심 토큰 필터링"""
    if not w:
        return ""
    # 양끝 따옴표/기호 제거
    w = re.sub(r"^[\'\"‘’“”]+|[\'\"‘’“”]+$", "", w.strip())
    w = w.strip(string.punctuation + "·…")
    # 공백 압축
    w = re.sub(r"\s+", " ", w)
    # 영문은 소문자
    if re.fullmatch(r"[A-Za-z0-9 \-_/]+", w):
        w = w.lower()
    return w

def latest(globpat: str):
    """주어진 패턴에 맞는 최신 파일을 반환합니다."""
    files = sorted(glob.glob(globpat))
    return files[-1] if files else None

def load_config():
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"top_n_keywords": 50, "stopwords": [], "dedup_threshold": 0.90, "min_docfreq": 2}

def clean_text(t: str) -> str:
    """텍스트 정규화: 태그 제거, 유니코드 정규화, 이모티콘/반복 문자 처리"""
    if not t:
        return ""
    t = re.sub(r"<.+?>", " ", t)  # HTML 태그 제거
    t = unicodedata.normalize("NFKC", t)  # 유니코드 정규화
    t = normalize(t)  # soynlp 기본 정규화
    t = emoticon_normalize(t, num_repeats=2)  # ㅋㅋㅋ → ㅋㅋ
    t = repeat_normalize(t, num_repeats=2)    # 반복 문자 축약
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
        for j in range(i + 1, len(docs)):
            if sim[i, j] >= threshold:
                removed.add(j)
    return [docs[i] for i in keep]

def build_docs(meta_items):
    """메타데이터에서 문서 생성"""
    docs = []
    for it in meta_items:
        title = clean_text(it.get("title") or it.get("title_og"))
        body = clean_text(it.get("body") or it.get("description") or it.get("description_og"))
        doc = (title + " " + body).strip()
        if doc:
            docs.append(doc)
    return docs

# ====== stopwords 합집합 로드(설정 + 확장 사전) ======
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
STOPWORDS = set(x for x in (STOP_CFG | STOP_EXT) if x)

# ====== KRWordRank 기반 키워드 추출 ======
def extract_keywords_krwordrank(docs, topk=30):
    """데이터 양에 따라 KRWordRank 파라미터 자동 조정"""
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
        # 정규화 → 조사/어미 제거
        w_norm = normalize_keyword(w)
        w_norm = strip_korean_particle(w_norm)
        w_norm = strip_verb_ending(w_norm)
        if len(w_norm) < 2:
            continue
        # stopwords(합집합) 적용: 정규화 경량화로 비교
        if norm_kw_light(w_norm) in STOPWORDS:
            continue
        # 숫자/기호만 필터링
        if re.fullmatch(r"[0-9\W_]+", w_norm):
            continue
        results.append({"keyword": w_norm, "score": float(score)})
    return results

# [추가] 키워드 후처리 함수
def postprocess_keywords(docs, keywords, min_docfreq=1):
    """문서 빈도 필터링, 토큰 정규화, 중복 병합"""
    # 문서 빈도 계산
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
        # stopwords(합집합) 최종 필터
        if norm_kw_light(w) in STOPWORDS:
            continue

        # 문서 빈도 필터 (정확 일치 + 부분 일치)
        exact_df = df.get(w, 0)
        approx_df = max((df[t] for t in df if w in t or t in w), default=0)
        if max(exact_df, approx_df) < min_docfreq:
            continue

        if w not in merged or merged[w]["score"] < k["score"]:
            merged[w] = {"keyword": w, "score": float(k["score"])}

    return sorted(merged.values(), key=lambda x: x["score"], reverse=True)

def build_tfidf(docs):
    """TF-IDF 벡터라이저 생성"""
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vec.fit_transform(docs)
    return vec, X

# ====== 추가: 엔터티/이벤트 추출 유틸 ======
ENT_ORG  = set(_load_lines(os.path.join(DICT_DIR, "entities_org.txt")))
BRANDS   = set(_load_lines(os.path.join(DICT_DIR, "brands.txt")))
ALIASMAP = _load_json(os.path.join(DICT_DIR, "product_alias.json"), {})

def alias_merge(term: str) -> str:
    t = term.strip()
    for canon, alist in ALIASMAP.items():
        if t == canon:
            return t
        for a in alist:
            if t == a:
                return canon
    return t

def extract_entities(text: str) -> List[Tuple[str, str]]:
    """
    간단 엔터티 추출: ORG/BRAND/PRODUCT(숫자 붙은 모델) 중심
    """
    ents = []
    if not text:
        return ents
    toks = re.findall(r"[가-힣A-Za-z0-9\-\+\.]{2,}", text)
    toks = [t.strip() for t in toks if len(t.strip()) >= 2]
    for t in toks:
        if norm_kw_light(t) in STOPWORDS:
            continue
        if t in ENT_ORG:
            ents.append((t, "ORG"))
            continue
        if t in BRANDS:
            ents.append((t, "BRAND"))
            continue
        if re.match(r"^[가-힣A-Za-z]+[0-9]{1,3}[A-Za-z]?$", t):
            ents.append((t, "PRODUCT"))
    ents = [(alias_merge(e), typ) for e, typ in ents]
    return ents

EVENT_PATTERNS = [
    ("LAUNCH", r"(출시|공개|발표|선보였|공식 오픈)"),
    ("PARTNERSHIP", r"(제휴|파트너십|MOU|공동개발|공급 계약)"),
    ("INVEST", r"(투자|유치|라운드|증자|펀드)"),
    ("ORDER", r"(수주|납품|계약 체결)"),
    ("CERT", r"(인증|승인|허가|표준|규격)"),
    ("REGUL", r"(규제|지침|정책|보조금|예산)")
]

def extract_events(text: str) -> List[Tuple[str, str]]:
    evts = []
    if not text:
        return evts
    sents = re.split(r"(?<=[\.!?다])\s+", text)
    for s in sents:
        for etype, pat in EVENT_PATTERNS:
            if re.search(pat, s):
                evts.append((etype, s.strip()))
    return evts

def save_entities_events(meta_items: List[dict]):
    """
    meta에서 body/raw_body/title 등을 이용해 entities.csv, events.csv 저장
    """
    os.makedirs("outputs/export", exist_ok=True)
    ent_freq = {}      # (term,type) -> count
    ent_samples = {}   # term -> sample_count
    events_rows = []   # type, sentence, url, date

    from timeutil import to_kst_date_str

    for it in meta_items:
        url = it.get("url") or ""
        date_raw = it.get("published_time") or it.get("pubDate_raw") or ""
        try:
            date_str = to_kst_date_str(date_raw)
        except Exception:
            date_str = ""
        text = (it.get("body") or it.get("description") or "") or ""

        # 엔터티
        ents = extract_entities(text)
        seen_terms = set()
        for term, typ in ents:
            key = (term, typ)
            ent_freq[key] = ent_freq.get(key, 0) + 1
            if term not in seen_terms:
                ent_samples[term] = ent_samples.get(term, 0) + 1
                seen_terms.add(term)

        # 이벤트(원문 가까운 raw_body가 있으면 우선)
        base = (it.get("raw_body") or text)
        evs = extract_events(base)
        for et, sent in evs:
            events_rows.append({
                "type": et,
                "sentence": sent[:2000],
                "url": url,
                "date": date_str
            })

    # entities.csv
    import csv
    with open("outputs/export/entities.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["entity", "type", "freq", "sample_count"])
        for (term, typ), cnt in sorted(ent_freq.items(), key=lambda x: (-x[1], x[0][0]))[:1000]:
            w.writerow([term, typ, cnt, ent_samples.get(term, 0)])

    # events.csv
    with open("outputs/export/events.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["type", "sentence", "url", "date"])
        for row in events_rows[:2000]:
            w.writerow([row["type"], row["sentence"], row["url"], row["date"]])

def build_tfidf(docs):
    """TF-IDF 벡터라이저 생성"""
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vec.fit_transform(docs)
    return vec, X

def main():
    t0 = time.time()
    cfg = CFG  # 이미 로드된 전역 CFG 사용
    topk = int(cfg.get("top_n_keywords", 50))
    # min_docfreq 설정(없으면 기본 2)
    min_docfreq = int(cfg.get("min_docfreq", 2))

    meta_path = latest("data/news_meta_*.json")
    if not meta_path:
        print("[ERROR] data/news_meta_*.json 없음. 모듈 A부터 실행 필요")
        raise SystemExit(1)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta_items = json.load(f)

    # 문서 생성
    docs = build_docs(meta_items)
    if not docs:
        print("[ERROR] 문서가 비어 있음")
        raise SystemExit(1)

    # 코사인 유사도 기반 중복 제거
    pre_n = len(docs)
    docs = dedup_docs_by_cosine(docs, threshold=0.90)
    post_n = len(docs)
    print(f"[INFO] 문서 중복 제거: {pre_n} -> {post_n}")

    # 키워드 추출 → 후처리
    keywords = extract_keywords_krwordrank(docs, topk=topk)
    keywords = postprocess_keywords(docs, keywords, min_docfreq=min_docfreq)

    # TF-IDF(추후 활용)
    _vec, _X = build_tfidf(docs)

    # 엔터티/이벤트 내보내기
    try:
        meta_path2 = latest("data/news_meta_*.json")
        if meta_path2:
            with open(meta_path2, "r", encoding="utf-8") as _f:
                _items = json.load(_f)
            save_entities_events(_items)
            print("[INFO] entities/events exported")
    except Exception as e:
        print("[WARN] entities/events export failed:", repr(e))

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

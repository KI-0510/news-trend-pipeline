import os
import json
import re
import glob
import unicodedata
import time
from datetime import datetime

import tomotopy as tp
import google.generativeai as genai


def latest(globpat: str):
    files = sorted(glob.glob(globpat))
    return files[-1] if files else None


def clean_text(t: str) -> str:
    if not t:
        return ""
    t = re.sub(r"<.+?>", " ", t)
    t = unicodedata.normalize("NFKC", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def load_warehouse(days=30):
    files = sorted(glob.glob("data/warehouse/*.jsonl"))[-days:]
    docs, dates = [], []
    
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    title = clean_text((obj.get("title") or "").strip())
                    if not title:
                        continue
                    docs.append(title)  # 간단히 제목으로 문서 구성(원하면 확장 가능)
                    dates.append(obj.get("published"))
                except Exception:
                    continue
    return docs, dates


def load_today_meta():
    meta_path = latest("data/news_meta_*.json")
    if not meta_path:
        return [], []
    
    with open(meta_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    
    docs, dates = [], []
    for it in items:
        title = clean_text(it.get("title") or it.get("title_og"))
        desc = clean_text(it.get("description") or it.get("description_og"))
        doc = (title + " " + desc).strip()
        
        if doc:
            docs.append(doc)
            # pubDate_raw에서 날짜만 근사 추출
            d = it.get("published_time") or it.get("pubDate_raw") or ""
            m = re.search(r"(\d{4}).?(\d{2}).?(\d{2})", d)
            if m:
                dates.append(f"{m.group(1)}-{m.group(2)}-{m.group(3)}")
            else:
                dates.append(datetime.today().strftime("%Y-%m-%d"))
    return docs, dates


def simple_tokenize_ko(text: str):
    toks = re.findall(r"[가-힣A-Za-z0-9]+", text)
    toks = [t.lower() for t in toks if len(t) >= 2]
    return toks


def lda_topics(docs, k=6, topn=8, min_cf=2, iters=150):
    mdl = tp.LDAModel(k=k, alpha=0.1, eta=0.01, min_cf=min_cf)
    for d in docs:
        mdl.add_doc(simple_tokenize_ko(d))
    
    if mdl.num_docs == 0:
        return {"topics": [], "doc_topics": []}
    
    mdl.burn_in = 50
    for _ in range(iters):
        mdl.train(10)
    
    topics = []
    for ti in range(mdl.k):
        words = mdl.get_topic_words(ti, top_n=topn)
        topics.append({
            "topic_id": ti,
            "top_words": [{"word": w, "prob": float(p)} for w, p in words]
        })
    
    doc_topics = []
    for di in range(mdl.num_docs):
        dist = mdl.docs[di].get_topics(top_n=3)
        doc_topics.append([{"topic_id": tid, "prob": float(prob)} for tid, prob in dist])
    
    return {"topics": topics, "doc_topics": doc_topics}


def timeseries_by_date(dates):
    counts = {}
    for d in dates:
        if not d:
            continue
        counts[d] = counts.get(d, 0) + 1
    
    daily = [{"date": k, "count": v} for k, v in sorted(counts.items())]
    return {"daily": daily}


def gemini_insight(api_key: str, model: str, context: dict, max_tokens=1024):
    genai.configure(api_key=api_key)
    gmodel = genai.GenerativeModel(model)
    
    prompt = (
        "다음은 한국어 뉴스에서 추출한 토픽과 날짜별 기사 수 요약입니다.\n"
        "요청:\n"
        "1) 상위 토픽을 3\~5개 주제로 묶어 핵심 맥락 설명(2\~3문장)\n"
        "2) 최근 변화/스파이크가 있으면 2문장으로 짚기\n"
        "3) 실무 인사이트 3가지 bullet(구체적 액션)\n"
        f"데이터: {json.dumps(context, ensure_ascii=False)}"
    )
    
    resp = gmodel.generate_content(prompt)
    return (resp.text or "")[:max_tokens]


def main():
    t0 = time.time()
    
    # 1) 데이터 로드: 누적 우선
    docs, dates = load_warehouse(days=30)
    if not docs:
        docs, dates = load_today_meta()
    
    if not docs:
        print("[ERROR] 사용할 문서가 없습니다.")
        raise SystemExit(1)
    
    # 2) 토픽 모델링
    k = 6  # 필요 시 5\~8로 조정
    lda = lda_topics(docs, k=k, topn=8, min_cf=2, iters=150)
    
    # 3) 간단 시계열
    ts = timeseries_by_date(dates)
    
    # 4) 저장
    os.makedirs("outputs", exist_ok=True)
    
    with open("outputs/topics.json", "w", encoding="utf-8") as f:
        json.dump(lda, f, ensure_ascii=False, indent=2)
    
    with open("outputs/trend_timeseries.json", "w", encoding="utf-8") as f:
        json.dump(ts, f, ensure_ascii=False, indent=2)
    
    # 5) Gemini 인사이트(키 없으면 DRY 메시지)
    insight_text = "DRY RUN: 키 미설정으로 요약 생략."
    api_key = os.getenv("GEMINI_API_KEY", "")
    model = "gemini-1.5-flash"
    
    if api_key:
        try:
            context = {
                "topics": lda.get("topics", [])[:6],
                "timeseries": ts.get("daily", [])
            }
            insight_text = gemini_insight(api_key, model, context, max_tokens=1024)
        except Exception as e:
            print(f"[WARN] Gemini 요약 실패: {e}")
            insight_text = "요약 생성 실패(로그 참조)."
    
    insights = {
        "summary": insight_text,
        "top_topics": lda.get("topics", [])[:5],
        "evidence": {"timeseries": ts.get("daily", [])[-14:]}  # 최근 2주
    }
    
    with open("outputs/trend_insights.json", "w", encoding="utf-8") as f:
        json.dump(insights, f, ensure_ascii=False, indent=2)
    
    print(
        f"[INFO] 모듈 C 완료 | topics={len(lda.get('topics', []))} | "
        f"ts_days={len(ts.get('daily', []))} | "
        f"경과(초)={round(time.time()-t0, 2)}"
    )


if __name__ == "__main__":
    import time
    main()

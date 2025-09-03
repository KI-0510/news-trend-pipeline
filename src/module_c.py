import os
import json
import re
import glob
import unicodedata
import time
import datetime
import tomotopy as tp
import google.generativeai as genai
from email.utils import parsedate_to_datetime


def latest(globpat: str):
    files = sorted(glob.glob(globpat))
    return files[-1] if files else None


def to_date(s: str) -> str:
    today = datetime.date.today()
    if not s or not isinstance(s, str):
        return today.strftime("%Y-%m-%d")
    s = s.strip()
    
    try:
        iso = s.replace("Z", "+00:00")
        dt = datetime.datetime.fromisoformat(iso)
        d = dt.date()
    except Exception:
        try:
            dt = parsedate_to_datetime(s)
            d = dt.date()
        except Exception:
            m = re.search(r"(\d{4}).*?(\d{1,2}).*?(\d{1,2})", s)
            if m:
                y, mm, dd = int(m.group(1)), int(m.group(2)), int(m.group(3))
                try:
                    d = datetime.date(y, mm, dd)
                except Exception:
                    d = today
            else:
                d = today
    
    if d > today:
        d = today
    return d.strftime("%Y-%m-%d")


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
                    docs.append(title)  # 간단: 제목만
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
            d_raw = it.get("published_time") or it.get("pubDate_raw") or ""
            dates.append(to_date(d_raw))
    return docs, dates


def simple_tokenize_ko(text: str):
    toks = re.findall(r"[가-힣A-Za-z0-9]+", text)
    toks = [t.lower() for t in toks if len(t) >= 2]
    return toks


def lda_topics(docs, k=6, topn=8, min_cf=2, iters=150):
    mdl = tp.LDAModel(k=k, alpha=0.1, eta=0.01, min_cf=min_cf)
    for d in docs:
        mdl.add_doc(simple_tokenize_ko(d))
    
    if len(mdl.docs) == 0:  # num_docs 대신 len(mdl.docs) 사용
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
    for di in range(len(mdl.docs)):  # num_docs 대신 len(mdl.docs) 사용
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


def gemini_insight(api_key: str, model: str, context: dict, max_tokens=2048, temperature=0.6):
    genai.configure(api_key=api_key)
    gmodel = genai.GenerativeModel(model)
    
    # 1차 요청 프롬프트(구조 유지, 길이는 여유)
    prompt = (
        "아래는 한국어 뉴스에서 추출한 토픽과 날짜별 기사 수 요약입니다.\n"
        "요청:\n"
        "1) 상위 토픽을 3~5개 주제로 묶어 핵심 맥락 설명(2~3문장)\n"
        "2) 최근 변화/스파이크가 있으면 2문장으로 짚기\n"
        "3) 실무 인사이트 3가지 bullet(구체적 액션)\n"
        "주의: 문장 중간에 끊지 말고 완결된 문장으로 끝내세요.\n"
        f"데이터: {json.dumps(context, ensure_ascii=False)}"
    )
    
    resp = gmodel.generate_content(
        prompt,
        generation_config={
            "max_output_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
        }
    )
    
    text = (getattr(resp, "text", None) or "").strip()
    
    # 문장 완결성 검사: 마침표/물음표/느낌표/종결문자 없이 끝나면 후속 요청 한 번 더
    if text and not re.search(r"[\.!?]\$|[다요음]\s*\$", text):
        cont = gmodel.generate_content(
            "방금 작성한 응답의 마지막 문장부터 이어서 3~5문장으로 마무리해줘. 반복은 피하고 결론을 명확히.",
            generation_config={"max_output_tokens": 384, "temperature": temperature}
        )
        cont_text = (getattr(cont, "text", None) or "").strip()
    
        if cont_text:
            text = text + ("\n" if not text.endswith("\n") else "") + cont_text
    
    return text


def main():
    t0 = time.time()
    
    docs, dates = load_warehouse(days=30)
    if not docs:
        docs, dates = load_today_meta()
    
    if not docs:
        print("[ERROR] 사용할 문서가 없습니다.")
        raise SystemExit(1)
    
    k = 6  # 5~8 권장
    lda = lda_topics(docs, k=k, topn=8, min_cf=2, iters=150)
    ts = timeseries_by_date(dates)
    
    os.makedirs("outputs", exist_ok=True)
    
    with open("outputs/topics.json", "w", encoding="utf-8") as f:
        json.dump(lda, f, ensure_ascii=False, indent=2)
    
    with open("outputs/trend_timeseries.json", "w", encoding="utf-8") as f:
        json.dump(ts, f, ensure_ascii=False, indent=2)
    
    insight_text = "DRY RUN: 키 미설정으로 요약 생략."
    api_key = os.getenv("GEMINI_API_KEY", "")
    
    if api_key and len(api_key.strip()) > 20:
        try:
            # 컨텍스트를 더 가볍게: 토픽 5개, 각 토픽 상위 6~8개 단어만, 시계열 최근 30개 지점
            raw_topics = lda.get("topics", [])[:5]
            compact_topics = []
            
            for t in raw_topics:
                words = t.get("top_words", [])[:8]
                compact_topics.append({
                    "topic_id": t.get("topic_id"),
                    "top_words": [w["word"] for w in words]
                })
            
            compact_ts = ts.get("daily", [])[-30:]
            
            context = {"topics": compact_topics, "timeseries": compact_ts}
            
            # config.json의 llm.max_output_tokens 있으면 반영
            try:
                with open("config.json", "r", encoding="utf-8") as f:
                    cfg_all = json.load(f)
                max_tokens = int(cfg_all.get("llm", {}).get("max_output_tokens", 2048))
            except Exception:
                max_tokens = 2048
            
            insight_text = gemini_insight(
                api_key=api_key,
                model="gemini-1.5-pro",
                context=context,
                max_tokens=max_tokens,
                temperature=0.6
            )
            
        except Exception as e:
            print(f"[WARN] Gemini 요약 실패: {e}")
            insight_text = "요약 생성 실패(로그 참조)."
    else:
        print("[WARN] GEMINI_API_KEY 비어있거나 비정상 길이 — 요약 생략")
        
        
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
    main()
    

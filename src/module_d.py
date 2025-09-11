# -*- coding: utf-8 -*-
import os
import re
import json
import csv
import glob
import unicodedata
import datetime
import time
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict, Counter

# ========== 공통 로드 ==========

def latest(globpat: str):
    files = sorted(glob.glob(globpat))
    return files[-1] if files else None

def load_json(path: str, default=None):
    if default is None:
        default = None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def save_json(path: str, obj: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# ========== LLM 설정/유틸(원복) ==========
# config.json에 llm_config가 있다면 사용, 없으면 합리적 기본값
def load_config():
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def llm_config(cfg: dict) -> dict:
    llm = cfg.get("llm") or {}
    return {
        "model": llm.get("model", "gemini-1.5-flash"),
        "max_output_tokens": int(llm.get("max_output_tokens", 2048)),
        "temperature": float(llm.get("temperature", 0.3)),
    }

CFG = load_config()
LLM = llm_config(CFG)

# 재시도 데코레이터(간단)
def retry(max_attempts=3, backoff=0.8, exceptions=(Exception,), circuit_trip=4):
    def deco(fn):
        def wrapper(*args, **kwargs):
            last = None
            for i in range(max_attempts):
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    last = e
                    time.sleep(backoff * (2**i))
            raise last
        return wrapper
    return deco

# LLM 출력 정리/파서
def strip_code_fence(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"^```[\t ]*\w*[\t ]*\n", "", t, flags=re.M)
    t = re.sub(r"\n```[\t ]*$", "", t, flags=re.M)
    return t.strip()

def clean_json_text(t: str) -> str:
    t = strip_code_fence(t or "")
    t = t.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    t = re.sub(r"[\u0000-\u0008\u000B\u000C\u000E-\u001F\u2028\u2029]", "", t)
    t = re.sub(r",\s*(\}|\])", r"\1", t)
    t = t.lstrip("\ufeff")
    return t.strip()

def parse_json_array_or_object(t: str):
    s = clean_json_text(t)
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict) and isinstance(obj.get("ideas"), list):
            return obj["ideas"]
    except Exception:
        return None
    return None

def extract_balanced_array(t: str):
    s = clean_json_text(t)
    start = s.find("[")
    if start == -1:
        return None
    depth, end = 0, -1
    for i, ch in enumerate(s[start:], start):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end == -1:
        return None
    payload = s[start:end+1]
    try:
        arr = json.loads(payload)
        return arr if isinstance(arr, list) else None
    except Exception:
        return None

def extract_objects_sequence(t: str, max_items=10):
    s = clean_json_text(t)
    out, i, n = [], 0, len(s)
    while i < n and len(out) < max_items:
        start = s.find("{", i)
        if start == -1:
            break
        depth, end = 0, -1
        j = start
        while j < n:
            ch = s[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = j
                    break
            j += 1
        if end == -1:
            break
        chunk = s[start:end+1]
        try:
            obj = json.loads(clean_json_text(chunk))
            if isinstance(obj, dict):
                out.append(obj)
        except Exception:
            pass
        i = end + 1
    return out

def extract_ndjson_lines(t: str, max_items=10):
    ideas = []
    for line in (t or "").splitlines():
        if len(ideas) >= max_items:
            break
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^[\-\*\d\.\)\s]+", "", line)
        try:
            obj = json.loads(clean_json_text(line))
            if isinstance(obj, dict):
                ideas.append(obj)
                continue
        except Exception:
            objs = extract_objects_sequence(line, max_items=2)
            for o in objs:
                if isinstance(o, dict):
                    ideas.append(o)
                    if len(ideas) >= max_items:
                        break
    return ideas

def extract_ideas_any(text: str, want=5):
    arr = parse_json_array_or_object(text)
    if isinstance(arr, list) and arr:
        return arr
    arr2 = extract_balanced_array(text)
    if isinstance(arr2, list) and arr2:
        return arr2
    objs = extract_objects_sequence(text, max_items=want)
    if objs:
        return objs
    nd = extract_ndjson_lines(text, max_items=want)
    if nd:
        return nd
    return None

# ========== 날짜/텍스트 유틸 ==========

def clean_text(t: str) -> str:
    if not t:
        return ""
    t = re.sub(r"<.+?>", " ", t)
    t = unicodedata.normalize("NFKC", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def to_kst_date_str(s: str) -> str:
    from email.utils import parsedate_to_datetime
    try:
        if not s:
            raise ValueError
        s2 = s.replace("Z", "+00:00")
        dt = datetime.datetime.fromisoformat(s2)
        d = dt.date()
    except Exception:
        try:
            dt = parsedate_to_datetime(s)
            d = dt.date()
        except Exception:
            m = re.search(r"(\d{4}).*?(\d{1,2}).*?(\d{1,2})", s or "")
            if m:
                y, mm, dd = int(m.group(1)), int(m.group(2)), int(m.group(3))
                d = datetime.date(y, mm, dd)
            else:
                d = datetime.date.today()
    today = datetime.date.today()
    if d > today:
        d = today
    return d.strftime("%Y-%m-%d")

# ========== 회사×토픽 매트릭스 ==========

def extract_orgs(text: str) -> List[str]:
    if not text:
        return []
    toks = re.findall(r"[가-힣A-Za-z0-9\-\+\.]{2,}", text)
    orgs = []
    for t in toks:
        if re.match(r"^[가-힣A-Za-z]+[0-9]{1,3}[A-Za-z]?$", t):
            continue
        if t in ("기자", "무단전재", "재배포", "관련기사", "사진", "영상"):
            continue
        if len(t) >= 2 and not re.fullmatch(r"[0-9\W_]+", t):
            orgs.append(t)
    cnt = Counter(orgs)
    out = [w for w, _ in cnt.most_common(10)]
    return out

def load_topic_labels(topics_obj: Dict[str, Any], topn=5) -> List[Dict[str, Any]]:
    labels = []
    for t in topics_obj.get("topics", []):
        words = [w.get("word","") for w in (t.get("top_words") or [])][:topn]
        labels.append({"topic_id": t.get("topic_id"), "words": [w for w in words if w]})
    return labels

def export_company_topic_matrix(meta_items: List[Dict[str,Any]], topic_labels: List[Dict[str,Any]]) -> None:
    os.makedirs("outputs/export", exist_ok=True)
    topic_wordsets = []
    for tl in topic_labels:
        ws = set([w for w in tl["words"] if w])
        topic_wordsets.append((tl["topic_id"], ws))
    matrix = defaultdict(lambda: defaultdict(int))
    for it in meta_items:
        text = (it.get("body") or it.get("description") or "") or ""
        if not text:
            continue
        orgs = extract_orgs(text)
        low = text.lower()
        for org in orgs:
            for tid, ws in topic_wordsets:
                hit = 0
                for w in ws:
                    if w and (w.lower() in low):
                        hit += 1
                if hit > 0:
                    matrix[org][tid] += hit
    all_tids = sorted(set([tid for _, d in matrix.items() for tid in d.keys()]))
    with open("outputs/export/company_topic_matrix.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["org"] + [f"topic_{tid}" for tid in all_tids])
        for org, row in sorted(matrix.items(), key=lambda x: x[0].lower()):
            w.writerow([org] + [row.get(tid, 0) for tid in all_tids])

# ========== 스코어링/증거 ==========

def clamp01(x): 
    return max(0.0, min(1.0, float(x)))

def normalize_score(x, lo, hi):
    if hi <= lo:
        return 0.0
    return clamp01((x - lo) / (hi - lo))

NEG_WORDS = ["논란", "우려", "리스크", "규제", "지연", "지적", "하락", "부진", "적자", "연기"]

def pick_evidence(term: str, items: List[Dict[str,Any]], limit=3):
    ev = []
    term_low = (term or "").lower()
    for it in items:
        base = (it.get("raw_body") or it.get("body") or it.get("description") or "")
        if not base:
            continue
        low = base.lower()
        if term_low in low:
            sents = re.split(r"(?<=[\.!?다])\s+", base)
            for s in sents:
                if term_low in (s or "").lower():
                    ev.append({
                        "sentence": (s or "").strip()[:400],
                        "url": it.get("url") or "",
                        "date": to_kst_date_str(it.get("published_time") or it.get("pubDate_raw") or "")
                    })
                    if len(ev) >= limit:
                        return ev
    return ev

# ========== LLM 프롬프트/정규화 ==========

def build_schema_hint() -> Dict[str, Any]:
    # Check D 축소 스키마를 그대로 유도
    return {
        "idea": "아이디어 한 줄 제목",
        "problem": "해결하려는 문제(2-3문장, 220자 이내)",
        "target_customer": "핵심 타깃(산업/직군/조직 규모 명확히)",
        "value_prop": "핵심 가치제안(차별점, 180자 이내)",
        "solution": ["핵심 기능 bullet 최대 4개"],
        "risks": ["리스크/규제 bullet 3개 내"],
        "priority_score": "우선순위 점수(0.0~5.0, 숫자)"
    }

def load_context_for_prompt() -> Dict[str, Any]:
    keywords = load_json("outputs/keywords.json", default={"keywords": []}) or {"keywords": []}
    topics = load_json("outputs/topics.json", default={"topics": []}) or {"topics": []}
    insights = load_json("outputs/trend_insights.json", default={"summary": "", "top_topics": [], "evidence": {}}) or {"summary": "", "top_topics": [], "evidence": {}}
    trend_strength_path = "outputs/export/trend_strength.csv"
    events_path = "outputs/export/events.csv"

    # 요약 길이 제한
    summary = (insights.get("summary") or "").strip()
    if len(summary) > 1200:
        summary = summary[:1200] + "…"

    kw_simple = [{"keyword": k.get("keyword",""), "score": k.get("score",0)} for k in (keywords.get("keywords") or [])[:20]]
    tp_simple = []
    for t in (topics.get("topics") or [])[:6]:
        words = [w.get("word","") for w in (t.get("top_words") or [])][:6]
        tp_simple.append({"topic_id": t.get("topic_id"), "words": words})

    # trend_strength 상위 일부(스파이크 지표)만 가져오기
    trend_rows = []
    try:
        with open(trend_strength_path, "r", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                try:
                    trend_rows.append({
                        "term": r["term"],
                        "cur": int(r.get("cur",0) or 0),
                        "z_like": float(r.get("z_like",0.0) or 0.0)
                    })
                except Exception:
                    continue
        trend_rows = sorted(trend_rows, key=lambda x: (x["z_like"], x["cur"]), reverse=True)[:30]
    except Exception:
        trend_rows = []

    # 이벤트 유형 요약(최근 기사에서 어떤 이벤트가 등장하는지)
    evt_summary = Counter()
    try:
        with open(events_path, "r", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                et = r.get("type")
                if et:
                    evt_summary[et] += 1
    except Exception:
        pass
    events_simple = dict(evt_summary)

    return {
        "summary": summary,
        "keywords": kw_simple,
        "topics": tp_simple,
        "trends": trend_rows,
        "events": events_simple
    }

def build_prompt(context: Dict[str, Any], want: int = 5) -> str:
    schema = build_schema_hint()
    return (
        "다음 컨텍스트(키워드/토픽/요약/스파이크/이벤트)를 기반으로 B2B에 유의미한 신사업 아이디어를 제안해 주세요.\n"
        f"- 아이디어 개수: 최대 {want}개\n"
        "- JSON 배열 형식만 출력하세요. 배열 이외의 텍스트를 출력하지 마세요.\n"
        "- 각 아이템은 아래 스키마 키를 정확히 사용하세요.\n"
        f"스키마: {json.dumps(schema, ensure_ascii=False)}\n"
        "- 제약:\n"
        "  1) 서로 유사한 테마/표현 금지(중복 금지). 각 아이디어의 '차별화 포인트'를 1문장 포함.\n"
        "  2) 카테고리 분산: [제품/서비스/플랫폼/파트너십(조달)/데이터·분석] 중 서로 다른 카테고리에서 최소 3개 이상 포함.\n"
        "  3) 산업 맥락: 디스플레이/사이니지/전자 제조·조달/모빌리티-디스플레이 접점을 최소 2개 아이디어에 반영.\n"
        "  4) 지역 명시: KR/JP/EU 중 하나를 각 아이디어에 1개 이상 명시.\n"
        "  5) 'Why now'를 각 아이디어에 1문장으로 포함(최근 스파이크/이벤트/정책을 근거로).\n"
        "- solution, risks는 리스트로 주세요.\n"
        "- priority_score는 0.0~5.0의 숫자(float)로 주세요.\n"
        "컨텍스트:\n"
        f"{json.dumps(context, ensure_ascii=False)}"
    )
    
# ========== Gemini 호출 ==========

@retry(max_attempts=3, backoff=0.8, exceptions=(Exception,), circuit_trip=4)
def _gen_content(model, prompt, max_tokens, temperature):
    return model.generate_content(
        prompt,
        generation_config={
            "max_output_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
        }
    )

def call_gemini(prompt: str) -> str:
    import google.generativeai as genai
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY 환경 변수가 없습니다.")
    genai.configure(api_key=api_key)
    model_name = str(LLM.get("model", "gemini-1.5-flash"))
    max_tokens = int(LLM.get("max_output_tokens", 2048))
    temperature = float(LLM.get("temperature", 0.3))
    model = genai.GenerativeModel(model_name)
    resp = _gen_content(model, prompt, max_tokens, temperature)
    text = (getattr(resp, "text", None) or "").strip()
    return text

# ========== 결과 정규화/보강 ==========

def as_list(x, max_len=4) -> List[str]:
    if isinstance(x, list):
        out = []
        for v in x[:max_len]:
            if v is None:
                continue
            s = str(v).strip()
            if s:
                out.append(s)
        return out
    if x is None:
        return []
    s = str(x).strip()
    return [s] if s else []

def clip_text(s: Optional[str], max_chars: int) -> str:
    s = (s or "").strip()
    if len(s) <= max_chars:
        return s
    return s[:max_chars].rstrip() + "…"

def to_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)

def normalize_item(it: Dict[str, Any]) -> Dict[str, Any]:
    idea = it.get("idea") or it.get("title") or it.get("name") or ""
    problem = it.get("problem") or it.get("pain") or ""
    target = it.get("target_customer") or it.get("target") or it.get("audience") or ""
    value = it.get("value_prop") or it.get("value") or it.get("description") or ""
    solution = it.get("solution") or it.get("solutions") or []
    risks = it.get("risks") or it.get("risk") or []
    score = it.get("priority_score", it.get("score", 0))
    score = max(0.0, min(5.0, to_float(score, 0.0)))

    idea = clip_text(idea, 100)
    problem = clip_text(problem, 300)
    target = clip_text(target, 120)
    value = clip_text(value, 220)
    solution = as_list(solution, max_len=4)
    risks = as_list(risks, max_len=3)

    out = {
        "idea": idea,
        "problem": problem,
        "target_customer": target,
        "value_prop": value,
        "solution": solution,
        "risks": risks,
        "priority_score": round(score, 1),
        "title": idea,
        "score": round(score, 1)  # 0~5 스케일 그대로도 유지
    }
    return out

# ========== 기회 생성(LLM + 신호 보강) ==========

def load_trend_strength_csv(path: str) -> List[Dict[str,Any]]:
    rows = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                r["term"] = r.get("term","")
                r["cur"] = int(r.get("cur", 0) or 0)
                try:
                    r["z_like"] = float(r.get("z_like", 0.0) or 0.0)
                except Exception:
                    r["z_like"] = 0.0
                r["total"] = int(r.get("total", 0) or 0)
                rows.append(r)
    except Exception:
        pass
    return rows

def load_events_csv(path: str) -> List[Dict[str,str]]:
    rows = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                rows.append(r)
    except Exception:
        pass
    return rows

def make_opportunities_llm(meta_items: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    # 1) 컨텍스트 구성 → 프롬프트
    want = 5
    context = load_context_for_prompt()
    prompt = build_prompt(context, want=want)

    # 2) LLM 호출
    try:
        raw_text = call_gemini(prompt)
    except Exception as e:
        print(f"[ERROR] Gemini 호출 실패: {e}")
        return []

    # 3) 파싱/정규화
    ideas_raw = extract_ideas_any(raw_text, want=want) or []
    items = []
    for it in ideas_raw:
        try:
            norm = normalize_item(it)
            if norm["idea"] and norm["value_prop"]:
                items.append(norm)
        except Exception:
            continue

    # 4) 상위 5개만 (priority_score 기준)
    items.sort(key=lambda x: x.get("priority_score", 0.0), reverse=True)
    return items[:want]

def enrich_with_signals(ideas: List[Dict[str,Any]],
                        meta_items: List[Dict[str,Any]],
                        trend_rows: List[Dict[str,Any]],
                        events_rows: List[Dict[str,str]]) -> List[Dict[str,Any]]:
    # 트렌드 인덱스
    trend_idx = {r.get("term",""): r for r in trend_rows}
    # 이벤트 요약
    event_hit = defaultdict(int)
    for r in events_rows:
        et = r.get("type")
        if et:
            event_hit[et] += 1

    ts_cur_vals, ts_z_vals = [], []
    for it in ideas:
        term = it.get("idea","")
        tr = trend_idx.get(term, {})
        ts_cur_vals.append(int(tr.get("cur",0)))
        try:
            ts_z_vals.append(float(tr.get("z_like",0.0)))
        except Exception:
            ts_z_vals.append(0.0)
    cur_hi = max([1] + ts_cur_vals) if ts_cur_vals else 1
    cur_lo = min([0] + ts_cur_vals) if ts_cur_vals else 0
    z_hi  = max([0.0] + ts_z_vals) if ts_z_vals else 0.0
    z_lo  = min([0.0] + ts_z_vals) if ts_z_vals else 0.0

    out = []
    for it in ideas:
        term = (it.get("idea") or "").strip()
        tr = trend_idx.get(term, {})
        cur = int(tr.get("cur", 0))
        z   = float(tr.get("z_like", 0.0) or 0.0)

        # 시장/시급/실행/리스크
        s_market = 0.75 * normalize_score(cur, cur_lo, cur_hi) + 0.25 * normalize_score(it.get("priority_score",0.0), 0.0, 5.0)
        s_urg = normalize_score(z, z_lo, z_hi)
        evt_boost = 0.0
        if event_hit.get("LAUNCH",0)>0: evt_boost += 0.10
        if event_hit.get("PARTNERSHIP",0)>0: evt_boost += 0.08
        if event_hit.get("INVEST",0)>0: evt_boost += 0.07
        if event_hit.get("ORDER",0)>0: evt_boost += 0.06
        if event_hit.get("CERT",0)>0: evt_boost += 0.05
        if event_hit.get("REGUL",0)>0: evt_boost += 0.04
        s_urg = clamp01(s_urg + evt_boost)

        s_feas = 0.6  # 기본값

        evid = pick_evidence(term, meta_items, limit=3)
        risk = 0.0
        for e in evid:
            if any(nw in (e.get("sentence") or "") for nw in NEG_WORDS):
                risk += 0.10
        risk = clamp01(risk)

        # 0~100 확장 점수(리포트 테이블 가독성 위해 추가 유지)
        score100 = 100.0 * (0.40 * s_market + 0.35 * s_urg + 0.25 * s_feas) - 100.0 * risk
        score100 = max(0.0, min(100.0, score100))

        it["score"] = round(score100, 2)
        it["score_breakdown"] = {
            "market": round(s_market, 3),
            "urgency": round(s_urg, 3),
            "feasibility": round(s_feas, 3),
            "risk": round(risk, 3),
            "notes": {
                "cur": cur,
                "z_like": round(z,3),
                "events_any": sum(event_hit.values()),
            }
        }
        it["evidence"] = evid if isinstance(evid, list) else []
        # Check D 호환 필드 보강(이미 채워져 있어도 재보장)
        it["title"] = it.get("title") or it["idea"]
        it["problem"] = it.get("problem") or f"{it['idea']} 관련 과제가 상존함."
        it["target_customer"] = it.get("target_customer") or "기업(B2B)"
        it["value_prop"] = it.get("value_prop") or f"{it['idea']} 도입 가치(비용/품질/경험 개선)."
        it["solution"] = it.get("solution") or ["파일럿→제휴→인증 확보"]
        it["risks"] = it.get("risks") or ["규제/표준 불확실성", "비용/ROI 불확실성"]
        it["priority_score"] = it.get("priority_score", 0.0)
        out.append(it)

    # priority_score(0~5) 우선 정렬, 동점 시 score(0~100) 보조 정렬
    out.sort(key=lambda x: (x.get("priority_score",0.0), x.get("score",0.0)), reverse=True)
    return out

# ========== 메인 ==========

def main():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/export", exist_ok=True)

    # 입력 로드
    meta_path = latest("data/news_meta_*.json")
    meta_items = load_json(meta_path, [])
    keywords_obj = load_json("outputs/keywords.json", {"keywords":[]})
    topics_obj   = load_json("outputs/topics.json", {"topics":[]})

    # 회사×토픽 매트릭스
    topic_labels = load_topic_labels(topics_obj, topn=5)
    try:
        export_company_topic_matrix(meta_items, topic_labels)
        print("[INFO] company_topic_matrix.csv exported")
    except Exception as e:
        print("[WARN] company_topic_matrix export failed:", repr(e))

    # LLM 생성 → 신호로 보강
    try:
        ideas_llm = make_opportunities_llm(meta_items)
    except Exception as e:
        print("[ERROR] LLM stage failed:", repr(e))
        ideas_llm = []

    trend_rows = load_trend_strength_csv("outputs/export/trend_strength.csv")
    events_rows = load_events_csv("outputs/export/events.csv")

    ideas_final = enrich_with_signals(ideas_llm, meta_items, trend_rows, events_rows)

    save_json("outputs/biz_opportunities.json", {"ideas": ideas_final})
    print("[INFO] Module D done | ideas=%d" % len(ideas_final))

if __name__ == "__main__":
    main()

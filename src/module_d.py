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

# ========== 공통 로드/유틸 ==========
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

# --- LLM 안전 호출 유틸(모듈 C와 동일) ---
def _gen_cfg(llm: dict):
    return {
        "temperature": float(llm.get("temperature", 0.3)),
        "max_output_tokens": int(llm.get("max_output_tokens", 2048)),  # D는 2048~4096 권장
    }

def _extract_text(resp):
    try:
        cands = getattr(resp, "candidates", []) or []
        if not cands:
            return ""
        parts = getattr(cands[0], "content", None)
        parts = getattr(parts, "parts", []) if parts else []
        texts = []
        for p in parts:
            t = getattr(p, "text", None)
            if t:
                texts.append(t)
        return " ".join(texts).strip()
    except Exception:
        return ""

def llm_generate(model, prompt, llm_cfg, shrink_hint=None):
    gen_cfg = _gen_cfg(llm_cfg)
    resp = model.generate_content(prompt, generation_config=gen_cfg)
    txt = _extract_text(resp)
    try:
        fr = getattr(resp.candidates[0], "finish_reason", None)
    except Exception:
        fr = None
    if (not txt) and fr == 2 and shrink_hint:  # MAX_TOKENS
        cfg2 = dict(gen_cfg)
        cfg2["max_output_tokens"] = int(gen_cfg.get("max_output_tokens", 2048)) * 2
        resp2 = model.generate_content(f"{prompt}\n{shrink_hint}", generation_config=cfg2)
        txt = _extract_text(resp2)
    if (not txt) and fr == 3:  # SAFETY
        resp3 = model.generate_content(prompt + "\n- 안전 가이드라인을 준수하고 민감/금지 컨텐츠는 언급하지 마세요.", generation_config=gen_cfg)
        txt = _extract_text(resp3)
    return txt or ""

# ========== 회사×토픽 매트릭스: ORG 잡음 제거 ==========
ORG_BAD_PATTERNS = [
    r"^\d{1,2}일$", r"^\d{4}$", r"^\d+(억원|조원|달러|원)$",
    r"^[0-9,\.]+(달러|원)$", r"^\d{1,3}(천|만|억|조)"
]
ORG_STOP = {
    "국내","대한민국","서울","경제자유구역","경제자유구역은","개선","경쟁력","거래일보다",
    "기업","시장","브랜드","글로벌","전문","최대","지난해","최근","일자리","지역","생활","기술","디지털","스마트",
    "tv","tv와","전자","디스플레이","11일","이라며","이라고"
}

def norm_org_token(t: str) -> str:
    t = (t or "").strip()
    if t.endswith("의") and len(t) >= 3:
        t = t[:-1]
    if len(t) >= 3 and t[-1] in ("은","는","이","가","을","를","과","와"):
        t = t[:-1]
    return t

def is_bad_org_token(t: str) -> bool:
    if not t or len(t) < 2:
        return True
    base = t.lower()
    if base in ORG_STOP:
        return True
    if re.fullmatch(r"^[0-9\W_]+$", base):
        return True
    for pat in ORG_BAD_PATTERNS:
        if re.fullmatch(pat, t):
            return True
    return False

def extract_orgs(text: str) -> List[str]:
    if not text:
        return []
    toks = re.findall(r"[가-힣A-Za-z0-9\-\+\.]{2,}", text)
    toks = [norm_org_token(t) for t in toks if t and len(t.strip()) >= 2]
    # 화이트리스트(있으면 우선)
    def _load_lines(p):
        try:
            with open(p, encoding="utf-8") as f:
                return [x.strip() for x in f if x.strip()]
        except Exception:
            return []
    ENT_ORG = set(_load_lines("data/dictionaries/entities_org.txt"))
    BRANDS  = set(_load_lines("data/dictionaries/brands.txt"))
    cand = []
    for t in toks:
        if is_bad_org_token(t):
            continue
        if re.match(r"^[가-힣A-Za-z]+[0-9]{1,3}[A-Za-z]?$", t):  # 모델형 제외
            continue
        cand.append(t)
    cnt = Counter(cand)
    out = []
    for w, c in cnt.most_common(50):
        if w in ENT_ORG or w in BRANDS:
            out.append(w)
        elif c >= 2 and len(w) >= 2:
            out.append(w)
        if len(out) >= 15:
            break
    # 최종 보호막
    out = [o for o in out if not is_bad_org_token(o)]
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
        orgs = [o for o in extract_orgs(text) if not is_bad_org_token(o)]
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

# ========== LLM 프롬프트/파서/정규화 ==========
def build_schema_hint() -> Dict[str, Any]:
    return {
        "idea": "아이디어 한 줄 제목",
        "problem": "해결하려는 문제(2-3문장, 220자 이내)",
        "target_customer": "핵심 타깃(산업/직군/조직 규모 명확히)",
        "value_prop": "핵심 가치제안(차별점, 180자 이내)",
        "solution": ["핵심 기능 bullet 최대 4개"],
        "risks": ["리스크/규제 bullet 3개 내"],
        "priority_score": "우선순위 점수(0.0~5.0, 숫자)"
    }

def build_prompt(context: Dict[str, Any], want: int = 5) -> str:
    schema = build_schema_hint()
    return (
        "다음 컨텍스트(키워드/토픽/요약/스파이크/이벤트)를 기반으로 B2B에 유의미한 신사업 아이디어를 제안해 주세요.\n"
        f"- 아이디어 개수: 정확히 {want}개(반드시 5개)\n"
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

def strip_code_fence(text: str) -> str:
    t = (text or "").strip()
    tick3 = "\x60\x60\x60"  # 백틱 3개를 안전하게 표현 [백틱 = \x60]
    # 멀티라인 정규식으로 코드펜스 오프너/클로저 제거
    open_pat = re.compile(rf"^{tick3}[\t ]*\w*[\t ]*\n", re.M)
    close_pat = re.compile(rf"\n{tick3}[\t ]*$", re.M)
    t = open_pat.sub("", t)
    t = close_pat.sub("", t)
    return t

def clean_json_text2(t: str) -> str:
    t = strip_code_fence(t or "")
    tick3 = "\x60\x60\x60"  # ``` 를 안전하게 표현
    if tick3 in t:
        t = t.replace(f"{tick3}json", tick3).replace(tick3, "\n")
    t = t.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    t = re.sub(r"[\u0000-\u0008\u000B\u000C\u000E-\u001F\u2028\u2029]", "", t)
    t = re.sub(r",\s*(\}|\])", r"\1", t)
    t = t.lstrip("\ufeff").strip()
    return t


def parse_json_array_or_object(t: str):
    s = clean_json_text2(t)
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
    s = clean_json_text2(t)
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
    s = clean_json_text2(t)
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
            obj = json.loads(clean_json_text2(chunk))
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
        line = re.sub(r"^[\-*\d\.\)\s]+", "", line)
        try:
            obj = json.loads(clean_json_text2(line))
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
        "score": round(score, 1)
    }
    return out

# ========== LLM 호출 ==========
def load_config2():
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def llm_config2(cfg: dict) -> dict:
    llm = cfg.get("llm") or {}
    return {
        "model": llm.get("model", "gemini-1.5-flash"),
        "max_output_tokens": int(llm.get("max_output_tokens", 2048)),
        "temperature": float(llm.get("temperature", 0.3)),
    }

CFG2 = load_config2()
LLM2 = llm_config2(CFG2)

def load_context_for_prompt() -> Dict[str, Any]:
    keywords = load_json("outputs/keywords.json", default={"keywords": []}) or {"keywords": []}
    topics = load_json("outputs/topics.json", default={"topics": []}) or {"topics": []}
    insights = load_json("outputs/trend_insights.json", default={"summary": "", "top_topics": [], "evidence": {}}) or {"summary": "", "top_topics": [], "evidence": {}}
    trend_strength_path = "outputs/export/trend_strength.csv"
    events_path = "outputs/export/events.csv"

    summary = (insights.get("summary") or "").strip()
    if len(summary) > 1200:
        summary = summary[:1200] + "…"

    kw_simple = [{"keyword": k.get("keyword",""), "score": k.get("score",0)} for k in (keywords.get("keywords") or [])[:20]]
    tp_simple = []
    for t in (topics.get("topics") or [])[:6]:
        words = [w.get("word","") for w in (t.get("top_words") or [])][:6]
        tp_simple.append({"topic_id": t.get("topic_id"), "words": words})

    trend_rows = []
    try:
        with open(trend_strength_path, "r", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                try:
                    trend_rows.append({"term": r["term"], "cur": int(r.get("cur",0) or 0), "z_like": float(r.get("z_like",0.0) or 0.0)})
                except Exception:
                    continue
        trend_rows = sorted(trend_rows, key=lambda x: (x["z_like"], x["cur"]), reverse=True)[:30]
    except Exception:
        trend_rows = []

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

    return {"summary": summary, "keywords": kw_simple, "topics": tp_simple, "trends": trend_rows, "events": events_simple}

def make_opportunities_llm(meta_items: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    # 프롬프트 구성
    want = 5
    context = load_context_for_prompt()
    prompt = build_prompt(context, want=want)

    # LLM 호출 (안전 래퍼)
    import google.generativeai as genai
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        print("[ERROR] GEMINI_API_KEY 환경 변수가 없습니다.")
        return []
    genai.configure(api_key=api_key)
    model_name = str(LLM2.get("model", "gemini-1.5-flash"))
    model = genai.GenerativeModel(model_name)

    raw_text = llm_generate(
        model,
        prompt,
        LLM2,
        shrink_hint="- 표는 생략하고 6~8줄 서술형으로 요약하세요."
    )
    if not raw_text:
        print("[ERROR] LLM 응답이 비어 있습니다.")
        return []

    # 파싱 → 정규화
    ideas_raw = extract_ideas_any(raw_text, want=want) or []
    items = []
    for it in ideas_raw:
        try:
            norm = normalize_item(it)
            if norm["idea"] and norm["value_prop"]:
                items.append(norm)
        except Exception:
            continue

    items.sort(key=lambda x: x.get("priority_score", 0.0), reverse=True)
    return items[:want]

# ========== 신호 보강 ==========
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

def enrich_with_signals(ideas: List[Dict[str,Any]],
                        meta_items: List[Dict[str,Any]],
                        trend_rows: List[Dict[str,Any]],
                        events_rows: List[Dict[str,str]]) -> List[Dict[str,Any]]:
    trend_idx = {r.get("term",""): r for r in trend_rows}
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

        s_feas = 0.6

        evid = pick_evidence(term, meta_items, limit=3)
        risk = 0.0
        for e in evid:
            if any(nw in (e.get("sentence") or "") for nw in NEG_WORDS):
                risk += 0.10
        risk = clamp01(risk)

        score100 = 100.0 * (0.40 * s_market + 0.35 * s_urg + 0.25 * s_feas) - 100.0 * risk
        score100 = max(0.0, min(100.0, score100))

        it["score"] = round(score100, 2)
        it["score_breakdown"] = {
            "market": round(s_market, 3),
            "urgency": round(s_urg, 3),
            "feasibility": round(s_feas, 3),
            "risk": round(risk, 3),
            "notes": {"cur": cur, "z_like": round(z,3), "events_any": sum(event_hit.values())}
        }
        it["evidence"] = evid if isinstance(evid, list) else []
        it["title"] = it.get("title") or it["idea"]
        it["problem"] = it.get("problem") or f"{it['idea']} 관련 과제가 상존함."
        it["target_customer"] = it.get("target_customer") or "기업(B2B)"
        it["value_prop"] = it.get("value_prop") or f"{it['idea']} 도입 가치(비용/품질/경험 개선)."
        it["solution"] = it.get("solution") or ["파일럿→제휴→인증 확보"]
        it["risks"] = it.get("risks") or ["규제/표준 불확실성", "비용/ROI 불확실성"]
        it["priority_score"] = it.get("priority_score", 0.0)

        out.append(it)

    out.sort(key=lambda x: (x.get("priority_score",0.0), x.get("score",0.0)), reverse=True)
    return out

# ========== Top5 보장 ==========
def fill_opportunities_to_five(ideas: list, keywords_obj: dict, want: int = 5) -> list:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    existing = ideas[:]
    if len(existing) >= want:
        return existing[:want]

    cands = [(k.get("keyword",""), float(k.get("score", 0))) for k in (keywords_obj.get("keywords") or [])[:50]]
    cands = [c for c in cands if c[0] and len(c[0]) >= 2]

    titles = [it.get("idea") or it.get("title") or "" for it in existing if (it.get("idea") or it.get("title"))]
    vec = TfidfVectorizer(analyzer="char", ngram_range=(3,5))

    if titles:
        M = vec.fit_transform(titles + [c[0] for c in cands])
        base = M[:len(titles)]
        pool = M[len(titles):]
    else:
        M = vec.fit_transform([c[0] for c in cands])
        base = None
        pool = M

    used = set(titles)
    for i, (term, _) in enumerate(cands):
        if len(existing) >= want:
            break
        if term in used:
            continue
        if base is not None:
            sim = cosine_similarity(pool[i:i+1], base).max() if base.shape[0] > 0 else 0.0
            if sim >= 0.6:
                continue
        sk = {
            "idea": term, "title": term,
            "problem": f"{term} 관련 시장/도입/규격 이슈를 해결할 기회.",
            "target_customer": "기업(B2B)",
            "value_prop": f"{term} 도입으로 비용/품질/경험을 개선.",
            "solution": ["파일럿", "파트너십", "인증/규격 검토", "조달/유통 테스트"],
            "risks": ["규제/표준 불확실성", "ROI 불확실성"],
            "priority_score": 3.0,
            "score": 60.0,
            "score_breakdown": {"market":0.5,"urgency":0.5,"feasibility":0.6,"risk":0.0,"notes":{}},
            "evidence": []
        }
        existing.append(sk)
        used.add(term)

    return existing[:want]

# ========== 메인 ==========
def main():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/export", exist_ok=True)

    meta_path = latest("data/news_meta_*.json")
    meta_items = load_json(meta_path, [])

    keywords_obj = load_json("outputs/keywords.json", {"keywords":[]})
    topics_obj   = load_json("outputs/topics.json", {"topics":[]})

    trend_rows = load_trend_strength_csv("outputs/export/trend_strength.csv")
    events_rows = load_events_csv("outputs/export/events.csv")

    topic_labels = load_topic_labels(topics_obj, topn=5)
    try:
        export_company_topic_matrix(meta_items, topic_labels)
        print("[INFO] company_topic_matrix.csv exported")
    except Exception as e:
        print("[WARN] company_topic_matrix export failed:", repr(e))

    try:
        ideas_llm = make_opportunities_llm(meta_items)
    except Exception as e:
        print("[ERROR] LLM stage failed:", repr(e))
        ideas_llm = []

    ideas_final = enrich_with_signals(ideas_llm, meta_items, trend_rows, events_rows)

    # Top5 보장
    ideas_final = fill_opportunities_to_five(ideas_final, keywords_obj, want=5)

    save_json("outputs/biz_opportunities.json", {"ideas": ideas_final})
    print("[INFO] Module D done | ideas=%d" % len(ideas_final))

if __name__ == "__main__":
    main()

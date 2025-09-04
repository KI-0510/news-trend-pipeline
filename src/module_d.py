import os
import json
import re
import time
import google.generativeai as genai

# ---------- 공통 유틸 ----------

def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def strip_code_fence(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"^```[\t ]*\w*[\t ]*\n", "", t, flags=re.M)   # 시작 펜스
    t = re.sub(r"\n```[\t ]*$", "", t, flags=re.M)            # 끝 펜스
    return t.strip()

def now_ts():
    import time as _t
    return _t.strftime("%Y%m%dT%H%M%SZ", _t.gmtime())

def dump_debug(text: str, tag="ndjson"):
    try:
        os.makedirs("outputs/debug", exist_ok=True)
        fp = f"outputs/debug/module_d_raw_{tag}_{now_ts()}.txt"
        with open(fp, "w", encoding="utf-8") as f:
            f.write(text or "")
        print(f"[INFO] raw dump: {fp}")
    except Exception:
        pass

def clean_json_text(t: str) -> str:
    # 스마트쿼트/제어문자/트레일링 콤마 정리
    t = strip_code_fence(t or "")
    t = t.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    t = re.sub(r"[\u0000-\u0008\u000B\u000C\u000E-\u001F\u2028\u2029]", "", t)
    t = re.sub(r",\s*(\}|\])", r"\1", t)
    t = t.lstrip("\ufeff")
    return t.strip()

def normalize_list_field(v):
    if v is None:
        return []
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return []
        parts = [p.strip("-•* \t") for p in re.split(r"[\n;,]+|•|-|\*", s) if p.strip()]
        return parts if parts else [s]
    return [str(v)]

def postprocess_ideas(ideas):
    required = ["idea", "problem", "target_customer", "value_prop",
                "solution", "poc_plan", "risks", "roadmap_3m", "metrics", "priority_score"]
    seen, out = set(), []
    for it in ideas:
        if not isinstance(it, dict):
            continue
        for k in required:
            it.setdefault(k, "" if k != "priority_score" else 3.0)
        for k in ["solution", "poc_plan", "risks", "metrics"]:
            it[k] = normalize_list_field(it.get(k))
        try:
            s = float(it.get("priority_score", 3))
        except Exception:
            s = 3.0
        it["priority_score"] = max(1.0, min(5.0, round(s, 1)))
        title = (it.get("idea") or "").strip()
        if not title:
            continue
        key = title.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    out = sorted(out, key=lambda x: x.get("priority_score", 0), reverse=True)
    return out

def compact_context():
    topics = load_json("outputs/topics.json") or {}
    keywords = load_json("outputs/keywords.json") or {}
    ts = load_json("outputs/trend_timeseries.json") or {}
    insights = load_json("outputs/trend_insights.json") or {}

    raw_topics = (topics.get("topics") or [])[:5]
    compact_topics = []
    for t in raw_topics:
        words = (t.get("top_words") or [])[:8]
        compact_topics.append({
            "topic_id": t.get("topic_id"),
            "top_words": [w["word"] for w in words if "word" in w][:8]
        })

    kw_list = (keywords.get("keywords") or [])[:20]
    compact_kws = [k["keyword"] for k in kw_list if "keyword" in k][:20]
    compact_ts = (ts.get("daily") or [])[-30:]

    summary = (insights.get("summary") or "").strip()
    if summary:
        summary = re.sub(r"\s+", " ", summary)[:400]

    return {
        "topics": compact_topics,
        "keywords": compact_kws,
        "timeseries": compact_ts,
        "insight_hint": summary
    }

# ---------- LLM 호출 ----------

def call_gemini(api_key, prompt, max_tokens=1800, temperature=0.15):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    resp = model.generate_content(
        prompt,
        generation_config={
            "max_output_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.8,
            "response_mime_type": "text/plain"
        }
    )
    return (getattr(resp, "text", None) or "").strip()

# ---------- NDJSON 마커/균형 추출 ----------

MARK_START = "<<<NDJSON>>>"
MARK_END = "<<<END>>>"

def extract_objects_from_text(payload: str, max_items=10):
    """
    텍스트에서 중괄호 균형으로 JSON 객체들을 차례로 추출.
    줄 경계 무시, 붙어있는 경우도 파싱.
    """
    s = clean_json_text(payload)
    out = []
    i = 0
    while i < len(s) and len(out) < max_items:
        # 다음 객체 시작 찾기
        start = s.find("{", i)
        if start == -1:
            break
        depth = 0
        end = -1
        j = start
        # 문자열 내부 따옴표/이스케이프는 간단히 무시(실무상 충분)
        while j < len(s):
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

# 기존 parse_ndjson_marked 교체
def parse_ndjson_marked(text: str, need=5):
    """
    마커 구간이 여러 번 등장해도 모두 순회하면서 객체를 수집.
    1) 각 구간 라인 파싱(NDJSON)
    2) 부족하면 해당 구간 전체에서 균형 기반 보강
    3) 구간 순회하며 need개 채우면 종료
    """
    if not text:
        return None

    t = strip_code_fence(text)
    ideas = []

    def _collect_from_payload(payload: str):
        nonlocal ideas
        # 1) 라인 단위
        for line in payload.splitlines():
            if len(ideas) >= need:
                break
            line = line.strip()
            if not line:
                continue
            # 선행 불릿/번호/마커 조각 제거
            line = re.sub(r"^[\-\*\d\.\)\s]+", "", line)
            if line.startswith("<<<") or line.endswith(">>>"):
                continue
            cj = clean_json_text(line)
            try:
                obj = json.loads(cj)
                if isinstance(obj, dict):
                    ideas.append(obj)
                    continue
            except Exception:
                pass
            # 라인 안에 객체가 붙어있는 케이스
            objs = extract_objects_from_text(cj, max_items=2)
            for o in objs:
                if isinstance(o, dict):
                    ideas.append(o)
                    if len(ideas) >= need:
                        break

        # 2) 부족하면 전체 payload에서 균형 추출 보강
        if len(ideas) < need:
            extras = extract_objects_from_text(payload, max_items=10)
            # 중복 제거
            seen = {json.dumps(x, sort_keys=True) for x in ideas}
            for o in extras:
                jkey = json.dumps(o, sort_keys=True)
                if jkey not in seen and isinstance(o, dict):
                    ideas.append(o)
                    seen.add(jkey)
                if len(ideas) >= need:
                    break

    # 마커 구간 반복 추출
    pos = 0
    while len(ideas) < need and True:
        s = t.find(MARK_START, pos)
        if s == -1:
            break
        e = t.find(MARK_END, s + len(MARK_START))
        if e == -1:
            # 마지막 END가 없으면 중단
            break
        payload = t[s + len(MARK_START):e].strip()
        if payload:
            _collect_from_payload(payload)
        pos = e + len(MARK_END)

    return ideas if ideas else None

# ---------- 프롬프트 ----------

def build_prompt_ndjson(ctx, retry=False):
    schema_hint = {
        "idea": "아이디어 한 줄 제목",
        "problem": "해결하려는 문제(2-3문장, 280자 이내)",
        "target_customer": "핵심 타깃(산업/직군/조직 규모 명확히)",
        "value_prop": "핵심 가치제안(차별점, 200자 이내)",
        "solution": ["핵심 기능 bullet 4개 이내"],
        "poc_plan": ["3-6주 PoC 과제 bullet 3-5개"],
        "risks": ["리스크/규제 bullet 3-5개"],
        "roadmap_3m": ["3개월 로드맵 bullet 3-5개"],
        "metrics": ["KPI 3-5개"],
        "priority_score": 3.5
    }
    try:
        cfg = json.load(open("config.json", "r", encoding="utf-8"))
        domain_hints = cfg.get("domain_hints", [])
    except Exception:
        domain_hints = []
    domain_text = (
        f"(가능하면 서로 다른 도메인 반영: {', '.join(domain_hints)})"
        if domain_hints else "(서로 다른 타깃/도메인으로 다양화)"
    )
    strict = (
        "- 출력은 반드시 단 한 번의 마커 구간만 사용한다(아래 마커 1회만).\n"
        "- NDJSON: 각 줄은 반드시 { 로 시작하고 } 로 끝난다. 총 5줄만 출력. 빈 줄/불릿/번호/코드펜스/추가 마커 금지.\n"
        "- 마커 밖에는 어떤 텍스트도 쓰지 마라.\n"
    )
    stricter = "- RETRY MODE: 위 제약 위반 시 실패로 간주. 반드시 준수.\n" if retry else ""

    return (
        "아래 맥락을 바탕으로 한국 시장에 맞춘 디스플레이 및 인접 산업 중심의 신사업 아이디어를 정확히 5개 생성해.\n"
        "- 출력 형식: NDJSON (줄당 JSON 1개, 총 5줄)\n"
        "- 서로 다른 타깃/도메인/채널/BM으로 다양화. " + domain_text + "\n"
        "- 각 필드는 스키마와 길이 제한을 지킬 것(과도한 장문 금지). null/빈 문자열 금지.\n"
        + strict + stricter +
        f"맥락: {json.dumps(ctx, ensure_ascii=False)}\n"
        f"스키마: {json.dumps(schema_hint, ensure_ascii=False)}\n"
        f"{MARK_START}\n"
        "{...}\n"
        "{...}\n"
        "{...}\n"
        "{...}\n"
        "{...}\n"
        f"{MARK_END}"
    )

def build_prompt_one(ctx, used_titles=None, used_targets=None):
    schema_hint = {
        "idea": "아이디어 한 줄 제목",
        "problem": "해결하려는 문제(2-3문장, 280자 이내)",
        "target_customer": "핵심 타깃(산업/직군/조직 규모)",
        "value_prop": "핵심 가치제안(차별점, 200자 이내)",
        "solution": ["핵심 기능 bullet 4개 이내"],
        "poc_plan": ["3-6주 PoC 과제 bullet 3-5개"],
        "risks": ["리스크/규제 bullet 3-5개"],
        "roadmap_3m": ["3개월 로드맵 bullet 3-5개"],
        "metrics": ["KPI 3-5개"],
        "priority_score": 3.5
    }
    used_titles = sorted(list(used_titles or []))[:10]
    used_targets = sorted(list(used_targets or []))[:10]
    guard = {"used_titles": used_titles, "used_targets": used_targets}
    return (
        "아래 맥락을 바탕으로 신사업 아이디어 1개를 JSON 객체 1줄로 출력해.\n"
        f"- 반드시 {MARK_START} 와 {MARK_END} 사이에만 출력하고, 줄은 { '{' } 로 시작해서 { '}' } 로 끝나야 한다.\n"
        "- 불릿/번호/여분 텍스트 금지. 필수 필드 모두 포함. null/빈 문자열 금지.\n"
        f"맥락: {json.dumps(ctx, ensure_ascii=False)}\n"
        f"중복 회피 힌트: {json.dumps(guard, ensure_ascii=False)}\n"
        f"스키마: {json.dumps(schema_hint, ensure_ascii=False)}\n"
        f"{MARK_START}{{ JSON 객체 }}{MARK_END}"
    )

# ---------- 메인 ----------

def main():
    t0 = time.time()
    api_key = (os.getenv("GEMINI_API_KEY", "") or "").strip()
    ctx = compact_context()
    ideas = []

    # config에서 llm.max_output_tokens
    try:
        cfg_all = json.load(open("config.json", "r", encoding="utf-8"))
        max_tokens_cfg = int(cfg_all.get("llm", {}).get("max_output_tokens", 1800))
    except Exception:
        max_tokens_cfg = 1800

    if api_key and len(api_key) > 20:
        # NDJSON 1차
        text = call_gemini(api_key, build_prompt_ndjson(ctx, retry=False), max_tokens=max_tokens_cfg, temperature=0.15)
        dump_debug(text, tag="ndjson_try1")
        arr = parse_ndjson_marked(text, need=5)
        
        # NDJSON 2차(엄격 모드)
        if not (arr and isinstance(arr, list)):
            print("[WARN] NDJSON 파싱 실패 → 2차 엄격 모드")
            text2 = call_gemini(api_key, build_prompt_ndjson(ctx, retry=True), max_tokens=max_tokens_cfg, temperature=0.1)
            dump_debug(text2, tag="ndjson_try2")
            arr = parse_ndjson_marked(text2, need=5)

        # 3) 폴백: 단건 × N으로 부족분 채우기
        need = 5 - len(ideas)
        if need > 0:
            uniq_titles, uniq_targets = set(), set()
            for it in ideas:
                t = (it.get("idea") or "").strip().lower()
                c = (it.get("target_customer") or "").strip().lower()
                if t: uniq_titles.add(t)
                if c: uniq_targets.add(c)

            for _ in range(15):
                if len(ideas) >= 5:
                    break
                one_text = call_gemini(
                    api_key,
                    build_prompt_one(ctx, uniq_titles, uniq_targets),
                    max_tokens=820, temperature=0.15
                )
                dump_debug(one_text, tag="one")
                # 마커 구간 추출
                s = one_text.find(MARK_START); e = one_text.rfind(MARK_END)
                if s == -1 or e == -1 or e <= s:
                    continue
                line = one_text[s + len(MARK_START):e].strip()
                line = clean_json_text(line)
                obj = None
                try:
                    obj = json.loads(line)
                except Exception:
                    # 라인에 객체가 여러 개 붙은 케이스 대비
                    objs = extract_objects_from_text(line, max_items=1)
                    if objs:
                        obj = objs[0]
                if isinstance(obj, dict):
                    title = (obj.get("idea") or "").strip().lower()
                    cust  = (obj.get("target_customer") or "").strip().lower()
                    if title and (title not in uniq_titles) and (cust not in uniq_targets):
                        ideas.append(obj)
                        uniq_titles.add(title)
                        if cust: uniq_targets.add(cust)
    else:
        print("[WARN] GEMINI_API_KEY 비정상 → DRY 아이디어 3개 생성")
        ideas = [
            {"idea": "DRY 샘플 1", "problem": "", "target_customer": "", "value_prop": "",
             "solution": [], "poc_plan": [], "risks": [], "roadmap_3m": [], "metrics": [], "priority_score": 3.0},
            {"idea": "DRY 샘플 2", "problem": "", "target_customer": "", "value_prop": "",
             "solution": [], "poc_plan": [], "risks": [], "roadmap_3m": [], "metrics": [], "priority_score": 3.0},
            {"idea": "DRY 샘플 3", "problem": "", "target_customer": "", "value_prop": "",
             "solution": [], "poc_plan": [], "risks": [], "roadmap_3m": [], "metrics": [], "priority_score": 3.0},
        ]

    # 후처리 및 최종 개수 보정
    ideas = postprocess_ideas(ideas)
    if len(ideas) > 5:
        ideas = ideas[:5]
    if len(ideas) < 3:
        needed = 3 - len(ideas)
        for i in range(needed):
            ideas.append({
                "idea": f"보강 샘플 {i+1}",
                "problem": "아이디어 수가 부족하여 자동 보강.",
                "target_customer": "내부 검토",
                "value_prop": "파이프라인 안정성 보장",
                "solution": ["샘플 항목"],
                "poc_plan": ["샘플 항목"],
                "risks": ["샘플 항목"],
                "roadmap_3m": ["샘플 항목"],
                "metrics": ["샘플 항목"],
                "priority_score": 2.5
            })

    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/biz_opportunities.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"ideas": ideas}, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 모듈 D 완료 | ideas={len(ideas)} | 출력={out_path} | 경과(초)={round(time.time()-t0,2)}")

if __name__ == "__main__":
    main()

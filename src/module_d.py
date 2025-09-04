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

# 스마트쿼트/제어문자/트레일링 콤마 정리
def clean_json_text(t: str) -> str:
    t = strip_code_fence(t)
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

def call_gemini(api_key, prompt, max_tokens=1800, temperature=0.2):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    resp = model.generate_content(
        prompt,
        generation_config={
            "max_output_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
            "response_mime_type": "text/plain"  # 텍스트로 받아 NDJSON 파싱
        }
    )
    return (getattr(resp, "text", None) or "").strip()

# ---------- NDJSON 마커 파싱 ----------

MARK_START = "<<<NDJSON>>>"
MARK_END = "<<<END>>>"

def parse_ndjson_marked(text: str):
    """
    마커 사이 NDJSON을 라인 단위로 파싱.
    성공 시 list[dict] 반환, 실패 시 None.
    """
    if not text:
        return None
    t = strip_code_fence(text)
    s = t.find(MARK_START)
    e = t.rfind(MARK_END)
    if s == -1 or e == -1 or e <= s:
        return None
    payload = t[s + len(MARK_START):e].strip()
    if not payload:
        return None

    ideas = []
    for line in payload.splitlines():
        line = line.strip()
        if not line:
            continue
        # 마커/번호/불릿 같은 잡텍스트 제거 시도
        line = re.sub(r"^[\-\*\d\.\)\s]+", "", line)
        cj = clean_json_text(line)
        try:
            obj = json.loads(cj)
            if isinstance(obj, dict):
                ideas.append(obj)
        except Exception:
            # 라인 내부에 객체가 둘 들어가는 등 비정형 대응: 중괄호 균형으로 첫 객체만 추출
            m = re.search(r"(\{.*\})", cj, re.S)
            if m:
                try:
                    obj = json.loads(clean_json_text(m.group(1)))
                    if isinstance(obj, dict):
                        ideas.append(obj)
                except Exception:
                    pass
    return ideas if ideas else None

# ---------- 프롬프트 ----------

def build_prompt_ndjson(ctx):
    """
    한 번에 5개 객체를 NDJSON(각 줄 JSON 한 개)로 생성.
    마커(MARK_START ~ MARK_END) 사이에만 출력.
    """
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
    return (
        "아래 맥락(토픽, 키워드, 시계열, 요약)을 바탕으로 한국 시장에 맞춘 신사업 아이디어를 정확히 5개 생성해.\n"
        "- 출력 형식은 NDJSON: 각 줄에 JSON 객체 1개(총 5줄)만 출력.\n"
        "- 반드시 마커 사이에만 출력. 마커 밖에는 아무것도 쓰지 마.\n"
        "- 서로 다른 타깃/도메인/채널/BM으로 다양화. " + domain_text + "\n"
        "- 각 필드는 스키마와 길이 제한을 지킬 것(과도한 장문 금지). null/빈 문자열 금지.\n"
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
        "아래 맥락을 바탕으로 신사업 아이디어 1개만 JSON 객체로 출력해.\n"
        f"- 반드시 {MARK_START} 와 {MARK_END} 사이에 순수 JSON만 1줄로 출력.\n"
        "- 이미 생성된 아이디어(제목/타깃)와 겹치지 않게 새로운 타깃/도메인/채널/BM 선택.\n"
        "- 필수 필드 모두 포함. null/빈 문자열 금지.\n"
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
        # 1) NDJSON 한 방에(마커) 시도
        try:
            text = call_gemini(api_key, build_prompt_ndjson(ctx), max_tokens=max_tokens_cfg, temperature=0.2)
            dump_debug(text, tag="ndjson_try1")
            arr = parse_ndjson_marked(text)
            if isinstance(arr, list):
                ideas = arr
            else:
                print("[WARN] NDJSON 파싱 실패 → 2차 시도")
                text2 = call_gemini(api_key, build_prompt_ndjson(ctx), max_tokens=max_tokens_cfg, temperature=0.15)
                dump_debug(text2, tag="ndjson_try2")
                arr = parse_ndjson_marked(text2)
                if isinstance(arr, list):
                    ideas = arr
        except Exception as e:
            print(f"[WARN] NDJSON 생성 실패: {e}")

        # 2) 폴백: 단건 × N으로 채우기
        need = 5 - len(ideas)
        if need > 0:
            uniq_titles, uniq_targets = set(), set()
            for it in ideas:
                t = (it.get("idea") or "").strip().lower()
                c = (it.get("target_customer") or "").strip().lower()
                if t: uniq_titles.add(t)
                if c: uniq_targets.add(c)

            for _ in range(12):
                if len(ideas) >= 5:
                    break
                one_text = call_gemini(
                    api_key,
                    build_prompt_one(ctx, uniq_titles, uniq_targets),
                    max_tokens=820, temperature=0.2
                )
                dump_debug(one_text, tag="one")
                # 마커 구간만 추출 후 파싱(1줄 JSON)
                s = one_text.find(MARK_START); e = one_text.rfind(MARK_END)
                if s == -1 or e == -1 or e <= s:
                    continue
                line = one_text[s + len(MARK_START):e].strip()
                line = clean_json_text(line)
                try:
                    obj = json.loads(line)
                except Exception:
                    # 백업: 중괄호 첫 객체만
                    m = re.search(r"(\{.*\})", line, re.S)
                    if not m:
                        continue
                    try:
                        obj = json.loads(clean_json_text(m.group(1)))
                    except Exception:
                        continue
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

    if len(ideas) < 5 and api_key and len(api_key) > 20:
        # 아주 마지막 보강 라운드
        uniq_titles = { (it.get("idea") or "").strip().lower() for it in ideas }
        uniq_targets = { (it.get("target_customer") or "").strip().lower() for it in ideas }
        for _ in range(8):
            if len(ideas) >= 5:
                break
            one_text = call_gemini(
                api_key,
                build_prompt_one(ctx, uniq_titles, uniq_targets),
                max_tokens=820, temperature=0.2
            )
            dump_debug(one_text, tag="one_fill")
            s = one_text.find(MARK_START); e = one_text.rfind(MARK_END)
            if s == -1 or e == -1 or e <= s:
                continue
            line = clean_json_text(one_text[s + len(MARK_START):e].strip())
            try:
                obj = json.loads(line)
            except Exception:
                m = re.search(r"(\{.*\})", line, re.S)
                if not m:
                    continue
                try:
                    obj = json.loads(clean_json_text(m.group(1)))
                except Exception:
                    continue
            if isinstance(obj, dict):
                title = (obj.get("idea") or "").strip().lower()
                cust  = (obj.get("target_customer") or "").strip().lower()
                if title and (title not in uniq_titles) and (cust not in uniq_targets):
                    ideas.append(obj)
                    uniq_titles.add(title)
                    if cust: uniq_targets.add(cust)

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

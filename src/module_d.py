import os
import json
import re
import time
import random
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
    t = re.sub(r"^```[\t ]*\w*[\t ]*\n", "", t, flags=re.M)
    t = re.sub(r"\n```[\t ]*$", "", t, flags=re.M)
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
    t = strip_code_fence(t or "")
    t = t.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    t = re.sub(r"[\u0000-\u0008\u000B\u000C\u000E-\u001F\u2028\u2029]", "", t)
    t = re.sub(r",\s*(\}|\])", r"\1", t)
    t = t.lstrip("\ufeff")
    return t.strip()

# ---------- 텍스트 유사도(중복 방지) ----------

def _tokenize(s: str):
    toks = re.findall(r"[가-힣A-Za-z0-9]+", (s or "").lower())
    return [t for t in toks if len(t) >= 2]

def jaccard_sim(a_tokens, b_tokens):
    a, b = set(a_tokens), set(b_tokens)
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

def tf_cosine(a_tokens, b_tokens):
    from collections import Counter
    ca, cb = Counter(a_tokens), Counter(b_tokens)
    keys = set(ca) | set(cb)
    if not keys:
        return 0.0
    dot = sum(ca[k] * cb[k] for k in keys)
    na = sum(v * v for v in ca.values()) ** 0.5
    nb = sum(v * v for v in cb.values()) ** 0.5
    return (dot / (na * nb)) if na and nb else 0.0

def too_similar(idea_a: dict, idea_b: dict, jac_th=0.58, cos_th=0.82):
    sa = " ".join([
        idea_a.get("idea",""), idea_a.get("target_customer",""),
        idea_a.get("value_prop",""), " ".join(idea_a.get("solution", []))
    ])
    sb = " ".join([
        idea_b.get("idea",""), idea_b.get("target_customer",""),
        idea_b.get("value_prop",""), " ".join(idea_b.get("solution", []))
    ])
    ta, tb = _tokenize(sa), _tokenize(sb)
    j = jaccard_sim(ta, tb)
    c = tf_cosine(ta, tb)
    return (j >= jac_th) or (c >= cos_th)

def unique_append(pool: list, cand: dict, used_titles: set, used_targets: set) -> bool:
    title = (cand.get("idea") or "").strip().lower()
    targ  = (cand.get("target_customer") or "").strip().lower()
    if not title:
        return False
    if title in used_titles or (targ and targ in used_targets):
        return False
    for p in pool:
        if too_similar(p, cand):
            return False
    pool.append(cand)
    used_titles.add(title)
    if targ:
        used_targets.add(targ)
    return True

# ---------- 데이터 축약 ----------

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

def call_gemini(api_key, prompt, max_tokens=1500, temperature=0.15):
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
    s = clean_json_text(payload)
    out = []
    i = 0
    while i < len(s) and len(out) < max_items:
        start = s.find("{", i)
        if start == -1:
            break
        depth = 0
        end = -1
        j = start
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

def parse_ndjson_marked(text: str, need=5):
    if not text:
        return None
    t = strip_code_fence(text)
    ideas = []

    def _collect(payload: str):
        nonlocal ideas
        # 1) NDJSON 라인 파싱
        for line in payload.splitlines():
            if len(ideas) >= need:
                break
            line = line.strip()
            if not line:
                continue
            line = re.sub(r"^[\-\*\d\.\)\s]+", "", line)
            if line.startswith("<<<") or line.endswith(">>>"):
                continue
            cj = clean_json_text(line)
            try:
                obj = json.loads(cj)
                if isinstance(obj, dict):
                    ideas.append(obj); continue
            except Exception:
                pass
            # 2) 라인 내부 다중 객체 보강
            objs = extract_objects_from_text(cj, max_items=2)
            for o in objs:
                if isinstance(o, dict):
                    ideas.append(o)
                    if len(ideas) >= need:
                        break
        # 3) 전체 구간 균형 추출 보강
        if len(ideas) < need:
            extras = extract_objects_from_text(payload, max_items=10)
            seen = {json.dumps(x, sort_keys=True) for x in ideas}
            for o in extras:
                jkey = json.dumps(o, sort_keys=True)
                if jkey not in seen and isinstance(o, dict):
                    ideas.append(o)
                    seen.add(jkey)
                if len(ideas) >= need:
                    break

    pos = 0
    while len(ideas) < need:
        s = t.find(MARK_START, pos)
        if s == -1:
            break
        e = t.find(MARK_END, s + len(MARK_START))
        if e == -1:
            break
        payload = t[s + len(MARK_START):e].strip()
        if payload:
            _collect(payload)
        pos = e + len(MARK_END)

    return ideas if ideas else None

# ---------- 프롬프트(필드 축소 + 다양성 강화) ----------

def build_schema_hint():
    return {
        "idea": "아이디어 한 줄 제목",
        "problem": "해결하려는 문제(2-3문장, 220자 이내)",
        "target_customer": "핵심 타깃(산업/직군/조직 규모 명확히)",
        "value_prop": "핵심 가치제안(차별점, 180자 이내)",
        "solution": ["핵심 기능 bullet 최대 4개"],
        "risks": ["리스크/규제 bullet 3개 내"],
        "priority_score": 3.5
    }

def domain_text_from_config():
    try:
        cfg = json.load(open("config.json", "r", encoding="utf-8"))
        hints = cfg.get("domain_hints", [])
    except Exception:
        hints = []
    if hints:
        random.shuffle(hints)
        hints = hints[:3]
    focus = ", ".join(hints) if hints else "디스플레이 소재/부품/장비, OLED/MicroLED, AR/VR, 자동차용 디스플레이, 검사/메트롤로지"
    return f"(도메인 포커스: {focus})"

def build_prompt_ndjson(ctx, retry=False):
    schema_hint = build_schema_hint()
    strict = (
        "- 출력은 반드시 단 한 번의 마커 구간만 사용.\n"
        "- NDJSON: 각 줄은 반드시 { 로 시작하고 } 로 끝난다. 총 5줄만 출력. 빈 줄/불릿/번호/코드펜스/추가 마커 금지.\n"
        "- 마커 밖에는 어떤 텍스트도 쓰지 말 것.\n"
        "- 금지 표현: '뉴스 급증', '예측 서비스'(일반 미디어 분석), 'PR팀 대상'만을 위한 제안.\n"
        "- 중복 금지: 제목/타깃/가치제안/핵심기능이 유사한 안을 반복하지 말 것.\n"
    )
    stricter = "- RETRY MODE: 위 제약 위반 시 실패로 간주. 반드시 준수.\n" if retry else ""
    return (
        "아래 맥락을 바탕으로 한국 시장에 맞춘 '디스플레이 및 인접 산업' 중심 신사업 아이디어를 정확히 5개 생성해.\n"
        + domain_text_from_config() + "\n"
        "- 각 아이디어는 서로 다른 타깃·도메인·채널·BM을 반영.\n"
        "- 필드는 아래 스키마만 사용(불필요 필드 금지). 길이 제한 준수. null/빈 문자열 금지.\n"
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
    schema_hint = build_schema_hint()
    used_titles = sorted(list(used_titles or []))[:10]
    used_targets = sorted(list(used_targets or []))[:10]
    guard = {"used_titles": used_titles, "used_targets": used_targets}
    return (
        "아래 맥락을 바탕으로 '디스플레이 및 인접 산업' 중심 신사업 아이디어 1개를 JSON 1줄로 출력해.\n"
        f"- 반드시 {MARK_START} 와 {MARK_END} 사이에만 출력하고, 줄은 {{ 로 시작해서 }} 로 끝나야 한다.\n"
        "- 필드: idea, problem, target_customer, value_prop, solution, risks, priority_score (그 외 금지)\n"
        "- 길이 제한 준수, 중복/유사 금지(제목/타깃/가치/기능).\n"
        "- 금지 표현: '뉴스 급증', 'PR팀 전용', 미디어 예측 일반론.\n"
        f"맥락: {json.dumps(ctx, ensure_ascii=False)}\n"
        f"중복 회피 힌트: {json.dumps(guard, ensure_ascii=False)}\n"
        f"스키마: {json.dumps(schema_hint, ensure_ascii=False)}\n"
        f"{MARK_START}{{ JSON 객체 }}{MARK_END}"
    )

# ---------- 후처리 ----------

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
                "solution", "risks", "priority_score"]
    seen, out = set(), []
    for it in ideas:
        if not isinstance(it, dict):
            continue
        for k in required:
            it.setdefault(k, "" if k != "priority_score" else 3.0)
        # 리스트 표준화
        it["solution"] = normalize_list_field(it.get("solution"))
        it["risks"] = normalize_list_field(it.get("risks"))
        # 점수 정규화
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
    # 우선순위 정렬
    out = sorted(out, key=lambda x: x.get("priority_score", 0), reverse=True)
    return out

# ---------- 메인 ----------

def main():
    t0 = time.time()
    api_key = (os.getenv("GEMINI_API_KEY", "") or "").strip()
    ctx = compact_context()
    ideas = []

    try:
        cfg_all = json.load(open("config.json", "r", encoding="utf-8"))
        max_tokens_cfg = int(cfg_all.get("llm", {}).get("max_output_tokens", 1500))
    except Exception:
        max_tokens_cfg = 1500

    used_titles, used_targets = set(), set()

    if api_key and len(api_key) > 20:
        # 1) NDJSON 1차
        try:
            text = call_gemini(api_key, build_prompt_ndjson(ctx, retry=False),
                               max_tokens=max_tokens_cfg, temperature=0.15)
            dump_debug(text, tag="ndjson_try1")
            arr = parse_ndjson_marked(text, need=5)
            if isinstance(arr, list):
                for obj in arr:
                    unique_append(ideas, obj, used_titles, used_targets)
            # 2) NDJSON 2차(엄격)
            if len(ideas) < 5:
                print("[WARN] NDJSON 파싱 실패/부족 → 2차 엄격 모드")
                text2 = call_gemini(api_key, build_prompt_ndjson(ctx, retry=True),
                                    max_tokens=max_tokens_cfg, temperature=0.1)
                dump_debug(text2, tag="ndjson_try2")
                arr2 = parse_ndjson_marked(text2, need=5)
                if isinstance(arr2, list):
                    for obj in arr2:
                        if len(ideas) >= 5:
                            break
                        unique_append(ideas, obj, used_titles, used_targets)
        except Exception as e:
            print(f"[WARN] NDJSON 생성 실패: {e}")

        # 3) 폴백: 단건 × N으로 부족분 채우기(도메인 다양화 유도)
        if len(ideas) < 5:
            # 다양화 유도: 도메인 힌트 섞어서 여러 번 요청
            for _ in range(18):
                if len(ideas) >= 5:
                    break
                one_text = call_gemini(
                    api_key,
                    build_prompt_one(ctx, used_titles, used_targets),
                    max_tokens=820, temperature=0.15
                )
                dump_debug(one_text, tag="one")
                s = one_text.find(MARK_START); e = one_text.rfind(MARK_END)
                if s == -1 or e == -1 or e <= s:
                    continue
                line = one_text[s + len(MARK_START):e].strip()
                line = clean_json_text(line)
                obj = None
                try:
                    obj = json.loads(line)
                except Exception:
                    objs = extract_objects_from_text(line, max_items=1)
                    if objs:
                        obj = objs[0]
                if isinstance(obj, dict):
                    unique_append(ideas, obj, used_titles, used_targets)
                # 속도/비용 보호: 충분히 채워지면 중단
                if len(ideas) >= 5:
                    break
    else:
        print("[WARN] GEMINI_API_KEY 비정상 → DRY 아이디어 3개 생성")
        ideas = [
            {"idea": "DRY 샘플 1", "problem": "", "target_customer": "", "value_prop": "",
             "solution": [], "risks": [], "priority_score": 3.0},
            {"idea": "DRY 샘플 2", "problem": "", "target_customer": "", "value_prop": "",
             "solution": [], "risks": [], "priority_score": 3.0},
            {"idea": "DRY 샘플 3", "problem": "", "target_customer": "", "value_prop": "",
             "solution": [], "risks": [], "priority_score": 3.0},
        ]

    # 후처리 및 보정
    ideas = postprocess_ideas(ideas)

    # 5개 맞추기(아주 마지막 안전장치)
    if len(ideas) < 5 and api_key and len(api_key) > 20:
        for _ in range(8):
            if len(ideas) >= 5:
                break
            one_text = call_gemini(
                api_key,
                build_prompt_one(ctx, used_titles, used_targets),
                max_tokens=820, temperature=0.12
            )
            dump_debug(one_text, tag="one_fill")
            s = one_text.find(MARK_START); e = one_text.rfind(MARK_END)
            if s == -1 or e == -1 or e <= s:
                continue
            line = clean_json_text(one_text[s + len(MARK_START):e].strip())
            obj = None
            try:
                obj = json.loads(line)
            except Exception:
                objs = extract_objects_from_text(line, max_items=1)
                if objs:
                    obj = objs[0]
            if isinstance(obj, dict):
                unique_append(ideas, obj, used_titles, used_targets)

    # 최저 보장(3개)
    if len(ideas) < 3:
        needed = 3 - len(ideas)
        for i in range(needed):
            ideas.append({
                "idea": f"보강 샘플 {i+1}",
                "problem": "아이디어 수가 부족하여 자동 보강.",
                "target_customer": "내부 검토",
                "value_prop": "파이프라인 안정성 보장",
                "solution": ["샘플 항목"],
                "risks": ["샘플 리스크"],
                "priority_score": 2.5
            })

    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/biz_opportunities.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"ideas": ideas[:5]}, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 모듈 D 완료 | ideas={len(ideas[:5])} | 출력={out_path} | 경과(초)={round(time.time()-t0,2)}")

if __name__ == "__main__":
    main()

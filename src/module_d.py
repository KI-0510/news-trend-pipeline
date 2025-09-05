import os
import json
import re
import csv
import time
import google.generativeai as genai

# ---------- 유틸 ----------

def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def strip_code_fence(text: str) -> str:
    t = (text or "").strip()
    # 코드펜스/앞뒤 잡텍스트 제거
    t = re.sub(r"^```[\t ]*\w*[\t ]*\n", "", t, flags=re.M)
    t = re.sub(r"\n```[\t ]*$", "", t, flags=re.M)
    return t.strip()

def clean_json_text(t: str) -> str:
    # 스마트쿼트/제어문자/트레일링 콤마 정리
    t = strip_code_fence(t or "")
    t = t.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    t = re.sub(r"[\u0000-\u0008\u000B\u000C\u000E-\u001F\u2028\u2029]", "", t)
    t = re.sub(r",\s*(\}|\])", r"\1", t)
    t = t.lstrip("\ufeff")
    return t.strip()

def now_ts():
    import time as _t
    return _t.strftime("%Y%m%dT%H%M%SZ", _t.gmtime())

def dump_debug(text: str, tag="array"):
    try:
        os.makedirs("outputs/debug", exist_ok=True)
        fp = f"outputs/debug/module_d_raw_{tag}_{now_ts()}.txt"
        with open(fp, "w", encoding="utf-8") as f:
            f.write(text or "")
        print(f"[INFO] raw dump: {fp}")
    except Exception:
        pass

# ---------- 파서(이전 방식 복원 + 보강) ----------

def parse_json_array_or_object(t: str):
    """전체를 JSON으로 읽어 리스트 or {'ideas':[...]} 지원"""
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
    """첫 번째 [ ... ] 배열을 괄호 균형으로 추출 후 json.loads"""
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
    """텍스트 전역에서 { ... } 객체를 균형으로 연속 추출 → list[dict]"""
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
    """라인 단위로 JSON 객체를 읽는 NDJSON 보조 파서(마커 없이도 시도)"""
    ideas = []
    for line in (t or "").splitlines():
        if len(ideas) >= max_items:
            break
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^[\-\*\d\.\)\s]+", "", line)  # 불릿/번호 제거
        try:
            obj = json.loads(clean_json_text(line))
            if isinstance(obj, dict):
                ideas.append(obj)
                continue
        except Exception:
            # 라인 안에 객체 2개 이상 붙은 케이스
            objs = extract_objects_sequence(line, max_items=2)
            for o in objs:
                if isinstance(o, dict):
                    ideas.append(o)
                    if len(ideas) >= max_items:
                        break
    return ideas

def extract_ideas_any(text: str, want=5):
    """
    이전 파서 방식 복원:
    1) 전체 JSON 파싱(list or {'ideas':[...]} )
    2) 첫 배열 균형 추출
    3) 전역 객체 연속 추출
    4) NDJSON 라인 파싱
    """
    # 1
    arr = parse_json_array_or_object(text)
    if isinstance(arr, list) and arr:
        return arr
    # 2
    arr2 = extract_balanced_array(text)
    if isinstance(arr2, list) and arr2:
        return arr2
    # 3
    objs = extract_objects_sequence(text, max_items=want)
    if objs:
        return objs
    # 4
    nd = extract_ndjson_lines(text, max_items=want)
    if nd:
        return nd
    return None

# ---------- 스키마/프롬프트(필드 축소 버전 유지) ----------

def build_schema_hint():
    # PoC, 3개월 로드맵, KPI 제거(길이 절약)
    return {
        "idea": "아이디어 한 줄 제목",
        "problem": "해결하려는 문제(2-3문장, 220자 이내)",
        "target_customer": "핵심 타깃(산업/직군/조직 규모 명확히)",
        "value_prop": "핵심 가치제안(차별점, 180자 이내)",
        "solution": ["핵심 기능 bullet 최대 4개"],
        "risks": ["리스크/규제 bullet 3개 내"],
        "priority_score": 3.5
    }

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

def build_prompt_array(ctx):
    schema_hint = build_schema_hint()
    return (
        "아래 맥락(토픽/키워드/시계열/요약)을 바탕으로 '디스플레이 및 인접 산업' 중심 신사업 아이디어를 정확히 5개 생성해.\n"
        "- 반드시 순수 JSON만 반환. 코드펜스/설명/서문/후문 금지.\n"
        "- 출력 형식: JSON 배열([])로, 각 원소는 JSON 객체.\n"
        "- 필드 제한: idea, problem, target_customer, value_prop, solution, risks, priority_score (그 외 금지)\n"
        "- 각 아이디어는 서로 다른 타깃·도메인·채널·BM을 반영. 중복/유사 금지.\n"
        "- 길이 제한 준수. null/빈 문자열 금지.\n"
        f"맥락: {json.dumps(ctx, ensure_ascii=False)}\n"
        f"스키마: {json.dumps(schema_hint, ensure_ascii=False)}\n"
        "출력: JSON 배열([])만"
    )

# ---------- LLM 호출 ----------

def call_gemini_array(api_key, prompt, max_tokens=1400, temperature=0.15):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    # JSON MIME 지정(응답을 배열로 유도)
    resp = model.generate_content(
        prompt,
        generation_config={
            "max_output_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.8,
            "response_mime_type": "application/json"
        }
    )
    return (getattr(resp, "text", None) or "").strip()

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
    # 축소 스키마에 맞게 필수/타입 보정
    required = ["idea", "problem", "target_customer", "value_prop", "solution", "risks", "priority_score"]
    out, seen = [], set()
    for it in ideas or []:
        if not isinstance(it, dict):
            continue
        # 허용 필드만 남기고 기본값 채우기
        clean = {k: it.get(k) for k in required}
        for k in required:
            if k not in clean or clean[k] is None:
                clean[k] = [] if k in ("solution", "risks") else (3.0 if k == "priority_score" else "")
        clean["solution"] = normalize_list_field(clean.get("solution"))
        clean["risks"] = normalize_list_field(clean.get("risks"))
        # 점수 정규화
        try:
            s = float(clean.get("priority_score", 3))
        except Exception:
            s = 3.0
        clean["priority_score"] = max(1.0, min(5.0, round(s, 1)))
        # 중복 타이틀 제거(간단)
        title = (clean.get("idea") or "").strip()
        if not title:
            continue
        key = title.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(clean)
    # 우선순위 점수로 정렬
    out = sorted(out, key=lambda x: x.get("priority_score", 0), reverse=True)
    return out

# ---------- 메인 ----------

def main():
    t0 = time.time()
    api_key = (os.getenv("GEMINI_API_KEY", "") or "").strip()
    ctx = compact_context()
    ideas = []

    # 토큰 제한: 필드 축소했으니 1400 정도면 안전
    try:
        cfg_all = json.load(open("config.json", "r", encoding="utf-8"))
        max_tokens_cfg = int(cfg_all.get("llm", {}).get("max_output_tokens", 1400))
    except Exception:
        max_tokens_cfg = 1400

    if api_key and len(api_key) > 20:
        # 1차: 배열 모드(예전 파싱 전제)
        try:
            text1 = call_gemini_array(api_key, build_prompt_array(ctx), max_tokens=max_tokens_cfg, temperature=0.15)
            dump_debug(text1, tag="array_try1")
            arr1 = extract_ideas_any(text1, want=5)
            if isinstance(arr1, list) and arr1:
                ideas = arr1
        except Exception as e:
            print(f"[WARN] 배열 호출 실패(1차): {e}")

        # 2차: 그래도 파싱 실패 시 동일 프롬프트로 재시도(단, 폴백 생성 없음)
        if not ideas:
            try:
                text2 = call_gemini_array(api_key, build_prompt_array(ctx), max_tokens=max_tokens_cfg, temperature=0.12)
                dump_debug(text2, tag="array_try2")
                arr2 = extract_ideas_any(text2, want=5)
                if isinstance(arr2, list) and arr2:
                    ideas = arr2
            except Exception as e:
                print(f"[WARN] 배열 호출 실패(2차): {e}")
    else:
        print("[WARN] GEMINI_API_KEY 비정상 → 생성 생략(빈 결과 허용)")

    # 후처리(필드 정리/정렬). 폴백 생성은 하지 않음(요청사항)
    ideas = postprocess_ideas(ideas)

    # 개수 제한: 최대 5개, 모자라면 그대로 둠(빈칸 허용)
    if len(ideas) > 5:
        ideas = ideas[:5]

    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/biz_opportunities.json"
    with open("outputs/opportunities.csv", "w", encoding="utf-8", newline="") as cf:
        w = csv.writer(cf)
        w.writerow(["idea", "target_customer", "value_prop", "priority_score"])
        for it in ideas:
            w.writerow([
                it.get("idea", ""),
                it.get("target_customer", ""), 
                (it.get("value_prop", "") or "").replace("\n", " "),
                it.get("priority_score", "")  
            ]) 
    print(f"[INFO] CSV 저장: outputs/opportunities.csv")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"ideas": ideas}, f, ensure_ascii=False, indent=2)

    print(f"[INFO] 모듈 D 완료 | ideas={len(ideas)} | 출력={out_path} | 경과(초)={round(time.time()-t0,2)}")
    print(f"[INFO] SUMMARY | D | ideas={len(ideas)} titles={[ (it.get('idea') or '')[:12] for it in ideas[:3] ]}")

if __name__ == "__main__":
    main()

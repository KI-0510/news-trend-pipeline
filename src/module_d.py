# -*- coding: utf-8 -*-
import os
import json
import re
import time
from typing import List, Dict, Any, Optional

import google.generativeai as genai

# D1: 설정 일원화 적용
from config import load_config, llm_config
CFG = load_config()
LLM = llm_config(CFG)

# ---------- 유틸 ----------

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

def build_schema_hint() -> Dict[str, Any]:
    # 길이 최소화한 축소 스키마(롤백 기준)
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
        "다음 컨텍스트(키워드/토픽/요약)를 기반으로 신사업 아이디어를 제안해 주세요.\n"
        f"- 아이디어 개수: 최대 {want}개\n"
        "- JSON 배열 형식만 출력하세요. 배열 이외의 텍스트를 출력하지 마세요.\n"
        "- 각 아이템은 아래 스키마 키를 정확히 사용하세요.\n"
        f"스키마: {json.dumps(schema, ensure_ascii=False)}\n"
        "- solution, risks는 리스트로 주세요.\n"
        "- priority_score는 0.0~5.0의 숫자(float)로 주세요.\n"
        "컨텍스트:\n"
        f"{json.dumps(context, ensure_ascii=False)}"
    )

# ---------- LLM 호출 ----------

def call_gemini(prompt: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY 환경 변수가 없습니다.")
    genai.configure(api_key=api_key)
    model_name = str(LLM.get("model", "gemini-1.5-flash"))
    max_tokens = int(LLM.get("max_output_tokens", 2048))
    temperature = float(LLM.get("temperature", 0.3))

    model = genai.GenerativeModel(model_name)
    resp = model.generate_content(
        prompt,
        generation_config={
            "max_output_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
        }
    )
    text = (getattr(resp, "text", None) or "").strip()
    return text

# ---------- 아이디어 정규화/검증 ----------

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
    # 다양한 키 관대하게 흡수하되, 축소 스키마로 정규화
    idea = it.get("idea") or it.get("title") or it.get("name") or ""
    problem = it.get("problem") or it.get("pain") or ""
    target = it.get("target_customer") or it.get("target") or it.get("audience") or ""
    value = it.get("value_prop") or it.get("value") or it.get("description") or ""
    solution = it.get("solution") or it.get("solutions") or []
    risks = it.get("risks") or it.get("risk") or []

    score = it.get("priority_score", it.get("score", 0))
    score = max(0.0, min(5.0, to_float(score, 0.0)))

    # 길이 제한(샘플 톤)
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
        "priority_score": round(score, 1)
    }
    # 호환성 위해 title/score도 같이 둠(리포트 테이블 생성 시 유용)
    out["title"] = idea
    out["score"] = out["priority_score"]
    return out

def postprocess_ideas(raw_list: List[Dict[str, Any]], want=5) -> List[Dict[str, Any]]:
    items = []
    for it in (raw_list or []):
        try:
            norm = normalize_item(it)
            # 최소 요건: 아이디어 제목, 가치제안은 있어야 유의미
            if norm["idea"] and norm["value_prop"]:
                items.append(norm)
        except Exception:
            continue

    # 상위 5개만 (점수 내림차순)
    items.sort(key=lambda x: x.get("priority_score", 0.0), reverse=True)
    return items[:want]

# ---------- 컨텍스트 로딩 ----------

def load_context() -> Dict[str, Any]:
    keywords = load_json("outputs/keywords.json", default={"keywords": []}) or {"keywords": []}
    topics = load_json("outputs/topics.json", default={"topics": []}) or {"topics": []}
    insights = load_json("outputs/trend_insights.json", default={"summary": "", "top_topics": [], "evidence": {}}) or {"summary": "", "top_topics": [], "evidence": {}}

    # 요약 길이 제한(프롬프트 과도 길이 방지)
    summary = (insights.get("summary") or "").strip()
    if len(summary) > 1200:
        summary = summary[:1200] + "…"

    # 키워드 상위 20개만
    kw_simple = [{"keyword": k.get("keyword", ""), "score": k.get("score", 0)} for k in (keywords.get("keywords") or [])[:20]]

    # 토픽은 단어만 간추림
    tp_simple = []
    for t in (topics.get("topics") or [])[:6]:
        words = [w.get("word", "") for w in (t.get("top_words") or [])][:6]
        tp_simple.append({"topic_id": t.get("topic_id"), "words": words})

    return {
        "summary": summary,
        "keywords": kw_simple,
        "topics": tp_simple
    }

# ---------- 메인 ----------

def main():
    t0 = time.time()
    want = 5

    context = load_context()
    prompt = build_prompt(context, want=want)

    try:
        raw_text = call_gemini(prompt)
    except Exception as e:
        # LLM 호출 실패 시에도 빈 구조로 저장하여 파이프라인 계속
        print(f"[ERROR] Gemini 호출 실패: {e}")
        save_json("outputs/biz_opportunities.json", {"ideas": []})
        print("[INFO] SUMMARY | D | ideas=0 elapsed=%.2fs" % (time.time() - t0))
        return

    dump_debug(raw_text, tag="gen")

    ideas = extract_ideas_any(raw_text, want=want) or []
    if not isinstance(ideas, list):
        ideas = []

    # 후처리/정규화 + 상위 5개
    ideas_final = postprocess_ideas(ideas, want=want)

    save_json("outputs/biz_opportunities.json", {"ideas": ideas_final})
    print("[INFO] SUMMARY | D | ideas=%d elapsed=%.2fs" % (len(ideas_final), time.time() - t0))

if __name__ == "__main__":
    main()

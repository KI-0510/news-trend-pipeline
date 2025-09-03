import os
import json
import re
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
    # json ... 같은 코드 펜스 제거
    return re.sub(r"^```\s*\w*\s*\n|```\s*\n?", "", text.strip(), flags=re.M)


def extract_json_array(text: str):
    # 텍스트에서 최초의 [ ... ] 배열만 뽑아 파싱 시도
    t = strip_code_fence(text)
    m = re.search(r"($.*$)", t, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None


def normalize_list_field(v):
    # 문자열이면 한 줄 리스트로, None이면 빈 리스트로
    if v is None:
        return []
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return []
        # 문장부호/줄바꿈 기준으로 분할 시도
        parts = [p.strip("-• \t") for p in re.split(r"[\n;]|•|-|•", s) if p.strip()]
        return parts if parts else [s]
    # 그 외 타입은 문자열 변환 후 리스트로
    return [str(v)]


def postprocess_ideas(ideas):
    # 필수 필드 보정 + 중복 제거 + 점수 정리
    required = ["idea", "problem", "target_customer", "value_prop", "solution", "poc_plan", "risks", "roadmap_3m", "metrics", "priority_score"]
    seen, out = set(), []

    for it in ideas:
        if not isinstance(it, dict):
            continue
        # 빠진 키 기본값 채우기
        for k in required:
            it.setdefault(k, "" if k != "priority_score" else 3.0)
        # 리스트여야 할 필드 정규화
        for k in ["solution", "poc_plan", "risks", "metrics"]:
            it[k] = normalize_list_field(it.get(k))
        # 점수 보정
        try:
            s = float(it.get("priority_score", 3))
        except Exception:
            s = 3.0
        it["priority_score"] = max(1.0, min(5.0, round(s, 1)))
        title = (it.get("idea") or "").strip()
        key = title.lower()
        if not title or key in seen:
            continue
        seen.add(key)
        out.append(it)
    # 우선순위 점수 내림차순
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


# ---------- Gemini 호출 ----------

def call_gemini_array(api_key, prompt, max_tokens=2048, temperature=0.5):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    resp = model.generate_content(
        prompt,
        generation_config={
            "max_output_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
            "response_mime_type": "application/json"  # JSON 강제
        }
    )
    return (getattr(resp, "text", None) or "").strip()


def call_gemini_one(api_key, prompt_one, max_tokens=768, temperature=0.5):
    # 아이디어 1개씩 생성(폴백용)
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    resp = model.generate_content(
        prompt_one,
        generation_config={
            "max_output_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
            "response_mime_type": "application/json"
        }
    )
    return (getattr(resp, "text", None) or "").strip()


def build_prompt_array(ctx):
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

    # 도메인 힌트가 있을 경우 다양화 가이드
    try:
        cfg = json.load(open("config.json", "r", encoding="utf-8"))
        domain_hints = cfg.get("domain_hints", [])
    except Exception:
        domain_hints =[]

    domain_text = (
        f"(가능하면 서로 다른 도메인 반영: {', '.join(domain_hints)})"
        if domain_hints
        else "(서로 다른 타깃/도메인으로 다양화)"
    )

    return (
        "아래 맥락(토픽, 키워드, 시계열, 요약)을 바탕으로 한국 시장에 맞춘 신사업 아이디어를 정확히 5개 생성해.\n"
        "- 반드시 JSON 배열([])만 반환. 코드펜스/설명 금지.\n"
        "- 서로 다른 타깃/도메인/채널/BM으로 다양화. " + domain_text + "\n"
        "- 각 필드는 스키마와 길이 제한을 지킬 것(과도한 장문 금지).\n"
        f"맥락: {json.dumps(ctx, ensure_ascii=False)}\n"
        f"스키마: {json.dumps(schema_hint, ensure_ascii=False)}\n"
        "출력: 아이디어 5개 객체로 이뤄진 JSON 배열([])"
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

    used_titles = sorted(list(used_titles or[]))[:10]
    used_targets = sorted(list(used_targets or[]))[:10]

    guard = {
        "used_titles": used_titles,
        "used_targets": used_targets
    }

    return (
        "아래 맥락을 바탕으로 신사업 아이디어 1개만 JSON 객체로 반환해. 코드펜스/설명 금지, JSON만.\n"
        "- 이미 생성된 아이디어(제목/타깃)와 겹치지 않게 새로운 타깃/도메인/채널/BM 선택.\n"
        f"맥락: {json.dumps(ctx, ensure_ascii=False)}\n"
        f"중복 회피 힌트: {json.dumps(guard, ensure_ascii=False)}\n"
        f"스키마: {json.dumps(schema_hint, ensure_ascii=False)}\n"
        "출력: JSON 객체"
    )


# ---------- 메인 ----------

def main():
    t0 = time.time()
    api_key = (os.getenv("GEMINI_API_KEY", "") or "").strip()
    ctx = compact_context()
    ideas = []

    if api_key and len(api_key) > 20:
        # 1) 배열 한 번에 생성 시도
        try:
            text = call_gemini_array(api_key, build_prompt_array(ctx), max_tokens=1536, temperature=0.5)
            arr = extract_json_array(text)
            if arr and isinstance(arr, list):
                ideas = arr
            else:
                print("[WARN] 배열 JSON 파싱 실패 → 폴백 모드(단건 x 5)")
        except Exception as e:
            print(f"[WARN] 배열 생성 실패: {e} → 폴백 모드로 진행")

        # 2) 폴백: 단건 아이디어 5회 생성(중복 제거)
        if not ideas or len(ideas) < 3:
            uniq_titles, uniq_targets = set(), set()
        
            # 기존 아이디어에서 선점된 타이틀/타깃 수집
            for it in ideas:
                t = (it.get("idea") or "").strip().lower()
                c = (it.get("target_customer") or "").strip().lower()
                if t:
                    uniq_titles.add(t)
                if c:
                    uniq_targets.add(c)
        
            for _ in range(10):
                one_text = call_gemini_one(
                    api_key,
                    build_prompt_one(ctx, uniq_titles, uniq_targets),
                    max_tokens=820,
                    temperature=0.5
                )
                cleaned = strip_code_fence(one_text)
                m = re.search(r"(\{.*\})", cleaned, re.S)
                if not m:
                    continue
                obj = json.loads(m.group(1))
                if isinstance(obj, dict):
                    title = (obj.get("idea") or "").strip().lower()
                    cust = (obj.get("target_customer") or "").strip().lower()
                    if title and title not in uniq_titles and cust not in uniq_targets:
                        ideas.append(obj)
                        uniq_titles.add(title)
                        if cust:
                            uniq_targets.add(cust)
                if len(ideas) >= 5:
                    break
                except Exception:
                    continue
    else:
        print("[WARN] GEMINI_API_KEY 비정상 → DRY 아이디어 3개 생성")
        ideas = [
            {"idea": "DRY 샘플 1", "problem": "", "target_customer": "", "value_prop": "", "solution": [], "poc_plan": [], "risks": [], "roadmap_3m": [], "metrics": [], "priority_score": 3.0},
            {"idea": "DRY 샘플 2", "problem": "", "target_customer": "", "value_prop": "", "solution": [], "poc_plan": [], "risks": [], "roadmap_3m": [], "metrics": [], "priority_score": 3.0},
            {"idea": "DRY 샘플 3", "problem": "", "target_customer": "", "value_prop": "", "solution": [], "poc_plan": [], "risks": [], "roadmap_3m": [], "metrics": [], "priority_score": 3.0},
        ]

    ideas = postprocess_ideas(ideas)

    if len(ideas) < 3:
        # 마지막 안전장치: 최소 3개 보장
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

import os
import json
import re
import time

import google.generativeai as genai


def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def compact_context():
    topics = load_json("outputs/topics.json") or{}
    keywords = load_json("outputs/keywords.json") or{}
    ts = load_json("outputs/trend_timeseries.json") or{}
    insights = load_json("outputs/trend_insights.json") or{}

    # 토픽은 상위 5개, 토픽별 상위 단어 6-8개만 압축
    raw_topics = (topics.get("topics") or [])[:5]
    compact_topics =[]
    for t in raw_topics:
        words = (t.get("top_words") or [])[:8]
        compact_topics.append({
            "topic_id": t.get("topic_id"),
            "top_words": [w["word"] for w in words]
        })

    # 키워드는 상위 20개만(있으면)
    kw_list = (keywords.get("keywords") or [])[:20]
    compact_kws = [k["keyword"] for k in kw_list if "keyword" in k][:20]

    # 시계열 최근 30일 근사
    compact_ts = (ts.get("daily") or [])[-30:]

    # 인사이트 요약(있으면 1-2문장만)
    summary = (insights.get("summary") or "").strip()
    if summary:
        summary = re.sub(r"\s+", " ", summary)
        summary = summary[:400]

    return {
        "topics": compact_topics,
        "keywords": compact_kws,
        "timeseries": compact_ts,
        "insight_hint": summary
    }


def build_prompt(context):
    # JSON 구조 안내 + 평가 기준까지 포함(LLM이 구조 맞추도록)
    schema_hint = {
        "idea": "아이디어 한 줄 제목",
        "problem": "해결하려는 문제(2-3문장)",
        "target_customer": "핵심 타깃(특정 산업/조직/직군/지역 등)",
        "value_prop": "핵심 가치제안(경쟁 대비 차별점 포함)",
        "solution": "솔루션 개요(기능/흐름 중심, 4-6 bullet)",
        "poc_plan": "3-6주 PoC 계획(데이터/리소스/리스크 포함)",
        "risks": "주요 리스크/규제(보안/개인정보/저작권/비용 등, bullet)",
        "roadmap_3m": "3개월 로드맵(마일스톤 3-4개)",
        "metrics": "핵심 지표(KPI 3-5개)",
        "priority_score": "우선순위 점수(1-5): 시장성, 실현가능성, 수익화 잠재력 종합"
    }

    return (
        "다음 맥락(토픽, 키워드, 간단 시계열, 요약 힌트)을 바탕으로 한국 시장에 맞춘 신사업 아이디어를 5개 제안해줘.\n"
        "- 반드시 아래 JSON 배열 구조를 지켜서만 출력해줘. 설명 문장 없이 JSON만.\n"
        "- 각 아이디어는 실무자가 바로 검토 가능한 수준으로 구체화.\n"
        "- 공공/금융/제조/교육 등 맥락에 맞는 타깃을 명확히.\n"
        "- 개인정보/보안/규제 리스크를 현실적으로 포함.\n"
        f"맥락: {json.dumps(context, ensure_ascii=False)}\n\n"
        f"JSON 스키마 예시(키/형태만): {json.dumps(schema_hint, ensure_ascii=False)}\n"
        "출력: 위 스키마의 객체 5개로 이뤄진 JSON 배열만 반환"
    )


def call_gemini(api_key, prompt, max_tokens=2048, temperature=0.6):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    resp = model.generate_content(
        prompt,
        generation_config={
            "max_output_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9
        }
    )
    text = (getattr(resp, "text", None) or "").strip()
    return text


def safe_parse_json(text):
    # 응답이 JSON만 오도록 프롬프트했지만, 혹시 주변 텍스트가 섞일 수 있어 방어
    # 첫 [ ... ] 만 추출 시도
    m = re.search(r"(\[.*\])", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None


def postprocess_ideas(ideas):
    # 중복 아이디어(제목/핵심 키워드 겹침) 간단 제거 + 점수 범위 보정
    seen = set()
    out =[]
    for it in ideas:
        title = (it.get("idea") or "").strip()
        key = title.lower()
        if not title or key in seen:
            continue

        # 점수 보정
        try:
            s = float(it.get("priority_score", 3))
            it["priority_score"] = max(1.0, min(5.0, round(s, 1)))
        except Exception:
            it["priority_score"] = 3.0

        seen.add(key)
        out.append(it)

    # 우선순위 점수로 정렬(내림차순)
    out = sorted(out, key=lambda x: x.get("priority_score", 0), reverse=True)
    return out


def main():
    t0 = time.time()
    api_key = os.getenv("GEMINI_API_KEY", "")

    if not api_key or len(api_key.strip()) < 20:
        print("[WARN] GEMINI_API_KEY 미설정/비정상 — 더미 결과 생성")
        context = compact_context()
        dummy = [{
            "idea": "DRY RUN: 키 없음으로 샘플 아이디어",
            "problem": "데이터 부족",
            "target_customer": "내부 테스트",
            "value_prop": "파이프라인 검증용",
            "solution": "샘플",
            "poc_plan": "샘플",
            "risks": "샘플",
            "roadmap_3m": "샘플",
            "metrics": "샘플",
            "priority_score": 3.0
        }]
        ideas = dummy
    else:
        context = compact_context()
        prompt = build_prompt(context)
        text = call_gemini(api_key, prompt, max_tokens=2048, temperature=0.6)
        ideas = safe_parse_json(text) or[]

        if not ideas:
            print("[WARN] 1차 JSON 파싱 실패. 응답 일부를 로그로 남깁니다.")
            print(text[:600])
            # 구조 틀어졌을 때 후속 요청(간단 정정 프롬프트)
            fix_prompt = (
                "다음 텍스트에서 JSON 배열([ ... ])만 정확한 JSON으로 정리해서 반환해줘. 설명 없이 JSON만.\n"
                f"{text[:4000]}"
            )
            text2 = call_gemini(api_key, fix_prompt, max_tokens=2048, temperature=0.0)
            ideas = safe_parse_json(text2) or[]

    ideas = postprocess_ideas(ideas)

    if not ideas:
        ideas = [{"idea": "생성 실패(검수 필요)", "priority_score": 1.0}]

    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/biz_opportunities.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"ideas": ideas}, f, ensure_ascii=False, indent=2)

    print(f"[INFO] 모듈 D 완료 | ideas={len(ideas)} | 출력={out_path} | 경과(초)={round(time.time() - t0, 2)}")


if __name__ == "__main__":
    main()

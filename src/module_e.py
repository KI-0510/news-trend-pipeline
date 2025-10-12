# -*- coding: utf-8 -*-
import os
import re
import json
import csv
import glob
import unicodedata
import datetime
import time
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict, Counter
from src.config import load_config, llm_config
from src.utils import load_json, save_json, latest
from src.timeutil import to_date, kst_date_str, kst_run_suffix



# ========== 공통 로드/유틸 ==========
def to_date(s: str) -> str:
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


# ========== 스코어링/증거 ==========
def clamp01(x): 
    return max(0.0, min(1.0, float(x)))

def normalize_score(x, lo, hi):
    if hi <= lo:
        return 0.0
    return clamp01((x - lo) / (hi - lo))

NEG_WORDS = [
    "논란", "우려", "리스크", "규제", "지연", "지적", "하락", "부진", "적자", "연기",
    "보안", "개인정보", "불확실성", "기술 미성숙", "높은 비용", "복잡한 인증", "법적 리스크",
    "시장 불안정", "공급망 리스크", "정책 불확실성", "해킹", "위협", "취약점", "중단", "취소",
    "감소", "침체", "악화", "경고", "문제", "비판", "벌금", "소송", "분쟁", "해고", "감원",
    "축소", "퇴출", "손실", "손해", "파산", "부도", "불매", "보이콧", "사기", "횡령", "배임",
    "부패", "비리", "조작", "위조", "사건", "사고"
]


def extract_keywords_from_idea(idea_text: str, keywords_obj: dict) -> List[str]:
    """아이디어 문장에서 핵심 키워드를 추출합니다."""
    top_keywords = {k.get("keyword", "") for k in keywords_obj.get("keywords", [])}
    found_keywords = []
    for kw in top_keywords:
        if kw and kw.lower() in idea_text.lower():
            found_keywords.append(kw)
    
    nouns = re.findall(r"[가-힣A-Za-z]{2,}", idea_text)
    if not found_keywords and nouns:
        return nouns[:3]
    return found_keywords

def pick_evidence(idea_keywords: List[str], items: List[Dict[str,Any]], limit=3) -> List[Dict[str,Any]]:
    """ 문장의 점수를 매겨 가장 설득력 있는 근거를 선택하는 함수 """
    if not idea_keywords:
        return []
    
    candidate_sentences = []
    
    # 사업의 필요성과 연관된 가중치 키워드
    prob_keywords = ["문제", "어려움", "한계", "비용", "수율", "부족"]
    opp_keywords = ["기회", "요구", "필요", "성장", "가능성", "기대"]

    for it in items:
        base = (it.get("raw_body") or it.get("body") or it.get("description") or "")
        if not base:
            continue
        
        low_base = base.lower()
        # 아이디어 키워드 중 하나라도 본문에 포함된 경우에만 문장 분석 시작
        if any(kw.lower() in low_base for kw in idea_keywords):
            sents = re.split(r"(?<=[.!?다])\s+", base)
            for s in sents:
                s_strip = s.strip()
                if not s_strip:
                    continue

                # 문장 점수 계산
                score = 0
                # 1. 아이디어 핵심 키워드 포함 개수
                matched_keywords = sum(1 for kw in idea_keywords if kw.lower() in s_strip.lower())
                if matched_keywords == 0:
                    continue
                score += matched_keywords * 2

                # 2. 문제점/기회 키워드 포함 시 보너스 점수
                if any(pw in s_strip for pw in prob_keywords):
                    score += 5
                if any(ow in s_strip for ow in opp_keywords):
                    score += 3
                
                candidate_sentences.append({
                    "sentence": s_strip[:400],
                    "url": it.get("url") or "",
                    "date": to_date(it.get("published_time") or it.get("pubDate_raw") or ""),
                    "score": score
                })

    # 점수가 높은 순으로 정렬하여 상위 문장 선택
    sorted_candidates = sorted(candidate_sentences, key=lambda x: x['score'], reverse=True)
    
    # 중복 제거 및 최종 결과 반환
    final_evidence = []
    seen_sentences = set()
    for cand in sorted_candidates:
        if len(final_evidence) >= limit:
            break
        if cand['sentence'] not in seen_sentences:
            final_evidence.append(cand)
            seen_sentences.add(cand['sentence'])
            
    return final_evidence

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
    
    maturity_prompt_injection = ""
    if context.get("tech_maturity"):
        maturity_prompt_injection = (
            f"  5) 아래 '기술 성숙도' 정보를 반드시 참고하여, 각 아이디어가 어떤 기술 단계(Emerging, Growth, Maturity)에 있는지 전략적 타이밍 관점에서 언급하세요.\n"
            f"     (예: 이 아이디어는 아직 Emerging 단계인 OOO 기술에 선제적으로 진입하는 것입니다.)\n"
        )

    return (
        f"당신은 글로벌 1위 디스플레이 패널 제조 기업(B2B)의 '신사업 개발 총괄'입니다. "
        f"아래 컨텍스트를 기반으로 구체적인 신사업 아이디어를 제안해 주세요.\n"
        f"- 아이디어 개수: 정확히 {want}개\n"
        f"- JSON 배열 형식만 출력하세요. 설명은 필요 없습니다.\n"
        f"- 각 아이템은 아래 스키마 키를 정확히 사용하세요: {json.dumps(schema, ensure_ascii=False)}\n"
        f"- 제약 조건:\n"
        # --- ✨✨✨ 바로 이 부분이 수정되었습니다 ✨✨✨ ---
        f"  1) 아이디어는 **구체적인 '제품', '기술', '서비스', 또는 '공정 개선 방안'**의 형태여야 합니다. '플랫폼'이나 '솔루션' 같은 추상적인 개념 대신, **실체가 명확하고 현실적인 사업 아이템**을 제안해주세요. (예: '의료용 12K 고해상도 MicroLED 패널 개발', '롤러블 디스플레이 수율 20% 향상을 위한 AI 공정 모니터링 서비스')\n"
        # --- ✨✨✨ 여기까지 ---
        f"  2) 각 아이디어의 'problem' 항목에는 반드시 최신 트렌드나 'Why now' 관점을 1문장 이상 포함하여 문제의 시의성을 강조하세요.\n"
        f"  3) 'target_customer'는 '글로벌 완성차 OEM', '북미 빅테크 기업'처럼 구체적으로 명시하세요.\n"
        f"  4) 'priority_score'는 시장 잠재력, 기술 실현 가능성, 경쟁 강도를 고려하여 객관적으로 평가해주세요.\n"
        f"{maturity_prompt_injection}"
        f"컨텍스트:\n"
        f"{json.dumps(context, ensure_ascii=False)}"
    )

def strip_code_fence(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"^```[\t ]*\w*[\t ]*\n", "", t, flags=re.M)
    t = re.sub(r"\n```[\t ]*$", "", t, flags=re.M)
    return t.strip()

def clean_json_text2(t: str) -> str:
    t = strip_code_fence(t or "")
    t = t.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    t = re.sub(r"[\u0000-\u0008\u000B\u000C\u000E-\u001F\u2028\u2029]", "", t)
    t = re.sub(r",\s*(\}|\])", r"\1", t)
    t = t.lstrip("\ufeff")
    return t.strip()

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
        line = re.sub(r"^[\-\*\d\.\)\s]+", "", line)
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
CFG = load_config()
LLM = llm_config(CFG)

def load_context_for_prompt() -> Dict[str, Any]:
    keywords = load_json("outputs/keywords.json", default={"keywords": []}) or {"keywords": []}
    topics = load_json("outputs/topics.json", default={"topics": []}) or {"topics": []}
    insights = load_json("outputs/trend_insights.json", default={"summary": "", "top_topics": [], "evidence": {}}) or {"summary": "", "top_topics": [], "evidence": {}}
    trend_strength_path = "outputs/export/trend_strength.csv"
    events_path = "outputs/export/events.csv"
    tech_maturity = load_json("outputs/tech_maturity.json", default={"results": []}) or {"results": []}

    summary = (insights.get("summary") or "").strip()
    if len(summary) > 1200:
        summary = summary[:1200] + "…"

    kw_simple = [{"keyword": k.get("keyword",""), "score": k.get("score",0)} for k in (keywords.get("keywords") or [])[:20]]
    max_topics = int(CFG.get("llm_context_max_topics", 12))
    tp_simple = []
    for t in (topics.get("topics") or [])[:max_topics]:
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
                # 'types'와 'type'을 모두 안전하게 처리
                types_str = r.get("types") or r.get("type") or ""
                if types_str:
                    for etype in types_str.split(','):
                        evt_summary[etype.strip()] += 1
    except Exception:
        pass
    events_simple = dict(evt_summary)

    maturity_summary = []
    for res in tech_maturity.get("results", []):
        tech = res.get("technology")
        stage = res.get("analysis", {}).get("stage")
        if tech and stage and stage != "N/A":
            maturity_summary.append(f"- {tech}: {stage} 단계")
    maturity_context = "\n".join(maturity_summary)

    return {
        "summary": summary, 
        "keywords": kw_simple, 
        "topics": tp_simple, 
        "trends": trend_rows, 
        "events": events_simple,
        "tech_maturity": maturity_context
    }

def call_gemini(prompt: str) -> str:
    import google.generativeai as genai
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY 환경 변수가 없습니다.")
    genai.configure(api_key=api_key)
    model_name = str(LLM.get("model", "gemini-2.0-flash-001"))
    max_tokens = int(LLM.get("max_output_tokens", 2048))
    temperature = float(LLM.get("temperature", 0.3))
    model = genai.GenerativeModel(model_name)
    resp = model.generate_content(
        prompt,
        generation_config={"max_output_tokens": max_tokens, "temperature": temperature, "top_p": 0.9}
    )
    text = (getattr(resp, "text", None) or "").strip()
    return text

def make_opportunities_llm(meta_items: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    want = 5
    context = load_context_for_prompt()
    prompt = build_prompt(context, want=want)
    try:
        raw_text = call_gemini(prompt)
    except Exception as e:
        print(f"[ERROR] Gemini 호출 실패: {e}")
        return []
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
                try:
                    r["cur"] = int(r.get("cur", 0) or 0)
                    r["prev"] = int(r.get("prev", 0) or 0)
                    r["diff"] = int(r.get("diff", 0) or 0)
                    r["total"] = int(r.get("total", 0) or 0)
                    r["ma7"] = float(r.get("ma7", 0.0) or 0.0)
                    r["z_like"] = float(r.get("z_like", 0.0) or 0.0)
                    rows.append(r)
                except (ValueError, TypeError):
                    continue
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

def calculate_feasibility(idea_item: Dict[str, Any]) -> float:
    """ 아이디어 내용 기반으로 실현가능성 점수를 동적으로 계산 """
    base_score = 0.5  # 기본값 낮춤으로 변별력 확보

    text = f"{idea_item.get('idea', '')} {idea_item.get('problem', '')} {' '.join(idea_item.get('solution', []))} {' '.join(idea_item.get('risks', []))}"

    # 긍정 키워드 (점수 상승)
    positive_keywords = [
        '기존 기술', '파트너십', '검증된', '양산', '자동화', '효율 개선',
        '표준화', '레퍼런스 디자인', '공급망 확보', '상용화', '적용 사례', '기술 확보'
    ]
    for pk in positive_keywords:
        if pk in text:
            base_score += 0.05

    # 부정 키워드 (점수 하락)
    negative_keywords = [
        '장기 연구', '높은 비용', '신소재', '불확실성', '규제', '개인정보',
        '복잡한 인증', '기술 미성숙', '법적 리스크', '보안 문제', '시장 불안정'
    ]
    for nk in negative_keywords:
        if nk in text:
            base_score -= 0.05

    return clamp01(base_score)


def calculate_risk_score(idea_item: Dict[str, Any]) -> float:
    """ 리스크 평가 시스템 (0.1 ~ 0.5 분포 목표로 페널티 미세 조정) """
    score = 0.0
    
    # 1단계: 구조적 리스크 (LLM이 명시한 위험) - 페널티 하향
    structural_risks = idea_item.get('risks', [])
    score += len(structural_risks) * 0.05  # 항목당 0.10 -> 0.05

    # 분석할 전체 텍스트
    text_to_scan = f"{idea_item.get('idea', '')} {idea_item.get('problem', '')} {' '.join(idea_item.get('solution', []))} {' '.join(structural_risks)}"
    for e in idea_item.get("evidence", []):
        text_to_scan += " " + e.get("sentence", "")

    # 2단계: 치명적 리스크 키워드 - 페널티 하향
    high_risk_keywords = ["규제", "법적 리스크", "보안 문제", "해킹", "취약점", "소송"]
    score += sum(0.06 for rk in high_risk_keywords if rk in text_to_scan) # 0.08 -> 0.06

    # 3단계: 일반 리스크 키워드 - 페널티 하향
    medium_risk_keywords = [
        "개인정보", "불확실성", "기술 미성숙", "높은 비용", "복잡한 인증", 
        "시장 불안정", "공급망 리스크", "위협"
    ]
    score += sum(0.02 for rk in medium_risk_keywords if rk in text_to_scan) # 0.03 -> 0.02

    return clamp01(score) # 최종 점수는 0.0 ~ 1.0 사이로 보정

def enrich_with_signals(ideas: List[Dict[str,Any]],
                        meta_items: List[Dict[str,Any]],
                        trend_rows: List[Dict[str,Any]],
                        events_rows: List[Dict[str,str]],
                        cfg: Dict[str, Any],
                        keywords_obj: dict) -> List[Dict[str,Any]]:
    
    weights = cfg.get("score_weights", {})
    mkt_w = float(weights.get("market_weight", 0.40))
    urg_w = float(weights.get("urgency_weight", 0.35))
    feas_w = float(weights.get("feasibility_weight", 0.25))
    risk_p = float(weights.get("risk_penalty", 1.0))
    
    trend_idx = {r.get("term",""): r for r in trend_rows}
    event_hit = defaultdict(int)
    for r in events_rows:
        types_str = r.get("types", "")
        if types_str:
            for etype in types_str.split(','):
                event_hit[etype.strip()] += 1

    enriched_ideas = []
    for it in ideas:
        full_idea_text = f"{it.get('idea', '')} {it.get('problem', '')} {' '.join(it.get('solution', []))}"
        idea_keywords = extract_keywords_from_idea(full_idea_text, keywords_obj)
        
        cur, z = 0.0, 0.0
        if idea_keywords:
            related_trends = [trend_idx.get(kw) for kw in idea_keywords if trend_idx.get(kw)]
            if related_trends:
                cur_list = [float(t.get("cur", 0)) for t in related_trends]
                z_list = [float(t.get("z_like", 0.0)) for t in related_trends]
                if cur_list: cur = max(cur_list)
                if z_list: z = max(z_list)

        s_market = normalize_score(it.get("priority_score", 0.0), 0.0, 5.0) * 0.5 + normalize_score(cur, 0, 10) * 0.5
        s_market = clamp01(s_market)

        s_urg = normalize_score(z, 0, 3.0)
        evt_boost = sum(0.15 for etype in ["LAUNCH", "PARTNERSHIP"] if event_hit.get(etype, 0) > 0)
        s_urg = clamp01(s_urg + evt_boost)

        s_feas = calculate_feasibility(it)
        
        it["evidence"] = pick_evidence(idea_keywords, meta_items, limit=7)
        risk_score = calculate_risk_score(it)
      
        base_score = (mkt_w * s_market + urg_w * s_urg + feas_w * s_feas)
        final_score_raw = base_score * (1 - (risk_p * risk_score))
        
        it["score"] = round(final_score_raw * 10.0, 1)

        # --- 'risk_level' 관련 코드가 모두 제거되었습니다 ---
        it["score_breakdown"] = {
            "market": round(s_market, 3), 
            "urgency": round(s_urg, 3), 
            "feasibility": round(s_feas, 3), 
            "risk": round(risk_score, 3), 
            "notes": {"cur": cur, "z_like": z}
        }
        enriched_ideas.append(it)
    
    return enriched_ideas

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

    meta_items = load_json(latest("data/news_meta_*.json"), [])
    keywords_obj = load_json("outputs/keywords.json", {"keywords":[]})
    topics_obj   = load_json("outputs/topics.json", {"topics":[]})
    
    # 새로 추가: 분석 결과 로드
    analysis_summary = load_json("outputs/analysis_summary.json", {})
    
    trend_rows = load_trend_strength_csv("outputs/export/trend_strength.csv")
    events_rows = load_events_csv("outputs/export/events.csv")

    # 로그 보강
    print(f"[INFO] Loaded context data | trend_rows={len(trend_rows)}, events_rows={len(events_rows)}")
    print(f"[INFO] Analysis summary loaded | matrix_orgs={analysis_summary.get('matrix_stats', {}).get('num_orgs', 0)}")
    
    try:
        ideas_llm = make_opportunities_llm(meta_items)
    except Exception as e:
        print("[ERROR] LLM stage failed:", repr(e))
        ideas_llm = []

    # enrich_with_signals 호출 시 keywords_obj를 추가로 전달합니다.
    ideas_with_scores = enrich_with_signals(ideas_llm, meta_items, trend_rows, events_rows, CFG, keywords_obj)
    ideas_with_scores.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    
    ideas_final = fill_opportunities_to_five(ideas_with_scores, keywords_obj, want=5)

    save_json("outputs/biz_opportunities.json", {"ideas": ideas_final})
    print("[INFO] Module E done | ideas=%d" % len(ideas_final))

if __name__ == "__main__":
    main()

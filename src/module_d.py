# -*- coding: utf-8 -*-
import os
import re
import json
import csv
import glob
import unicodedata
import datetime
from typing import List, Dict, Any, Tuple
from collections import defaultdict, Counter

def latest(globpat: str):
    files = sorted(glob.glob(globpat))
    return files[-1] if files else None

def load_json(p, default=None):
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default if default is not None else {}

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

# ----- 간단 엔터티 추출(ORG/브랜드/모델 제외) -----
def extract_orgs(text: str) -> List[str]:
    if not text:
        return []
    toks = re.findall(r"[가-힣A-Za-z0-9\-\+\.]{2,}", text)
    orgs = []
    for t in toks:
        # 숫자 붙은 모델(아이폰17, 갤럭시S25, A19 등)은 제외
        if re.match(r"^[가-힣A-Za-z]+[0-9]{1,3}[A-Za-z]?$", t):
            continue
        # 흔한 잡단어 필터
        if t in ("기자", "무단전재", "재배포", "관련기사", "사진", "영상"):
            continue
        if len(t) >= 2 and not re.fullmatch(r"[0-9\W_]+", t):
            orgs.append(t)
    cnt = Counter(orgs)
    out = [w for w, _ in cnt.most_common(10)]
    return out

# ----- 토픽 라벨(상위 단어 묶음) -----
def load_topic_labels(topics_obj: Dict[str, Any], topn=5) -> List[Dict[str, Any]]:
    labels = []
    for t in topics_obj.get("topics", []):
        words = [w.get("word","") for w in (t.get("top_words") or [])][:topn]
        labels.append({"topic_id": t.get("topic_id"), "words": [w for w in words if w]})
    return labels

# ----- 회사×토픽 매트릭스 -----
def export_company_topic_matrix(meta_items: List[Dict[str,Any]], topic_labels: List[Dict[str,Any]]) -> None:
    """
    org가 들어있는 문서에서 topic 단어가 얼마나 같이 나오는지 세서 csv로 저장
    """
    os.makedirs("outputs/export", exist_ok=True)
    topic_wordsets = []
    for tl in topic_labels:
        ws = set([w for w in tl["words"] if w])
        topic_wordsets.append((tl["topic_id"], ws))

    matrix = defaultdict(lambda: defaultdict(int))  # org -> topic_id -> count
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

# ----- 스코어링/증거 -----
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

# --- 필드 합성 헬퍼 -------------------------------------------------
def infer_target_customer(sentences: list) -> str:
    base = "기업(B2B)"
    text = " ".join([s for s in sentences if s]).lower()
    if any(k in text for k in ["공공", "조달", "관공서", "지자체", "정부"]):
        return "공공/조달"
    if any(k in text for k in ["소비자", "일반 고객", "리테일", "매장", "고객경험"]):
        return "소비자(B2C)"
    if any(k in text for k in ["수주", "납품", "파트너", "mou", "제휴"]):
        return "기업(B2B)"
    return base

def synthesize_problem(term: str, sentences: list) -> str:
    has_neg = any(any(nw in (s or "") for nw in NEG_WORDS) for s in sentences)
    if has_neg:
        return f"{term} 관련 이슈(규제·품질·수요 변동)가 보도에서 확인되어 리스크 관리와 대응 전략 수립이 필요함."
    return f"{term} 수요/관심은 증가 중이나 표준·비용·도입 난이도 등 실행 과제가 상존함."

def synthesize_value_prop(term: str) -> str:
    return f"{term} 도입으로 비용 절감, 품질·성능 개선, 고객경험 향상 등 가시적 효과를 기대할 수 있음."

def synthesize_solution(term: str) -> str:
    return f"{term}에 대해 1) 파일럿 PoC 설계, 2) 핵심 파트너사 발굴·제휴(MOU), 3) 인증/규격 검토 및 조달/유통 채널 테스트를 순차 추진."

def extract_risks(sentences: list) -> list:
    risks = set()
    text = " ".join([s for s in sentences if s])
    for w in NEG_WORDS:
        if w in text:
            risks.add(w)
    if not risks:
        risks.update(["규제/표준 불확실성", "비용/ROI 불확실성"])
    return list(risks)

# ----- CSV 로더 -----
def load_trend_strength_csv(path: str) -> List[Dict[str,Any]]:
    rows = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                r["cur"] = int(r.get("cur", 0) or 0)
                r["prev"] = int(r.get("prev", 0) or 0)
                r["diff"] = int(r.get("diff", 0) or 0)
                try:
                    r["ma7"] = float(r.get("ma7", 0.0) or 0.0)
                except Exception:
                    r["ma7"] = 0.0
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

# ----- 기회 생성 -----
def make_opportunities(meta_items: List[Dict[str,Any]],
                       keywords_obj: Dict[str,Any],
                       trend_strength_rows: List[Dict[str,Any]],
                       events_rows: List[Dict[str,str]]) -> Dict[str,Any]:
    trend_idx = {}
    for r in trend_strength_rows:
        try:
            term = r["term"]
            trend_idx[term] = r
        except Exception:
            pass

    event_hit = defaultdict(int)
    for r in events_rows:
        et = r.get("type")
        if et:
            event_hit[et] += 1

    top_kw = (keywords_obj.get("keywords") or [])[:20]
    ideas = []

    ts_cur_vals = [int(trend_idx.get(k.get("keyword",""), {}).get("cur", 0)) for k in top_kw if k.get("keyword")]
    ts_z_vals   = [float(trend_idx.get(k.get("keyword",""), {}).get("z_like", 0.0)) for k in top_kw if k.get("keyword")]
    cur_hi = max([1] + ts_cur_vals); cur_lo = min([0] + ts_cur_vals) if ts_cur_vals else 0
    z_hi  = max([0.0] + ts_z_vals);  z_lo  = min([0.0] + ts_z_vals) if ts_z_vals else 0.0

    for k in top_kw:
        term = (k.get("keyword","") or "").strip()
        if not term:
            continue

        tr = trend_idx.get(term, {})
        cur = int(tr.get("cur", 0))
        z   = float(tr.get("z_like", 0.0))

        s_market = 0.75 * normalize_score(cur, cur_lo, cur_hi) + 0.25 * normalize_score(k.get("score",0.0), 0.0, (top_kw[0].get("score",1.0) or 1.0))
        s_urg = normalize_score(z, z_lo, z_hi)
        evt_boost = 0.0
        if event_hit.get("LAUNCH",0)>0: evt_boost += 0.10
        if event_hit.get("PARTNERSHIP",0)>0: evt_boost += 0.08
        if event_hit.get("INVEST",0)>0: evt_boost += 0.07
        if event_hit.get("ORDER",0)>0: evt_boost += 0.06
        if event_hit.get("CERT",0)>0: evt_boost += 0.05
        if event_hit.get("REGUL",0)>0: evt_boost += 0.04
        s_urg = clamp01(s_urg + evt_boost)

        s_feas = 0.6  # 간단 기본값

        evid = pick_evidence(term, meta_items, limit=3)
        risk = 0.0
        for e in evid:
            if any(nw in (e.get("sentence") or "") for nw in NEG_WORDS):
                risk += 0.10
        risk = clamp01(risk)

        score = 100.0 * (0.40 * s_market + 0.35 * s_urg + 0.25 * s_feas) - 100.0 * risk
        score = max(0.0, min(100.0, score))

        evid_sents = [e.get("sentence","") for e in (evid or []) if e.get("sentence")]
        problem = synthesize_problem(term, evid_sents)
        target_customer = infer_target_customer(evid_sents)
        value_prop = synthesize_value_prop(term)
        solution = synthesize_solution(term)
        risks_list = extract_risks(evid_sents)

        item = {
            "idea": term,                       # Check D 필수
            "title": term,                      # 호환
            "problem": problem,                 # Check D 필수
            "target_customer": target_customer, # Check D 필수
            "value_prop": value_prop,           # Check D 필수
            "solution": solution,               # Check D 필수
            "risks": risks_list,                # Check D 필수
            "priority_score": round(score, 2),  # Check D 필수
            "score": round(score, 2),
            "score_breakdown": {
                "market": round(s_market, 3),
                "urgency": round(s_urg, 3),
                "feasibility": round(s_feas, 3),
                "risk": round(risk, 3),
                "notes": {
                    "cur": cur,
                    "z_like": round(z,3),
                    "events_any": sum(event_hit.values()),
                }
            },
            "evidence": evid if isinstance(evid, list) else []
        }
        ideas.append(item)

    # 스키마 보정: 누락 대비
    for it in ideas:
        it["idea"] = (it.get("idea") or it.get("title") or "").strip() or "(no idea)"
        it["title"] = it.get("title") or it["idea"]
        it["problem"] = it.get("problem") or f"{it['idea']} 관련 과제가 상존함."
        it["target_customer"] = it.get("target_customer") or "기업(B2B)"
        it["value_prop"] = it.get("value_prop") or f"{it['idea']} 도입 가치(비용/품질/경험 개선)."
        it["solution"] = it.get("solution") or f"{it['idea']} 파일럿→제휴→인증 확보."
        it["risks"] = it.get("risks") or ["규제/표준 불확실성", "비용/ROI 불확실성"]
        it["priority_score"] = it.get("priority_score", it.get("score", 0))

    ideas = sorted(ideas, key=lambda x: x.get("priority_score", 0), reverse=True)
    return {"ideas": ideas}

def main():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/export", exist_ok=True)

    # 1) 입력 로드
    meta_path = latest("data/news_meta_*.json")
    meta_items = load_json(meta_path, [])
    keywords_obj = load_json("outputs/keywords.json", {"keywords":[]})
    topics_obj   = load_json("outputs/topics.json", {"topics":[]})

    trend_rows = load_trend_strength_csv("outputs/export/trend_strength.csv")
    events_rows = load_events_csv("outputs/export/events.csv")

    # 2) 회사×토픽 매트릭스 생성
    topic_labels = load_topic_labels(topics_obj, topn=5)
    try:
        export_company_topic_matrix(meta_items, topic_labels)
        print("[INFO] company_topic_matrix.csv exported")
    except Exception as e:
        print("[WARN] company_topic_matrix export failed:", repr(e))

    # 3) 기회 스코어링 + 근거 + Check D 필드 포함
    try:
        opp = make_opportunities(meta_items, keywords_obj, trend_rows, events_rows)
    except Exception as e:
        print("[ERROR] make_opportunities failed:", repr(e))
        opp = {"ideas": []}

    # 4) 저장
    with open("outputs/biz_opportunities.json", "w", encoding="utf-8") as f:
        json.dump(opp, f, ensure_ascii=False, indent=2)

    print("[INFO] Module D done | ideas=%d" % len(opp.get("ideas", [])))

if __name__ == "__main__":
    main()

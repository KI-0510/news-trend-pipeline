# -*- coding: utf-8 -*-
"""
signal_export.py
- 하루 1파일 정책(날짜별 최신 1개)
- 오늘(KST) 제외
- 발행일 기반 d/d+1 윈도우로 용어 시계열 집계
- trend_strength.csv / weak_signals.csv / events.csv 생성
"""

import os
import re
import csv
import glob
import json
from collections import defaultdict, Counter
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

# ===================== KST 시간/날짜 =====================
KST = timezone(timedelta(hours=9))

def kst_today_str() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d")

def kst_yesterday_str() -> str:
    return (datetime.now(KST) - timedelta(days=1)).strftime("%Y-%m-%d")

# ===================== 유틸 =====================
def latest(globpat: str):
    files = sorted(glob.glob(globpat))
    return files[-1] if files else None

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def clean_text(t: str) -> str:
    if not t: return ""
    t = re.sub(r"<.+?>", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def std_date_from_published(published: str, fallback: str) -> str:
    if published:
        try:
            iso = published.replace("Z", "+00:00")
            dt = datetime.fromisoformat(iso)
            return dt.date().strftime("%Y-%m-%d")
        except Exception:
            try:
                dt = parsedate_to_datetime(published)
                return dt.date().strftime("%Y-%m-%d")
            except Exception:
                pass
    return fallback

# ===================== 불용어/토큰화 =====================
EN_STOP = {
    "the","and","to","of","in","for","on","with","at","by","from","as","is","are","be","it",
    "that","this","an","a","or","if","we","you","they","he","she","was","were","been","than",
    "into","about","over","under","per","via"
}
KO_FUNC = {
    "하다","있다","되다","통해","이번","대한","것으로","밝혔다","다양한","함께","현재",
    "기자","대표","회장","주요","기준","위해","위한","지원","전략","정책","협력","확대",
    "말했다","강조했다","대상","대상으로","최근","지난해","생활","시장","스마트","디지털","글로벌",
    "그는","그녀는","이어","한편","또한","이날","이라며","이라고","모델을","성과를","받았다","서울","기반으로",
    "있는","있으며","있다는","이후","설명했다","전했다","계획이다","관계자는","따르면",
    "올해","내년","최대","신규","기존","국제","국내","세계","오전","오후",
    "등을","따라","있도록","지난","특히","대비","아니라","만에","의원은","라고",
    "있습니다","관련","한다","진행한다","예정이다","가능하다","있었다",
    "이상","넘어","제공한다","같은","했다","많은","그리고","같다","우리","하고",
    "때문에","이렇게","이런","등이","각각"
}
STOP_EXT = set(EN_STOP) | set(KO_FUNC)

def norm_tok(x: str) -> str:
    return x.lower()

def tokenize(text: str):
    toks = re.findall(r"[가-힣A-Za-z0-9]{2,}", text or "")
    toks = [norm_tok(x) for x in toks if x and x.lower() not in STOP_EXT]
    return toks

# ===================== 하루 1파일: 날짜별 최신 =====================
def group_by_day_latest(glob_pattern: str) -> dict:
    files = sorted(glob.glob(glob_pattern))
    by_day = defaultdict(list)
    for f in files:
        d = os.path.basename(f)[:10]
        by_day[d].append(f)
    latest_map = {}
    for d, fps in by_day.items():
        latest_map[d] = sorted(fps)[-1]
    return latest_map

# ===================== 이벤트 추출/저장 =====================
EVENT_MAP = {
    "LAUNCH":      [r"출시", r"론칭", r"발표", r"선보이", r"공개"],
    "PARTNERSHIP": [r"제휴", r"파트너십", r"업무협약", r"\bMOU\b", r"맞손"],
    "INVEST":      [r"투자", r"유치", r"라운드", r"시리즈 [ABCD]"],
    "ORDER":       [r"수주", r"계약 체결", r"납품 계약", r"공급 계약", r"수의 계약"],
    "CERT":        [r"인증", r"허가", r"승인", r"적합성 평가", r"CE ?인증", r"FDA ?승인"],
    "REGUL":       [r"규제", r"가이드라인", r"행정예고", r"고시", r"지침", r"제정", r"개정"],
}

def _pick_meta_path():
    p1 = "outputs/debug/news_meta_latest.json"
    if os.path.exists(p1): return p1
    return latest("data/news_meta_*.json")

def _detect_events_from_items(items: list) -> list:
    rows = []
    for it in items:
        title = (it.get("title") or it.get("title_og") or "").strip()
        body  = (it.get("body") or it.get("description") or it.get("description_og") or "").strip()
        text  = f"{title}\n{body}"
        date  = (it.get("date") or it.get("pubDate") or "")[:10]
        url   = it.get("url") or ""
        src   = it.get("source") or it.get("press") or ""
        for etype, pats in EVENT_MAP.items():
            for pat in pats:
                if re.search(pat, text, flags=re.IGNORECASE):
                    rows.append({"date": date or "", "type": etype, "title": title[:300], "url": url, "source": src})
                    break
    return rows

def _dedup_events(rows: list) -> list:
    seen, out = set(), []
    for r in rows:
        key = (r.get("date",""), r.get("type",""), r.get("title",""))
        if key in seen: continue
        seen.add(key); out.append(r)
    return out

def export_events(out_path="outputs/export/events.csv"):
    ensure_dir(os.path.dirname(out_path))
    meta_path = _pick_meta_path()
    if not meta_path:
        print("[INFO] events.csv skipped (no meta)")
        return
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            items = json.load(f) or []
    except Exception as e:
        print("[WARN] events: meta load failed:", repr(e))
        items = []
    rows = _detect_events_from_items(items)
    rows = _dedup_events(rows)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date","type","title","url","source"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[INFO] events.csv exported | rows={len(rows)}")

# ===================== 시계열(용어) 계산 헬퍼 =====================
def iter_jsonl(fp: str):
    try:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = (line or "").strip()
                if not line: continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue
    except Exception:
        return

def build_day_docs_published_based(days: int = 28) -> tuple[list, dict]:
    """
    발행일 기반 d/d+1 윈도우로 day_docs 구성.
    - 대상 날짜: 과거 days일 (오늘 제외)
    - 각 d에 대해 d, d+1 최신 jsonl만 스캔
    - 문서는 published==d 인 것만 포함(title 기준 텍스트)
    반환:
      days_sorted, day_docs(dict: day -> [text...])
    """
    latest_map = group_by_day_latest("data/warehouse/*.jsonl")
    if not latest_map:
        return [], {}

    today = kst_today_str()
    all_days_sorted = sorted([d for d in latest_map.keys() if d < today])
    target_days = all_days_sorted[-max(1, days):]

    day_docs = {d: [] for d in target_days}
    for d in target_days:
        d_plus_1 = (datetime.strptime(d, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        fps = []
        if d in latest_map: fps.append((d, latest_map[d]))
        if d_plus_1 in latest_map: fps.append((d_plus_1, latest_map[d_plus_1]))

        seen = set()
        for file_day, fp in fps:
            for obj in iter_jsonl(fp):
                title = clean_text(obj.get("title") or "")
                if not title: continue
                pub_day = std_date_from_published(obj.get("published") or obj.get("created_at") or "", fallback=file_day)
                if pub_day != d:
                    continue
                key = obj.get("url") or (title[:200] if title else "")
                if key in seen:
                    continue
                seen.add(key)
                day_docs[d].append(title)
    return target_days, day_docs

def moving_avg(vals, w=7):
    out = []
    for i in range(len(vals)):
        s = max(0, i - w + 1)
        seg = vals[s:i+1]
        out.append(sum(seg) / max(1, len(seg)))
    return out

def z_like(vals, ma):
    # 간단 안정형 z-like
    z = []
    for v, m in zip(vals, ma):
        z.append((v - m) / ((m**0.5) + 1.0))
    return z

def to_rows_from_day_docs(day_docs: dict) -> list:
    # term universe 추출
    terms = set()
    for docs in day_docs.values():
        for doc in docs:
            terms.update(tokenize(doc))

    dates = sorted(day_docs.keys())
    rows = []
    for t in sorted(terms):
        counts = []
        for d in dates:
            cnt = sum(1 for doc in day_docs[d] if t in tokenize(doc))
            counts.append(cnt)
        ma7 = moving_avg(counts, 7)
        z = z_like(counts, ma7)
        cur = counts[-1] if counts else 0
        prev = counts[-2] if len(counts) >= 2 else 0
        diff = cur - prev
        rows.append({
            "term": t,
            "dates": dates,
            "counts": counts,
            "cur": cur, "prev": prev, "diff": diff,
            "ma7": ma7[-1] if ma7 else 0.0,
            "z_like": z[-1] if z else 0.0,
            "total": sum(counts)
        })
    return rows

# ===================== 내보내기 =====================
def export_trend_strength(rows: list, out_path="outputs/export/trend_strength.csv"):
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["term", "cur", "prev", "diff", "ma7", "z_like", "total"])
        for r in sorted(rows, key=lambda x: (x["z_like"], x["diff"], x["cur"]), reverse=True)[:300]:
            w.writerow([r["term"], r["cur"], r["prev"], r["diff"], round(r["ma7"],3), round(r["z_like"],3), r["total"]])

def export_weak_signals(rows: list, out_path="outputs/export/weak_signals.csv"):
    ensure_dir(os.path.dirname(out_path))
    cand = []
    for r in rows:
        if r["total"] <= 15 and r["cur"] >= 2 and r["z_like"] > 0.8:
            cand.append(r)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["term", "cur", "prev", "diff", "ma7", "z_like", "total"])
        for r in sorted(cand, key=lambda x: (x["z_like"], x["cur"]), reverse=True)[:200]:
            w.writerow([r["term"], r["cur"], r["prev"], r["diff"], round(r["ma7"],3), round(r["z_like"],3), r["total"]])

# ===================== 메인 =====================
def main():
    print("[INFO] signal_export start | policy: one-file-per-day, exclude-today, publish(d/d+1)")
    # 1) 발행일 기반 d/d+1 윈도우로 day_docs 구성(오늘 제외)
    days, day_docs = build_day_docs_published_based(days=28)
    if not days:
        print("[WARN] no warehouse docs; skip trend exports")
        export_events()
        return

    # 2) 용어 시계열 행 생성
    rows = to_rows_from_day_docs(day_docs)

    # 3) 내보내기
    export_trend_strength(rows)
    export_weak_signals(rows)
    export_events()

    print(f"[INFO] signal_export done | days={len(days)} terms={len(rows)}")

if __name__ == "__main__":
    main()

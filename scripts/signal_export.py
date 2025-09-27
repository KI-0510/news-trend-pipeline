import os
import re
import glob
import json
import csv
import datetime
import unicodedata
from collections import defaultdict, Counter

DICT_DIR = "data/dictionaries"
def _load_lines(p):
    try:
        with open(p, encoding="utf-8") as f:
            return [x.strip() for x in f if x.strip()]
    except Exception:
        return []
STOP_EXT = set(_load_lines(os.path.join(DICT_DIR, "stopwords_ext.txt")))

def norm_tok(s):
    s = unicodedata.normalize("NFKC", s or "")
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def tokenize(t):
    toks = re.findall(r"[가-힣A-Za-z0-9]{2,}", t or "")
    toks = [norm_tok(x) for x in toks if x and x not in STOP_EXT]
    return toks

# ================= 데이터 로더 (모듈 C와 로직 통일) =================
def select_latest_files_per_day(glob_pattern: str):
    all_files = sorted(glob.glob(glob_pattern))
    daily_files = defaultdict(list)
    for f in all_files:
        date_key = os.path.basename(f)[:10]
        daily_files[date_key].append(f)
    
    latest_daily_files = []
    for date_key in sorted(daily_files.keys()):
        latest_file_for_day = sorted(daily_files[date_key])[-1]
        latest_daily_files.append(latest_file_for_day)
    return latest_daily_files

def load_stable_warehouse_data(days: int = 30):
    """
    D일자와 D+1일자 규칙을 적용하여 안정적인 시계열 데이터를 로드합니다.
    """
    warehouse_files = select_latest_files_per_day("data/warehouse/*.jsonl")
    file_map = {os.path.basename(f)[:10]: f for f in warehouse_files}
    
    if not file_map:
        return []
    
    sorted_dates = sorted(file_map.keys())
    # 분석 시작일은 요청일수(days)만큼 과거로 설정
    start_date = datetime.datetime.strptime(sorted_dates[-1], "%Y-%m-%d").date() - datetime.timedelta(days=days)
    # 마지막 날짜는 D+1일이 없으므로, 그 전날까지만 계산합니다.
    end_date = datetime.datetime.strptime(sorted_dates[-1], "%Y-%m-%d").date() - datetime.timedelta(days=1)
    
    rows = []
    current_date = start_date

    while current_date <= end_date:
        date_str_d = current_date.strftime("%Y-%m-%d")
        date_str_d_plus_1 = (current_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        
        files_to_check = []
        if date_str_d in file_map:
            files_to_check.append(file_map[date_str_d])
        if date_str_d_plus_1 in file_map:
            files_to_check.append(file_map[date_str_d_plus_1])
            
        for fp in files_to_check:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            obj = json.loads(line)
                            d_raw = obj.get("published") or obj.get("created_at") or os.path.basename(fp)[:10]
                            published_date = d_raw[:10]
                            
                            if published_date == date_str_d:
                                title = (obj.get("title") or "").strip()
                                toks = tokenize(title)
                                rows.append((date_str_d, toks))
                        except Exception:
                            continue
            except Exception:
                continue
        current_date += datetime.timedelta(days=1)
        
    return rows

# ================= 통계 계산 함수들 =================
def daily_counts(rows):
    by_day = defaultdict(Counter)
    for d, toks in rows:
        for t in toks:
            by_day[d][t] += 1
    return dict(sorted(by_day.items()))

def moving_avg(vals, w=7):
    out = []
    for i in range(len(vals)):
        s = max(0, i - w + 1)
        seg = vals[s:i+1]
        out.append(sum(seg) / max(1, len(seg)))
    return out

def z_like(vals, ma):
    # 편차 / (sqrt(ma)+1) 간단화
    z = []
    for v, m in zip(vals, ma):
        z.append((v - m) / ( (m**0.5) + 1.0 ))
    return z

def to_rows(dc):
    # dc: date -> Counter
    # terms universe
    terms = set()
    for d, c in dc.items():
        terms.update(c.keys())
    dates = sorted(dc.keys())
    rows = []
    for t in sorted(terms):
        counts = [dc[d].get(t, 0) for d in dates]
        ma7 = moving_avg(counts, 7)
        z = z_like(counts, ma7)
        # 최근 값
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

def export_trend_strength(rows):
    os.makedirs("outputs/export", exist_ok=True)
    with open("outputs/export/trend_strength.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["term", "cur", "prev", "diff", "ma7", "z_like", "total"])
        # 스파이크 기준 상위만
        for r in sorted(rows, key=lambda x: (x["z_like"], x["diff"], x["cur"]), reverse=True)[:300]:
            w.writerow([r["term"], r["cur"], r["prev"], r["diff"], round(r["ma7"],3), round(r["z_like"],3), r["total"]])

def export_weak_signals(rows):
    # 희소하면서 최근 증가세인 용어
    cand = []
    for r in rows:
        if r["total"] <= 15 and r["cur"] >= 2 and r["z_like"] > 0.8:
            cand.append(r)
    with open("outputs/export/weak_signals.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["term", "cur", "prev", "diff", "ma7", "z_like", "total"])
        for r in sorted(cand, key=lambda x: (x["z_like"], x["cur"]), reverse=True)[:200]:
            w.writerow([r["term"], r["cur"], r["prev"], r["diff"], round(r["ma7"],3), round(r["z_like"],3), r["total"]])


# ===== 이벤트 추출/저장 =====
EVENT_MAP = {
    "LAUNCH":      [r"출시", r"론칭", r"발표", r"선보이", r"공개"],
    "PARTNERSHIP": [r"제휴", r"파트너십", r"업무협약", r"\bMOU\b", r"맞손"],
    "INVEST":      [r"투자", r"유치", r"라운드", r"시리즈 [ABCD]"],
    "ORDER":       [r"수주", r"계약 체결", r"납품 계약", r"공급 계약", r"수의 계약"],
    "CERT":        [r"인증", r"허가", r"승인", r"적합성 평가", r"CE ?인증", r"FDA ?승인"],
    "REGUL":       [r"규제", r"가이드라인", r"행정예고", r"고시", r"지침", r"제정", r"개정"],
}

def _latest(path_glob: str):
    files = sorted(glob.glob(path_glob))
    return files[-1] if files else None

def _pick_meta_path():
    p1 = "outputs/debug/news_meta_latest.json"
    if os.path.exists(p1):
        return p1
    return _latest("data/news_meta_*.json")

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
                    rows.append({
                        "date": date or "",
                        "type": etype,
                        "title": title[:300],
                        "url": url,
                        "source": src
                    })
                    break
    return rows

def _dedup_events(rows: list) -> list:
    seen, out = set(), []
    for r in rows:
        key = (r.get("date",""), r.get("type",""), r.get("title",""))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out

def export_events(out_path="outputs/export/events.csv"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    meta_path = _pick_meta_path()
    if not meta_path:
        print("[INFO] events.csv skipped (no meta)")
        return
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            items = json.load(f)
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

# ================= 메인 =================
def main():
    # 하루 1파일 및 D+D+1 안정성 정책이 적용된 함수를 호출합니다.
    rows = load_stable_warehouse_data(days=30)
    
    dc = daily_counts(rows)
    rows2 = to_rows(dc)
    export_trend_strength(rows2)
    export_weak_signals(rows2)
    export_events()
    print("[INFO] signal_export | terms=", len(rows2))

if __name__ == "__main__":
    main()
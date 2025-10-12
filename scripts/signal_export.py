import os
import re
import glob
import json
import csv
import datetime
import unicodedata
from collections import defaultdict, Counter
from src.timeutil import to_date, kst_date_str, kst_run_suffix

# =================== 설정 ===================
DICT_DIR = "data/dictionaries"

def _load_lines(p):
    try:
        with open(p, encoding="utf-8") as f:
            return [x.strip() for x in f if x.strip()]
    except Exception:
        return []

STOP_EXT = set(_load_lines(os.path.join(DICT_DIR, "stopwords_ext.txt")))

# =================== 정규화/토큰화 ===================
def norm_tok(s):
    s = unicodedata.normalize("NFKC", s or "")
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def tokenize(t):
    toks = re.findall(r"[가-힣A-Za-z0-9]{2,}", t or "")
    toks = [norm_tok(x) for x in toks if x and x not in STOP_EXT]
    return toks

# =================== 어휘집 로드/정제 ===================
def load_signal_vocabulary():
    cfg = {}
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        print("[WARN] signal_export: config.json을 찾을 수 없습니다.")
    
    vocab = set(cfg.get("domain_hints", []))
    vocab.update(_load_lines(os.path.join(DICT_DIR, "brands.txt")))
    vocab.update(_load_lines(os.path.join(DICT_DIR, "entities_org.txt")))
    vocab = {norm_tok(v) for v in vocab if v}

    # 1차 정제: 범주형/숫자·단위/기간 제거
    _GENERIC_STOP = {"미국","유럽","산업","업계","공급","시장","관련","분야","글로벌","국내","해외","업체"}
    _RE_UNIT   = re.compile(r"^\d+(hz|w|mah|nm|mm|cm|kg|g|gb|tb|mhz|ghz|nit|nits)\$", re.I)
    _RE_PERIOD = re.compile(r"^\d{1,4}(년|월|분기)\$")
    _RE_COUNT  = re.compile(r"^\d+(위|종|개국|명|가지)\$")
    _RE_MIXED  = re.compile(r"^\d+[a-z가-힣]+\$", re.I)

    def _vocab_noise(x: str) -> bool:
        if x in _GENERIC_STOP: return True
        if x.isdigit(): return True
        if _RE_UNIT.match(x) or _RE_PERIOD.match(x) or _RE_COUNT.match(x) or _RE_MIXED.match(x):
            return True
        return False

    vocab = {t for t in vocab if not _vocab_noise(t)}
    return vocab

# =================== 데이터 로딩(안정) ===================
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
    files = select_latest_files_per_day("data/warehouse/*.jsonl")
    file_map = {os.path.basename(f)[:10]: f for f in files}
    if not file_map:
        return []
    sorted_dates = sorted(file_map.keys())
    start_date = datetime.datetime.strptime(sorted_dates[-1], "%Y-%m-%d").date() - datetime.timedelta(days=days)
    end_date   = datetime.datetime.strptime(sorted_dates[-1], "%Y-%m-%d").date() - datetime.timedelta(days=1)

    rows = []
    current_date = start_date
    while current_date <= end_date:
        d0 = current_date.strftime("%Y-%m-%d")
        d1 = (current_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        files_to_check = []
        if d0 in file_map: files_to_check.append(file_map[d0])
        if d1 in file_map: files_to_check.append(file_map[d1])

        for fp in files_to_check:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            obj = json.loads(line)
                            d_raw = obj.get("published") or obj.get("created_at") or os.path.basename(fp)[:10]
                            published_date = (d_raw or "")[:10]
                            if published_date == d0:
                                title = (obj.get("title") or "").strip()
                                toks = tokenize(title)
                                rows.append((d0, toks))
                        except Exception:
                            continue
            except Exception:
                continue
        current_date += datetime.timedelta(days=1)
    return rows

# =================== 통계/지표 ===================
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
    z = []
    for v, m in zip(vals, ma):
        zv = (v - m) / ((m**0.5) + 1.0)
        z.append(zv)
    return z

def to_rows(dc):
    terms = set()
    for d, c in dc.items():
        terms.update(c.keys())
    dates = sorted(dc.keys())
    rows = []
    for t in sorted(terms):
        counts = [dc[d].get(t, 0) for d in dates]
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

# =================== 이벤트 규칙 ===================
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
        date_raw = it.get("published_time") or it.get("pubDate_raw") or ""
        date = to_date(date_raw)
        url = it.get("url") or ""

        detected_types = []
        for etype, pats in EVENT_MAP.items():
            for pat in pats:
                if re.search(pat, text, flags=re.IGNORECASE):
                    detected_types.append(etype)
                    break
        if detected_types:
            rows.append({
                "date": date or "",
                "types": ",".join(sorted(detected_types)),
                "title": title[:300],
                "url": url
            })
    return rows

def _dedup_events(rows: list) -> list:
    seen_titles, seen_urls, out = set(), set(), []
    for r in rows:
        title = r.get("title", "")
        url = r.get("url", "")
        if not title or not url or title in seen_titles or url in seen_urls:
            continue
        seen_titles.add(title)
        seen_urls.add(url)
        out.append(r)
    return out

# =================== CSV Export ===================
def export_trend_strength(rows):
    os.makedirs("outputs/export", exist_ok=True)
    final_path = "outputs/export/trend_strength.csv"
    tmp_path = final_path + ".tmp"

    bad_generic = {"공급","산업","업계","시장","관련","분야"}
    filtered = [r for r in rows if r["term"] not in bad_generic and r["cur"] >= 1]
    filtered.sort(key=lambda x: (x["z_like"], x["diff"], x["cur"]), reverse=True)
    topk = filtered[:300]

    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["term","cur","prev","diff","ma7","z_like","total"])
        for r in topk:
            w.writerow([r["term"], r["cur"], r["prev"], r["diff"], round(float(r["ma7"]),3), round(float(r["z_like"]),3), r["total"]])
    os.replace(tmp_path, final_path)
    print(f"[INFO] trend_strength | rows={len(topk)}")

def export_weak_signals(rows):
    os.makedirs("outputs/export", exist_ok=True)
    final_path = "outputs/export/weak_signals.csv"
    tmp_path = final_path + ".tmp"

    generic_stop = {"미국","유럽","산업","업계","공급","시장","관련","분야","글로벌","국내","해외","업체"}
    cand = []
    pre_cand = []  # 1차 조건만 통과(디버그용)

    for r in rows:
        term = r["term"]
        if term in generic_stop:
            continue

        z = float(r["z_like"])
        tot = int(r["total"])

        # 최근성 필터(스테이징 전)
        if r["cur"] >= 1 and r["prev"] <= 3 and r["diff"] >= 0:
            pre_cand.append(r)

            # 스테이징 컷
            pass_cond = False
            if z > 1.0 and tot <= 40:
                pass_cond = True
            elif z > 0.9 and tot <= 25:
                pass_cond = True

            if pass_cond:
                cand.append(r)

    # 후보 과다 시 2단 컷(보수형)
    if len(cand) > 50:
        cand = [r for r in cand if float(r["z_like"]) > 1.4 and r["total"] <= 20]

    cand.sort(key=lambda x: (x["z_like"], x["cur"], x["diff"], -x["total"], -x["prev"]), reverse=True)

    # 디버그 덤프
    os.makedirs("outputs/debug", exist_ok=True)
    with open("outputs/debug/weak_candidates.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["term","cur","prev","diff","total","ma7","z_like"])
        for r in sorted(pre_cand, key=lambda x: (x["z_like"], x["cur"], -x["total"]), reverse=True):
            w.writerow([r["term"], r["cur"], r["prev"], r["diff"], r["total"], round(float(r["ma7"]),3), round(float(r["z_like"]),3)])

    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["term","cur","prev","diff","ma7","z_like","total"])
        for r in cand[:50]:
            w.writerow([r["term"], r["cur"], r["prev"], r["diff"], round(float(r["ma7"]),3), round(float(r["z_like"]),3), r["total"]])
    os.replace(tmp_path, final_path)
    print(f"[INFO] weak_signals | pre={len(pre_cand)} final={len(cand)}")
    # 분포 힌트
    z_over_1 = sum(1 for r in pre_cand if float(r["z_like"]) > 1.0)
    z_09_10 = sum(1 for r in pre_cand if 0.9 < float(r["z_like"]) <= 1.0)
    z_085_09 = sum(1 for r in pre_cand if 0.85 < float(r["z_like"]) <= 0.9)
    print(f"[DEBUG] z>1.0:{z_over_1} | 0.9<z<=1.0:{z_09_10} | 0.85<z<=0.9:{z_085_09}")

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
        print(f"[WARN] events: meta load failed: {repr(e)}")
        items = []

    rows = _detect_events_from_items(items)
    rows = _dedup_events(rows)

    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date", "types", "title", "url"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    os.replace(tmp_path, out_path)
    print(f"[INFO] events.csv exported | rows={len(rows)}")

# =================== 메인 ===================
def main():
    # 1) 신호 어휘집
    signal_vocab = load_signal_vocabulary()
    print(f"[INFO] 신호 어휘집 로드/정제 완료: {len(signal_vocab)}개")

    # 2) 데이터 로드
    rows_raw = load_stable_warehouse_data(days=30)

    # 3) 어휘집 기반 필터링
    filtered_rows = []
    for d, toks in rows_raw:
        qualified = [t for t in toks if t in signal_vocab]
        if qualified:
            filtered_rows.append((d, qualified))
    print(f"[INFO] 원본 기사수={len(rows_raw)} → 어휘집 필터 후={len(filtered_rows)}")

    # 4) 통계 변환
    dc = daily_counts(filtered_rows)
    rows_stat = to_rows(dc)
    print(f"[INFO] 통계 대상 term 수={len(rows_stat)}")

    # 5) 오늘자(cur>0) 디버그 덤프
    os.makedirs("outputs/debug", exist_ok=True)
    with open("outputs/debug/today_terms.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["term","cur","prev","diff","total","ma7","z_like"])
        for r in sorted(rows_stat, key=lambda x: (x["cur"], x["z_like"], x["total"]), reverse=True):
            if r["cur"] > 0:
                w.writerow([r["term"], r["cur"], r["prev"], r["diff"], r["total"], round(float(r["ma7"]),3), round(float(r["z_like"]),3)])
    print(f"[DEBUG] today_terms.csv exported")

    # 6) Export
    export_trend_strength(rows_stat)
    export_weak_signals(rows_stat)
    export_events()

if __name__ == "__main__":
    main()

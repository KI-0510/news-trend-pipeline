# -*- coding: utf-8 -*-
import os, re, glob, json, csv, datetime, unicodedata
from collections import defaultdict, Counter

DICT_DIR = "data/dictionaries"
def _load_lines(p):
    try:
        with open(p, encoding="utf-8") as f:
            return [x.strip() for x in f if x.strip()]
    except Exception:
        return []
STOP_EXT = set(_load_lines(os.path.join(DICT_DIR, "stopwords_ext.txt")))

# ===== 하루 1파일 선택 유틸 =====
def _date_from_name(fp: str) -> str:
    # 파일명 앞 10자 기준(YYYY-MM-DD)
    base = os.path.basename(fp)
    return base[:10]

def _list_warehouse_files(pattern="data/warehouse/*.jsonl"):
    return sorted(glob.glob(pattern))

def _pick_one_per_day(files: list, strategy="latest") -> list:
    from collections import defaultdict
    by_day = defaultdict(list)
    for fp in files:
        d = _date_from_name(fp)
        by_day[d].append(fp)
    picked = []
    for d, fps in by_day.items():
        fps.sort()  # 시간 포함 파일명은 사전순=시간순
        if strategy == "latest":
            picked.append(fps[-1])
        elif strategy == "earliest":
            picked.append(fps[0])
        elif strategy == "random":
            import random
            picked.append(random.choice(fps))
        else:
            picked.append(fps[-1])
    return sorted(picked)

def load_warehouse_unique_per_day(days=30, strategy="latest"):
    """
    기존 load_warehouse 대체: 같은 날짜 파일이 여러 개여도 하루 1개만 로드.
    """
    rows = []
    files = _list_warehouse_files()
    # days 윈도 적용: 뒤에서부터 days일의 고유 날짜만 남기기
    # 1) 우선 전체를 날짜 그룹핑 후 날짜 정렬
    from collections import defaultdict
    by_day = defaultdict(list)
    for fp in files:
        by_day[_date_from_name(fp)].append(fp)
    days_sorted = sorted(by_day.keys())[-max(1, days):]  # 최근 days일만
    # 2) 해당 날짜들에서 전략대로 1개씩 선택
    pick_pool = []
    for d in days_sorted:
        fps = sorted(by_day[d])
        if strategy == "latest":
            pick_pool.append(fps[-1])
        elif strategy == "earliest":
            pick_pool.append(fps[0])
        elif strategy == "random":
            import random
            pick_pool.append(random.choice(fps))
        else:
            pick_pool.append(fps[-1])
    # 3) 파일 로드
    for fp in pick_pool:
        d = _date_from_name(fp)
        try:
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    title = (obj.get("title") or "").strip()
                    toks = tokenize(title)
                    rows.append((d, " ".join(toks)))
        except Exception:
            continue
    return rows
    
def norm_tok(s):
    s = unicodedata.normalize("NFKC", s or "")
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def tokenize(t):
    toks = re.findall(r"[가-힣A-Za-z0-9]{2,}", t or "")
    toks = [norm_tok(x) for x in toks if x and x not in STOP_EXT]
    return toks

def to_date_from_name(fp):
    # 파일명: YYYY-MM-DD-hhmm-KST.jsonl
    base = os.path.basename(fp)
    d = base[:10]
    return d

def load_warehouse(days=30):
    files = sorted(glob.glob("data/warehouse/*.jsonl"))[-days:]
    rows = []
    for fp in files:
        d = to_date_from_name(fp)
        try:
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    title = obj.get("title") or ""
                    toks = tokenize(title)
                    rows.append((d, toks))
        except Exception:
            continue
    return rows

def daily_counts(rows):
    # rows: list of (date, tokens)
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

def main():
    rows = load_warehouse(days=30)
    dc = daily_counts(rows)
    rows2 = to_rows(dc)
    export_trend_strength(rows2)
    export_weak_signals(rows2)
    print("[INFO] signal_export | terms=", len(rows2))

if __name__ == "__main__":
    main()

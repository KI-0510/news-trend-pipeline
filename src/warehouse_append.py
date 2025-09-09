import os
import json
import glob
import re
import sys
import datetime
from email.utils import parsedate_to_datetime
from timeutil import now_kst, kst_date_str, kst_run_suffix

def latest(globpat):
    files = sorted(glob.glob(globpat))
    return files[-1] if files else None

def to_date(s: str) -> str:
    today = datetime.date.today()
    if not s or not isinstance(s, str):
        return today.strftime("%Y-%m-%d")
    s = s.strip()
    
    # 1) ISO-8601
    try:
        iso = s.replace("Z", "+00:00")
        dt = datetime.datetime.fromisoformat(iso)
        d = dt.date()
    except Exception:
        # 2) RFC2822 (예: Wed, 03 Sep 2025 11:22:33 +0900)
        try:
            dt = parsedate_to_datetime(s)
            d = dt.date()
        except Exception:
            # 3) 정규식 추출 + 실제 달력 검증
            m = re.search(r"(\d{4}).*?(\d{1,2}).*?(\d{1,2})", s)
            if m:
                y, mm, dd = int(m.group(1)), int(m.group(2)), int(m.group(3))
                try:
                    d = datetime.date(y, mm, dd)
                except Exception:
                    d = today
            else:
                d = today
    
    # 미래 날짜 방지
    if d > today:
        d = today
    return d.strftime("%Y-%m-%d")

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def load_existing_urls(path):
    urls = set()
    if not os.path.exists(path):
        return urls
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                u = obj.get("url")
                if u:
                    urls.add(u)
            except Exception:
                continue
    return urls

def main():
    meta_path = latest("data/news_meta_*.json")
    if not meta_path:
        print("[ERROR] meta 파일이 없습니다. Module A부터 실행하세요.")
        sys.exit(1)
    
    with open(meta_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    
    ensure_dir("data/warehouse")
    
    appended, skipped = 0, 0
    
    for it in items:
        url = (it.get("url") or "").strip()
        if not url:
            skipped += 1
            continue
        
        # description 값을 JSON 파일에서 직접 가져옵니다.
        description_text = it.get("description", "")
        if not description_text:
             skipped += 1
             continue
        
        d_raw = it.get("published_time") or it.get("pubDate_raw") or ""
        published = to_date(d_raw)
        
        row = {
            "url": url,
            "title": it.get("title"),
            "description": description_text, # 테스트 이후 주석 처리, 'full_text' 대신 JSON에서 읽은 'description'을 사용
            "site_name": it.get("site_name"),
            "_query": it.get("_query") or it.get("query"),
            "published": published,
            "created_at": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"
        }

        date_part = kst_date_str()            # 예: 2025-09-08
        run_part  = kst_run_suffix()          # 예: 0712-KST
        out_path = f"data/warehouse/{date_part}-{run_part}.jsonl"
        existing = load_existing_urls(out_path)
        
        if url in existing:
            skipped += 1
            continue
        
        with open(out_path, "a", encoding="utf-8") as wf:
            wf.write(json.dumps(row, ensure_ascii=False) + "\n")
        
        appended += 1
    
    print(f"[INFO] warehouse append | appended={appended} skipped={skipped}")

if __name__ == "__main__":
    main()

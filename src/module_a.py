import os
import json
import time
import random
import re
import html
import requests
from bs4 import BeautifulSoup
from src.config import load_config
from src.timeutil import kst_date_str, kst_run_suffix


CFG = load_config()
NAVER_API = "https://openapi.naver.com/v1/search/news.json"

def naver_headers():
    return {
        "X-Naver-Client-Id": os.getenv("NAVER_CLIENT_ID", ""),
        "X-Naver-Client-Secret": os.getenv("NAVER_CLIENT_SECRET", ""),
        "User-Agent": "Mozilla/5.0"
    }

def http_get(url, params=None, headers=None, timeout=10, max_retry=3):
    for i in range(max_retry):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code == 429:
                wait = 30 + i * 15
                print(f"[WARN] 429 Too Many Requests, wait {wait}s")
                time.sleep(wait)
                continue  # 재시도
            if r.status_code >= 500:
                raise requests.HTTPError(f"5xx {r.status_code}")
            r.raise_for_status()
            return r
        except Exception:
            if i == max_retry - 1:
                raise
            time.sleep(1.2 * (2 ** i) + random.random())

def prefer_link(item):
    return item.get("originallink") or item.get("link") or ""

def fetch_naver_news(query, display=30, pages=2):
    items = []
    for p in range(pages):
        start = 1 + p * display
        if start > 1000: break
        params = {
            "query": query,
            "display": display,
            "start": start,
            "sort": "date"
        }
        r = http_get(NAVER_API, params=params, headers=naver_headers(), timeout=10, max_retry=3)
        data = r.json()
        batch = data.get("items", [])
        if not batch:
            break
        for it in batch:
            it["_query"] = query if query else "unknown"
        items.extend(batch)
        time.sleep(0.3)
    return items

def dedup_by_url(items):
    seen, out = set(),[]
    for it in items:
        url = prefer_link(it)
        if "_query" not in it or it["_query"] is None:
            it["_query"] = "unknown"
        if url and url not in seen:
            seen.add(url)
            out.append(it)
    return out

def expand_with_og(url):
    meta = {
        "url": url,
        "site_name": None,
        "title_og": None,
        "description_og": None,
        "published_time": None
    }
    try:
        r = http_get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10, max_retry=2)
        soup = BeautifulSoup(r.text, "lxml")
        def og(name):
            tag = soup.find("meta", property=name)
            return tag["content"].strip() if tag and tag.has_attr("content") else None
        meta["site_name"] = og("og:site_name")
        meta["title_og"] = og("og:title")
        meta["description_og"] = og("og:description")
        meta["published_time"] = og("article:published_time")
    except Exception:
        pass
    return meta

def clean_html(s):
    if not s:
        return s
    s = re.sub(r"<.+?>", " ", s)
    s = html.unescape(s)
    return s.strip()

def main():
    t0 = time.time()
    
    # dry_run: 환경 변수가 우선시되며, 없을 경우 CFG 값 사용. 기본값은 True
    dry_run = (os.getenv("DRY_RUN", str(CFG.get("dry_run", True))).lower() == "true")
    
    # queries: 반드시 비어있지 않은 리스트여야 함. 기본값은 ["unknown"]
    q_raw = CFG.get("queries", ["unknown"])
    queries = q_raw if isinstance(q_raw, list) and q_raw else ["unknown"]
    
    # display: 1~100 사이로 제한. 기본값 10
    display = int(CFG.get("per_query_display", 10))
    display = max(1, min(display, 100))
    
    # pages: 최소 1, dry_run=False일 경우 최소 1
    pages = int(CFG.get("pages", 1))
    pages = max(1, pages)
    if not dry_run:
        pages = max(1, pages)
    
    print(f"[INFO] queries={queries} dry_run={dry_run} display={display} pages={pages}")
    
    all_items =[]
    for q in queries:
        batch = fetch_naver_news(q, display=display, pages=pages)
        all_items.extend(batch)
        print(f"[INFO] query={q} | fetched={len(batch)} | total={len(all_items)}")
    clean_items = dedup_by_url(all_items)
    
    meta_list =[]
    for it in clean_items:
        url = prefer_link(it)
        meta = expand_with_og(url)
        meta["title"] = clean_html(it.get("title"))
        meta["description"] = clean_html(it.get("description"))
        meta["pubDate_raw"] = it.get("pubDate")
        meta["_query"] = it.get("_query")
        meta_list.append(meta)
        time.sleep(0.15)

    os.makedirs("data", exist_ok=True)
    ts_str = f"{kst_date_str()}-{kst_run_suffix()}"
    raw_path = f"data/news_clean_{ts_str}.json"
    meta_path = f"data/news_meta_{ts_str}.json"

    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(clean_items, f, ensure_ascii=False, indent=2)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_list, f, ensure_ascii=False, indent=2)

    print(f"[INFO] 저장 완료: {raw_path}, {meta_path} | 총 수집(중복 제거 후): {len(clean_items)} | 경과(초): {round(time.time()-t0,2)}")

    
if __name__ == "__main__":
    main()

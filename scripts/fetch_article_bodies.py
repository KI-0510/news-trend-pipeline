import os
import json
import glob
import time
import hashlib
import re
import unicodedata
from typing import List, Dict, Any
import trafilatura


def latest(globpat: str) -> str:
    files = sorted(glob.glob(globpat))
    return files[-1] if files else None


def clean_text(t: str) -> str:
    if not t:
        return ""
    t = unicodedata.normalize("NFKC", t)
    t = re.sub(r"<.+?>", " ", t)
    t = re.sub(r"\\s+", " ", t).strip()
    return t


def sha1(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()


def fetch_body(url: str, timeout: int = 12) -> str:
    if not url:
        return ""
    
    # 캐시 체크
    os.makedirs("data/article_cache", exist_ok=True)
    key = sha1(url)
    cache_path = os.path.join("data/article_cache", key + ".txt")
    
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = f.read().strip()
                if len(cached) >= 300:
                    return cached
        except Exception:
            pass
    
    # 다운로드 + 본문 추출
    try:
        downloaded = trafilatura.fetch_url(url, timeout=timeout, no_ssl=True)
        if downloaded:
            text = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_formatting=False,
                favor_recall=True,      # 최대한 많이 긁기
                with_metadata=False
            ) or ""
            text = clean_text(text)
        else:
            text = ""
    except Exception:
        text = ""
    
    # 캐시 저장(유의미 길이만)
    if len(text) >= 300:
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                f.write(text)
        except Exception:
            pass
    
    return text


def main() -> int:
    meta_path = latest("data/news_meta_*.json")
    if not meta_path:
        print("[ERROR] news_meta_* 파일을 찾지 못했습니다.")
        return 1
    
    with open(meta_path, "r", encoding="utf-8") as f:
        items: List[Dict[str, Any]] = json.load(f)
    
    updated = 0
    for it in items:
        url = it.get("url") or ""
        
        # 이미 body가 충분히 있으면 스킵
        body_now = it.get("body") or ""
        if len(body_now) >= 300:
            continue
        
        body = fetch_body(url)
        if len(body) >= 300:
            it["body"] = body
            # 모듈 B가 description을 쓰므로, 안전하게 동기화
            it["description"] = body
            updated += 1
        
        # 너무 빠른 연속 요청 방지(서버 배려)
        time.sleep(0.3)
    
    # 덮어쓰기 저장
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    
    print(f"[INFO] fetch_article_bodies | updated={updated} | file={os.path.basename(meta_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

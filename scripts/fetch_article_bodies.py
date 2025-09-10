import os
import json
import glob
import time
import hashlib
import re
import unicodedata
from typing import List, Dict, Any
import trafilatura
from trafilatura.settings import use_config

# 본문 최소 길이(환경변수로 조절 가능): 기본 120자
MIN_LEN = int(os.environ.get("BODY_MIN_LEN", "120"))

def latest(globpat: str) -> str:
    files = sorted(glob.glob(globpat))
    return files[-1] if files else None

def clean_text(t: str) -> str:
    if not t:
        return ""
    t = unicodedata.normalize("NFKC", t)
    t = re.sub(r"<.+?>", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def sha1(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()

def pick_url(it: Dict[str, Any]) -> str:
    """
    URL 선택 우선순위: url > canonical > link > origin_url
    네이버 도메인 우선 선택 + AMP/모바일 정규화 + n.news 표준화
    """
    cand = [it.get("url"), it.get("canonical"), it.get("link"), it.get("origin_url")]
    cand = [c.strip() for c in cand if c]
    if not cand:
        return ""
    def _naver_score(u: str) -> int:
        h = re.sub(r"^https?://([^/]+).*$", r"\1", (u or "").lower())
        return 2 if ("n.news.naver.com" in h or "news.naver.com" in h or "m.news.naver.com" in h) else 1
    cand.sort(key=_naver_score, reverse=True)
    u = cand[0]
    # AMP 경로 제거
    u = re.sub(r"/amp(/|$).*", "/", u)
    # 네이버 모바일 → n.news 표준화
    u = u.replace("m.news.naver.com", "n.news.naver.com")
    u = u.replace("news.naver.com", "n.news.naver.com")
    return u

def _make_config():
    cfg = use_config()
    cfg.set("DEFAULT", "user_agent",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
    cfg.set("DEFAULT", "timeout", "12")
    return cfg

def _strip_html_tags(s: str) -> str:
    # 간단 태그 제거(줄바꿈 의미는 공백으로 정리)
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    s = re.sub(r"</p\s*>", "\n", s, flags=re.I)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def extract_naver_body(html: str) -> str:
    """
    네이버 전용 본문 추출:
    - 신규: #dic_area
    - 구형: #articleBodyContents
    """
    if not html:
        return ""
    m = re.search(r'<div[^>]+id=["\']dic_area["\'][^>]*>(.*?)</div>', html, flags=re.I | re.S)
    if not m:
        m = re.search(r'<div[^>]+id=["\']articleBodyContents["\'][^>]*>(.*?)</div>', html, flags=re.I | re.S)
    txt = _strip_html_tags(m.group(1)) if m else ""
    # 꼬리문구/이메일 간단 정리
    txt = re.sub(r"\s*ⓒ.*?무단전재.*?$", "", txt, flags=re.I)
    txt = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def fetch_body(url: str, timeout: int = 12) -> str:
    if not url:
        return ""
    # 캐시
    os.makedirs("data/article_cache", exist_ok=True)
    key = sha1(url)
    cache_path = os.path.join("data/article_cache", key + ".txt")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = f.read().strip()
                if len(cached) >= MIN_LEN:
                    return cached
        except Exception:
            pass

    cfg = _make_config()
    text = ""

    # 1차: trafilatura.fetch_url → extract
    try:
        downloaded = trafilatura.fetch_url(url, config=cfg, timeout=timeout, no_ssl=True)
        if downloaded:
            text = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_formatting=False,
                favor_recall=True,
                with_metadata=False
            ) or ""
            text = clean_text(text)
    except Exception:
        text = ""

    # 2차: requests로 HTML 받아 네이버 전용 파서 → trafilatura 순서 폴백
    if len(text) < MIN_LEN:
        try:
            import requests
            headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                                     "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
            r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
            if r.ok and r.text:
                host = re.sub(r"^https?://([^/]+).*$", r"\1", (url or "").lower())
                t1 = extract_naver_body(r.text) if "naver.com" in host else ""
                t2 = trafilatura.extract(
                    r.text,
                    include_comments=False,
                    include_formatting=False,
                    favor_recall=True,
                    with_metadata=False
                ) or ""
                t1 = clean_text(t1)
                t2 = clean_text(t2)
                candidate = t1 if len(t1) >= len(t2) else t2
                if len(candidate) > len(text):
                    text = candidate
        except Exception:
            pass

    # 캐시 저장
    if len(text) >= MIN_LEN:
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
    tried = 0
    per_domain = {}

    for it in items:
        url = pick_url(it)
        if not url:
            continue

        # 이미 충분한 본문이 있으면 스킵
        body_now = (it.get("body") or "").strip()
        if len(body_now) >= MIN_LEN:
            continue

        tried += 1
        domain = re.sub(r"^https?://([^/]+)/?.*$", r"\1", url) if url else "-"
        per_domain.setdefault(domain, {"ok": 0, "fail": 0})

        body = fetch_body(url)
        if len(body) >= MIN_LEN:
            it["body"] = body
            it["description"] = body  # 모듈 B에서 description 참조 → 동기화
            updated += 1
            per_domain[domain]["ok"] += 1
        else:
            per_domain[domain]["fail"] += 1

        # 서버 배려(속도 조절)
        time.sleep(0.3)

    # 저장
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    # 로그
    print(f"[INFO] fetch_article_bodies | tried={tried} updated={updated} | file={os.path.basename(meta_path)}")
    if per_domain:
        stats = ", ".join(f"{d}: ok={v['ok']}, fail={v['fail']}" for d, v in list(per_domain.items())[:15])
        print("[DEBUG] per-domain:", stats)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())

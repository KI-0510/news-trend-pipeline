import os
import json
import glob
import time
import hashlib
import re
import unicodedata
from typing import List, Dict, Any, Tuple, Optional
import trafilatura
from trafilatura.settings import use_config
from src.utils import load_json, save_json, latest, clean_text
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
import threading



# 본문 최소 길이(환경변수로 조절). 기본 120자
MIN_LEN = int(os.environ.get("BODY_MIN_LEN", "120"))
MAX_WORKERS = int(os.environ.get("FETCH_MAX_WORKERS", "8"))
PER_DOMAIN_LIMIT = int(os.environ.get("FETCH_PER_DOMAIN", "3"))

_domain_locks: Dict[str, threading.Semaphore] = {}
_domain_lock_global = threading.Lock()


def sha1(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()


def pick_url(it: Dict[str, Any]) -> str:
    """
    URL 우선순위: url > canonical > link > origin_url
    네이버 도메인 우선 + AMP 제거 + n.news 표준화
    """
    cand = [it.get("url"), it.get("canonical"), it.get("link"), it.get("origin_url")]
    cand = [c.strip() for c in cand if c]
    if not cand:
        return ""

    def _naver_score(u: str) -> int:
        h = re.sub(r"^https?://([^/]+).", r"\1", (u or "").lower())
        return 2 if ("n.news.naver.com" in h or "news.naver.com" in h or "m.news.naver.com" in h) else 1
    
    cand.sort(key=_naver_score, reverse=True)
    u = cand[0]
    u = re.sub(r"/amp(/|).", "/", u)  # AMP 경로 제거
    u = u.replace("m.news.naver.com", "n.news.naver.com")
    u = u.replace("news.naver.com", "n.news.naver.com")
    return u


def _make_config():
    cfg = use_config()
    cfg.set("DEFAULT", "user_agent", "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36")
    cfg.set("DEFAULT", "timeout", "12")
    return cfg

def _domain_semaphore(url: str) -> threading.Semaphore:
    host = urlparse(url).netloc
    with _domain_lock_global:
        sem = _domain_locks.get(host)
        if sem is None:
            sem = _domain_locks[host] = threading.Semaphore(PER_DOMAIN_LIMIT)
        return sem



# -------- 정제 유틸(노이즈 컷) --------
def _remove_after_anchors(text: str) -> str:
    """ 특정 앵커가 나오면 그 이후는 노이즈로 간주하고 컷. """
    anchors = [
        r"^앱토 한마디\b",
        r"^BEST댓글\b",
        r"^댓글\b",
        r"^이 기사를 공유합니다\b",
        r"^제원표\b",
        r"^POINT\b",
        r"^관련기사\b",
        r"^기사원문\b"
    ]
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out = []
    for ln in lines:
        if any(re.search(p, ln.strip()) for p in anchors):
            break
        out.append(ln)
    return "\n".join(out)


def _strip_common_noise(text: str) -> str:
    """ 기자/저작권/이메일/URL/가격/캡션/UX 문구 제거(보수적). """
    t = text
    t = re.sub(r"\[[^[]\n]{0,80}기자]", " ", t)  # [매체 = ... 기자]
    t = re.sub(r"[-–—]\s기자명?\s[^\s]기자", " ", t)  # - 기자명 ... 기자
    t = re.sub(r"\b[\w.-]+@[\w.-]+.\w+\b", " ", t)  # 이메일
    t = re.sub(r"(무단전재\s및\s재배포\s금지|Copyright\s*©[^,\n]+)", " ", t, flags=re.I)
    t = re.sub(r"(BEST댓글|댓글삭제|댓글수정|이 기사를 공유합니다)", " ", t)  # UX
    t = re.sub(r"https?://\S+", " ", t)  # URL
    t = re.sub(r"\b[0-9]{1,3}(?:,[0-9]{3})+(?:\s원(?:부터)?)?\b", " ", t)  # 가격
    t = t.replace("▲", " ")
    t = re.sub(r"^\s사진\s*=\s*.*$", " ", t, flags=re.M)
    t = re.sub(r"[│•▪◆■※☆★▷▶▸▹◀◁◾◼︎]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _dedup_sentences(text: str) -> str:
    """ 완전 동일 문장/문단 중복 제거(순서 유지). """
    parts = re.split(r"(?<=[.!?다])\s+", text)
    seen, out = set(), []
    for s in parts:
        s = s.strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return " ".join(out)


def sanitize_article(text: str) -> str:
    """
    전체 정제 파이프라인: 앵커 컷 → 공통 노이즈 제거 → '제원/스펙' 표 이전 컷 → 중복 제거.
    """
    if not text:
        return ""
    t = _remove_after_anchors(text)
    t = _strip_common_noise(t)
    m = re.search(r"(제원|스펙|사양)\s*표", t)
    if m:
        t = t[:m.start()].strip()
    t = _dedup_sentences(t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# -------- 네이버 전용 파서 --------
def _strip_html_tags(s: str) -> str:
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    s = re.sub(r"</p\s*>", "\n", s, flags=re.I)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_naver_body(html: str) -> str:
    if not html:
        return ""
    m = re.search(r'<div[^>]+id=["\']dic_area["\'][^>]>(.*?)', html, flags=re.I | re.S)
    if not m:
        m = re.search(r'<div[^>]+id=["\']articleBodyContents["\'][^>]>(.*?)', html, flags=re.I | re.S)
    
    txt = _strip_html_tags(m.group(1)) if m else ""
    txt = re.sub(r"\s*ⓒ.?무단전재.?$", "", txt, flags=re.I)
    txt = re.sub(r"\b[\w.-]+@[\w.-]+.\w+\b", "", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


# -------- 본문 수집 --------
def fetch_body(url: str, timeout: int = 12) -> Tuple[str, str]:
    """
    반환: (sanitized_text, raw_text)
    - sanitized_text: 정제된 본문(캐시 저장 기준)
    - raw_text: 정제 전 추출결과(신호 추출용)
    """
    if not url:
        return "", ""

    os.makedirs("data/article_cache", exist_ok=True)
    key = sha1(url)
    cache_path = os.path.join("data/article_cache", key + ".txt")
    cache_raw_path = os.path.join("data/article_cache", key + ".raw.txt")

    # 캐시 히트(정제본 우선)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = f.read().strip()
                if len(cached) >= MIN_LEN:
                    # raw 캐시가 있으면 같이 읽어 반환
                    raw = ""
                    if os.path.exists(cache_raw_path):
                        try:
                            with open(cache_raw_path, "r", encoding="utf-8") as rf:
                                raw = rf.read().strip()
                        except Exception:
                            pass
                    return cached, raw
        except Exception:
            pass

    cfg = _make_config()
    text_raw = ""

    # 1차: trafilatura.fetch_url → extract
    try:
        downloaded = trafilatura.fetch_url(url, config=cfg, timeout=timeout, no_ssl=True)
        if downloaded:
            t = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_formatting=False,
                favor_recall=True,
                with_metadata=False
            ) or ""
            text_raw = clean_text(t)
    except Exception:
        pass

    # 2차: requests로 HTML → 네이버 전용 파서 → trafilatura(HTML)
    if len(text_raw) < MIN_LEN:
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
                cand = t1 if len(t1) >= len(t2) else t2
                if len(cand) > len(text_raw):
                    text_raw = cand
        except Exception:
            pass

    # 3) 정제는 항상 최종에서 1회 적용
    text_san = sanitize_article(text_raw)

    # 캐시 저장(정제본 + raw)
    if len(text_san) >= MIN_LEN:
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                f.write(text_san)
        except Exception:
            pass
        # raw도 별도 캐시(신호 추출용)
        try:
            with open(cache_raw_path, "w", encoding="utf-8") as rf:
                rf.write(text_raw)
        except Exception:
            pass

    return text_san, text_raw

def _process_one(it: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    개별 기사 처리.
    성공 시 (업데이트된 item, domain) 반환, 실패/스킵 시 (None, domain) 반환.
    """
    url = pick_url(it)
    domain = re.sub(r"^https?://([^/]+)/?.*$", r"\1", url) if url else "-"
    if not url:
        return None, domain
    body_now = (it.get("body") or "").strip()
    if len(body_now) >= MIN_LEN:
        # raw_body/description_short 보강만 수행
        if not it.get("raw_body"):
            it["raw_body"] = it.get("raw_body") or ""
        if not it.get("description_short"):
            it["description_short"] = make_description_short(body_now)
        return it, domain
    try:
        sem = _domain_semaphore(url)
        with sem:
            body_san, body_raw = fetch_body(url)
        if len(body_san) >= MIN_LEN:
            it["raw_body"] = body_raw or ""
            it["body"] = body_san
            it["description"] = body_san
            it["description_short"] = make_description_short(body_san)
            return it, domain
        return None, domain
    except Exception:
        return None, domain



def make_description_short(text: str, target_min=400, target_max=600) -> str:
    """
    본문 첫 문단/문장 기준으로 400~600자 요약 생성(문장 경계에서 자르기 시도).
    """
    if not text:
        return ""
    txt = text.strip()
    # 문단 기준
    para = txt.split("\n")[0].strip()
    base = para if len(para) >= target_min else txt
    
    if len(base) <= target_max:
        return base
    
    # 문장 경계에서 컷
    cut = base[:target_max]
    m = re.search(r"[.!?다]\s", cut[::-1])  # 뒤에서부터 첫 종결부호 찾기
    
    if m:
        idx = len(cut) - m.start()
        return cut[:idx].strip()
    
    return cut.strip()


def main() -> int:
    meta_path = latest("data/news_meta_*.json")
    if not meta_path:
        print("[ERROR] news_meta_ 파일을 찾지 못했습니다.")
        return 1

    with open(meta_path, "r", encoding="utf-8") as f:
        items: List[Dict[str, Any]] = json.load(f)

    tried, updated = 0, 0
    per_domain: Dict[str, Dict[str, int]] = {}

    # 대상 인덱스 수집(이미 본문 있는 항목은 그대로 두고 보강만)
    indices = list(range(len(items)))

    # 병렬 처리
    results: Dict[int, Tuple[Optional[Dict[str, Any]], Optional[str]]] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        fut_map = {ex.submit(_process_one, items[i]): i for i in indices}
        for fut in as_completed(fut_map):
            i = fut_map[fut]
            try:
                res_item, domain = fut.result()
            except Exception:
                res_item, domain = None, "-"
            tried += 1
            if domain not in per_domain:
                per_domain[domain] = {"ok": 0, "fail": 0}
            if res_item is not None:
                items[i] = res_item
                updated += 1
                per_domain[domain]["ok"] += 1
            else:
                per_domain[domain]["fail"] += 1

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(f"[INFO] fetch_article_bodies | tried={tried} updated={updated} | file={os.path.basename(meta_path)} | workers={MAX_WORKERS} per_domain={PER_DOMAIN_LIMIT}")
    if per_domain:
        stats = ", ".join(f"{d}: ok={v['ok']}, fail={v['fail']}" for d, v in list(per_domain.items())[:15])
        print("[DEBUG] per-domain:", stats)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

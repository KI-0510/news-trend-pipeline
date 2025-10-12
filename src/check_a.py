import json
import glob
import sys
from typing import NoReturn  # NoReturn 추가 임포트
from src.utils import load_json, save_json, latest

def fail(msg: str) -> NoReturn:
    """에러 메시지를 출력하고 프로그램을 종료합니다."""
    print(f"[ERROR] {msg}")
    sys.exit(1)

def main():
    meta_path = latest("data/news_meta_*.json")
    if not meta_path:
        fail("data/news_meta_*.json 없음")

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            items = json.load(f)
    except json.JSONDecodeError:
        fail("JSON 파싱 실패")
    except FileNotFoundError:
        fail("파일을 찾을 수 없음")

    if not isinstance(items, list) or len(items) == 0:
        fail("메타 리스트가 비어 있음")

    # 필수 필드: url, title, _query (module_a.py에서 저장하는 필드명)
    required = ["url", "title", "_query"]
    bad_indices = []

    for idx, item in enumerate(items):
        # 1) 필수 필드 존재 여부 검증
        for k in required:
            if k not in item:
                bad_indices.append((idx, f"필드 누락: {k}"))
                break  # 필드가 하나라도 없으면 더 이상 검증하지 않음

        else:  # 모든 필수 필드가 있을 때만 추가 검증 수행
            url = (item.get("url") or "").strip()
            title = (item.get("title") or "").strip()
            query = (item.get("_query") or "").strip()

            # 2) URL 형식 검증
            if not url.startswith(("http://", "https://")):
                bad_indices.append((idx, f"URL 형식 이상: {url[:60]}..."))
                continue

            # 3) 제목 최소 길이 검증
            if len(title) < 2:
                bad_indices.append((idx, "제목 길이 < 2"))
                continue

            # 4) 쿼리 존재 여부 검증
            if not query:
                bad_indices.append((idx, "쿼리(검색어) 비어 있음"))
                continue

    if bad_indices:
        # 문제 있는 첫 5개만 출력하고 실패 처리
        for i, (idx, reason) in enumerate(bad_indices[:5], 1):
            print(f"[ERROR] {i}. index={idx} | 이유={reason}")
        fail(f"유효성 검사 실패: 총 {len(bad_indices)}건")

    # 샘플 정보 출력 (로그 가독성)
    sample = items[0] if items else {}
    print(f"[INFO] Check A OK | 파일: {meta_path} | 건수: {len(items)}")
    print(f"[INFO] sample_site={sample.get('site_name') or 'N/A'} | "
          f"sample_time={sample.get('published_time') or 'N/A'} | "
          f"sample_query={sample.get('_query') or 'N/A'}")

if __name__ == "__main__":
    main()

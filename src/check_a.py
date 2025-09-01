import json
import glob
import sys

def latest(path_glob):
    paths = sorted(glob.glob(path_glob))
    return paths[-1] if paths else None

def main():
    meta_path = latest("data/news_meta_*.json")
    if not meta_path:
        print("[ERROR] data/news_meta_*.json 없음")
        sys.exit(1)

    with open(meta_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    if not isinstance(items, list) or len(items) == 0:
        print("[ERROR] 메타 리스트 비었음")
        sys.exit(1)

    required = ["url", "title", "_query"]

    for idx, item in enumerate(items):
        # 필수 필드 검증
        for k in required:
            if k not in item:
                print(f"[ERROR] 필드 누락: {k} (index={idx})")
                sys.exit(1)

        # URL 검증
        url = item.get("url")
        if not url or not url.startswith(("http://", "https://")):
            print(f"[ERROR] 유효하지 않은 URL (index={idx}): {url}")
            sys.exit(1)

        # 제목 검증
        title = item.get("title", "")
        if len(title.strip()) < 2:
            print(f"[ERROR] 제목이 너무 짧음 (index={idx}): {title}")
            sys.exit(1)

    print(f"[INFO] Check A OK | 파일: {meta_path} | 건수: {len(items)}")

if __name__ == "__main__":
    main()

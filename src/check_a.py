import json, glob, sys

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
    sample = items[0]
    required = ["url", "title"]
    for k in required:
        if k not in sample:
            print(f"[ERROR] 필드 누락: {k}")
            sys.exit(1)
    print(f"[INFO] Check A OK | 파일: {meta_path} | 건수: {len(items)}")

if name == "main":
    main()

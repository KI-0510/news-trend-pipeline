import json
import sys


def fail(msg):
    print(f"[ERROR] {msg}")
    sys.exit(1)


def main():
    path = "outputs/keywords.json"
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        fail("outputs/keywords.json 없음 (모듈 B 실행 실패)")
    except json.JSONDecodeError:
        fail("keywords.json 파싱 실패")

    kws = data.get("keywords", [])
    
    if not isinstance(kws, list) or len(kws) < 10:
        fail("키워드 개수 부족(<10)")

    for i, k in enumerate(kws[:10], 1):
        if "keyword" not in k or "score" not in k:
            fail(f"필드 누락(keyword/score) at {i}")
        if not isinstance(k["score"], (int, float)) or k["score"] <= 0:
            fail(f"score가 유효하지 않음 at {i}")

    print(f"[INFO] Check B OK | 키워드 수={len(kws)}")


if __name__ == "__main__":
    main()

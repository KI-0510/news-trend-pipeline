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

    # 고유 키워드 검증
    uniq = {k["keyword"] for k in kws if "keyword" in k}
    if len(uniq) < 10:
        fail("고유 키워드 개수 < 10")

    # 키워드 길이 검증
    for i, k in enumerate(kws, 1):
        if "keyword" not in k or "score" not in k:
            fail(f"필드 누락(keyword/score) at {i}")
        keyword = k["keyword"]
        if len(keyword) < 2:
            fail(f"키워드 길이 < 2 at {i}")
        if not isinstance(k["score"], (int, float)) or k["score"] <= 0:
            fail(f"score가 유효하지 않음 at {i}")

    # 점수 단조 감소 검증 (상위 5개)
    for i in range(min(5, len(kws) - 1)):
        if kws[i]["score"] < kws[i + 1]["score"]:
            print(f"\[WARN\] 상위 점수 정렬 이상 (pos {i + 1})")

    print(f"[INFO] Check B OK | 키워드 수={len(kws)} | 고유 키워드={len(uniq)}")


if __name__ == "__main__":
    main()

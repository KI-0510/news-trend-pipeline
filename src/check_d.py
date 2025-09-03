import json
import sys


def fail(msg):
    print(f"[ERROR] {msg}")
    sys.exit(1)


def main():
    path = "outputs/biz_opportunities.json"
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        fail("biz_opportunities.json 로드 실패")

    ideas = data.get("ideas")
    if not isinstance(ideas, list) or len(ideas) < 3:
        fail("아이디어 최소 3개 미만")

    required = [
        "idea",
        "problem",
        "target_customer",
        "value_prop",
        "solution",
        "poc_plan",
        "risks",
        "roadmap_3m",
        "metrics",
        "priority_score",
    ]

    for idx, it in enumerate(ideas[:5], 1):  # 앞쪽 5건만 샘플 검증
        for k in required:
            if k not in it:
                fail(f"필드 누락: {k} (index={idx})")

    print(f"[INFO] Check D OK | ideas={len(ideas)}")


if __name__ == "__main__":
    main()

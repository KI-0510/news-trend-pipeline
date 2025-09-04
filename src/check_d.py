import json
import sys

def warn(msg):
    print(f"[WARN] {msg}")

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
    if ideas is None or not isinstance(ideas, list):
        fail("ideas 필드가 누락되었거나 형식이 잘못됨")

    n = len(ideas)
    if n == 0:
        warn("아이디어가 0개입니다(폴백 비활성화 설정).")
    elif n < 3:
        warn(f"아이디어가 적습니다({n}개).")

    # 필드(축소 버전)
    required = [
        "idea",
        "problem",
        "target_customer",
        "value_prop",
        "solution",
        "risks",
        "priority_score",
    ]

    # 앞쪽 최대 5건 샘플 검증(있을 때만)
    for idx, it in enumerate(ideas[:5], 1):
        if not isinstance(it, dict):
            fail(f"아이디어 형식 오류(index={idx})")
        for k in required:
            if k not in it:
                fail(f"필드 누락: {k} (index={idx})")

    print(f"[INFO] Check D OK | ideas={n}")

if __name__ == "__main__":
    main()

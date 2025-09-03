import os
import sys

def fail(msg):
    print(f"[ERROR] {msg}")
    sys.exit(1)

def main():
    if not os.path.exists("outputs/report.md"):
        fail("outputs/report.md 없음")

    with open("outputs/report.md", "r", encoding="utf-8") as f:
        txt = f.read()

    required = [
        "Executive Summary",
        "Key Metrics",
        "Top Keywords",
        "Topics",
        "Trend",
        "Insights",
        "Opportunities",
        "Appendix"
    ]

    for h in required:
        if h not in txt:
            fail(f"report.md 섹션 누락: {h}")

    print("[INFO] Check E OK | report.md 생성 확인")

if __name__ == "__main__":
    main()

import os
import sys

def fail(msg):
    print(f"[ERROR] {msg}")
    sys.exit(1)

def main():
    path = "outputs/report.md"
    if not os.path.exists(path):
        fail(f"{path} 없음")
    print("[INFO] Check F OK | report.md 존재 확인")

if __name__ == "__main__":
    main()
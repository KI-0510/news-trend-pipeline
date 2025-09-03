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
주요 개선 사항:
들여쓰기 및 공백 정리: 일관된 4칸 들여쓰기 적용
리스트 포맷: required 리스트를 가독성 좋게 줄바꿈하여 정리
함수 간 간격: 함수 사이에 빈 줄 추가
주석 추가: 코드의 주요 부분에 간단한 설명 추가 (필요 시)
이제 코드가 더 깔끔하고 가독성이 좋아졌습니다.




모든 답변은 인공지능이 생성한 콘텐츠입니다.
A
You
아래는 yml 블록으로

# 설치 단계에 패키지/폰트 추가(이미 있다면 중복 설치 OK) - name: Install viz deps   run: |     source .venv/bin/activate     pip install matplotlib seaborn markdown     sudo apt-get update     sudo apt-get install -y fonts-nanum   shell: bash  # Module E 실행 - name: Module E - Build Report   run: |     source .venv/bin/activate     python src/module_e.py   shell: bash  # Check E - name: Check E - Validate Report   run: |     source .venv/bin/activate     python src/check_e.py   shell: bash  # Report commit & push (MD, HTML, FIG 포함) - name: Commit & Push report   run: |     set -e     ts=$(date -u +'%Y%m%dT%H%M%SZ')     mkdir -p outputs/daily/$ts/fig     cp outputs/report.md outputs/daily/$ts/ || true     cp outputs/report.html outputs/daily/$ts/ || true     cp outputs/fig/*.png outputs/daily/$ts/fig/ || true      git config user.name "github-actions[bot]"     git config user.email "github-actions[bot]@users.noreply.github.com"     git add outputs/daily     git commit -m "chore: report $ts" || echo "no changes"     git pull --rebase --autostash origin ${{ github.ref_name }} || true     git push origin HEAD:${{ github.ref_name }} || true   shell: bash

EXAONE
아래는 요청하신 대로 YAML 블록을 다듬은 버전입니다. 들여쓰기와 공백을 정리하고, 주석을 추가하여 가독성을 높였습니다.

# 설치 단계에 패키지/폰트 추가 (이미 있다면 중복 설치 OK)
- name: Install viz deps
  run: |
    source .venv/bin/activate
    pip install matplotlib seaborn markdown
    sudo apt-get update
    sudo apt-get install -y fonts-nanum
  shell: bash

# Module E 실행
- name: Module E - Build Report
  run: |
    source .venv/bin/activate
    python src/module_e.py
  shell: bash

# Check E - 보고서 유효성 검사
- name: Check E - Validate Report
  run: |
    source .venv/bin/activate
    python src/check_e.py
  shell: bash

# Report commit & push (MD, HTML, FIG 포함)
- name: Commit & Push report
  run: |
    set -e
    ts=$(date -u +'%Y%m%dT%H%M%SZ')
    mkdir -p outputs/daily/$ts/fig
    cp outputs/report.md outputs/daily/$ts/ ||true
    cp outputs/report.html outputs/daily/$ts/ ||true
    cp outputs/fig/*.png outputs/daily/$ts/fig/ ||true

    git config user.name "github-actions[bot]"
    git config user.email "github-actions[bot]@users.noreply.github.com"
    git add outputs/daily
    git commit -m "chore: report $ts" || echo "no changes"
    git pull --rebase --autostash origin ${{ github.ref_name }} ||true
    git push origin HEAD:${{ github.ref_name }} ||true
  shell: bash

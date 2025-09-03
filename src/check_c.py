import json
import sys


def fail(msg):
    print(f"[ERROR] {msg}")
    sys.exit(1)


def main():
    # topics
    try:
        topics = json.load(open("outputs/topics.json", "r", encoding="utf-8"))
    except Exception:
        fail("outputs/topics.json 로드 실패")
    
    if "topics" not in topics or not isinstance(topics["topics"], list):
        fail("topics.json 구조 오류(topics)")
    
    # timeseries
    try:
        ts = json.load(open("outputs/trend_timeseries.json", "r", encoding="utf-8"))
    except Exception:
        fail("outputs/trend_timeseries.json 로드 실패")
    
    if "daily" not in ts or not isinstance(ts["daily"], list):
        fail("trend_timeseries.json 구조 오류(daily)")
    
    # insights
    try:
        ins = json.load(open("outputs/trend_insights.json", "r", encoding="utf-8"))
    except Exception:
        fail("outputs/trend_insights.json 로드 실패")
    
    required = ["summary", "top_topics", "evidence"]
    for k in required:
        if k not in ins:
            fail(f"trend_insights.json 필드 누락: {k}")
    
    print(f"[INFO] Check C OK | topics={len(topics['topics'])} | ts_days={len(ts['daily'])}")


if __name__ == "__main__":
    main()
    

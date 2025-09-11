# -*- coding: utf-8 -*-
import os
import json
import re
import glob
import unicodedata
import time
import datetime
from typing import List, Dict, Any, Tuple
from email.utils import parsedate_to_datetime
from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

try:
    from config import load_config, llm_config
except Exception:
    def load_config(): return {}
    def llm_config(cfg: dict) -> dict:
        llm = cfg.get("llm") or {}
        return {"model": llm.get("model","gemini-1.5-flash"),
                "max_output_tokens": int(llm.get("max_output_tokens",2048)),
                "temperature": float(llm.get("temperature",0.3))}

CFG = load_config()
LLM = llm_config(CFG)

def latest(globpat: str):
    files = sorted(glob.glob(globpat))
    return files[-1] if files else None

def to_kst_date_str(s: str) -> str:
    try:
        s2 = (s or "").replace("Z","+00:00")
        dt = datetime.datetime.fromisoformat(s2)
        return dt.date().strftime("%Y-%m-%d")
    except Exception:
        try:
            dt = parsedate_to_datetime(s or "")
            return dt.date().strftime("%Y-%m-%d")
        except Exception:
            return datetime.date.today().strftime("%Y-%m-%d")

def clean_text(t: str) -> str:
    if not t: return ""
    t = re.sub(r"<.+?>", " ", t)
    t = unicodedata.normalize("NFKC", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def load_today_meta() -> List[str]:
    meta_path = latest("data/news_meta_*.json")
    if not meta_path:
        return []
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            items = json.load(f)
    except Exception:
        return []
    docs = []
    for it in items:
        title = clean_text((it.get("title") or it.get("title_og") or "").strip())
        body  = clean_text((it.get("body") or it.get("description") or it.get("description_og") or "").strip())
        doc = (title + " " + body).strip()
        if doc:
            docs.append(doc)
    return docs

EN_STOP = {
    "the","and","to","of","in","for","on","with","at","by","from","as","is","are","be","it",
    "that","this","an","a","or","if","we","you","they","he","she","was","were","been","than",
    "into","about","over","under","per","via"
}
KO_FUNC = {"하다","있다","되다","통해","이번","대한","것으로","밝혔다","다양한","함께","현재","기자","대표","회장"}

def build_topics(docs: List[str], k_candidates=(7,8,9,10,11), max_features=8000, min_df=4, topn=10) -> Dict[str, Any]:
    if not docs:
        return {"topics": []}
    vec = CountVectorizer(
        ngram_range=(1,2),
        max_features=max_features,
        min_df=min_df,
        token_pattern=r"[가-힣A-Za-z0-9_]{2,}",
        stop_words=list(EN_STOP)
    )
    X = vec.fit_transform(docs)
    vocab = vec.get_feature_names_out()
    if X.shape[1] == 0:
        return {"topics": []}

    def topic_words(lda, n_top=topn):
        comps = lda.components_
        topics = []
        for tid, comp in enumerate(comps):
            idx = comp.argsort()[-n_top:][::-1]
            words = [vocab[i] for i in idx]
            topics.append((tid, words))
        return topics

    def is_bad_topic(words):
        bad = 0
        for w in words:
            base = w.split()[0] if " " in w else w
            if base in KO_FUNC or base.lower() in EN_STOP:
                bad += 1
        return (bad / max(1, len(words))) >= 0.3  # 30%로 강화

    best_score = -1.0
    best_topics = None

    for k in k_candidates:
        lda = LatentDirichletAllocation(n_components=k, learning_method="batch", random_state=42, max_iter=15)
        _ = lda.fit_transform(X)
        ts = topic_words(lda, n_top=topn)
        good = sum(1 for _, ws in ts if not is_bad_topic(ws))
        score = good / float(k)
        if score > best_score:
            best_score = score
            best_topics = ts

    topics_obj = {"topics": []}
    if not best_topics:
        return topics_obj

    for tid, words in best_topics:
        filtered = [w for w in words if (w.split()[0] if " " in w else w) not in KO_FUNC and w.lower() not in EN_STOP]
        if not filtered:
            filtered = words
        topics_obj["topics"].append({
            "topic_id": int(tid),
            "top_words": [{"word": w} for w in filtered[:topn]]
        })
    return topics_obj

def main():
    os.makedirs("outputs", exist_ok=True)
    docs = load_today_meta()
    topics_obj = build_topics(docs, k_candidates=(7,8,9,10,11), max_features=8000, min_df=4, topn=10)
    with open("outputs/topics.json", "w", encoding="utf-8") as f:
        json.dump(topics_obj, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Module C done | topics={len(topics_obj.get('topics', []))} | docs={len(docs)}")

if __name__ == "__main__":
    main()

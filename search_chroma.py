import math
import os
import re
import time
from collections import Counter

import chromadb
import jieba
import requests

TOP_K = 5
FILE_K = 5
COARSE_K = 20
RRF_K = 60
K1 = 1.2
B = 0.75

CODE_PATTERN = r"[a-zA-Z0-9_][a-zA-Z0-9_./\-]*"

STOPWORDS = set("""
是 的 了 吗 呢 吧 啊 什么 怎么 怎么样 如何 为什么 为啥 哪些 哪个 哪里
这个 那个 一个 一下 可以 能 会 要 想 在 有 和 与 或 就 都 也 还 但 而且 以及
我 你 他 她 它 我们 你们 它们 这 那 么 做 用 到 把 被 给 对 从 为 以 之 其
""".split())


def _sf_key():
    v = os.environ.get("SILICONFLOW_API_KEY")
    if v:
        return v.strip()
    try:
        with open("key_sf.txt",encoding="utf-8")as f:
            return f.read().strip()
    except FileNotFoundError:
        raise RuntimeError(
            "找不到 SILICONFLOW_API_KEY。\n"
            "  本地：建个 key_sf.txt\n"
            "  服务器：docker run 时带 --env-file .env"
        )

sf_key = _sf_key()

def embed(texts):
    last_err = None
    for attempt in range(3):
        try:
            r = requests.post(
                "https://api.siliconflow.cn/v1/embeddings",
                headers=({"Authorization":f"Bearer {sf_key}","Content-Type":"application/json"}),
                json={"model":"BAAI/bge-m3","input":texts},
                timeout=60,
            )
            if r.status_code != 200:
                raise RuntimeError(f"embedding 接口返回{r.status_code}:{r.text[:200]}")
            d = r.json()
            if d.get("data") is None:
                raise RuntimeError(f"embedding 出错:{d}")
            out = [None] * len(texts)
            for item in d["data"]:
                out[item["index"]] = item["embedding"]
            return out
        except Exception as e:
            last_err = e
            if attempt < 2:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"embedding 连续失败 3 次：{last_err}")


client = chromadb.PersistentClient(path="./chroma_db")
col = client.get_collection("notes",embedding_function=None)


def cut_words(text):
    codes = re.findall(CODE_PATTERN,text)
    rest = re.sub(CODE_PATTERN," ",text)
    words = [w for w in jieba.lcut(rest) if re.search(r"\w",w) and w not in STOPWORDS]
    words.extend(codes)
    return words

_all = col.get()
DOC_IDS = _all["ids"]
DOCS = _all["documents"]
METAS = _all["metadatas"]
ID_TO_IDX = {i: n for n, i in enumerate(DOC_IDS)}

DOC_WORDS =[cut_words(d) for d in DOCS]
N_DOCS = len(DOCS)
AVG_LEN = sum(len(w) for w in DOC_WORDS) / N_DOCS


DF = Counter()
for _ws in DOC_WORDS:
    for _w in set(_ws):
        DF[_w] += 1

def vector_rank(question, top_k=COARSE_K):
    qv = embed([question])[0]
    res = col.query(query_embeddings=[qv], n_results=top_k)
    return [(ID_TO_IDX[res["ids"][0][n]],1 - res["distances"][0][n])
            for n in range(len(res["ids"][0]))]

def bm25_rank(question, top_k=COARSE_K):
    q_words = cut_words(question)
    scores = []
    for i, doc_words in enumerate(DOC_WORDS):
        dl = len(doc_words)
        s = 0.0
        for w in q_words:
            f = doc_words.count(w)
            if f == 0:
                continue
            n = DF.get(w, 0)
            idf = math.log((N_DOCS-n+0.5)/(n+0.5)+1)
            tf = f*(K1+1)/(f+K1*(1-B+B*dl/AVG_LEN))
            s += idf * tf
        scores.append((s, i))
    scores.sort(reverse=True)
    return [(i, s) for s, i in scores[:top_k]]


def rrf_fuse(rank_lists, top_k = TOP_K, k=RRF_K):
    total = {}
    for ranks in rank_lists:
        for rank, (idx,_score) in enumerate(ranks,1):
            total[idx] = total.get(idx, 0.0) + 1.0/(k+rank)
    pairs = sorted(total.items(), key=lambda kv: -kv[1])
    return [(idx,s)for idx,s in pairs[:top_k]]


def search_notes(question,top_k=TOP_K, file_k = FILE_K):
    fused = rrf_fuse([
        vector_rank(question, COARSE_K),
        bm25_rank(question, COARSE_K),
    ],top_k=COARSE_K)
    best = {}
    for idx, s in fused:
        src = METAS[idx]["source"]
        if src not in best or s >best[src][1]:
            best[src] = (idx, s)
    ranked = sorted(best.items(),key=lambda kv:-kv[1][1])[:min(file_k,top_k)]
    return [{
        "text":DOCS[idx],
        "source":src,
        "score":s,
    }for src, (idx, s)in ranked]
















    



def build_context(hits):
    text =""
    sources = []
    for h in hits:
        text += f"【来源：{h['source']}】\n{h['text']}\n\n"
        if h["source"] not in sources:
            sources.append(h["source"])
    return text, sources

if __name__ == "__main__":
    for q in ["变量是什么", "RAG 是什么"]:
        print("=" * 60)
        print(f"  问：{q}")
        print("=" * 60)
        for h in search_notes(q):
            print(f"    {h['score']:.5f}  【{h['source']}】")
            print(f"        {h['text'][:45]!r}")
        print()
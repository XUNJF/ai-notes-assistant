import requests
import chromadb
import os

def _sf_key():
    v = os.environ.get("SILICONFLOW_API_KEY")
    if v:
        return v.strip()
    try:
        with open("key_sf.txt", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        raise RuntimeError(
            "找不到 SILICONFLOW_API_KEY。\n"
            "  本地：建个 key_sf.txt\n"
            "  服务器：docker run 时带 --env-file .env"
        )

sf_key = _sf_key()
TOP_K = 5


def embed(texts):    # 把一批文字转成向量
    r = requests.post(
        "https://api.siliconflow.cn/v1/embeddings",
        headers={"Authorization": f"Bearer {sf_key}", "Content-Type": "application/json"},
        json={"model": "BAAI/bge-m3", "input": texts},
        timeout=60,
    )
    d = r.json()
    if d.get("data") is None:
        raise RuntimeError(f"embedding出错：{d}")

    out = [None] * len(texts)
    for item in d["data"]:
        out[item["index"]] = item["embedding"]
    return out


client = chromadb.PersistentClient(path="./chroma_db")
col = client.get_collection("notes", embedding_function=None)


def search_notes(question, top_k=TOP_K, where=None):    # 查笔记，返回一个列表
    """返回 [{"text": 原文, "source": 来源, "score": 相似度}, ...]"""
    qv = embed([question])[0]

    kwargs = {"query_embeddings": [qv], "n_results": top_k}
    if where:
        kwargs["where"] = where

    res = col.query(**kwargs)

    hits = []
    for i in range(len(res["ids"][0])):
        hits.append({
            "text": res["documents"][0][i],
            "source": res["metadatas"][0][i]["source"],
            "score": 1 - res["distances"][0][i],      # cosine 空间下：1-距离=相似度
        })
    return hits


def build_context(hits):    # 把查到的几块拼成给模型看的一段资料
    """返回 (拼好的资料文字, 去重后的来源列表)"""
    text = ""
    sources = []
    for h in hits:
        text += f"【来源：{h['source']}】\n{h['text']}\n\n"
        if h["source"] not in sources:
            sources.append(h["source"])
    return text, sources


if __name__ == "__main__":
    META_DOCS = ["成都JD分析-20份.md", "杭州JD分析-22份.md", "21周学习计划.md",
                 "零基础到AI应用开发-全职学习路线.md"]

    for q in ["RAG 是什么？", "怎么让页面自动滚到底部", "top_k 是什么"]:
        print("=" * 62)
        print(f"  问：{q}")
        print("=" * 62)

        hits = search_notes(q, where={"source": {"$nin": META_DOCS}})
        for h in hits:
            print(f"    {h['score']:.4f}  【{h['source']}】")
            print(f"        {h['text'][:50]!r}")

        text, sources = build_context(hits)
        print(f"    来源汇总：{sources}")
        print()

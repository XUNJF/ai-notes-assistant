import requests
import chromadb

sf_key = open("key_sf.txt", encoding="utf-8").read().strip()
TOP_K = 5


def embed(texts):    # 一次把一批文字转成向量
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


def search(question, top_k=TOP_K, where=None):    # 查笔记，返回 (拼好的文字, 来源列表)
    qv = embed([question])[0]

    kwargs = {"query_embeddings": [qv], "n_results": top_k}
    if where:
        kwargs["where"] = where

    res = col.query(**kwargs)

    text = ""
    sources = []
    for i in range(len(res["ids"][0])):
        md = res["metadatas"][0][i]
        doc = res["documents"][0][i]
        sim = 1 - res["distances"][0][i]
        text += f"【来源：{md['source']}】\n{doc}\n\n"
        if md["source"] not in sources:
            sources.append(md["source"])
        print(f"    {sim:.4f}  {md['source']:<22} {doc[:30]!r}")

    return text, sources


if __name__ == "__main__":
    for q in ["RAG 是什么？", "怎么让页面自动滚到底部", "top_k 是什么"]:
        print(f"\n问：{q}")
        text, sources = search(q)
        print(f"    来源汇总：{sources}")

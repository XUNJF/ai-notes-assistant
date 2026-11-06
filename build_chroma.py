import json
import requests
import chromadb

sf_key = open("key_sf.txt", encoding="utf-8").read().strip()


def embed(texts):    
    r = requests.post(
        "https://api.siliconflow.cn/v1/embeddings",
        headers={"Authorization": f"Bearer {sf_key}", "Content-Type": "application/json"},
        json={"model": "BAAI/bge-m3", "input": texts},
        timeout=60,
    )
    d = r.json()
    if d.get("data") is None:
        raise RuntimeError(f"embedding 出错：{d}")
    out = [None] * len(texts)
    for item in d["data"]:
        out[item["index"]] = item["embedding"]
    return out


client = chromadb.PersistentClient(path="./chroma_db")    
col = client.get_or_create_collection(
    "notes",
    embedding_function=None,                
    metadata={"hnsw:space": "cosine"},      
)

data = json.load(open("index.json", encoding="utf-8"))
chunks = data["chunks"]
print(f"  index.json 里 {len(chunks)} 块")

BATCH = 32                                  
for i in range(0, len(chunks), BATCH):
    batch = chunks[i:i + BATCH]
    col.upsert(
        ids=[f"c{i + j}" for j in range(len(batch))],
        documents=[c["text"] for c in batch],
        embeddings=embed([c["text"] for c in batch]),
        metadatas=[{"source": c["source"]} for c in batch],
    )
    print(f"    已加 {min(i + BATCH, len(chunks))}/{len(chunks)}")

print(f"\n  集合里现在有 {col.count()} 条")
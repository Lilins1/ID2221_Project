# 文件名: chroma_similarity_search.py
from chromadb import HttpClient
import urllib.request
import json

# ===============================
# 配置 Chroma 和 Ollama
# ===============================
CHROMA_HOST = "localhost"       # 宿主机访问容器
CHROMA_PORT = 8000
COLLECTION_NAME = "ori_pqau_medical_qa_rag"

OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL_NAME = "embeddinggemma:latest"

# ===============================
# 生成嵌入向量
# ===============================
def get_embedding(text: str) -> list:
    payload = {"model": MODEL_NAME, "prompt": text}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode("utf-8"))
    return result["embedding"]

# ===============================
# 查询 Chroma 相似文档
# ===============================
def query_chroma(query_text: str, top_k: int = 5):
    # 1. 获取查询向量
    embedding = get_embedding(query_text)

    # 2. 连接 Chroma
    client = HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    # 3. 查询相似文档
    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    # 4. 输出结果
    print(f"查询: {query_text}")
    print(f"找到 {len(docs)} 条最相似文档\n")
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances), start=1):
        print(f"相似文档 {i}:")
        print("相似度分数:", dist)
        print("内容:", doc)
        print("元数据:", json.dumps(meta, ensure_ascii=False))
        print("-" * 50)

# ===============================
# 主程序
# ===============================
if __name__ == "__main__":
    while True:
        query = input("请输入查询文本（退出请输入 'exit'）：\n> ").strip()
        if query.lower() == "exit":
            break
        query_chroma(query, top_k=5)

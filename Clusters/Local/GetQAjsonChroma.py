# 文件名: chroma_similarity_export.py
from chromadb import HttpClient
import urllib.request
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# ===============================
# 配置 Chroma 和 Ollama
# ===============================
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION_NAME = "ori_pqau_medical_qa_rag"
# COLLECTION_NAME = "ori_pqau_medical_qa_rag"
OUTPUT_JSON = "ori_pqau_chroma_query_results.json"


OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL_NAME = "embeddinggemma:latest"

OUTPUT_JSON = "Selecteddata/test_set_chroma_query_results.json"
TOP_K = 500       # 每条查询返回前 K 个最相似文档
BATCH_SIZE = 50  # 分批处理
MAX_WORKERS = 5  # 并行嵌入线程数

# ===============================
# 生成嵌入向量
# ===============================
def get_embedding(text: str) -> list:
    """调用 Ollama API 生成文本嵌入"""
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
# 辅助函数
# ===============================
def parse_list_field(value):
    """将字符串或 JSON 数组恢复为 Python 列表"""
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else [parsed]
        except json.JSONDecodeError:
            return [value]
    elif isinstance(value, list):
        return value
    else:
        return []

# ===============================
# 查询 Chroma 并导出
# ===============================
def query_and_export(queries, output_file):
    client = HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    all_results = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_query = {executor.submit(get_embedding, q): q for q in queries}

        for future in as_completed(future_to_query):
            query_text = future_to_query[future]
            try:
                embedding = future.result()
            except Exception as e:
                print(f"生成嵌入失败: {query_text[:50]}..., 错误: {e}")
                continue

            # 查询 Chroma
            results = collection.query(
                query_embeddings=[embedding],
                n_results=TOP_K,
                include=["documents", "metadatas", "distances"]
            )

            docs = results["documents"][0]
            metas = results["metadatas"][0]

            for doc_text, meta in zip(docs, metas):
                doc_id = meta.get("id") or meta.get("doc_id") or str(len(all_results)+1)
                all_results[doc_id] = {
                    "QUESTION": meta.get("question", ""),
                    "CONTEXTS": parse_list_field(meta.get("contexts", [])),
                    "LABELS": parse_list_field(meta.get("labels", [])),
                    "MESHES": parse_list_field(meta.get("meshes", [])),
                    "YEAR": meta.get("year", ""),
                    "reasoning_required_pred": meta.get("reasoning_required_pred", ""),
                    "reasoning_free_pred": meta.get("reasoning_free_pred", ""),
                    "final_decision": meta.get("final_decision", ""),
                    "LONG_ANSWER": meta.get("long_answer", ""),
                }

                # 分批保存，防止内存过大
                if len(all_results) >= BATCH_SIZE:
                    with open(output_file, "a", encoding="utf-8") as f:
                        json.dump(all_results, f, ensure_ascii=False, indent=4)
                        f.write("\n")  # 换行隔开批次
                    all_results.clear()

    # 写入剩余的数据
    if all_results:
        with open(output_file, "a", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)
            f.write("\n")

    print(f"已导出所有文档到 {output_file}")

# ===============================
# 主程序
# ===============================
if __name__ == "__main__":
    queries = [
        "cancer"
    ]

    # 如果文件存在，先清空
    if os.path.exists(OUTPUT_JSON):
        os.remove(OUTPUT_JSON)

    query_and_export(queries, OUTPUT_JSON)

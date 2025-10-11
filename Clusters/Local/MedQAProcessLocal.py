import json
import os
import traceback
from tenacity import retry, stop_after_attempt, wait_fixed
import urllib.request
from tqdm import tqdm
import chromadb

# ----------------------
# 配置参数
# ----------------------
OLLAMA_HOST = "localhost"
OLLAMA_PORT = 11434
OLLAMA_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/embeddings"
MODEL_NAME = "embeddinggemma:latest"

CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
# CHROMA_COLLECTION = "test_set_medical_qa_rag"

# INPUT_JSON = "../../Data/pubmedqa/test_set.json"
# OUTPUT_JSON = "embeddedData/test_set_with_embedding.json"
CHROMA_COLLECTION = "ori_pqau_medical_qa_rag"

INPUT_JSON = "../../Data/pubmedqa/ori_pqau.json"
OUTPUT_JSON = "embeddedData/ori_pqau_with_embedding.json"

INPUT_JSON_ABS = os.path.abspath(INPUT_JSON)
print(f"输入文件绝对路径: {INPUT_JSON_ABS}")

BATCH_SIZE = 100  # 每处理多少条写一次数据库

# ----------------------
# 嵌入向量生成函数
# ----------------------
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def call_ollama_embedding(text: str) -> list:
    """调用Ollama API生成嵌入向量，带有重试机制"""
    if not text or len(text.strip()) < 5:
        # 如果文本太短或为空，返回空列表，避免不必要的API调用
        return []
    try:
        payload = {"model": MODEL_NAME, "prompt": text.strip()}
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url=OLLAMA_URL,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
        if "embedding" in result:
            return result["embedding"]
        raise ValueError("Ollama返回结果中无embedding字段")
    except Exception as e:
        # 抛出更具体的异常，便于重试逻辑捕获
        raise ConnectionError(f"Ollama调用失败: {str(e)}")

# ----------------------
# 元数据安全处理 (核心修改)
# ----------------------
def safe_metadata_value(value):
    """
    确保元数据值是ChromaDB支持的类型。
    对于列表，将其序列化为JSON字符串以保留结构。
    """
    if value is None:
        return ""  # 空值返回空字符串
    if isinstance(value, list):
        # === 修改核心 ===
        # 将列表转换为JSON字符串，而不是用逗号连接
        # 这样在读取时可以轻松地恢复为列表
        return json.dumps(value, ensure_ascii=False) if value else "[]"
    # 其他类型直接转为字符串
    return str(value).strip()

# ----------------------
# 批量写入ChromaDB的辅助函数
# ----------------------
def _write_batch_to_chroma(collection, batch):
    """将一个批次的数据写入ChromaDB"""
    try:
        metadatas = [{
            "id": doc["doc_id"],
            "question": safe_metadata_value(doc["QUESTION"]),
            "labels": safe_metadata_value(doc["LABELS"]),
            "meshes": safe_metadata_value(doc["MESHES"]),
            "year": safe_metadata_value(doc["YEAR"]),
            "reasoning_required_pred": safe_metadata_value(doc["reasoning_required_pred"]),
            "reasoning_free_pred": safe_metadata_value(doc["reasoning_free_pred"]),
            "final_decision": safe_metadata_value(doc["final_decision"]),
            "contexts": safe_metadata_value(doc["CONTEXTS"]),
            "long_answer": safe_metadata_value(doc["LONG_ANSWER"])
        } for doc in batch]

        collection.add(
            ids=[doc["doc_id"] for doc in batch],
            documents=[doc["rag_text"] for doc in batch],
            embeddings=[doc["embedding"] for doc in batch],
            metadatas=metadatas
        )
        tqdm.write(f"成功写入批次 {batch[0]['doc_id']} - {batch[-1]['doc_id']}，共 {len(batch)} 条")
    except Exception as e:
        tqdm.write(f"错误：写入批次 {batch[0]['doc_id']} - {batch[-1]['doc_id']} 失败: {e}")
        traceback.print_exc()

# ----------------------
# 主处理流程
# ----------------------
def process_and_write_to_chroma():
    """主函数，处理数据并写入ChromaDB"""
    if not os.path.exists(INPUT_JSON_ABS):
        raise FileNotFoundError(f"输入文件不存在：{INPUT_JSON_ABS}")

    with open(INPUT_JSON_ABS, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    total_docs = len(raw_data)
    print(f"成功读取 {total_docs} 个文档")

    # 连接Chroma
    chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    chroma_client.heartbeat()
    print("成功连接到Chroma服务")

    collection = chroma_client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"embedding_model": MODEL_NAME}
    )

    batch_docs = []
    success_count = 0
    processed_docs_for_json = []

    # 使用tqdm创建进度条
    for doc_id, doc_content in tqdm(raw_data.items(), total=total_docs, desc="处理并写入文档"):
        doc = {
            "QUESTION": doc_content.get("QUESTION", ""),
            "CONTEXTS": doc_content.get("CONTEXTS", []),
            "LONG_ANSWER": doc_content.get("LONG_ANSWER", ""),
            "LABELS": doc_content.get("LABELS", []),
            "MESHES": doc_content.get("MESHES", []),
            "YEAR": doc_content.get("YEAR", ""),
            "reasoning_required_pred": doc_content.get("reasoning_required_pred", ""),
            "reasoning_free_pred": doc_content.get("reasoning_free_pred", ""),
            "final_decision": doc_content.get("final_decision", "")
        }

        # 准备用于RAG的文本
        rag_text = "\n".join([doc["QUESTION"]] + doc.get("CONTEXTS", []) + [doc["LONG_ANSWER"]]).strip()

        try:
            embedding = call_ollama_embedding(rag_text)
            if not embedding:
                tqdm.write(f"警告：文档 {doc_id} 的文本过短或无效，无法生成嵌入向量，已跳过。")
                continue
        except Exception as e:
            tqdm.write(f"错误：文档 {doc_id} 嵌入生成失败 - {str(e)}，已跳过。")
            continue

        processed_doc = {
            "doc_id": doc_id,
            **doc,
            "rag_text": rag_text,
            "embedding": embedding
        }

        batch_docs.append(processed_doc)
        # 只有成功处理的文档才会被添加到最终的JSON文件中
        processed_docs_for_json.append(processed_doc)

        # 批量写入
        if len(batch_docs) >= BATCH_SIZE:
            _write_batch_to_chroma(collection, batch_docs)
            success_count += len(batch_docs)
            batch_docs = []

    # 写入最后不足一个批次的文档
    if batch_docs:
        _write_batch_to_chroma(collection, batch_docs)
        success_count += len(batch_docs)

    print(f"\n全部写入完成，共成功写入 {success_count}/{total_docs} 条数据到ChromaDB")

    # 保存包含嵌入向量的JSON文件
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        # 为了方便阅读，将结果保存为一个字典，key是doc_id
        final_json_output = {item['doc_id']: item for item in processed_docs_for_json}
        json.dump(final_json_output, f, ensure_ascii=False, indent=2)
    print(f"已保存处理结果到：{OUTPUT_JSON}")

# ----------------------
# 程序入口
# ----------------------
if __name__ == "__main__":
    try:
        process_and_write_to_chroma()
    except Exception as e:
        print(f"主流程发生严重错误：{str(e)}")
        traceback.print_exc()

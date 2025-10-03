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
CHROMA_COLLECTION = "ori_pqau_medical_qa_rag"

# 文件路径
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
    if not text or len(text.strip()) < 5:
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
        raise Exception("Ollama返回结果中无embedding字段")
    except Exception as e:
        raise Exception(f"Ollama调用失败: {str(e)}")

# ----------------------
# 元数据安全处理
# ----------------------
def safe_metadata_value(value):
    if value is None:
        return ""
    if isinstance(value, list):
        return ",".join([str(item).strip() for item in value]) if value else ""
    return str(value).strip()

# ----------------------
# 边处理边写入 Chroma
# ----------------------
def process_and_write_to_chroma():
    if not os.path.exists(INPUT_JSON_ABS):
        raise FileNotFoundError(f"输入文件不存在：{INPUT_JSON_ABS}")
    
    with open(INPUT_JSON_ABS, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    total_docs = len(raw_data)
    print(f"成功读取 {total_docs} 个文档")
    
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

        rag_text = "\n".join([doc["QUESTION"], "\n".join(doc["CONTEXTS"]), doc["LONG_ANSWER"]]).strip()
        
        try:
            embedding = call_ollama_embedding(rag_text)
            if not embedding:
                tqdm.write(f"警告：文档 {doc_id} 嵌入向量为空，跳过")
                continue
        except Exception as e:
            tqdm.write(f"错误：文档 {doc_id} 嵌入生成失败 - {str(e)}，跳过")
            continue

        processed_doc = {
            "doc_id": doc_id,
            **doc,
            "rag_text": rag_text,
            "embedding": embedding
        }

        batch_docs.append(processed_doc)
        processed_docs_for_json.append(processed_doc)

        # 批量写入
        if len(batch_docs) >= BATCH_SIZE:
            _write_batch_to_chroma(collection, batch_docs)
            success_count += len(batch_docs)
            batch_docs = []

    # 写入最后不足一批的文档
    if batch_docs:
        _write_batch_to_chroma(collection, batch_docs)
        success_count += len(batch_docs)

    print(f"全部写入完成，共写入 {success_count}/{total_docs} 条数据")

    # 保存 JSON 文件
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(processed_docs_for_json, f, ensure_ascii=False, indent=2)
    print(f"已保存处理结果到：{OUTPUT_JSON}")

    del chroma_client
    print("已断开Chroma连接")


def _write_batch_to_chroma(collection, batch):
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

# ----------------------
# 主流程
# ----------------------
if __name__ == "__main__":
    try:
        process_and_write_to_chroma()
    except Exception as e:
        print(f"主流程错误：{str(e)}")
        traceback.print_exc()

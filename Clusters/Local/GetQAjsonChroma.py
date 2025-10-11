from chromadb import HttpClient
import urllib.request
import json
import os
import re # 导入 re 模块用于生成安全文件名
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# ===============================
# 配置 Chroma 和 Ollama
# ===============================
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
# 确保这里的集合名称与你写入数据时使用的名称一致
COLLECTION_NAME = "ori_pqau_medical_qa_rag"

OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL_NAME = "embeddinggemma:latest"

TOP_K = 3000     # 每条查询返回前 K 个最相似文档
MAX_WORKERS = 5  # 并行嵌入线程数

# ===============================
# 生成嵌入向量 (无变化)
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
# 辅助函数 (核心修改)
# ===============================
def parse_list_field(value, field_name=None):
    """
    解析元数据字段的更新策略：
     - 如果已经是 list，直接返回。
     - 如果是 str，优先尝试将其作为 JSON 列表解析。
     - 如果 JSON 解析失败，则回退到旧的分割逻辑（为了兼容）。
    """
    if isinstance(value, list):
        return [v.strip() if isinstance(v, str) else v for v in value]

    if isinstance(value, str):
        # === 核心修改：优先尝试将字符串作为JSON解析 ===
        # 因为我们新的存储脚本将列表保存为JSON字符串
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                # 解析成功，且结果是列表，直接返回
                return [p.strip() if isinstance(p, str) else p for p in parsed]
        except json.JSONDecodeError:
            # 如果不是一个有效的JSON字符串，则忽略错误，继续执行下面的旧逻辑
            pass

        # === 以下为旧逻辑/回退逻辑，用于兼容老数据 ===
        # 如果JSON解析失败，它可能是一个普通的、未格式化的字符串
        if field_name == "contexts" and '\n' in value:
            return [item.strip() for item in value.split('\n') if item.strip()]

        if field_name == "meshes" and "," in value:
            return [item.strip() for item in value.split(",") if item.strip()]

        # 如果以上都不适用，则将整个字符串作为一个元素返回
        return [value.strip()]

    # 其他类型（如None）返回空列表
    return []

def sanitize_filename(text: str, max_length: int = 100) -> str:
    """根据输入文本创建一个安全的文件名"""
    # 移除非法字符
    text = re.sub(r'[\\/*?:"<>|]', "", text)
    # 用下划线替换空格
    text = text.replace(" ", "_")
    # 截断到最大长度
    return text[:max_length]

# ===============================
# 查询 Chroma 并导出 (逻辑无变化, 但现在会调用更新后的解析函数)
# ===============================
def query_and_export(queries, output_base_dir):
    client = HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    collection = client.get_collection(name=COLLECTION_NAME) # 使用 get_collection 更安全

    # 创建时间子文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_path = os.path.join(output_base_dir, timestamp)
    os.makedirs(folder_path, exist_ok=True)
    print(f"结果将保存到目录: {folder_path}")

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
                include=["metadatas", "distances"] # documents 字段通常和 rag_text 重复，可以不取
            )

            query_specific_results = {}
            metas = results.get("metadatas", [[]])[0]
            if not metas:
                print(f"查询 '{query_text[:50]}...' 没有返回结果。")
                continue

            for idx, meta in enumerate(metas, start=1):
                base_id = str(meta.get("id") or meta.get("doc_id") or idx)
                doc_id = base_id
                suffix = 1
                while doc_id in query_specific_results:
                    doc_id = f"{base_id}_{suffix}"
                    suffix += 1

                # 这里的 parse_list_field 现在会正确处理JSON字符串
                query_specific_results[doc_id] = {
                    "QUESTION": meta.get("question", ""),
                    "CONTEXTS": parse_list_field(meta.get("contexts"), field_name="contexts"),
                    "LABELS": parse_list_field(meta.get("labels"), field_name="labels"),
                    "MESHES": parse_list_field(meta.get("meshes"), field_name="meshes"),
                    "YEAR": meta.get("year", ""),
                    "reasoning_required_pred": meta.get("reasoning_required_pred", ""),
                    "reasoning_free_pred": meta.get("reasoning_free_pred", ""),
                    "final_decision": meta.get("final_decision", ""),
                    "LONG_ANSWER": meta.get("long_answer", ""),
                }

            # === 修改：为当前查询生成并写入独立文件 ===
            # 新的文件命名规则：query的第一个单词 + "pubmedqaSet.json"
            first_word = query_text.split()[0] if query_text else "default_query"
            # 仍然使用 sanitize_filename 来确保第一个单词不含非法字符
            safe_first_word = sanitize_filename(first_word)
            new_filename = f"{safe_first_word}_PubmedqaSet.json"
            output_file = os.path.join(folder_path, new_filename)

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(query_specific_results, f, ensure_ascii=False, indent=4)

            print(f"  -> 已导出查询结果到 {output_file}")

# ===============================
# 主程序
# ===============================
if __name__ == "__main__":

    queries = [
        "Cancer: Cancer is a broad term for a class of diseases characterized by abnormal cells that grow and divide uncontrollably, with the potential to invade and destroy normal body tissue. These malignant cells can spread throughout the body via the blood and lymph systems in a process called metastasis. Cancers are often classified by the type of cell from which they originate, such as carcinoma (from epithelial cells), sarcoma (from connective tissue), leukemia (from blood-forming tissue), and lymphoma (from immune cells). Causes are multifactorial, including genetic mutations, exposure to carcinogens (like tobacco smoke or UV radiation), and certain viral infections. Treatment depends on the type and stage of cancer and may include surgery, chemotherapy, radiation therapy, immunotherapy, and targeted therapy.",
        "Common_cold: The common cold is a highly contagious, self-limiting viral infection of the upper respiratory tract, primarily affecting the nose and throat. It is caused by over 200 different viruses, with rhinoviruses being the most frequent culprit. Transmission occurs through airborne droplets from coughs and sneezes or by touching contaminated surfaces and then touching the face. Symptoms typically appear 1-3 days after exposure and include a runny or stuffy nose, sore throat, cough, sneezing, and sometimes mild body aches or a low-grade fever. There is no cure for the common cold; treatment is symptomatic, focusing on rest, hydration, and over-the-counter medications to relieve discomfort. Most people recover within 7-10 days.",
        "Bone_fracture: A bone fracture is a medical condition where there is a partial or complete break in the continuity of a bone. Fractures are typically caused by high-force impact or stress, such as from trauma (falls, accidents) or overuse (stress fractures in athletes). They can also occur due to certain medical conditions that weaken the bones, like osteoporosis. Fractures are classified in several ways, including whether they are 'closed' (skin is intact) or 'open'/'compound' (bone pierces the skin). Symptoms include severe pain, swelling, bruising, deformity in the affected area, and an inability to put weight on or use the limb. Treatment involves immobilizing the bone with a cast or splint to allow it to heal, and in more severe cases, surgical intervention with pins, plates, or screws may be necessary to realign the bone fragments.",
        "Diabetes: Diabetes, formally known as Diabetes Mellitus, is a chronic metabolic disorder characterized by high blood sugar (glucose) levels over a prolonged period. This occurs either because the pancreas does not produce enough insulin (Type 1 Diabetes) or because the body's cells do not respond effectively to the insulin that is produced (Type 2 Diabetes). Insulin is a hormone crucial for allowing glucose from the bloodstream to enter cells to be used for energy. Common symptoms include frequent urination, increased thirst, and increased hunger. If left unmanaged, diabetes can lead to serious long-term complications, including cardiovascular disease, kidney failure, nerve damage, and blindness. Management involves monitoring blood glucose levels, a healthy diet, regular physical activity, and, for many, oral medications or insulin injections."
    ]

    output_base_dir = "Selecteddata"
    query_and_export(queries, output_base_dir)


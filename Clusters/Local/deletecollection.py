from chromadb import HttpClient

# 连接 Chroma
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000

client = HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

# 删除集合
collection_name = "subtest_set_medical_qa_rag"

try:
    client.delete_collection(name=collection_name)
    print(f"集合 {collection_name} 已删除")
except Exception as e:
    print(f"删除失败: {str(e)}")

from chromadb import HttpClient

# ===============================
# 连接 Chroma
# ===============================
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000

client = HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

# ===============================
# 列出所有集合
# ===============================
def list_collections(client):
    collections = client.list_collections()
    if not collections:
        print("当前没有集合。")
    else:
        print("当前集合列表：")
        for c in collections:
            print(f"- {c.name}")
    return collections

# ===============================
# 删除集合
# ===============================
collection_name = "ori_pqau_medical_qa_rag"

print("\n=== 删除前 ===")
list_collections(client)

try:
    client.delete_collection(name=collection_name)
    print(f"\n集合 {collection_name} 已删除")
except Exception as e:
    print(f"\n删除失败: {str(e)}")

print("\n=== 删除后 ===")
list_collections(client)

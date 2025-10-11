from chromadb import HttpClient

# ===============================
# Connect to Chroma
# ===============================
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000

client = HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

# ===============================
# List all collections
# ===============================
def list_collections(client):
    collections = client.list_collections()
    if not collections:
        print("No collections found.")
    else:
        print("Current collections:")
        for c in collections:
            print(f"- {c.name}")
    return collections

# ===============================
# Delete a specific collection
# ===============================
collection_name = "ori_pqau_medical_qa_rag"

print("\n=== Before Deletion ===")
list_collections(client)

try:
    client.delete_collection(name=collection_name)
    print(f"\nCollection '{collection_name}' has been deleted successfully.")
except Exception as e:
    print(f"\nDeletion failed: {str(e)}")

print("\n=== After Deletion ===")
list_collections(client)

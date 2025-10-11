# File: chroma_similarity_search.py
from chromadb import HttpClient
import urllib.request
import json

# ===============================
# Configuration: Chroma & Ollama
# ===============================
CHROMA_HOST = "localhost"       # Host machine accessing the container
CHROMA_PORT = 8000
COLLECTION_NAME = "ori_pqau_medical_qa_rag"

OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL_NAME = "embeddinggemma:latest"

# ===============================
# Generate Embedding Vector
# ===============================
def get_embedding(text: str) -> list:
    """Call Ollama to generate an embedding vector for the given text."""
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
# Query Similar Documents in Chroma
# ===============================
def query_chroma(query_text: str, top_k: int = 5):
    """Retrieve the top-K most similar documents from Chroma."""
    # 1. Generate query embedding
    embedding = get_embedding(query_text)

    # 2. Connect to Chroma
    client = HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    # 3. Query similar documents
    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    # 4. Display results
    print(f"Query: {query_text}")
    print(f"Found {len(docs)} most similar documents\n")
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances), start=1):
        print(f"Similar Document {i}:")
        print("Similarity Score:", dist)
        print("Content:", doc)
        print("Metadata:", json.dumps(meta, ensure_ascii=False))
        print("-" * 50)

# ===============================
# Main Program
# ===============================
if __name__ == "__main__":
    while True:
        query = input("Enter your query (type 'exit' to quit):\n> ").strip()
        if query.lower() == "exit":
            break
        query_chroma(query, top_k=5)

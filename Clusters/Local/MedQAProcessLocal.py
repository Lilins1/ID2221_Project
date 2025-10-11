import json
import os
import traceback
from tenacity import retry, stop_after_attempt, wait_fixed
import urllib.request
from tqdm import tqdm
import chromadb

# ----------------------
# Configuration
# ----------------------
OLLAMA_HOST = "localhost"
OLLAMA_PORT = 11434
OLLAMA_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/embeddings"
MODEL_NAME = "embeddinggemma:latest"

CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
CHROMA_COLLECTION = "ori_pqau_medical_qa_rag"

INPUT_JSON = "../../Data/pubmedqa/ori_pqau.json"
OUTPUT_JSON = "embeddedData/ori_pqau_with_embedding.json"

INPUT_JSON_ABS = os.path.abspath(INPUT_JSON)
print(f"Input file absolute path: {INPUT_JSON_ABS}")

BATCH_SIZE = 100  # Number of documents per batch to write into ChromaDB

# ----------------------
# Embedding Generation
# ----------------------
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def call_ollama_embedding(text: str) -> list:
    """Call the Ollama API to generate embeddings, with retry mechanism."""
    if not text or len(text.strip()) < 5:
        # Skip very short or empty texts to avoid unnecessary API calls
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
        raise ValueError("No 'embedding' field in Ollama response.")
    except Exception as e:
        # Raise explicit error for retry logic
        raise ConnectionError(f"Ollama call failed: {str(e)}")

# ----------------------
# Metadata Handling
# ----------------------
def safe_metadata_value(value):
    """
    Ensure that metadata values are compatible with ChromaDB.
    Lists are serialized as JSON strings to preserve structure.
    """
    if value is None:
        return ""  # Replace None with empty string
    if isinstance(value, list):
        # Serialize lists into JSON instead of joining with commas
        # This allows easy reconstruction later
        return json.dumps(value, ensure_ascii=False) if value else "[]"
    return str(value).strip()

# ----------------------
# Write Batch to ChromaDB
# ----------------------
def _write_batch_to_chroma(collection, batch):
    """Write a batch of documents into ChromaDB."""
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
        tqdm.write(f"Successfully wrote batch {batch[0]['doc_id']} - {batch[-1]['doc_id']} ({len(batch)} items)")
    except Exception as e:
        tqdm.write(f"Error writing batch {batch[0]['doc_id']} - {batch[-1]['doc_id']}: {e}")
        traceback.print_exc()

# ----------------------
# Main Processing Function
# ----------------------
def process_and_write_to_chroma():
    """Main function to process data and write embeddings into ChromaDB."""
    if not os.path.exists(INPUT_JSON_ABS):
        raise FileNotFoundError(f"Input file not found: {INPUT_JSON_ABS}")

    with open(INPUT_JSON_ABS, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    total_docs = len(raw_data)
    print(f"Successfully loaded {total_docs} documents")

    # Connect to Chroma
    chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    chroma_client.heartbeat()
    print("Connected to Chroma service successfully")

    collection = chroma_client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"embedding_model": MODEL_NAME}
    )

    batch_docs = []
    success_count = 0
    processed_docs_for_json = []

    # Use tqdm progress bar for better visibility
    for doc_id, doc_content in tqdm(raw_data.items(), total=total_docs, desc="Processing and writing documents"):
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

        # Construct text for RAG embedding
        rag_text = "\n".join([doc["QUESTION"]] + doc.get("CONTEXTS", []) + [doc["LONG_ANSWER"]]).strip()

        try:
            embedding = call_ollama_embedding(rag_text)
            if not embedding:
                tqdm.write(f"Warning: Document {doc_id} is too short or invalid, skipped.")
                continue
        except Exception as e:
            tqdm.write(f"Error: Failed to generate embedding for document {doc_id} - {str(e)}. Skipped.")
            continue

        processed_doc = {
            "doc_id": doc_id,
            **doc,
            "rag_text": rag_text,
            "embedding": embedding
        }

        batch_docs.append(processed_doc)
        processed_docs_for_json.append(processed_doc)  # Only add successfully processed docs

        # Write in batches
        if len(batch_docs) >= BATCH_SIZE:
            _write_batch_to_chroma(collection, batch_docs)
            success_count += len(batch_docs)
            batch_docs = []

    # Write remaining documents
    if batch_docs:
        _write_batch_to_chroma(collection, batch_docs)
        success_count += len(batch_docs)

    print(f"\nAll data written. Successfully inserted {success_count}/{total_docs} documents into ChromaDB.")

    # Save processed data with embeddings to JSON
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        final_json_output = {item['doc_id']: item for item in processed_docs_for_json}
        json.dump(final_json_output, f, ensure_ascii=False, indent=2)
    print(f"Saved processed results to: {OUTPUT_JSON}")

# ----------------------
# Entry Point
# ----------------------
if __name__ == "__main__":
    try:
        process_and_write_to_chroma()
    except Exception as e:
        print(f"Critical error in main process: {str(e)}")
        traceback.print_exc()

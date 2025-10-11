from chromadb import HttpClient
import urllib.request
import json
import os
import re  # For safe filename generation
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# ===============================
# Configuration: Chroma & Ollama
# ===============================
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
# Ensure this matches the collection name used during data insertion
COLLECTION_NAME = "ori_pqau_medical_qa_rag"

OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL_NAME = "embeddinggemma:latest"

TOP_K = 3000     # Number of most similar documents to retrieve per query
MAX_WORKERS = 5  # Number of parallel embedding threads

# ===============================
# Generate Embeddings
# ===============================
def get_embedding(text: str) -> list:
    """Generate a text embedding using the Ollama API."""
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
# Helper Functions
# ===============================
def parse_list_field(value, field_name=None):
    """
    Parse metadata fields with flexible strategies:
     - If already a list, return it directly.
     - If a string, first try to parse it as a JSON list.
     - If JSON parsing fails, fall back to legacy split-based parsing.
    """
    if isinstance(value, list):
        return [v.strip() if isinstance(v, str) else v for v in value]

    if isinstance(value, str):
        # === Preferred: try parsing as JSON string ===
        # New data stores lists as serialized JSON strings
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [p.strip() if isinstance(p, str) else p for p in parsed]
        except json.JSONDecodeError:
            # Not a valid JSON string — fall back to legacy logic
            pass

        # === Legacy / compatibility fallback ===
        if field_name == "contexts" and '\n' in value:
            return [item.strip() for item in value.split('\n') if item.strip()]

        if field_name == "meshes" and "," in value:
            return [item.strip() for item in value.split(",") if item.strip()]

        # If none of the above, return as single-element list
        return [value.strip()]

    # For None or unexpected types, return an empty list
    return []

def sanitize_filename(text: str, max_length: int = 100) -> str:
    """Create a filesystem-safe filename from text."""
    text = re.sub(r'[\\/*?:"<>|]', "", text)  # Remove illegal characters
    text = text.replace(" ", "_")             # Replace spaces with underscores
    return text[:max_length]                  # Truncate if too long

# ===============================
# Query Chroma and Export Results
# ===============================
def query_and_export(queries, output_base_dir):
    client = HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    collection = client.get_collection(name=COLLECTION_NAME)  # safer than create/get hybrid

    # Create timestamped output folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_path = os.path.join(output_base_dir, timestamp)
    os.makedirs(folder_path, exist_ok=True)
    print(f"Results will be saved to: {folder_path}")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_query = {executor.submit(get_embedding, q): q for q in queries}

        for future in as_completed(future_to_query):
            query_text = future_to_query[future]
            try:
                embedding = future.result()
            except Exception as e:
                print(f"Embedding generation failed for: {query_text[:50]}..., Error: {e}")
                continue

            # Query Chroma for the most similar entries
            results = collection.query(
                query_embeddings=[embedding],
                n_results=TOP_K,
                include=["metadatas", "distances"]
            )

            query_specific_results = {}
            metas = results.get("metadatas", [[]])[0]
            if not metas:
                print(f"No results for query: '{query_text[:50]}...'")
                continue

            for idx, meta in enumerate(metas, start=1):
                base_id = str(meta.get("id") or meta.get("doc_id") or idx)
                doc_id = base_id
                suffix = 1
                while doc_id in query_specific_results:
                    doc_id = f"{base_id}_{suffix}"
                    suffix += 1

                # Use updated parser for all metadata fields
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

            # === Save one output file per query ===
            first_word = query_text.split()[0] if query_text else "default_query"
            safe_first_word = sanitize_filename(first_word)
            new_filename = f"{safe_first_word}_PubmedqaSet.json"
            output_file = os.path.join(folder_path, new_filename)

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(query_specific_results, f, ensure_ascii=False, indent=4)

            print(f"  -> Exported query results to {output_file}")

# ===============================
# Main Script
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

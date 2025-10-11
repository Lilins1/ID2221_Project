#  MedPubQA Distributed RAG Platform

##  Overview

This project builds a **distributed storage and computation platform** for creating a **medical question-answering RAG (Retrieval-Augmented Generation)** dataset based on the **MedPubQA** database.

The distributed cluster integrates:

- **HDFS** → Distributed medical QA data storage  
- **Apache Spark** → Distributed computation and preprocessing  
- **Ollama (Docker)** → Medical content embedding generation  
- **Redis + Chroma** → RAG embedding index and similarity search  

This system is designed to support **data filtering and selection** for **Mixture-of-Experts (MoE) LLM models**, especially during **progressive pruning** and evaluation stages.

---

## System Architecture

```plaintext
+-------------------+
|     Ollama API    |  ← Embedding generation
+-------------------+
          ↓
+-------------------+
|      Spark        |  ← Distributed computation
+-------------------+
          ↓
+-------------------+
|       HDFS        |  ← Distributed storage
+-------------------+
          ↓
+-------------------+
| Redis / Chroma DB |  ← Vector index + retrieval
+-------------------+

## Setup and Usage

### **Step 1 — Start the Cluster**

Use the following Docker Compose file to build and start all components (except **Ollama**):

```bash
docker-compose -f Clusters/docker-compose.yml up -d
This starts:

🗄️ HDFS (NameNode + DataNodes)

⚡ Redis

🧩 Chroma vector database

🔥 Spark master node

All ports are already pre-configured in docker-compose.yml.

Step 2 — Upload Medical QA Data
Upload your medical QA dataset to HDFS using the NameNode Web UI or command line:

hdfs dfs -put ./data/test_set.json /id2221/MedevalRaw/
Step 3 — Preprocess Raw Data
Run the following script on Spark to parse and convert the raw data (e.g., XML → JSON):

spark-submit Clusters/app-code/process_xml_to_json.py
This creates a structured JSON dataset stored in HDFS.

Step 4 — Generate RAG Embeddings
Run the main processing script to compute and store embeddings:

spark-submit Clusters/app-code/MedQAProcess.py
This step:

Calls the Ollama API to generate embeddings

Stores embeddings and metadata in Chroma

Saves the processed data back into HDFS

Step 5 — Query RAG Data
Ensure Chroma is running properly, then query stored RAG data using:

python Clusters/Local/chromaSearchContext.py
This script searches the RAG index and returns the most relevant medical QA entries.

Step 6 — Batch Query and Export Results
To perform batch queries and export the complete QA content as JSON:

python Clusters/Local/GetQAjsonChroma.py
The resulting file will include both the questions and full answers retrieved from Chroma.

Maintenance and Utilities
Delete a Chroma Collection
If you need to delete an existing RAG vector database:

python Clusters/Local/deletecollection.py
Local Processing Script
Clusters/Local/MedQAProcessLocal.py
is a local utility for interacting with HDFS, Chroma, Redis, and Spark directly — useful for debugging or standalone testing.
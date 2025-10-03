from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import col, concat_ws
from pyspark.sql.types import ArrayType, FloatType, StructType, StructField, StringType
import os
import json
import asyncio
import aiohttp

# ----------------------
# 配置参数
# ----------------------
OLLAMA_HOST = "host.docker.internal"
OLLAMA_PORT = 11434
OLLAMA_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/embeddings"
MODEL_NAME = "embeddinggemma:latest"
CHROMA_HOST = os.getenv("CHROMA_HOST", "chroma")
CHROMA_PORT = os.getenv("CHROMA_PORT", "8000")
CHROMA_COLLECTION = "test_set_medical_qa_rag"
HDFS_OUTPUT_PATH = "hdfs://namenode:9000/id2221/MedevalProcessed/test_set_with_embedding"
HDFS_INPUT_PATH = "hdfs://namenode:9000/id2221/MedevalRaw/test_set.json"
# CHROMA_COLLECTION = "ori_pqau_medical_qa_rag"
# HDFS_OUTPUT_PATH = "hdfs://namenode:9000/id2221/MedevalProcessed/ori_pqau_with_embedding"
# HDFS_INPUT_PATH = "hdfs://namenode:9000/id2221/MedevalRaw/ori_pqau.json"
PARTITION_NUM = 2500  # 大文件可以适当增加分区数

# ----------------------
# 初始化 Spark
# ----------------------
spark = SparkSession.builder \
    .appName("Async-Embedding-Chroma-HDFS") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "16g") \
    .config("spark.executor.memoryOverhead", "4g")\
    .config("spark.sql.jsonReader.multiLine", "true") \
    .config("spark.sql.shuffle.partitions", PARTITION_NUM) \
    .getOrCreate()

# ----------------------
# 异步 Ollama API
# ----------------------
async def fetch_embedding(session, text):
    if not text or len(text.strip()) < 5:
        return []
    payload = {"model": MODEL_NAME, "prompt": text.strip()}
    async with session.post(OLLAMA_URL, json=payload, timeout=30) as resp:
        result = await resp.json()
        return result.get("embedding", [])

async def fetch_embeddings_batch(texts):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_embedding(session, t) for t in texts]
        return await asyncio.gather(*tasks)

# ----------------------
# 数据读取与解析
# ----------------------
def read_large_json():
    raw_df = spark.read.option("multiLine", "true").option("inferSchema", "false").json(HDFS_INPUT_PATH)
    doc_ids = raw_df.columns
    print(f"从顶层 JSON 中识别到 {len(doc_ids)} 个文档ID")
    
    required_fields = {
        "QUESTION": "", "CONTEXTS": [], "LONG_ANSWER": "", "LABELS": [], "MESHES": [],
        "YEAR": "", "reasoning_required_pred": "", "reasoning_free_pred": "", "final_decision": ""
    }

    def explode_and_filter_docs(row):
        for doc_id in doc_ids:
            doc_content = row[doc_id]
            if doc_content is None:
                continue
            content_dict = doc_content.asDict() if hasattr(doc_content, "asDict") else dict(doc_content)
            filtered = {field: content_dict.get(field, default) for field, default in required_fields.items()}
            yield (
                doc_id,
                filtered["QUESTION"],
                filtered["CONTEXTS"],
                filtered["LONG_ANSWER"],
                filtered["LABELS"],
                filtered["MESHES"],
                filtered["YEAR"],
                filtered["reasoning_required_pred"],
                filtered["reasoning_free_pred"],
                filtered["final_decision"]
            )

    doc_rdd = raw_df.rdd.flatMap(explode_and_filter_docs)
    target_schema = StructType([
        StructField("doc_id", StringType(), False),
        StructField("QUESTION", StringType(), True),
        StructField("CONTEXTS", ArrayType(StringType()), True),
        StructField("LONG_ANSWER", StringType(), True),
        StructField("LABELS", ArrayType(StringType()), True),
        StructField("MESHES", ArrayType(StringType()), True),
        StructField("YEAR", StringType(), True),
        StructField("reasoning_required_pred", StringType(), True),
        StructField("reasoning_free_pred", StringType(), True),
        StructField("final_decision", StringType(), True)
    ])
    return spark.createDataFrame(doc_rdd, schema=target_schema).repartition(PARTITION_NUM)

# ----------------------
# 安全元数据处理
# ----------------------
def safe_metadata_value(value):
    if value is None:
        return ""
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    return str(value)

# ----------------------
# 分批处理函数
# ----------------------
def process_batch(batch, add_with_retry):
    texts = [row.rag_text for row in batch]
    embeddings = asyncio.run(fetch_embeddings_batch(texts))

    metas = [
        {
            "id": safe_metadata_value(row.doc_id),
            "question": safe_metadata_value(row.QUESTION),
            "labels": safe_metadata_value(row.LABELS),
            "meshes": safe_metadata_value(row.MESHES),
            "year": safe_metadata_value(row.YEAR),
            "reasoning_required_pred": safe_metadata_value(row.reasoning_required_pred),
            "reasoning_free_pred": safe_metadata_value(row.reasoning_free_pred),
            "final_decision": safe_metadata_value(row.final_decision),
            "contexts": safe_metadata_value(row.CONTEXTS),
            "long_answer": safe_metadata_value(row.LONG_ANSWER)
        } for row in batch
    ]

    try:
        add_with_retry(
            ids=[row.doc_id for row in batch],
            docs=texts,
            embeds=embeddings,
            metadata=metas
        )
    except Exception:
        import traceback
        print(f"Chroma写入失败: {traceback.format_exc()}")

    for row_obj, embed in zip(batch, embeddings):
        yield Row(**row_obj.asDict(), embedding=embed)

# ----------------------
# 逐条处理 partition
# ----------------------
def process_partition(partition):
    import chromadb
    from tenacity import retry, stop_after_attempt, wait_fixed

    chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    collection = chroma_client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"embedding_model": MODEL_NAME}
    )

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def add_with_retry(ids, docs, embeds, metadata):
        collection.add(ids=ids, documents=docs, embeddings=embeds, metadatas=metadata)

    batch = []
    for row in partition:
        batch.append(row)
        if len(batch) >= 50:
            yield from process_batch(batch, add_with_retry)
            batch.clear()
    if batch:
        yield from process_batch(batch, add_with_retry)

# ----------------------
# 主流程
# ----------------------
try:
    parsed_df = read_large_json()
    parsed_df = parsed_df.fillna({
        "YEAR": "", "QUESTION": "", "LONG_ANSWER": "",
        "reasoning_required_pred": "", "reasoning_free_pred": "", "final_decision": ""
    })
    parsed_df = parsed_df.withColumn(
        "rag_text",
        concat_ws("\n", col("QUESTION"), concat_ws("\n", col("CONTEXTS")), col("LONG_ANSWER"))
    )

    new_rdd = parsed_df.rdd.mapPartitions(process_partition)
    new_schema = StructType(parsed_df.schema.fields + [StructField("embedding", ArrayType(FloatType()), True)])
    df_with_embedding = spark.createDataFrame(new_rdd, schema=new_schema)

    # 写 HDFS
    df_with_embedding.write.mode("overwrite").option("maxRecordsPerFile", 1000).json(HDFS_OUTPUT_PATH)
    print(f"已保存到 HDFS: {HDFS_OUTPUT_PATH}")

except Exception as e:
    print(f"处理失败: {str(e)}")
finally:
    spark.stop()

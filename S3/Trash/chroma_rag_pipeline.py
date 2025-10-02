import os
import uuid
import boto3
import json
from botocore.exceptions import ClientError
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Optional

# 加载环境变量
load_dotenv()

class MedicalRAGPipeline:
    def __init__(self):
        """初始化RAG流水线组件"""
        # 初始化S3客户端
        with open("../config.json") as f:
            cfg = json.load(f)

        self.s3_client = boto3.client(
            cfg["service"],
            aws_access_key_id=cfg["aws_access_key_id"],
            aws_secret_access_key=cfg["aws_secret_access_key"],
            region_name=cfg["region_name"]
        )
        
        # 初始化ChromaDB客户端（本地模式）
        self.chroma_client = chromadb.PersistentClient(path="./medical_chroma_db")
        
        # 初始化医学领域嵌入模型（使用BioBERT衍生模型）
        self.embedding_func = embedding_functions.HuggingFaceEmbeddingFunction(
            model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
        )
        
        # 初始化文本切分器（针对医学文档优化）
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,           # 每个片段的字符数
            chunk_overlap=50,         # 片段重叠部分，保持上下文连续性
            separators=["\n\n", "\n", ". ", " ", ""],  # 优先按段落和句子切分
            length_function=len
        )
        
        # 获取或创建Chroma集合
        self.collection = self.chroma_client.get_or_create_collection(
            name="medical_articles",
            embedding_function=self.embedding_func,
            metadata={"description": "存储医学文章片段及其向量"}
        )

    def load_document_from_s3(self, bucket_name: str, s3_key: str) -> Optional[Dict]:
        """从S3加载文档内容"""
        try:
            response = self.s3_client.get_object(Bucket=bucket_name, Key=s3_key)
            content = response['Body'].read().decode('utf-8', errors='ignore')
            
            # 提取基本元数据
            metadata = {
                "s3_bucket": bucket_name,
                "s3_key": s3_key,
                "s3_path": f"s3://{bucket_name}/{s3_key}",
                "file_name": os.path.basename(s3_key),
                "file_size": len(content)
            }
            
            return {
                "content": content,
                "metadata": metadata
            }
        except ClientError as e:
            print(f"❌ 从S3加载文档失败: {str(e)}")
            return None
        except Exception as e:
            print(f"❌ 处理文档时出错: {str(e)}")
            return None

    def split_document(self, content: str) -> List[str]:
        """将文档切分为适合RAG的片段"""
        return self.text_splitter.split_text(content)

    def process_and_store_document(self, bucket_name: str, s3_key: str, category: str) -> bool:
        """处理文档并存储到ChromaDB"""
        # 1. 从S3加载文档
        doc_data = self.load_document_from_s3(bucket_name, s3_key)
        if not doc_data:
            return False
        
        # 2. 切分文档
        chunks = self.split_document(doc_data["content"])
        if not chunks:
            print("❌ 文档切分后没有内容")
            return False
        
        print(f"✅ 文档切分完成: {len(chunks)}个片段")
        
        # 3. 为每个片段准备数据
        ids = []
        documents = []
        metadatas = []
        
        for i, chunk in enumerate(chunks):
            # 生成唯一ID
            chunk_id = f"{uuid.uuid4()}"
            
            # 构建元数据（包含S3路径和分类信息）
            chunk_metadata = {
                **doc_data["metadata"],  # 继承文档级元数据
                "chunk_id": i,
                "category": category,    # 医学分类（如"lung_cancer"）
                "chunk_length": len(chunk)
            }
            
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append(chunk_metadata)
        
        # 4. 存储到ChromaDB
        try:
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
                # 嵌入向量会自动通过embedding_function生成
            )
            print(f"✅ 成功存储{len(chunks)}个片段到ChromaDB")
            return True
        except Exception as e:
            print(f"❌ 存储到ChromaDB失败: {str(e)}")
            return False

    def batch_process_documents(self, bucket_name: str, s3_prefix: str, category: str) -> Dict:
        """批量处理S3指定前缀下的所有文档"""
        stats = {
            "total": 0,
            "success": 0,
            "failed": 0
        }
        
        try:
            # 列出S3前缀下的所有文件
            response = self.s3_client.list_objects_v2(Bucket=bucket_name, Prefix=s3_prefix)
            if 'Contents' not in response:
                print(f"❌ 未找到文件: s3://{bucket_name}/{s3_prefix}")
                return stats
            
            # 批量处理每个文件
            for obj in response['Contents']:
                s3_key = obj['Key']
                # 跳过目录（仅处理文件）
                if s3_key.endswith('/'):
                    continue
                
                stats["total"] += 1
                print(f"\n处理文件 {stats['total']}/{len(response['Contents'])}: {s3_key}")
                
                if self.process_and_store_document(bucket_name, s3_key, category):
                    stats["success"] += 1
                else:
                    stats["failed"] += 1
            
            print(f"\n📊 批量处理完成: 共{stats['total']}个, 成功{stats['success']}个, 失败{stats['failed']}个")
            return stats
        except Exception as e:
            print(f"❌ 批量处理出错: {str(e)}")
            return stats

    def retrieve_relevant_chunks(self, query: str, n_results: int = 5, category: Optional[str] = None) -> List[Dict]:
        """检索与查询相关的文档片段"""
        # 构建过滤条件（可选）
        where_clause = {"category": category} if category else None
        
        # 执行检索
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_clause
        )
        
        # 整理结果
        retrieved = []
        for i in range(len(results['ids'][0])):
            retrieved.append({
                "id": results['ids'][0][i],
                "content": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "similarity_score": 1 - results['distances'][0][i]  # 转换为相似度分数（0-1）
            })
        
        return retrieved

    def get_document_from_s3(self, s3_path: str) -> Optional[str]:
        """从S3获取完整文档内容"""
        try:
            # 解析S3路径
            bucket = s3_path.split("//")[1].split("/")[0]
            key = "/".join(s3_path.split("//")[1].split("/")[1:])
            
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            return response['Body'].read().decode('utf-8', errors='ignore')
        except Exception as e:
            print(f"❌ 获取完整文档失败: {str(e)}")
            return None

if __name__ == "__main__":
    # 初始化RAG流水线
    rag_pipeline = MedicalRAGPipeline()
    
    # 配置参数（请替换为你的实际信息）
    BUCKET_NAME = "id2221"
    S3_PREFIX = "10_01/domain_partitioned/"  # S3中文档所在路径
    DOCUMENT_CATEGORY = "lung_cancer"  # 文档分类
    
    # 1. 批量处理文档并存储到ChromaDB
    print("===== 开始批量处理文档 =====")
    rag_pipeline.batch_process_documents(BUCKET_NAME, S3_PREFIX, DOCUMENT_CATEGORY)
    
    # 2. 测试检索功能
    print("\n===== 测试检索功能 =====")
    test_queries = [
        "肺癌的靶向治疗最新进展",
        "非小细胞肺癌的免疫治疗方法",
        "奥希替尼在肺癌治疗中的应用"
    ]
    
    for query in test_queries:
        print(f"\n查询: {query}")
        print("------------------------")
        results = rag_pipeline.retrieve_relevant_chunks(
            query=query,
            n_results=3,
            category=DOCUMENT_CATEGORY
        )
        
        for i, result in enumerate(results, 1):
            print(f"\n结果 {i} (相似度: {result['similarity_score']:.4f}):")
            print(f"来源: {result['metadata']['s3_path']}")
            print(f"片段: {result['content'][:200]}...")

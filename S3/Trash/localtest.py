import os
import uuid
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Optional

# 加载环境变量（如果需要配置嵌入模型路径等）
load_dotenv()

class LocalMedicalRAG:
    def __init__(self):
        """初始化本地RAG组件（无需S3）"""
        # 初始化ChromaDB（本地持久化存储）
        self.chroma_client = chromadb.PersistentClient(path="./local_medical_chroma_db")
        
        # 初始化医学嵌入模型
        self.embedding_func = embedding_functions.HuggingFaceEmbeddingFunction(
            model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
        )
        
        # 初始化文本切分器（针对医学文档优化）
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # 获取或创建Chroma集合
        self.collection = self.chroma_client.get_or_create_collection(
            name="local_medical_docs",
            embedding_function=self.embedding_func,
            metadata={"description": "本地医学文档RAG测试"}
        )

    def load_local_document(self, file_path: str) -> Optional[Dict]:
        """从本地加载文本文件"""
        try:
            # 检查文件是否存在
            if not os.path.exists(file_path):
                print(f"❌ 文件不存在: {file_path}")
                return None
            
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # 提取元数据
            metadata = {
                "local_path": file_path,
                "file_name": os.path.basename(file_path),
                "file_size": len(content),
                "last_modified": os.path.getmtime(file_path)
            }
            
            return {
                "content": content,
                "metadata": metadata
            }
        except Exception as e:
            print(f"❌ 加载本地文件失败: {str(e)}")
            return None

    def split_document(self, content: str) -> List[str]:
        """切分文档为片段"""
        return self.text_splitter.split_text(content)

    def process_and_store_local(self, file_path: str, category: str) -> bool:
        """处理本地文件并存储到ChromaDB"""
        # 1. 加载本地文档
        doc_data = self.load_local_document(file_path)
        if not doc_data:
            return False
        
        # 2. 切分文档
        chunks = self.split_document(doc_data["content"])
        if not chunks:
            print("❌ 文档切分后无内容")
            return False
        
        print(f"✅ 文档切分完成: {len(chunks)}个片段")
        
        # 3. 准备存储数据
        ids = [f"{uuid.uuid4()}" for _ in chunks]
        metadatas = [{
            **doc_data["metadata"],
            "chunk_id": i,
            "category": category
        } for i in range(len(chunks))]
        
        # 4. 存储到ChromaDB
        try:
            self.collection.add(
                ids=ids,
                documents=chunks,
                metadatas=metadatas
            )
            print(f"✅ 成功存储到ChromaDB: {os.path.basename(file_path)}")
            return True
        except Exception as e:
            print(f"❌ 存储失败: {str(e)}")
            return False

    def batch_process_local_dir(self, dir_path: str, category: str) -> Dict:
        """批量处理本地目录下的所有文本文件"""
        stats = {"total": 0, "success": 0, "failed": 0}
        
        if not os.path.isdir(dir_path):
            print(f"❌ 目录不存在: {dir_path}")
            return stats
        
        # 遍历目录下的所有txt文件
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.endswith(".txt"):  # 只处理txt文件
                    file_path = os.path.join(root, file)
                    stats["total"] += 1
                    print(f"\n处理文件 {stats['total']}: {file}")
                    
                    if self.process_and_store_local(file_path, category):
                        stats["success"] += 1
                    else:
                        stats["failed"] += 1
        
        print(f"\n📊 批量处理完成: 共{stats['total']}个, 成功{stats['success']}个, 失败{stats['failed']}个")
        return stats

    def retrieve_local(self, query: str, n_results: int = 3, category: Optional[str] = None) -> List[Dict]:
        """检索相关文档片段"""
        where_clause = {"category": category} if category else None
        
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_clause
        )
        
        # 整理结果
        return [{
            "id": results['ids'][0][i],
            "content": results['documents'][0][i],
            "metadata": results['metadatas'][0][i],
            "similarity_score": 1 - results['distances'][0][i]
        } for i in range(len(results['ids'][0]))]

if __name__ == "__main__":
    # 初始化本地RAG
    local_rag = LocalMedicalRAG()
    
    # 配置你的本地文件路径（请修改为实际路径）
    # 单个文件测试
    SINGLE_FILE_PATH = r"C:\Users\Ruizhe\Desktop\Study\ID2221\Project\S3\LocalData\PMC000xxxxxxtxt\PMC466938.txt"  # 你的本地文件路径
    # 目录批量测试（如果有多个文件）
    DIR_PATH = r"C:\Users\Ruizhe\Desktop\Study\ID2221\Project\S3\LocalData\PMC000xxxxxxtxt"  # 包含多个txt文件的目录
    DOC_CATEGORY = "medical_research"  # 文档分类
    
    # 1. 处理单个文件（二选一，根据需要注释）
    print("===== 处理单个文件 =====")
    local_rag.process_and_store_local(SINGLE_FILE_PATH, DOC_CATEGORY)
    
    # # 2. 批量处理目录（如果有多个文件，取消注释）
    # print("\n===== 批量处理目录 =====")
    # local_rag.batch_process_local_dir(DIR_PATH, DOC_CATEGORY)
    
    # 3. 测试检索功能
    print("\n===== 测试检索 =====")
    test_queries = [
        "肺癌的诊断方法",
        "靶向治疗的副作用",
        "临床试验的结果分析"
    ]
    
    for query in test_queries:
        print(f"\n查询: {query}")
        print("-------------------")
        results = local_rag.retrieve_local(
            query=query,
            n_results=2,
            category=DOC_CATEGORY
        )
        
        for i, res in enumerate(results, 1):
            print(f"结果 {i} (相似度: {res['similarity_score']:.4f})")
            print(f"来源: {res['metadata']['local_path']}")
            print(f"片段: {res['content'][:200]}...")

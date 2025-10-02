from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
# 导入Ollama的大语言模型类
from langchain_community.llms import Ollama
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# --- 1. 配置参数 ---
CHROMA_PERSIST_DIR = "./chroma_db_ollama"  # 确保存储和加载的是同一个路径
OLLAMA_EMBEDDING_MODEL = "embeddinggemma:latest"
# ✅ 修改：指定要使用的Ollama大语言模型
OLLAMA_LLM_MODEL = "deepseek-r1:8b"  # 你也可以换成 "qwen:7b" 等


def main():
    print(f"--- 医学文献智能问答系统 (本地模型: {OLLAMA_LLM_MODEL}) ---")

    # --- 2. 加载已存在的向量数据库 ---
    print("正在加载本地知识库...")
    try:
        embedding_function = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)
        vector_db = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embedding_function
        )
    except Exception as e:
        print(f"错误：无法加载向量数据库。请先运行 'build_index.py' (Ollama版本)。 {e}")
        return

    # --- 3. 准备RAG的核心组件 ---
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})

    # Prompt模板可以保持不变，但可以根据模型的特点微调
    template = """
    You are a professional medical research assistant. Use the following pieces of context to answer the question at the end. 
    If you don't know the answer from the context, just say that you don't know, don't try to make up an answer.
    Answer in Chinese.

    Context:
    {context}

    Question:
    {question}

    Answer (in Chinese):
    """
    prompt = ChatPromptTemplate.from_template(template)

    # ✅ 修改：加载Ollama大语言模型
    llm = Ollama(model=OLLAMA_LLM_MODEL)

    # --- 4. 构建RAG处理链 (此部分不变) ---
    rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    print("知识库加载完成，可以开始提问了 (输入 '退出' 来结束程序)。\n")

    # --- 5. 创建一个交互式查询循环 (此部分不变) ---
    while True:
        user_question = input("请输入您的问题: ")
        if user_question.lower() == '退出':
            break

        print("\n正在检索并生成答案 (这可能需要一些时间)...")

        # 调用RAG链来获取答案
        answer = rag_chain.invoke(user_question)

        print("\n💡 答案:")
        print(answer)
        print("-" * 50)

        relevant_docs = retriever.get_relevant_documents(user_question)
        print("🔍 参考文献片段:")
        for i, doc in enumerate(relevant_docs):
            print(f"  片段 {i + 1} (来自PMID: {doc.metadata.get('pmid', 'N/A')}):")
            print(f"  > {doc.page_content[:200]}...\n")

    print("感谢使用，再见！")


if __name__ == "__main__":
    main()
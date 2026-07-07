# modules/rag_chain.py
import os
from dotenv import load_dotenv
load_dotenv()
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough

# 加载根目录的.env文件（因为当前文件在modules里，所以用 ../ 回到根目录）



def build_rag_chain(persist_dir: str = "./chroma_db"):
    """构建完整的RAG查询链"""
    print("◆  开始加载嵌入模型...")
    # 1. 初始化嵌入模型
    embeddings = HuggingFaceEmbeddings(
        model_name="D:/models/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print("✅ 嵌入模型加载完成")

    print("◆  加载向量数据库...")
    # 2. 加载已持久化的向量库
    vector_db = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )
    retriever = vector_db.as_retriever(search_kwargs={"k": 4})
    print("✅ 数据库加载完成")

    # 3. 初始化大模型
    llm = ChatOpenAI(
        model="deepseek-chat",
        openai_api_key=os.environ.get("API_KEY"),
        base_url="https://api.deepseek.com/v1",
        temperature=0.0,
        timeout=30,
    )

    # 4. 提示词 + 输出解析器
    parser = JsonOutputParser()
    prompt = PromptTemplate(
        template="根据上下文提取竞品价格信息，仅输出JSON：\n{context}\n{format_instructions}",
        input_variables=["context"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    # 5. 组装LCEL RAG链
    rag_chain = (
        {
            "context": retriever | (lambda docs: "\n".join(d.page_content for d in docs))
        }
        | prompt
        | llm
        | parser
    )

    return rag_chain


if __name__ == "__main__":
    # 单独运行这个文件时测试RAG
    rag_chain = build_rag_chain()
    print("◆  开始执行 RAG 查询...")
    result = rag_chain.invoke("竞品A最近有什么价格变动？")
    print("✅ 结构化价格情报：", result)
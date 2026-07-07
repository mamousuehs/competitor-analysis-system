# data_loader.py
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def initialize_vector_db():
    """初始化向量数据库，灌入模拟数据"""
    # 1. 初始化嵌入模型
    embeddings = HuggingFaceEmbeddings(
        model_name=r"B:\MySoftwarework\Python\MyCode\CAS\competitor-analysis-system\models\bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # 2. 模拟数据
    mock_texts = [
        "竞品A在2026年7月5日宣布全线产品降价15%，标准版年费从999元降至849元。",
        "竞品A推出了新的入门级套餐，定价为每月49元，直接对标本企业的轻量版。",
        "根据行业论坛爆料，竞品A在618大促期间，高级版实际成交价低至599元/年。",
        "竟品A的CEO在采访中表示，下半年将采取激进的价格策略抢占市场份额。",
    ]

    # 3. 创建文档对象
    docs = [Document(page_content=text, metadata={"source": "mock"}) for text in mock_texts]

    # 4. 文本分割
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    split_docs = text_splitter.split_documents(docs)

    # 5. 创建向量数据库
    vector_db = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory="./chroma_db",
    )

    print(f"灌库完成!共存入{len(split_docs)}条向量数据。")


if __name__ == "__main__":
    initialize_vector_db()


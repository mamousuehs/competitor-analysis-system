"""Vector database initialization and document loading."""

from __future__ import annotations

import csv
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "bge-m3"
DEFAULT_VECTOR_DB_PATH = PROJECT_ROOT / "chroma_db"
DEFAULT_NEWS_CSV = PROJECT_ROOT / "data" / "raw" / "news.csv"


def _build_embeddings(model_path: str | Path = DEFAULT_MODEL_PATH) -> HuggingFaceEmbeddings:
    model_path = Path(model_path)
    if not model_path.exists() or not any(model_path.iterdir()):
        raise FileNotFoundError(
            f"未找到本地嵌入模型：{model_path}。请先把 bge-m3 模型文件拷贝到该目录。"
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=str(model_path),
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return embeddings


def _load_csv_documents(csv_path: str | Path = DEFAULT_NEWS_CSV) -> list[Document]:
    csv_path = Path(csv_path)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return []

    docs: list[Document] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return []

        for index, row in enumerate(reader, start=1):
            text_parts = [
                str(row.get(key, "")).strip()
                for key in ("title", "content", "summary", "text", "正文", "标题")
                if row.get(key)
            ]
            content = "\n".join(text_parts).strip() or "\n".join(
                str(value).strip() for value in row.values() if value
            )
            if content:
                docs.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source": row.get("source") or row.get("url") or str(csv_path),
                            "row": index,
                        },
                    )
                )
    return docs


def _mock_documents() -> list[Document]:
    mock_texts = [
        "竞品A在2026年7月5日宣布全线产品降价15%，标准版年费从999元降至849元。",
        "竞品A推出了新的入门级套餐，定价为每月49元，直接对标本企业的轻量版。",
        "根据行业论坛爆料，竞品A在618大促期间，高级版实际成交价低至599元/年。",
        "竞品A的CEO在采访中表示，下半年将采取激进的价格策略抢占市场份额。",
        "竞品B发布专业版订阅，首年优惠价1299元/年，续费价格1599元/年。",
        "竞品C宣布企业版从每席位89元/月上调至99元/月，调整将于2026年8月1日生效。",
    ]
    return [Document(page_content=text, metadata={"source": "mock"}) for text in mock_texts]


def init_and_fill_vector_db(
    persist_dir: str | Path = DEFAULT_VECTOR_DB_PATH,
    csv_path: str | Path = DEFAULT_NEWS_CSV,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> Chroma:
    """Initialize Chroma and fill it with CSV documents or fallback mock data."""
    embeddings = _build_embeddings(model_path)
    docs = _load_csv_documents(csv_path) or _mock_documents()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    split_docs = text_splitter.split_documents(docs)

    vector_db = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=str(persist_dir),
    )

    print(f"灌库完成！共存入 {len(split_docs)} 条向量数据。")
    return vector_db


def initialize_vector_db() -> Chroma:
    """Backward-compatible entry point used by older course code."""
    return init_and_fill_vector_db()


if __name__ == "__main__":
    initialize_vector_db()


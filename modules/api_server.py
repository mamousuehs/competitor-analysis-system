"""FastAPI service for competitor intelligence workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from modules.agent_core import build_agent_executor, build_runnable_sequence
from modules.data_loader import DEFAULT_VECTOR_DB_PATH, init_and_fill_vector_db
from modules.monitoring_pipeline import build_demo_documents, collect_sources, run_pipeline
from modules.rag_chain import build_rag_chain
from modules.tools import crawl_web_text, summarize_url

app = FastAPI(title="Competitor Analysis System", version="0.1.0")
agent_executor = None


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)


class SummaryRequest(BaseModel):
    product: str
    competitor: str
    key_points: str


class UrlRequest(BaseModel):
    url: str


class MonitorRequest(BaseModel):
    news_urls: list[str] = Field(default_factory=list)
    rss_feeds: list[str] = Field(default_factory=list)
    forum_urls: list[str] = Field(default_factory=list)
    forum_content_class: str | None = None
    chunk_size: int = Field(default=600, ge=100, le=3000)
    chunk_overlap: int = Field(default=100, ge=0, le=1000)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/rag/price")
def rag_price(request: QueryRequest) -> dict[str, Any]:
    if not Path(DEFAULT_VECTOR_DB_PATH).exists():
        init_and_fill_vector_db()
    chain = build_rag_chain()
    return chain.invoke(request.query)


@app.post("/summary")
def summary(request: SummaryRequest) -> dict[str, str]:
    chain = build_runnable_sequence()
    result = chain.invoke(request.model_dump())
    return {"summary": result}


@app.post("/crawl")
def crawl(request: UrlRequest) -> dict[str, Any]:
    return crawl_web_text(request.url)


@app.post("/crawl/summarize")
def crawl_summary(request: UrlRequest) -> dict[str, str]:
    return {"summary": summarize_url(request.url)}


@app.post("/chat")
def chat(request: QueryRequest) -> dict[str, Any]:
    global agent_executor
    if agent_executor is None:
        agent_executor = build_agent_executor(verbose=False)
    return agent_executor.invoke({"input": request.query})


@app.post("/monitor/collect")
def monitor_collect(request: MonitorRequest) -> dict[str, Any]:
    documents, summary = collect_sources(
        news_urls=request.news_urls,
        rss_feeds=request.rss_feeds,
        forum_urls=request.forum_urls,
        forum_content_class=request.forum_content_class,
    )
    previews = [
        {
            "title": doc.metadata.get("title", ""),
            "source_type": doc.metadata.get("source_type", "unknown"),
            "source": doc.metadata.get("source", ""),
            "content_preview": doc.page_content[:200],
        }
        for doc in documents[:10]
    ]
    return {"summary": summary, "previews": previews}


@app.post("/monitor/pipeline")
def monitor_pipeline(request: MonitorRequest) -> dict[str, Any]:
    documents, collect_summary = collect_sources(
        news_urls=request.news_urls,
        rss_feeds=request.rss_feeds,
        forum_urls=request.forum_urls,
        forum_content_class=request.forum_content_class,
    )
    result = run_pipeline(
        documents,
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
    )
    return {"collect_summary": collect_summary, "pipeline": result}


@app.post("/monitor/demo")
def monitor_demo() -> dict[str, Any]:
    documents = build_demo_documents()
    return run_pipeline(documents)

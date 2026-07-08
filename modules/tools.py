"""LangChain tools and web crawling helpers for competitor intelligence."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool

from modules.llm_client import get_deepseek_llm
from modules.prompts import CRAWL_SUMMARY_PROMPT


def extract_keywords(text: str, max_chars: int = 80) -> str:
    """Simple preprocessing step used in the day-3 RunnableSequence task."""
    compact_text = re.sub(r"\s+", " ", text).strip()
    return compact_text[:max_chars]


def crawl_web_text(url: str, timeout: int = 10, max_chars: int = 4000) -> dict[str, Any]:
    """Fetch readable text and headings from a web page."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("仅支持 http/https URL。")

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; CompetitorAnalysisBot/1.0)",
    }
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or response.encoding

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    headings = [
        heading.get_text(" ", strip=True)
        for heading in soup.find_all(["h1", "h2", "h3"])
        if heading.get_text(" ", strip=True)
    ][:20]
    paragraphs = [
        p.get_text(" ", strip=True)
        for p in soup.find_all(["p", "li"])
        if p.get_text(" ", strip=True)
    ]
    text = "\n".join([title, *headings, *paragraphs]).strip()

    return {
        "url": url,
        "title": title,
        "headings": headings,
        "text": text[:max_chars],
    }


def summarize_url(url: str) -> str:
    """Crawl a URL and summarize the competitor intelligence signals with LLM."""
    page = crawl_web_text(url)
    chain = CRAWL_SUMMARY_PROMPT | get_deepseek_llm(temperature=0.2)
    response = chain.invoke({"url": url, "content": page["text"]})
    return response.content


@tool
def extract_price_info(news_text: str) -> str:
    """从新闻或网页文本中抽取竞品价格、折扣、套餐和促销信息。"""
    llm = get_deepseek_llm()
    prompt = (
        "只抽取与价格有关的信息，包括产品、原价、现价、折扣、促销时间和影响判断。"
        f"\n\n原文：{news_text}"
    )
    return llm.invoke(prompt).content


@tool
def extract_new_product(news_text: str) -> str:
    """从新闻或网页文本中抽取竞品新产品、新功能和版本发布信息。"""
    llm = get_deepseek_llm()
    prompt = f"只抽取新产品、新功能、版本发布相关信息，并判断对市场的影响。\n\n原文：{news_text}"
    return llm.invoke(prompt).content


@tool
def crawl_competitor_news(url: str) -> str:
    """抓取给定 URL 的标题和正文，并生成竞品情报摘要。"""
    return summarize_url(url)


tools = [extract_price_info, extract_new_product, crawl_competitor_news]

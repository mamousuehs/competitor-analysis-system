"""LLM client helpers for DeepSeek-compatible OpenAI API calls."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def get_deepseek_llm(temperature: float = 0.0, timeout: int = 60) -> ChatOpenAI:
    """Create a DeepSeek chat model through LangChain's OpenAI-compatible client."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("未找到 DEEPSEEK_API_KEY，请先在项目根目录 .env 中配置。")

    return ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        api_key=api_key,
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        temperature=temperature,
        timeout=timeout,
    )

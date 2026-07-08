"""Small smoke demos for the course tasks.

Run examples:
    python test.py runnable
    python test.py crawl https://example.com

LLM demos require DEEPSEEK_API_KEY in .env.
"""

from __future__ import annotations

import sys

from modules.agent_core import build_runnable_sequence
from modules.tools import crawl_web_text


def demo_runnable() -> None:
    chain = build_runnable_sequence()
    result = chain.invoke(
        {
            "product": "企业知识库系统",
            "competitor": "竞品A",
            "key_points": "竞品A降价15%，新增49元/月入门套餐，并加强权限管理、报表分析和企业集成能力。",
        }
    )
    print(result)


def demo_crawl(url: str) -> None:
    page = crawl_web_text(url)
    print("标题：", page["title"])
    print("一级摘要文本：")
    print(page["text"][:1000])


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "runnable"
    if command == "crawl":
        demo_crawl(sys.argv[2])
    else:
        demo_runnable()

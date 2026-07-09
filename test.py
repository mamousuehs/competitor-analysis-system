"""Small smoke demos for the course tasks.

Run examples:
    python test.py runnable
    python test.py crawl https://example.com

LLM demos require DEEPSEEK_API_KEY in .env.
"""

from __future__ import annotations

import sys

from modules.agent_core import build_runnable_sequence
from modules.monitoring_pipeline import build_demo_documents, run_pipeline
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


def demo_day4_pipeline() -> None:
    result = run_pipeline(build_demo_documents())
    print("第四天数据管道已完成。")
    print("原始文档数：", result["raw_documents"])
    print("分块数：", result["chunks"])
    print("清洗后行数：", result["cleaned_rows"])
    print("标注报告：", result["report"])
    print("导出文件：", result["files"])


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "runnable"
    if command == "crawl":
        demo_crawl(sys.argv[2])
    elif command == "day4":
        demo_day4_pipeline()
    else:
        demo_runnable()

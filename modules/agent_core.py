"""Agent and memory workflows for competitor intelligence conversations."""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_classic.memory import ConversationBufferMemory

from modules.llm_client import get_deepseek_llm
from modules.prompts import COMPETITOR_SUMMARY_PROMPT
from modules.tools import crawl_competitor_news, extract_keywords, extract_new_product, extract_price_info


def build_llm_chain():
    """Day-3 LLMChain style exercise using PromptTemplate + DeepSeek."""
    try:
        from langchain_classic.chains import LLMChain
    except ImportError as exc:  # pragma: no cover
        raise ImportError("LLMChain 需要安装 langchain。") from exc

    return LLMChain(llm=get_deepseek_llm(temperature=0.2), prompt=COMPETITOR_SUMMARY_PROMPT)


def build_runnable_sequence():
    """RunnableSequence exercise with a keyword preprocessing step."""
    return (
        RunnableLambda(
            lambda x: {
                "product": x["product"],
                "competitor": x["competitor"],
                "key_points": extract_keywords(x["key_points"]),
            }
        )
        | COMPETITOR_SUMMARY_PROMPT
        | get_deepseek_llm(temperature=0.2)
        | StrOutputParser()
    )


class MemoryCompetitorAgent:
    """Small LangChain-compatible conversation wrapper with memory and tools."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=False)
        self.llm = get_deepseek_llm(temperature=0.1)

    def _maybe_use_tool(self, user_input: str) -> str:
        if user_input.startswith("http://") or user_input.startswith("https://"):
            return crawl_competitor_news.invoke(user_input)
        if "抓取" in user_input and "http" in user_input:
            url = user_input[user_input.find("http") :].split()[0]
            return crawl_competitor_news.invoke(url)
        if any(word in user_input for word in ("价格", "降价", "涨价", "促销", "套餐")):
            return extract_price_info.invoke(user_input)
        if any(word in user_input for word in ("新产品", "新功能", "发布", "版本")):
            return extract_new_product.invoke(user_input)
        return ""

    def invoke(self, inputs: dict) -> dict:
        user_input = inputs["input"]
        memory_vars = self.memory.load_memory_variables({})
        tool_context = self._maybe_use_tool(user_input)
        prompt = f"""你是竞品情报分析助手，请结合历史对话、工具上下文和用户问题回答。

历史对话：
{memory_vars.get("chat_history", "")}

工具上下文：
{tool_context or "无"}

用户问题：
{user_input}
"""
        output = self.llm.invoke(prompt).content
        self.memory.save_context({"input": user_input}, {"output": output})
        return {"input": user_input, "output": output, "tool_context": tool_context}


def build_agent_executor(verbose: bool = True) -> MemoryCompetitorAgent:
    """Create a multi-turn competitor intelligence assistant with memory."""
    return MemoryCompetitorAgent(verbose=verbose)


def run_memory_chat_demo() -> None:
    """Small interactive console for the day-3 Memory + crawler task."""
    agent_executor = build_agent_executor(verbose=True)
    print("竞品情报多轮对话已启动。输入 exit 退出。")

    while True:
        user_input = input("\n你：").strip()
        if user_input.lower() in {"exit", "quit", "q"}:
            break
        if not user_input:
            continue
        result = agent_executor.invoke({"input": user_input})
        print(f"助手：{result['output']}")


if __name__ == "__main__":
    sequence = build_runnable_sequence()
    print(
        sequence.invoke(
            {
                "product": "企业知识库系统",
                "competitor": "竞品A",
                "key_points": "竞品A降价15%，推出49元/月入门套餐，同时加强企业版权限管理和报表功能。",
            }
        )
    )

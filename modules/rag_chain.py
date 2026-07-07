# rag_chain.py
import os
import json
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda

env_path = r"B:\MySoftwarework\Python\MyCode\CAS\competitor-analysis-system\.env"
load_dotenv(dotenv_path=env_path)

api_key = os.environ.get("DEEPSEEK_API_KEY")

# 初始化 LLM
LLM = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=api_key,
    base_url="https://api.deepseek.com/v1",
    temperature=0.0,
    timeout=30,
)

# 提示词 + 解析器
parser = JsonOutputParser()
prompt = PromptTemplate(
    template="""你是一个专业的竞品价格情报分析师。请从以下上下文中提取商品价格变动信息，并严格按照指定的JSON格式输出。

上下文：
{context}

请输出以下JSON结构（必须是纯JSON，不要添加markdown标记、代码块符号或任何解释文字）：
{{
  "query": "用户的查询意图简要概括",
  "analysis_time": "分析时间（当前日期）",
  "price_changes": [
    {{
      "product": "产品名称（精确完整）",
      "change_type": "降价/涨价/新上市/促销活动",
      "old_price": 原价数字（无单位，若无则为null）,
      "new_price": 现价数字（无单位）,
      "price_unit": "元/年 或 元/月",
      "change_percentage": "变动百分比（如-15%，若无则为null）",
      "effective_date": "生效日期（YYYY-MM-DD格式，若无则为null）",
      "source": "信息来源",
      "impact_assessment": "对本企业的影响评估（高/中/低）"
    }}
  ],
  "summary": "所有价格变动的综合分析摘要（不超过200字）"
}}

提取规则：
1. 只提取明确提及价格数字的产品
2. 如果同一产品有多次价格变动，分别列出
3. 金额只保留数字，如999元/年 → old_price: 999, price_unit: "元/年"
4. 变动百分比需计算或直接提取，如"降价15%" → change_percentage: "-15%"
5. 影响评估根据价格变动幅度判断：降价>10%为高影响，5-10%为中影响，<5%为低影响
6. 新上市产品old_price设为null
7. 必须是合法JSON，字符串使用双引号""",
    input_variables=["context"],
)


def build_rag_chain():
    """延迟加载嵌入模型和向量库"""
    print("开始加载嵌入模型...")
    embeddings = HuggingFaceEmbeddings(
        model_name=r"B:\MySoftwarework\Python\MyCode\CAS\competitor-analysis-system\models\bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print("嵌入模型加载完成")

    print("加载向量数据库...")
    vector_db = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
    )
    retriever = vector_db.as_retriever(search_kwargs={"k": 4})
    print("数据库加载完成")

    # 组装 LCEL 链
    rag_chain = (
            {
                "context": retriever | RunnableLambda(lambda docs: "\n".join(doc.page_content for doc in docs))
            }
            | prompt
            | LLM
            | parser
    )

    return rag_chain

# 格式化输出 并且将结果保存到price_intelligence.json文件当中
def format_output(result):
    """格式化输出价格情报报告"""
    print("\n" + "=" * 70)
    print("                    📊 竞品价格情报分析报告")
    print("=" * 70)

    print(f"\n  🔍 查询意图: {result['query']}")
    print(f"  📅 分析时间: {result['analysis_time']}")
    print(f"  📊 变动数量: {len(result['price_changes'])} 条")

    print(f"\n  {'─' * 66}")
    print(f"  📋 价格变动明细")
    print(f"  {'─' * 66}")

    for i, change in enumerate(result['price_changes'], 1):
        print(f"\n  【{i}】{change['product']}")
        print(f"      ├─ 变动类型: {change['change_type']}")

        if change['old_price'] is not None:
            print(f"      ├─ 原    价: ¥{change['old_price']}/{change['price_unit'].split('/')[1]}")
        else:
            print(f"      ├─ 原    价: 无")

        print(f"      ├─ 现    价: ¥{change['new_price']}/{change['price_unit'].split('/')[1]}")

        if change['change_percentage']:
            print(f"      ├─ 变    幅: {change['change_percentage']}")

        if change['effective_date']:
            print(f"      ├─ 生效日期: {change['effective_date']}")
        else:
            print(f"      ├─ 生效日期: 未明确")

        print(f"      ├─ 信息来源: {change['source']}")

        impact_icon = "🔴" if change['impact_assessment'] == '高' else "🟡" if change[
                                                                                 'impact_assessment'] == '中' else "🟢"
        print(f"      ├─ 影响评估: {impact_icon} {change['impact_assessment']}")
        print(f"      └─ 价格单位: {change['price_unit']}")

    print(f"\n  {'─' * 66}")
    print(f"  💡 综合分析")
    print(f"  {'─' * 66}")

    summary = result['summary']
    line_width = 60
    for i in range(0, len(summary), line_width):
        if i == 0:
            print(f"  {summary[i:i + line_width]}")
        else:
            print(f"  {summary[i:i + line_width]}")

    print(f"\n{'=' * 70}")

    with open('price_intelligence.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  ✅ 完整报告已保存至: price_intelligence.json")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    rag_chain = build_rag_chain()
    print("开始执行 RAG 查询...")
    result = rag_chain.invoke("竞品A最近有什么价格变动？")
    format_output(result)
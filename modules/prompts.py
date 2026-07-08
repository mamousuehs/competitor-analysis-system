"""Prompt templates used by the competitor intelligence system."""

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

PRICE_JSON_PARSER = JsonOutputParser()

PRICE_INTELLIGENCE_PROMPT = PromptTemplate(
    template="""你是一个专业的竞品价格情报分析师。请从上下文中提取商品价格变动信息，并严格输出合法 JSON。

用户问题：
{query}

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
      "new_price": 现价数字（无单位，若无则为null）,
      "price_unit": "元/年 或 元/月，若无则为null",
      "change_percentage": "变动百分比（如-15%，若无则为null）",
      "effective_date": "生效日期（YYYY-MM-DD格式，若无则为null）",
      "source": "信息来源",
      "impact_assessment": "对本企业的影响评估（高/中/低）"
    }}
  ],
  "summary": "所有价格变动的综合分析摘要（不超过200字）"
}}

提取规则：
1. 只提取明确提及价格数字的产品或套餐
2. 如果同一产品有多次价格变动，分别列出
3. 金额只保留数字，如999元/年 -> old_price: 999, price_unit: "元/年"
4. 变动百分比需计算或直接提取，如"降价15%" -> change_percentage: "-15%"
5. 影响评估根据价格变动幅度判断：降价>10%为高影响，5-10%为中影响，<5%为低影响
6. 新上市产品 old_price 设为 null
7. 没有找到价格变动时，price_changes 输出空数组，并在 summary 中说明原因""",
    input_variables=["query", "context"],
)

COMPETITOR_SUMMARY_PROMPT = PromptTemplate(
    input_variables=["product", "competitor", "key_points"],
    template="""你是一个竞品情报分析师。请根据以下信息，生成一份简洁的竞品分析摘要。

产品名称：{product}
竞争对手：{competitor}
关键差异点：{key_points}

输出要求：
- 先给出3条要点
- 再给出不超过120字的综合判断
- 语气专业、直接，避免空泛表述""",
)

CRAWL_SUMMARY_PROMPT = PromptTemplate(
    input_variables=["url", "content"],
    template="""你是一个竞品情报分析师。以下是从网页抓取的文本，请提炼与竞品动态、价格、产品、市场动作有关的信息。

来源URL：{url}

网页文本：
{content}

请输出：
1. 关键标题或事实
2. 可能的业务影响
3. 后续需要人工确认的点""",
)

AGENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是竞品情报分析助手。你可以调用工具抽取价格、产品和网页信息；回答要基于已知上下文，必要时说明不确定性。",
        ),
        ("placeholder", "{chat_history}"),
        ("user", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

"""营销分析助手：Claude + 工具调用 + RAG。

Claude 可以使用三个工具：
- search_playbook：检索策略文档（RAG）
- get_segment_profiles：调用部署在线上的 API，获取四个客群的概况
- predict_customer：调用线上 API，预测某个客户的响应概率和所属客群

交互运行（在项目根目录）：python -m llm.assistant
"""
import json
import os

import anthropic
import httpx
from dotenv import load_dotenv

from llm.retriever import PlaybookRetriever

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://customer-intelligence-h23l.onrender.com")
MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-5-5")
MAX_TOOL_ROUNDS = 6

SYSTEM_PROMPT = """You are a marketing analytics assistant for a retail company.
You help marketers understand four customer segments and decide whom to target in campaigns.

Rules:
- Base every number and recommendation on tool results. Never invent figures.
- For questions about segments, strategy, campaigns or definitions, call search_playbook first.
  The playbook is written in English, so always write the search query in English,
  even when the user writes in another language.
- For questions about a specific customer, call predict_customer. If the user leaves out
  some fields, use the schema defaults and state briefly which values you assumed.
- When you use the playbook, name the section you relied on, e.g. (Playbook: Segment: Young Online Browsers).
- If the tools do not contain the answer, say so instead of guessing.
- Reply in the same language as the user. Keep answers concise and practical."""

TOOLS = [
    {
        "name": "search_playbook",
        "description": "Search the customer segment playbook: segment profiles, buying behaviour, "
                       "campaign history, recommended strategies, how to read model scores, "
                       "data limitations and definitions. Returns the most relevant sections.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query in English."},
                "top_k": {"type": "integer", "default": 3,
                          "description": "Number of sections to return (1-5)."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_segment_profiles",
        "description": "Get live summary statistics for the four segments from the deployed API: "
                       "customers, average income, age, spend, children, web visits, response rate, "
                       "share of customers and share of revenue.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "predict_customer",
        "description": "Predict one customer's probability of responding to the next campaign and "
                       "assign their segment, using the deployed machine learning model. "
                       "Spend fields are totals over the last 2 years. "
                       "Purchases are counts over the same period.",
        "input_schema": {
            "type": "object",
            "properties": {
                "birth_year": {"type": "integer", "description": "Between 1920 and 2000."},
                "education": {"type": "string", "enum": ["Basic", "2n Cycle", "Graduation", "Master", "PhD"],
                              "default": "Graduation"},
                "marital_status": {"type": "string",
                                   "enum": ["Single", "Married", "Together", "Divorced", "Widow"],
                                   "default": "Married"},
                "income": {"type": "number", "description": "Yearly household income in dollars."},
                "kids_at_home": {"type": "integer", "default": 0},
                "teens_at_home": {"type": "integer", "default": 0},
                "enrolled_date": {"type": "string",
                                  "description": "Date the customer joined, YYYY-MM-DD, "
                                                 "between 2012-07-30 and 2014-06-29."},
                "recency_days": {"type": "integer", "description": "Days since last purchase, 0-365."},
                "spend_wine": {"type": "number", "default": 0},
                "spend_fruit": {"type": "number", "default": 0},
                "spend_meat": {"type": "number", "default": 0},
                "spend_fish": {"type": "number", "default": 0},
                "spend_sweets": {"type": "number", "default": 0},
                "spend_gold": {"type": "number", "default": 0},
                "purchases_web": {"type": "integer", "default": 0},
                "purchases_catalog": {"type": "integer", "default": 0},
                "purchases_store": {"type": "integer", "default": 0},
                "purchases_deals": {"type": "integer", "default": 0},
                "web_visits_month": {"type": "integer", "default": 0},
                "accepted_cmp1": {"type": "integer", "enum": [0, 1], "default": 0},
                "accepted_cmp2": {"type": "integer", "enum": [0, 1], "default": 0},
                "accepted_cmp3": {"type": "integer", "enum": [0, 1], "default": 0},
                "accepted_cmp4": {"type": "integer", "enum": [0, 1], "default": 0},
                "accepted_cmp5": {"type": "integer", "enum": [0, 1], "default": 0},
                "complained": {"type": "integer", "enum": [0, 1], "default": 0},
            },
            "required": ["birth_year", "enrolled_date", "recency_days"],
        },
    },
]

retriever = PlaybookRetriever()
http = httpx.Client(base_url=API_BASE_URL, timeout=90)  # 免费套餐冷启动可能要等一会


def run_tool(name: str, args: dict):
    """执行 Claude 请求的工具，返回结果（会被转成 JSON 交回给 Claude）。"""
    try:
        if name == "search_playbook":
            return retriever.search(args["query"], top_k=min(max(args.get("top_k", 3), 1), 5))
        if name == "get_segment_profiles":
            r = http.get("/segments")
        elif name == "predict_customer":
            r = http.post("/predict", json=args)
        else:
            return {"error": f"Unknown tool: {name}"}
        if r.status_code != 200:
            return {"error": f"API returned {r.status_code}", "detail": r.text[:500]}
        return r.json()
    except httpx.HTTPError as e:
        return {"error": f"Could not reach the API: {e}"}


def ask(question: str, history: list | None = None, client: anthropic.Anthropic | None = None) -> dict:
    """回答一个问题。返回答案文本、调用过的工具，以及更新后的对话记录。"""
    client = client or anthropic.Anthropic()
    messages = list(history or []) + [{"role": "user", "content": question}]
    tool_calls = []
    nudged = False

    for _ in range(MAX_TOOL_ROUNDS):
        response = client.messages.create(
            model=MODEL, max_tokens=1500, system=SYSTEM_PROMPT, tools=TOOLS, messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            answer = "".join(b.text for b in response.content if b.type == "text").strip()
            if not answer and not nudged:
                # 偶尔模型拿到工具结果后没有输出文字，提醒它给出最终答案（只提醒一次）
                nudged = True
                messages.append({"role": "user",
                                 "content": "Please give your final answer based on the tool results above."})
                continue
            return {"answer": answer, "tool_calls": tool_calls, "messages": messages,
                    "stop_reason": response.stop_reason}

        results = []
        for block in response.content:
            if block.type == "tool_use":
                output = run_tool(block.name, block.input)
                tool_calls.append({"name": block.name, "input": block.input})
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": json.dumps(output, ensure_ascii=False)})
        messages.append({"role": "user", "content": results})

    answer = "（工具调用次数超过上限，未能得出答案）"
    return {"answer": answer, "tool_calls": tool_calls, "messages": messages}


def main():
    client = anthropic.Anthropic()
    history = []
    print(f"营销分析助手（模型：{MODEL}）。输入问题，输入 exit 退出。")
    while True:
        try:
            q = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"exit", "quit", "q"}:
            break
        if not q:
            continue
        result = ask(q, history, client)
        for call in result["tool_calls"]:
            print(f"  [调用工具] {call['name']} {json.dumps(call['input'], ensure_ascii=False)[:120]}")
        print(f"\n助手：{result['answer']}")
        history = result["messages"]


if __name__ == "__main__":
    main()

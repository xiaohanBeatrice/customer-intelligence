"""评估营销分析助手：20 个问题，自动检查答案和工具调用。

每个问题有两项检查：
- 工具：助手是否调用了合适的工具（tools 里任意一个即可）
- 答案：回答里是否包含标准答案的关键信息（must_include 里每一组至少命中一个）

运行（在项目根目录）：python -m llm.eval
结果保存到 llm/eval_results.json
"""
import json
import time
from pathlib import Path

import anthropic

from llm.assistant import MODEL, ask

PLAYBOOK_OR_API = ["search_playbook", "get_segment_profiles"]

CASES = [
    # ---- 客群事实 ----
    {"q": "Which segment generates the largest share of revenue, and how much?",
     "tools": PLAYBOOK_OR_API, "must_include": [["Affluent High-Value"], ["52"]]},
    {"q": "哪个客群对营销活动的响应率最低？是多少？",
     "tools": PLAYBOOK_OR_API, "must_include": [["Price-Sensitive New"], ["4.3", "4%"]]},
    {"q": "What share of all customers are Young Online Browsers?",
     "tools": PLAYBOOK_OR_API, "must_include": [["26.8", "27%"]]},
    {"q": "Affluent High-Value 客群的平均收入是多少？",
     "tools": PLAYBOOK_OR_API,
     "must_include": [["76,767", "76767", "76,766", "76,8", "7.7万", "77,000", "$77k"]]},
    {"q": "Apart from stores, which purchase channel matters most for Affluent High-Value customers?",
     "tools": ["search_playbook"], "must_include": [["catalog", "目录"]]},
    {"q": "Established Families 的消费里，葡萄酒占多大比例？",
     "tools": ["search_playbook"], "must_include": [["59"]]},
    {"q": "How many times per month do Young Online Browsers visit the website on average?",
     "tools": PLAYBOOK_OR_API, "must_include": [["7.3"]]},
    # ---- 跨客群发现 ----
    {"q": "Do customers who accepted earlier campaigns respond more? By how much?",
     "tools": ["search_playbook"], "must_include": [["40.8", "41%"], ["8.4", "8%"]]},
    {"q": "Should we rely on discounts to drive value? What does the data say?",
     "tools": ["search_playbook"], "must_include": [["6.6", "Affluent"]]},
    # ---- 模型 ----
    {"q": "response_probability 是什么意思？will_respond 用的阈值是多少？",
     "tools": ["search_playbook"], "must_include": [["0.5", "50%"]]},
    {"q": "How accurate is the response model? Give the ROC-AUC.",
     "tools": ["search_playbook"], "must_include": [["0.90", "0.9 ", "90%"]]},
    {"q": "如果只联系预测概率最高的 20% 客户，能覆盖多少实际响应者？",
     "tools": ["search_playbook"], "must_include": [["70"]]},
    # ---- 策略 ----
    {"q": "Price-Sensitive New 这个客群应该怎么做营销？",
     "tools": ["search_playbook"],
     "must_include": [["low-cost", "低成本", "coupon", "优惠券", "不要", "do not over-invest", "成本低"]]},
    {"q": "When should we send a win-back message to Affluent High-Value customers?",
     "tools": ["search_playbook"], "must_include": [["60"]]},
    {"q": "Which earlier campaign worked best for Young Online Browsers?",
     "tools": ["search_playbook"],
     "must_include": [["campaign 3", "第 3", "第3", "活动 3", "活动3", "第三", "cmp3"]]},
    # ---- 定义与局限 ----
    {"q": "数据里的 2n Cycle 学历是什么意思？是怎么处理的？",
     "tools": ["search_playbook"],
     "must_include": [["Undergraduate", "本科以下", "未定义", "没有定义", "not define"]]},
    {"q": "How many customers per group do we need for an A/B test to detect a lift from 15.6% to 20%?",
     "tools": ["search_playbook"], "must_include": [["600"]]},
    # ---- 单个客户预测 ----
    {"q": "A customer born in 1970 with a PhD, single, income $85,000, joined on 2012-09-01, last purchase "
          "5 days ago. Over 2 years: wine $900, meat $600, fish $120, fruit $80, sweets $80, gold $90; "
          "6 web, 7 catalog and 10 store purchases, 1 deal purchase, 2 web visits a month, accepted "
          "campaigns 1 and 5. Which segment, and should we target them?",
     "tools": ["predict_customer"], "must_include": [["Affluent High-Value"]]},
    {"q": "一位 1985 年出生的客户，已婚，年收入 28000，家里有一个小孩，2014-02-10 入会，90 天没有消费了。"
          "两年里只买了 3 次门店、1 次网购（其中 1 次用了折扣），总共花了大约 30 美元，每月访问网站 7 次。"
          "他属于哪个客群？值得给他发营销吗？",
     "tools": ["predict_customer"], "must_include": [["Young Online Browsers", "Price-Sensitive New"]]},
    # ---- 超出范围 ----
    {"q": "What was the company's total revenue in 2025?",
     "tools": [], "must_include": [["not", "no data", "don't", "doesn't", "cannot", "can't",
                                    "无法", "没有", "不包含"]]},
]


def check(case: dict, result: dict) -> dict:
    used = [c["name"] for c in result["tool_calls"]]
    tool_ok = (not case["tools"]) or any(t in used for t in case["tools"])
    text = result["answer"].lower()
    answer_ok = all(any(opt.lower() in text for opt in group) for group in case["must_include"])
    return {"tool_ok": tool_ok, "answer_ok": answer_ok, "passed": tool_ok and answer_ok, "tools_used": used}


def main():
    client = anthropic.Anthropic()
    rows = []
    for i, case in enumerate(CASES, 1):
        start = time.time()
        result = ask(case["q"], client=client)
        row = {"id": i, "question": case["q"], **check(case, result),
               "seconds": round(time.time() - start, 1), "answer": result["answer"],
               "stop_reason": result.get("stop_reason")}
        rows.append(row)
        mark = "PASS" if row["passed"] else "FAIL"
        tool_mark = "✓" if row["tool_ok"] else "✗"
        answer_mark = "✓" if row["answer_ok"] else "✗"
        print(f"[{mark}] {i:2d}. 工具{tool_mark} 答案{answer_mark}  {row['seconds']:5.1f}s  {case['q'][:50]}")

    n = len(rows)
    summary = {
        "model": MODEL,
        "n_questions": n,
        "pass_rate": round(sum(r["passed"] for r in rows) / n, 3),
        "tool_accuracy": round(sum(r["tool_ok"] for r in rows) / n, 3),
        "answer_accuracy": round(sum(r["answer_ok"] for r in rows) / n, 3),
        "avg_seconds": round(sum(r["seconds"] for r in rows) / n, 1),
    }
    print(f"\n模型 {MODEL}：通过 {summary['pass_rate']:.0%}，工具选择 {summary['tool_accuracy']:.0%}，"
          f"答案正确 {summary['answer_accuracy']:.0%}，平均 {summary['avg_seconds']}s/题")
    out = Path(__file__).with_name("eval_results.json")
    out.write_text(json.dumps({"summary": summary, "results": rows}, ensure_ascii=False, indent=2))
    print(f"详细结果已保存到 {out}")


if __name__ == "__main__":
    main()

"""把一个客户的原始信息转换成模型需要的特征。

这里的计算必须和 dbt 模型 customer_features.sql 保持一致，
api_tests/test_features.py 会自动核对两边的结果。
"""
from datetime import date
from decimal import ROUND_HALF_UP, Decimal

import pandas as pd

# 训练数据里最晚的入会日期。年龄和入会时长都以它为基准，
# 和 customer_features.sql 里的 max(enrolled_date) 一致。
REFERENCE_DATE = date(2014, 6, 29)


def round_half_up(x: float, digits: int = 1) -> float:
    """和 SQL 的 round() 一样四舍五入（Python 自带的 round 是「四舍六入五成双」）。"""
    return float(Decimal(str(x)).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP))


def education_group(education: str) -> str:
    if education in ("Master", "PhD"):
        return "Postgraduate"
    if education == "Graduation":
        return "Graduate"
    return "Undergraduate"


def clean_marital_status(status: str) -> str:
    return "Single" if status in ("Alone", "Absurd", "YOLO") else status


def preferred_channel(web: int, catalog: int, store: int) -> str:
    if web + catalog + store == 0:
        return "None"
    if store >= max(web, catalog):
        return "Store"
    if web >= catalog:
        return "Web"
    return "Catalog"


def build_features(c: dict) -> pd.DataFrame:
    """输入一个客户的原始字段（dict），返回只有一行的特征表。"""
    marital = clean_marital_status(c["marital_status"])
    living_with = "Partner" if marital in ("Married", "Together") else "Alone"
    children = c["kids_at_home"] + c["teens_at_home"]
    spend = ["spend_wine", "spend_fruit", "spend_meat", "spend_fish", "spend_sweets", "spend_gold"]
    total_spent = sum(c[k] for k in spend)
    total_purchases = c["purchases_web"] + c["purchases_catalog"] + c["purchases_store"]

    row = {
        **{k: c[k] for k in spend},
        "age": REFERENCE_DATE.year - c["birth_year"],
        "income": c.get("income"),
        "children": children,
        "family_size": children + (2 if living_with == "Partner" else 1),
        "tenure_days": (REFERENCE_DATE - c["enrolled_date"]).days,
        "recency_days": c["recency_days"],
        "total_spent": total_spent,
        "purchases_web": c["purchases_web"],
        "purchases_catalog": c["purchases_catalog"],
        "purchases_store": c["purchases_store"],
        "purchases_deals": c["purchases_deals"],
        "web_visits_month": c["web_visits_month"],
        "total_purchases": total_purchases,
        "avg_order_value": round_half_up(total_spent / total_purchases) if total_purchases else None,
        "total_campaigns_accepted": sum(c[f"accepted_cmp{i}"] for i in range(1, 6)),
        "complained": c["complained"],
        "education_group": education_group(c["education"]),
        "marital_status": marital,
        "preferred_channel": preferred_channel(
            c["purchases_web"], c["purchases_catalog"], c["purchases_store"]),
    }
    return pd.DataFrame([row])
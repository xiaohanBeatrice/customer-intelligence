"""核对 Python 特征计算和 dbt 的 customer_features 表是否一致。

API 用 Python 计算特征，训练用的是 dbt 算出来的特征。两边如果不一致，
模型上线后的效果就会和训练时不同（training-serving skew）。
需要先运行 dbt build；没有数据库文件时自动跳过。
"""
from pathlib import Path

import pytest

from api.features import build_features

DB = Path(__file__).resolve().parent.parent / "warehouse.duckdb"
CHECK_COLS = ["age", "children", "family_size", "tenure_days", "total_spent",
              "total_purchases", "avg_order_value", "total_campaigns_accepted",
              "education_group", "marital_status", "preferred_channel"]


@pytest.mark.skipif(not DB.exists(), reason="需要先运行 dbt build")
def test_python_features_match_dbt():
    import duckdb

    with duckdb.connect(str(DB), read_only=True) as con:
        df = con.sql("""
            select s.*, f.* exclude (customer_id)
            from stg_customers s
            join customer_features f using (customer_id)
            where f.is_model_ready
        """).df()

    mismatches = []
    for _, row in df.iterrows():
        raw = row.to_dict()
        raw["enrolled_date"] = row["enrolled_date"].date()
        mine = build_features(raw).iloc[0]
        for col in CHECK_COLS:
            a, b = mine[col], row[col]
            if isinstance(b, str):
                same = a == b
            else:
                same = abs(float(a) - float(b)) < 1e-6
            if not same:
                mismatches.append((row["customer_id"], col, a, b))

    assert not mismatches, f"发现 {len(mismatches)} 处不一致，例如：{mismatches[:5]}"
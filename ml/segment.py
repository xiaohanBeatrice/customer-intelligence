"""
客户分群：标准化 → PCA → K-Means，结果写回 DuckDB 并保存模型。

运行方式（在项目根目录）：python ml/segment.py
"""
import json
from pathlib import Path

import duckdb
import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DB_PATH = "warehouse.duckdb"
ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)

# 和原 notebook 一样的 8 个分群特征
FEATURES = [
    "income", "age", "total_spent", "children",
    "tenure_days", "total_purchases", "web_visits_month", "recency_days",
]
N_CLUSTERS = 4
SEED = 42


def load_data() -> pd.DataFrame:
    with duckdb.connect(DB_PATH, read_only=True) as con:
        return con.sql("select * from customer_features where is_model_ready").df()


def name_segments(profile: pd.DataFrame) -> dict:
    """
    按规则给群命名，而不是按 K-Means 的随机编号手动对应。

    规则：平均消费最高 → Affluent High-Value；第二 → Established Families；
    剩下两个里，平均年龄更小的 → Young Online Browsers，另一个 → Price-Sensitive New。
    """
    by_spend = profile.sort_values("total_spent", ascending=False).index.tolist()
    rest = profile.loc[by_spend[2:]].sort_values("age")
    return {
        by_spend[0]: "Affluent High-Value",
        by_spend[1]: "Established Families",
        rest.index[0]: "Young Online Browsers",
        rest.index[1]: "Price-Sensitive New",
    }


def main():
    df = load_data()
    X = df[FEATURES]

    # 1. 选 K：打印每个 K 的轮廓系数作参考
    prep = Pipeline([("scale", StandardScaler()), ("pca", PCA(n_components=3, random_state=SEED))])
    X_pca = prep.fit_transform(X)
    print("K   silhouette")
    for k in range(2, 9):
        labels = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(X_pca)
        print(f"{k}   {silhouette_score(X_pca, labels):.3f}")

    # 2. 训练最终模型（K=4，业务上更好解释）
    model = Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=3, random_state=SEED)),
        ("kmeans", KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=SEED)),
    ])
    df["cluster"] = model.fit_predict(X)

    # 3. 按规则命名
    profile = df.groupby("cluster")[FEATURES].mean()
    names = name_segments(profile)
    df["segment"] = df["cluster"].map(names)

    summary = df.groupby("segment").agg(
        customers=("customer_id", "count"),
        avg_income=("income", "mean"),
        avg_age=("age", "mean"),
        avg_spent=("total_spent", "mean"),
        avg_children=("children", "mean"),
        avg_web_visits=("web_visits_month", "mean"),
        response_rate=("responded", "mean"),
    ).round(2)
    summary["share_of_customers"] = (summary["customers"] / len(df)).round(3)
    summary["share_of_revenue"] = (
        df.groupby("segment")["total_spent"].sum() / df["total_spent"].sum()
    ).round(3)
    print("\n", summary.to_string())

    # 4. 保存：模型 + 编号到名字的映射 + 各群概况
    joblib.dump({"model": model, "features": FEATURES, "names": names},
                ARTIFACTS / "segmenter.joblib")
    (ARTIFACTS / "segment_profiles.json").write_text(
        json.dumps(summary.reset_index().to_dict(orient="records"), indent=2))

    # 5. 写回 DuckDB，方便查询和 API 使用
    with duckdb.connect(DB_PATH) as con:
        con.register("seg", df[["customer_id", "cluster", "segment"]])
        con.sql("create or replace table customer_segments as select * from seg")
    print(f"\n已保存 artifacts/segmenter.joblib，并写入表 customer_segments（{len(df)} 行）")


if __name__ == "__main__":
    main()
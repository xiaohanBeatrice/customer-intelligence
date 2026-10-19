"""预测客户是否会响应下一次营销活动(responded = 1)。

对比逻辑回归、随机森林、XGBoost，用 5 折交叉验证选最佳模型，
在留出的测试集上评估，保存模型和指标。

运行方式（在项目根目录）：python ml/train_response.py
"""
import json
from pathlib import Path

import duckdb
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

DB_PATH = "warehouse.duckdb"
ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)
SEED = 42
TARGET = "responded"

NUMERIC = [
    "age", "income", "children", "family_size", "tenure_days", "recency_days",
    "spend_wine", "spend_fruit", "spend_meat", "spend_fish", "spend_sweets", "spend_gold",
    "total_spent", "purchases_web", "purchases_catalog", "purchases_store", "purchases_deals",
    "web_visits_month", "total_purchases", "avg_order_value",
    "total_campaigns_accepted", "complained",
]
CATEGORICAL = ["education_group", "marital_status", "preferred_channel"]


def load_data() -> pd.DataFrame:
    with duckdb.connect(DB_PATH, read_only=True) as con:
        return con.sql("select * from customer_features where is_model_ready").df()


def make_preprocessor(scale: bool) -> ColumnTransformer:
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:  # 逻辑回归需要标准化；树模型不需要
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])


def main():
    df = load_data()
    X, y = df[NUMERIC + CATEGORICAL], df[TARGET]
    print(f"样本 {len(df)}，正例比例 {y.mean():.3f}")

    # 先切出 20% 测试集，整个选模型过程都不碰它
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    candidates = {
        "logistic_regression": Pipeline([
            ("prep", make_preprocessor(scale=True)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]),
        "random_forest": Pipeline([
            ("prep", make_preprocessor(scale=False)),
            ("clf", RandomForestClassifier(n_estimators=400, min_samples_leaf=3,
                                           class_weight="balanced", random_state=SEED, n_jobs=-1)),
        ]),
        "xgboost": Pipeline([
            ("prep", make_preprocessor(scale=False)),
            ("clf", XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8,
                                  scale_pos_weight=pos_weight, eval_metric="logloss",
                                  random_state=SEED)),
        ]),
    }

    # 5 折交叉验证（只用训练集）
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_results = {}
    print("\n模型                  ROC-AUC          PR-AUC")
    for name, pipe in candidates.items():
        s = cross_validate(pipe, X_train, y_train, cv=cv,
                           scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision"})
        cv_results[name] = {
            "roc_auc_mean": round(s["test_roc_auc"].mean(), 4),
            "roc_auc_std": round(s["test_roc_auc"].std(), 4),
            "pr_auc_mean": round(s["test_pr_auc"].mean(), 4),
        }
        r = cv_results[name]
        print(f"{name:20}  {r['roc_auc_mean']:.3f} ± {r['roc_auc_std']:.3f}    {r['pr_auc_mean']:.3f}")

    # 选交叉验证 ROC-AUC 最高的，在全部训练集上重新训练
    best_name = max(cv_results, key=lambda n: cv_results[n]["roc_auc_mean"])
    best = candidates[best_name].fit(X_train, y_train)

    # 在测试集上做最终评估
    proba = best.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    test_metrics = {
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "pr_auc": round(average_precision_score(y_test, proba), 4),
        "baseline_positive_rate": round(y_test.mean(), 4),
    }
    print(f"\n最佳模型：{best_name}")
    print(f"测试集 ROC-AUC {test_metrics['roc_auc']:.3f}，PR-AUC {test_metrics['pr_auc']:.3f}"
          f"（随机猜测的 PR-AUC ≈ {test_metrics['baseline_positive_rate']:.3f}）")
    print(classification_report(y_test, pred, digits=3))

    # 业务视角：按预测概率排序，联系前 20% 的客户，能覆盖多少真正会响应的人？
    top = pd.Series(proba).rank(ascending=False, pct=True) <= 0.2
    capture = y_test.reset_index(drop=True)[top].sum() / y_test.sum()
    test_metrics["top20pct_capture_rate"] = round(float(capture), 4)
    print(f"联系预测概率最高的 20% 客户，可覆盖 {capture:.1%} 的实际响应者")

    joblib.dump({"model": best, "features": NUMERIC + CATEGORICAL, "model_name": best_name},
                ARTIFACTS / "response_model.joblib")
    (ARTIFACTS / "response_metrics.json").write_text(json.dumps({
        "best_model": best_name, "cv": cv_results, "test": test_metrics,
        "n_train": len(X_train), "n_test": len(X_test),
    }, indent=2))
    print("\n已保存 artifacts/response_model.joblib 和 artifacts/response_metrics.json")


if __name__ == "__main__":
    main()
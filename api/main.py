"""客户分群与营销响应预测 API。

本地运行（在项目根目录）：uvicorn api.main:app --reload
然后打开 http://127.0.0.1:8000/docs
"""
import json
from datetime import date
from pathlib import Path
from typing import Literal, Optional

import joblib
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from api.features import build_features

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"

# 启动时加载一次模型，之后每个请求直接用
segmenter = joblib.load(ARTIFACTS / "segmenter.joblib")
responder = joblib.load(ARTIFACTS / "response_model.joblib")
segment_profiles = json.loads((ARTIFACTS / "segment_profiles.json").read_text())
metrics = json.loads((ARTIFACTS / "response_metrics.json").read_text())

app = FastAPI(
    title="Customer Intelligence API",
    description="客户分群与营销活动响应预测（基于 Kaggle Customer Personality Analysis 公开数据集）",
    version="1.0.0",
)


class Customer(BaseModel):
    """一个客户的原始信息，字段含义和 stg_customers 表一致。"""
    birth_year: int = Field(..., ge=1920, le=2000, examples=[1975])
    education: Literal["Basic", "2n Cycle", "Graduation", "Master", "PhD"] = "Graduation"
    marital_status: Literal["Single", "Married", "Together", "Divorced", "Widow",
                            "Alone", "Absurd", "YOLO"] = "Married"
    income: Optional[float] = Field(None, ge=0, le=200000, examples=[65000])
    kids_at_home: int = Field(0, ge=0, le=5)
    teens_at_home: int = Field(0, ge=0, le=5)
    enrolled_date: date = Field(..., examples=["2013-03-15"])
    recency_days: int = Field(..., ge=0, le=365, examples=[20])
    spend_wine: float = Field(0, ge=0)
    spend_fruit: float = Field(0, ge=0)
    spend_meat: float = Field(0, ge=0)
    spend_fish: float = Field(0, ge=0)
    spend_sweets: float = Field(0, ge=0)
    spend_gold: float = Field(0, ge=0)
    purchases_web: int = Field(0, ge=0)
    purchases_catalog: int = Field(0, ge=0)
    purchases_store: int = Field(0, ge=0)
    purchases_deals: int = Field(0, ge=0)
    web_visits_month: int = Field(0, ge=0)
    accepted_cmp1: int = Field(0, ge=0, le=1)
    accepted_cmp2: int = Field(0, ge=0, le=1)
    accepted_cmp3: int = Field(0, ge=0, le=1)
    accepted_cmp4: int = Field(0, ge=0, le=1)
    accepted_cmp5: int = Field(0, ge=0, le=1)
    complained: int = Field(0, ge=0, le=1)


def assign_segment(features) -> str:
    X = features[segmenter["features"]].copy()
    # 分群模型训练时没有缺失收入；缺失时用训练数据的平均收入代替
    if X["income"].isna().any():
        X["income"] = X["income"].fillna(segmenter["model"].named_steps["scale"].mean_[0])
    cluster = int(segmenter["model"].predict(X)[0])
    return segmenter["names"][cluster]

@app.get("/", include_in_schema=False)
def root():
    """打开根地址时自动跳转到接口文档。"""
    return RedirectResponse(url="/docs")

@app.get("/health")
def health():
    """服务状态和模型信息。"""
    return {
        "status": "ok",
        "response_model": responder["model_name"],
        "test_roc_auc": metrics["test"]["roc_auc"],
    }


@app.get("/segments")
def list_segments():
    """四个客群的概况（人数、平均收入、消费、响应率等）。"""
    return segment_profiles


@app.post("/segment")
def segment(customer: Customer):
    """判断一个客户属于哪个客群。"""
    features = build_features(customer.model_dump())
    return {"segment": assign_segment(features)}


@app.post("/predict")
def predict(customer: Customer):
    """预测一个客户响应下一次营销活动的概率，同时给出所属客群。"""
    features = build_features(customer.model_dump())
    proba = float(responder["model"].predict_proba(features[responder["features"]])[0, 1])
    return {
        "response_probability": round(proba, 4),
        "will_respond": proba >= 0.5,
        "segment": assign_segment(features),
    }
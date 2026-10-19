"""API 接口测试。运行：pytest"""
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

SEGMENTS = {"Affluent High-Value", "Established Families",
            "Price-Sensitive New", "Young Online Browsers"}

# 一个高收入、高消费、最近刚买过的客户
HIGH_VALUE = {
    "birth_year": 1970, "education": "PhD", "marital_status": "Single",
    "income": 85000, "kids_at_home": 0, "teens_at_home": 0,
    "enrolled_date": "2012-09-01", "recency_days": 5,
    "spend_wine": 900, "spend_fruit": 80, "spend_meat": 600,
    "spend_fish": 120, "spend_sweets": 80, "spend_gold": 90,
    "purchases_web": 6, "purchases_catalog": 7, "purchases_store": 10,
    "purchases_deals": 1, "web_visits_month": 2,
    "accepted_cmp1": 1, "accepted_cmp5": 1,
}

# 一个低收入、低消费、很久没买的客户
LOW_VALUE = {
    "birth_year": 1985, "education": "Graduation", "marital_status": "Married",
    "income": 28000, "kids_at_home": 1, "teens_at_home": 0,
    "enrolled_date": "2014-02-10", "recency_days": 90,
    "spend_wine": 10, "spend_fruit": 2, "spend_meat": 8,
    "spend_fish": 3, "spend_sweets": 1, "spend_gold": 5,
    "purchases_web": 1, "purchases_catalog": 0, "purchases_store": 3,
    "purchases_deals": 1, "web_visits_month": 7,
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_list_segments():
    r = client.get("/segments")
    assert r.status_code == 200
    assert {s["segment"] for s in r.json()} == SEGMENTS


def test_segment_returns_known_name():
    r = client.post("/segment", json=HIGH_VALUE)
    assert r.status_code == 200
    assert r.json()["segment"] in SEGMENTS


def test_high_value_customer_segment():
    assert client.post("/segment", json=HIGH_VALUE).json()["segment"] == "Affluent High-Value"


def test_predict_probability_in_range():
    body = client.post("/predict", json=HIGH_VALUE).json()
    assert 0 <= body["response_probability"] <= 1
    assert body["segment"] in SEGMENTS


def test_high_value_more_likely_to_respond():
    high = client.post("/predict", json=HIGH_VALUE).json()["response_probability"]
    low = client.post("/predict", json=LOW_VALUE).json()["response_probability"]
    assert high > low


def test_missing_income_is_handled():
    customer = {**LOW_VALUE, "income": None}
    r = client.post("/predict", json=customer)
    assert r.status_code == 200


def test_invalid_input_rejected():
    bad = {**HIGH_VALUE, "education": "Kindergarten"}
    assert client.post("/predict", json=bad).status_code == 422
    assert client.post("/predict", json={**HIGH_VALUE, "recency_days": -1}).status_code == 422
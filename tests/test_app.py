import pytest
from fastapi.testclient import TestClient

from app import app

client = TestClient(app)
NORMAL = {"credit_score": 760, "annual_income_k": 90.0, "loan_amount_k": 15.0, "debt_to_income": 0.18, "employment_years": 8.0, "missed_payments_2y": 0}
RISKY = {"credit_score": 560, "annual_income_k": 35.0, "loan_amount_k": 30.0, "debt_to_income": 0.55, "employment_years": 0.5, "missed_payments_2y": 4}


# ---------- given tests ----------
def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_model_info_has_threshold():                            # checks TODO 3c
    body = client.get("/model-info").json()
    assert {"model_source", "threshold", "features"} <= set(body)


def test_predict_returns_valid_response():
    r = client.post("/predict", json=NORMAL)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["probability"] <= 1
    assert isinstance(body["flagged"], bool)


def test_risky_case_scores_higher_than_normal():
    normal = client.post("/predict", json=NORMAL).json()
    risky = client.post("/predict", json=RISKY).json()
    assert risky["probability"] > normal["probability"]
    assert risky["flagged"] is True


def test_actions_are_filled_in_and_differ():                    # checks TODO 3b
    normal = client.post("/predict", json=NORMAL).json()["recommended_action"]
    risky = client.post("/predict", json=RISKY).json()["recommended_action"]
    assert "TODO" not in (normal + risky)
    assert normal != risky


# ---------- your tests ----------
def test_out_of_range_input_is_rejected():
    # TODO 4a: send a request with an impossible value and assert the API answers 422.
    # HINT: r = client.post("/predict", json={**NORMAL, "credit_score": 100}); assert r.status_code == 422
    # THINK: test both edges? (a value just below your minimum AND just above your maximum)
    bad_values = [("credit_score", 100), ("credit_score", 900), ("debt_to_income", -0.1), ("debt_to_income", 1.5),
                  ("employment_years", -1), ("employment_years", 61), ("missed_payments_2y", -1), ("missed_payments_2y", 51)]
    for field, value in bad_values:
        r = client.post("/predict", json={**NORMAL, field: value})
        assert r.status_code == 422, f"{field}={value} should be rejected, got {r.status_code}"


def test_missing_field_is_rejected():
    # TODO 4b: remove one field from NORMAL, send it, and assert 422.
    # HINT: request = dict(NORMAL); request.pop("missed_payments_2y")
    for field in NORMAL:
        request = dict(NORMAL)
        request.pop(field)
        r = client.post("/predict", json=request)
        assert r.status_code == 422, f"a request without {field} should be rejected, got {r.status_code}"


def test_my_own_idea():
    # TODO 4c (optional, recommended): write a test of your own. Ideas:
    #   - a higher missed_payments_2y never LOWERS the probability (a "behaviour" test)
    #   - a text value such as "credit_score": "high" is rejected with 422
    #   - /predict answers in under 200 ms
    #   - changing a value the model should ignore (e.g. sending it twice) gives the same answer
    # Behaviour test: more missed payments must never LOWER the predicted risk (all else equal).
    probabilities = [client.post("/predict", json={**NORMAL, "missed_payments_2y": m}).json()["probability"]
                     for m in (0, 2, 4, 6, 8)]
    assert probabilities == sorted(probabilities), f"risk should not fall as missed payments rise: {probabilities}"

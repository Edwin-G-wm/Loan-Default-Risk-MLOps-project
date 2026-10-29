"""FastAPI service for Loan Default Risk."""
import logging
import os
import time

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

FEATURES = ['credit_score', 'annual_income_k', 'loan_amount_k', 'debt_to_income', 'employment_years', 'missed_payments_2y']

# TODO 3d: use the same THRESHOLD you chose in train.py (TODO 2a). Replace "____" with that number, e.g. "0.35".
# THINK: two copies of one number can drift apart. How could the API read it from MLflow or metrics.json?
DEFAULT_THRESHOLD = "0.35"          # same value as THRESHOLD in train.py (TODO 2a)
if DEFAULT_THRESHOLD == "____":
    raise RuntimeError("TODO 3d in app.py: set DEFAULT_THRESHOLD to the threshold you chose in train.py")
THRESHOLD = float(os.getenv("THRESHOLD", DEFAULT_THRESHOLD))
DECLINE_THRESHOLD = 0.7              # from this probability we suggest declining or a smaller loan (TODO 3b)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("loan-risk-api")


def load_model():
    uri = os.getenv("MODEL_URI")                 # e.g. models:/loan-default-model@production
    if uri:
        import mlflow.sklearn
        return mlflow.sklearn.load_model(uri), uri
    path = os.getenv("MODEL_PATH", "model.joblib")
    return joblib.load(path), path


model, MODEL_SOURCE = load_model()
log.info("Model loaded from %s", MODEL_SOURCE)

app = FastAPI(title="Loan Default Risk API", version="1.0.0")


# TODO 3a: set limits (ge = minimum, le = maximum) so impossible values get a 422 error.
# One field is done for you as an example. Training data ranges (from df.describe() in Part 1):
#   credit_score               363 - 850
#   annual_income_k            12 - 353.6
#   loan_amount_k              1.6 - 176.8
#   debt_to_income             0 - 0.73
#   employment_years           0 - 45
#   missed_payments_2y         0 - 9
# Options for how wide to make the limits:
#   Option A: exactly the training min/max  -> safest for the model, but rejects some real requests
#   Option B: real-world limits             -> e.g. 0-1 for yes/no fields, 0-23 for hours
# THINK: what should happen to a value that is possible but outside the training data? Reject? Warn?
# OUR CHOICE: Option B, the real-world limits from the project brief. A possible value outside the training
#   range (e.g. income above 353.6k) is accepted; the model extrapolates, so production should also log a warning.
class Request(BaseModel):
    credit_score: float = Field(..., ge=300, le=850, description="Credit score (300-850)", examples=[560])
    annual_income_k: float = Field(..., ge=0, le=2000, description="Annual income (thousands)", examples=[35.0])
    loan_amount_k: float = Field(..., ge=0.5, le=500, description="Loan amount (thousands)", examples=[30.0])
    debt_to_income: float = Field(..., ge=0, le=1, description="Monthly debt / monthly income", examples=[0.55])
    employment_years: float = Field(..., ge=0, le=60, description="Years in current job", examples=[0.5])
    missed_payments_2y: int = Field(..., ge=0, le=50, description="Missed payments in the last 2 years", examples=[4])


class Prediction(BaseModel):
    probability: float
    flagged: bool
    recommended_action: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    # TODO 3c: return a dict with the keys "model_source", "threshold" and "features".
    # HINT: the values are MODEL_SOURCE, THRESHOLD and FEATURES (all defined above).
    # THINK: who would call this endpoint, and when? (someone debugging a strange prediction?)
    return {"model_source": MODEL_SOURCE, "threshold": THRESHOLD, "features": FEATURES}


@app.post("/predict", response_model=Prediction)
def predict(request: Request):
    start = time.perf_counter()
    X = pd.DataFrame([request.model_dump()])[FEATURES]
    probability = float(model.predict_proba(X)[0, 1])
    flagged = probability >= THRESHOLD

    # TODO 3b: turn the prediction into an action for the loan officer and, indirectly, the applicant.
    #   Option A (2 levels): if flagged -> "Send to a loan officer for review", else "Approve"
    #   Option B (3 levels): also "Decline, or offer a smaller loan" when probability >= 0.7
    #   Option C (your own): e.g. include the probability in the message
    # THINK: who reads this? What would make them trust (or ignore) it?
    if probability >= DECLINE_THRESHOLD:
        action = "Decline, or offer a smaller loan"
    elif flagged:
        action = "Send to a loan officer for review"
    else:
        action = "Approve"

    log.info("predict p=%.3f flagged=%s latency_ms=%.1f", probability, flagged, (time.perf_counter() - start) * 1000)
    return Prediction(probability=round(probability, 4), flagged=flagged, recommended_action=action)

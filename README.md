# Loan Default Risk

An end-to-end MLOps project: model tracked and registered in MLflow, served with FastAPI, containerised with
Docker and tested on every push with GitHub Actions.

| File | Purpose |
|---|---|
| `data.py` | Generates the dataset |
| `train.py` | Trains candidate models, tracks them in MLflow, registers the best as `@production` |
| `app.py` | FastAPI service: `/predict`, `/health`, `/model-info`, docs at `/docs` |
| `tests/` | Data and API tests (pytest) |
| `Dockerfile` | Container for the API |
| `.github/workflows/ci.yml` | Train, test, build and smoke-test on every push |

```bash
pip install -r requirements.txt
python train.py && python -m pytest -q
uvicorn app:app --reload                          # http://127.0.0.1:8000/docs
docker build -t loan-risk-api . && docker run -p 8000:8000 loan-risk-api
```

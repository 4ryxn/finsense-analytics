# FinSense Analytics Agent Notes

This repository is intentionally a compact single-user Streamlit portfolio project.

Do not expand the scope into an enterprise application. In particular, do not add React, FastAPI, authentication, a database, Docker, Redis, Celery, OCR, paid APIs, LLMs, admin dashboards, notifications, Power BI, deep learning, cloud infrastructure, or microservices.

Keep changes small, typed, testable, and aligned with the existing Python modules. Prefer Pandas, NumPy, SciPy, scikit-learn, Plotly, Streamlit, pytest, and Ruff only.

Current V2 modules include scoring, scenarios, and reporting. Preserve deterministic local processing and validate with:

- `python scripts/generate_sample_data.py`
- `python -m ruff format .`
- `python -m ruff format --check .`
- `python -m ruff check .`
- `python -m pytest`
- Streamlit local health check
- `git diff --check`

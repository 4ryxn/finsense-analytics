# FinSense Analytics

FinSense Analytics is a deliberately compact Streamlit portfolio project for INR personal-finance analytics, lightweight machine learning, and data-quality reporting.

## Features

- Built-in deterministic 36-month sample dataset
- CSV upload processed in memory only
- ETL validation, standardization, duplicate removal, and rejected-row reporting
- Overview KPIs, monthly cash flow, category mix, and recent transactions
- Data Quality tab with downloadable cleaned CSV
- Exploratory analysis for trends, categories, merchants, timing, histograms, boxplots, and summary statistics
- Next-month expense forecasting with chronological validation
- Isolation Forest anomaly detection with factual explanations
- Rule-based insights from actual calculations

## Stack

Python 3.11+, Streamlit, Pandas, NumPy, SciPy, scikit-learn, Plotly, joblib, pytest, and Ruff.

## Architecture

- `app.py`: Streamlit entry point
- `src/finsense/etl.py`: CSV contract validation and cleaning report
- `src/finsense/analytics.py`: KPIs, summaries, EDA aggregations
- `src/finsense/features.py`: monthly features, lags, rolling averages, cyclical month features
- `src/finsense/forecasting.py`: baseline, Linear Regression, Random Forest, holdout metrics, permutation importance
- `src/finsense/anomalies.py`: deterministic Isolation Forest pipeline
- `src/finsense/insights.py`: deterministic rule-based insights
- `src/finsense/ui.py`: Streamlit layout and Plotly views
- `scripts/generate_sample_data.py`: deterministic non-sensitive sample data generator

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python scripts/generate_sample_data.py
streamlit run app.py
```

## CSV Schema

Required columns:

- `date`: parseable date
- `transaction_type`: `income` or `expense`
- `category`: transaction category
- `merchant`: merchant name; missing values become `Unknown`
- `amount`: positive numeric value
- `payment_method`: payment method

Optional:

- `transaction_id`: duplicates are removed deterministically where present

Amounts are positive and interpreted as INR for this MVP. Invalid date, amount, type, or critical category records are rejected.

## ML Methodology

Forecasting aggregates expenses monthly and builds prior-period features only, including lags, rolling averages, growth, diversity, frequency, and cyclical month encodings. It uses chronological holdout validation, never random shuffling. A seasonal baseline is compared with Linear Regression and Random Forest models. The best model is selected by holdout MAE, with RMSE and R-squared reported where valid. Forecasts are estimates, not guaranteed financial advice.

Anomaly detection uses a scikit-learn pipeline with scaled numeric features and one-hot encoded categorical features feeding a deterministic Isolation Forest. Anomalies mean unusual transactions, not fraud.

## Evaluation Metrics

- MAE: average absolute forecast error in INR
- RMSE: larger errors receive more penalty
- R-squared: reported when the holdout size makes it meaningful

## Limitations

This is a small-data educational dashboard. It does not persist uploads, authenticate users, call external APIs, provide financial advice, or claim statistically valid confidence intervals.

## Testing

```bash
python -m ruff format --check .
python -m ruff check .
python -m pytest
```

## Streamlit Deployment

Deploy the repository to Streamlit Community Cloud or another Streamlit-compatible host. Use `app.py` as the entry point and install dependencies from `requirements.txt`.

## Screenshots

Screenshots can be added after launching the app locally.

## Dataset Generation

The sample data is deterministic with a fixed random seed. It covers July 2023 through June 2026 with monthly salary, occasional freelance and investment income, common household expense categories, seasonal spending patterns, varied merchants and payment methods, and a few deliberately unusual expenses for anomaly demonstration.


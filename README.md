# FinSense Analytics

[![Live App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://finsense-analytics-4ryxn.streamlit.app)

**Live Demo:** https://finsense-analytics-4ryxn.streamlit.app

FinSense Analytics is a compact portfolio-grade Streamlit Data Science project for synthetic INR personal-finance analytics. It demonstrates ingestion, validation, ETL, exploratory analysis, statistical analysis, feature engineering, forecasting, anomaly detection, explainability, financial scoring, scenario analysis, and reporting without adding enterprise infrastructure.

## Key Features

- Deterministic synthetic four-year transaction dataset
- Main-area "Analyze Your Data" CSV import wizard processed in memory only
- Schema validation, cleaning, duplicate removal, rejected-row reasons, and cleaning report download
- Executive dashboard with income, expense, net cash flow, savings rate, budget utilization, health score, and anomaly count
- EDA tabs for trends, categories, merchants, calendar behavior, distributions, boxplots, percentiles, growth, and concentration
- Forecast and budget-risk page with seasonal baseline, Linear Regression, Random Forest, Gradient Boosting, MAE selection, residual-derived prediction range, and feature importance
- Isolation Forest anomaly detection with severity labels and factual explanations
- Transparent Financial Health Score from 0 to 100
- Scenario planner for expense reduction, one-time upcoming expenses, budget gap, and savings-goal progress
- Downloads for cleaned CSV, cleaning report, anomalies, monthly summary, model comparison, and a self-contained HTML report

## Analyze Your Own Data

Use **Upload My Transactions** at the top of the app to open the import wizard. The workflow is:

1. Upload file
2. Detect columns
3. Review mapping
4. Preview cleaned data
5. Analyze

Supported CSV layouts:

- Existing FinSense template
- Separate debit and credit columns
- One signed amount column
- Amount plus transaction-type column

Recognized column names include common variations of date, value date, description, narration, remarks, debit, withdrawal, credit, deposit, amount, type, category, merchant, payment method, currency, and transaction ID. Balance columns are ignored and never used as transaction amounts.

When merchant or category is missing, FinSense derives merchant names from transaction descriptions and applies transparent keyword rules. Examples include Swiggy/Zomato as Dining, rent as Housing, Airtel/Jio/gas/power as Utilities, Netflix/Spotify/Prime as Subscriptions, and unknown descriptions as Other. This is deterministic rule-based categorization, not AI categorization.

Privacy controls:

- Uploaded files are processed in memory only
- Files over 10 MB are rejected
- Unknown columns are ignored
- Transaction contents are not logged by the app
- Full account numbers should be removed before upload
- Malformed, empty, and unsupported CSVs receive clear validation messages

## Screenshots

Add screenshots after running locally:

- Executive Overview
- Forecast & Budget Risk
- Financial Health & Scenario Planner

## Data Science Lifecycle

1. Ingest sample or uploaded CSV.
2. Validate required columns and critical values.
3. Clean dates, amounts, transaction types, categories, merchants, and payment methods.
4. Transform transactions into monthly analytical and modeling features.
5. Explore trends, distributions, merchants, categories, and calendar behavior.
6. Evaluate forecasting models chronologically without target leakage.
7. Score anomaly patterns and financial health deterministically.
8. Produce recommendations, scenarios, and downloadable reports.

## Architecture

- `app.py`: Streamlit entry point and page orchestration
- `src/finsense/etl.py`: validation, cleaning, duplicate handling, cleaning report
- `src/finsense/analytics.py`: KPIs, summaries, filters, EDA aggregations
- `src/finsense/features.py`: monthly modeling features, lags, rolling averages, cyclical month features
- `src/finsense/forecasting.py`: baseline, Linear Regression, Random Forest, Gradient Boosting, metrics, ranges, importance
- `src/finsense/imports.py`: uploaded CSV parsing, column detection, mapping validation, deterministic enrichment
- `src/finsense/anomalies.py`: deterministic Isolation Forest pipeline and severity labels
- `src/finsense/scoring.py`: Financial Health Score components, weights, bands, suggestions
- `src/finsense/scenarios.py`: scenario planner calculations
- `src/finsense/insights.py`: deterministic insights and recommendations
- `src/finsense/reporting.py`: CSV and escaped HTML report generation
- `src/finsense/ui.py`: Streamlit/Plotly presentation
- `scripts/generate_sample_data.py`: deterministic synthetic data generator

## Dataset Schema

Required columns:

- `date`: parseable date
- `transaction_type`: `income` or `expense`
- `category`: transaction category
- `merchant`: merchant name; missing values become `Unknown`
- `amount`: positive numeric value in INR
- `payment_method`: payment method

Optional:

- `transaction_id`: exact and transaction-ID duplicates are removed where present

The built-in sample dataset is synthetic and non-sensitive. It includes salary growth, inflation, recurring merchants, seasonality, weekday/weekend behavior, and controlled unusual expenses.

## Forecasting Methodology

Expenses are aggregated monthly. Features used to predict month `t` are available strictly before month `t`, including previous-month income, previous-month savings, previous-month savings rate, previous-month transaction frequency, previous-month expense growth, expense lags, rolling expense averages, merchant/category diversity, weekend ratio, and cyclical month encodings. Models are evaluated chronologically and selected by validation MAE, not R².

The seasonal baseline remains the benchmark. An ML model is selected only when it beats the baseline validation MAE by at least 2%; otherwise the simpler baseline is preferred and the app reports that ML provided no material uplift. The app compares:

- Seasonal naive baseline
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

The prediction range is derived from historical validation residuals. It is not a statistically guaranteed confidence interval. Negative R² is shown honestly and means the model underperformed a mean-based reference on the validation period.

## Anomaly Methodology

Anomaly detection scores expense transactions with a deterministic scikit-learn pipeline using scaled amount/time features and one-hot encoded category, merchant, and payment-method features. Isolation Forest flags unusual transactions and assigns severity from anomaly-score quantiles. Anomaly means unusual, not fraudulent.

## Health-Score Methodology

The Financial Health Score is deterministic and educational. Components are scored from 0 to 100 and combined with these weights:

- Savings rate: 30%
- Expense stability: 20%
- Budget adherence: 20%
- Income-to-expense ratio: 20%
- Anomaly burden: 10%

Bands are Strong, Stable, Watch, and At Risk. The app explains reasons and suggestions but does not provide professional financial advice.

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python scripts/generate_sample_data.py
streamlit run app.py
```

## Test Commands

```bash
python -m ruff format .
python -m ruff format --check .
python -m ruff check .
python -m pytest
```

## Deployment

Deploy to Streamlit Community Cloud with `app.py` as the entry point and `requirements.txt` as the runtime dependency file. No database, Docker, external assets, paid APIs, or background services are required.

## Limitations

- Educational portfolio project, not enterprise production software
- No authentication or persistence
- Uploaded files are processed in memory only
- Forecasts are estimates and may underperform on filtered or short histories
- Prediction ranges are validation-residual ranges, not formal confidence intervals
- Anomaly flags are unusual patterns, not proof of fraud

## Future Improvements

- Optional richer sample scenarios
- More robust walk-forward backtesting visualizations
- Additional user-controlled anomaly sensitivity
- Exportable PDF report from the existing HTML report

## Responsible Use

FinSense Analytics is not professional financial advice. Use it to explore data quality, analytics, and modeling techniques on synthetic or personally owned transaction data.

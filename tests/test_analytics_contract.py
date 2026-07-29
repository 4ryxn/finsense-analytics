from __future__ import annotations

from io import BytesIO

import pandas as pd

from finsense.analytics import apply_filters, monthly_summary, monthly_summary_bytes
from tests.test_features_forecasting import make_monthly_data


def test_analytics_public_import_and_monthly_summary_export_contract() -> None:
    df = apply_filters(make_monthly_data(3))
    exported = monthly_summary_bytes(df)
    parsed = pd.read_csv(BytesIO(exported), parse_dates=["month"])
    expected = monthly_summary(df).reset_index(drop=True)
    expected.columns.name = None

    pd.testing.assert_frame_equal(parsed, expected)

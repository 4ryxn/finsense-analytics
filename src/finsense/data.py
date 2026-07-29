from __future__ import annotations

from pathlib import Path

import pandas as pd

from finsense.config import SAMPLE_DATA_PATH, TEMPLATE_PATH
from finsense.etl import CleaningReport, clean_transactions


def load_sample_transactions(path: Path = SAMPLE_DATA_PATH) -> tuple[pd.DataFrame, CleaningReport]:
    return load_transactions(path)


def load_transactions(path_or_buffer: str | Path | object) -> tuple[pd.DataFrame, CleaningReport]:
    raw = pd.read_csv(path_or_buffer)
    return clean_transactions(raw)


def template_bytes(path: Path = TEMPLATE_PATH) -> bytes:
    return path.read_bytes()


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

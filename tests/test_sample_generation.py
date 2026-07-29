from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_sample_generation_is_deterministic(tmp_path: Path) -> None:
    path = Path("data/sample_transactions.csv")
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    subprocess.run([sys.executable, "scripts/generate_sample_data.py"], check=True)
    after = hashlib.sha256(path.read_bytes()).hexdigest()
    df = pd.read_csv(path, parse_dates=["date"])
    assert before == after
    assert len(df) >= 1_600
    assert df["date"].min() == pd.Timestamp("2022-07-01")
    assert df["date"].max() >= pd.Timestamp("2026-06-25")
    assert {"Travel", "Dining", "Subscriptions", "Groceries"}.issubset(set(df["category"]))

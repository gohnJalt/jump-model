"""Run: python3 tests/test_data.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import align, fix_breaks

idx = pd.bdate_range("2020-07-20", "2020-07-31")
raw = pd.Series(1000.0, idx)
raw[raw.index >= "2020-07-27"] = 10.0  # ÷100 rebase in the data
fixed = fix_breaks(raw, [("2020-07-27", 100)])
assert np.allclose(fixed, 10.0), fixed
assert fix_breaks(fixed, [("2020-07-27", 100)]).equals(fixed)  # already-adjusted data is left alone

eq = pd.Series([1.0, 2.0, 3.0], pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-15"]))
fx = pd.Series([10.0, 11.0], pd.to_datetime(["2024-01-02", "2024-01-03"]))
m = align(eq, fx)
assert list(m["usdtry"].iloc[:2]) == [10.0, 11.0] and m["fx_lag_days"].iloc[1] == 0
assert np.isnan(m["usdtry"].iloc[2])  # 12-day-old FX must not be carried forward
print("ok")

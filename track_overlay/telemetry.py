"""Column discovery, numeric normalization, and unit conversion."""

import re
import numpy as np
import pandas as pd

from .settings import ATM_PSI, KPA_TO_PSI


def pick(columns, pattern):
    for col in columns:
        if re.search(pattern, col, re.I):
            return col
    return None


def series_num(df, col):
    return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)


def detect_scale(df, col, kind):
    if col is None:
        return 1.0
    mx = float(np.nanmax(series_num(df, col)))
    if not np.isfinite(mx):
        return 1.0

    if kind == "ign":
        return 10.0 if mx > 100.0 else 1.0

    if kind == "lambda":
        if mx > 50 and mx <= 5000:
            return 1000.0
        if mx > 5 and mx <= 50:
            return 10.0
        if mx > 5000 and mx <= 50000:
            return 10000.0
        return 1.0

    if mx >= 1000 and mx < 10000:
        return 10.0
    if mx >= 10000:
        return 100.0
    return 1.0


def kpa_to_psi(v): return v * KPA_TO_PSI


def psi_abs_to_psig(psi_abs): return psi_abs - ATM_PSI


def kelvin_to_f(k): return (k - 273.15) * 9.0 / 5.0 + 32.0

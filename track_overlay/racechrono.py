"""RaceChrono CSV loading."""

import re
import numpy as np
import pandas as pd

RACECHRONO_MPS_TO_MPH = 2.2369362920544


def _load_racechrono_csv(path):
    # RaceChrono has a small text header then a CSV header line starting with "timestamp,"
    header_idx = None
    with open(path, "r", errors="ignore") as f:
        for i, line in enumerate(f):
            if line.lower().startswith("timestamp,"):
                header_idx = i
                break
    if header_idx is None:
        raise SystemExit(f"Could not find RaceChrono header row in: {path}")

    df = pd.read_csv(path, skiprows=header_idx, engine="python")
    # First 2 rows after header are units/source rows (non-numeric "timestamp")
    df["timestamp_num"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.loc[np.isfinite(df["timestamp_num"])].copy()

    # Pick speed column (RaceChrono sometimes has duplicates like speed & speed.1)
    speed_col = "speed.1" if "speed.1" in df.columns else (
        "speed" if "speed" in df.columns else None)
    if speed_col is None:
        raise SystemExit("RaceChrono CSV missing speed column.")

    def col_or_none(name):
        return name if name in df.columns else None

    out = {
        "t_unix": pd.to_numeric(df["timestamp_num"], errors="coerce").to_numpy(dtype=np.float64),
        "speed_mph": pd.to_numeric(df[speed_col], errors="coerce").to_numpy(dtype=np.float64) * RACECHRONO_MPS_TO_MPH,
        "lap_number": pd.to_numeric(df.get("lap_number", np.nan), errors="coerce").to_numpy(dtype=np.float64),
        "lat_g": pd.to_numeric(df.get("lateral_acc", np.nan), errors="coerce").to_numpy(dtype=np.float64),
        "lon_g": pd.to_numeric(df.get("longitudinal_acc", np.nan), errors="coerce").to_numpy(dtype=np.float64),
        # GPS position (if present in export)
        "gps_lat": pd.to_numeric(df.get("latitude", np.nan), errors="coerce").to_numpy(dtype=np.float64),
        "gps_lon": pd.to_numeric(df.get("longitude", np.nan), errors="coerce").to_numpy(dtype=np.float64),
    }
    # Basic cleanup
    m = np.isfinite(out["t_unix"]) & np.isfinite(out["speed_mph"])
    for k in out:
        out[k] = out[k][m]
    return out

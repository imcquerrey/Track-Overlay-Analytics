"""Time parsing and display formatting."""

import re
import numpy as np
import pandas as pd


def _mmss_to_s(mmss):
    if mmss is None:
        return None
    mm, ss = mmss.split(":")
    return int(mm) * 60 + int(ss)


def _mmss_or_ss_to_s(v):
    """Parse 'm:ss', 'mm:ss', optionally with .mmm, or plain seconds string '20.4'."""
    if v is None:
        return None
    if isinstance(v, (int, float)) and not (isinstance(v, float) and (v != v)):
        return float(v)
    s = str(v).strip()
    if s == "" or s.lower() == "none":
        return None
    if re.match(r"^\d+(?:\.\d+)?$", s):
        return float(s)
    m = re.match(r"^(?P<m>\d+):(?P<s>\d{1,2})(?:\.(?P<ms>\d{1,3}))?$", s)
    if not m:
        raise ValueError(
            f"Bad time format: {v!r} (use seconds like 20.4 or m:ss(.mmm) like 0:20.400)")
    mm = int(m.group("m"))
    ss = int(m.group("s"))
    ms = m.group("ms")
    frac = 0.0
    if ms is not None:
        frac = int(ms.ljust(3, "0")) / 1000.0
    return mm * 60.0 + ss + frac


def _fmt_mmss_mmm(s: float) -> str:
    """Format seconds as m:ss.mmm."""
    if s is None:
        return "None"
    sign = "-" if s < 0 else ""
    s = abs(float(s))
    m = int(s // 60)
    sec = s - m * 60
    return f"{sign}{m}:{sec:06.3f}"


def time_to_seconds(t):
    if pd.isna(t):
        return np.nan
    s = str(t).strip()
    m = re.match(r"^(?P<h>\d{2}):(?P<m>\d{2}):(?P<sec>\d{2})\.(?P<ms>\d+)$", s)
    if not m:
        return np.nan
    ms = int(m.group("ms")[:3].ljust(3, "0"))
    return int(m.group("h")) * 3600 + int(m.group("m")) * 60 + int(m.group("sec")) + ms / 1000.0


def _fmt_laptime(sec):
    if not np.isfinite(sec):
        return "--:--.---"
    sec = float(max(0.0, sec))
    m = int(sec // 60.0)
    s = sec - 60.0 * m
    return f"{m:d}:{s:06.3f}"

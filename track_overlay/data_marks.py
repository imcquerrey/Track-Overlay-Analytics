"""Parsing for per-session trim and synchronization marks."""

from __future__ import annotations

from collections.abc import Iterable


def load_data_marks(path: str, allowed_names: Iterable[str] | None = None) -> dict[str, dict[str, object]]:
    allowed = None if allowed_names is None else set(allowed_names)
    results: dict[str, dict[str, object]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|")
            if len(parts) != 6:
                raise ValueError(
                    f"Invalid format at {path}:{line_number}: {line}")
            file_name, trim_start, trim_end, video_sync, log_sync, fine_tune_ms = parts
            if allowed is not None and file_name not in allowed:
                continue
            results[file_name] = {
                "VIDEO_TRIM_START_S": trim_start,
                "VIDEO_TRIM_END_S": trim_end,
                "VIDEO_SYNC_AT_MMSS": video_sync,
                "LOG_SYNC_AT_MMSS": log_sync,
                "SYNC_FINE_TUNE_MS": int(fine_tune_ms),
            }
    return results

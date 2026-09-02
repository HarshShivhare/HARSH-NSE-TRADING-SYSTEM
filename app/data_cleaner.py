from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["date", "open", "high", "low", "close", "volume"]
REGULAR_START_MIN = 9 * 60 + 15
REGULAR_END_MIN = 15 * 60 + 30


@dataclass(frozen=True)
class CleanResult:
    data: pd.DataFrame
    audit: pd.DataFrame


def _normalise_dates(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def classify_sessions(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Classify symbol-sessions without altering raw OHLCV values.

    Conservative policy:
    - Any session containing impossible OHLC, non-positive prices, negative volume,
      invalid numeric data or invalid timestamps is quarantined in full.
    - Any session containing bars outside 09:15-15:30 is treated as a special/non-regular
      session and excluded from the regular intraday strategy.
    - No OHLC values are repaired or fabricated.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    w = df.copy()
    w["date"] = _normalise_dates(w["date"])
    for c in ["open", "high", "low", "close", "volume"]:
        w[c] = pd.to_numeric(w[c], errors="coerce")

    # Invalid timestamps cannot be assigned to a session. They are reported separately.
    invalid_date_rows = int(w["date"].isna().sum())
    valid = w.dropna(subset=["date"]).copy()
    valid["session"] = valid["date"].dt.date

    numeric_bad = valid[["open", "high", "low", "close", "volume"]].isna().any(axis=1)
    nonpositive_price = ~(valid[["open", "high", "low", "close"]] > 0).all(axis=1)
    negative_volume = valid["volume"] < 0
    high_bad = valid["high"] < valid[["open", "close", "low"]].max(axis=1)
    low_bad = valid["low"] > valid[["open", "close", "high"]].min(axis=1)
    invalid_ohlcv = numeric_bad | nonpositive_price | negative_volume | high_bad | low_bad

    mins = valid["date"].dt.hour * 60 + valid["date"].dt.minute
    outside_regular = (mins < REGULAR_START_MIN) | (mins > REGULAR_END_MIN)

    rows: list[dict] = []
    if invalid_date_rows:
        rows.append({
            "symbol": symbol,
            "session": pd.NaT,
            "action": "QUARANTINE_ROWS",
            "reason": "invalid_timestamp",
            "rows_affected": invalid_date_rows,
        })

    for session, idx in valid.groupby("session").groups.items():
        idx = list(idx)
        bad_count = int(invalid_ohlcv.loc[idx].sum())
        outside_count = int(outside_regular.loc[idx].sum())
        if bad_count:
            rows.append({
                "symbol": symbol,
                "session": pd.Timestamp(session),
                "action": "EXCLUDE_SESSION",
                "reason": "invalid_ohlcv",
                "rows_affected": len(idx),
                "bad_rows": bad_count,
                "outside_regular_rows": outside_count,
            })
        elif outside_count:
            rows.append({
                "symbol": symbol,
                "session": pd.Timestamp(session),
                "action": "EXCLUDE_SESSION",
                "reason": "special_or_nonregular_session",
                "rows_affected": len(idx),
                "bad_rows": 0,
                "outside_regular_rows": outside_count,
            })

    return pd.DataFrame(rows)


def clean_market_data(df: pd.DataFrame, symbol: str) -> CleanResult:
    audit = classify_sessions(df, symbol)
    w = df.copy()
    w["date"] = pd.to_datetime(w["date"], errors="coerce")

    # Invalid timestamps are always removed; no value repair is performed.
    w = w.dropna(subset=["date"]).copy()
    w["session"] = w["date"].dt.date

    excluded_dates: set = set()
    if not audit.empty:
        sess = audit.loc[audit["action"] == "EXCLUDE_SESSION", "session"].dropna()
        excluded_dates = {pd.Timestamp(x).date() for x in sess}
    if excluded_dates:
        w = w.loc[~w["session"].isin(excluded_dates)].copy()

    # After session quarantine, retain regular-session bars only as a second safety rail.
    mins = w["date"].dt.hour * 60 + w["date"].dt.minute
    w = w.loc[(mins >= REGULAR_START_MIN) & (mins <= REGULAR_END_MIN)].copy()
    w = w.drop(columns=["session"]).sort_values("date").reset_index(drop=True)
    return CleanResult(data=w, audit=audit)


def summarize_cleaning(raw_df: pd.DataFrame, clean: CleanResult, symbol: str) -> dict:
    raw_dates = pd.to_datetime(raw_df["date"], errors="coerce") if "date" in raw_df else pd.Series(dtype="datetime64[ns]")
    raw_sessions = int(raw_dates.dropna().dt.date.nunique()) if len(raw_dates) else 0
    clean_dates = pd.to_datetime(clean.data["date"], errors="coerce") if not clean.data.empty else pd.Series(dtype="datetime64[ns]")
    clean_sessions = int(clean_dates.dropna().dt.date.nunique()) if len(clean_dates) else 0
    excluded = clean.audit.loc[clean.audit["action"] == "EXCLUDE_SESSION"] if not clean.audit.empty else pd.DataFrame()
    return {
        "symbol": symbol,
        "raw_rows": int(len(raw_df)),
        "clean_rows": int(len(clean.data)),
        "rows_removed": int(len(raw_df) - len(clean.data)),
        "raw_sessions": raw_sessions,
        "clean_sessions": clean_sessions,
        "sessions_removed": raw_sessions - clean_sessions,
        "invalid_sessions_removed": int((excluded.get("reason", pd.Series(dtype=str)) == "invalid_ohlcv").sum()) if not excluded.empty else 0,
        "special_sessions_removed": int((excluded.get("reason", pd.Series(dtype=str)) == "special_or_nonregular_session").sum()) if not excluded.empty else 0,
    }

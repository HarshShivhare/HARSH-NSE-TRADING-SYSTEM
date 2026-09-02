from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import hashlib
import re

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["date", "open", "high", "low", "close", "volume"]
FILENAME_RE = re.compile(
    r"^NSE_(?P<symbol>.+?)_(?P<interval>\w+)_(?P<start>\d{8})_(?P<end>\d{8})\.(?:parquet|csv)$",
    re.IGNORECASE,
)


@dataclass
class ValidationResult:
    files: pd.DataFrame
    issues: pd.DataFrame
    duplicate_symbols: pd.DataFrame


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _expected_interval_minutes(interval: str | None) -> int | None:
    if not interval:
        return None
    m = re.fullmatch(r"(\d+)minute", interval.lower())
    return int(m.group(1)) if m else None


def validate_files(files: Iterable[Path], expected_symbols: list[str] | None = None) -> ValidationResult:
    records: list[dict] = []
    issues: list[dict] = []
    expected_set = {s.strip().upper() for s in (expected_symbols or []) if s.strip()}
    seen_symbols: list[str] = []

    for path in files:
        meta = FILENAME_RE.match(path.name)
        symbol = meta.group("symbol").upper() if meta else path.stem.upper()
        interval = meta.group("interval") if meta else None
        seen_symbols.append(symbol)

        try:
            df = _load(path)
        except Exception as exc:
            issues.append({"file": path.name, "symbol": symbol, "severity": "ERROR", "issue": f"cannot_read: {exc}"})
            records.append({"file": path.name, "symbol": symbol, "status": "FAIL", "rows": 0})
            continue

        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            issues.append({"file": path.name, "symbol": symbol, "severity": "ERROR", "issue": f"missing_columns: {','.join(missing)}"})
            records.append({"file": path.name, "symbol": symbol, "status": "FAIL", "rows": len(df)})
            continue

        work = df.copy()
        work["date"] = pd.to_datetime(work["date"], errors="coerce")
        bad_dates = int(work["date"].isna().sum())
        if bad_dates:
            issues.append({"file": path.name, "symbol": symbol, "severity": "ERROR", "issue": f"invalid_dates: {bad_dates}"})
        work = work.dropna(subset=["date"]).sort_values("date")

        duplicate_ts = int(work["date"].duplicated().sum())
        if duplicate_ts:
            issues.append({"file": path.name, "symbol": symbol, "severity": "ERROR", "issue": f"duplicate_timestamps: {duplicate_ts}"})

        numeric = work[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")
        invalid_numeric = int(numeric.isna().any(axis=1).sum())
        if invalid_numeric:
            issues.append({"file": path.name, "symbol": symbol, "severity": "ERROR", "issue": f"invalid_ohlcv_rows: {invalid_numeric}"})

        price_positive = (numeric[["open", "high", "low", "close"]] > 0).all(axis=1)
        bad_price = int((~price_positive).sum())
        if bad_price:
            issues.append({"file": path.name, "symbol": symbol, "severity": "ERROR", "issue": f"nonpositive_price_rows: {bad_price}"})

        high_bad = numeric["high"] < numeric[["open", "close", "low"]].max(axis=1)
        low_bad = numeric["low"] > numeric[["open", "close", "high"]].min(axis=1)
        bad_ohlc = int((high_bad | low_bad).sum())
        if bad_ohlc:
            issues.append({"file": path.name, "symbol": symbol, "severity": "ERROR", "issue": f"ohlc_consistency_rows: {bad_ohlc}"})

        neg_vol = int((numeric["volume"] < 0).sum())
        if neg_vol:
            issues.append({"file": path.name, "symbol": symbol, "severity": "ERROR", "issue": f"negative_volume_rows: {neg_vol}"})

        if work.empty:
            first_ts = last_ts = pd.NaT
            sessions = 0
            median_bars = min_bars = max_bars = 0
            off_grid = 0
            out_of_hours = 0
        else:
            first_ts, last_ts = work["date"].iloc[0], work["date"].iloc[-1]
            sessions_series = work["date"].dt.date
            counts = work.groupby(sessions_series).size()
            sessions = int(counts.size)
            median_bars = float(counts.median()) if len(counts) else 0
            min_bars = int(counts.min()) if len(counts) else 0
            max_bars = int(counts.max()) if len(counts) else 0

            mins = work["date"].dt.hour * 60 + work["date"].dt.minute
            out_of_hours = int(((mins < 9 * 60 + 15) | (mins > 15 * 60 + 30)).sum())
            expected_min = _expected_interval_minutes(interval)
            off_grid = 0
            if expected_min:
                offset = mins - (9 * 60 + 15)
                off_grid = int(((offset % expected_min) != 0).sum())

        if out_of_hours:
            # Special NSE sessions such as Diwali Muhurat trading are legitimate source data.
            # V7 excludes any non-regular session from the regular intraday strategy instead
            # of treating those candles as corrupt.
            mins2 = work["date"].dt.hour * 60 + work["date"].dt.minute
            out_mask = (mins2 < 9 * 60 + 15) | (mins2 > 15 * 60 + 30)
            out_sessions = work.loc[out_mask, "date"].dt.date.nunique()
            issues.append({"file": path.name, "symbol": symbol, "severity": "INFO", "issue": f"special_or_nonregular_bars: {out_of_hours} across {out_sessions} sessions"})
        if off_grid:
            # Evening special sessions are anchored differently from 09:15, so their grid
            # should not create a warning. Only inspect regular-session bars for grid drift.
            regular = (mins >= 9 * 60 + 15) & (mins <= 15 * 60 + 30)
            expected_min = _expected_interval_minutes(interval)
            reg_off_grid = 0
            if expected_min:
                reg_offset = mins[regular] - (9 * 60 + 15)
                reg_off_grid = int(((reg_offset % expected_min) != 0).sum())
            if reg_off_grid:
                issues.append({"file": path.name, "symbol": symbol, "severity": "WARN", "issue": f"bars_off_interval_grid_regular_session: {reg_off_grid}"})

        if meta and not work.empty:
            filename_start = pd.Timestamp(meta.group("start"))
            filename_end = pd.Timestamp(meta.group("end"))
            actual_start = pd.Timestamp(first_ts).tz_localize(None).normalize() if pd.Timestamp(first_ts).tzinfo else pd.Timestamp(first_ts).normalize()
            actual_end = pd.Timestamp(last_ts).tz_localize(None).normalize() if pd.Timestamp(last_ts).tzinfo else pd.Timestamp(last_ts).normalize()
            # Requested boundaries may be weekends/holidays, so allow a small edge difference.
            if abs((actual_start - filename_start).days) > 10:
                issues.append({"file": path.name, "symbol": symbol, "severity": "WARN", "issue": f"filename_start_mismatch: requested={filename_start.date()} actual={actual_start.date()}"})
            if abs((filename_end - actual_end).days) > 10:
                issues.append({"file": path.name, "symbol": symbol, "severity": "WARN", "issue": f"filename_end_mismatch: requested={filename_end.date()} actual={actual_end.date()}"})

        error_count = sum(1 for x in issues if x["file"] == path.name and x["severity"] == "ERROR")
        warn_count = sum(1 for x in issues if x["file"] == path.name and x["severity"] == "WARN")
        info_count = sum(1 for x in issues if x["file"] == path.name and x["severity"] == "INFO")
        records.append({
            "file": path.name,
            "symbol": symbol,
            "interval": interval,
            "rows": int(len(df)),
            "sessions": sessions,
            "first_timestamp": first_ts,
            "last_timestamp": last_ts,
            "median_bars_per_session": median_bars,
            "min_bars_per_session": min_bars,
            "max_bars_per_session": max_bars,
            "duplicate_timestamps": duplicate_ts,
            "errors": error_count,
            "warnings": warn_count,
            "infos": info_count,
            "status": "FAIL" if error_count else ("WARN" if warn_count else "PASS"),
            "sha256": _sha256(path),
        })

    if expected_set:
        seen_set = set(seen_symbols)
        for s in sorted(expected_set - seen_set):
            issues.append({"file": "", "symbol": s, "severity": "ERROR", "issue": "expected_symbol_missing"})
        for s in sorted(seen_set - expected_set):
            issues.append({"file": "", "symbol": s, "severity": "WARN", "issue": "unexpected_symbol_present"})

    dup = pd.Series(seen_symbols).value_counts()
    dup = dup[dup > 1].rename_axis("symbol").reset_index(name="file_count") if len(dup) else pd.DataFrame(columns=["symbol", "file_count"])
    for _, row in dup.iterrows():
        issues.append({"file": "", "symbol": row["symbol"], "severity": "ERROR", "issue": f"duplicate_symbol_files: {int(row['file_count'])}"})

    return ValidationResult(pd.DataFrame(records), pd.DataFrame(issues), dup)


def write_validation_reports(result: ValidationResult, report_dir: Path) -> dict[str, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    files_path = report_dir / "data_files.csv"
    issues_path = report_dir / "data_issues.csv"
    dup_path = report_dir / "duplicate_symbols.csv"
    result.files.to_csv(files_path, index=False)
    result.issues.to_csv(issues_path, index=False)
    result.duplicate_symbols.to_csv(dup_path, index=False)
    return {"files": files_path, "issues": issues_path, "duplicate_symbols": dup_path}

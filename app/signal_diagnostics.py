from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import time

import numpy as np
import pandas as pd

from .data_cleaner import clean_market_data
from .strategy import StrategyConfig, prepare_features


@dataclass
class SignalDiagnosticResult:
    events: pd.DataFrame
    stage_summary: pd.DataFrame
    drop_one_summary: pd.DataFrame
    by_entry_time: pd.DataFrame
    by_vwap_extension: pd.DataFrame
    by_or_extension_atr: pd.DataFrame
    by_gap: pd.DataFrame
    by_rvol: pd.DataFrame


def _safe_mean(s: pd.Series) -> float | None:
    x = pd.to_numeric(s, errors="coerce").dropna()
    return float(x.mean()) if len(x) else None


def _summary(df: pd.DataFrame, label_col: str, label: str) -> dict:
    if df.empty:
        return {
            label_col: label, "events": 0, "positive_30m": None,
            "avg_fwd_15m_pct": None, "avg_fwd_30m_pct": None,
            "avg_fwd_60m_pct": None, "avg_fwd_120m_pct": None,
            "avg_eod_pct": None, "avg_mfe_60m_atr": None,
            "avg_mae_60m_atr": None, "failure_60m_rate": None,
            "avg_vwap_extension_pct": None, "avg_or_extension_atr": None,
        }
    return {
        label_col: label,
        "events": int(len(df)),
        "positive_30m": float((df["fwd_30m_pct"] > 0).mean()) if df["fwd_30m_pct"].notna().any() else None,
        "avg_fwd_15m_pct": _safe_mean(df["fwd_15m_pct"]),
        "avg_fwd_30m_pct": _safe_mean(df["fwd_30m_pct"]),
        "avg_fwd_60m_pct": _safe_mean(df["fwd_60m_pct"]),
        "avg_fwd_120m_pct": _safe_mean(df["fwd_120m_pct"]),
        "avg_eod_pct": _safe_mean(df["eod_pct"]),
        "avg_mfe_60m_atr": _safe_mean(df["mfe_60m_atr"]),
        "avg_mae_60m_atr": _safe_mean(df["mae_60m_atr"]),
        "failure_60m_rate": _safe_mean(df["failure_60m"].astype(float)),
        "avg_vwap_extension_pct": _safe_mean(df["entry_vs_vwap_pct"]),
        "avg_or_extension_atr": _safe_mean(df["entry_vs_or_high_atr"]),
    }


def _add_intraday_context(f: pd.DataFrame) -> pd.DataFrame:
    out = f.copy()
    typical = (out["high"] + out["low"] + out["close"]) / 3.0
    pv = typical * out["volume"]
    out["vwap"] = pv.groupby(out["session"]).cumsum() / out["volume"].groupby(out["session"]).cumsum().replace(0, np.nan)

    out["or_break"] = out["close"] > out["or_high"]
    out["prev_high_break"] = out["close"] > out["prev_high"]

    def first_break_time(day: pd.DataFrame, col: str) -> pd.Series:
        hits = day.loc[day[col], "date"]
        first = hits.iloc[0] if len(hits) else pd.NaT
        return pd.Series(first, index=day.index)

    out["first_or_break_time"] = out.groupby("session", group_keys=False).apply(
        lambda d: first_break_time(d, "or_break"), include_groups=False
    )
    out["first_prev_high_break_time"] = out.groupby("session", group_keys=False).apply(
        lambda d: first_break_time(d, "prev_high_break"), include_groups=False
    )
    return out


def _stage_masks(f: pd.DataFrame, cfg: StrategyConfig) -> dict[str, pd.Series]:
    complete = f[["prev_close", "prev_high", "daily_sma", "or_high", "rvol", "atr"]].notna().all(axis=1)
    time_ok = (f["time"] >= cfg.earliest_entry) & (f["time"] <= cfg.latest_entry)
    trend = f["prev_close"] > f["daily_sma"]
    gap = f["gap_pct"] >= cfg.gap_min_pct
    prev_high = f["close"] > f["prev_high"]
    opening_range = f["close"] > f["or_high"]
    rvol = f["rvol"] >= cfg.rvol_min

    base = complete & time_ok
    return {
        "01_complete_time": base,
        "02_plus_trend": base & trend,
        "03_plus_gap": base & trend & gap,
        "04_plus_prev_high": base & trend & gap & prev_high,
        "05_plus_opening_range": base & trend & gap & prev_high & opening_range,
        "06_full_plus_rvol": base & trend & gap & prev_high & opening_range & rvol,
        "drop_trend": base & gap & prev_high & opening_range & rvol,
        "drop_gap": base & trend & prev_high & opening_range & rvol,
        "drop_prev_high": base & trend & gap & opening_range & rvol,
        "drop_opening_range": base & trend & gap & prev_high & rvol,
        "drop_rvol": base & trend & gap & prev_high & opening_range,
    }


def _event_from_signal(day: pd.DataFrame, sig_idx: int, symbol: str, stage: str) -> dict | None:
    # Mirror the backtester: signal on close, enter on next bar open.
    if sig_idx + 1 >= len(day):
        return None
    signal = day.loc[sig_idx]
    entry_idx = sig_idx + 1
    entry_row = day.loc[entry_idx]
    entry = float(entry_row["open"])
    atr = float(signal["atr"]) if pd.notna(signal["atr"]) else np.nan
    if not np.isfinite(entry) or entry <= 0:
        return None

    event = {
        "stage": stage,
        "symbol": symbol,
        "session": pd.Timestamp(signal["session"]),
        "signal_time": signal["date"],
        "entry_time": entry_row["date"],
        "entry": entry,
        "signal_close": float(signal["close"]),
        "day_open": float(day.iloc[0]["open"]),
        "prev_high": float(signal["prev_high"]),
        "or_high": float(signal["or_high"]),
        "vwap": float(signal["vwap"]) if pd.notna(signal["vwap"]) else np.nan,
        "atr": atr,
        "gap_pct": float(signal["gap_pct"]),
        "rvol": float(signal["rvol"]),
    }

    event["entry_vs_vwap_pct"] = (entry / event["vwap"] - 1.0) * 100.0 if event["vwap"] else np.nan
    event["entry_vs_open_pct"] = (entry / event["day_open"] - 1.0) * 100.0 if event["day_open"] else np.nan
    event["entry_vs_or_high_atr"] = (entry - event["or_high"]) / atr if np.isfinite(atr) and atr > 0 else np.nan
    event["entry_vs_prev_high_atr"] = (entry - event["prev_high"]) / atr if np.isfinite(atr) and atr > 0 else np.nan

    for source, name in [(signal.get("first_or_break_time"), "minutes_since_or_break"),
                         (signal.get("first_prev_high_break_time"), "minutes_since_prev_high_break")]:
        if pd.isna(source):
            event[name] = np.nan
        else:
            event[name] = (pd.Timestamp(signal["date"]) - pd.Timestamp(source)).total_seconds() / 60.0

    # 5-minute data: use the bar at/after the requested horizon, capped to the session.
    horizons = {15: "fwd_15m_pct", 30: "fwd_30m_pct", 60: "fwd_60m_pct", 120: "fwd_120m_pct"}
    entry_ts = pd.Timestamp(entry_row["date"])
    for minutes, col in horizons.items():
        target_ts = entry_ts + pd.Timedelta(minutes=minutes)
        later = day.loc[(day.index >= entry_idx) & (pd.to_datetime(day["date"]) >= target_ts)]
        if later.empty:
            event[col] = np.nan
        else:
            px = float(later.iloc[0]["close"])
            event[col] = (px / entry - 1.0) * 100.0

    eod = float(day.iloc[-1]["close"])
    event["eod_pct"] = (eod / entry - 1.0) * 100.0

    window_end = entry_ts + pd.Timedelta(minutes=60)
    window = day.loc[(day.index >= entry_idx) & (pd.to_datetime(day["date"]) <= window_end)]
    if not window.empty and np.isfinite(atr) and atr > 0:
        event["mfe_60m_atr"] = (float(window["high"].max()) - entry) / atr
        event["mae_60m_atr"] = (float(window["low"].min()) - entry) / atr
    else:
        event["mfe_60m_atr"] = np.nan
        event["mae_60m_atr"] = np.nan

    # Failure = closes back below either breakout level within 60m after entry.
    if window.empty:
        event["failure_60m"] = False
    else:
        fail_level = max(event["or_high"], event["prev_high"])
        event["failure_60m"] = bool((window["close"] < fail_level).any())

    return event


def _first_events_for_mask(f: pd.DataFrame, mask: pd.Series, symbol: str, stage: str,
                           start_date, end_date) -> list[dict]:
    work = f.loc[mask].copy()
    if start_date is not None:
        work = work.loc[pd.to_datetime(work["session"]).dt.date >= start_date]
    if end_date is not None:
        work = work.loc[pd.to_datetime(work["session"]).dt.date <= end_date]
    if work.empty:
        return []

    # One independent event per symbol/session/stage: first qualifying signal only.
    first_idx = work.groupby("session", sort=True).head(1).index
    events: list[dict] = []
    for idx in first_idx:
        session = f.loc[idx, "session"]
        day = f.loc[f["session"] == session].reset_index(drop=False)
        positions = day.index[day["index"] == idx]
        if len(positions) == 0:
            continue
        ev = _event_from_signal(day.drop(columns=["index"]), int(positions[0]), symbol, stage)
        if ev is not None:
            events.append(ev)
    return events


def _bucket_summary(events: pd.DataFrame, col: str, bins, labels, out_col: str) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    full = events.loc[events["stage"] == "06_full_plus_rvol"].copy()
    if full.empty:
        return pd.DataFrame()
    full[out_col] = pd.cut(pd.to_numeric(full[col], errors="coerce"), bins=bins, labels=labels, include_lowest=True)
    rows = [_summary(g, out_col, str(bucket)) for bucket, g in full.groupby(out_col, observed=True)]
    return pd.DataFrame(rows)


def run_signal_diagnostics(
    files: Iterable[Path],
    cfg: StrategyConfig,
    session_start: str | None = None,
    session_end: str | None = None,
) -> SignalDiagnosticResult:
    files = list(files)
    start_date = pd.Timestamp(session_start).date() if session_start else None
    end_date = pd.Timestamp(session_end).date() if session_end else None
    all_events: list[dict] = []
    started = time.monotonic()

    for i, path in enumerate(files, start=1):
        symbol = path.name.split("_")[1] if "_" in path.name else path.stem
        print(f"[{i}/{len(files)}] V9 signal diagnostics: {symbol} | elapsed {(time.monotonic()-started)/60:.1f}m", flush=True)
        raw = pd.read_parquet(path)
        cleaned = clean_market_data(raw, symbol).data
        if cleaned.empty:
            continue
        f = _add_intraday_context(prepare_features(cleaned, cfg))
        masks = _stage_masks(f, cfg)
        for stage, mask in masks.items():
            all_events.extend(_first_events_for_mask(f, mask, symbol, stage, start_date, end_date))

    events = pd.DataFrame(all_events)
    if events.empty:
        empty = pd.DataFrame()
        return SignalDiagnosticResult(events, empty, empty, empty, empty, empty, empty, empty)

    stage_names = [
        "01_complete_time", "02_plus_trend", "03_plus_gap", "04_plus_prev_high",
        "05_plus_opening_range", "06_full_plus_rvol",
    ]
    stage_summary = pd.DataFrame([
        _summary(events.loc[events["stage"] == s], "stage", s) for s in stage_names
    ])
    drop_names = ["drop_trend", "drop_gap", "drop_prev_high", "drop_opening_range", "drop_rvol"]
    drop_one_summary = pd.DataFrame([
        _summary(events.loc[events["stage"] == s], "variant", s) for s in drop_names
    ])

    full = events.loc[events["stage"] == "06_full_plus_rvol"].copy()
    if full.empty:
        by_entry_time = by_gap = by_rvol = pd.DataFrame()
    else:
        full["entry_bucket"] = pd.to_datetime(full["entry_time"]).dt.strftime("%H:%M")
        rows = [_summary(g, "entry_time", str(k)) for k, g in full.groupby("entry_bucket")]
        by_entry_time = pd.DataFrame(rows).sort_values("entry_time") if rows else pd.DataFrame()
        by_gap = _bucket_summary(events, "gap_pct", [-np.inf, 1.5, 2, 3, 5, np.inf], ["<=1.5", "1.5-2", "2-3", "3-5", ">5"], "gap_bucket")
        by_rvol = _bucket_summary(events, "rvol", [-np.inf, 2, 3, 5, np.inf], ["<=2", "2-3", "3-5", ">5"], "rvol_bucket")

    by_vwap_extension = _bucket_summary(
        events, "entry_vs_vwap_pct", [-np.inf, 0, 0.25, 0.5, 1.0, 2.0, np.inf],
        ["<=0%", "0-.25%", ".25-.5%", ".5-1%", "1-2%", ">2%"], "vwap_extension_bucket"
    )
    by_or_extension_atr = _bucket_summary(
        events, "entry_vs_or_high_atr", [-np.inf, 0, 0.25, 0.5, 1.0, np.inf],
        ["<=0 ATR", "0-.25 ATR", ".25-.5 ATR", ".5-1 ATR", ">1 ATR"], "or_extension_bucket"
    )

    return SignalDiagnosticResult(
        events=events,
        stage_summary=stage_summary,
        drop_one_summary=drop_one_summary,
        by_entry_time=by_entry_time,
        by_vwap_extension=by_vwap_extension,
        by_or_extension_atr=by_or_extension_atr,
        by_gap=by_gap,
        by_rvol=by_rvol,
    )


def write_signal_diagnostic_reports(result: SignalDiagnosticResult, report_dir: Path) -> dict[str, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    tables = {
        "stage_summary": result.stage_summary,
        "drop_one_summary": result.drop_one_summary,
        "by_entry_time": result.by_entry_time,
        "by_vwap_extension": result.by_vwap_extension,
        "by_or_extension_atr": result.by_or_extension_atr,
        "by_gap": result.by_gap,
        "by_rvol": result.by_rvol,
    }
    for name, table in tables.items():
        path = report_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        outputs[name] = path
    events_path = report_dir / "signal_events.parquet"
    result.events.to_parquet(events_path, index=False)
    outputs["events"] = events_path
    return outputs

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .strategy import StrategyConfig, prepare_features


@dataclass(frozen=True)
class DiagnosticResult:
    aggregate: pd.DataFrame
    per_symbol: pd.DataFrame
    threshold_sensitivity: pd.DataFrame
    data_quality: pd.DataFrame


def _count(mask: pd.Series, f: pd.DataFrame) -> tuple[int, int]:
    """Return matching bars and unique trading sessions."""
    m = mask.fillna(False)
    return int(m.sum()), int(f.loc[m, "session"].nunique())


def _masks(f: pd.DataFrame, cfg: StrategyConfig) -> dict[str, pd.Series]:
    time_ok = (f["time"] >= cfg.earliest_entry) & (f["time"] <= cfg.latest_entry)
    trend_ok = f["prev_close"] > f["daily_sma"]
    gap_ok = f["gap_pct"] >= cfg.gap_min_pct
    prev_high_ok = f["close"] > f["prev_high"]
    or_ok = f["close"] > f["or_high"]
    rvol_ok = f["rvol"] >= cfg.rvol_min
    complete = f[["prev_close", "prev_high", "daily_sma", "or_high", "rvol", "atr"]].notna().all(axis=1)
    return {
        "complete_features": complete,
        "time_window": time_ok,
        "trend_prev_close_gt_sma": trend_ok,
        "gap": gap_ok,
        "prev_day_high_break": prev_high_ok,
        "opening_range_break": or_ok,
        "rvol": rvol_ok,
    }


def _symbol_diagnostics(df: pd.DataFrame, symbol: str, cfg: StrategyConfig) -> tuple[pd.DataFrame, dict, list[dict]]:
    f = prepare_features(df, cfg)
    masks = _masks(f, cfg)

    # Cumulative funnel. Feature completeness comes first so missing warm-up data is visible.
    order = [
        "complete_features",
        "time_window",
        "trend_prev_close_gt_sma",
        "gap",
        "prev_day_high_break",
        "opening_range_break",
        "rvol",
    ]

    cumulative = pd.Series(True, index=f.index)
    rows: list[dict] = []
    for step in order:
        independent_bars, independent_days = _count(masks[step], f)
        cumulative &= masks[step].fillna(False)
        cumulative_bars, cumulative_days = _count(cumulative, f)
        rows.append({
            "symbol": symbol,
            "step": step,
            "independent_bars": independent_bars,
            "independent_sessions": independent_days,
            "cumulative_bars": cumulative_bars,
            "cumulative_sessions": cumulative_days,
        })

    # Data quality/warm-up visibility.
    sessions = int(f["session"].nunique())
    sma_sessions = int(f.loc[f["daily_sma"].notna(), "session"].nunique())
    rvol_sessions = int(f.loc[f["rvol"].notna(), "session"].nunique())
    complete_sessions = int(f.loc[masks["complete_features"], "session"].nunique())
    q = {
        "symbol": symbol,
        "rows": len(f),
        "sessions": sessions,
        "first_date": f["date"].min(),
        "last_date": f["date"].max(),
        "sma_ready_sessions": sma_sessions,
        "rvol_ready_sessions": rvol_sessions,
        "complete_feature_sessions": complete_sessions,
        "sma_ready_pct": (100.0 * sma_sessions / sessions) if sessions else 0.0,
        "rvol_ready_pct": (100.0 * rvol_sessions / sessions) if sessions else 0.0,
        "complete_ready_pct": (100.0 * complete_sessions / sessions) if sessions else 0.0,
    }

    # Threshold sensitivity WITHOUT changing the backtest. Counts potential bars after
    # completeness + time + trend + both breakout conditions, then asks how gap/RVOL prune them.
    base = (
        masks["complete_features"]
        & masks["time_window"]
        & masks["trend_prev_close_gt_sma"]
        & masks["prev_day_high_break"]
        & masks["opening_range_break"]
    ).fillna(False)

    sensitivity: list[dict] = []
    gap_levels = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
    rvol_levels = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    for gap in gap_levels:
        for rvol in rvol_levels:
            mask = base & (f["gap_pct"] >= gap) & (f["rvol"] >= rvol)
            bars, days = _count(mask, f)
            sensitivity.append({
                "symbol": symbol,
                "gap_min_pct": gap,
                "rvol_min": rvol,
                "candidate_bars": bars,
                "candidate_sessions": days,
            })

    return pd.DataFrame(rows), q, sensitivity


def diagnose_files(files: Iterable[Path], cfg: StrategyConfig) -> DiagnosticResult:
    funnel_frames: list[pd.DataFrame] = []
    quality_rows: list[dict] = []
    sensitivity_rows: list[dict] = []

    for path in files:
        symbol = path.name.split("_")[1] if "_" in path.name else path.stem
        print(f"Diagnosing {symbol}: {path.name}")
        df = pd.read_parquet(path)
        funnel, quality, sensitivity = _symbol_diagnostics(df, symbol, cfg)
        funnel_frames.append(funnel)
        quality_rows.append(quality)
        sensitivity_rows.extend(sensitivity)

    per_symbol = pd.concat(funnel_frames, ignore_index=True) if funnel_frames else pd.DataFrame()
    quality = pd.DataFrame(quality_rows)
    sensitivity = pd.DataFrame(sensitivity_rows)

    if per_symbol.empty:
        aggregate = pd.DataFrame()
    else:
        # Sum bar/session counts across symbols. "Sessions" here means symbol-sessions,
        # intentionally, because each stock-day is a separate opportunity.
        aggregate = (
            per_symbol.groupby("step", sort=False)[
                ["independent_bars", "independent_sessions", "cumulative_bars", "cumulative_sessions"]
            ]
            .sum()
            .reset_index()
        )

    if not sensitivity.empty:
        sensitivity = (
            sensitivity.groupby(["gap_min_pct", "rvol_min"], as_index=False)
            [["candidate_bars", "candidate_sessions"]]
            .sum()
            .sort_values(["gap_min_pct", "rvol_min"])
            .reset_index(drop=True)
        )

    return DiagnosticResult(
        aggregate=aggregate,
        per_symbol=per_symbol,
        threshold_sensitivity=sensitivity,
        data_quality=quality,
    )


def write_diagnostic_reports(result: DiagnosticResult, report_dir: Path) -> dict[str, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "aggregate_funnel": report_dir / "aggregate_funnel.csv",
        "per_symbol_funnel": report_dir / "per_symbol_funnel.csv",
        "threshold_sensitivity": report_dir / "threshold_sensitivity.csv",
        "data_quality": report_dir / "data_quality.csv",
    }
    result.aggregate.to_csv(paths["aggregate_funnel"], index=False)
    result.per_symbol.to_csv(paths["per_symbol_funnel"], index=False)
    result.threshold_sensitivity.to_csv(paths["threshold_sensitivity"], index=False)
    result.data_quality.to_csv(paths["data_quality"], index=False)
    return paths

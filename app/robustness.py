from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .backtest import BacktestConfig, backtest_files, summarize_trades
from .strategy import StrategyConfig


def _bootstrap_mean_ci(values: pd.Series, samples: int = 1000, seed: int = 42) -> tuple[float | None, float | None]:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(x) < 2 or samples <= 0:
        return None, None
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=float)
    for i in range(samples):
        means[i] = rng.choice(x, size=len(x), replace=True).mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _split_row(
    trades: pd.DataFrame,
    initial_capital: float,
    split: str,
    gap_min: float,
    rvol_min: float,
    target_r: float,
    opening_range: int,
    bootstrap_samples: int,
) -> dict:
    s = summarize_trades(trades, initial_capital)
    ci_lo, ci_hi = _bootstrap_mean_ci(
        trades["r_multiple"] if not trades.empty and "r_multiple" in trades else pd.Series(dtype=float),
        bootstrap_samples,
    )
    return {
        "split": split,
        "gap_min": gap_min,
        "rvol_min": rvol_min,
        "target_r": target_r,
        "opening_range": opening_range,
        "trades": s["trades"],
        "wins": s["wins"],
        "win_rate": s["win_rate"],
        "profit_factor": s["profit_factor"],
        "expectancy_r": s["expectancy_r"],
        "expectancy_ci_low": ci_lo,
        "expectancy_ci_high": ci_hi,
        "net_pnl": s["net_pnl"],
        "return_pct": s["return_pct"],
        "max_drawdown_pct": s["max_drawdown_pct"],
        "avg_mfe_r": s["avg_mfe_r"],
        "avg_mae_r": s["avg_mae_r"],
    }


def run_development_grid(
    files: Iterable[Path],
    base_strategy: StrategyConfig,
    bt_cfg: BacktestConfig,
    gaps: list[float],
    rvols: list[float],
    targets: list[float],
    opening_ranges: list[int],
    dev_start: str,
    dev_end: str,
    validation_start: str,
    validation_end: str,
    min_trades_per_split: int = 10,
    bootstrap_samples: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []

    for opening_range in opening_ranges:
        for gap in gaps:
            for rvol in rvols:
                for target in targets:
                    cfg = replace(
                        base_strategy,
                        opening_range_minutes=opening_range,
                        gap_min_pct=gap,
                        rvol_min=rvol,
                        target_r=target,
                    )
                    for split, start, end in [
                        ("DEV", dev_start, dev_end),
                        ("VALIDATION", validation_start, validation_end),
                    ]:
                        trades = backtest_files(
                            files,
                            cfg,
                            bt_cfg,
                            session_start=start,
                            session_end=end,
                            quiet=True,
                        )
                        rows.append(
                            _split_row(
                                trades,
                                bt_cfg.initial_capital,
                                split,
                                gap,
                                rvol,
                                target,
                                opening_range,
                                bootstrap_samples,
                            )
                        )

    results = pd.DataFrame(rows)
    if results.empty:
        return results, pd.DataFrame()

    key = ["gap_min", "rvol_min", "target_r", "opening_range"]
    wide = results.pivot(index=key, columns="split")
    wide.columns = [f"{metric.lower()}_{split.lower()}" for metric, split in wide.columns]
    ranking = wide.reset_index()

    for col in [
        "trades_dev", "trades_validation", "expectancy_r_dev", "expectancy_r_validation",
        "profit_factor_dev", "profit_factor_validation", "expectancy_ci_low_dev",
        "expectancy_ci_low_validation",
    ]:
        if col not in ranking:
            ranking[col] = np.nan

    ranking["passes_min_trades"] = (
        (ranking["trades_dev"].fillna(0) >= min_trades_per_split)
        & (ranking["trades_validation"].fillna(0) >= min_trades_per_split)
    )
    ranking["positive_both"] = (
        (ranking["expectancy_r_dev"].fillna(-np.inf) > 0)
        & (ranking["expectancy_r_validation"].fillna(-np.inf) > 0)
        & (ranking["profit_factor_dev"].fillna(0) > 1)
        & (ranking["profit_factor_validation"].fillna(0) > 1)
    )
    ranking["ci_positive_both"] = (
        (ranking["expectancy_ci_low_dev"].fillna(-np.inf) > 0)
        & (ranking["expectancy_ci_low_validation"].fillna(-np.inf) > 0)
    )
    ranking["robust_gate"] = ranking["passes_min_trades"] & ranking["positive_both"]
    ranking["worst_split_expectancy_r"] = ranking[["expectancy_r_dev", "expectancy_r_validation"]].min(axis=1)
    ranking["total_trades_dev_validation"] = ranking["trades_dev"].fillna(0) + ranking["trades_validation"].fillna(0)
    ranking = ranking.sort_values(
        ["robust_gate", "ci_positive_both", "worst_split_expectancy_r", "total_trades_dev_validation"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    return results, ranking


def run_final_oos(
    files: Iterable[Path],
    strategy_cfg: StrategyConfig,
    bt_cfg: BacktestConfig,
    oos_start: str,
    oos_end: str,
    bootstrap_samples: int = 5000,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    trades = backtest_files(
        files,
        strategy_cfg,
        bt_cfg,
        session_start=oos_start,
        session_end=oos_end,
        quiet=True,
    )
    summary = summarize_trades(trades, bt_cfg.initial_capital)
    ci_lo, ci_hi = _bootstrap_mean_ci(
        trades["r_multiple"] if not trades.empty else pd.Series(dtype=float),
        bootstrap_samples,
    )
    summary = dict(summary)
    summary["expectancy_ci_low"] = ci_lo
    summary["expectancy_ci_high"] = ci_hi

    if trades.empty:
        return summary, trades, pd.DataFrame()

    t = trades.copy()
    dt = pd.to_datetime(t["exit_time"])
    if getattr(dt.dt, "tz", None) is not None:
        dt = dt.dt.tz_localize(None)
    t["quarter"] = dt.dt.to_period("Q").astype(str)

    rows = []
    for quarter, q in t.groupby("quarter", sort=True):
        s = summarize_trades(q, bt_cfg.initial_capital)
        qlo, qhi = _bootstrap_mean_ci(q["r_multiple"], min(bootstrap_samples, 2000))
        rows.append({
            "quarter": quarter,
            "trades": s["trades"],
            "win_rate": s["win_rate"],
            "profit_factor": s["profit_factor"],
            "expectancy_r": s["expectancy_r"],
            "expectancy_ci_low": qlo,
            "expectancy_ci_high": qhi,
            "net_pnl": s["net_pnl"],
            "max_drawdown_pct": s["max_drawdown_pct"],
        })
    quarterly = pd.DataFrame(rows)
    return summary, trades, quarterly


def write_robustness_reports(
    report_dir: Path,
    results: pd.DataFrame | None = None,
    ranking: pd.DataFrame | None = None,
    final_trades: pd.DataFrame | None = None,
    final_quarterly: pd.DataFrame | None = None,
    final_summary: dict | None = None,
) -> dict[str, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    if results is not None:
        p = report_dir / "grid_results.csv"; results.to_csv(p, index=False); paths["grid_results"] = p
    if ranking is not None:
        p = report_dir / "ranking.csv"; ranking.to_csv(p, index=False); paths["ranking"] = p
    if final_trades is not None:
        p = report_dir / "final_oos_trades.parquet"; final_trades.to_parquet(p, index=False); paths["final_oos_trades"] = p
    if final_quarterly is not None:
        p = report_dir / "final_oos_by_quarter.csv"; final_quarterly.to_csv(p, index=False); paths["final_oos_by_quarter"] = p
    if final_summary is not None:
        p = report_dir / "final_oos_summary.csv"; pd.DataFrame([final_summary]).to_csv(p, index=False); paths["final_oos_summary"] = p
    return paths

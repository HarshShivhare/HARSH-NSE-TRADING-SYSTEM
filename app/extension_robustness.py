from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Iterable
import itertools
import time

import numpy as np
import pandas as pd

from .backtest import BacktestConfig, backtest_files, summarize_trades
from .robustness import _Spinner, _bootstrap_mean_ci
from .strategy import StrategyConfig


def _parse_optional_float(value: str) -> float | None:
    text = str(value).strip().lower()
    if text in {"none", "off", "na", "nan", ""}:
        return None
    return float(text)


def parse_optional_float_csv(text: str) -> list[float | None]:
    return [_parse_optional_float(v) for v in text.split(",") if v.strip()]


def parse_bool_csv(text: str) -> list[bool]:
    out: list[bool] = []
    for raw in text.split(","):
        v = raw.strip().lower()
        if not v:
            continue
        if v in {"on", "true", "1", "yes"}:
            out.append(True)
        elif v in {"off", "false", "0", "no"}:
            out.append(False)
        else:
            raise ValueError(f"Unsupported trend mode: {raw!r}. Use on/off.")
    return out


def _split_row(
    trades: pd.DataFrame,
    initial_capital: float,
    split: str,
    require_trend: bool,
    max_vwap_extension_pct: float | None,
    max_or_extension_atr: float | None,
    rvol_min: float,
    bootstrap_samples: int,
) -> dict:
    s = summarize_trades(trades, initial_capital)
    ci_lo, ci_hi = _bootstrap_mean_ci(
        trades["r_multiple"] if not trades.empty and "r_multiple" in trades else pd.Series(dtype=float),
        bootstrap_samples,
    )
    return {
        "split": split,
        "trend": "ON" if require_trend else "OFF",
        "max_vwap_extension_pct": max_vwap_extension_pct,
        "max_or_extension_atr": max_or_extension_atr,
        "rvol_min": rvol_min,
        "trades": s["trades"],
        "wins": s["wins"],
        "win_rate": s["win_rate"],
        "profit_factor": s["profit_factor"],
        "expectancy_r": s["expectancy_r"],
        "expectancy_ci_low": ci_lo,
        "expectancy_ci_high": ci_hi,
        "gross_pnl": s["gross_pnl"],
        "charges": s["charges"],
        "net_pnl": s["net_pnl"],
        "return_pct": s["return_pct"],
        "max_drawdown_pct": s["max_drawdown_pct"],
        "avg_mfe_r": s["avg_mfe_r"],
        "avg_mae_r": s["avg_mae_r"],
    }


def run_extension_development_grid(
    files: Iterable[Path],
    base_strategy: StrategyConfig,
    bt_cfg: BacktestConfig,
    trend_modes: list[bool],
    vwap_maxes: list[float | None],
    or_maxes: list[float | None],
    rvols: list[float],
    dev_start: str,
    dev_end: str,
    validation_start: str,
    validation_end: str,
    min_trades_per_split: int = 20,
    bootstrap_samples: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    combos = list(itertools.product(trend_modes, vwap_maxes, or_maxes, rvols))
    total_jobs = len(combos) * 2
    job = 0
    started_all = time.monotonic()

    print(f"V10 extension robustness: {len(combos)} combinations / {total_jobs} split backtests", flush=True)
    print(
        f"Fixed structure: gap>={base_strategy.gap_min_pct:g}% | OR={base_strategy.opening_range_minutes}m | "
        f"target={base_strategy.target_r:g}R | stop={base_strategy.stop_mode}/{base_strategy.atr_multiple:g} ATR",
        flush=True,
    )

    for combo_idx, (trend, vwap_max, or_max, rvol) in enumerate(combos, start=1):
        cfg = replace(
            base_strategy,
            require_trend=trend,
            max_vwap_extension_pct=vwap_max,
            max_or_extension_atr=or_max,
            rvol_min=rvol,
        )
        for split, start, end in [
            ("DEV", dev_start, dev_end),
            ("VALIDATION", validation_start, validation_end),
        ]:
            job += 1
            vwap_label = "none" if vwap_max is None else f"{vwap_max:g}%"
            or_label = "none" if or_max is None else f"{or_max:g}ATR"
            message = (
                f"[{job}/{total_jobs}] combo {combo_idx}/{len(combos)} {split} "
                f"trend={'ON' if trend else 'OFF'} vwap<={vwap_label} ORext<={or_label} rvol>={rvol:g}"
            )
            with _Spinner(message):
                trades = backtest_files(files, cfg, bt_cfg, session_start=start, session_end=end, quiet=True)
            elapsed = time.monotonic() - started_all
            print(f"✓ {message} -> {len(trades)} trades | elapsed {elapsed/60:.1f}m", flush=True)
            rows.append(
                _split_row(
                    trades,
                    bt_cfg.initial_capital,
                    split,
                    trend,
                    vwap_max,
                    or_max,
                    rvol,
                    bootstrap_samples,
                )
            )

    results = pd.DataFrame(rows)
    if results.empty:
        return results, pd.DataFrame()

    # Pivot needs stable string keys because pandas drops NaN keys. Keep numeric
    # threshold columns in the result rows, but rank using explicit labels.
    ranked_source = results.copy()
    ranked_source["vwap_key"] = ranked_source["max_vwap_extension_pct"].map(lambda x: "NONE" if pd.isna(x) else f"{float(x):g}")
    ranked_source["or_key"] = ranked_source["max_or_extension_atr"].map(lambda x: "NONE" if pd.isna(x) else f"{float(x):g}")
    key = ["trend", "vwap_key", "or_key", "rvol_min"]
    wide = ranked_source.pivot(index=key, columns="split")
    wide.columns = [f"{metric.lower()}_{split.lower()}" for metric, split in wide.columns]
    ranking = wide.reset_index()
    ranking["max_vwap_extension_pct"] = ranking["vwap_key"].map(lambda x: np.nan if x == "NONE" else float(x))
    ranking["max_or_extension_atr"] = ranking["or_key"].map(lambda x: np.nan if x == "NONE" else float(x))

    needed = [
        "trades_dev", "trades_validation", "expectancy_r_dev", "expectancy_r_validation",
        "profit_factor_dev", "profit_factor_validation", "expectancy_ci_low_dev",
        "expectancy_ci_low_validation",
    ]
    for col in needed:
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

    front = ["trend", "max_vwap_extension_pct", "max_or_extension_atr", "rvol_min"]
    rest = [c for c in ranking.columns if c not in front and c not in {"vwap_key", "or_key"}]
    ranking = ranking[front + rest]
    return results, ranking


def write_extension_reports(report_dir: Path, results: pd.DataFrame, ranking: pd.DataFrame) -> dict[str, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    results_path = report_dir / "extension_grid_results.csv"
    ranking_path = report_dir / "extension_ranking.csv"
    results.to_csv(results_path, index=False)
    ranking.to_csv(ranking_path, index=False)
    return {"grid_results": results_path, "ranking": ranking_path}

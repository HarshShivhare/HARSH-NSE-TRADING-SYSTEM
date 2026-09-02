from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Iterable

import pandas as pd

from .backtest import BacktestConfig, backtest_files, summarize_trades
from .strategy import StrategyConfig


def _summarize_run(trades: pd.DataFrame, capital: float, label: str, target_r: float) -> dict:
    s = summarize_trades(trades, capital)
    return {
        "split": label,
        "target_r": target_r,
        "trades": s["trades"],
        "wins": s["wins"],
        "win_rate": s["win_rate"],
        "profit_factor": s["profit_factor"],
        "expectancy_r": s["expectancy_r"],
        "gross_pnl": s["gross_pnl"],
        "charges": s["charges"],
        "net_pnl": s["net_pnl"],
        "return_pct": s["return_pct"],
        "max_drawdown_pct": s["max_drawdown_pct"],
        "avg_mfe_r": s["avg_mfe_r"],
        "avg_mae_r": s["avg_mae_r"],
    }


def target_sensitivity(
    files: Iterable[Path],
    strategy_cfg: StrategyConfig,
    bt_cfg: BacktestConfig,
    targets: list[float],
    dev_start: str,
    dev_end: str,
    test_start: str,
    test_end: str,
) -> tuple[pd.DataFrame, dict[tuple[str, float], pd.DataFrame]]:
    rows: list[dict] = []
    trade_sets: dict[tuple[str, float], pd.DataFrame] = {}
    for target in targets:
        cfg = replace(strategy_cfg, target_r=float(target))
        print(f"\n--- Target {target:.2f}R / DEVELOPMENT {dev_start} -> {dev_end} ---")
        dev = backtest_files(files, cfg, bt_cfg, session_start=dev_start, session_end=dev_end, quiet=True)
        rows.append(_summarize_run(dev, bt_cfg.initial_capital, "DEVELOPMENT", target))
        trade_sets[("DEVELOPMENT", target)] = dev

        print(f"--- Target {target:.2f}R / OUT-OF-SAMPLE {test_start} -> {test_end} ---")
        test = backtest_files(files, cfg, bt_cfg, session_start=test_start, session_end=test_end, quiet=True)
        rows.append(_summarize_run(test, bt_cfg.initial_capital, "OUT_OF_SAMPLE", target))
        trade_sets[("OUT_OF_SAMPLE", target)] = test
    return pd.DataFrame(rows), trade_sets


def write_research_reports(table: pd.DataFrame, report_dir: Path) -> dict[str, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "target_sensitivity.csv"
    table.to_csv(path, index=False)
    return {"target_sensitivity": path}

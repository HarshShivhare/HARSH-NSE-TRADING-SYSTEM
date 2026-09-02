from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
import math

import numpy as np
import pandas as pd

from .costs import IntradayEquityCostModel
from .risk import position_size
from .strategy import StrategyConfig, prepare_features, signal_mask
from .data_cleaner import clean_market_data


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 500_000.0
    risk_pct: float = 0.005
    max_trades_per_day: int = 1
    slippage_bps_each_side: float = 5.0


def _apply_slippage(price: float, bps: float, side: str) -> float:
    adj = bps / 10_000.0
    return price * (1 + adj) if side == "buy" else price * (1 - adj)


def _choose_stop(row: pd.Series, cfg: StrategyConfig, entry: float) -> float:
    if cfg.stop_mode == "atr":
        return entry - cfg.atr_multiple * float(row["atr"])
    if cfg.stop_mode == "opening_range":
        return float(row["or_low"])
    if cfg.stop_mode == "breakout_candle":
        return float(row["low"])
    raise ValueError(f"Unsupported stop_mode: {cfg.stop_mode}")


def run_symbol_backtest(
    df: pd.DataFrame,
    symbol: str,
    strategy_cfg: StrategyConfig,
    bt_cfg: BacktestConfig,
    costs: IntradayEquityCostModel | None = None,
    session_start: str | None = None,
    session_end: str | None = None,
) -> pd.DataFrame:
    costs = costs or IntradayEquityCostModel()
    f = prepare_features(df, strategy_cfg)
    f["signal"] = signal_mask(f, strategy_cfg)

    trades: list[dict] = []
    equity = bt_cfg.initial_capital

    start_date = pd.Timestamp(session_start).date() if session_start else None
    end_date = pd.Timestamp(session_end).date() if session_end else None

    for session, day in f.groupby("session", sort=True):
        if start_date and session < start_date:
            continue
        if end_date and session > end_date:
            continue
        day = day.reset_index(drop=True)
        signal_indices = day.index[day["signal"]].tolist()
        if not signal_indices:
            continue

        # V5: one entry per symbol/session by default. If max_trades_per_day is raised,
        # subsequent entries are allowed only after the prior trade has exited.
        daily_count = 0
        last_exit_idx = -1
        session_signal_bars = len(signal_indices)

        for sig_idx in signal_indices:
            if daily_count >= bt_cfg.max_trades_per_day:
                break
            if sig_idx <= last_exit_idx:
                continue
            if sig_idx + 1 >= len(day):
                continue

            signal_row = day.loc[sig_idx]
            entry_idx = sig_idx + 1
            entry_row = day.loc[entry_idx]
            raw_entry = float(entry_row["open"])

            # V10 extension-aware execution guards. The signal is generated on the
            # previous 5-minute close; at the next bar open we can choose not to
            # chase a breakout that has become too extended.
            signal_vwap = float(signal_row["vwap"]) if pd.notna(signal_row.get("vwap")) else np.nan
            vwap_extension_pct = (raw_entry / signal_vwap - 1.0) * 100.0 if np.isfinite(signal_vwap) and signal_vwap > 0 else np.nan
            signal_atr = float(signal_row["atr"]) if pd.notna(signal_row.get("atr")) else np.nan
            signal_or_high = float(signal_row["or_high"]) if pd.notna(signal_row.get("or_high")) else np.nan
            or_extension_atr = (raw_entry - signal_or_high) / signal_atr if np.isfinite(signal_atr) and signal_atr > 0 and np.isfinite(signal_or_high) else np.nan

            if strategy_cfg.max_vwap_extension_pct is not None:
                if not np.isfinite(vwap_extension_pct) or vwap_extension_pct > strategy_cfg.max_vwap_extension_pct:
                    continue
            if strategy_cfg.max_or_extension_atr is not None:
                if not np.isfinite(or_extension_atr) or or_extension_atr > strategy_cfg.max_or_extension_atr:
                    continue

            entry = _apply_slippage(raw_entry, bt_cfg.slippage_bps_each_side, "buy")
            stop = _choose_stop(signal_row, strategy_cfg, entry)
            if not math.isfinite(stop) or stop <= 0 or stop >= entry:
                continue

            risk_per_share = entry - stop
            target = entry + strategy_cfg.target_r * risk_per_share
            qty = position_size(equity, bt_cfg.risk_pct, entry, stop)
            if qty <= 0:
                continue

            exit_price_raw = float(day.iloc[-1]["close"])
            exit_reason = "EOD"
            exit_idx = len(day) - 1
            exit_time = day.iloc[-1]["date"]

            # Track favorable/adverse excursion from actual entry until exit.
            max_high = raw_entry
            min_low = raw_entry

            for j in range(entry_idx, len(day)):
                bar = day.loc[j]
                bar_low = float(bar["low"])
                bar_high = float(bar["high"])
                max_high = max(max_high, bar_high)
                min_low = min(min_low, bar_low)
                stop_hit = bar_low <= stop
                target_hit = bar_high >= target
                # Conservative same-bar ambiguity: assume stop first.
                if stop_hit:
                    exit_price_raw = stop
                    exit_reason = "STOP"
                    exit_idx = j
                    exit_time = bar["date"]
                    break
                if target_hit:
                    exit_price_raw = target
                    exit_reason = "TARGET"
                    exit_idx = j
                    exit_time = bar["date"]
                    break

            exit_price = _apply_slippage(exit_price_raw, bt_cfg.slippage_bps_each_side, "sell")
            gross_pnl = (exit_price - entry) * qty
            charge_breakdown = costs.estimate(entry, exit_price, qty)
            net_pnl = gross_pnl - charge_breakdown["total"]
            initial_risk = risk_per_share * qty
            gross_r = gross_pnl / initial_risk if initial_risk else np.nan
            net_r = net_pnl / initial_risk if initial_risk else np.nan
            mfe_r = (max_high - entry) / risk_per_share if risk_per_share else np.nan
            mae_r = (min_low - entry) / risk_per_share if risk_per_share else np.nan
            equity += net_pnl

            entry_ts = pd.Timestamp(entry_row["date"])
            exit_ts = pd.Timestamp(exit_time)
            duration_minutes = (exit_ts - entry_ts).total_seconds() / 60.0

            trades.append(
                {
                    "symbol": symbol,
                    "session": pd.Timestamp(session),
                    "session_signal_bars": session_signal_bars,
                    "signal_time": signal_row["date"],
                    "entry_time": entry_row["date"],
                    "exit_time": exit_time,
                    "entry": entry,
                    "stop": stop,
                    "target": target,
                    "exit": exit_price,
                    "qty": qty,
                    "gap_pct": signal_row["gap_pct"],
                    "rvol": signal_row["rvol"],
                    "atr": signal_row["atr"],
                    "entry_vs_vwap_pct": vwap_extension_pct,
                    "entry_vs_or_high_atr": or_extension_atr,
                    "trend_required": strategy_cfg.require_trend,
                    "max_vwap_extension_pct": strategy_cfg.max_vwap_extension_pct,
                    "max_or_extension_atr": strategy_cfg.max_or_extension_atr,
                    "exit_reason": exit_reason,
                    "gross_pnl": gross_pnl,
                    "charges": charge_breakdown["total"],
                    "net_pnl": net_pnl,
                    "gross_r": gross_r,
                    "r_multiple": net_r,
                    "mfe_r": mfe_r,
                    "mae_r": mae_r,
                    "duration_minutes": duration_minutes,
                    "equity_after": equity,
                }
            )
            daily_count += 1
            last_exit_idx = exit_idx

    return pd.DataFrame(trades)

def _max_consecutive_losses(pnl: pd.Series) -> int:
    max_run = run = 0
    for x in pnl:
        if x < 0:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return max_run


def summarize_trades(trades: pd.DataFrame, initial_capital: float) -> dict:
    if trades.empty:
        return {
            "trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
            "gross_pnl": 0.0, "charges": 0.0, "net_pnl": 0.0,
            "return_pct": 0.0, "profit_factor": None, "expectancy_r": None,
            "avg_win": None, "avg_loss": None, "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0, "sharpe_daily": None,
            "max_consecutive_losses": 0, "avg_realized_r": None,
            "avg_mfe_r": None, "avg_mae_r": None, "avg_duration_minutes": None,
        
        }

    t = trades.sort_values(["exit_time", "symbol"]).copy()
    pnl = t["net_pnl"].astype(float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())

    equity = initial_capital + pnl.cumsum()
    peak = equity.cummax()
    dd = equity - peak
    dd_pct = dd / peak.replace(0, np.nan)

    daily = t.groupby(pd.to_datetime(t["exit_time"]).dt.date)["net_pnl"].sum()
    daily_returns = daily / initial_capital
    sharpe = None
    if len(daily_returns) > 1 and daily_returns.std(ddof=1) > 0:
        sharpe = float(np.sqrt(252) * daily_returns.mean() / daily_returns.std(ddof=1))

    return {
        "trades": int(len(t)),
        "wins": int((pnl > 0).sum()),
        "losses": int((pnl < 0).sum()),
        "win_rate": float((pnl > 0).mean()),
        "gross_pnl": float(t["gross_pnl"].sum()),
        "charges": float(t["charges"].sum()),
        "net_pnl": float(pnl.sum()),
        "return_pct": float(pnl.sum() / initial_capital * 100.0),
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else None,
        "expectancy_r": float(t["r_multiple"].mean()),
        "avg_win": float(wins.mean()) if len(wins) else None,
        "avg_loss": float(losses.mean()) if len(losses) else None,
        "max_drawdown": float(dd.min()),
        "max_drawdown_pct": float(dd_pct.min() * 100.0),
        "sharpe_daily": sharpe,
        "max_consecutive_losses": _max_consecutive_losses(pnl),
        "avg_realized_r": float(t["r_multiple"].mean()),
        "avg_mfe_r": float(t["mfe_r"].mean()),
        "avg_mae_r": float(t["mae_r"].mean()),
        "avg_duration_minutes": float(t["duration_minutes"].mean()),
    }


def backtest_files(
    files: Iterable[Path],
    strategy_cfg: StrategyConfig,
    bt_cfg: BacktestConfig,
    session_start: str | None = None,
    session_end: str | None = None,
    quiet: bool = False,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in files:
        symbol = path.name.split("_")[1] if "_" in path.name else path.stem
        if not quiet:
            print(f"Backtesting {symbol}: {path.name}")
        raw_df = pd.read_parquet(path)
        cleaned = clean_market_data(raw_df, symbol)
        df = cleaned.data
        trades = run_symbol_backtest(df, symbol, strategy_cfg, bt_cfg, session_start=session_start, session_end=session_end)
        if not quiet:
            print(f"  {len(trades)} trades")
        if not trades.empty:
            frames.append(trades)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["exit_time", "symbol"]).reset_index(drop=True)


def write_reports(
    trades: pd.DataFrame,
    report_dir: Path,
    initial_capital: float,
    strategy_cfg: StrategyConfig,
    bt_cfg: BacktestConfig,
) -> dict[str, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_trades(trades, initial_capital)

    summary_df = pd.DataFrame([{
        **summary,
        **{f"strategy_{k}": v for k, v in asdict(strategy_cfg).items()},
        **{f"backtest_{k}": v for k, v in asdict(bt_cfg).items()},
    }])
    summary_path = report_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)

    trades_path = report_dir / "trades.parquet"
    trades.to_parquet(trades_path, index=False)

    by_symbol_path = report_dir / "by_symbol.csv"
    by_month_path = report_dir / "by_month.csv"
    by_year_path = report_dir / "by_year.csv"
    by_exit_path = report_dir / "by_exit_reason.csv"
    entry_time_path = report_dir / "by_entry_time.csv"
    diagnostics_path = report_dir / "execution_diagnostics.csv"

    if trades.empty:
        for path in [by_symbol_path, by_month_path, by_year_path, by_exit_path, entry_time_path, diagnostics_path]:
            pd.DataFrame().to_csv(path, index=False)
    else:
        by_symbol = trades.groupby("symbol").agg(
            trades=("net_pnl", "size"),
            net_pnl=("net_pnl", "sum"),
            avg_r=("r_multiple", "mean"),
            avg_mfe_r=("mfe_r", "mean"),
            avg_mae_r=("mae_r", "mean"),
            win_rate=("net_pnl", lambda s: (s > 0).mean()),
        ).sort_values("net_pnl", ascending=False)
        by_symbol.to_csv(by_symbol_path)

        temp = trades.copy()
        exit_dt = pd.to_datetime(temp["exit_time"])
        # Avoid timezone warning before converting to monthly Period.
        if getattr(exit_dt.dt, "tz", None) is not None:
            exit_dt = exit_dt.dt.tz_localize(None)
        temp["month"] = exit_dt.dt.to_period("M").astype(str)
        temp["year"] = exit_dt.dt.year
        temp.groupby("month").agg(
            trades=("net_pnl", "size"), net_pnl=("net_pnl", "sum"), avg_r=("r_multiple", "mean")
        ).to_csv(by_month_path)
        temp.groupby("year").agg(
            trades=("net_pnl", "size"), net_pnl=("net_pnl", "sum"), avg_r=("r_multiple", "mean")
        ).to_csv(by_year_path)

        trades.groupby("exit_reason").agg(
            trades=("net_pnl", "size"),
            pct=("net_pnl", lambda s: len(s) / len(trades)),
            net_pnl=("net_pnl", "sum"),
            avg_net_r=("r_multiple", "mean"),
            avg_gross_r=("gross_r", "mean"),
            avg_duration_minutes=("duration_minutes", "mean"),
        ).sort_values("trades", ascending=False).to_csv(by_exit_path)

        entry_dt = pd.to_datetime(trades["entry_time"])
        entry_bucket = entry_dt.dt.strftime("%H:%M")
        pd.DataFrame({"entry_time_bucket": entry_bucket, "net_pnl": trades["net_pnl"], "r": trades["r_multiple"]}).groupby("entry_time_bucket").agg(
            trades=("net_pnl", "size"), net_pnl=("net_pnl", "sum"), avg_r=("r", "mean")
        ).to_csv(entry_time_path)

        diagnostics = pd.DataFrame([
            {
                "candidate_symbol_sessions_entered": int(trades[["symbol", "session"]].drop_duplicates().shape[0]),
                "actual_trades": int(len(trades)),
                "signal_bars_on_entered_sessions": int(trades.groupby(["symbol", "session"])["session_signal_bars"].max().sum()),
                "target_exits": int((trades["exit_reason"] == "TARGET").sum()),
                "stop_exits": int((trades["exit_reason"] == "STOP").sum()),
                "eod_exits": int((trades["exit_reason"] == "EOD").sum()),
                "avg_winner_r": float(trades.loc[trades["net_pnl"] > 0, "r_multiple"].mean()) if (trades["net_pnl"] > 0).any() else np.nan,
                "avg_loser_r": float(trades.loc[trades["net_pnl"] < 0, "r_multiple"].mean()) if (trades["net_pnl"] < 0).any() else np.nan,
                "avg_realized_r": float(trades["r_multiple"].mean()),
                "avg_mfe_r": float(trades["mfe_r"].mean()),
                "avg_mae_r": float(trades["mae_r"].mean()),
                "avg_winner_duration_minutes": float(trades.loc[trades["net_pnl"] > 0, "duration_minutes"].mean()) if (trades["net_pnl"] > 0).any() else np.nan,
                "avg_loser_duration_minutes": float(trades.loc[trades["net_pnl"] < 0, "duration_minutes"].mean()) if (trades["net_pnl"] < 0).any() else np.nan,
            }
        ])
        diagnostics.to_csv(diagnostics_path, index=False)

    return {
        "summary": summary_path,
        "trades": trades_path,
        "by_symbol": by_symbol_path,
        "by_month": by_month_path,
        "by_year": by_year_path,
        "by_exit_reason": by_exit_path,
        "by_entry_time": entry_time_path,
        "execution_diagnostics": diagnostics_path,
    }


from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def _agg(grouped) -> pd.DataFrame:
    def pf(s: pd.Series):
        pos = s[s > 0].sum()
        neg = abs(s[s < 0].sum())
        return pos / neg if neg > 0 else np.nan

    return grouped.agg(
        trades=("net_pnl", "size"),
        wins=("net_pnl", lambda s: int((s > 0).sum())),
        win_rate=("net_pnl", lambda s: float((s > 0).mean())),
        net_pnl=("net_pnl", "sum"),
        avg_r=("r_multiple", "mean"),
        avg_mfe_r=("mfe_r", "mean"),
        avg_mae_r=("mae_r", "mean"),
        profit_factor=("net_pnl", pf),
    ).reset_index()


def stability_tables(trades: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if trades.empty:
        return {k: pd.DataFrame() for k in ["by_year", "by_month", "by_symbol", "by_gap", "by_rvol", "by_entry_time", "concentration"]}

    t = trades.copy()
    exit_dt = pd.to_datetime(t["exit_time"])
    local = exit_dt.dt.tz_localize(None) if getattr(exit_dt.dt, "tz", None) is not None else exit_dt
    t["year"] = local.dt.year
    t["month"] = local.dt.to_period("M").astype(str)
    entry_dt = pd.to_datetime(t["entry_time"])
    t["entry_time_bucket"] = entry_dt.dt.strftime("%H:%M")
    t["gap_bucket"] = pd.cut(t["gap_pct"], bins=[-np.inf, 1.0, 1.5, 2.0, 3.0, 5.0, np.inf], right=False,
                              labels=["<1.0", "1.0-1.5", "1.5-2.0", "2.0-3.0", "3.0-5.0", ">=5.0"])
    t["rvol_bucket"] = pd.cut(t["rvol"], bins=[-np.inf, 1.5, 2.0, 3.0, 5.0, np.inf], right=False,
                               labels=["<1.5", "1.5-2.0", "2.0-3.0", "3.0-5.0", ">=5.0"])

    by_year = _agg(t.groupby("year", dropna=False))
    by_month = _agg(t.groupby("month", dropna=False))
    by_symbol = _agg(t.groupby("symbol", dropna=False)).sort_values("net_pnl", ascending=False)
    by_gap = _agg(t.groupby("gap_bucket", observed=True, dropna=False))
    by_rvol = _agg(t.groupby("rvol_bucket", observed=True, dropna=False))
    by_entry = _agg(t.groupby("entry_time_bucket", dropna=False))

    by_trade = t[["symbol", "session", "net_pnl", "r_multiple"]].copy().sort_values("net_pnl", ascending=False)
    total = float(t["net_pnl"].sum())
    gross_profit = float(t.loc[t["net_pnl"] > 0, "net_pnl"].sum())
    top1 = float(by_trade.head(1)["net_pnl"].sum())
    top3 = float(by_trade.head(3)["net_pnl"].sum())
    top5 = float(by_trade.head(5)["net_pnl"].sum())
    sym = t.groupby("symbol")["net_pnl"].sum().sort_values(ascending=False)
    top_symbol = float(sym.head(1).sum()) if len(sym) else 0.0
    concentration = pd.DataFrame([{
        "trades": len(t),
        "total_net_pnl": total,
        "gross_positive_pnl": gross_profit,
        "top_1_trade_pnl": top1,
        "top_3_trades_pnl": top3,
        "top_5_trades_pnl": top5,
        "top_symbol_pnl": top_symbol,
        "top_1_trade_share_of_positive_pnl": top1 / gross_profit if gross_profit > 0 else np.nan,
        "top_3_trade_share_of_positive_pnl": top3 / gross_profit if gross_profit > 0 else np.nan,
        "top_5_trade_share_of_positive_pnl": top5 / gross_profit if gross_profit > 0 else np.nan,
        "top_symbol_share_of_positive_pnl": top_symbol / gross_profit if gross_profit > 0 else np.nan,
    }])

    return {
        "by_year": by_year,
        "by_month": by_month,
        "by_symbol": by_symbol,
        "by_gap": by_gap,
        "by_rvol": by_rvol,
        "by_entry_time": by_entry,
        "concentration": concentration,
    }


def write_stability_reports(tables: dict[str, pd.DataFrame], report_dir: Path) -> dict[str, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, df in tables.items():
        path = report_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        paths[name] = path
    return paths

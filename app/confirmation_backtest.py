from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Iterable
import math
import time

import numpy as np
import pandas as pd

from .costs import IntradayEquityCostModel
from .data_cleaner import clean_market_data
from .strategy import StrategyConfig, prepare_features, signal_mask


DELAY_MIN = 5
HORIZONS_MIN = (30, 60, 120)
CONFIRMATIONS = ("DELAY_ONLY", "ABOVE_OR", "ABOVE_OR_VWAP")
EXIT_MODES = ("TIME", "STOP1R_TIME")


@dataclass
class ConfirmationBacktestResult:
    trades: pd.DataFrame
    summary: pd.DataFrame
    capital_example: pd.DataFrame


def _split_name(session, dev_start, dev_end, validation_start, validation_end) -> str | None:
    d = pd.Timestamp(session).date()
    if pd.Timestamp(dev_start).date() <= d <= pd.Timestamp(dev_end).date():
        return "DEV"
    if pd.Timestamp(validation_start).date() <= d <= pd.Timestamp(validation_end).date():
        return "VALIDATION"
    return None


def _first_at_or_after(day: pd.DataFrame, ts: pd.Timestamp) -> int | None:
    idx = day.index[pd.to_datetime(day["date"]) >= ts]
    return None if len(idx) == 0 else int(idx[0])


def _apply_slippage(price: float, bps: float, side: str) -> float:
    adj = bps / 10_000.0
    return price * (1 + adj) if side == "buy" else price * (1 - adj)


def _bootstrap_mean_ci(values: pd.Series, samples: int = 1000, seed: int = 42) -> tuple[float, float]:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(x) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=float)
    for i in range(samples):
        means[i] = rng.choice(x, size=len(x), replace=True).mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _confirmation_ok(mode: str, entry_row: pd.Series, or_high: float) -> bool:
    close = float(entry_row["close"])
    above_or = np.isfinite(or_high) and close >= or_high
    vwap = float(entry_row["vwap"]) if pd.notna(entry_row.get("vwap")) else np.nan
    above_vwap = np.isfinite(vwap) and close >= vwap
    if mode == "DELAY_ONLY":
        return True
    if mode == "ABOVE_OR":
        return bool(above_or)
    if mode == "ABOVE_OR_VWAP":
        return bool(above_or and above_vwap)
    raise ValueError(f"Unknown confirmation mode: {mode}")


def _trade_variant(
    day: pd.DataFrame,
    entry_idx: int,
    horizon_min: int,
    exit_mode: str,
    risk_unit: float,
    slippage_bps: float,
    account_capital: float,
    risk_pct: float,
    costs: IntradayEquityCostModel,
) -> dict | None:
    entry_row = day.loc[entry_idx]
    entry_ts = pd.Timestamp(entry_row["date"])
    target_idx = _first_at_or_after(day, entry_ts + pd.Timedelta(minutes=horizon_min))
    if target_idx is None:
        return None

    raw_entry = float(entry_row["open"])
    entry = _apply_slippage(raw_entry, slippage_bps, "buy")
    if not np.isfinite(entry) or entry <= 0 or risk_unit <= 0:
        return None

    stop = entry - risk_unit
    if stop <= 0:
        return None

    exit_idx = target_idx
    raw_exit = float(day.loc[target_idx, "close"])
    exit_reason = f"TIME_{horizon_min}M"

    if exit_mode == "STOP1R_TIME":
        for j in range(entry_idx, target_idx + 1):
            if float(day.loc[j, "low"]) <= stop:
                raw_exit = stop
                exit_idx = j
                exit_reason = "STOP_1R"
                break

    exit_price = _apply_slippage(raw_exit, slippage_bps, "sell")
    gross_per_share = exit_price - entry
    gross_r = gross_per_share / risk_unit

    risk_budget = account_capital * risk_pct
    qty_by_risk = math.floor(risk_budget / risk_unit)
    qty_by_cash = math.floor(account_capital / entry)
    qty = max(0, min(qty_by_risk, qty_by_cash))
    if qty <= 0:
        return None

    charge_breakdown = costs.estimate(entry, exit_price, qty)
    charges = float(charge_breakdown["total"])
    gross_pnl = gross_per_share * qty
    net_pnl = gross_pnl - charges
    actual_risk = risk_unit * qty
    net_r = net_pnl / actual_risk if actual_risk > 0 else np.nan
    net_pct_capital = net_pnl / account_capital * 100.0
    buy_notional = entry * qty

    return {
        "exit_time": day.loc[exit_idx, "date"],
        "exit_reason": exit_reason,
        "entry": entry,
        "exit": exit_price,
        "stop": stop,
        "qty_1l": qty,
        "buy_notional_1l": buy_notional,
        "risk_budget_1l": risk_budget,
        "actual_risk_1l": actual_risk,
        "gross_pnl_1l": gross_pnl,
        "charges_1l": charges,
        "net_pnl_1l": net_pnl,
        "net_pct_1l": net_pct_capital,
        "gross_r": gross_r,
        "net_r": net_r,
    }


def run_confirmation_backtest(
    files: Iterable[Path],
    scfg: StrategyConfig,
    dev_start: str = "2023-09-01",
    dev_end: str = "2025-06-30",
    validation_start: str = "2025-07-01",
    validation_end: str = "2025-12-31",
    account_capital: float = 100_000.0,
    risk_pct: float = 0.005,
    slippage_bps_each_side: float = 5.0,
    bootstrap_samples: int = 1000,
) -> ConfirmationBacktestResult:
    files = list(files)
    rows: list[dict] = []
    costs = IntradayEquityCostModel()
    started = time.monotonic()

    for i, path in enumerate(files, start=1):
        symbol = path.name.split("_")[1] if "_" in path.name else path.stem
        print(f"[{i}/{len(files)}] V13 confirmation backtest: {symbol} | trades={len(rows)} | elapsed {(time.monotonic()-started)/60:.1f}m", flush=True)
        raw = pd.read_parquet(path)
        cleaned = clean_market_data(raw, symbol).data
        if cleaned.empty:
            continue
        f = prepare_features(cleaned, scfg)
        f["signal"] = signal_mask(f, scfg)

        for session, day in f.groupby("session", sort=True):
            split = _split_name(session, dev_start, dev_end, validation_start, validation_end)
            if split is None:
                continue
            day = day.reset_index(drop=True)
            candidates = day.index[day["signal"]].tolist()
            if not candidates:
                continue
            sig_idx = int(candidates[0])
            signal_row = day.loc[sig_idx]
            signal_ts = pd.Timestamp(signal_row["date"])
            entry_idx = _first_at_or_after(day, signal_ts + pd.Timedelta(minutes=DELAY_MIN))
            if entry_idx is None or entry_idx <= sig_idx:
                continue
            entry_row = day.loc[entry_idx]
            atr = float(entry_row["atr"]) if pd.notna(entry_row.get("atr")) else np.nan
            if not np.isfinite(atr) or atr <= 0:
                continue
            risk_unit = scfg.atr_multiple * atr
            or_high = float(signal_row["or_high"]) if pd.notna(signal_row.get("or_high")) else np.nan

            for confirmation in CONFIRMATIONS:
                if not _confirmation_ok(confirmation, entry_row, or_high):
                    continue
                for horizon in HORIZONS_MIN:
                    for exit_mode in EXIT_MODES:
                        trade = _trade_variant(
                            day, entry_idx, horizon, exit_mode, risk_unit,
                            slippage_bps_each_side, account_capital, risk_pct, costs,
                        )
                        if trade is None:
                            continue
                        trade.update({
                            "split": split,
                            "symbol": symbol,
                            "session": pd.Timestamp(session),
                            "signal_time": signal_row["date"],
                            "entry_time": entry_row["date"],
                            "delay_min": DELAY_MIN,
                            "confirmation": confirmation,
                            "horizon_min": horizon,
                            "exit_mode": exit_mode,
                            "atr": atr,
                            "risk_unit": risk_unit,
                            "gap_pct": float(signal_row["gap_pct"]),
                            "rvol": float(signal_row["rvol"]),
                            "entry_close_above_or": bool(float(entry_row["close"]) >= or_high) if np.isfinite(or_high) else False,
                            "entry_close_above_vwap": bool(float(entry_row["close"]) >= float(entry_row["vwap"])) if pd.notna(entry_row.get("vwap")) else False,
                        })
                        rows.append(trade)

    trades = pd.DataFrame(rows)
    if trades.empty:
        return ConfirmationBacktestResult(trades, pd.DataFrame(), pd.DataFrame())

    summary_rows = []
    example_rows = []
    group_cols = ["split", "confirmation", "exit_mode", "horizon_min"]
    for keys, g in trades.groupby(group_cols):
        split, confirmation, exit_mode, horizon = keys
        net_r = pd.to_numeric(g["net_r"], errors="coerce").dropna()
        lo, hi = _bootstrap_mean_ci(net_r, samples=bootstrap_samples, seed=1300 + int(horizon))
        pnl = pd.to_numeric(g["net_pnl_1l"], errors="coerce")
        summary_rows.append({
            "split": split,
            "confirmation": confirmation,
            "exit_mode": exit_mode,
            "horizon_min": int(horizon),
            "trades": int(len(g)),
            "win_rate": float((pnl > 0).mean()),
            "gross_expectancy_r": float(pd.to_numeric(g["gross_r"], errors="coerce").mean()),
            "net_expectancy_r": float(net_r.mean()),
            "net_ci_low": lo,
            "net_ci_high": hi,
            "avg_net_pct_1l": float(pd.to_numeric(g["net_pct_1l"], errors="coerce").mean()),
            "avg_net_pnl_1l": float(pnl.mean()),
            "median_net_pnl_1l": float(pnl.median()),
            "avg_charges_1l": float(pd.to_numeric(g["charges_1l"], errors="coerce").mean()),
            "stop_rate": float((g["exit_reason"] == "STOP_1R").mean()),
        })
        example_rows.append({
            "split": split,
            "confirmation": confirmation,
            "exit_mode": exit_mode,
            "horizon_min": int(horizon),
            "starting_capital": account_capital,
            "risk_pct": risk_pct * 100.0,
            "risk_budget_per_trade": account_capital * risk_pct,
            "trades": int(len(g)),
            "avg_invested_notional": float(pd.to_numeric(g["buy_notional_1l"], errors="coerce").mean()),
            "avg_net_pnl_per_trade": float(pnl.mean()),
            "avg_return_on_1l_pct_per_trade": float(pd.to_numeric(g["net_pct_1l"], errors="coerce").mean()),
            "sample_sum_pnl_independent_trades": float(pnl.sum()),
        })

    return ConfirmationBacktestResult(trades, pd.DataFrame(summary_rows), pd.DataFrame(example_rows))


def _svg_net_expectancy(summary: pd.DataFrame, confirmation: str, exit_mode: str) -> str:
    data = summary[(summary["confirmation"] == confirmation) & (summary["exit_mode"] == exit_mode)].copy()
    if data.empty:
        return "<p>No data.</p>"
    width, height, left, top, bottom = 900, 390, 75, 35, 65
    horizons = sorted(data["horizon_min"].unique())
    vals = pd.to_numeric(data["net_expectancy_r"], errors="coerce").dropna()
    extent = max(.10, float(vals.abs().max()) * 1.35 if len(vals) else .10)
    ymin, ymax = -extent, extent
    plot_h = height - top - bottom
    group_w = (width - left - 30) / max(1, len(horizons))
    bar_w = 36
    def ymap(v): return top + (ymax - v) / (ymax - ymin) * plot_h
    zero = ymap(0)
    parts = [f'<svg viewBox="0 0 {width} {height}">', f'<line x1="{left}" y1="{zero:.1f}" x2="{width-20}" y2="{zero:.1f}" class="axis"/>']
    for i, horizon in enumerate(horizons):
        cx = left + group_w * (i + .5)
        parts.append(f'<text x="{cx:.1f}" y="{height-25}" text-anchor="middle">{horizon}m</text>')
        for si, split in enumerate(("DEV", "VALIDATION")):
            row = data[(data.horizon_min == horizon) & (data.split == split)]
            if row.empty:
                continue
            v = float(row.iloc[0].net_expectancy_r)
            x = cx - 40 + si * bar_w
            y = min(ymap(v), zero)
            h = abs(ymap(v) - zero)
            cls = "bar-dev" if split == "DEV" else "bar-val"
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w-5}" height="{max(1,h):.1f}" class="{cls}"><title>{split}: {v:+.3f}R</title></rect>')
    parts.append('</svg>')
    return ''.join(parts)


def _svg_rupee_example(example: pd.DataFrame, confirmation: str, exit_mode: str) -> str:
    data = example[(example["confirmation"] == confirmation) & (example["exit_mode"] == exit_mode)].copy()
    if data.empty:
        return "<p>No data.</p>"
    width, height, left, top, bottom = 900, 390, 75, 35, 65
    horizons = sorted(data["horizon_min"].unique())
    vals = pd.to_numeric(data["avg_net_pnl_per_trade"], errors="coerce").dropna()
    extent = max(100.0, float(vals.abs().max()) * 1.35 if len(vals) else 100.0)
    ymin, ymax = -extent, extent
    plot_h = height - top - bottom
    group_w = (width - left - 30) / max(1, len(horizons))
    bar_w = 36
    def ymap(v): return top + (ymax - v) / (ymax - ymin) * plot_h
    zero = ymap(0)
    parts = [f'<svg viewBox="0 0 {width} {height}">', f'<line x1="{left}" y1="{zero:.1f}" x2="{width-20}" y2="{zero:.1f}" class="axis"/>']
    for i, horizon in enumerate(horizons):
        cx = left + group_w * (i + .5)
        parts.append(f'<text x="{cx:.1f}" y="{height-25}" text-anchor="middle">{horizon}m</text>')
        for si, split in enumerate(("DEV", "VALIDATION")):
            row = data[(data.horizon_min == horizon) & (data.split == split)]
            if row.empty:
                continue
            v = float(row.iloc[0].avg_net_pnl_per_trade)
            x = cx - 40 + si * bar_w
            y = min(ymap(v), zero)
            h = abs(ymap(v) - zero)
            cls = "bar-dev" if split == "DEV" else "bar-val"
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w-5}" height="{max(1,h):.1f}" class="{cls}"><title>{split}: ₹{v:+,.0f} per trade</title></rect>')
    parts.append('</svg>')
    return ''.join(parts)


def write_confirmation_reports(result: ConfirmationBacktestResult, report_dir: Path, scfg: StrategyConfig, account_capital: float, risk_pct: float) -> dict[str, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    for name, table in (("summary", result.summary), ("capital_example", result.capital_example)):
        path = report_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        outputs[name] = path
    trades_path = report_dir / "confirmation_trades.parquet"
    result.trades.to_parquet(trades_path, index=False)
    outputs["trades"] = trades_path

    focus_confirmation = "ABOVE_OR_VWAP"
    focus_exit = "STOP1R_TIME"
    focus = result.capital_example[(result.capital_example.confirmation == focus_confirmation) & (result.capital_example.exit_mode == focus_exit)].copy()
    focus_table = focus[[c for c in ["split","horizon_min","trades","risk_budget_per_trade","avg_invested_notional","avg_net_pnl_per_trade","avg_return_on_1l_pct_per_trade"] if c in focus.columns]]

    css = '''body{font-family:system-ui,-apple-system,sans-serif;margin:28px;background:#f7f7f8;color:#171717}.wrap{max-width:1100px;margin:auto}.card{background:white;border:1px solid #ddd;border-radius:14px;padding:20px;margin:16px 0;box-shadow:0 1px 4px #0001}.muted{color:#666}.axis{stroke:#666;stroke-width:1}.bar-dev{fill:#496f9b}.bar-val{fill:#bd6b4d}svg text{font-size:12px;fill:#555}table{border-collapse:collapse;width:100%;font-size:12px}th,td{border-bottom:1px solid #eee;padding:7px;text-align:right}th:first-child,td:first-child{text-align:left}.legend span{display:inline-block;margin-right:18px}.sw{width:12px;height:12px;display:inline-block;margin-right:5px}.dev{background:#496f9b}.val{background:#bd6b4d}.note{background:#fff8df;border:1px solid #ead99b;border-radius:10px;padding:14px}'''
    config_text = (f"Frozen hypothesis: +{DELAY_MIN}m entry | Trend={'ON' if scfg.require_trend else 'OFF'} | Gap ≥ {scfg.gap_min_pct:g}% | "
                   f"RVOL ≥ {scfg.rvol_min:g} | OR={scfg.opening_range_minutes}m | ATR risk unit={scfg.atr_multiple:g}×ATR")
    html = f'''<!doctype html><html><head><meta charset="utf-8"><title>V13 Confirmation Backtest</title><style>{css}</style></head><body><div class="wrap">
<h1>V13 +5m Confirmation Trade Construction</h1><p class="muted">{escape(config_text)}</p><p class="legend"><span><i class="sw dev"></i>DEV</span><span><i class="sw val"></i>VALIDATION</span></p>
<div class="card"><h2>Net expectancy — confirmed above OR + VWAP, 1R protective stop</h2><p>After slippage and Indian intraday transaction-cost model.</p>{_svg_net_expectancy(result.summary, focus_confirmation, focus_exit)}</div>
<div class="card"><h2>₹1,00,000 account example</h2><p>Account starts at ₹1,00,000 for each trade, risk budget is {risk_pct*100:.2f}% = ₹{account_capital*risk_pct:,.0f} per trade, and quantity is capped by available cash (no leverage).</p>{_svg_rupee_example(result.capital_example, focus_confirmation, focus_exit)}
{focus_table.to_html(index=False, float_format=lambda x:f'{x:.4f}') if not focus_table.empty else '<p>No focus trades.</p>'}
<p class="note"><b>How to read this:</b> “avg net P/L per trade” is the historical sample average for one independently sized ₹1L account trade. It is not a compounded portfolio return. Signals can overlap across stocks, and V13 does not yet allocate shared capital across simultaneous positions.</p></div>
<div class="card"><h2>All confirmation / exit variants</h2>{result.summary.to_html(index=False,float_format=lambda x:f'{x:.4f}')}</div>
<div class="card"><h2>₹1L illustrative economics — all variants</h2>{result.capital_example.to_html(index=False,float_format=lambda x:f'{x:.4f}')}</div>
<p class="muted">Research only. 2026 FINAL OOS is intentionally not read by V13.</p></div></body></html>'''
    dashboard = report_dir / "confirmation_dashboard.html"
    dashboard.write_text(html, encoding="utf-8")
    outputs["dashboard"] = dashboard
    return outputs

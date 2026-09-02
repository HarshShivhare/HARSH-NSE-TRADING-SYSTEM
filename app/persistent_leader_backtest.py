from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import math

import numpy as np
import pandas as pd

from .costs import IntradayEquityCostModel
from .data_cleaner import clean_market_data
from .persistent_leader_discovery import run_persistent_leader_discovery
from .strategy import StrategyConfig, prepare_features

# V18 freezes the broad V17 finding instead of tuning the best timestamp.
# Candidates may appear at 10:30, 11:00, or 12:00 and must be TOP10 for >=3
# consecutive research checkpoints, above VWAP, and RVOL >= 1.5.
ENTRY_VARIANTS = ("IMMEDIATE", "VWAP_RECLAIM", "NEW_HIGH")
STOP_MODES = ("ATR1", "ATR1_5")
EXIT_MODES = ("TIME60", "TIME120", "TARGET2R_EOD")
SELECTIONS = ("TOP1", "TOP2", "TOP3")
TRADE_SNAPSHOT_TIMES = ("10:30", "11:00", "12:00")
ENTRY_SEARCH_MINUTES = 30


@dataclass
class PersistentLeaderBacktestResult:
    candidates: pd.DataFrame
    intents: pd.DataFrame
    trades: pd.DataFrame
    portfolio_summary: pd.DataFrame


def _apply_slippage(price: float, bps: float, side: str) -> float:
    a = bps / 10_000.0
    return price * (1 + a) if side == "buy" else price * (1 - a)


def _candidate_rank(events: pd.DataFrame) -> pd.DataFrame:
    c = events[
        events["snapshot_time"].isin(TRADE_SNAPSHOT_TIMES)
        & (events["top10_streak"] >= 3)
        & events["above_vwap"]
        & (events["rvol"] >= 1.5)
    ].copy()
    if c.empty:
        return c
    # Transparent causal ordering using only information known at the checkpoint.
    c = c.sort_values(
        ["split", "timestamp", "momentum_percentile", "top10_streak", "rvol", "vwap_pct", "symbol"],
        ascending=[True, True, False, False, False, False, True],
    )
    c["rank_at_snapshot"] = c.groupby(["split", "timestamp"]).cumcount() + 1
    return c.reset_index(drop=True)


def _find_entry(day: pd.DataFrame, snapshot_ts: pd.Timestamp, variant: str):
    ids = day.index[pd.to_datetime(day["date"]) == snapshot_ts]
    if len(ids) == 0:
        return None
    sidx = int(ids[0])
    if sidx + 1 >= len(day):
        return None
    snap = day.loc[sidx]
    atr = float(snap["atr"]) if pd.notna(snap["atr"]) else np.nan
    if not np.isfinite(atr) or atr <= 0:
        return None

    if variant == "IMMEDIATE":
        confirm_idx = sidx
    else:
        end = snapshot_ts + pd.Timedelta(minutes=ENTRY_SEARCH_MINUTES)
        future = day.index[(pd.to_datetime(day["date"]) > snapshot_ts) & (pd.to_datetime(day["date"]) <= end)]
        confirm_idx = None
        if variant == "VWAP_RECLAIM":
            for j in future:
                r = day.loc[j]
                vw = float(r["vwap"]) if pd.notna(r["vwap"]) else np.nan
                if np.isfinite(vw) and float(r["low"]) <= vw and float(r["close"]) >= vw:
                    confirm_idx = int(j); break
        elif variant == "NEW_HIGH":
            prior_high = float(day.loc[:sidx, "high"].max())
            for j in future:
                if float(day.loc[j, "high"]) > prior_high:
                    confirm_idx = int(j); break
        else:
            raise ValueError(variant)
        if confirm_idx is None:
            return None

    entry_idx = confirm_idx + 1
    if entry_idx >= len(day) or str(day.loc[entry_idx, "time"]) > "14:45":
        return None
    return entry_idx, atr


def _exit_path(day: pd.DataFrame, entry_idx: int, raw_entry: float, atr: float, stop_mode: str, exit_mode: str):
    stop_dist = atr if stop_mode == "ATR1" else 1.5 * atr
    raw_stop = raw_entry - stop_dist
    if raw_stop <= 0:
        return None
    if exit_mode == "TIME60":
        end_ts = pd.Timestamp(day.loc[entry_idx, "date"]) + pd.Timedelta(minutes=60)
        target = None
    elif exit_mode == "TIME120":
        end_ts = pd.Timestamp(day.loc[entry_idx, "date"]) + pd.Timedelta(minutes=120)
        target = None
    elif exit_mode == "TARGET2R_EOD":
        end_ts = pd.Timestamp(day.iloc[-1]["date"])
        target = raw_entry + 2.0 * stop_dist
    else:
        raise ValueError(exit_mode)
    ids = day.index[(day.index >= entry_idx) & (pd.to_datetime(day["date"]) >= end_ts)]
    end_idx = int(ids[0]) if len(ids) else int(day.index[-1])
    raw_exit = float(day.loc[end_idx, "close"]); reason = "TIME" if target is None else "EOD"
    for j in range(entry_idx, end_idx + 1):
        lo, hi = float(day.loc[j, "low"]), float(day.loc[j, "high"])
        # Conservative same-bar ordering: stop before target.
        if lo <= raw_stop:
            return int(j), raw_stop, "STOP", stop_dist
        if target is not None and hi >= target:
            return int(j), target, "TARGET2R", stop_dist
    return end_idx, raw_exit, reason, stop_dist


def _build_intents(files: Iterable[Path], cfg: StrategyConfig, candidates: pd.DataFrame, slippage_bps: float):
    wanted = set(candidates["symbol"].unique()) if not candidates.empty else set()
    rows = []
    for path in files:
        symbol = path.name.split("_")[1] if "_" in path.name else path.stem
        if symbol not in wanted:
            continue
        clean = clean_market_data(pd.read_parquet(path), symbol).data
        if clean.empty:
            continue
        f = prepare_features(clean, cfg)
        for session, day0 in f.groupby("session", sort=True):
            cc = candidates[(candidates.symbol == symbol) & (candidates.session == pd.Timestamp(session))]
            if cc.empty:
                continue
            day = day0.sort_values("date").reset_index(drop=True)
            for _, c in cc.iterrows():
                for entry_variant in ENTRY_VARIANTS:
                    found = _find_entry(day, pd.Timestamp(c.timestamp), entry_variant)
                    if found is None:
                        continue
                    entry_idx, atr = found
                    raw_entry = float(day.loc[entry_idx, "open"])
                    entry = _apply_slippage(raw_entry, slippage_bps, "buy")
                    for stop_mode in STOP_MODES:
                        for exit_mode in EXIT_MODES:
                            xp = _exit_path(day, entry_idx, raw_entry, atr, stop_mode, exit_mode)
                            if xp is None:
                                continue
                            exit_idx, raw_exit, reason, stop_dist = xp
                            exit_price = _apply_slippage(raw_exit, slippage_bps, "sell")
                            rows.append({
                                "split": c.split, "symbol": symbol, "session": pd.Timestamp(session),
                                "snapshot_time": c.snapshot_time, "signal_time": c.timestamp,
                                "rank_at_snapshot": int(c.rank_at_snapshot), "momentum_percentile": float(c.momentum_percentile),
                                "top10_streak": int(c.top10_streak), "rvol": float(c.rvol), "vwap_pct": float(c.vwap_pct),
                                "entry_variant": entry_variant, "stop_mode": stop_mode, "exit_mode": exit_mode,
                                "entry_time": day.loc[entry_idx, "date"], "exit_time": day.loc[exit_idx, "date"],
                                "entry_price": entry, "exit_price": exit_price, "risk_per_share": entry - (raw_entry - stop_dist),
                                "exit_reason": reason,
                            })
    return pd.DataFrame(rows)


def _simulate_variant(intents: pd.DataFrame, selection: str, capital: float, risk_pct: float):
    max_positions = int(selection[-1])
    x = intents[intents.rank_at_snapshot <= max_positions].sort_values(["entry_time", "rank_at_snapshot", "symbol"]).copy()
    cash = float(capital); realized = 0.0; open_pos = []; trades = []; costs = IntradayEquityCostModel()

    def close_due(ts=None, force=False):
        nonlocal cash, realized, open_pos
        remain = []
        for p in open_pos:
            if force or pd.Timestamp(p["exit_time"]) <= ts:
                proceeds = p["exit_price"] * p["qty"]
                ch = costs.estimate(p["entry_price"], p["exit_price"], p["qty"])
                gross = (p["exit_price"] - p["entry_price"]) * p["qty"]
                net = gross - ch["total"]
                cash += proceeds - (ch["stt"] + ch["exchange"] + ch["sebi"] + ch["gst"])
                # Buy-side brokerage/stamp were reserved at entry below; add sell-side economics through full net reconciliation.
                cash = p["cash_before_entry"] - p["entry_price"] * p["qty"] - p["buy_reserve"] + proceeds - (ch["total"] - p["buy_reserve"])
                realized += net
                p.update({"gross_pnl": gross, "charges": ch["total"], "net_pnl": net, "equity_after": capital + realized})
                trades.append(p)
            else:
                remain.append(p)
        open_pos = remain

    # Recompute cash robustly from realized P&L and cost basis of open positions after every event.
    def refresh_cash():
        invested = sum(p["entry_price"] * p["qty"] for p in open_pos)
        return max(0.0, capital + realized - invested)

    for _, r in x.iterrows():
        ts = pd.Timestamp(r.entry_time)
        close_due(ts); cash = refresh_cash()
        if any(p["symbol"] == r.symbol for p in open_pos) or len(open_pos) >= max_positions:
            continue
        equity = capital + realized
        slots = max_positions - len(open_pos)
        cash_cap = cash / max(1, slots)
        risk_budget = equity * risk_pct
        rps = float(r.risk_per_share)
        if not np.isfinite(rps) or rps <= 0:
            continue
        qty = min(math.floor(risk_budget / rps), math.floor(cash_cap / float(r.entry_price)))
        if qty <= 0:
            continue
        buy_turnover = float(r.entry_price) * qty
        # Reserve estimated buy-side friction so cash is never overstated. Final charges are reconciled at exit.
        est = costs.estimate(float(r.entry_price), float(r.entry_price), qty)
        buy_reserve = min(buy_turnover * costs.brokerage_rate, costs.brokerage_cap_per_order) + buy_turnover * costs.stamp_buy_rate
        p = r.to_dict(); p.update({"selection": selection, "qty": qty, "cash_before_entry": cash,
                                   "buy_reserve": buy_reserve, "capital_at_entry": equity})
        open_pos.append(p); cash = refresh_cash()

    close_due(force=True); cash = refresh_cash()
    t = pd.DataFrame(trades)
    ending = capital + realized
    return t, {
        "selection": selection, "trades": len(t), "starting_capital": capital, "net_pnl": realized,
        "ending_capital": ending, "return_pct": (ending / capital - 1.0) * 100.0,
        "win_rate": float((t.net_pnl > 0).mean()) if len(t) else np.nan,
        "total_charges": float(t.charges.sum()) if len(t) else 0.0,
        "profit_factor": (float(t.loc[t.net_pnl > 0, "net_pnl"].sum()) / -float(t.loc[t.net_pnl < 0, "net_pnl"].sum())) if len(t) and (t.net_pnl < 0).any() else np.nan,
    }


def run_persistent_leader_backtest(files: Iterable[Path], cfg: StrategyConfig, dev_start, dev_end, validation_start, validation_end,
                                   account_capital=100000.0, risk_pct=.005, slippage_bps_each_side=5.0, bootstrap_samples=500):
    files = list(files)
    base = run_persistent_leader_discovery(files, cfg, dev_start, dev_end, validation_start, validation_end, bootstrap_samples)
    candidates = _candidate_rank(base.events)
    if candidates.empty:
        return PersistentLeaderBacktestResult(candidates, pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    intents = _build_intents(files, cfg, candidates, slippage_bps_each_side)
    all_trades, summaries = [], []
    for split in ("DEV", "VALIDATION"):
        si = intents[intents.split == split]
        for entry_variant in ENTRY_VARIANTS:
            for stop_mode in STOP_MODES:
                for exit_mode in EXIT_MODES:
                    vi = si[(si.entry_variant == entry_variant) & (si.stop_mode == stop_mode) & (si.exit_mode == exit_mode)]
                    for selection in SELECTIONS:
                        t, s = _simulate_variant(vi, selection, account_capital, risk_pct)
                        s.update({"split": split, "entry_variant": entry_variant, "stop_mode": stop_mode, "exit_mode": exit_mode})
                        summaries.append(s)
                        if not t.empty:
                            t["split"] = split; t["entry_variant"] = entry_variant; t["stop_mode"] = stop_mode; t["exit_mode"] = exit_mode
                            all_trades.append(t)
    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    summary = pd.DataFrame(summaries)
    return PersistentLeaderBacktestResult(candidates, intents, trades, summary)


def write_persistent_leader_backtest_reports(r: PersistentLeaderBacktestResult, report_dir: Path):
    report_dir.mkdir(parents=True, exist_ok=True); paths = {}
    for name, df in (("candidates", r.candidates), ("intents", r.intents), ("trades", r.trades), ("portfolio_summary", r.portfolio_summary)):
        p = report_dir / (name + (".parquet" if name in ("candidates", "intents", "trades") else ".csv"))
        df.to_parquet(p, index=False) if p.suffix == ".parquet" else df.to_csv(p, index=False)
        paths[name] = p
    if not r.portfolio_summary.empty:
        s = r.portfolio_summary.copy()
        pivot = s.pivot_table(index=["entry_variant", "stop_mode", "exit_mode", "selection"], columns="split", values="return_pct", aggfunc="first").reset_index()
        if "DEV" in pivot and "VALIDATION" in pivot:
            pivot["worst_split_return_pct"] = pivot[["DEV", "VALIDATION"]].min(axis=1)
            pivot = pivot.sort_values("worst_split_return_pct", ascending=False)
        table = pivot.to_html(index=False, float_format=lambda x: f"{x:.2f}")
        html = f'''<!doctype html><meta charset="utf-8"><title>V18 Persistent Leader Backtest</title>
<style>body{{font-family:system-ui;margin:30px;max-width:1300px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:7px;border-bottom:1px solid #ddd;font-size:12px}}.warn{{padding:12px;background:#fff3cd}}</style>
<h1>V18 Persistent-Leader Tradable Backtest</h1><p class="warn"><b>DEV + Validation only; 2026 locked.</b> Shared ₹1L capital, causal TOP1/TOP2/TOP3 selection, 0.5% equity risk/trade, cash-only sizing, 5 bps slippage each side, and Indian intraday charges.</p>
<h2>₹1L portfolio return comparison</h2>{table}<p>Candidate definition is frozen broadly from V17: 10:30 onward, TOP10 for at least 3 consecutive checkpoints, above VWAP, RVOL ≥1.5. This is still a historical research simulation, not a live-trading recommendation.</p>'''
        hp = report_dir / "persistent_leader_backtest_dashboard.html"; hp.write_text(html, encoding="utf-8"); paths["dashboard"] = hp
    return paths

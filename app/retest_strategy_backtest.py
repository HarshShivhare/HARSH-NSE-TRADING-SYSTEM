from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import math
import time

import numpy as np
import pandas as pd

from .costs import IntradayEquityCostModel
from .data_cleaner import clean_market_data
from .strategy import StrategyConfig, prepare_features, signal_mask

# V15 freezes the broad V14 discovery result. These are not CLI optimisation knobs.
MAX_RETEST_MIN = 10
RETEST_RVOL_MIN = 1.5
RETEST_RVOL_MAX = 3.0
HORIZONS_MIN = (60, 120)
STOP_MODES = ("PULLBACK_LOW", "ATR1", "ATR1_5")
SELECTIONS = ("ALL", "TOP3", "TOP2", "TOP1")


@dataclass
class RetestStrategyResult:
    candidates: pd.DataFrame
    trades: pd.DataFrame
    summary: pd.DataFrame
    selection_summary: pd.DataFrame


def _split_name(session, dev_start, dev_end, validation_start, validation_end):
    d = pd.Timestamp(session).date()
    if pd.Timestamp(dev_start).date() <= d <= pd.Timestamp(dev_end).date():
        return "DEV"
    if pd.Timestamp(validation_start).date() <= d <= pd.Timestamp(validation_end).date():
        return "VALIDATION"
    return None


def _apply_slippage(price: float, bps: float, side: str) -> float:
    a = bps / 10_000.0
    return price * (1 + a) if side == "buy" else price * (1 - a)


def _bootstrap_ci(values, samples=1000, seed=42):
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    if len(x) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(samples)])
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))


def _profit_factor(pnl: pd.Series) -> float:
    p = pd.to_numeric(pnl, errors="coerce").dropna()
    wins = p[p > 0].sum(); losses = -p[p < 0].sum()
    if losses == 0:
        return np.inf if wins > 0 else np.nan
    return float(wins / losses)


def _candidate(day: pd.DataFrame, sig_idx: int, symbol: str, split: str):
    """First valid OR retest/reclaim within 10m; hypothetical entry is NEXT bar open."""
    s = day.loc[sig_idx]; sig_ts = pd.Timestamp(s["date"])
    atr = float(s["atr"]); orh = float(s["or_high"])
    if not (np.isfinite(atr) and atr > 0 and np.isfinite(orh)):
        return None
    end = sig_ts + pd.Timedelta(minutes=MAX_RETEST_MIN)
    ids = day.index[(pd.to_datetime(day["date"]) > sig_ts) & (pd.to_datetime(day["date"]) <= end)]
    reclaim_idx = None
    for j in ids:
        r = day.loc[j]
        # Same broad V14 retest zone: low reaches ORH + 0.25 ATR or closer, candle reclaims/closes >= ORH.
        if float(r["low"]) <= orh + .25 * atr and float(r["close"]) >= orh:
            rv = float(r["rvol"]) if pd.notna(r.get("rvol")) else np.nan
            vw = float(r["vwap"]) if pd.notna(r.get("vwap")) else np.nan
            if np.isfinite(rv) and RETEST_RVOL_MIN <= rv < RETEST_RVOL_MAX and np.isfinite(vw) and float(r["close"]) >= vw:
                reclaim_idx = int(j); break
    if reclaim_idx is None or reclaim_idx + 1 >= len(day):
        return None
    r = day.loc[reclaim_idx]; e = day.loc[reclaim_idx + 1]
    # Entry must still be in the configured intraday window.
    if str(e["time"]) > "14:45":
        return None
    pre = day.loc[sig_idx:reclaim_idx]
    rv = float(r["rvol"]); vw = float(r["vwap"])
    return {
        "split": split, "symbol": symbol, "session": pd.Timestamp(s["session"]),
        "signal_time": s["date"], "retest_time": r["date"], "entry_time": e["date"],
        "entry_idx": int(reclaim_idx + 1), "signal_idx": int(sig_idx), "retest_idx": int(reclaim_idx),
        "minutes_to_retest": (pd.Timestamp(r["date"]) - sig_ts).total_seconds()/60,
        "signal_rvol": float(s["rvol"]), "retest_rvol": rv, "gap_pct": float(s["gap_pct"]),
        "atr": atr, "or_high": orh, "retest_vwap_pct": (float(r["close"])/vw - 1)*100,
        "reclaim_vs_or_atr": (float(r["close"])-orh)/atr,
        "pullback_low": float(pre["low"].min()),
    }


def _rank_candidates(c: pd.DataFrame) -> pd.DataFrame:
    """Rank only candidates known at the same entry timestamp; never compare with future candidates."""
    if c.empty: return c
    x = c.copy()
    # Transparent, frozen lexicographic ranking: faster retest, RVOL nearer middle of V14's 1.5-3 band,
    # then stronger original breakout RVOL. No future return enters this ordering.
    x["rvol_mid_distance"] = (x["retest_rvol"] - 2.25).abs()
    x = x.sort_values(["entry_time","minutes_to_retest","rvol_mid_distance","signal_rvol","symbol"],
                      ascending=[True,True,True,False,True])
    x["rank_at_entry"] = x.groupby("entry_time").cumcount() + 1
    return x.sort_values(["entry_time","rank_at_entry"]).reset_index(drop=True)


def _trade(day, c, stop_mode, horizon, capital, risk_pct, slip_bps, costs):
    i = int(c["entry_idx"]); row = day.loc[i]; raw_entry = float(row["open"])
    entry = _apply_slippage(raw_entry, slip_bps, "buy"); atr = float(c["atr"])
    if stop_mode == "PULLBACK_LOW": raw_stop = float(c["pullback_low"])
    elif stop_mode == "ATR1": raw_stop = raw_entry - atr
    elif stop_mode == "ATR1_5": raw_stop = raw_entry - 1.5*atr
    else: raise ValueError(stop_mode)
    risk_unit = entry - raw_stop
    if not (np.isfinite(risk_unit) and risk_unit > 0 and raw_stop > 0): return None
    end_ts = pd.Timestamp(row["date"]) + pd.Timedelta(minutes=horizon)
    end_ids = day.index[pd.to_datetime(day["date"]) >= end_ts]
    if len(end_ids) == 0: return None
    end_idx = int(end_ids[0]); exit_idx=end_idx; raw_exit=float(day.loc[end_idx,"close"]); reason=f"TIME_{horizon}M"
    for j in range(i, end_idx+1):
        if float(day.loc[j,"low"]) <= raw_stop:
            exit_idx=j; raw_exit=raw_stop; reason="STOP"; break
    exit_price = _apply_slippage(raw_exit, slip_bps, "sell")
    risk_budget=capital*risk_pct; qty=min(math.floor(risk_budget/risk_unit), math.floor(capital/entry))
    if qty <= 0: return None
    gross=(exit_price-entry)*qty; charges=float(costs.estimate(entry,exit_price,qty)["total"]); net=gross-charges
    actual_risk=risk_unit*qty
    return {"stop_mode":stop_mode,"horizon_min":horizon,"entry":entry,"stop":raw_stop,"exit":exit_price,
            "exit_time":day.loc[exit_idx,"date"],"exit_reason":reason,"qty_1l":qty,"buy_notional_1l":entry*qty,
            "gross_pnl_1l":gross,"charges_1l":charges,"net_pnl_1l":net,
            "gross_r":gross/actual_risk if actual_risk else np.nan,"net_r":net/actual_risk if actual_risk else np.nan}


def run_retest_strategy_backtest(files: Iterable[Path], cfg: StrategyConfig, dev_start, dev_end, validation_start, validation_end,
                                  account_capital=100000.0, risk_pct=.005, slippage_bps_each_side=5.0, bootstrap_samples=1000):
    files=list(files); candidates=[]; day_cache={}; started=time.monotonic()
    for n,path in enumerate(files,1):
        symbol=path.name.split("_")[1] if "_" in path.name else path.stem
        clean=clean_market_data(pd.read_parquet(path),symbol).data
        if clean.empty: continue
        f=prepare_features(clean,cfg); f["signal"]=signal_mask(f,cfg)
        for session,d0 in f.groupby("session",sort=True):
            split=_split_name(session,dev_start,dev_end,validation_start,validation_end)
            if not split: continue
            day=d0.sort_values("date").reset_index(drop=True); ids=day.index[day["signal"]]
            if len(ids)==0: continue
            c=_candidate(day,int(ids[0]),symbol,split)
            if c:
                candidates.append(c); day_cache[(symbol,str(pd.Timestamp(session).date()))]=day
        print(f"\rV15 {n}/{len(files)} {symbol:14s} candidates={len(candidates)} elapsed={(time.monotonic()-started)/60:.1f}m",end="",flush=True)
    print()
    cand=_rank_candidates(pd.DataFrame(candidates))
    if cand.empty: return RetestStrategyResult(cand,pd.DataFrame(),pd.DataFrame(),pd.DataFrame())
    rows=[]; costs=IntradayEquityCostModel()
    for _,c in cand.iterrows():
        day=day_cache[(c.symbol,str(pd.Timestamp(c.session).date()))]
        for stop in STOP_MODES:
            for h in HORIZONS_MIN:
                t=_trade(day,c,stop,h,account_capital,risk_pct,slippage_bps_each_side,costs)
                if t:
                    base={k:c[k] for k in ["split","symbol","session","signal_time","retest_time","entry_time","minutes_to_retest","signal_rvol","retest_rvol","gap_pct","retest_vwap_pct","reclaim_vs_or_atr","rank_at_entry"]}
                    base.update(t); rows.append(base)
    trades=pd.DataFrame(rows)
    if trades.empty:return RetestStrategyResult(cand,trades,pd.DataFrame(),pd.DataFrame())
    selected=[]
    for selection in SELECTIONS:
        lim={"ALL":np.inf,"TOP3":3,"TOP2":2,"TOP1":1}[selection]
        g=trades[trades.rank_at_entry<=lim].copy(); g["selection"]=selection; selected.append(g)
    allsel=pd.concat(selected,ignore_index=True)
    sums=[]
    for (split,selection,stop,h),g in allsel.groupby(["split","selection","stop_mode","horizon_min"]):
        nr=g.net_r.dropna(); lo,hi=_bootstrap_ci(nr,bootstrap_samples,1500+int(h))
        sums.append({"split":split,"selection":selection,"stop_mode":stop,"horizon_min":int(h),"trades":len(g),
                     "win_rate":float((g.net_pnl_1l>0).mean()),"gross_expectancy_r":float(g.gross_r.mean()),
                     "net_expectancy_r":float(nr.mean()),"profit_factor_net":_profit_factor(g.net_pnl_1l),
                     "net_ci_low":lo,"net_ci_high":hi,"avg_net_pnl_1l":float(g.net_pnl_1l.mean()),
                     "median_net_pnl_1l":float(g.net_pnl_1l.median()),"avg_charges_1l":float(g.charges_1l.mean()),
                     "stop_rate":float((g.exit_reason=="STOP").mean())})
    summary=pd.DataFrame(sums).sort_values(["selection","stop_mode","horizon_min","split"])
    ss=[]
    for (split,selection),g in allsel.groupby(["split","selection"]):
        # Unique candidate count independent of stop/horizon variants.
        u=g[["symbol","entry_time"]].drop_duplicates()
        ss.append({"split":split,"selection":selection,"selected_candidates":len(u),
                   "entry_timestamps":u.entry_time.nunique(),"avg_candidates_per_entry_time":len(u)/max(1,u.entry_time.nunique())})
    return RetestStrategyResult(cand,allsel,summary,pd.DataFrame(ss))


def write_retest_strategy_reports(r: RetestStrategyResult, report_dir: Path):
    report_dir.mkdir(parents=True,exist_ok=True); paths={}
    for name,df in [("candidates",r.candidates),("trades",r.trades),("summary",r.summary),("selection_summary",r.selection_summary)]:
        p=report_dir/(name+('.parquet' if name in ('candidates','trades') else '.csv'))
        df.to_parquet(p,index=False) if p.suffix=='.parquet' else df.to_csv(p,index=False); paths[name]=p
    if r.summary.empty: return paths
    piv=r.summary.pivot_table(index=["selection","stop_mode","horizon_min"],columns="split",values="net_expectancy_r",aggfunc="first").reset_index()
    if "DEV" in piv and "VALIDATION" in piv: piv["worst_split_net_r"]=piv[["DEV","VALIDATION"]].min(axis=1)
    table=piv.sort_values("worst_split_net_r",ascending=False).to_html(index=False,float_format=lambda x:f"{x:.4f}") if "worst_split_net_r" in piv else piv.to_html(index=False)
    html=f'''<!doctype html><meta charset="utf-8"><title>V15 Retest Strategy</title><style>body{{font-family:system-ui;margin:30px;max-width:1200px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:7px;border-bottom:1px solid #ddd;font-size:12px}}</style><h1>V15 Retest Strategy + Candidate Ranking</h1><p><b>DEV + Validation only. 2026 remains locked.</b> Candidate definition is frozen from V14: retest/reclaim within 10m, above VWAP, retest RVOL 1.5–3. Entry is next-bar open. Ranking is pre-entry only and performed among candidates sharing the same entry timestamp.</p><h2>Net expectancy comparison</h2>{table}<p>₹1L figures are independent-trade illustrations, not a shared-capital portfolio. Shared-capital simulation belongs after a strategy survives this gate.</p>'''
    hp=report_dir/'retest_strategy_dashboard.html';hp.write_text(html,encoding='utf-8');paths['dashboard']=hp
    return paths

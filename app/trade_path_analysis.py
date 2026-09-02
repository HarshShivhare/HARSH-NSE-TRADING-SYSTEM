from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Iterable
import math
import time

import numpy as np
import pandas as pd

from .backtest import BacktestConfig, _apply_slippage, _choose_stop
from .costs import IntradayEquityCostModel
from .data_cleaner import clean_market_data
from .risk import position_size
from .strategy import StrategyConfig, prepare_features, signal_mask


POSITIVE_BARRIERS = (0.25, 0.50, 0.75, 1.00, 1.50, 2.00)
NEGATIVE_BARRIERS = (0.25, 0.50, 0.75, 1.00, 1.50)
HORIZONS_MIN = (5, 15, 30, 60, 90, 120)
ORDER_PAIRS = ((0.50, 0.50), (1.00, 0.50), (1.00, 1.00), (1.50, 1.00), (2.00, 1.00))


@dataclass
class TradePathResult:
    events: pd.DataFrame
    barrier_hits: pd.DataFrame
    barrier_ordering: pd.DataFrame
    time_profile: pd.DataFrame
    lifecycle: pd.DataFrame


def _split_name(session, dev_start, dev_end, validation_start, validation_end) -> str | None:
    d = pd.Timestamp(session).date()
    if pd.Timestamp(dev_start).date() <= d <= pd.Timestamp(dev_end).date():
        return "DEV"
    if pd.Timestamp(validation_start).date() <= d <= pd.Timestamp(validation_end).date():
        return "VALIDATION"
    return None


def _first_touch(day: pd.DataFrame, entry_idx: int, level: float, side: str) -> tuple[int | None, float | None]:
    entry_ts = pd.Timestamp(day.loc[entry_idx, "date"])
    for j in range(entry_idx, len(day)):
        bar = day.loc[j]
        hit = float(bar["high"]) >= level if side == "up" else float(bar["low"]) <= level
        if hit:
            minutes = (pd.Timestamp(bar["date"]) - entry_ts).total_seconds() / 60.0
            return j, minutes
    return None, None


def _at_or_after(day: pd.DataFrame, entry_idx: int, target_ts: pd.Timestamp) -> pd.Series | None:
    later = day.loc[(day.index >= entry_idx) & (pd.to_datetime(day["date"]) >= target_ts)]
    return None if later.empty else later.iloc[0]


def _qualifying_event(
    day: pd.DataFrame,
    sig_idx: int,
    symbol: str,
    split: str,
    scfg: StrategyConfig,
    bcfg: BacktestConfig,
    costs: IntradayEquityCostModel,
) -> dict | None:
    if sig_idx + 1 >= len(day):
        return None
    signal = day.loc[sig_idx]
    entry_idx = sig_idx + 1
    entry_row = day.loc[entry_idx]
    raw_entry = float(entry_row["open"])

    # Match V10 execution guards at the next-bar open.
    signal_vwap = float(signal["vwap"]) if pd.notna(signal.get("vwap")) else np.nan
    signal_atr = float(signal["atr"]) if pd.notna(signal.get("atr")) else np.nan
    signal_or_high = float(signal["or_high"]) if pd.notna(signal.get("or_high")) else np.nan
    vwap_ext = (raw_entry / signal_vwap - 1.0) * 100.0 if np.isfinite(signal_vwap) and signal_vwap > 0 else np.nan
    or_ext = (raw_entry - signal_or_high) / signal_atr if np.isfinite(signal_atr) and signal_atr > 0 and np.isfinite(signal_or_high) else np.nan
    if scfg.max_vwap_extension_pct is not None:
        if not np.isfinite(vwap_ext) or vwap_ext > scfg.max_vwap_extension_pct:
            return None
    if scfg.max_or_extension_atr is not None:
        if not np.isfinite(or_ext) or or_ext > scfg.max_or_extension_atr:
            return None

    entry = _apply_slippage(raw_entry, bcfg.slippage_bps_each_side, "buy")
    stop = _choose_stop(signal, scfg, entry)
    if not math.isfinite(stop) or stop <= 0 or stop >= entry:
        return None
    risk_per_share = entry - stop
    qty = position_size(bcfg.initial_capital, bcfg.risk_pct, entry, stop)
    if qty <= 0 or risk_per_share <= 0:
        return None

    event = {
        "split": split,
        "symbol": symbol,
        "session": pd.Timestamp(signal["session"]),
        "signal_time": signal["date"],
        "entry_time": entry_row["date"],
        "entry": entry,
        "raw_entry": raw_entry,
        "stop": stop,
        "risk_per_share": risk_per_share,
        "qty": qty,
        "gap_pct": float(signal["gap_pct"]),
        "rvol": float(signal["rvol"]),
        "atr": signal_atr,
        "entry_vs_vwap_pct": vwap_ext,
        "entry_vs_or_high_atr": or_ext,
    }

    # First touch for each favorable/adverse barrier. Same-bar ordering is handled
    # conservatively later: adverse wins when both barriers first touch on same bar.
    for r in POSITIVE_BARRIERS:
        idx, minutes = _first_touch(day, entry_idx, entry + r * risk_per_share, "up")
        event[f"up_{r:g}r_idx"] = idx
        event[f"up_{r:g}r_min"] = minutes
    for r in NEGATIVE_BARRIERS:
        idx, minutes = _first_touch(day, entry_idx, entry - r * risk_per_share, "down")
        event[f"down_{r:g}r_idx"] = idx
        event[f"down_{r:g}r_min"] = minutes

    entry_ts = pd.Timestamp(entry_row["date"])
    for minutes in HORIZONS_MIN:
        target_ts = entry_ts + pd.Timedelta(minutes=minutes)
        row = _at_or_after(day, entry_idx, target_ts)
        window = day.loc[(day.index >= entry_idx) & (pd.to_datetime(day["date"]) <= target_ts)]
        if row is None:
            event[f"close_{minutes}m_r"] = np.nan
        else:
            event[f"close_{minutes}m_r"] = (float(row["close"]) - entry) / risk_per_share
        if window.empty:
            event[f"mfe_{minutes}m_r"] = np.nan
            event[f"mae_{minutes}m_r"] = np.nan
        else:
            event[f"mfe_{minutes}m_r"] = (float(window["high"].max()) - entry) / risk_per_share
            event[f"mae_{minutes}m_r"] = (float(window["low"].min()) - entry) / risk_per_share

    eod_raw = float(day.iloc[-1]["close"])
    eod_exit = _apply_slippage(eod_raw, bcfg.slippage_bps_each_side, "sell")
    event["eod_gross_r"] = (eod_exit - entry) / risk_per_share

    initial_risk = risk_per_share * qty
    for reward, adverse in ORDER_PAIRS:
        up_idx = event.get(f"up_{reward:g}r_idx")
        down_idx = event.get(f"down_{adverse:g}r_idx")
        if up_idx is not None and not pd.isna(up_idx) and (down_idx is None or pd.isna(down_idx) or up_idx < down_idx):
            outcome = "UP_FIRST"
            exit_raw = entry + reward * risk_per_share
        elif down_idx is not None and not pd.isna(down_idx):
            outcome = "DOWN_FIRST"
            exit_raw = entry - adverse * risk_per_share
        else:
            outcome = "NEITHER_EOD"
            exit_raw = eod_raw
        exit_px = _apply_slippage(float(exit_raw), bcfg.slippage_bps_each_side, "sell")
        gross_pnl = (exit_px - entry) * qty
        charge = costs.estimate(entry, exit_px, qty)["total"]
        event[f"pair_{reward:g}_{adverse:g}_outcome"] = outcome
        event[f"pair_{reward:g}_{adverse:g}_gross_r"] = gross_pnl / initial_risk
        event[f"pair_{reward:g}_{adverse:g}_net_r"] = (gross_pnl - charge) / initial_risk
        event[f"pair_{reward:g}_{adverse:g}_cost_r"] = charge / initial_risk
    return event


def run_trade_path_analysis(
    files: Iterable[Path],
    scfg: StrategyConfig,
    bcfg: BacktestConfig,
    dev_start: str = "2023-09-01",
    dev_end: str = "2025-06-30",
    validation_start: str = "2025-07-01",
    validation_end: str = "2025-12-31",
) -> TradePathResult:
    files = list(files)
    events: list[dict] = []
    costs = IntradayEquityCostModel()
    started = time.monotonic()

    for i, path in enumerate(files, start=1):
        symbol = path.name.split("_")[1] if "_" in path.name else path.stem
        print(f"[{i}/{len(files)}] V11 trade-path: {symbol} | events={len(events)} | elapsed {(time.monotonic()-started)/60:.1f}m", flush=True)
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
            for sig_idx in candidates:
                ev = _qualifying_event(day, sig_idx, symbol, split, scfg, bcfg, costs)
                if ev is not None:
                    events.append(ev)
                    break  # one independent event per symbol/session

    e = pd.DataFrame(events)
    if e.empty:
        empty = pd.DataFrame()
        return TradePathResult(e, empty, empty, empty, empty)

    hit_rows = []
    for split, g in e.groupby("split"):
        for side, barriers in (("UP", POSITIVE_BARRIERS), ("DOWN", NEGATIVE_BARRIERS)):
            for r in barriers:
                col = f"{'up' if side == 'UP' else 'down'}_{r:g}r_min"
                hit = g[col].notna()
                hit_rows.append({
                    "split": split, "side": side, "barrier_r": r, "events": len(g),
                    "hit_count": int(hit.sum()), "hit_rate": float(hit.mean()),
                    "median_minutes_if_hit": float(g.loc[hit, col].median()) if hit.any() else np.nan,
                })
    barrier_hits = pd.DataFrame(hit_rows)

    ordering_rows = []
    for split, g in e.groupby("split"):
        for reward, adverse in ORDER_PAIRS:
            prefix = f"pair_{reward:g}_{adverse:g}"
            outcomes = g[f"{prefix}_outcome"]
            ordering_rows.append({
                "split": split, "reward_r": reward, "adverse_r": adverse, "events": len(g),
                "up_first_rate": float((outcomes == "UP_FIRST").mean()),
                "down_first_rate": float((outcomes == "DOWN_FIRST").mean()),
                "neither_eod_rate": float((outcomes == "NEITHER_EOD").mean()),
                "gross_expectancy_r": float(g[f"{prefix}_gross_r"].mean()),
                "avg_cost_r": float(g[f"{prefix}_cost_r"].mean()),
                "net_expectancy_r": float(g[f"{prefix}_net_r"].mean()),
            })
    barrier_ordering = pd.DataFrame(ordering_rows)

    time_rows = []
    for split, g in e.groupby("split"):
        for minutes in HORIZONS_MIN:
            c = pd.to_numeric(g[f"close_{minutes}m_r"], errors="coerce")
            time_rows.append({
                "split": split, "minutes": minutes, "events": int(c.notna().sum()),
                "positive_rate": float((c.dropna() > 0).mean()) if c.notna().any() else np.nan,
                "avg_close_r": float(c.mean()), "median_close_r": float(c.median()),
                "avg_mfe_r": float(pd.to_numeric(g[f"mfe_{minutes}m_r"], errors="coerce").mean()),
                "avg_mae_r": float(pd.to_numeric(g[f"mae_{minutes}m_r"], errors="coerce").mean()),
            })
    time_profile = pd.DataFrame(time_rows)

    life_rows = []
    for split, g in e.groupby("split"):
        life_rows.append({
            "split": split, "events": len(g),
            "avg_eod_gross_r": float(g["eod_gross_r"].mean()),
            "avg_mfe_60m_r": float(g["mfe_60m_r"].mean()),
            "avg_mae_60m_r": float(g["mae_60m_r"].mean()),
            "median_mfe_60m_r": float(g["mfe_60m_r"].median()),
            "median_mae_60m_r": float(g["mae_60m_r"].median()),
            "p_mfe_ge_0_5r": float((g["mfe_60m_r"] >= 0.5).mean()),
            "p_mfe_ge_1r": float((g["mfe_60m_r"] >= 1.0).mean()),
            "p_mae_le_neg_0_5r": float((g["mae_60m_r"] <= -0.5).mean()),
            "p_mae_le_neg_1r": float((g["mae_60m_r"] <= -1.0).mean()),
        })
    lifecycle = pd.DataFrame(life_rows)
    return TradePathResult(e, barrier_hits, barrier_ordering, time_profile, lifecycle)


def _svg_barrier_chart(table: pd.DataFrame) -> str:
    data = table.loc[table["side"] == "UP"].copy()
    if data.empty:
        return "<p>No barrier data.</p>"
    width, height, left, top, bottom = 820, 360, 70, 35, 55
    plot_h = height - top - bottom
    groups = sorted(data["barrier_r"].unique())
    splits = [s for s in ["DEV", "VALIDATION"] if s in set(data["split"])]
    bar_w = 22
    group_w = max(70, (width-left-30)/max(1,len(groups)))
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img">']
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" class="axis"/>')
    parts.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-20}" y2="{height-bottom}" class="axis"/>')
    for tick in [0, .25, .5, .75, 1.0]:
        y = top + plot_h*(1-tick)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-20}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end">{tick:.0%}</text>')
    for gi, r in enumerate(groups):
        cx = left + group_w*(gi+.5)
        parts.append(f'<text x="{cx:.1f}" y="{height-25}" text-anchor="middle">+{r:g}R</text>')
        for si, split in enumerate(splits):
            row = data[(data["barrier_r"]==r)&(data["split"]==split)]
            if row.empty: continue
            rate = float(row.iloc[0]["hit_rate"])
            h = plot_h*rate
            x = cx - (len(splits)*bar_w)/2 + si*bar_w
            cls = "bar-dev" if split=="DEV" else "bar-val"
            parts.append(f'<rect x="{x:.1f}" y="{top+plot_h-h:.1f}" width="{bar_w-3}" height="{h:.1f}" class="{cls}"><title>{split} +{r:g}R: {rate:.1%}</title></rect>')
    parts.append('</svg>')
    return ''.join(parts)


def _svg_time_chart(table: pd.DataFrame) -> str:
    if table.empty: return "<p>No time-profile data.</p>"
    width, height, left, top, bottom = 820, 360, 70, 35, 55
    xs = sorted(table["minutes"].unique())
    vals = pd.to_numeric(table["avg_close_r"], errors="coerce").dropna()
    ymax = max(0.25, float(vals.abs().max())*1.25 if len(vals) else 0.25)
    ymin = -ymax
    plot_w, plot_h = width-left-30, height-top-bottom
    xmap = {m: left + plot_w*i/max(1,len(xs)-1) for i,m in enumerate(xs)}
    def ymap(v): return top + (ymax-v)/(ymax-ymin)*plot_h
    parts=[f'<svg viewBox="0 0 {width} {height}">']
    zero=ymap(0)
    parts.append(f'<line x1="{left}" y1="{zero:.1f}" x2="{width-20}" y2="{zero:.1f}" class="axis"/>')
    for m in xs:
        parts.append(f'<text x="{xmap[m]:.1f}" y="{height-25}" text-anchor="middle">{m}m</text>')
    for split, cls in [("DEV","line-dev"),("VALIDATION","line-val")]:
        g=table[table["split"]==split].sort_values("minutes")
        pts=' '.join(f'{xmap[int(r.minutes)]:.1f},{ymap(float(r.avg_close_r)):.1f}' for r in g.itertuples() if pd.notna(r.avg_close_r))
        if pts: parts.append(f'<polyline points="{pts}" class="{cls}"/>')
        for r in g.itertuples():
            if pd.notna(r.avg_close_r):
                parts.append(f'<circle cx="{xmap[int(r.minutes)]:.1f}" cy="{ymap(float(r.avg_close_r)):.1f}" r="4" class="dot"><title>{split} {r.minutes}m: {r.avg_close_r:+.3f}R</title></circle>')
    parts.append('</svg>')
    return ''.join(parts)


def _svg_order_chart(table: pd.DataFrame) -> str:
    if table.empty: return "<p>No barrier-order data.</p>"
    width,height,left,top,bottom=820,360,95,35,70
    labels=[f"+{r:g}R / -{a:g}R" for r,a in ORDER_PAIRS]
    splits=[s for s in ["DEV","VALIDATION"] if s in set(table["split"])]
    plot_h=height-top-bottom; group_w=(width-left-25)/len(labels); bar_w=22
    parts=[f'<svg viewBox="0 0 {width} {height}">']
    for tick in [0,.25,.5,.75,1]:
        y=top+plot_h*(1-tick)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-20}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end">{tick:.0%}</text>')
    for i,(reward,adverse) in enumerate(ORDER_PAIRS):
        cx=left+group_w*(i+.5)
        parts.append(f'<text x="{cx:.1f}" y="{height-38}" text-anchor="middle">{labels[i]}</text>')
        for si,split in enumerate(splits):
            row=table[(table.reward_r==reward)&(table.adverse_r==adverse)&(table.split==split)]
            if row.empty: continue
            v=float(row.iloc[0].up_first_rate); h=plot_h*v; x=cx-(len(splits)*bar_w)/2+si*bar_w
            cls="bar-dev" if split=="DEV" else "bar-val"
            parts.append(f'<rect x="{x:.1f}" y="{top+plot_h-h:.1f}" width="{bar_w-3}" height="{h:.1f}" class="{cls}"><title>{split}: {v:.1%} favorable first</title></rect>')
    parts.append('</svg>')
    return ''.join(parts)


def _svg_expectancy_chart(table: pd.DataFrame) -> str:
    if table.empty: return "<p>No expectancy data.</p>"
    width,height,left,top,bottom=820,380,100,40,80
    values=pd.concat([table["gross_expectancy_r"],table["net_expectancy_r"]]).dropna()
    extent=max(.25,float(values.abs().max())*1.35 if len(values) else .25)
    ymin,ymax=-extent,extent; plot_h=height-top-bottom; group_w=(width-left-25)/len(ORDER_PAIRS); bar_w=18
    def ymap(v): return top+(ymax-v)/(ymax-ymin)*plot_h
    zero=ymap(0); parts=[f'<svg viewBox="0 0 {width} {height}">',f'<line x1="{left}" y1="{zero:.1f}" x2="{width-20}" y2="{zero:.1f}" class="axis"/>']
    for i,(reward,adverse) in enumerate(ORDER_PAIRS):
        cx=left+group_w*(i+.5); parts.append(f'<text x="{cx:.1f}" y="{height-42}" text-anchor="middle">+{reward:g}/-{adverse:g}R</text>')
        # Four bars: DEV gross/net, VALIDATION gross/net.
        rows=[]
        for split in ["DEV","VALIDATION"]:
            r=table[(table.reward_r==reward)&(table.adverse_r==adverse)&(table.split==split)]
            if not r.empty: rows.append((split,"gross",float(r.iloc[0].gross_expectancy_r))); rows.append((split,"net",float(r.iloc[0].net_expectancy_r)))
        total=len(rows)
        for bi,(split,kind,v) in enumerate(rows):
            x=cx-total*bar_w/2+bi*bar_w; y=ymap(max(v,0)); h=abs(ymap(v)-zero); y=min(y,zero)
            cls=("bar-dev" if split=="DEV" else "bar-val") + (" gross" if kind=="gross" else " net")
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w-3}" height="{max(1,h):.1f}" class="{cls}"><title>{split} {kind}: {v:+.3f}R</title></rect>')
    parts.append('</svg>'); return ''.join(parts)


def write_trade_path_reports(result: TradePathResult, report_dir: Path, scfg: StrategyConfig) -> dict[str, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    tables = {
        "barrier_hits": result.barrier_hits,
        "barrier_ordering": result.barrier_ordering,
        "time_profile": result.time_profile,
        "lifecycle": result.lifecycle,
    }
    for name, table in tables.items():
        path=report_dir/f"{name}.csv"; table.to_csv(path,index=False); outputs[name]=path
    events_path=report_dir/"trade_path_events.parquet"; result.events.to_parquet(events_path,index=False); outputs["events"]=events_path

    dashboard=report_dir/"trade_path_dashboard.html"
    config_text=(f"Trend={'ON' if scfg.require_trend else 'OFF'} | Gap ≥ {scfg.gap_min_pct:g}% | RVOL ≥ {scfg.rvol_min:g} | "
                 f"OR={scfg.opening_range_minutes}m | stop={scfg.stop_mode}/{scfg.atr_multiple:g} ATR | "
                 f"VWAP cap={scfg.max_vwap_extension_pct if scfg.max_vwap_extension_pct is not None else 'none'} | "
                 f"OR-extension cap={scfg.max_or_extension_atr if scfg.max_or_extension_atr is not None else 'none'} ATR")
    css='''body{font-family:system-ui,-apple-system,sans-serif;margin:28px;background:#f7f7f8;color:#171717}.wrap{max-width:1080px;margin:auto}.card{background:white;border:1px solid #ddd;border-radius:14px;padding:20px;margin:16px 0;box-shadow:0 1px 4px #0001}h1,h2{margin:.2em 0 .5em}.muted{color:#666}.axis{stroke:#555;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}.bar-dev{fill:#496f9b}.bar-val{fill:#bd6b4d}.gross{opacity:.45}.net{opacity:1}.line-dev{fill:none;stroke:#496f9b;stroke-width:3}.line-val{fill:none;stroke:#bd6b4d;stroke-width:3}.dot{fill:#333}svg text{font-size:12px;fill:#555}table{border-collapse:collapse;width:100%;font-size:13px}th,td{border-bottom:1px solid #eee;padding:7px;text-align:right}th:first-child,td:first-child{text-align:left}.legend span{display:inline-block;margin-right:18px}.sw{width:12px;height:12px;display:inline-block;margin-right:5px}.dev{background:#496f9b}.val{background:#bd6b4d}'''
    html=f'''<!doctype html><html><head><meta charset="utf-8"><title>V11 Trade Path Dashboard</title><style>{css}</style></head><body><div class="wrap">
<h1>V11 Trade Path + Barrier Analysis</h1><p class="muted">{escape(config_text)}</p><p class="legend"><span><i class="sw dev"></i>DEV</span><span><i class="sw val"></i>VALIDATION</span></p>
<div class="card"><h2>Favorable barrier hit probability</h2><p>How often price touches each positive R barrier at any point after entry.</p>{_svg_barrier_chart(result.barrier_hits)}</div>
<div class="card"><h2>Barrier ordering</h2><p>Probability the favorable barrier is reached before the adverse barrier. Same-bar ambiguity is counted adverse-first.</p>{_svg_order_chart(result.barrier_ordering)}</div>
<div class="card"><h2>Average path after entry</h2><p>Average close return in R at fixed horizons. This shows whether continuation appears quickly and whether it persists.</p>{_svg_time_chart(result.time_profile)}</div>
<div class="card"><h2>Gross vs net expectancy by barrier pair</h2><p>Actual slippage and the configured Indian intraday cost model are included in net R. Faded bars are gross; solid bars are net.</p>{_svg_expectancy_chart(result.barrier_ordering)}</div>
<div class="card"><h2>Lifecycle summary</h2>{result.lifecycle.to_html(index=False,float_format=lambda x:f'{x:.4f}')}</div>
<div class="card"><h2>Barrier ordering table</h2>{result.barrier_ordering.to_html(index=False,float_format=lambda x:f'{x:.4f}')}</div>
<p class="muted">Research diagnostic only. 2026 FINAL OOS is intentionally not read by this command.</p></div></body></html>'''
    dashboard.write_text(html,encoding="utf-8"); outputs["dashboard"]=dashboard
    return outputs

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Iterable
import time

import numpy as np
import pandas as pd

from .data_cleaner import clean_market_data
from .strategy import StrategyConfig, prepare_features, signal_mask


DELAYS_MIN = (5, 15, 30, 60)
FORWARD_HORIZONS_MIN = (15, 30, 60, 120)
DIST_BINS = (-np.inf, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, np.inf)
DIST_LABELS = ("<-2R", "-2..-1R", "-1..-0.5R", "-0.5..0R", "0..0.5R", "0.5..1R", "1..2R", ">2R")


@dataclass
class PostBreakoutResult:
    events: pd.DataFrame
    delay_summary: pd.DataFrame
    condition_summary: pd.DataFrame
    distribution_summary: pd.DataFrame
    concentration_summary: pd.DataFrame
    histogram: pd.DataFrame


def _split_name(session, dev_start, dev_end, validation_start, validation_end) -> str | None:
    d = pd.Timestamp(session).date()
    if pd.Timestamp(dev_start).date() <= d <= pd.Timestamp(dev_end).date():
        return "DEV"
    if pd.Timestamp(validation_start).date() <= d <= pd.Timestamp(validation_end).date():
        return "VALIDATION"
    return None


def _first_at_or_after(day: pd.DataFrame, ts: pd.Timestamp) -> int | None:
    mask = pd.to_datetime(day["date"]) >= ts
    idx = day.index[mask]
    return None if len(idx) == 0 else int(idx[0])


def _trimmed_mean(values: pd.Series, trim: float = 0.10) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna().sort_values().to_numpy()
    if len(x) == 0:
        return np.nan
    k = int(len(x) * trim)
    if 2 * k >= len(x):
        return float(np.mean(x))
    return float(np.mean(x[k:len(x)-k]))


def _positive_concentration(values: pd.Series, top_fraction: float) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna()
    pos = x[x > 0].sort_values(ascending=False)
    total = float(pos.sum())
    if total <= 0 or pos.empty:
        return np.nan
    n = max(1, int(np.ceil(len(pos) * top_fraction)))
    return float(pos.head(n).sum() / total)


def _bootstrap_mean_ci(values: pd.Series, samples: int = 1000, seed: int = 42) -> tuple[float, float]:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(x) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=float)
    for i in range(samples):
        means[i] = rng.choice(x, size=len(x), replace=True).mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _event_for_delay(day: pd.DataFrame, sig_idx: int, delay_min: int, symbol: str, split: str, atr_multiple: float) -> dict | None:
    signal = day.loc[sig_idx]
    signal_ts = pd.Timestamp(signal["date"])
    entry_idx = _first_at_or_after(day, signal_ts + pd.Timedelta(minutes=delay_min))
    if entry_idx is None or entry_idx <= sig_idx:
        return None

    entry_row = day.loc[entry_idx]
    entry = float(entry_row["open"])
    atr = float(entry_row["atr"]) if pd.notna(entry_row.get("atr")) else np.nan
    if not np.isfinite(atr) or atr <= 0 or not np.isfinite(entry) or entry <= 0:
        return None
    risk_unit = atr_multiple * atr
    if risk_unit <= 0:
        return None

    pre = day.loc[sig_idx + 1:entry_idx].copy()
    if pre.empty:
        return None
    or_high = float(signal["or_high"]) if pd.notna(signal.get("or_high")) else np.nan
    signal_high = float(signal["high"])
    pre_low = float(pre["low"].min())
    pre_high = float(pre["high"].max())
    pre_close_min = float(pre["close"].min())
    entry_vwap = float(entry_row["vwap"]) if pd.notna(entry_row.get("vwap")) else np.nan
    below_vwap = bool(((pre["low"] <= pre["vwap"]) & pre["vwap"].notna()).any())
    close_below_or = bool((pre["close"] < or_high).any()) if np.isfinite(or_high) else False
    recovered_or = bool(float(entry_row["close"]) >= or_high) if np.isfinite(or_high) else False
    touched_or_zone = bool(pre_low <= or_high + 0.25 * atr) if np.isfinite(or_high) else False
    stayed_above_or = bool(pre_close_min >= or_high) if np.isfinite(or_high) else False

    ev = {
        "split": split,
        "symbol": symbol,
        "session": pd.Timestamp(signal["session"]),
        "signal_time": signal["date"],
        "delay_min": delay_min,
        "entry_time": entry_row["date"],
        "entry": entry,
        "atr": atr,
        "risk_unit": risk_unit,
        "gap_pct": float(signal["gap_pct"]),
        "rvol": float(signal["rvol"]),
        "entry_vs_vwap_pct": (entry / entry_vwap - 1.0) * 100.0 if np.isfinite(entry_vwap) and entry_vwap > 0 else np.nan,
        "entry_vs_or_high_atr": (entry - or_high) / atr if np.isfinite(or_high) else np.nan,
        "stayed_above_or_high": stayed_above_or,
        "pulled_back_to_or_high": touched_or_zone and not close_below_or,
        "went_below_vwap": below_vwap,
        "made_new_high_before_entry": pre_high > signal_high,
        "failed_breakout_recovered": close_below_or and recovered_or,
    }

    entry_ts = pd.Timestamp(entry_row["date"])
    for horizon in FORWARD_HORIZONS_MIN:
        target_idx = _first_at_or_after(day, entry_ts + pd.Timedelta(minutes=horizon))
        if target_idx is None:
            ev[f"ret_{horizon}m_r"] = np.nan
            ev[f"mfe_{horizon}m_r"] = np.nan
            ev[f"mae_{horizon}m_r"] = np.nan
            continue
        row = day.loc[target_idx]
        window = day.loc[entry_idx:target_idx]
        ev[f"ret_{horizon}m_r"] = (float(row["close"]) - entry) / risk_unit
        ev[f"mfe_{horizon}m_r"] = (float(window["high"].max()) - entry) / risk_unit
        ev[f"mae_{horizon}m_r"] = (float(window["low"].min()) - entry) / risk_unit
    return ev


def run_post_breakout_analysis(
    files: Iterable[Path],
    scfg: StrategyConfig,
    dev_start: str = "2023-09-01",
    dev_end: str = "2025-06-30",
    validation_start: str = "2025-07-01",
    validation_end: str = "2025-12-31",
    bootstrap_samples: int = 1000,
) -> PostBreakoutResult:
    files = list(files)
    events: list[dict] = []
    started = time.monotonic()

    for i, path in enumerate(files, start=1):
        symbol = path.name.split("_")[1] if "_" in path.name else path.stem
        print(f"[{i}/{len(files)}] V12 post-breakout: {symbol} | events={len(events)} | elapsed {(time.monotonic()-started)/60:.1f}m", flush=True)
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
            sig_idx = int(candidates[0])  # one independent base signal per symbol/session
            for delay in DELAYS_MIN:
                ev = _event_for_delay(day, sig_idx, delay, symbol, split, scfg.atr_multiple)
                if ev is not None:
                    events.append(ev)

    e = pd.DataFrame(events)
    if e.empty:
        empty = pd.DataFrame()
        return PostBreakoutResult(e, empty, empty, empty, empty, empty)

    delay_rows = []
    distribution_rows = []
    concentration_rows = []
    histogram_rows = []
    for (split, delay), g in e.groupby(["split", "delay_min"]):
        for horizon in FORWARD_HORIZONS_MIN:
            r = pd.to_numeric(g[f"ret_{horizon}m_r"], errors="coerce").dropna()
            lo, hi = _bootstrap_mean_ci(r, samples=bootstrap_samples, seed=42 + delay + horizon)
            delay_rows.append({
                "split": split, "delay_min": delay, "horizon_min": horizon, "events": len(r),
                "positive_rate": float((r > 0).mean()) if len(r) else np.nan,
                "avg_r": float(r.mean()) if len(r) else np.nan,
                "median_r": float(r.median()) if len(r) else np.nan,
                "trimmed_mean_r": _trimmed_mean(r),
                "ci_low": lo, "ci_high": hi,
                "avg_mfe_r": float(pd.to_numeric(g[f"mfe_{horizon}m_r"], errors="coerce").mean()),
                "avg_mae_r": float(pd.to_numeric(g[f"mae_{horizon}m_r"], errors="coerce").mean()),
            })

        r120 = pd.to_numeric(g["ret_120m_r"], errors="coerce").dropna()
        distribution_rows.append({
            "split": split, "delay_min": delay, "events": len(r120),
            "mean_r": float(r120.mean()), "median_r": float(r120.median()), "trimmed_mean_r": _trimmed_mean(r120),
            "p10": float(r120.quantile(.10)), "p25": float(r120.quantile(.25)), "p75": float(r120.quantile(.75)),
            "p90": float(r120.quantile(.90)), "p95": float(r120.quantile(.95)),
        })
        concentration_rows.append({
            "split": split, "delay_min": delay, "events": len(r120),
            "top_1pct_positive_share": _positive_concentration(r120, .01),
            "top_5pct_positive_share": _positive_concentration(r120, .05),
            "top_10pct_positive_share": _positive_concentration(r120, .10),
        })
        if len(r120):
            buckets = pd.cut(r120, bins=DIST_BINS, labels=DIST_LABELS, right=False)
            counts = buckets.value_counts(sort=False)
            for label in DIST_LABELS:
                count = int(counts.get(label, 0))
                histogram_rows.append({
                    "split": split, "delay_min": delay, "bucket": label,
                    "count": count, "rate": count / len(r120),
                })

    condition_cols = [
        "stayed_above_or_high", "pulled_back_to_or_high", "went_below_vwap",
        "made_new_high_before_entry", "failed_breakout_recovered",
    ]
    condition_rows = []
    for (split, delay), g in e.groupby(["split", "delay_min"]):
        for condition in condition_cols:
            for value in (True, False):
                s = g[g[condition] == value]
                r = pd.to_numeric(s["ret_120m_r"], errors="coerce").dropna()
                if len(r) == 0:
                    continue
                condition_rows.append({
                    "split": split, "delay_min": delay, "condition": condition,
                    "condition_value": value, "events": len(r),
                    "positive_rate_120m": float((r > 0).mean()),
                    "avg_120m_r": float(r.mean()), "median_120m_r": float(r.median()),
                    "trimmed_mean_120m_r": _trimmed_mean(r),
                })

    return PostBreakoutResult(
        e,
        pd.DataFrame(delay_rows),
        pd.DataFrame(condition_rows),
        pd.DataFrame(distribution_rows),
        pd.DataFrame(concentration_rows),
        pd.DataFrame(histogram_rows),
    )


def _svg_line_delay(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "<p>No data.</p>"
    width, height, left, top, bottom = 900, 390, 75, 35, 60
    plot_w, plot_h = width-left-30, height-top-bottom
    delays = sorted(summary["delay_min"].unique())
    horizons = sorted(summary["horizon_min"].unique())
    vals = pd.to_numeric(summary["avg_r"], errors="coerce").dropna()
    extent = max(.25, float(vals.abs().max()) * 1.3 if len(vals) else .25)
    ymin, ymax = -extent, extent
    xmap = {d: left + plot_w*i/max(1, len(delays)-1) for i, d in enumerate(delays)}
    def ymap(v): return top + (ymax-v)/(ymax-ymin)*plot_h
    parts = [f'<svg viewBox="0 0 {width} {height}">']
    parts.append(f'<line x1="{left}" y1="{ymap(0):.1f}" x2="{width-20}" y2="{ymap(0):.1f}" class="axis"/>')
    for d in delays:
        parts.append(f'<text x="{xmap[d]:.1f}" y="{height-25}" text-anchor="middle">+{d}m</text>')
    classes = ["line-a", "line-b", "line-c", "line-d"]
    for hi, horizon in enumerate(horizons):
        g = summary[(summary["split"] == "VALIDATION") & (summary["horizon_min"] == horizon)].sort_values("delay_min")
        pts = ' '.join(f'{xmap[int(r.delay_min)]:.1f},{ymap(float(r.avg_r)):.1f}' for r in g.itertuples() if pd.notna(r.avg_r))
        if pts:
            parts.append(f'<polyline points="{pts}" class="{classes[hi % len(classes)]}"><title>Validation +{horizon}m horizon</title></polyline>')
    parts.append('</svg>')
    return ''.join(parts)


def _svg_dev_val_120(summary: pd.DataFrame) -> str:
    data = summary[summary["horizon_min"] == 120].copy()
    if data.empty:
        return "<p>No data.</p>"
    width, height, left, top, bottom = 900, 390, 75, 35, 60
    delays = sorted(data["delay_min"].unique())
    vals = pd.to_numeric(data["avg_r"], errors="coerce").dropna()
    extent = max(.25, float(vals.abs().max())*1.35 if len(vals) else .25)
    ymin, ymax = -extent, extent; plot_h = height-top-bottom
    group_w = (width-left-30)/max(1,len(delays)); bar_w=32
    def ymap(v): return top+(ymax-v)/(ymax-ymin)*plot_h
    zero=ymap(0); parts=[f'<svg viewBox="0 0 {width} {height}">', f'<line x1="{left}" y1="{zero:.1f}" x2="{width-20}" y2="{zero:.1f}" class="axis"/>']
    for i, delay in enumerate(delays):
        cx=left+group_w*(i+.5); parts.append(f'<text x="{cx:.1f}" y="{height-25}" text-anchor="middle">Entry +{delay}m</text>')
        for si, split in enumerate(["DEV","VALIDATION"]):
            row=data[(data.delay_min==delay)&(data.split==split)]
            if row.empty: continue
            v=float(row.iloc[0].avg_r); x=cx-38+si*bar_w; y=min(ymap(v),zero); h=abs(ymap(v)-zero)
            cls="bar-dev" if split=="DEV" else "bar-val"
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w-5}" height="{max(1,h):.1f}" class="{cls}"><title>{split}: {v:+.3f}R</title></rect>')
    parts.append('</svg>'); return ''.join(parts)


def _svg_histogram(hist: pd.DataFrame, delay: int = 30) -> str:
    data=hist[hist.delay_min==delay].copy()
    if data.empty: return "<p>No histogram data.</p>"
    width,height,left,top,bottom=900,390,80,35,75; plot_h=height-top-bottom
    labels=list(DIST_LABELS); group_w=(width-left-25)/len(labels); bar_w=25
    ymax=max(.10,float(data.rate.max())*1.2)
    parts=[f'<svg viewBox="0 0 {width} {height}">']
    for i,label in enumerate(labels):
        cx=left+group_w*(i+.5); parts.append(f'<text x="{cx:.1f}" y="{height-30}" text-anchor="middle">{escape(label)}</text>')
        for si,split in enumerate(["DEV","VALIDATION"]):
            row=data[(data.bucket==label)&(data.split==split)]
            if row.empty: continue
            v=float(row.iloc[0].rate); h=plot_h*v/ymax; x=cx-28+si*bar_w; cls="bar-dev" if split=="DEV" else "bar-val"
            parts.append(f'<rect x="{x:.1f}" y="{top+plot_h-h:.1f}" width="{bar_w-4}" height="{h:.1f}" class="{cls}"><title>{split} {label}: {v:.1%}</title></rect>')
    parts.append('</svg>'); return ''.join(parts)


def write_post_breakout_reports(result: PostBreakoutResult, report_dir: Path, scfg: StrategyConfig) -> dict[str, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    tables = {
        "delay_summary": result.delay_summary,
        "condition_summary": result.condition_summary,
        "distribution_summary": result.distribution_summary,
        "concentration_summary": result.concentration_summary,
        "histogram": result.histogram,
    }
    for name, table in tables.items():
        path = report_dir / f"{name}.csv"; table.to_csv(path, index=False); outputs[name] = path
    events_path = report_dir / "post_breakout_events.parquet"; result.events.to_parquet(events_path, index=False); outputs["events"] = events_path

    dashboard = report_dir / "post_breakout_dashboard.html"
    config_text = (f"Trend={'ON' if scfg.require_trend else 'OFF'} | Gap ≥ {scfg.gap_min_pct:g}% | RVOL ≥ {scfg.rvol_min:g} | "
                   f"OR={scfg.opening_range_minutes}m | risk unit={scfg.atr_multiple:g} ATR | delayed entries={','.join(map(str,DELAYS_MIN))}m")
    css='''body{font-family:system-ui,-apple-system,sans-serif;margin:28px;background:#f7f7f8;color:#171717}.wrap{max-width:1100px;margin:auto}.card{background:white;border:1px solid #ddd;border-radius:14px;padding:20px;margin:16px 0;box-shadow:0 1px 4px #0001}.muted{color:#666}.axis{stroke:#666;stroke-width:1}.bar-dev{fill:#496f9b}.bar-val{fill:#bd6b4d}.line-a{fill:none;stroke:#496f9b;stroke-width:3}.line-b{fill:none;stroke:#bd6b4d;stroke-width:3}.line-c{fill:none;stroke:#648f62;stroke-width:3}.line-d{fill:none;stroke:#8b6ba8;stroke-width:3}svg text{font-size:12px;fill:#555}table{border-collapse:collapse;width:100%;font-size:12px}th,td{border-bottom:1px solid #eee;padding:7px;text-align:right}th:first-child,td:first-child{text-align:left}.legend span{display:inline-block;margin-right:18px}.sw{width:12px;height:12px;display:inline-block;margin-right:5px}.dev{background:#496f9b}.val{background:#bd6b4d}'''
    html=f'''<!doctype html><html><head><meta charset="utf-8"><title>V12 Post-Breakout Dashboard</title><style>{css}</style></head><body><div class="wrap">
<h1>V12 Delayed Entry + Retest Research</h1><p class="muted">{escape(config_text)}</p><p class="legend"><span><i class="sw dev"></i>DEV</span><span><i class="sw val"></i>VALIDATION</span></p>
<div class="card"><h2>120-minute average R by delayed entry</h2><p>Direct DEV vs Validation comparison. A real effect should improve broadly, not in one isolated delay.</p>{_svg_dev_val_120(result.delay_summary)}</div>
<div class="card"><h2>Validation forward-return profile</h2><p>Average R after each delayed entry, shown across 15/30/60/120-minute horizons.</p>{_svg_line_delay(result.delay_summary)}</div>
<div class="card"><h2>120-minute R distribution at +30m entry</h2><p>Shape matters more than the mean: this reveals whether a few large winners are carrying the average.</p>{_svg_histogram(result.histogram,30)}</div>
<div class="card"><h2>Distribution summary</h2>{result.distribution_summary.to_html(index=False,float_format=lambda x:f'{x:.4f}')}</div>
<div class="card"><h2>Positive-return concentration</h2>{result.concentration_summary.to_html(index=False,float_format=lambda x:f'{x:.4f}')}</div>
<div class="card"><h2>Pre-entry condition analysis</h2>{result.condition_summary.to_html(index=False,float_format=lambda x:f'{x:.4f}')}</div>
<p class="muted">Diagnostic research only. 2026 FINAL OOS is intentionally not read by V12.</p></div></body></html>'''
    dashboard.write_text(html, encoding="utf-8"); outputs["dashboard"] = dashboard
    return outputs

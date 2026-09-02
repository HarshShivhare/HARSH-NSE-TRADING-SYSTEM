from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import time

import numpy as np
import pandas as pd

from .data_cleaner import clean_market_data
from .strategy import StrategyConfig, prepare_features

# V16 is discovery-only. These broad checkpoints / cohorts are frozen and are not optimisation knobs.
SNAPSHOT_TIMES = ("09:45", "10:00", "10:30", "11:00", "12:00")
FORWARD_MINUTES = (15, 30, 60, 120)
COHORTS = (("TOP5", .95), ("TOP10", .90), ("TOP20", .80))


@dataclass
class CrossSectionalMomentumResult:
    events: pd.DataFrame
    cohort_summary: pd.DataFrame
    feature_summary: pd.DataFrame
    persistence_summary: pd.DataFrame


def _split_name(session, dev_start, dev_end, validation_start, validation_end):
    d = pd.Timestamp(session).date()
    if pd.Timestamp(dev_start).date() <= d <= pd.Timestamp(dev_end).date():
        return "DEV"
    if pd.Timestamp(validation_start).date() <= d <= pd.Timestamp(validation_end).date():
        return "VALIDATION"
    return None


def _future_close(day: pd.DataFrame, i: int, minutes: int):
    target = pd.Timestamp(day.loc[i, "date"]) + pd.Timedelta(minutes=minutes)
    ids = day.index[pd.to_datetime(day["date"]) >= target]
    if len(ids) == 0:
        return np.nan
    return float(day.loc[int(ids[0]), "close"])


def _symbol_snapshots(path: Path, cfg: StrategyConfig, dev_start, dev_end, validation_start, validation_end):
    symbol = path.name.split("_")[1] if "_" in path.name else path.stem
    clean = clean_market_data(pd.read_parquet(path), symbol).data
    if clean.empty:
        return []
    f = prepare_features(clean, cfg)
    rows = []
    for session, d0 in f.groupby("session", sort=True):
        split = _split_name(session, dev_start, dev_end, validation_start, validation_end)
        if not split:
            continue
        day = d0.sort_values("date").reset_index(drop=True)
        if day.empty:
            continue
        day_open = float(day.loc[0, "open"])
        if not np.isfinite(day_open) or day_open <= 0:
            continue
        for clock in SNAPSHOT_TIMES:
            ids = day.index[day["time"] == clock]
            if len(ids) == 0:
                continue
            i = int(ids[0]); r = day.loc[i]
            close = float(r["close"]); atr = float(r["atr"]) if pd.notna(r["atr"]) else np.nan
            vwap = float(r["vwap"]) if pd.notna(r["vwap"]) else np.nan
            out = {
                "split": split, "symbol": symbol, "session": pd.Timestamp(session), "snapshot_time": clock,
                "timestamp": r["date"], "close": close,
                "day_return_pct": (close / day_open - 1.0) * 100.0,
                "gap_pct": float(r["gap_pct"]) if pd.notna(r["gap_pct"]) else np.nan,
                "rvol": float(r["rvol"]) if pd.notna(r["rvol"]) else np.nan,
                "vwap_pct": (close / vwap - 1.0) * 100.0 if np.isfinite(vwap) and vwap > 0 else np.nan,
                "atr": atr,
                "above_vwap": bool(np.isfinite(vwap) and close > vwap),
            }
            for h in FORWARD_MINUTES:
                fc = _future_close(day, i, h)
                out[f"fwd_{h}m_pct"] = (fc / close - 1.0) * 100.0 if np.isfinite(fc) else np.nan
                out[f"fwd_{h}m_atr"] = (fc - close) / atr if np.isfinite(fc) and np.isfinite(atr) and atr > 0 else np.nan
            rows.append(out)
    return rows


def _bootstrap_ci(values, samples=1000, seed=42):
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    if len(x) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(samples)])
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))


def run_cross_sectional_momentum(files: Iterable[Path], cfg: StrategyConfig, dev_start, dev_end,
                                 validation_start, validation_end, bootstrap_samples=1000):
    files = list(files); rows = []; started = time.monotonic()
    for n, path in enumerate(files, 1):
        got = _symbol_snapshots(path, cfg, dev_start, dev_end, validation_start, validation_end)
        rows.extend(got)
        symbol = path.name.split("_")[1] if "_" in path.name else path.stem
        print(f"\rV16 {n}/{len(files)} {symbol:14s} snapshots={len(rows)} elapsed={(time.monotonic()-started)/60:.1f}m", end="", flush=True)
    print()
    e = pd.DataFrame(rows)
    if e.empty:
        return CrossSectionalMomentumResult(e, pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    # Cross-sectional market proxy: same-timestamp median of the available universe.
    grp = e.groupby(["split", "timestamp"], sort=False)
    e["market_median_return_pct"] = grp["day_return_pct"].transform("median")
    e["excess_vs_median_pct"] = e["day_return_pct"] - e["market_median_return_pct"]
    e["momentum_percentile"] = grp["day_return_pct"].rank(method="average", pct=True)
    e["universe_count"] = grp["symbol"].transform("count")

    # Cohorts are pure cross-sectional strength first. Descriptive confirmations are reported separately below.
    sums = []
    for (split, clock), g in e.groupby(["split", "snapshot_time"]):
        for name, q in COHORTS:
            c = g[g["momentum_percentile"] >= q]
            for h in FORWARD_MINUTES:
                vals = c[f"fwd_{h}m_atr"].dropna(); lo, hi = _bootstrap_ci(vals, bootstrap_samples, 1600+h)
                sums.append({"split": split, "snapshot_time": clock, "cohort": name, "events": len(vals),
                             "avg_excess_vs_median_pct": float(c["excess_vs_median_pct"].mean()),
                             "avg_rvol": float(c["rvol"].mean()), "above_vwap_rate": float(c["above_vwap"].mean()),
                             "horizon_min": h, "avg_fwd_atr": float(vals.mean()) if len(vals) else np.nan,
                             "median_fwd_atr": float(vals.median()) if len(vals) else np.nan,
                             "positive_rate": float((vals > 0).mean()) if len(vals) else np.nan,
                             "ci_low": lo, "ci_high": hi})
    cohort = pd.DataFrame(sums)

    # Broad, pre-observable confirmations on TOP10 only; not a combinatorial grid.
    fs = []
    top = e[e["momentum_percentile"] >= .90].copy()
    variants = {
        "TOP10_ALL": pd.Series(True, index=top.index),
        "TOP10_ABOVE_VWAP": top["above_vwap"],
        "TOP10_RVOL_GE_1_5": top["rvol"] >= 1.5,
        "TOP10_ABOVE_VWAP_RVOL_GE_1_5": top["above_vwap"] & (top["rvol"] >= 1.5),
    }
    for label, mask in variants.items():
        gg = top[mask]
        for (split, clock), g in gg.groupby(["split", "snapshot_time"]):
            for h in (60, 120):
                v = g[f"fwd_{h}m_atr"].dropna(); lo, hi = _bootstrap_ci(v, bootstrap_samples, 1700+h)
                fs.append({"variant": label, "split": split, "snapshot_time": clock, "horizon_min": h,
                           "events": len(v), "avg_fwd_atr": float(v.mean()) if len(v) else np.nan,
                           "median_fwd_atr": float(v.median()) if len(v) else np.nan,
                           "positive_rate": float((v > 0).mean()) if len(v) else np.nan, "ci_low": lo, "ci_high": hi})
    feature = pd.DataFrame(fs)

    # Persistence: if a stock is TOP10 now, is it still TOP10 at the next research checkpoint?
    ps = []
    order = list(SNAPSHOT_TIMES)
    for split, g in e.groupby("split"):
        for a, b in zip(order[:-1], order[1:]):
            ga = g[(g.snapshot_time == a) & (g.momentum_percentile >= .90)][["session", "symbol"]].drop_duplicates()
            gb = g[(g.snapshot_time == b) & (g.momentum_percentile >= .90)][["session", "symbol"]].drop_duplicates()
            if ga.empty:
                continue
            merged = ga.merge(gb.assign(still_top10=True), on=["session", "symbol"], how="left")
            ps.append({"split": split, "from_time": a, "to_time": b, "top10_events": len(ga),
                       "still_top10_rate": float(merged.still_top10.astype("boolean").fillna(False).mean())})
    persistence = pd.DataFrame(ps)
    return CrossSectionalMomentumResult(e, cohort, feature, persistence)


def write_cross_sectional_momentum_reports(r: CrossSectionalMomentumResult, report_dir: Path):
    report_dir.mkdir(parents=True, exist_ok=True); paths = {}
    for name, df in [("events", r.events), ("cohort_summary", r.cohort_summary),
                     ("feature_summary", r.feature_summary), ("persistence_summary", r.persistence_summary)]:
        p = report_dir / (name + (".parquet" if name == "events" else ".csv"))
        df.to_parquet(p, index=False) if p.suffix == ".parquet" else df.to_csv(p, index=False)
        paths[name] = p
    if r.cohort_summary.empty:
        return paths
    focus = r.cohort_summary[(r.cohort_summary.cohort == "TOP10") & (r.cohort_summary.horizon_min.isin([60,120]))]
    table = focus.to_html(index=False, float_format=lambda x: f"{x:.4f}")
    ftable = r.feature_summary.to_html(index=False, float_format=lambda x: f"{x:.4f}") if not r.feature_summary.empty else ""
    ptable = r.persistence_summary.to_html(index=False, float_format=lambda x: f"{x:.4f}") if not r.persistence_summary.empty else ""
    html = f'''<!doctype html><meta charset="utf-8"><title>V16 Cross-sectional Momentum Discovery</title>
<style>body{{font-family:system-ui;margin:30px;max-width:1400px}}table{{border-collapse:collapse;width:100%;margin-bottom:28px}}td,th{{padding:7px;border-bottom:1px solid #ddd;font-size:12px}}.warn{{padding:12px;background:#fff3cd}}</style>
<h1>V16 Cross-sectional Relative Strength + Abnormal Momentum Discovery</h1>
<p class="warn"><b>DEV + Validation only. 2026 remains locked.</b> This is discovery, not a trading backtest. Relative strength is measured against the same-timestamp cross-sectional universe median because NIFTY index history is not yet part of the dataset. Current NIFTY100 membership used historically still has survivorship bias.</p>
<h2>TOP10 momentum cohort — forward continuation</h2>{table}
<h2>Broad confirmation study</h2>{ftable}
<h2>Momentum persistence</h2>{ptable}
<p>No capital allocation, stops, targets, brokerage or slippage are applied in V16. If a stable cross-sectional effect survives DEV and Validation, the next version should convert only that frozen effect into a tradable construction.</p>'''
    hp = report_dir / "cross_sectional_momentum_dashboard.html"; hp.write_text(html, encoding="utf-8"); paths["dashboard"] = hp
    return paths

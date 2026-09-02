from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .cross_sectional_momentum import (
    FORWARD_MINUTES,
    SNAPSHOT_TIMES,
    run_cross_sectional_momentum,
)
from .strategy import StrategyConfig

# V17 is discovery-only. These definitions are frozen before seeing V17 results.
# A leader is TOP10 at the current checkpoint. PERSIST_2/PERSIST_3 require that
# the same stock was TOP10 at the immediately preceding 1/2 checkpoints too.
PERSISTENCE_LEVELS = (1, 2, 3)
FOCUS_HORIZONS = (30, 60, 120)


@dataclass
class PersistentLeaderResult:
    events: pd.DataFrame
    leader_summary: pd.DataFrame
    spread_summary: pd.DataFrame
    fade_summary: pd.DataFrame
    transition_summary: pd.DataFrame


def _bootstrap_mean_ci(values, samples=1000, seed=42):
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    if len(x) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(samples)])
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))


def _session_spread_table(g: pd.DataFrame, leader_mask: pd.Series, benchmark_mask: pd.Series, value_col: str):
    x = g[["session", value_col]].copy()
    x["leader"] = leader_mask.reindex(g.index).fillna(False).to_numpy(bool)
    x["benchmark"] = benchmark_mask.reindex(g.index).fillna(False).to_numpy(bool)
    rows = []
    for session, d in x.groupby("session", sort=False):
        a = d.loc[d["leader"], value_col].dropna()
        b = d.loc[d["benchmark"], value_col].dropna()
        if len(a) and len(b):
            rows.append({"session": session, "leader_mean": float(a.mean()), "benchmark_mean": float(b.mean()),
                         "spread": float(a.mean() - b.mean())})
    return pd.DataFrame(rows)


def _add_persistence(events: pd.DataFrame) -> pd.DataFrame:
    e = events.copy()
    e["is_top10"] = e["momentum_percentile"] >= .90
    order = {clock: i for i, clock in enumerate(SNAPSHOT_TIMES)}
    e["snapshot_order"] = e["snapshot_time"].map(order)
    e = e.sort_values(["split", "session", "symbol", "snapshot_order"]).reset_index(drop=True)
    e["top10_streak"] = 0

    for _, idx in e.groupby(["split", "session", "symbol"], sort=False).groups.items():
        streak = 0
        last_order = None
        for i in idx:
            current_order = int(e.at[i, "snapshot_order"])
            consecutive = last_order is not None and current_order == last_order + 1
            if bool(e.at[i, "is_top10"]):
                streak = streak + 1 if consecutive else 1
            else:
                streak = 0
            e.at[i, "top10_streak"] = streak
            last_order = current_order

    # Fading leader = TOP10 at immediately previous checkpoint but not TOP10 now.
    e["was_top10_prev"] = False
    for _, idx in e.groupby(["split", "session", "symbol"], sort=False).groups.items():
        prev_top = False
        prev_order = None
        for i in idx:
            current_order = int(e.at[i, "snapshot_order"])
            consecutive = prev_order is not None and current_order == prev_order + 1
            e.at[i, "was_top10_prev"] = bool(prev_top and consecutive)
            prev_top = bool(e.at[i, "is_top10"])
            prev_order = current_order
    e["is_fading_leader"] = e["was_top10_prev"] & ~e["is_top10"]
    return e


def _leader_rows(e: pd.DataFrame, bootstrap_samples: int):
    rows = []
    for (split, clock), g in e.groupby(["split", "snapshot_time"], sort=True):
        for streak in PERSISTENCE_LEVELS:
            mask = g["top10_streak"] >= streak
            # One broad confirmation carried forward from V16; not a parameter grid.
            variants = {
                f"PERSIST_{streak}": mask,
                f"PERSIST_{streak}_CONFIRMED": mask & g["above_vwap"] & (g["rvol"] >= 1.5),
            }
            for label, m in variants.items():
                c = g[m]
                if c.empty:
                    continue
                for h in FOCUS_HORIZONS:
                    v = c[f"fwd_{h}m_atr"].dropna()
                    if v.empty:
                        continue
                    lo, hi = _bootstrap_mean_ci(v, bootstrap_samples, seed=17100 + h + streak)
                    rows.append({
                        "split": split, "snapshot_time": clock, "cohort": label, "events": len(v),
                        "sessions": int(c.loc[v.index, "session"].nunique()),
                        "avg_excess_vs_median_pct": float(c.loc[v.index, "excess_vs_median_pct"].mean()),
                        "avg_rvol": float(c.loc[v.index, "rvol"].mean()),
                        "above_vwap_rate": float(c.loc[v.index, "above_vwap"].mean()),
                        "horizon_min": h, "avg_fwd_atr": float(v.mean()), "median_fwd_atr": float(v.median()),
                        "positive_rate": float((v > 0).mean()), "ci_low": lo, "ci_high": hi,
                    })
    return pd.DataFrame(rows)


def _spread_rows(e: pd.DataFrame, bootstrap_samples: int):
    rows = []
    for (split, clock), g in e.groupby(["split", "snapshot_time"], sort=True):
        bottom50 = g["momentum_percentile"] <= .50
        for streak in PERSISTENCE_LEVELS:
            leader = g["top10_streak"] >= streak
            variants = {
                f"PERSIST_{streak}": leader,
                f"PERSIST_{streak}_CONFIRMED": leader & g["above_vwap"] & (g["rvol"] >= 1.5),
            }
            for label, lmask in variants.items():
                if not lmask.any():
                    continue
                rest = ~lmask
                for h in FOCUS_HORIZONS:
                    col = f"fwd_{h}m_atr"
                    for bench_name, bmask in (("REST", rest), ("BOTTOM50", bottom50)):
                        d = _session_spread_table(g, lmask, bmask, col)
                        if d.empty:
                            continue
                        lo, hi = _bootstrap_mean_ci(d["spread"], bootstrap_samples,
                                                    seed=17200 + h + streak + (0 if bench_name == "REST" else 50))
                        rows.append({
                            "split": split, "snapshot_time": clock, "cohort": label,
                            "benchmark": bench_name, "horizon_min": h,
                            "leader_events": int(lmask.sum()), "paired_sessions": len(d),
                            "leader_session_avg_atr": float(d["leader_mean"].mean()),
                            "benchmark_session_avg_atr": float(d["benchmark_mean"].mean()),
                            "spread_atr": float(d["spread"].mean()),
                            "spread_ci_low": lo, "spread_ci_high": hi,
                            "positive_spread_session_rate": float((d["spread"] > 0).mean()),
                        })
    return pd.DataFrame(rows)


def _fade_rows(e: pd.DataFrame, bootstrap_samples: int):
    rows = []
    f = e[e["is_fading_leader"]].copy()
    for (split, clock), g in f.groupby(["split", "snapshot_time"], sort=True):
        for h in FOCUS_HORIZONS:
            v = g[f"fwd_{h}m_atr"].dropna()
            if v.empty:
                continue
            lo, hi = _bootstrap_mean_ci(v, bootstrap_samples, seed=17300 + h)
            rows.append({"split": split, "snapshot_time": clock, "events": len(v), "horizon_min": h,
                         "avg_fwd_atr": float(v.mean()), "median_fwd_atr": float(v.median()),
                         "positive_rate": float((v > 0).mean()), "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def _transition_rows(e: pd.DataFrame):
    rows = []
    for split, g in e.groupby("split", sort=True):
        for clock in SNAPSHOT_TIMES:
            c = g[g["snapshot_time"] == clock]
            if c.empty:
                continue
            rows.append({
                "split": split, "snapshot_time": clock, "events": len(c),
                "top10_events": int(c["is_top10"].sum()),
                "persist2_events": int((c["top10_streak"] >= 2).sum()),
                "persist3_events": int((c["top10_streak"] >= 3).sum()),
                "fading_events": int(c["is_fading_leader"].sum()),
            })
    return pd.DataFrame(rows)


def run_persistent_leader_discovery(files: Iterable[Path], cfg: StrategyConfig, dev_start, dev_end,
                                    validation_start, validation_end, bootstrap_samples=1000):
    # Reuse V16's frozen snapshot construction so V17 changes the question, not the underlying data treatment.
    base = run_cross_sectional_momentum(
        files, cfg, dev_start, dev_end, validation_start, validation_end, bootstrap_samples
    )
    if base.events.empty:
        empty = pd.DataFrame()
        return PersistentLeaderResult(empty, empty, empty, empty, empty)
    e = _add_persistence(base.events)
    return PersistentLeaderResult(
        events=e,
        leader_summary=_leader_rows(e, bootstrap_samples),
        spread_summary=_spread_rows(e, bootstrap_samples),
        fade_summary=_fade_rows(e, bootstrap_samples),
        transition_summary=_transition_rows(e),
    )


def write_persistent_leader_reports(r: PersistentLeaderResult, report_dir: Path):
    report_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    frames = [
        ("events", r.events),
        ("leader_summary", r.leader_summary),
        ("spread_summary", r.spread_summary),
        ("fade_summary", r.fade_summary),
        ("transition_summary", r.transition_summary),
    ]
    for name, df in frames:
        p = report_dir / (name + (".parquet" if name == "events" else ".csv"))
        if p.suffix == ".parquet":
            df.to_parquet(p, index=False)
        else:
            df.to_csv(p, index=False)
        paths[name] = p

    def html_table(df):
        return df.to_html(index=False, float_format=lambda x: f"{x:.4f}") if not df.empty else "<p>No rows.</p>"

    focus_leaders = r.leader_summary[
        r.leader_summary["cohort"].isin(["PERSIST_2", "PERSIST_2_CONFIRMED", "PERSIST_3", "PERSIST_3_CONFIRMED"])
        & r.leader_summary["horizon_min"].isin([60, 120])
    ] if not r.leader_summary.empty else r.leader_summary
    focus_spreads = r.spread_summary[
        r.spread_summary["cohort"].isin(["PERSIST_2", "PERSIST_2_CONFIRMED", "PERSIST_3", "PERSIST_3_CONFIRMED"])
        & (r.spread_summary["benchmark"] == "BOTTOM50")
        & r.spread_summary["horizon_min"].isin([60, 120])
    ] if not r.spread_summary.empty else r.spread_summary

    dashboard = report_dir / "persistent_leader_dashboard.html"
    dashboard.write_text(f'''<!doctype html><meta charset="utf-8"><title>V17 Persistent Leader Discovery</title>
<style>body{{font-family:system-ui;margin:30px;max-width:1500px}}table{{border-collapse:collapse;width:100%;margin:14px 0 30px}}td,th{{padding:7px;border-bottom:1px solid #ddd;font-size:12px;text-align:right}}th:first-child,td:first-child{{text-align:left}}.warn{{padding:12px;background:#fff3cd}}code{{background:#f5f5f5;padding:2px 4px}}</style>
<h1>V17 Persistent Leader + Cross-sectional Spread Discovery</h1>
<p class="warn"><b>DEV + Validation only. 2026 remains locked.</b> Discovery only; no trading/capital allocation. V17 deliberately reuses V16 snapshots and the same-timestamp universe-median market proxy. Current NIFTY100 membership used historically still has survivorship bias.</p>
<p><b>Frozen definitions:</b> TOP10 = current cross-sectional top decile. PERSIST_2/PERSIST_3 = TOP10 for 2/3 consecutive research checkpoints. CONFIRMED adds above VWAP + RVOL ≥ 1.5. Spread confidence intervals are bootstrapped from <b>session-level</b> leader-minus-benchmark means to reduce false precision from treating stocks on the same day as independent.</p>
<h2>Transitions / sample sizes</h2>{html_table(r.transition_summary)}
<h2>Persistent-leader continuation — focus</h2>{html_table(focus_leaders)}
<h2>Persistent leader vs Bottom 50 — session-clustered spread</h2>{html_table(focus_spreads)}
<h2>Fading leaders</h2>{html_table(r.fade_summary)}
<h2>All leader cohorts</h2>{html_table(r.leader_summary)}
<h2>All cross-sectional spreads</h2>{html_table(r.spread_summary)}
<p><b>Interpretation rule:</b> promotion requires a broad effect that repeats in DEV and Validation, preferably with positive session-clustered spread versus BOTTOM50. Do not promote a single timestamp/horizon merely because it is the best-looking cell.</p>''', encoding="utf-8")
    paths["dashboard"] = dashboard
    return paths

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import time

import numpy as np
import pandas as pd

from .data_cleaner import clean_market_data
from .opportunity_ranking_discovery import run_opportunity_ranking_discovery
from .strategy import StrategyConfig, prepare_features

# V20 intentionally keeps a small, pre-declared matrix. The purpose is to test whether
# entry structure improves a causal opportunity rank, not to optimise dozens of knobs.
SNAPSHOTS = ("10:30", "11:00")
SCORE_FAMILIES = ("MOMENTUM_ONLY", "FULL_4")
TOP_K = (1, 3)
ENTRY_VARIANTS = ("IMMEDIATE", "VWAP_RECLAIM", "PULLBACK_NEW_HIGH", "CONSOLIDATION_BREAKOUT")
HORIZONS = (30, 60, 120)
ENTRY_WINDOW_MIN = 30


@dataclass
class OpportunityEntryResult:
    candidates: pd.DataFrame
    entries: pd.DataFrame
    entry_summary: pd.DataFrame
    paired_improvement: pd.DataFrame


def _bootstrap_ci(values, samples=1000, seed=42):
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    if len(x) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(samples)])
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))


def _next_bar(day: pd.DataFrame, idx: int):
    j = idx + 1
    return j if j < len(day) else None


def _window_indices(day: pd.DataFrame, snapshot_idx: int, minutes: int = ENTRY_WINDOW_MIN):
    ts = pd.Timestamp(day.loc[snapshot_idx, "date"])
    end = ts + pd.Timedelta(minutes=minutes)
    return [int(i) for i in day.index[(day.index > snapshot_idx) & (pd.to_datetime(day["date"]) <= end)]]


def _entry_index(day: pd.DataFrame, snapshot_idx: int, variant: str, snapshot_atr: float):
    if variant == "IMMEDIATE":
        return _next_bar(day, snapshot_idx)

    scan = _window_indices(day, snapshot_idx)
    if not scan or not np.isfinite(snapshot_atr) or snapshot_atr <= 0:
        return None

    snapshot_close = float(day.loc[snapshot_idx, "close"])
    snapshot_high = float(day.loc[snapshot_idx, "high"])

    if variant == "VWAP_RECLAIM":
        for i in scan:
            r = day.loc[i]
            vwap = float(r["vwap"]) if pd.notna(r["vwap"]) else np.nan
            if np.isfinite(vwap) and float(r["low"]) <= vwap and float(r["close"]) >= vwap:
                return _next_bar(day, i)
        return None

    if variant == "PULLBACK_NEW_HIGH":
        pullback_seen = False
        for i in scan:
            r = day.loc[i]
            low = float(r["low"])
            vwap = float(r["vwap"]) if pd.notna(r["vwap"]) else np.nan
            if not pullback_seen:
                depth = snapshot_close - low
                if .25 * snapshot_atr <= depth <= 1.0 * snapshot_atr:
                    pullback_seen = True
                continue
            if float(r["close"]) > snapshot_high and np.isfinite(vwap) and float(r["close"]) >= vwap:
                return _next_bar(day, i)
        return None

    if variant == "CONSOLIDATION_BREAKOUT":
        # First three completed bars after the snapshot define a fixed 15-minute box.
        box_ids = scan[:3]
        if len(box_ids) < 3:
            return None
        box = day.loc[box_ids]
        box_high = float(box["high"].max())
        box_low = float(box["low"].min())
        if box_high - box_low > snapshot_atr:
            return None
        for i in scan[3:]:
            r = day.loc[i]
            vwap = float(r["vwap"]) if pd.notna(r["vwap"]) else np.nan
            if float(r["close"]) > box_high and np.isfinite(vwap) and float(r["close"]) >= vwap:
                return _next_bar(day, i)
        return None

    raise ValueError(f"Unknown entry variant: {variant}")


def _path_metrics(day: pd.DataFrame, entry_idx: int, horizon_min: int, entry_price: float, atr: float):
    entry_ts = pd.Timestamp(day.loc[entry_idx, "date"])
    target = entry_ts + pd.Timedelta(minutes=horizon_min)
    after = day[(day.index >= entry_idx) & (pd.to_datetime(day["date"]) <= target)]
    if after.empty or not np.isfinite(atr) or atr <= 0:
        return np.nan, np.nan, np.nan, np.nan
    target_rows = day.index[pd.to_datetime(day["date"]) >= target]
    if len(target_rows) == 0:
        return np.nan, np.nan, np.nan, np.nan
    exit_close = float(day.loc[int(target_rows[0]), "close"])
    fwd_atr = (exit_close - entry_price) / atr
    fwd_pct = (exit_close / entry_price - 1.0) * 100.0
    mfe = (float(after["high"].max()) - entry_price) / atr
    mae = (float(after["low"].min()) - entry_price) / atr
    return fwd_atr, fwd_pct, mfe, mae


def _load_feature_days(files: Iterable[Path], cfg: StrategyConfig, wanted: set[tuple[str, str]]):
    days = {}
    files = list(files); started = time.monotonic()
    wanted_symbols = {s for s, _ in wanted}
    for n, path in enumerate(files, 1):
        symbol = path.name.split("_")[1] if "_" in path.name else path.stem
        if symbol not in wanted_symbols:
            continue
        clean = clean_market_data(pd.read_parquet(path), symbol).data
        if clean.empty:
            continue
        f = prepare_features(clean, cfg)
        for session, d0 in f.groupby("session", sort=False):
            key = (symbol, str(pd.Timestamp(session).date()))
            if key in wanted:
                days[key] = d0.sort_values("date").reset_index(drop=True)
        print(f"\rV20 feature paths {n}/{len(files)} {symbol:14s} cached_days={len(days)} elapsed={(time.monotonic()-started)/60:.1f}m", end="", flush=True)
    print()
    return days


def _candidate_table(events: pd.DataFrame):
    rows = []
    base = events[events["snapshot_time"].isin(SNAPSHOTS)].copy()
    for family in SCORE_FAMILIES:
        rank_col = f"rank_{family}"
        score_col = f"score_{family}"
        for k in TOP_K:
            picked = base[base[rank_col] <= k]
            for _, r in picked.iterrows():
                rows.append({
                    "split": r["split"], "symbol": r["symbol"], "session": r["session"],
                    "snapshot_time": r["snapshot_time"], "snapshot_timestamp": r["timestamp"],
                    "score_family": family, "top_k": k, "rank": float(r[rank_col]),
                    "score": float(r[score_col]), "snapshot_close": float(r["close"]),
                    "snapshot_atr": float(r["atr"]), "snapshot_rvol": float(r["rvol"]),
                    "snapshot_vwap_pct": float(r["vwap_pct"]), "snapshot_gap_pct": float(r["gap_pct"]),
                    "snapshot_day_return_pct": float(r["day_return_pct"]),
                })
    return pd.DataFrame(rows)


def _build_entries(candidates: pd.DataFrame, days: dict):
    rows = []
    for _, c in candidates.iterrows():
        key = (c["symbol"], str(pd.Timestamp(c["session"]).date()))
        day = days.get(key)
        if day is None or day.empty:
            continue
        ids = day.index[day["time"] == c["snapshot_time"]]
        if len(ids) == 0:
            continue
        si = int(ids[0]); satr = float(c["snapshot_atr"])
        for variant in ENTRY_VARIANTS:
            ei = _entry_index(day, si, variant, satr)
            out = c.to_dict(); out["entry_variant"] = variant
            out["triggered"] = ei is not None
            if ei is None or ei >= len(day):
                out.update({"entry_time": pd.NaT, "entry_price": np.nan, "entry_atr": np.nan, "entry_delay_min": np.nan})
                for h in HORIZONS:
                    out.update({f"fwd_{h}m_atr": np.nan, f"fwd_{h}m_pct": np.nan, f"mfe_{h}m_atr": np.nan, f"mae_{h}m_atr": np.nan})
                rows.append(out); continue
            er = day.loc[ei]
            entry_price = float(er["open"])
            eatr = float(er["atr"]) if pd.notna(er["atr"]) else satr
            out.update({
                "entry_time": er["date"], "entry_price": entry_price, "entry_atr": eatr,
                "entry_delay_min": (pd.Timestamp(er["date"]) - pd.Timestamp(c["snapshot_timestamp"])).total_seconds() / 60.0,
            })
            for h in HORIZONS:
                fr, fp, mfe, mae = _path_metrics(day, ei, h, entry_price, eatr)
                out.update({f"fwd_{h}m_atr": fr, f"fwd_{h}m_pct": fp, f"mfe_{h}m_atr": mfe, f"mae_{h}m_atr": mae})
            rows.append(out)
    return pd.DataFrame(rows)


def _summary(entries: pd.DataFrame, bootstrap_samples: int):
    rows = []
    keys = ["split", "score_family", "top_k", "snapshot_time", "entry_variant"]
    for key, g in entries.groupby(keys, sort=True):
        split, family, k, clock, variant = key
        base_candidates = len(g)
        triggered = g[g["triggered"]]
        for h in HORIZONS:
            col = f"fwd_{h}m_atr"
            v = triggered[col].dropna()
            sess = triggered.groupby("session")[col].mean().dropna()
            lo, hi = _bootstrap_ci(sess, bootstrap_samples, 20000 + h + int(k))
            rows.append({
                "split": split, "score_family": family, "top_k": int(k), "snapshot_time": clock,
                "entry_variant": variant, "horizon_min": h, "candidates": base_candidates,
                "triggered": int(triggered["triggered"].sum()), "trigger_rate": float(triggered.shape[0] / base_candidates) if base_candidates else np.nan,
                "events": len(v), "sessions": int(sess.size), "avg_fwd_atr": float(v.mean()) if len(v) else np.nan,
                "median_fwd_atr": float(v.median()) if len(v) else np.nan, "positive_rate": float((v > 0).mean()) if len(v) else np.nan,
                "avg_mfe_atr": float(triggered[f"mfe_{h}m_atr"].mean()) if len(triggered) else np.nan,
                "avg_mae_atr": float(triggered[f"mae_{h}m_atr"].mean()) if len(triggered) else np.nan,
                "avg_entry_delay_min": float(triggered["entry_delay_min"].mean()) if len(triggered) else np.nan,
                "session_avg_fwd_atr": float(sess.mean()) if len(sess) else np.nan,
                "session_ci_low": lo, "session_ci_high": hi,
            })
    return pd.DataFrame(rows)


def _paired_improvement(entries: pd.DataFrame, bootstrap_samples: int):
    rows = []
    id_cols = ["split", "score_family", "top_k", "snapshot_time", "symbol", "session"]
    immediate = entries[entries.entry_variant == "IMMEDIATE"]
    for variant in ("VWAP_RECLAIM", "PULLBACK_NEW_HIGH", "CONSOLIDATION_BREAKOUT"):
        alt = entries[(entries.entry_variant == variant) & entries.triggered]
        paired = alt.merge(immediate, on=id_cols, suffixes=("_alt", "_imm"))
        for (split, family, k, clock), g in paired.groupby(["split", "score_family", "top_k", "snapshot_time"], sort=True):
            for h in HORIZONS:
                a = f"fwd_{h}m_atr_alt"; b = f"fwd_{h}m_atr_imm"
                d = g[["session", a, b]].dropna().copy()
                if d.empty:
                    continue
                d["delta"] = d[a] - d[b]
                session_delta = d.groupby("session")["delta"].mean()
                lo, hi = _bootstrap_ci(session_delta, bootstrap_samples, 20100 + h + int(k))
                rows.append({
                    "split": split, "score_family": family, "top_k": int(k), "snapshot_time": clock,
                    "entry_variant": variant, "horizon_min": h, "paired_events": len(d), "paired_sessions": int(session_delta.size),
                    "alt_avg_fwd_atr": float(d[a].mean()), "immediate_avg_fwd_atr": float(d[b].mean()),
                    "improvement_atr": float(d["delta"].mean()), "session_improvement_atr": float(session_delta.mean()),
                    "improvement_ci_low": lo, "improvement_ci_high": hi,
                    "improvement_positive_session_rate": float((session_delta > 0).mean()),
                })
    return pd.DataFrame(rows)


def run_opportunity_entry_discovery(files: Iterable[Path], cfg: StrategyConfig, dev_start, dev_end,
                                    validation_start, validation_end, bootstrap_samples=1000):
    files = list(files)
    rank = run_opportunity_ranking_discovery(
        files, cfg, dev_start, dev_end, validation_start, validation_end, bootstrap_samples
    )
    if rank.events.empty:
        z = pd.DataFrame(); return OpportunityEntryResult(z, z, z, z)
    candidates = _candidate_table(rank.events)
    wanted = {(r.symbol, str(pd.Timestamp(r.session).date())) for r in candidates.itertuples()}
    days = _load_feature_days(files, cfg, wanted)
    entries = _build_entries(candidates, days)
    return OpportunityEntryResult(
        candidates=candidates,
        entries=entries,
        entry_summary=_summary(entries, bootstrap_samples),
        paired_improvement=_paired_improvement(entries, bootstrap_samples),
    )


def write_opportunity_entry_reports(r: OpportunityEntryResult, report_dir: Path):
    report_dir.mkdir(parents=True, exist_ok=True); paths = {}
    for name, df in [("candidates", r.candidates), ("entries", r.entries),
                     ("entry_summary", r.entry_summary), ("paired_improvement", r.paired_improvement)]:
        p = report_dir / (name + (".parquet" if name in {"candidates", "entries"} else ".csv"))
        df.to_parquet(p, index=False) if p.suffix == ".parquet" else df.to_csv(p, index=False)
        paths[name] = p
    summary = r.entry_summary[r.entry_summary.horizon_min.isin([60, 120])] if not r.entry_summary.empty else r.entry_summary
    paired = r.paired_improvement[r.paired_improvement.horizon_min.isin([60, 120])] if not r.paired_improvement.empty else r.paired_improvement
    html = f'''<!doctype html><meta charset="utf-8"><title>V20 Opportunity Rank + Entry Discovery</title>
<style>body{{font-family:system-ui;margin:30px;max-width:1500px}}table{{border-collapse:collapse;width:100%;margin-bottom:28px}}td,th{{padding:7px;border-bottom:1px solid #ddd;font-size:12px}}.warn{{padding:12px;background:#fff3cd}}</style>
<h1>V20 Opportunity Rank + Entry Discovery</h1>
<p class="warn"><b>DEV + Validation only. 2026 remains locked.</b> Discovery only: no capital, brokerage, slippage, stop or target. Rankings are causal at 10:30/11:00; entry patterns use only bars available after the rank timestamp. Current NIFTY100 historical membership remains survivorship-biased.</p>
<h2>Entry Structure Forward Returns</h2>{summary.to_html(index=False, float_format=lambda x: f"{x:.4f}") if not summary.empty else '<p>No data</p>'}
<h2>Matched Improvement vs Immediate Entry</h2>{paired.to_html(index=False, float_format=lambda x: f"{x:.4f}") if not paired.empty else '<p>No data</p>'}
<p>Frozen score families: MOMENTUM_ONLY and FULL_4. Frozen selection sizes: Top1 and Top3. Frozen snapshots: 10:30 and 11:00. Entry variants: next-bar immediate; VWAP touch/reclaim; 0.25–1.0 ATR pullback followed by a close above snapshot high; and a 15-minute consolidation no wider than 1 ATR followed by breakout.</p>'''
    hp = report_dir / "opportunity_entry_dashboard.html"; hp.write_text(html, encoding="utf-8"); paths["dashboard"] = hp
    return paths

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .cross_sectional_momentum import run_cross_sectional_momentum, SNAPSHOT_TIMES
from .strategy import StrategyConfig

# V19 is deliberately a small, frozen discovery study. These are broad rank families,
# not an optimisation grid. All inputs are known at the snapshot timestamp.
SCORE_FAMILIES = {
    "MOMENTUM_ONLY": ("momentum_rank",),
    "MOMENTUM_VOLUME": ("momentum_rank", "rvol_rank"),
    "QUALITY_3": ("momentum_rank", "rvol_rank", "vwap_rank"),
    "FULL_4": ("momentum_rank", "rvol_rank", "vwap_rank", "gap_rank"),
}
TOP_K = (1, 3, 5)
HORIZONS = (30, 60, 120)


@dataclass
class OpportunityRankingResult:
    events: pd.DataFrame
    rank_summary: pd.DataFrame
    winner_capture: pd.DataFrame
    score_spread: pd.DataFrame


def _bootstrap_ci(values, samples=1000, seed=42):
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    if len(x) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(samples)])
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))


def _add_scores(events: pd.DataFrame) -> pd.DataFrame:
    e = events.copy()
    keys = ["split", "timestamp"]
    grp = e.groupby(keys, sort=False)

    # Cross-sectional percentile ranks; higher always means stronger/more active.
    e["momentum_rank"] = grp["day_return_pct"].rank(method="average", pct=True)
    e["rvol_rank"] = grp["rvol"].rank(method="average", pct=True)
    e["vwap_rank"] = grp["vwap_pct"].rank(method="average", pct=True)
    e["gap_rank"] = grp["gap_pct"].rank(method="average", pct=True)

    for name, cols in SCORE_FAMILIES.items():
        e[f"score_{name}"] = e[list(cols)].mean(axis=1, skipna=False)
        e[f"rank_{name}"] = e.groupby(keys, sort=False)[f"score_{name}"].rank(method="first", ascending=False)
    return e


def _session_selected(g: pd.DataFrame, score_family: str, k: int) -> pd.DataFrame:
    return g[g[f"rank_{score_family}"] <= k]


def _rank_summary(e: pd.DataFrame, bootstrap_samples: int) -> pd.DataFrame:
    rows = []
    for (split, clock), g in e.groupby(["split", "snapshot_time"], sort=True):
        for family in SCORE_FAMILIES:
            for k in TOP_K:
                selected = _session_selected(g, family, k)
                for h in HORIZONS:
                    col = f"fwd_{h}m_atr"
                    v = selected[col].dropna()
                    # Cluster by session: each day contributes one selected-basket mean.
                    session_means = selected.groupby("session")[col].mean().dropna()
                    lo, hi = _bootstrap_ci(session_means, bootstrap_samples, 19000 + h + k)
                    rows.append({
                        "split": split, "snapshot_time": clock, "score_family": family, "top_k": k,
                        "horizon_min": h, "events": len(v), "sessions": int(session_means.size),
                        "avg_fwd_atr": float(v.mean()) if len(v) else np.nan,
                        "median_fwd_atr": float(v.median()) if len(v) else np.nan,
                        "positive_rate": float((v > 0).mean()) if len(v) else np.nan,
                        "session_avg_fwd_atr": float(session_means.mean()) if len(session_means) else np.nan,
                        "session_ci_low": lo, "session_ci_high": hi,
                    })
    return pd.DataFrame(rows)


def _winner_capture(e: pd.DataFrame) -> pd.DataFrame:
    """Non-causal labels are used only to score the causal ranking, never as inputs."""
    rows = []
    for (split, clock), g in e.groupby(["split", "snapshot_time"], sort=True):
        for h in (60, 120):
            col = f"fwd_{h}m_atr"
            valid = g[g[col].notna()].copy()
            if valid.empty:
                continue
            # Future winner = top 5% future ATR return within the same timestamp/session universe.
            valid["future_pct"] = valid.groupby("timestamp")[col].rank(method="average", pct=True)
            valid["future_winner"] = valid["future_pct"] >= .95
            winner_count = int(valid["future_winner"].sum())
            for family in SCORE_FAMILIES:
                for k in TOP_K:
                    picked = valid[valid[f"rank_{family}"] <= k]
                    captured = int(picked["future_winner"].sum())
                    rows.append({
                        "split": split, "snapshot_time": clock, "score_family": family, "top_k": k,
                        "horizon_min": h, "future_winners": winner_count, "selected_events": len(picked),
                        "winners_captured": captured,
                        "winner_capture_rate": captured / winner_count if winner_count else np.nan,
                        "precision": captured / len(picked) if len(picked) else np.nan,
                    })
    return pd.DataFrame(rows)


def _score_spread(e: pd.DataFrame, bootstrap_samples: int) -> pd.DataFrame:
    rows = []
    for (split, clock), g in e.groupby(["split", "snapshot_time"], sort=True):
        for family in SCORE_FAMILIES:
            for k in TOP_K:
                leaders = g[g[f"rank_{family}"] <= k]
                # Same-day bottom half of the same causal score is the benchmark.
                bottom = g[g[f"score_{family}"] <= g.groupby("timestamp")[f"score_{family}"].transform("median")]
                for h in (60, 120):
                    col = f"fwd_{h}m_atr"
                    a = leaders.groupby("session")[col].mean().rename("leader")
                    b = bottom.groupby("session")[col].mean().rename("bottom")
                    d = pd.concat([a, b], axis=1).dropna()
                    if d.empty:
                        continue
                    spread = d["leader"] - d["bottom"]
                    lo, hi = _bootstrap_ci(spread, bootstrap_samples, 19100 + h + k)
                    rows.append({
                        "split": split, "snapshot_time": clock, "score_family": family, "top_k": k,
                        "horizon_min": h, "paired_sessions": len(d),
                        "leader_session_avg_atr": float(d["leader"].mean()),
                        "bottom_session_avg_atr": float(d["bottom"].mean()),
                        "spread_atr": float(spread.mean()), "spread_ci_low": lo, "spread_ci_high": hi,
                        "positive_spread_session_rate": float((spread > 0).mean()),
                    })
    return pd.DataFrame(rows)


def run_opportunity_ranking_discovery(files: Iterable[Path], cfg: StrategyConfig, dev_start, dev_end,
                                      validation_start, validation_end, bootstrap_samples=1000):
    base = run_cross_sectional_momentum(
        files, cfg, dev_start, dev_end, validation_start, validation_end, bootstrap_samples
    )
    if base.events.empty:
        z = pd.DataFrame()
        return OpportunityRankingResult(z, z, z, z)
    e = _add_scores(base.events)
    return OpportunityRankingResult(
        events=e,
        rank_summary=_rank_summary(e, bootstrap_samples),
        winner_capture=_winner_capture(e),
        score_spread=_score_spread(e, bootstrap_samples),
    )


def write_opportunity_ranking_reports(r: OpportunityRankingResult, report_dir: Path):
    report_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, df in [("events", r.events), ("rank_summary", r.rank_summary),
                     ("winner_capture", r.winner_capture), ("score_spread", r.score_spread)]:
        p = report_dir / (name + (".parquet" if name == "events" else ".csv"))
        df.to_parquet(p, index=False) if p.suffix == ".parquet" else df.to_csv(p, index=False)
        paths[name] = p

    focus = r.rank_summary[(r.rank_summary.snapshot_time.isin(["10:30", "11:00", "12:00"])) &
                           (r.rank_summary.horizon_min.isin([60, 120]))]
    capture = r.winner_capture[(r.winner_capture.snapshot_time.isin(["10:30", "11:00", "12:00"]))]
    spread = r.score_spread[(r.score_spread.snapshot_time.isin(["10:30", "11:00", "12:00"]))]
    html = f'''<!doctype html><meta charset="utf-8"><title>V19 Opportunity Ranking Discovery</title>
<style>body{{font-family:system-ui;margin:30px;max-width:1500px}}table{{border-collapse:collapse;width:100%;margin-bottom:28px}}td,th{{padding:7px;border-bottom:1px solid #ddd;font-size:12px}}.warn{{padding:12px;background:#fff3cd}}</style>
<h1>V19 Opportunity Ranking Discovery</h1>
<p class="warn"><b>DEV + Validation only. 2026 remains locked.</b> Discovery only: no capital, entries, stops or trading P/L. Future-winner labels are evaluation targets only and are never ranking inputs. Current NIFTY100 membership used historically remains survivorship-biased.</p>
<h2>Causal Top-K Forward Returns</h2>{focus.to_html(index=False, float_format=lambda x: f"{x:.4f}")}
<h2>Future Winner Capture</h2>{capture.to_html(index=False, float_format=lambda x: f"{x:.4f}")}
<h2>Top-K vs Bottom-Half Score Spread</h2>{spread.to_html(index=False, float_format=lambda x: f"{x:.4f}")}
<p>Score families are frozen equal-weight percentile composites: momentum only; momentum+RVOL; momentum+RVOL+VWAP strength; and those three plus gap rank.</p>'''
    hp = report_dir / "opportunity_ranking_dashboard.html"; hp.write_text(html, encoding="utf-8"); paths["dashboard"] = hp
    return paths

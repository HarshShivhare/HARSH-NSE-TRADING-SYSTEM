from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StrategyConfig:
    gap_min_pct: float = 1.0
    opening_range_minutes: int = 15
    rvol_min: float = 1.5
    rvol_lookback_days: int = 20
    daily_sma_days: int = 200
    atr_period: int = 14
    stop_mode: str = "atr"  # atr | opening_range | breakout_candle
    atr_multiple: float = 1.5
    target_r: float = 2.0
    earliest_entry: str = "09:30"
    latest_entry: str = "14:45"


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    # Kite timestamps generally include Asia/Kolkata offset. Preserve timezone if present.
    out = out.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    out["session"] = out["date"].dt.date
    out["time"] = out["date"].dt.strftime("%H:%M")
    return out


def _daily_features(df: pd.DataFrame, sma_days: int) -> pd.DataFrame:
    daily = (
        df.groupby("session", as_index=False)
        .agg(
            day_open=("open", "first"),
            day_high=("high", "max"),
            day_low=("low", "min"),
            day_close=("close", "last"),
            day_volume=("volume", "sum"),
        )
    )
    daily["prev_close"] = daily["day_close"].shift(1)
    daily["prev_high"] = daily["day_high"].shift(1)
    daily["prev_low"] = daily["day_low"].shift(1)
    # Critical: SMA200 is based on completed DAILY closes, not 200 five-minute bars.
    daily["daily_sma"] = daily["day_close"].rolling(sma_days, min_periods=sma_days).mean().shift(1)
    daily["gap_pct"] = ((daily["day_open"] / daily["prev_close"]) - 1.0) * 100.0
    return daily


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def prepare_features(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    out = _normalise(df)
    daily = _daily_features(out, cfg.daily_sma_days)
    out = out.merge(daily[["session", "prev_close", "prev_high", "prev_low", "daily_sma", "gap_pct"]], on="session", how="left")

    # Opening-range high/low from the first N minutes of regular trading.
    session_start = pd.to_datetime(out["session"].astype(str) + " 09:15:00")
    if out["date"].dt.tz is not None:
        session_start = session_start.dt.tz_localize(out["date"].dt.tz)
    elapsed = (out["date"].reset_index(drop=True) - session_start.reset_index(drop=True)).dt.total_seconds() / 60.0
    in_or = (elapsed >= 0) & (elapsed < cfg.opening_range_minutes)
    or_rows = out.loc[in_or].groupby("session").agg(or_high=("high", "max"), or_low=("low", "min")).reset_index()
    out = out.merge(or_rows, on="session", how="left")

    # Slot-relative volume: current bar volume / mean volume for the same clock slot
    # over prior sessions. This avoids comparing 09:20 volume with midday volume.
    slot_mean = (
        out.groupby("time")["volume"]
        .transform(lambda s: s.shift(1).rolling(cfg.rvol_lookback_days, min_periods=max(5, cfg.rvol_lookback_days // 2)).mean())
    )
    out["rvol"] = out["volume"] / slot_mean.replace(0, np.nan)
    out["atr"] = _atr(out, cfg.atr_period)

    return out


def signal_mask(features: pd.DataFrame, cfg: StrategyConfig) -> pd.Series:
    f = features
    time_ok = (f["time"] >= cfg.earliest_entry) & (f["time"] <= cfg.latest_entry)
    trend_ok = f["prev_close"] > f["daily_sma"]
    gap_ok = f["gap_pct"] >= cfg.gap_min_pct
    breakout_ok = (f["close"] > f["prev_high"]) & (f["close"] > f["or_high"])
    rvol_ok = f["rvol"] >= cfg.rvol_min
    complete = f[["prev_close", "prev_high", "daily_sma", "or_high", "rvol", "atr"]].notna().all(axis=1)
    return complete & time_ok & trend_ok & gap_ok & breakout_ok & rvol_ok

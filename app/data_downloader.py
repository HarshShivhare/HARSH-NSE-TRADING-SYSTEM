from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import random
import time
from typing import Iterable

import pandas as pd

from .zerodha import get_kite

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
META_DIR = DATA_DIR / "_meta"
DATA_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class DownloadResult:
    symbol: str
    status: str
    rows: int = 0
    path: str = ""
    error: str = ""


def get_instruments(exchange: str = "NSE", refresh: bool = False) -> pd.DataFrame:
    """Fetch and locally cache the Zerodha instrument dump for the day."""
    cache_path = META_DIR / f"instruments_{exchange}_{datetime.now():%Y%m%d}.parquet"
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)

    kite = get_kite(require_access_token=True)
    df = pd.DataFrame(kite.instruments(exchange))
    if df.empty:
        raise RuntimeError(f"No instruments returned for exchange {exchange}")
    df.to_parquet(cache_path, index=False)
    return df


def find_instrument(
    symbol: str,
    exchange: str = "NSE",
    instruments: pd.DataFrame | None = None,
) -> dict:
    df = instruments if instruments is not None else get_instruments(exchange)
    symbol_upper = symbol.strip().upper()
    row = df[df["tradingsymbol"].astype(str).str.upper() == symbol_upper]
    if row.empty:
        raise ValueError(f"Symbol not found: {exchange}:{symbol_upper}")

    # Prefer normal NSE equity rows if duplicate symbols ever occur.
    if "segment" in row.columns:
        eq = row[row["segment"].astype(str).str.upper().eq("NSE")]
        if not eq.empty:
            row = eq
    return row.iloc[0].to_dict()


def _chunks(start: datetime, end: datetime, days_per_chunk: int = 60):
    """Yield non-overlapping date ranges. 60 days is intentionally conservative."""
    current = start
    while current <= end:
        chunk_end = min(current + timedelta(days=days_per_chunk) - timedelta(seconds=1), end)
        yield current, chunk_end
        current = chunk_end + timedelta(seconds=1)


def _historical_with_retry(
    kite,
    *,
    instrument_token: int,
    from_date: datetime,
    to_date: datetime,
    interval: str,
    max_retries: int = 5,
    base_delay: float = 1.0,
):
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return kite.historical_data(
                instrument_token=instrument_token,
                from_date=from_date,
                to_date=to_date,
                interval=interval,
                continuous=False,
                oi=False,
            )
        except Exception as exc:  # Kite raises several request/network exception types.
            last_error = exc
            if attempt >= max_retries:
                break
            delay = base_delay * (2**attempt) + random.uniform(0, 0.4)
            print(
                f"    Request failed ({type(exc).__name__}). "
                f"Retry {attempt + 1}/{max_retries} in {delay:.1f}s..."
            )
            time.sleep(delay)
    raise RuntimeError(
        f"Historical request failed after {max_retries + 1} attempts: {last_error}"
    ) from last_error


def _save_frame(df: pd.DataFrame, path: Path, save_format: str) -> None:
    if save_format == "csv":
        df.to_csv(path, index=False)
    elif save_format == "parquet":
        df.to_parquet(path, index=False)
    else:
        raise ValueError("save_format must be 'csv' or 'parquet'")


def download_historical(
    symbol: str,
    start: datetime,
    end: datetime,
    interval: str = "5minute",
    exchange: str = "NSE",
    save_format: str = "parquet",
    pause_seconds: float = 0.40,
    days_per_chunk: int = 60,
    max_retries: int = 5,
    instruments: pd.DataFrame | None = None,
    overwrite: bool = False,
) -> Path:
    """Download one symbol and save a clean chronological file."""
    if end < start:
        raise ValueError("end must be on or after start")

    symbol = symbol.strip().upper()
    safe_start = start.strftime("%Y%m%d")
    safe_end = end.strftime("%Y%m%d")
    filename = f"{exchange}_{symbol}_{interval}_{safe_start}_{safe_end}.{save_format}"
    path = DATA_DIR / filename

    if path.exists() and not overwrite:
        print(f"  {symbol}: already exists, skipping ({path.name})")
        return path

    kite = get_kite(require_access_token=True)
    instrument = find_instrument(symbol, exchange, instruments=instruments)
    token = int(instrument["instrument_token"])

    frames: list[pd.DataFrame] = []
    chunks = list(_chunks(start, end, days_per_chunk=days_per_chunk))
    for idx, (chunk_start, chunk_end) in enumerate(chunks, start=1):
        print(
            f"  {symbol}: chunk {idx}/{len(chunks)} "
            f"{chunk_start:%Y-%m-%d} -> {chunk_end:%Y-%m-%d}"
        )
        candles = _historical_with_retry(
            kite,
            instrument_token=token,
            from_date=chunk_start,
            to_date=chunk_end,
            interval=interval,
            max_retries=max_retries,
        )
        if candles:
            frames.append(pd.DataFrame(candles))
        time.sleep(pause_seconds)

    if not frames:
        raise RuntimeError(f"No data returned for {exchange}:{symbol}")

    df = pd.concat(frames, ignore_index=True)
    if "date" not in df.columns:
        raise RuntimeError(f"Unexpected historical data format for {symbol}: no 'date' column")
    df.drop_duplicates(subset=["date"], inplace=True)
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    _save_frame(df, path, save_format)
    return path


def load_symbols_file(path: str | Path) -> list[str]:
    """Read one trading symbol per line; blank lines and # comments are ignored."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Symbols file not found: {p}")
    symbols: list[str] = []
    for raw in p.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            symbols.append(line.upper())
    return _dedupe(symbols)


def _dedupe(symbols: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for s in symbols:
        s = s.strip().upper()
        if s and s not in seen:
            seen.add(s)
            result.append(s)
    return result


def download_bulk(
    symbols: Iterable[str],
    start: datetime,
    end: datetime,
    interval: str = "5minute",
    exchange: str = "NSE",
    save_format: str = "parquet",
    pause_seconds: float = 0.40,
    days_per_chunk: int = 60,
    max_retries: int = 5,
    overwrite: bool = False,
) -> Path:
    """Download many symbols, continue on failures, and write a summary CSV."""
    symbols = _dedupe(symbols)
    if not symbols:
        raise ValueError("No symbols supplied")

    instruments = get_instruments(exchange)
    results: list[DownloadResult] = []
    print(
        f"Bulk download: {len(symbols)} symbols | {start:%Y-%m-%d} -> {end:%Y-%m-%d} "
        f"| {interval}"
    )

    for idx, symbol in enumerate(symbols, start=1):
        print(f"\n[{idx}/{len(symbols)}] {exchange}:{symbol}")
        try:
            path = download_historical(
                symbol=symbol,
                start=start,
                end=end,
                interval=interval,
                exchange=exchange,
                save_format=save_format,
                pause_seconds=pause_seconds,
                days_per_chunk=days_per_chunk,
                max_retries=max_retries,
                instruments=instruments,
                overwrite=overwrite,
            )
            if save_format == "parquet":
                rows = len(pd.read_parquet(path, columns=["date"]))
            else:
                rows = len(pd.read_csv(path, usecols=["date"]))
            results.append(DownloadResult(symbol, "ok", rows, str(path), ""))
            print(f"  Saved {rows:,} rows -> {path.name}")
        except Exception as exc:
            results.append(DownloadResult(symbol, "failed", 0, "", str(exc)))
            print(f"  FAILED: {exc}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = META_DIR / f"bulk_download_{stamp}.csv"
    pd.DataFrame([r.__dict__ for r in results]).to_csv(summary_path, index=False)

    ok = sum(r.status == "ok" for r in results)
    failed = len(results) - ok
    print(f"\nCompleted: {ok} succeeded, {failed} failed")
    print(f"Summary: {summary_path}")
    return summary_path

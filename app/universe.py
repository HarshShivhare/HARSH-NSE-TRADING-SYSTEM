from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import io

import pandas as pd
import requests

from .data_downloader import get_instruments


NIFTY_INDEX_URLS = {
    "nifty100": "https://www.niftyindices.com/IndexConstituent/ind_nifty100list.csv",
}


@dataclass
class UniverseResult:
    index_name: str
    source_url: str
    requested_symbols: list[str]
    matched_symbols: list[str]
    missing_symbols: list[str]
    output_path: Path
    audit_path: Path


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        s = str(value).strip().upper()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _find_symbol_column(df: pd.DataFrame) -> str:
    normalized = {str(c).strip().lower(): c for c in df.columns}
    for candidate in ["symbol", "tradingsymbol", "ticker"]:
        if candidate in normalized:
            return normalized[candidate]
    raise RuntimeError(f"Could not find symbol column in index CSV. Columns: {list(df.columns)}")


def fetch_index_symbols(index_name: str = "nifty100", timeout: int = 30) -> tuple[list[str], str]:
    key = index_name.strip().lower()
    if key not in NIFTY_INDEX_URLS:
        raise ValueError(f"Unsupported index: {index_name}. Supported: {', '.join(sorted(NIFTY_INDEX_URLS))}")
    url = NIFTY_INDEX_URLS[key]
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; NSE-Trading-Research/1.0)",
        "Accept": "text/csv,application/csv,text/plain,*/*",
    }
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text))
    symbol_col = _find_symbol_column(df)
    symbols = _dedupe(df[symbol_col].dropna().astype(str))
    if not symbols:
        raise RuntimeError("Index constituent file returned no symbols")
    return symbols, url


def build_current_nse_universe(
    index_name: str = "nifty100",
    output_path: str | Path = "config/nifty100_symbols.txt",
    refresh_instruments: bool = False,
) -> UniverseResult:
    requested, source_url = fetch_index_symbols(index_name)
    instruments = get_instruments("NSE", refresh=refresh_instruments)
    if "tradingsymbol" not in instruments.columns:
        raise RuntimeError("Zerodha instrument dump has no tradingsymbol column")

    available = set(instruments["tradingsymbol"].astype(str).str.upper())
    matched = [s for s in requested if s in available]
    missing = [s for s in requested if s not in available]

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "# CURRENT NIFTY 100 constituents mapped to Zerodha NSE symbols.",
        "# Research warning: this is a CURRENT membership list and therefore has survivorship bias",
        "# when applied to historical dates. Do not treat this as a point-in-time historical universe.",
    ]
    out.write_text("\n".join(header + matched) + "\n")

    audit_path = out.with_suffix(".audit.csv")
    audit_rows = ([{"symbol": s, "status": "matched"} for s in matched]
                  + [{"symbol": s, "status": "missing_in_kite"} for s in missing])
    pd.DataFrame(audit_rows).to_csv(audit_path, index=False)

    return UniverseResult(
        index_name=index_name,
        source_url=source_url,
        requested_symbols=requested,
        matched_symbols=matched,
        missing_symbols=missing,
        output_path=out,
        audit_path=audit_path,
    )

# NSE Trading System V7

V7 makes data quality a first-class part of the research pipeline. Raw Zerodha files stay immutable. Research/backtest/diagnostics automatically use a conservative cleaned view.

## What changed from V6

- **Raw Parquet files are never edited.**
- Sessions containing impossible OHLC / invalid OHLCV are quarantined in full.
- Sessions containing bars outside normal NSE cash hours (09:15-15:30), including Muhurat evening sessions, are excluded from the regular intraday strategy.
- V7 does **not** fabricate or repair OHLC values.
- `validate-data` reports special/non-regular sessions as INFO, not corruption.
- New `clean-audit` command shows exactly which symbol-days are removed and why.
- `backtest`, `diagnose`, `research`, and `stability` automatically use the cleaned view.
- New `stability` command breaks results down by year, stock, gap bucket, RVOL bucket, entry time, and profit concentration.

## Setup

Use Python 3.12 and copy your existing `.env` and `data/` folder into V7. Do not copy `.venv`; creating a fresh one is safest.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1. Validate raw source files

```bash
python main.py validate-data \
  --data-glob 'data/NSE_*_5minute_20230901_20260901.parquet' \
  --symbols-file config/starter_symbols.txt
```

Known source issues in the current 10-stock sample are expected to remain visible as validation errors. The raw files are intentionally preserved.

## 2. Audit the research-clean view

```bash
python main.py clean-audit \
  --data-glob 'data/NSE_*_5minute_20230901_20260901.parquet'
```

Expected from the currently inspected sample:

- 2023-11-12 Muhurat session excluded for each stock.
- 2024-11-01 Muhurat session excluded for each stock.
- 2024-06-25 excluded for BHARTIARTL, HDFCBANK, ICICIBANK because the 09:15 candle has impossible OHLC values.

This command writes an audit CSV and never edits the input Parquet files.

## 3. Rerun baseline on cleaned data

```bash
python main.py backtest \
  --data-glob 'data/NSE_*_5minute_20230901_20260901.parquet' \
  --capital 500000 \
  --risk-pct 0.005 \
  --gap-min 1.0 \
  --opening-range 15 \
  --rvol-min 1.5 \
  --stop-mode atr \
  --atr-multiple 1.5 \
  --target-r 2.0 \
  --max-trades-day 1
```

## 4. Controlled target sensitivity on cleaned data

```bash
python main.py research \
  --data-glob 'data/NSE_*_5minute_20230901_20260901.parquet' \
  --gap-min 1.0 \
  --opening-range 15 \
  --rvol-min 1.5 \
  --stop-mode atr \
  --atr-multiple 1.5 \
  --targets '0.75,1.0,1.25,1.5,2.0,2.5,3.0' \
  --dev-start 2024-06-01 \
  --dev-end 2025-12-31 \
  --test-start 2026-01-01 \
  --test-end 2026-09-01
```

Use DEVELOPMENT for parameter decisions. Do not tune against OUT_OF_SAMPLE.

## 5. Stability / regime analysis

Start with the unchanged 2R baseline:

```bash
python main.py stability \
  --data-glob 'data/NSE_*_5minute_20230901_20260901.parquet' \
  --gap-min 1.0 \
  --opening-range 15 \
  --rvol-min 1.5 \
  --stop-mode atr \
  --atr-multiple 1.5 \
  --target-r 2.0
```

Outputs include:

- `by_year.csv`
- `by_month.csv`
- `by_symbol.csv`
- `by_gap.csv`
- `by_rvol.csv`
- `by_entry_time.csv`
- `concentration.csv`
- `trades.parquet`

The concentration report helps identify whether a small number of trades or one stock dominate results.

## Research caveat

V7 is still a **signal/strategy research engine**, not the final portfolio simulator. Capital is still evolved separately per symbol before trades are aggregated. Before claiming final portfolio returns we need a shared-capital, event-driven portfolio engine with global daily limits and concurrent-position handling.

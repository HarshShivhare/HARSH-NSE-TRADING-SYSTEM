# NSE Trading System V6

V6 is a research-validation release. It does **not** add live trading.

## What changed

- `validate-data`: checks copied Parquet/CSV files for expected symbols, readable schema, timestamp duplicates, OHLC consistency, invalid prices/volume, date coverage, interval grid, session counts, and SHA-256 fingerprints.
- Backtests accept `--session-start` / `--session-end`. Features are still calculated from the full file first, so SMA200 and RVOL warm-up data are preserved.
- `research`: controlled target-R sensitivity with a DEVELOPMENT split and a separate OUT-OF-SAMPLE split.
- Default research targets: 0.75R, 1R, 1.25R, 1.5R, 2R, 2.5R, 3R.

## Important limitation

V6 still runs each symbol with its own evolving equity before aggregating trades. This is useful for signal/exit research, but it is **not yet a final shared-capital portfolio simulator**. Do not interpret aggregate return/drawdown as production portfolio performance. A later version should implement shared capital, global daily trade limits, and concurrent-position controls.

## 1. Validate your copied data first

```bash
python main.py validate-data \
  --data-glob 'data/NSE_*_5minute_20230901_20260901.parquet' \
  --symbols-file config/starter_symbols.txt
```

A clean run should show 10 files, no FAIL rows, and no ERROR issues. WARNs can be reviewed individually.

## 2. Reconfirm the baseline if desired

```bash
python main.py backtest \
  --data-glob 'data/NSE_*_5minute_20230901_20260901.parquet' \
  --gap-min 1.0 \
  --opening-range 15 \
  --rvol-min 1.5 \
  --stop-mode atr \
  --atr-multiple 1.5 \
  --target-r 2.0 \
  --max-trades-day 1
```

## 3. Run controlled target sensitivity

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

The first ~200 sessions of the 2023 file serve as indicator warm-up. Parameter choices must be made from DEVELOPMENT only. OUT_OF_SAMPLE should be inspected only after rules are frozen.

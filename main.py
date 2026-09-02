from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import glob

from app.zerodha import login_url, generate_access_token, profile
from app.data_downloader import download_historical, download_bulk, load_symbols_file
from app.backtest import BacktestConfig, backtest_files, summarize_trades, write_reports
from app.strategy import StrategyConfig
from app.diagnostics import diagnose_files, write_diagnostic_reports
from app.data_validator import validate_files, write_validation_reports
from app.research import target_sensitivity, write_research_reports


def _matched_files(pattern: str) -> list[Path]:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    return [p for p in files if p.is_file() and "/_meta/" not in str(p)]


def cmd_login_url(_args):
    print("Open this URL in your browser and log in to Zerodha:")
    print(login_url())


def cmd_token(args):
    session = generate_access_token(args.request_token)
    print("\nAccess token generated successfully.")
    print("Add this to your local .env as KITE_ACCESS_TOKEN.")
    print(session["access_token"])


def cmd_profile(_args):
    p = profile()
    print({k: p.get(k) for k in ["user_id", "user_name", "email", "broker"]})


def cmd_download(args):
    path = download_historical(
        symbol=args.symbol,
        start=datetime.fromisoformat(args.start),
        end=datetime.fromisoformat(args.end),
        interval=args.interval,
        save_format=args.format,
        overwrite=args.overwrite,
    )
    print(f"Saved: {path}")


def cmd_bulk_download(args):
    symbols = []
    if args.symbols:
        symbols.extend(s.strip() for s in args.symbols.split(",") if s.strip())
    if args.symbols_file:
        symbols.extend(load_symbols_file(args.symbols_file))
    if not symbols:
        raise SystemExit("Provide --symbols or --symbols-file")
    download_bulk(
        symbols=symbols,
        start=datetime.fromisoformat(args.start),
        end=datetime.fromisoformat(args.end),
        interval=args.interval,
        save_format=args.format,
        pause_seconds=args.pause,
        days_per_chunk=args.chunk_days,
        max_retries=args.retries,
        overwrite=args.overwrite,
    )


def _strategy_config_from_args(args):
    return StrategyConfig(
        gap_min_pct=args.gap_min,
        opening_range_minutes=args.opening_range,
        rvol_min=args.rvol_min,
        rvol_lookback_days=args.rvol_lookback,
        daily_sma_days=args.sma_days,
        atr_period=args.atr_period,
        stop_mode=getattr(args, "stop_mode", "atr"),
        atr_multiple=getattr(args, "atr_multiple", 1.5),
        target_r=getattr(args, "target_r", 2.0),
        earliest_entry=args.earliest_entry,
        latest_entry=args.latest_entry,
    )


def _bt_config_from_args(args):
    return BacktestConfig(args.capital, args.risk_pct, args.max_trades_day, args.slippage_bps)


def cmd_validate_data(args):
    files = _matched_files(args.data_glob)
    if not files:
        raise SystemExit(f"No files matched: {args.data_glob}")
    expected = load_symbols_file(args.symbols_file) if args.symbols_file else None
    result = validate_files(files, expected)
    print("\n=== V6 DATA VALIDATION ===")
    print(f"files checked     : {len(result.files)}")
    if not result.files.empty:
        print(f"PASS              : {(result.files['status'] == 'PASS').sum()}")
        print(f"WARN              : {(result.files['status'] == 'WARN').sum()}")
        print(f"FAIL              : {(result.files['status'] == 'FAIL').sum()}")
        cols = ["symbol","rows","sessions","first_timestamp","last_timestamp","median_bars_per_session","errors","warnings","status"]
        print("\n" + result.files[cols].to_string(index=False))
    if result.issues.empty:
        print("\nNo validation issues found.")
    else:
        print("\nIssues:")
        print(result.issues.to_string(index=False))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = Path(args.report_dir) / f"validation_{stamp}"
    paths = write_validation_reports(result, report_dir)
    print(f"\nValidation reports: {report_dir.resolve()}")
    for name, path in paths.items():
        print(f"  {name:20s}: {path.name}")
    if not result.issues.empty and (result.issues["severity"] == "ERROR").any():
        raise SystemExit(2)


def cmd_diagnose(args):
    files = _matched_files(args.data_glob)
    if not files:
        raise SystemExit(f"No files matched: {args.data_glob}")
    result = diagnose_files(files, _strategy_config_from_args(args))
    print("\n=== V6 SIGNAL FUNNEL ===")
    if not result.aggregate.empty:
        print(f"{'step':30s} {'indep bars':>10s} {'indep days':>11s} {'cum bars':>10s} {'cum days':>10s}")
        for _, row in result.aggregate.iterrows():
            print(f"{str(row['step']):30s} {int(row['independent_bars']):10d} {int(row['independent_sessions']):11d} {int(row['cumulative_bars']):10d} {int(row['cumulative_sessions']):10d}")
    if not result.data_quality.empty:
        total = int(result.data_quality['sessions'].sum()); complete = int(result.data_quality['complete_feature_sessions'].sum())
        print("\n=== DATA / WARM-UP ===")
        print(f"symbol-sessions              : {total}")
        print(f"complete-feature sessions    : {complete}")
        print(f"complete-feature coverage    : {(100*complete/total if total else 0):.2f}%")
    if not result.threshold_sensitivity.empty:
        print("\n=== GAP / RVOL SENSITIVITY (candidate sessions) ===")
        print(result.threshold_sensitivity.pivot(index='gap_min_pct', columns='rvol_min', values='candidate_sessions').to_string())
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = Path(args.report_dir) / f"diagnostics_{stamp}"
    write_diagnostic_reports(result, report_dir)
    print(f"\nDiagnostic reports: {report_dir.resolve()}")


def cmd_backtest(args):
    files = _matched_files(args.data_glob)
    if not files:
        raise SystemExit(f"No files matched: {args.data_glob}")
    scfg, bcfg = _strategy_config_from_args(args), _bt_config_from_args(args)
    trades = backtest_files(files, scfg, bcfg, session_start=args.session_start, session_end=args.session_end)
    summary = summarize_trades(trades, bcfg.initial_capital)
    print("\n=== V6 BACKTEST SUMMARY ===")
    for k,v in summary.items(): print(f"{k:24s}: {v:,.4f}" if isinstance(v,float) else f"{k:24s}: {v}")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = Path(args.report_dir) / f"backtest_{stamp}"
    paths = write_reports(trades, report_dir, bcfg.initial_capital, scfg, bcfg)
    print(f"\nReports: {report_dir.resolve()}")
    for name,path in paths.items(): print(f"  {name:20s}: {path.name}")


def cmd_research(args):
    files = _matched_files(args.data_glob)
    if not files:
        raise SystemExit(f"No files matched: {args.data_glob}")
    scfg, bcfg = _strategy_config_from_args(args), _bt_config_from_args(args)
    targets = [float(x.strip()) for x in args.targets.split(",") if x.strip()]
    table, _ = target_sensitivity(files, scfg, bcfg, targets, args.dev_start, args.dev_end, args.test_start, args.test_end)
    print("\n=== V6 TARGET SENSITIVITY ===")
    display = table.copy()
    for c in ["win_rate","return_pct","max_drawdown_pct"]:
        if c in display: display[c] = display[c].map(lambda x: round(x,4) if x is not None else x)
    print(display.to_string(index=False))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = Path(args.report_dir) / f"research_{stamp}"
    paths = write_research_reports(table, report_dir)
    print(f"\nResearch reports: {report_dir.resolve()}")
    for name,path in paths.items(): print(f"  {name:20s}: {path.name}")
    print("\nIMPORTANT: Choose/freeze parameters using DEVELOPMENT only. OUT_OF_SAMPLE is for confirmation, not tuning.")


def add_strategy_args(p, include_exit=True):
    p.add_argument("--gap-min", type=float, default=1.0)
    p.add_argument("--opening-range", type=int, default=15)
    p.add_argument("--rvol-min", type=float, default=1.5)
    p.add_argument("--rvol-lookback", type=int, default=20)
    p.add_argument("--sma-days", type=int, default=200)
    p.add_argument("--atr-period", type=int, default=14)
    p.add_argument("--earliest-entry", default="09:30")
    p.add_argument("--latest-entry", default="14:45")
    if include_exit:
        p.add_argument("--stop-mode", choices=["atr","opening_range","breakout_candle"], default="atr")
        p.add_argument("--atr-multiple", type=float, default=1.5)
        p.add_argument("--target-r", type=float, default=2.0)


def add_bt_args(p):
    p.add_argument("--capital", type=float, default=500000.0)
    p.add_argument("--risk-pct", type=float, default=0.005)
    p.add_argument("--max-trades-day", type=int, default=1)
    p.add_argument("--slippage-bps", type=float, default=5.0)


def build_parser():
    parser = argparse.ArgumentParser(description="NSE trading system V6 - validation + controlled research")
    sub = parser.add_subparsers(required=True)
    p=sub.add_parser("login-url"); p.set_defaults(func=cmd_login_url)
    p=sub.add_parser("token"); p.add_argument("request_token"); p.set_defaults(func=cmd_token)
    p=sub.add_parser("profile"); p.set_defaults(func=cmd_profile)
    p=sub.add_parser("download"); p.add_argument("symbol"); p.add_argument("--start",required=True); p.add_argument("--end",required=True); p.add_argument("--interval",default="5minute"); p.add_argument("--format",choices=["parquet","csv"],default="parquet"); p.add_argument("--overwrite",action="store_true"); p.set_defaults(func=cmd_download)
    p=sub.add_parser("bulk-download"); src=p.add_mutually_exclusive_group(required=True); src.add_argument("--symbols"); src.add_argument("--symbols-file"); p.add_argument("--start",required=True); p.add_argument("--end",required=True); p.add_argument("--interval",default="5minute"); p.add_argument("--format",choices=["parquet","csv"],default="parquet"); p.add_argument("--chunk-days",type=int,default=60); p.add_argument("--pause",type=float,default=.40); p.add_argument("--retries",type=int,default=5); p.add_argument("--overwrite",action="store_true"); p.set_defaults(func=cmd_bulk_download)

    p=sub.add_parser("validate-data", help="Verify copied historical files before research")
    p.add_argument("--data-glob",default="data/NSE_*_5minute_*.parquet"); p.add_argument("--symbols-file",default="config/starter_symbols.txt"); p.add_argument("--report-dir",default="reports"); p.set_defaults(func=cmd_validate_data)

    p=sub.add_parser("diagnose"); p.add_argument("--data-glob",default="data/NSE_*_5minute_*.parquet"); p.add_argument("--report-dir",default="reports"); add_strategy_args(p,False); p.set_defaults(func=cmd_diagnose)

    p=sub.add_parser("backtest"); p.add_argument("--data-glob",default="data/NSE_*_5minute_*.parquet"); p.add_argument("--report-dir",default="reports"); p.add_argument("--session-start"); p.add_argument("--session-end"); add_strategy_args(p,True); add_bt_args(p); p.set_defaults(func=cmd_backtest)

    p=sub.add_parser("research", help="Target sensitivity with development/out-of-sample split")
    p.add_argument("--data-glob",default="data/NSE_*_5minute_*.parquet"); p.add_argument("--report-dir",default="reports"); p.add_argument("--targets",default="0.75,1.0,1.25,1.5,2.0,2.5,3.0"); p.add_argument("--dev-start",default="2024-06-01"); p.add_argument("--dev-end",default="2025-12-31"); p.add_argument("--test-start",default="2026-01-01"); p.add_argument("--test-end",default="2026-09-01"); add_strategy_args(p,True); add_bt_args(p); p.set_defaults(func=cmd_research)
    return parser


if __name__ == "__main__":
    args=build_parser().parse_args(); args.func(args)

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
from app.data_cleaner import clean_market_data, summarize_cleaning
from app.stability import stability_tables, write_stability_reports
from app.robustness import run_development_grid, run_final_oos, write_robustness_reports
from app.universe import build_current_nse_universe
from app.signal_diagnostics import run_signal_diagnostics, write_signal_diagnostic_reports
from app.extension_robustness import (
    run_extension_development_grid,
    write_extension_reports,
    parse_optional_float_csv,
    parse_bool_csv,
)
from app.trade_path_analysis import run_trade_path_analysis, write_trade_path_reports
from app.post_breakout_analysis import run_post_breakout_analysis, write_post_breakout_reports


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


def cmd_universe(args):
    print(f"Building current {args.index.upper()} universe...", flush=True)
    result = build_current_nse_universe(
        index_name=args.index,
        output_path=args.output,
        refresh_instruments=args.refresh_instruments,
    )
    print("\n=== V8.1 UNIVERSE ===")
    print(f"index                 : {result.index_name.upper()}")
    print(f"constituents requested: {len(result.requested_symbols)}")
    print(f"matched Zerodha NSE   : {len(result.matched_symbols)}")
    print(f"missing in Kite       : {len(result.missing_symbols)}")
    print(f"symbols file          : {result.output_path.resolve()}")
    print(f"audit file            : {result.audit_path.resolve()}")
    if result.missing_symbols:
        print("missing symbols       : " + ", ".join(result.missing_symbols))
    print("\nWARNING: this is the CURRENT index membership, not a point-in-time historical universe.")
    print("Using it for older dates introduces survivorship bias. V8.1 records this explicitly.")


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
        require_trend=getattr(args, "require_trend", True),
        max_vwap_extension_pct=getattr(args, "max_vwap_extension_pct", None),
        max_or_extension_atr=getattr(args, "max_or_extension_atr", None),
    )


def _bt_config_from_args(args):
    return BacktestConfig(args.capital, args.risk_pct, args.max_trades_day, args.slippage_bps)


def cmd_validate_data(args):
    files = _matched_files(args.data_glob)
    if not files:
        raise SystemExit(f"No files matched: {args.data_glob}")
    expected = load_symbols_file(args.symbols_file) if args.symbols_file else None
    result = validate_files(files, expected)
    print("\n=== V7 DATA VALIDATION ===")
    print(f"files checked     : {len(result.files)}")
    if not result.files.empty:
        print(f"PASS              : {(result.files['status'] == 'PASS').sum()}")
        print(f"WARN              : {(result.files['status'] == 'WARN').sum()}")
        print(f"FAIL              : {(result.files['status'] == 'FAIL').sum()}")
        print(f"INFO issues       : {int(result.files.get('infos', 0).sum()) if 'infos' in result.files else 0}")
        cols = ["symbol","rows","sessions","first_timestamp","last_timestamp","median_bars_per_session","errors","warnings","infos","status"]
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


def cmd_clean_audit(args):
    files = _matched_files(args.data_glob)
    if not files:
        raise SystemExit(f"No files matched: {args.data_glob}")
    summaries = []
    audits = []
    for path in files:
        symbol = path.name.split("_")[1] if "_" in path.name else path.stem
        raw = __import__("pandas").read_parquet(path)
        result = clean_market_data(raw, symbol)
        summaries.append(summarize_cleaning(raw, result, symbol))
        if not result.audit.empty:
            a = result.audit.copy()
            a["file"] = path.name
            audits.append(a)
    import pandas as pd
    summary = pd.DataFrame(summaries)
    audit = pd.concat(audits, ignore_index=True) if audits else pd.DataFrame()
    print("\n=== V7 CLEANING AUDIT ===")
    print(summary.to_string(index=False))
    if audit.empty:
        print("\nNo sessions excluded.")
    else:
        print("\nExcluded/quarantined sessions:")
        cols = [c for c in ["symbol","session","reason","rows_affected","bad_rows","outside_regular_rows"] if c in audit.columns]
        print(audit[cols].to_string(index=False))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = Path(args.report_dir) / f"cleaning_{stamp}"
    report_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(report_dir / "cleaning_summary.csv", index=False)
    audit.to_csv(report_dir / "cleaning_audit.csv", index=False)
    print(f"\nCleaning reports: {report_dir.resolve()}")
    print("Raw parquet files were NOT modified.")


def cmd_stability(args):
    files = _matched_files(args.data_glob)
    if not files:
        raise SystemExit(f"No files matched: {args.data_glob}")
    scfg, bcfg = _strategy_config_from_args(args), _bt_config_from_args(args)
    trades = backtest_files(files, scfg, bcfg, session_start=args.session_start, session_end=args.session_end, quiet=True)
    summary = summarize_trades(trades, bcfg.initial_capital)
    tables = stability_tables(trades)
    print("\n=== V7 STABILITY SUMMARY ===")
    for k in ["trades","wins","win_rate","profit_factor","expectancy_r","net_pnl","avg_mfe_r","avg_mae_r"]:
        v = summary.get(k)
        print(f"{k:24s}: {v:,.4f}" if isinstance(v, float) else f"{k:24s}: {v}")
    for name in ["by_year","by_symbol","by_gap","by_rvol","concentration"]:
        print(f"\n--- {name.upper()} ---")
        print(tables[name].to_string(index=False) if not tables[name].empty else "No data")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = Path(args.report_dir) / f"stability_{stamp}"
    paths = write_stability_reports(tables, report_dir)
    trades.to_parquet(report_dir / "trades.parquet", index=False)
    print(f"\nStability reports: {report_dir.resolve()}")
    for name, path in paths.items():
        print(f"  {name:20s}: {path.name}")


def cmd_diagnose(args):
    files = _matched_files(args.data_glob)
    if not files:
        raise SystemExit(f"No files matched: {args.data_glob}")
    result = diagnose_files(files, _strategy_config_from_args(args))
    print("\n=== V7 SIGNAL FUNNEL ===")
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
    print("\n=== V7 BACKTEST SUMMARY ===")
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
    print("\n=== V7 TARGET SENSITIVITY ===")
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



def _csv_floats(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def _csv_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]



def cmd_signal_diagnostics(args):
    files = _matched_files(args.data_glob)
    if not files:
        raise SystemExit(f"No files matched: {args.data_glob}")
    cfg = _strategy_config_from_args(args)
    result = run_signal_diagnostics(
        files, cfg, session_start=args.session_start, session_end=args.session_end
    )
    print("\n=== V9 SIGNAL DIAGNOSTICS ===")
    print(f"files analyzed : {len(files)}")
    print(f"events         : {len(result.events)} across all diagnostic variants")
    for title, table in [
        ("CUMULATIVE FILTER STAGES", result.stage_summary),
        ("DROP-ONE FILTER TEST", result.drop_one_summary),
        ("FULL SIGNAL BY VWAP EXTENSION", result.by_vwap_extension),
        ("FULL SIGNAL BY OR EXTENSION / ATR", result.by_or_extension_atr),
        ("FULL SIGNAL BY ENTRY TIME", result.by_entry_time),
        ("FULL SIGNAL BY GAP", result.by_gap),
        ("FULL SIGNAL BY RVOL", result.by_rvol),
    ]:
        print(f"\n--- {title} ---")
        print(table.to_string(index=False) if not table.empty else "No data")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = Path(args.report_dir) / f"signal_diagnostics_{stamp}"
    paths = write_signal_diagnostic_reports(result, report_dir)
    print(f"\nV9 reports: {report_dir.resolve()}")
    for name, path in paths.items():
        print(f"  {name:24s}: {path.name}")
    print("\nResearch note: these are diagnostic forward-return measurements, not a tradable strategy or final OOS test.")


def cmd_extension_robustness(args):
    files = _matched_files(args.data_glob)
    if not files:
        raise SystemExit(f"No files matched: {args.data_glob}")
    scfg, bcfg = _strategy_config_from_args(args), _bt_config_from_args(args)
    try:
        trend_modes = parse_bool_csv(args.trend_modes)
        vwap_maxes = parse_optional_float_csv(args.vwap_maxes)
        or_maxes = parse_optional_float_csv(args.or_maxes)
        rvols = _csv_floats(args.rvols)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    results, ranking = run_extension_development_grid(
        files=files,
        base_strategy=scfg,
        bt_cfg=bcfg,
        trend_modes=trend_modes,
        vwap_maxes=vwap_maxes,
        or_maxes=or_maxes,
        rvols=rvols,
        dev_start=args.dev_start,
        dev_end=args.dev_end,
        validation_start=args.validation_start,
        validation_end=args.validation_end,
        min_trades_per_split=args.min_trades,
        bootstrap_samples=args.bootstrap_samples,
    )

    print("\n=== V10 EXTENSION-AWARE ROBUSTNESS — DEV + VALIDATION ONLY ===")
    print(f"grid combinations          : {len(ranking)}")
    print(f"robust-gate combinations   : {int(ranking['robust_gate'].sum()) if not ranking.empty else 0}")
    print("\nTop candidates (2026 OOS remains locked):")
    cols = [
        "trend", "max_vwap_extension_pct", "max_or_extension_atr", "rvol_min",
        "trades_dev", "expectancy_r_dev", "profit_factor_dev",
        "trades_validation", "expectancy_r_validation", "profit_factor_validation",
        "expectancy_ci_low_dev", "expectancy_ci_low_validation",
        "robust_gate", "ci_positive_both", "worst_split_expectancy_r",
    ]
    print(ranking[cols].head(args.top).to_string(index=False) if not ranking.empty else "No results")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = Path(args.report_dir) / f"extension_robustness_{stamp}"
    paths = write_extension_reports(report_dir, results, ranking)
    print(f"\nV10 reports: {report_dir.resolve()}")
    for name, path in paths.items():
        print(f"  {name:22s}: {path.name}")
    print("\n2026 FINAL OOS was NOT evaluated and this command has no final stage by design.")
    print("Only promote a broad, stable parameter neighborhood; do not tune to one winning cell.")

def cmd_trade_path(args):
    files = _matched_files(args.data_glob)
    if not files:
        raise SystemExit(f"No files matched: {args.data_glob}")
    args.require_trend = (args.trend == "on")
    scfg, bcfg = _strategy_config_from_args(args), _bt_config_from_args(args)
    result = run_trade_path_analysis(
        files, scfg, bcfg,
        dev_start=args.dev_start, dev_end=args.dev_end,
        validation_start=args.validation_start, validation_end=args.validation_end,
    )
    print("\n=== V11 TRADE PATH + BARRIER ANALYSIS — DEV + VALIDATION ONLY ===")
    print(f"events : {len(result.events)}")
    print("\n--- LIFECYCLE ---")
    print(result.lifecycle.to_string(index=False) if not result.lifecycle.empty else "No events")
    print("\n--- BARRIER ORDERING + COST DECOMPOSITION ---")
    print(result.barrier_ordering.to_string(index=False) if not result.barrier_ordering.empty else "No events")
    print("\n--- TIME PROFILE ---")
    print(result.time_profile.to_string(index=False) if not result.time_profile.empty else "No events")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = Path(args.report_dir) / f"trade_path_{stamp}"
    paths = write_trade_path_reports(result, report_dir, scfg)
    print(f"\nV11 reports: {report_dir.resolve()}")
    for name, path in paths.items():
        print(f"  {name:22s}: {path.name}")
    print(f"\nOpen graphical dashboard: {paths['dashboard'].resolve()}")
    print("2026 FINAL OOS was NOT evaluated. This command only reads DEV + VALIDATION dates.")


def cmd_post_breakout(args):
    files = _matched_files(args.data_glob)
    if not files:
        raise SystemExit(f"No files matched: {args.data_glob}")
    args.require_trend = (args.trend == "on")
    scfg = _strategy_config_from_args(args)
    result = run_post_breakout_analysis(
        files, scfg,
        dev_start=args.dev_start, dev_end=args.dev_end,
        validation_start=args.validation_start, validation_end=args.validation_end,
        bootstrap_samples=args.bootstrap_samples,
    )
    print("\n=== V12 DELAYED ENTRY + RETEST ANALYSIS — DEV + VALIDATION ONLY ===")
    print(f"events : {len(result.events)} across delayed-entry variants")
    print("\n--- DELAY / HORIZON SUMMARY ---")
    print(result.delay_summary.to_string(index=False) if not result.delay_summary.empty else "No events")
    print("\n--- 120M DISTRIBUTION SUMMARY ---")
    print(result.distribution_summary.to_string(index=False) if not result.distribution_summary.empty else "No events")
    print("\n--- POSITIVE RETURN CONCENTRATION ---")
    print(result.concentration_summary.to_string(index=False) if not result.concentration_summary.empty else "No events")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = Path(args.report_dir) / f"post_breakout_{stamp}"
    paths = write_post_breakout_reports(result, report_dir, scfg)
    print(f"\nV12 reports: {report_dir.resolve()}")
    for name, path in paths.items():
        print(f"  {name:24s}: {path.name}")
    print(f"\nOpen graphical dashboard: {paths['dashboard'].resolve()}")
    print("2026 FINAL OOS was NOT evaluated. V12 only reads DEV + VALIDATION dates.")


def cmd_robustness(args):
    files = _matched_files(args.data_glob)
    if not files:
        raise SystemExit(f"No files matched: {args.data_glob}")
    scfg, bcfg = _strategy_config_from_args(args), _bt_config_from_args(args)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = Path(args.report_dir) / f"robustness_{stamp}"

    if args.stage == "develop":
        results, ranking = run_development_grid(
            files=files,
            base_strategy=scfg,
            bt_cfg=bcfg,
            gaps=_csv_floats(args.gaps),
            rvols=_csv_floats(args.rvols),
            targets=_csv_floats(args.targets),
            opening_ranges=_csv_ints(args.opening_ranges),
            dev_start=args.dev_start,
            dev_end=args.dev_end,
            validation_start=args.validation_start,
            validation_end=args.validation_end,
            min_trades_per_split=args.min_trades,
            bootstrap_samples=args.bootstrap_samples,
        )
        print("\n=== V8 ROBUSTNESS — DEVELOPMENT + VALIDATION ONLY ===")
        print(f"grid combinations          : {len(ranking)}")
        print(f"robust-gate combinations   : {int(ranking['robust_gate'].sum()) if not ranking.empty else 0}")
        print("\nTop candidates (OOS intentionally hidden):")
        cols = [
            "gap_min","rvol_min","target_r","opening_range",
            "trades_dev","expectancy_r_dev","profit_factor_dev",
            "trades_validation","expectancy_r_validation","profit_factor_validation",
            "expectancy_ci_low_dev","expectancy_ci_low_validation",
            "robust_gate","ci_positive_both","worst_split_expectancy_r",
        ]
        print(ranking[cols].head(args.top).to_string(index=False) if not ranking.empty else "No results")
        paths = write_robustness_reports(report_dir, results=results, ranking=ranking)
        print("\nFINAL OOS was NOT evaluated. Freeze a parameter set before using --stage final.")
    else:
        summary, trades, quarterly = run_final_oos(
            files=files,
            strategy_cfg=scfg,
            bt_cfg=bcfg,
            oos_start=args.oos_start,
            oos_end=args.oos_end,
            bootstrap_samples=args.bootstrap_samples,
        )
        print("\n=== V8 FINAL OOS CHECK ===")
        print(f"frozen parameters: gap>={scfg.gap_min_pct}, RVOL>={scfg.rvol_min}, OR={scfg.opening_range_minutes}m, target={scfg.target_r}R")
        for k in ["trades","wins","win_rate","profit_factor","expectancy_r","expectancy_ci_low","expectancy_ci_high","net_pnl","return_pct","max_drawdown_pct"]:
            v = summary.get(k)
            print(f"{k:24s}: {v:,.4f}" if isinstance(v, float) else f"{k:24s}: {v}")
        print("\nQuarterly OOS stability:")
        print(quarterly.to_string(index=False) if not quarterly.empty else "No OOS trades")
        paths = write_robustness_reports(report_dir, final_trades=trades, final_quarterly=quarterly, final_summary=summary)
        print("\nIMPORTANT: Do not retune parameters from this OOS result. If you do, OOS is no longer unseen.")

    print(f"\nRobustness reports: {report_dir.resolve()}")
    for name, path in paths.items():
        print(f"  {name:22s}: {path.name}")

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
    parser = argparse.ArgumentParser(description="NSE trading system V8 - robust development/validation/OOS research")
    sub = parser.add_subparsers(required=True)
    p=sub.add_parser("login-url"); p.set_defaults(func=cmd_login_url)
    p=sub.add_parser("token"); p.add_argument("request_token"); p.set_defaults(func=cmd_token)
    p=sub.add_parser("profile"); p.set_defaults(func=cmd_profile)
    p=sub.add_parser("download"); p.add_argument("symbol"); p.add_argument("--start",required=True); p.add_argument("--end",required=True); p.add_argument("--interval",default="5minute"); p.add_argument("--format",choices=["parquet","csv"],default="parquet"); p.add_argument("--overwrite",action="store_true"); p.set_defaults(func=cmd_download)
    p=sub.add_parser("universe", help="V8.1 build a current liquid NSE research universe")
    p.add_argument("--index", choices=["nifty100"], default="nifty100")
    p.add_argument("--output", default="config/nifty100_symbols.txt")
    p.add_argument("--refresh-instruments", action="store_true")
    p.set_defaults(func=cmd_universe)
    p=sub.add_parser("bulk-download"); src=p.add_mutually_exclusive_group(required=True); src.add_argument("--symbols"); src.add_argument("--symbols-file"); p.add_argument("--start",required=True); p.add_argument("--end",required=True); p.add_argument("--interval",default="5minute"); p.add_argument("--format",choices=["parquet","csv"],default="parquet"); p.add_argument("--chunk-days",type=int,default=60); p.add_argument("--pause",type=float,default=.40); p.add_argument("--retries",type=int,default=5); p.add_argument("--overwrite",action="store_true"); p.set_defaults(func=cmd_bulk_download)

    p=sub.add_parser("validate-data", help="Verify copied historical files before research")
    p.add_argument("--data-glob",default="data/NSE_*_5minute_*.parquet"); p.add_argument("--symbols-file",default="config/starter_symbols.txt"); p.add_argument("--report-dir",default="reports"); p.set_defaults(func=cmd_validate_data)

    p=sub.add_parser("clean-audit", help="Show exactly which symbol-sessions V7 excludes without modifying raw files")
    p.add_argument("--data-glob",default="data/NSE_*_5minute_*.parquet"); p.add_argument("--report-dir",default="reports"); p.set_defaults(func=cmd_clean_audit)

    p=sub.add_parser("diagnose"); p.add_argument("--data-glob",default="data/NSE_*_5minute_*.parquet"); p.add_argument("--report-dir",default="reports"); add_strategy_args(p,False); p.set_defaults(func=cmd_diagnose)

    p=sub.add_parser("backtest"); p.add_argument("--data-glob",default="data/NSE_*_5minute_*.parquet"); p.add_argument("--report-dir",default="reports"); p.add_argument("--session-start"); p.add_argument("--session-end"); add_strategy_args(p,True); add_bt_args(p); p.set_defaults(func=cmd_backtest)

    p=sub.add_parser("stability", help="Cleaned-data stability analysis by year/stock/gap/RVOL/entry time")
    p.add_argument("--data-glob",default="data/NSE_*_5minute_*.parquet"); p.add_argument("--report-dir",default="reports"); p.add_argument("--session-start"); p.add_argument("--session-end"); add_strategy_args(p,True); add_bt_args(p); p.set_defaults(func=cmd_stability)



    p=sub.add_parser("signal-diagnostics", help="V9 filter ablation + forward-return / entry-extension diagnostics")
    p.add_argument("--data-glob",default="data/NSE_*_5minute_*.parquet")
    p.add_argument("--report-dir",default="reports")
    p.add_argument("--session-start",default="2023-09-01")
    p.add_argument("--session-end",default="2025-12-31")
    add_strategy_args(p,False)
    p.set_defaults(func=cmd_signal_diagnostics)

    p=sub.add_parser("extension-robustness", help="V10 extension-aware breakout robustness; DEV + validation only")
    p.add_argument("--data-glob",default="data/NSE_*_5minute_*.parquet")
    p.add_argument("--report-dir",default="reports")
    p.add_argument("--trend-modes",default="on,off",help="Comma list: on,off")
    p.add_argument("--vwap-maxes",default="none,1.0,0.75",help="Max entry extension above signal VWAP in percent; use none for no cap")
    p.add_argument("--or-maxes",default="none,0.5,0.25",help="Max entry extension above OR high in ATR; use none for no cap")
    p.add_argument("--rvols",default="1.5,3.0")
    p.add_argument("--dev-start",default="2023-09-01"); p.add_argument("--dev-end",default="2025-06-30")
    p.add_argument("--validation-start",default="2025-07-01"); p.add_argument("--validation-end",default="2025-12-31")
    p.add_argument("--min-trades",type=int,default=20); p.add_argument("--bootstrap-samples",type=int,default=1000); p.add_argument("--top",type=int,default=20)
    add_strategy_args(p,True); add_bt_args(p); p.set_defaults(func=cmd_extension_robustness)

    p=sub.add_parser("trade-path", help="V11 trade path, barrier ordering, cost decomposition + graphical dashboard")
    p.add_argument("--data-glob",default="data/NSE_*_5minute_*.parquet")
    p.add_argument("--report-dir",default="reports")
    p.add_argument("--dev-start",default="2023-09-01"); p.add_argument("--dev-end",default="2025-06-30")
    p.add_argument("--validation-start",default="2025-07-01"); p.add_argument("--validation-end",default="2025-12-31")
    p.add_argument("--trend",choices=["on","off"],default="off",help="Trend filter mode; V11 default follows the best broad V10 family")
    p.add_argument("--max-vwap-extension-pct",type=float,default=None,help="Optional next-bar entry cap above signal VWAP in percent")
    p.add_argument("--max-or-extension-atr",type=float,default=0.5,help="Optional next-bar entry cap above OR high in ATR; default 0.5")
    add_strategy_args(p,True); add_bt_args(p)
    p.set_defaults(func=cmd_trade_path, rvol_min=3.0)

    p=sub.add_parser("post-breakout", help="V12 delayed-entry, retest/confirmation and distribution research")
    p.add_argument("--data-glob",default="data/NSE_*_5minute_*.parquet")
    p.add_argument("--report-dir",default="reports")
    p.add_argument("--dev-start",default="2023-09-01"); p.add_argument("--dev-end",default="2025-06-30")
    p.add_argument("--validation-start",default="2025-07-01"); p.add_argument("--validation-end",default="2025-12-31")
    p.add_argument("--trend",choices=["on","off"],default="off")
    p.add_argument("--bootstrap-samples",type=int,default=1000)
    add_strategy_args(p,True)
    p.set_defaults(func=cmd_post_breakout, rvol_min=3.0, max_or_extension_atr=0.5, max_vwap_extension_pct=None)

    p=sub.add_parser("robustness", help="V8 parameter robustness with protected final OOS")
    p.add_argument("--stage", choices=["develop","final"], default="develop")
    p.add_argument("--data-glob",default="data/NSE_*_5minute_*.parquet"); p.add_argument("--report-dir",default="reports")
    p.add_argument("--gaps",default="1.0,1.5,2.0"); p.add_argument("--rvols",default="1.5,3.0,5.0"); p.add_argument("--targets",default="1.5,2.0,2.5"); p.add_argument("--opening-ranges",default="15")
    p.add_argument("--dev-start",default="2023-09-01"); p.add_argument("--dev-end",default="2025-06-30")
    p.add_argument("--validation-start",default="2025-07-01"); p.add_argument("--validation-end",default="2025-12-31")
    p.add_argument("--oos-start",default="2026-01-01"); p.add_argument("--oos-end",default="2026-08-31")
    p.add_argument("--min-trades",type=int,default=10); p.add_argument("--bootstrap-samples",type=int,default=1000); p.add_argument("--top",type=int,default=15)
    add_strategy_args(p,True); add_bt_args(p); p.set_defaults(func=cmd_robustness)

    p=sub.add_parser("research", help="Target sensitivity with development/out-of-sample split")
    p.add_argument("--data-glob",default="data/NSE_*_5minute_*.parquet"); p.add_argument("--report-dir",default="reports"); p.add_argument("--targets",default="0.75,1.0,1.25,1.5,2.0,2.5,3.0"); p.add_argument("--dev-start",default="2024-06-01"); p.add_argument("--dev-end",default="2025-12-31"); p.add_argument("--test-start",default="2026-01-01"); p.add_argument("--test-end",default="2026-09-01"); add_strategy_args(p,True); add_bt_args(p); p.set_defaults(func=cmd_research)
    return parser


if __name__ == "__main__":
    args=build_parser().parse_args(); args.func(args)

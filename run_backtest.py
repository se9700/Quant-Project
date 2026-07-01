"""End-to-end backtest: data -> valuation -> triple-barrier labels ->
meta-model (walk-forward) -> portfolio simulation -> report.

Usage:
    python run_backtest.py [--config config.yaml] [--refresh-data]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from quantfolio.backtest import run_backtest
from quantfolio.config import load_config
from quantfolio.data import load_earnings, load_fundamentals, load_prices
from quantfolio.features import build_features
from quantfolio.fundamentals import build_earnings_factors, build_fundamental_factors
from quantfolio.labeling import generate_events, label_events
from quantfolio.metrics import performance_stats, plot_results, print_report, trade_stats
from quantfolio.model import bet_size, final_feature_importance, walk_forward_probabilities
from quantfolio.valuation import build_value_scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--refresh-data", action="store_true", help="re-download instead of using cache")
    args = ap.parse_args()

    cfg = load_config(args.config)
    lab, sig, val, mdl, bt = (cfg["labeling"], cfg["signals"], cfg["valuation"],
                              cfg["model"], cfg["backtest"])

    print("Loading prices...")
    tickers = cfg["universe"]
    prices = load_prices(tickers + [cfg["benchmark"]], cfg["start"], cfg["end"],
                         cfg["data_cache"], refresh=args.refresh_data)
    bench_rets = prices[cfg["benchmark"]].pct_change()
    prices = prices[[t for t in tickers if t in prices.columns]]
    print(f"  {prices.shape[1]} tickers, {prices.shape[0]} days "
          f"({prices.index[0].date()} .. {prices.index[-1].date()})")

    print("Loading fundamentals and earnings...")
    fundamentals = load_fundamentals(list(prices.columns), cfg["data_cache"],
                                     refresh=args.refresh_data)
    earnings = load_earnings(list(prices.columns), cfg["data_cache"],
                             refresh=args.refresh_data)
    value_score, value_rank = build_value_scores(prices, fundamentals,
                                                 val["reporting_lag_days"])
    fund_factors = build_fundamental_factors(prices, fundamentals, val["reporting_lag_days"])
    earn_factors = build_earnings_factors(prices, earnings)

    print("Building features and events...")
    feats = build_features(prices, value_score, value_rank, lab["vol_span"],
                           fund_factors, earn_factors)
    events = generate_events(prices, value_rank, sig["fast_sma"], sig["slow_sma"],
                             val["min_value_rank"], lab["event_every"])
    print(f"  {len(events)} primary-signal events")

    print("Applying triple-barrier labels...")
    labeled = label_events(prices, events, feats, lab["vol_span"], lab["pt_mult"],
                           lab["sl_mult"], lab["max_holding_days"])
    if labeled.empty:
        sys.exit("No labeled events. Check data and signal parameters.")
    print(f"  {len(labeled)} labeled events, base hit rate {labeled['label'].mean():.1%}")

    print("Walk-forward meta-model...")
    labeled["prob"] = walk_forward_probabilities(
        labeled, prices.index,
        n_estimators=mdl["n_estimators"], max_depth=mdl["max_depth"],
        retrain_freq=mdl["retrain_freq"], min_train_events=mdl["min_train_events"])
    labeled["size"] = bet_size(labeled["prob"], mdl["min_prob"])
    scored = labeled["prob"].notna().sum()
    print(f"  {scored} events scored out-of-sample, "
          f"{(labeled['size'] > 0).sum()} trades taken")

    print("Simulating portfolio...")
    result = run_backtest(prices, labeled[labeled["size"] > 0],
                          cost_bps=bt["cost_bps"],
                          max_gross_exposure=bt["max_gross_exposure"],
                          max_positions=bt["max_positions"])

    # compare only over the period when the model could actually trade
    live = labeled.loc[labeled["prob"].notna(), "signal_date"]
    start_live = live.min() if not live.empty else result["returns_net"].index[0]
    strat = performance_stats(result["returns_net"].loc[start_live:], "strategy")
    bench = performance_stats(bench_rets.loc[start_live:], "benchmark")
    importance = final_feature_importance(labeled, **mdl)
    print(f"\nLive evaluation window: {start_live.date()} .. {prices.index[-1].date()}")
    print_report(strat, bench, trade_stats(labeled), importance)

    report_dir = Path(cfg["report_dir"])
    labeled.to_csv(report_dir / "trades.csv", index=False)
    plot_results(result["equity"].loc[start_live:], bench_rets, result["exposure"],
                 report_dir / "equity_curve.png")
    print(f"\nSaved {report_dir / 'trades.csv'} and {report_dir / 'equity_curve.png'}")


if __name__ == "__main__":
    main()

"""Forward-looking view of your CURRENT holdings.

1. Meta-model view: scores each holding's present technical+valuation setup
   using a random forest trained on every resolved triple-barrier event.
2. Monte Carlo: block-bootstraps joint daily returns to project the
   portfolio over the next N trading days (distribution, VaR, fan chart).
3. Risk decomposition: each holding's share of portfolio variance.

Usage:
    python portfolio_outlook.py [--portfolio portfolio.csv] [--horizon 21]
                                [--sims 5000] [--config config.yaml]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from quantfolio.config import load_config
from quantfolio.data import load_earnings, load_fundamentals, load_prices
from quantfolio.features import build_features
from quantfolio.fundamentals import build_earnings_factors, build_fundamental_factors
from quantfolio.labeling import generate_events, label_events
from quantfolio.outlook import (
    load_portfolio,
    model_view,
    plot_outlook,
    simulate_portfolio,
)
from quantfolio.valuation import build_value_scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--portfolio", default="portfolio.csv")
    ap.add_argument("--horizon", type=int, default=21, help="trading days ahead")
    ap.add_argument("--sims", type=int, default=5000)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--refresh-data", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    lab, sig, val, mdl = cfg["labeling"], cfg["signals"], cfg["valuation"], cfg["model"]

    holdings_tickers = pd.read_csv(args.portfolio)["ticker"].str.upper().str.strip().tolist()
    # union of config universe and holdings so cross-sectional value ranks
    # stay meaningful and the model trains on the full event history
    tickers = sorted(set(cfg["universe"]) | set(holdings_tickers))

    print("Loading prices, fundamentals, and earnings...")
    prices = load_prices(tickers, cfg["start"], cfg["end"], cfg["data_cache"],
                         refresh=args.refresh_data)
    fundamentals = load_fundamentals(list(prices.columns), cfg["data_cache"],
                                     refresh=args.refresh_data)
    earnings = load_earnings(list(prices.columns), cfg["data_cache"],
                             refresh=args.refresh_data)
    value_score, value_rank = build_value_scores(prices, fundamentals,
                                                 val["reporting_lag_days"])
    fund_factors = build_fundamental_factors(prices, fundamentals, val["reporting_lag_days"])
    earn_factors = build_earnings_factors(prices, earnings)
    feats = build_features(prices, value_score, value_rank, lab["vol_span"],
                           fund_factors, earn_factors)

    print("Training meta-model on full event history...")
    events = generate_events(prices, value_rank, sig["fast_sma"], sig["slow_sma"],
                             val["min_value_rank"], lab["event_every"])
    labeled = label_events(prices, events, feats, lab["vol_span"], lab["pt_mult"],
                           lab["sl_mult"], lab["max_holding_days"])
    if labeled.empty:
        sys.exit("No labeled events to train on.")

    weights = load_portfolio(args.portfolio, prices)
    asof = prices.index[-1].date()
    print(f"\nPortfolio as of {asof}  ({len(weights)} holdings)")

    view = model_view(weights, prices, feats, labeled, sig["fast_sma"],
                      sig["slow_sma"], lab["pt_mult"], lab["sl_mult"],
                      n_estimators=mdl["n_estimators"], max_depth=mdl["max_depth"])

    sim = simulate_portfolio(prices, weights, horizon=args.horizon, n_sims=args.sims)
    view["risk_share"] = sim["risk_contrib"]

    print("\n=== Holdings: model view (horizon = barrier window, ~20d) ===")
    out = view.copy()
    for col, fmt in [("weight", "{:.1%}"), ("value_rank", "{:.2f}"),
                     ("ann_vol", "{:.1%}"), ("mom_63", "{:+.1%}"),
                     ("meta_prob", "{:.2f}"), ("exp_ret_h", "{:+.2%}"),
                     ("risk_share", "{:.1%}")]:
        if col in out.columns:
            out[col] = out[col].map(lambda v, f=fmt: f.format(v) if pd.notna(v) else "n/a")
    print(out.to_string())
    print("note: meta_prob for holdings with trend_on=False is out-of-distribution;")
    print("      the model never traded such setups; treat it as a caution flag.")

    p = sim["percentiles"]
    print(f"\n=== Monte Carlo: next {args.horizon} trading days "
          f"({args.sims} block-bootstrap paths) ===")
    print(f"median return     : {p[50]:+.2%}")
    print(f"25th .. 75th pct  : {p[25]:+.2%} .. {p[75]:+.2%}")
    print(f"5th  .. 95th pct  : {p[5]:+.2%} .. {p[95]:+.2%}")
    print(f"P(loss)           : {sim['prob_loss']:.1%}")
    print(f"VaR 95% / CVaR 95%: {sim['var_95']:.2%} / {sim['cvar_95']:.2%}")
    print(f"portfolio ann vol : {sim['ann_vol']:.1%}")

    report_dir = Path(cfg["report_dir"])
    view.to_csv(report_dir / "outlook_holdings.csv")
    plot_outlook(sim, args.horizon, report_dir / "outlook_fanchart.png")
    print(f"\nSaved {report_dir / 'outlook_holdings.csv'} and "
          f"{report_dir / 'outlook_fanchart.png'}")


if __name__ == "__main__":
    main()

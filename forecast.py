"""Forecast the current portfolio's forward returns (panel LSTM vs regularized
linear) and forward dividends (VAR), with honest, aligned model comparison.

Usage:
    python forecast.py [--portfolio portfolio.csv] [--config config.yaml]
                       [--model both|lstm|linear] [--universe config|sp500]
                       [--device auto|cpu|cuda] [--no-amp]
                       [--horizon 5] [--folds 3] [--epochs 25] [--refresh-data]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from quantfolio import forecast_linear, forecast_lstm
from quantfolio.config import load_config
from quantfolio.data import load_earnings, load_fundamentals, load_prices, load_sp500_universe
from quantfolio.features import FEATURE_NAMES, build_features
from quantfolio.forecast_dividends import forecast_dividends
from quantfolio.forecast_eval import walk_forward
from quantfolio.fundamentals import build_earnings_factors, build_fundamental_factors
from quantfolio.outlook import load_portfolio
from quantfolio.valuation import build_value_scores

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _resolve_device(choice: str):
    if not _HAS_TORCH:
        return None
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        return torch.device("cuda")
    return forecast_lstm.get_device()  # auto


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--portfolio", default="portfolio.csv")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model", choices=["both", "lstm", "linear"], default="both")
    ap.add_argument("--universe", choices=["config", "sp500"], default="config")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--folds", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--refresh-data", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    lab, val = cfg["labeling"], cfg["valuation"]
    fc_cfg = cfg.get("forecast", {})
    horizon = args.horizon or fc_cfg.get("horizon", 5)
    folds = args.folds or fc_cfg.get("folds", 3)
    epochs = args.epochs or fc_cfg.get("epochs", 25)
    seq_len = fc_cfg.get("seq_len", 30)
    use_amp = fc_cfg.get("use_amp", True) and not args.no_amp
    device = _resolve_device(args.device)
    if device is not None:
        print(f"Device: {device.type}  | AMP: {use_amp and device.type == 'cuda'}")

    holdings = pd.read_csv(args.portfolio)["ticker"].str.upper().str.strip().tolist()
    base = (load_sp500_universe(cfg["data_cache"], refresh=args.refresh_data)
            if args.universe == "sp500" else cfg["universe"])
    tickers = sorted(set(base) | set(holdings))
    print(f"Universe: {args.universe} ({len(tickers)} tickers)")

    print("Loading prices, fundamentals, earnings...")
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

    held = [t for t in holdings if t in prices.columns]
    weights = load_portfolio(args.portfolio, prices)
    raw = pd.read_csv(args.portfolio)
    raw["ticker"] = raw["ticker"].str.upper().str.strip()
    shares = raw.set_index("ticker")["shares"] if "shares" in raw.columns else None

    # ----- returns: aligned model comparison -----
    print(f"\nBuilding sequences (horizon={horizon}d, seq_len={seq_len})...")
    data = forecast_lstm.build_sequences(prices, feats, horizon=horizon, seq_len=seq_len)
    print(f"  {len(data['y'])} pooled sequences across {prices.shape[1]} tickers")

    model_fns = {}
    if args.model in ("both", "lstm"):
        model_fns["lstm"] = forecast_lstm.make_lstm_fn(
            data["n_features"], hidden=fc_cfg.get("hidden", 64),
            layers=fc_cfg.get("layers", 2), dropout=fc_cfg.get("dropout", 0.1),
            epochs=epochs, batch=fc_cfg.get("batch", 256), lr=fc_cfg.get("lr", 1e-3),
            patience=fc_cfg.get("patience", 4), device=device, use_amp=use_amp)
    if args.model in ("both", "linear"):
        model_fns["linear"] = forecast_linear.make_linear_fn(
            alpha=fc_cfg.get("alpha", 1.0), model=fc_cfg.get("linear_model", "ridge"),
            l1_ratio=fc_cfg.get("l1_ratio", 0.1))

    print(f"Walk-forward evaluation ({folds} folds): {', '.join(model_fns)} + const...")
    ev = walk_forward(data, model_fns, n_folds=folds, test_years=fc_cfg.get("test_years", 3.0))
    m = ev["metrics"]
    print(f"\n=== Out-of-sample forward {horizon}d-return forecast quality ===")
    print("(IC_xsection is the daily rank correlation a PM cares about; "
          "higher = real signal)")
    disp = m.copy()
    for c in ["RMSE", "IC_pooled", "IC_xsection"]:
        disp[c] = disp[c].map(lambda v: f"{v:.4f}" if pd.notna(v) else "n/a")
    disp["DirAcc"] = disp["DirAcc"].map(lambda v: f"{v:.1%}")
    print(disp.to_string())
    best = _verdict(m)

    # ----- live per-holding forecasts -----
    print("\nTraining final model(s) on all history for live forecast...")
    cols = {"weight": weights.reindex(held)}
    if "lstm" in model_fns:
        cols["lstm_pred"] = forecast_lstm.fit_full_and_forecast(
            data, prices, feats, held, seq_len=seq_len, hidden=fc_cfg.get("hidden", 64),
            layers=fc_cfg.get("layers", 2), dropout=fc_cfg.get("dropout", 0.1),
            epochs=epochs, batch=fc_cfg.get("batch", 256), lr=fc_cfg.get("lr", 1e-3),
            patience=fc_cfg.get("patience", 4), device=device, use_amp=use_amp)
    if "linear" in model_fns:
        cols["linear_pred"] = forecast_linear.fit_full_and_forecast(
            data, prices, feats, held, seq_len=seq_len, alpha=fc_cfg.get("alpha", 1.0),
            model=fc_cfg.get("linear_model", "ridge"), l1_ratio=fc_cfg.get("l1_ratio", 0.1))

    show = pd.DataFrame(cols)
    sort_col = f"{best}_pred" if f"{best}_pred" in show.columns else show.columns[-1]
    show = show.sort_values(sort_col, ascending=False)
    show_fmt = show.copy()
    show_fmt["weight"] = show_fmt["weight"].map(lambda v: f"{v:.1%}" if pd.notna(v) else "n/a")
    for c in [c for c in show.columns if c.endswith("_pred")]:
        show_fmt[c] = show_fmt[c].map(lambda v: f"{v:+.2%}" if pd.notna(v) else "n/a")
    print(f"\n=== Predicted next-{horizon}d return by holding "
          f"(sorted by {best}) ===")
    print(show_fmt.to_string())
    if f"{best}_pred" in show.columns:
        port = float((show["weight"].fillna(0) * show[f"{best}_pred"].fillna(0)).sum())
        print(f"weighted portfolio predicted {horizon}d return ({best}): {port:+.2%}")
    print("CAUTION: treat as a weak tilt signal, not a point forecast -- see IC above.")

    # linear coefficients (interpretability win of the linear model)
    if "linear" in model_fns:
        coef = forecast_linear.coefficients(
            data, FEATURE_NAMES, alpha=fc_cfg.get("alpha", 1.0),
            model=fc_cfg.get("linear_model", "ridge"), l1_ratio=fc_cfg.get("l1_ratio", 0.1))
        print("\n=== Linear model: top standardized coefficients ===")
        for name, v in coef.head(8).items():
            print(f"{name:16} {v:+.4f}")

    # ----- dividends: VAR -----
    print("\nForecasting forward dividends (VAR + Lintner fallback)...")
    dtab, dsum = forecast_dividends(held, shares, prices, cfg["data_cache"],
                                    refresh=args.refresh_data)
    print(f"VAR used: {dsum['var_used']}  (payers with enough history: "
          f"{', '.join(dsum['var_payers']) if dsum['var_payers'] else 'none'})")
    dshow = dtab[["ttm_dps", "fwd_annual_dps", "trailing_yield", "fwd_yield", "method"]].copy()
    dshow["ttm_dps"] = dshow["ttm_dps"].map(lambda v: f"{v:.2f}")
    dshow["fwd_annual_dps"] = dshow["fwd_annual_dps"].map(lambda v: f"{v:.2f}")
    dshow["trailing_yield"] = dshow["trailing_yield"].map(lambda v: f"{v:.2%}" if pd.notna(v) else "n/a")
    dshow["fwd_yield"] = dshow["fwd_yield"].map(lambda v: f"{v:.2%}" if pd.notna(v) else "n/a")
    print(dshow.to_string())
    if "portfolio_fwd_yield" in dsum:
        print(f"\nportfolio forward dividend yield : {dsum['portfolio_fwd_yield']:.2%}")
        print(f"expected annual dividend income  : "
              f"{dsum['fwd_annual_income']:,.2f} (on value {dsum['portfolio_value']:,.2f})")

    # ----- save -----
    rep = Path(cfg["report_dir"])
    m.to_csv(rep / "forecast_returns_metrics.csv")
    show.to_csv(rep / "forecast_returns_holdings.csv")
    dtab.to_csv(rep / "forecast_dividends.csv")
    _plot_scatter(ev, best, rep / "forecast_scatter.png", horizon)
    print(f"\nSaved metrics, holding forecasts, dividends, and scatter to {rep}/")


def _verdict(m: pd.DataFrame) -> str:
    """Print a blunt verdict and return the recommended model name."""
    have = [x for x in ("lstm", "linear") if x in m.index]
    if not have:
        return "const"
    ics = {x: m.loc[x, "IC_xsection"] for x in have}
    best = max(ics, key=lambda k: (ics[k] if np.isfinite(ics[k]) else -1))
    if not np.isfinite(ics[best]) or ics[best] <= 0:
        print("  verdict: no usable cross-sectional signal (IC <= 0). Do not trade on it.")
        return best
    if "lstm" in ics and "linear" in ics:
        if ics["linear"] >= ics["lstm"] + 0.005:
            print("  verdict: the LINEAR model wins -- the LSTM overfits at this "
                  "universe size. Use linear (or widen the universe: --universe sp500).")
        elif ics["lstm"] >= ics["linear"] + 0.005:
            print("  verdict: the LSTM beats the linear model on cross-sectional IC.")
        else:
            print("  verdict: LSTM and linear are about even; prefer linear (simpler).")
            best = "linear"
    print(f"  recommended model for live forecast: {best}")
    return best


def _plot_scatter(ev: dict, best: str, out_path: str | Path, horizon: int) -> None:
    p = ev["preds"].get(best)
    y = ev["oos"].get("y")
    if p is None or len(p) == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(p, y, s=4, alpha=0.2)
    ax.axhline(0, color="gray", lw=0.6)
    ax.axvline(0, color="gray", lw=0.6)
    lim = np.percentile(np.abs(np.concatenate([p, y])), 99)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(f"predicted {horizon}d return ({best})")
    ax.set_ylabel(f"realized {horizon}d return")
    ax.set_title(f"{best} out-of-sample: predicted vs realized")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()

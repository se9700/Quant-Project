from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def performance_stats(returns: pd.Series, label: str = "strategy") -> dict:
    r = returns.dropna()
    if r.empty or (r == 0).all():
        return {"label": label, "n_days": len(r)}
    equity = (1 + r).cumprod()
    years = len(r) / TRADING_DAYS
    cagr = equity.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    vol = r.std() * np.sqrt(TRADING_DAYS)
    downside = r[r < 0].std() * np.sqrt(TRADING_DAYS)
    dd = equity / equity.cummax() - 1
    return {
        "label": label,
        "CAGR": cagr,
        "AnnVol": vol,
        "Sharpe": (r.mean() * TRADING_DAYS) / vol if vol > 0 else np.nan,
        "Sortino": (r.mean() * TRADING_DAYS) / downside if downside > 0 else np.nan,
        "MaxDD": dd.min(),
        "n_days": len(r),
    }


def trade_stats(trades: pd.DataFrame) -> dict:
    taken = trades[trades["size"] > 0]
    if taken.empty:
        return {"n_trades": 0}
    wins = taken["ret"] > 0
    return {
        "n_trades": len(taken),
        "hit_rate": wins.mean(),
        "avg_ret": taken["ret"].mean(),
        "avg_win": taken.loc[wins, "ret"].mean(),
        "avg_loss": taken.loc[~wins, "ret"].mean(),
        "avg_days_held": taken["days_held"].mean(),
    }


def print_report(strat: dict, bench: dict, tstats: dict, importance: pd.Series) -> None:
    def pct(x):
        return f"{x:+.2%}" if isinstance(x, (int, float)) and np.isfinite(x) else "n/a"

    def num(x):
        return f"{x:.2f}" if isinstance(x, (int, float)) and np.isfinite(x) else "n/a"

    print("\n=== Performance (net of costs) ===")
    print(f"{'':18}{'Strategy':>12}{'Benchmark':>12}")
    for key in ["CAGR", "AnnVol", "MaxDD"]:
        print(f"{key:18}{pct(strat.get(key)):>12}{pct(bench.get(key)):>12}")
    for key in ["Sharpe", "Sortino"]:
        print(f"{key:18}{num(strat.get(key)):>12}{num(bench.get(key)):>12}")

    print("\n=== Trades (taken by the meta-model) ===")
    print(f"count        : {tstats.get('n_trades', 0)}")
    if tstats.get("n_trades", 0) > 0:
        print(f"hit rate     : {pct(tstats['hit_rate'])}")
        print(f"avg return   : {pct(tstats['avg_ret'])}")
        print(f"avg win/loss : {pct(tstats['avg_win'])} / {pct(tstats['avg_loss'])}")
        print(f"avg held     : {tstats['avg_days_held']:.1f} days")

    if not importance.empty:
        print("\n=== Meta-model feature importance ===")
        for name, val in importance.items():
            print(f"{name:16} {val:.3f}")


def plot_results(
    equity: pd.Series,
    bench_returns: pd.Series,
    exposure: pd.Series,
    out_path: str | Path,
) -> None:
    bench_eq = (1 + bench_returns.reindex(equity.index).fillna(0)).cumprod()
    dd = equity / equity.cummax() - 1

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1, 1]})
    axes[0].plot(equity.index, equity, label="Strategy (net)", lw=1.4)
    axes[0].plot(bench_eq.index, bench_eq, label="Benchmark", lw=1.2, alpha=0.8)
    axes[0].set_yscale("log")
    axes[0].set_title("Equity curve")
    axes[0].legend()
    axes[1].fill_between(dd.index, dd, 0, color="firebrick", alpha=0.6)
    axes[1].set_title("Drawdown")
    axes[2].plot(exposure.index, exposure, lw=0.8, color="gray")
    axes[2].set_title("Gross exposure")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)

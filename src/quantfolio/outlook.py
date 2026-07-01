from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .features import FEATURE_NAMES

TRADING_DAYS = 252


def load_portfolio(path: str | Path, prices: pd.DataFrame) -> pd.Series:
    """Read holdings and return normalized weights.

    CSV needs a `ticker` column plus either `shares` (converted to weights
    at the latest close) or `weight`.
    """
    df = pd.read_csv(path)
    df["ticker"] = df["ticker"].str.upper().str.strip()
    missing = [t for t in df["ticker"] if t not in prices.columns]
    if missing:
        raise ValueError(f"No price data for holdings: {missing}")
    last = prices.iloc[-1]
    if "shares" in df.columns:
        value = df.set_index("ticker")["shares"] * last[df["ticker"]].values
        weights = value / value.sum()
    elif "weight" in df.columns:
        weights = df.set_index("ticker")["weight"].astype(float)
        weights = weights / weights.sum()
    else:
        raise ValueError("portfolio file needs a 'shares' or 'weight' column")
    return weights


def model_view(
    weights: pd.Series,
    prices: pd.DataFrame,
    feats: dict[str, pd.DataFrame],
    labeled: pd.DataFrame,
    fast_sma: int,
    slow_sma: int,
    pt_mult: float,
    sl_mult: float,
    n_estimators: int = 400,
    max_depth: int = 6,
) -> pd.DataFrame:
    """Score each current holding with the meta-model.

    The model is fit on ALL historically resolved triple-barrier events
    (legitimate here: we are forecasting forward, not backtesting). The
    expected horizon return uses the barrier geometry:
        E[ret] ~ p * pt_mult * vol_to_barrier - (1-p) * sl_mult * vol_to_barrier
    expressed via realized avg win/loss of past trades for realism.

    Caveat shown in the output: the model was trained only on events where
    the trend signal was ON. For holdings with the signal OFF the probability
    is out-of-distribution and should be read as "the setup does not match
    anything the model was trained to like".
    """
    clf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=20,
        class_weight="balanced_subsample", random_state=42, n_jobs=-1,
    )
    clf.fit(labeled[FEATURE_NAMES].to_numpy(), labeled["label"].to_numpy())

    wins = labeled.loc[labeled["ret"] > 0, "ret"]
    losses = labeled.loc[labeled["ret"] <= 0, "ret"]
    avg_win = wins.mean() if not wins.empty else pt_mult * 0.02
    avg_loss = losses.mean() if not losses.empty else -sl_mult * 0.02

    sma_f = prices.rolling(fast_sma).mean().iloc[-1]
    sma_s = prices.rolling(slow_sma).mean().iloc[-1]
    today = prices.index[-1]

    rows = []
    for tk, w in weights.items():
        x = np.array([[feats[name].at[today, tk] for name in FEATURE_NAMES]])
        if not np.isfinite(x).all():
            rows.append({"ticker": tk, "weight": w})
            continue
        p = float(clf.predict_proba(x)[0, 1])
        rows.append({
            "ticker": tk,
            "weight": w,
            "trend_on": bool(sma_f[tk] > sma_s[tk]),
            "value_rank": float(feats["value_rank"].at[today, tk]),
            "ann_vol": float(feats["vol"].at[today, tk]) * np.sqrt(TRADING_DAYS),
            "mom_63": float(feats["mom_63"].at[today, tk]),
            "meta_prob": p,
            "exp_ret_h": p * avg_win + (1 - p) * avg_loss,
        })
    return pd.DataFrame(rows).set_index("ticker")


def simulate_portfolio(
    prices: pd.DataFrame,
    weights: pd.Series,
    horizon: int = 21,
    n_sims: int = 5000,
    block: int = 5,
    lookback_days: int = 1260,
    seed: int = 42,
) -> dict:
    """Stationary block bootstrap of joint daily returns.

    Sampling whole date-blocks keeps the cross-sectional correlation between
    holdings and some short-horizon autocorrelation intact, which a plain
    iid bootstrap or a normal approximation would destroy.
    """
    rets = prices[weights.index].pct_change().dropna().tail(lookback_days)
    R = rets.to_numpy()
    T = len(R)
    w = weights.to_numpy()
    rng = np.random.default_rng(seed)

    n_blocks = int(np.ceil(horizon / block))
    starts = rng.integers(0, T - block, size=(n_sims, n_blocks))
    # gather blocks: paths has shape (n_sims, horizon, n_assets)
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_sims, -1)[:, :horizon]
    paths = R[idx]  # fancy indexing

    port_daily = paths @ w
    cum = np.cumprod(1.0 + port_daily, axis=1)
    final = cum[:, -1] - 1.0

    # risk decomposition from the realized covariance
    cov = np.cov(R, rowvar=False) * TRADING_DAYS
    port_var = float(w @ cov @ w)
    mcr = (cov @ w) * w / port_var if port_var > 0 else np.zeros_like(w)

    return {
        "final_returns": final,
        "paths_cum": cum,
        "percentiles": {p: float(np.percentile(final, p)) for p in (5, 25, 50, 75, 95)},
        "prob_loss": float((final < 0).mean()),
        "var_95": float(-np.percentile(final, 5)),
        "cvar_95": float(-final[final <= np.percentile(final, 5)].mean()),
        "ann_vol": float(np.sqrt(port_var)),
        "risk_contrib": pd.Series(mcr, index=weights.index),
    }


def plot_outlook(sim: dict, horizon: int, out_path: str | Path) -> None:
    cum = sim["paths_cum"]
    days = np.arange(1, cum.shape[1] + 1)
    bands = [(5, 95, 0.15), (25, 75, 0.30)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [2, 1]})
    for lo, hi, alpha in bands:
        axes[0].fill_between(days, np.percentile(cum, lo, axis=0),
                             np.percentile(cum, hi, axis=0),
                             color="steelblue", alpha=alpha,
                             label=f"{lo}–{hi} pct")
    axes[0].plot(days, np.percentile(cum, 50, axis=0), color="navy", lw=1.5, label="median")
    axes[0].axhline(1.0, color="gray", lw=0.8, ls="--")
    axes[0].set_title(f"Simulated portfolio value, next {horizon} trading days")
    axes[0].set_xlabel("trading days ahead")
    axes[0].legend()

    axes[1].hist(sim["final_returns"], bins=60, color="steelblue", alpha=0.8)
    axes[1].axvline(0, color="gray", lw=0.8, ls="--")
    axes[1].axvline(-sim["var_95"], color="firebrick", lw=1.2, label=f"VaR95 {-sim['var_95']:+.1%}")
    axes[1].set_title("Distribution of horizon return")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)

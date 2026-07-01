from __future__ import annotations

import numpy as np
import pandas as pd


def build_value_scores(
    prices: pd.DataFrame,
    fundamentals: dict[str, pd.DataFrame],
    reporting_lag_days: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily cross-sectional value scores from quarterly statements.

    Returns (value_score, value_rank), both shaped like `prices`.
    value_score = mean z-score of TTM earnings yield and book-to-price.
    value_rank  = cross-sectional percentile (0..1), 0.5 where unknown so
    that periods without fundamentals trade on technicals alone.

    Fundamentals become usable only `reporting_lag_days` after the fiscal
    period end, to avoid using numbers before they were public.
    """
    ey = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    bp = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)

    for tk in prices.columns:
        f = fundamentals.get(tk)
        if f is None or f.empty or "net_income" not in f.columns:
            continue
        f = f.sort_index()
        ttm_ni = f["net_income"].rolling(4).sum()
        equity = f["equity"] if "equity" in f.columns else pd.Series(np.nan, index=f.index)
        shares = f["shares"] if "shares" in f.columns else pd.Series(np.nan, index=f.index)

        # shift availability: numbers known only after the reporting lag
        avail = f.index + pd.Timedelta(days=reporting_lag_days)
        ttm_ni.index = avail
        equity.index = avail
        shares.index = avail

        ttm_d = ttm_ni.reindex(prices.index, method="ffill")
        eq_d = equity.reindex(prices.index, method="ffill")
        sh_d = shares.reindex(prices.index, method="ffill")

        mcap = sh_d * prices[tk]
        ey[tk] = ttm_d / mcap
        bp[tk] = eq_d / mcap

    z_ey = _xs_zscore(ey)
    z_bp = _xs_zscore(bp)
    value_score = pd.concat([z_ey, z_bp]).groupby(level=0).mean()
    value_score = value_score.reindex(prices.index)
    value_rank = value_score.rank(axis=1, pct=True)
    value_rank = value_rank.fillna(0.5)
    return value_score, value_rank


def _xs_zscore(df: pd.DataFrame, clip: float = 3.0) -> pd.DataFrame:
    mu = df.mean(axis=1)
    sd = df.std(axis=1)
    z = df.sub(mu, axis=0).div(sd.replace(0, np.nan), axis=0)
    return z.clip(-clip, clip)

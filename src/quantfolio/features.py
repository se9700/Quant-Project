from __future__ import annotations

import numpy as np
import pandas as pd


def get_daily_vol(prices: pd.DataFrame, span: int = 50) -> pd.DataFrame:
    """EWMA estimate of daily return volatility, per ticker."""
    return prices.pct_change().ewm(span=span).std()


def build_features(
    prices: pd.DataFrame,
    value_score: pd.DataFrame,
    value_rank: pd.DataFrame,
    vol_span: int = 50,
    fund_factors: dict[str, pd.DataFrame] | None = None,
    earn_factors: dict[str, pd.DataFrame] | None = None,
) -> dict[str, pd.DataFrame]:
    """Per-date, per-ticker features fed to the meta-model.

    Everything here uses information available at the close of the same day,
    which is when events are triggered. `fund_factors` / `earn_factors` are the
    point-in-time panels from fundamentals.py; when omitted those features fall
    back to neutral values so the feature vector keeps a fixed length.
    """
    rets = prices.pct_change()
    vol = get_daily_vol(prices, vol_span)
    sma50 = prices.rolling(50).mean()
    sma200 = prices.rolling(200).mean()
    mom_63 = prices.pct_change(63)

    feats = {
        "vol": vol,
        "mom_21": prices.pct_change(21),
        "mom_63": mom_63,
        "mom_126": prices.pct_change(126),
        "rsi_14": _rsi(prices, 14),
        "dist_sma50": prices / sma50 - 1.0,
        "dist_sma200": prices / sma200 - 1.0,
        "trend_strength": sma50 / sma200 - 1.0,
        "ret_1": rets,
        "value_score": value_score.fillna(0.0),
        "value_rank": value_rank,
        # relative (cross-sectional) momentum: own 63d momentum vs the
        # universe average that day -> isolates stock-specific strength
        "rel_mom_63": mom_63.sub(mom_63.mean(axis=1), axis=0),
    }

    # statement-based factors (neutral where statements are missing)
    ff = fund_factors or {}
    nm = _align(ff.get("net_margin"), prices)
    feats["earn_growth_yoy"] = _align(ff.get("earn_growth_yoy"), prices).fillna(0.0)
    feats["net_margin"] = nm.fillna(0.0)
    feats["roe"] = _align(ff.get("roe"), prices).fillna(0.0)
    # cross-sectional quality rank: where each name's margin sits vs the
    # universe that day (0.5 where unknown) -> a relative quality signal
    feats["rank_quality"] = nm.rank(axis=1, pct=True).fillna(0.5)

    # earnings surprise / post-announcement drift (neutral 0 where unknown)
    ef = earn_factors or {}
    feats["earn_surprise"] = _align(ef.get("earn_surprise"), prices).fillna(0.0)
    return feats


def _align(panel: pd.DataFrame | None, prices: pd.DataFrame) -> pd.DataFrame:
    if panel is None:
        return pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    return panel.reindex(index=prices.index, columns=prices.columns)


FEATURE_NAMES = [
    "vol", "mom_21", "mom_63", "mom_126", "rsi_14",
    "dist_sma50", "dist_sma200", "trend_strength", "ret_1",
    "value_score", "value_rank", "rel_mom_63",
    "earn_growth_yoy", "net_margin", "roe", "rank_quality",
    "earn_surprise",
]


def features_at(feats: dict[str, pd.DataFrame], date, ticker: str) -> list[float]:
    return [float(feats[name].at[date, ticker]) for name in FEATURE_NAMES]


def _rsi(prices: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    delta = prices.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

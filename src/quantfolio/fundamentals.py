"""Point-in-time fundamental and earnings factors.

Everything here is shifted by the reporting lag (fundamentals) or anchored to
the announcement date (earnings) so a feature is only ever known when it would
actually have been public -- the same discipline used in valuation.py.

Factors returned as daily panels (index = trading days, columns = tickers),
left as NaN where unknown; features.py decides the neutral fill.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_fundamental_factors(
    prices: pd.DataFrame,
    fundamentals: dict[str, pd.DataFrame],
    reporting_lag_days: int = 60,
) -> dict[str, pd.DataFrame]:
    """Statement-based factors, made available `reporting_lag_days` after the
    period end and forward-filled onto trading days:

      earn_growth_yoy : single-quarter net-income YoY growth (momentum)
      net_margin      : net income / revenue (profitability level)
      roe             : annualized return on equity (quality level)

    Design note: yfinance's free tier currently exposes only ~5 quarterly
    statements per ticker. YoY-of-TTM growth needs 8 quarters and is therefore
    impossible here, so growth uses single-quarter YoY (needs 5), and the
    margin/ROE *levels* (need 1) are used instead of their changes so the
    panels are actually populated. These cover only the recent years where
    statements exist; the earnings-surprise factor carries the long history.
    """
    cols = prices.columns
    eg = pd.DataFrame(index=prices.index, columns=cols, dtype=float)
    nm = pd.DataFrame(index=prices.index, columns=cols, dtype=float)
    roe = pd.DataFrame(index=prices.index, columns=cols, dtype=float)

    for tk in cols:
        f = fundamentals.get(tk)
        if f is None or f.empty or "net_income" not in f.columns:
            continue
        f = f.sort_index()
        ni = f["net_income"]

        # single-quarter YoY growth (clipped: pct_change explodes near zero base)
        earn_growth = ni.pct_change(4).clip(-2.0, 2.0)

        if "revenue" in f.columns:
            net_margin = (ni / f["revenue"].replace(0, np.nan)).clip(-1.0, 1.0)
        else:
            net_margin = pd.Series(np.nan, index=f.index)

        if "equity" in f.columns:
            roe_q = (ni * 4 / f["equity"].replace(0, np.nan)).clip(-2.0, 2.0)
        else:
            roe_q = pd.Series(np.nan, index=f.index)

        avail = f.index + pd.Timedelta(days=reporting_lag_days)
        for series, panel in ((earn_growth, eg), (net_margin, nm), (roe_q, roe)):
            s = series.copy()
            s.index = avail
            panel[tk] = s.reindex(prices.index, method="ffill")

    return {"earn_growth_yoy": eg, "net_margin": nm, "roe": roe}


def build_earnings_factors(
    prices: pd.DataFrame,
    earnings: dict[str, pd.DataFrame],
    drift_window_days: int = 90,
) -> dict[str, pd.DataFrame]:
    """Earnings-surprise factor with post-announcement-drift decay.

    The most recent surprise becomes active the day AFTER the announcement
    (avoids same-day lookahead) and decays linearly to zero over
    `drift_window_days`, capturing post-earnings drift while not pretending a
    stale surprise still matters a year later.
    """
    surprise = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)

    for tk in prices.columns:
        e = earnings.get(tk)
        if e is None or e.empty or "surprise_pct" not in e.columns:
            continue
        s = e["surprise_pct"].dropna() / 100.0
        if s.empty:
            continue
        s = s.clip(-1.0, 1.0).sort_index()
        avail = s.index + pd.Timedelta(days=1)  # known next day
        edf = pd.DataFrame({"surprise": s.values, "edate": avail}, index=avail).sort_index()
        edf = edf[~edf.index.duplicated(keep="last")]
        daily = edf.reindex(prices.index, method="ffill")
        days_since = (prices.index - pd.DatetimeIndex(daily["edate"])).days
        decay = np.clip(1.0 - days_since / drift_window_days, 0.0, 1.0)
        surprise[tk] = daily["surprise"].to_numpy() * decay

    return {"earn_surprise": surprise}

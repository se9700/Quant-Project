"""Forward dividend forecasting for the current portfolio.

Method, and an honest note about it: dividends are sticky and smoothed
(Lintner) -- a firm's own dividend history explains most of its next
dividend, and cross-firm dynamics add little. A VAR is therefore a slightly
heavy tool here, and it is statistically fragile on the ~12-20 quarterly
points yfinance exposes. Two design decisions keep it from blowing up:

  * The VAR is fit on the trailing-12-month DPS series (a rolling 4-quarter
    sum), not raw quarterly dividends. Single quarters are lumpy -- annual
    payers and ETF distributions leave many $0 quarters whose logs explode a
    VAR. The TTM series is far smoother and closer to stationary.
  * VAR forecasts are clipped to a dividend-plausible band around the
    trailing level (dividends rarely move more than +-60% in a year). This
    catches the explosive extrapolation that short, noisy samples produce.

Payers without enough history (or non-payers) fall back to a Lintner-style
estimate: trailing-year DPS grown by its clipped median YoY growth (0 for
non-payers). Each holding's output states which method produced it.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

try:
    from statsmodels.tsa.api import VAR
except ImportError as e:  # pragma: no cover
    raise ImportError("forecast_dividends needs statsmodels") from e

MIN_VAR_QUARTERS = 16   # need a reasonable joint sample to fit a VAR
GROWTH_CLIP = 0.25      # cap implied annual dividend growth at +-25% (Lintner)
SAFETY_BAND = 0.60      # clip ANY forward forecast to +-60% of trailing year


def load_dividends(
    tickers: list[str], cache_dir: str | Path, refresh: bool = False
) -> dict[str, pd.Series]:
    """Per-share cash dividends (and ETF distributions) by ex-date."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    out: dict[str, pd.Series] = {}
    for tk in tickers:
        path = cache / f"{tk}_dividends.csv"
        if path.exists() and not refresh:
            s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
        else:
            try:
                s = yf.Ticker(tk).dividends
            except Exception as e:
                print(f"  warning: dividends failed for {tk}: {e}")
                s = pd.Series(dtype=float)
            if s is not None and not s.empty:
                s.index = pd.to_datetime(s.index).tz_localize(None)
                s.to_frame("dividend").to_csv(path)
            time.sleep(0.1)
        out[tk] = s if s is not None else pd.Series(dtype=float)
    return out


def quarterly_dps(divs: pd.Series) -> pd.Series:
    """Quarterly per-share dividend over the last ~12 years."""
    if divs is None or divs.empty:
        return pd.Series(dtype=float)
    q = divs.resample("QE").sum()
    return q[q.index >= (q.index.max() - pd.Timedelta(days=365 * 12))]


def ttm_dps(dps_q: pd.Series) -> pd.Series:
    """Trailing-12-month DPS at each quarter (rolling 4-quarter sum)."""
    if dps_q.empty:
        return pd.Series(dtype=float)
    return dps_q.rolling(4).sum().dropna()


def _lintner_forecast(ttm_series: pd.Series) -> float:
    """Forward annual DPS from trailing year x clipped median YoY growth."""
    if ttm_series.empty or ttm_series.iloc[-1] == 0:
        return 0.0
    ttm = float(ttm_series.iloc[-1])
    yoy = ttm_series.pct_change(4).dropna()
    g = float(np.clip(yoy.tail(8).median(), -GROWTH_CLIP, GROWTH_CLIP)) if not yoy.empty else 0.0
    return ttm * (1 + g)


def forecast_dividends(
    holdings: list[str],
    shares: pd.Series | None,
    prices: pd.DataFrame,
    cache_dir: str | Path,
    refresh: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Return per-holding forward-annual dividend forecast + portfolio roll-up."""
    divs = load_dividends(holdings, cache_dir, refresh=refresh)
    dps = {tk: quarterly_dps(divs.get(tk, pd.Series(dtype=float))) for tk in holdings}
    ttm = {tk: ttm_dps(dps[tk]) for tk in holdings}

    var_payers = [
        tk for tk in holdings
        if len(ttm[tk]) >= MIN_VAR_QUARTERS and ttm[tk].iloc[-1] > 0
    ]
    var_annual, var_used = ({}, False)
    if var_payers:
        var_annual, var_used = _fit_var(ttm, var_payers)

    last_px = prices.iloc[-1]
    rows = []
    for tk in holdings:
        trailing = float(ttm[tk].iloc[-1]) if len(ttm[tk]) else 0.0
        if tk in var_annual and np.isfinite(var_annual[tk]):
            fwd_annual = _clip_to_band(var_annual[tk], trailing)
            method = "VAR" if abs(fwd_annual - var_annual[tk]) < 1e-9 else "VAR (clipped)"
        else:
            fwd_annual, method = _lintner_forecast(ttm[tk]), "Lintner/TTM"
        px = float(last_px.get(tk, np.nan))
        rows.append({
            "ticker": tk,
            "ttm_dps": trailing,
            "fwd_annual_dps": float(fwd_annual),
            "price": px,
            "trailing_yield": trailing / px if px and np.isfinite(px) else np.nan,
            "fwd_yield": fwd_annual / px if px and np.isfinite(px) else np.nan,
            "method": method,
        })
    table = pd.DataFrame(rows).set_index("ticker")

    summary: dict = {"var_used": var_used, "var_payers": var_payers}
    if shares is not None:
        table["shares"] = shares.reindex(table.index)
        table["fwd_income"] = table["fwd_annual_dps"] * table["shares"]
        table["mkt_value"] = table["price"] * table["shares"]
        pv = table["mkt_value"].sum()
        summary["portfolio_value"] = float(pv)
        summary["fwd_annual_income"] = float(table["fwd_income"].sum())
        summary["portfolio_fwd_yield"] = float(table["fwd_income"].sum() / pv) if pv else np.nan
    return table, summary


def _clip_to_band(forecast: float, trailing: float) -> float:
    if trailing <= 0:
        return max(forecast, 0.0)
    lo, hi = trailing * (1 - SAFETY_BAND), trailing * (1 + SAFETY_BAND)
    return float(np.clip(forecast, lo, hi))


def _fit_var(ttm: dict[str, pd.Series], payers: list[str]) -> tuple[dict, bool]:
    """Fit a VAR on log trailing-year DPS over the payers' common quarters
    and forecast 4 quarters ahead. Returns {} on any numerical failure so the
    caller can fall back to Lintner."""
    frame = pd.DataFrame({tk: ttm[tk] for tk in payers}).dropna()
    if len(frame) < MIN_VAR_QUARTERS:
        return {}, False
    logf = np.log(frame.clip(lower=1e-6))
    # VAR cannot include constant series (e.g. a flat fixed dividend);
    # leave those to the Lintner fallback so VAR fits on what varies.
    nun = logf.nunique()
    varying = nun[nun > 2].index.tolist()
    if len(varying) < 2:
        return {}, False
    logv = logf[varying]
    k = logv.shape[1]
    if len(logv) < k + 6:
        return {}, False
    try:
        # lag 1 only: short samples cannot support more without overfitting
        res = VAR(logv).fit(maxlags=1, ic=None)
        fc = res.forecast(logv.values[-res.k_ar:], steps=4)
        # the 4-quarters-ahead TTM value IS the forward annual run-rate
        annual = np.exp(fc[-1])
        return {tk: float(annual[i]) for i, tk in enumerate(varying)}, True
    except Exception as e:
        print(f"  warning: VAR fit failed ({e}); using Lintner fallback")
        return {}, False

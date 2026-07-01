from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import yfinance as yf


def load_prices(
    tickers: list[str],
    start: str,
    end: str | None,
    cache_dir: str | Path,
    refresh: bool = False,
) -> pd.DataFrame:
    """Adjusted close prices, one column per ticker, cached to CSV per ticker."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    series = {}
    for tk in tickers:
        path = cache / f"{tk}_prices.csv"
        if path.exists() and not refresh:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
        else:
            df = yf.download(tk, start=start, end=end, auto_adjust=True, progress=False)
            if df.empty:
                print(f"  warning: no price data for {tk}, skipping")
                continue
            # yfinance >=0.2.40 returns MultiIndex columns even for one ticker
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.to_csv(path)
            time.sleep(0.2)  # be polite to Yahoo
        if "Close" in df.columns:
            series[tk] = df["Close"]
    prices = pd.DataFrame(series).sort_index()
    prices.index = pd.to_datetime(prices.index)
    return prices


def load_fundamentals(
    tickers: list[str],
    cache_dir: str | Path,
    refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Quarterly fundamentals per ticker: net_income, revenue, equity, shares.

    Indexed by fiscal period end date (ascending). yfinance only exposes
    roughly the last 4-5 years of quarterly statements; before that the
    valuation score falls back to neutral.
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    out: dict[str, pd.DataFrame] = {}
    for tk in tickers:
        path = cache / f"{tk}_fundamentals.csv"
        if path.exists() and not refresh:
            out[tk] = pd.read_csv(path, index_col=0, parse_dates=True)
            continue
        try:
            t = yf.Ticker(tk)
            inc = t.quarterly_income_stmt
            bs = t.quarterly_balance_sheet
            df = _extract_fundamentals(inc, bs, t)
        except Exception as e:  # yfinance raises a zoo of exceptions
            print(f"  warning: fundamentals failed for {tk}: {e}")
            df = pd.DataFrame(columns=["net_income", "equity", "shares"])
        if not df.empty:
            df.to_csv(path)
        out[tk] = df
        time.sleep(0.2)
    return out


def _first_row(stmt: pd.DataFrame, names: list[str]) -> pd.Series | None:
    for name in names:
        if name in stmt.index:
            return stmt.loc[name]
    return None


def _extract_fundamentals(inc: pd.DataFrame, bs: pd.DataFrame, t: yf.Ticker) -> pd.DataFrame:
    if inc is None or inc.empty:
        return pd.DataFrame(columns=["net_income", "revenue", "equity", "shares"])
    ni = _first_row(inc, ["Net Income", "Net Income Common Stockholders"])
    rev = _first_row(inc, ["Total Revenue", "Operating Revenue", "Total Revenues"])
    eq = _first_row(bs, ["Stockholders Equity", "Common Stock Equity"]) if bs is not None else None
    sh = _first_row(bs, ["Ordinary Shares Number", "Share Issued"]) if bs is not None else None
    df = pd.DataFrame({"net_income": ni})
    if rev is not None:
        df["revenue"] = rev
    if eq is not None:
        df["equity"] = eq
    if sh is not None:
        df["shares"] = sh
    if "shares" not in df.columns or df["shares"].isna().all():
        # fall back to today's share count applied to all history (approximation)
        shares_now = None
        try:
            shares_now = t.fast_info.get("shares")
        except Exception:
            pass
        df["shares"] = shares_now
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df.astype(float)


def load_sp500_universe(cache_dir: str | Path, refresh: bool = False) -> list[str]:
    """Current S&P 500 constituents (scraped from Wikipedia, cached).

    NOTE: this is the *current* membership applied to history -> survivorship
    bias, same caveat as the hand-picked universe. For a forecasting research
    panel that is acceptable; document it when presenting results. A
    point-in-time constituents source would remove the bias.
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "sp500_universe.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path)["ticker"].tolist()
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    tickers = tables[0]["Symbol"].astype(str).str.upper().str.replace(".", "-", regex=False)
    tickers = sorted(tickers.tolist())
    pd.DataFrame({"ticker": tickers}).to_csv(path, index=False)
    return tickers


def load_earnings(
    tickers: list[str],
    cache_dir: str | Path,
    refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Reported-vs-estimate earnings history per ticker, indexed by earnings
    date (tz-naive). Columns: eps_estimate, eps_reported, surprise_pct.

    Used to build a point-in-time earnings-surprise / post-announcement-drift
    feature: the surprise is known at the announcement, so it is a legitimate
    stand-in for "earnings-revision direction" (which yfinance only exposes as
    a non-historical snapshot that would leak future information).
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    out: dict[str, pd.DataFrame] = {}
    cols = ["eps_estimate", "eps_reported", "surprise_pct"]
    for tk in tickers:
        path = cache / f"{tk}_earnings.csv"
        if path.exists() and not refresh:
            out[tk] = pd.read_csv(path, index_col=0, parse_dates=True)
            continue
        try:
            raw = yf.Ticker(tk).get_earnings_dates(limit=40)
            df = _extract_earnings(raw)
        except Exception as e:
            print(f"  warning: earnings dates failed for {tk}: {e}")
            df = pd.DataFrame(columns=cols)
        if not df.empty:
            df.to_csv(path)
        out[tk] = df
        time.sleep(0.2)
    return out


def _extract_earnings(raw: pd.DataFrame) -> pd.DataFrame:
    cols = ["eps_estimate", "eps_reported", "surprise_pct"]
    if raw is None or raw.empty:
        return pd.DataFrame(columns=cols)
    est = _first_col(raw, ["EPS Estimate"])
    rep = _first_col(raw, ["Reported EPS"])
    sur = _first_col(raw, ["Surprise(%)", "Surprise (%)"])
    df = pd.DataFrame({"eps_estimate": est, "eps_reported": rep, "surprise_pct": sur})
    # tz-naive UTC index; keep only announced quarters (reported EPS present)
    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
    df = df[df["eps_reported"].notna()].sort_index()
    return df[~df.index.duplicated(keep="last")]


def _first_col(df: pd.DataFrame, names: list[str]) -> pd.Series | None:
    for name in names:
        if name in df.columns:
            return df[name]
    return None

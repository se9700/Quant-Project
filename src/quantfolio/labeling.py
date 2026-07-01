from __future__ import annotations

import numpy as np
import pandas as pd

from .features import FEATURE_NAMES, features_at, get_daily_vol


def generate_events(
    prices: pd.DataFrame,
    value_rank: pd.DataFrame,
    fast_sma: int,
    slow_sma: int,
    min_value_rank: float,
    event_every: int,
) -> list[tuple[pd.Timestamp, str]]:
    """Primary model: long when fast SMA > slow SMA and the stock is not
    expensive cross-sectionally. Emits an event at signal onset and then
    every `event_every` trading days while the signal stays on.
    """
    sma_f = prices.rolling(fast_sma).mean()
    sma_s = prices.rolling(slow_sma).mean()
    signal_on = (sma_f > sma_s) & (value_rank >= min_value_rank)

    events: list[tuple[pd.Timestamp, str]] = []
    for tk in prices.columns:
        on = signal_on[tk]
        counter = event_every  # fire immediately at first onset
        prev = False
        for date, flag in on.items():
            if flag and (not prev or counter >= event_every):
                events.append((date, tk))
                counter = 0
            else:
                counter += 1
            prev = bool(flag)
    events.sort(key=lambda e: e[0])
    return events


def label_events(
    prices: pd.DataFrame,
    events: list[tuple[pd.Timestamp, str]],
    feats: dict[str, pd.DataFrame],
    vol_span: int,
    pt_mult: float,
    sl_mult: float,
    max_holding_days: int,
) -> pd.DataFrame:
    """Apply the triple-barrier method to each event.

    Entry is the close of the day AFTER the signal (no same-bar execution).
    Barriers: +pt_mult*vol (profit take), -sl_mult*vol (stop loss), and a
    vertical barrier after max_holding_days. Label is 1 if the trade's exit
    return is positive — this is what the meta-model learns to predict.
    """
    vol = get_daily_vol(prices, vol_span)
    rows = []
    for signal_date, tk in events:
        close = prices[tk].dropna()
        if signal_date not in close.index:
            continue
        loc = close.index.get_loc(signal_date)
        if loc + 1 >= len(close):
            continue  # cannot enter, no next bar
        entry_date = close.index[loc + 1]
        v = vol.at[signal_date, tk]
        if not np.isfinite(v) or v <= 0:
            continue
        path = close.iloc[loc + 1 : loc + 2 + max_holding_days]
        if len(path) < 2:
            continue
        rets = path / path.iloc[0] - 1.0
        pt, sl = pt_mult * v, -sl_mult * v
        exit_date = path.index[-1]
        for d, r in rets.iloc[1:].items():
            if r >= pt or r <= sl:
                exit_date = d
                break
        exit_ret = float(rets.loc[exit_date])
        try:
            x = features_at(feats, signal_date, tk)
        except (KeyError, ValueError):
            continue
        if not all(np.isfinite(x)):
            continue
        rows.append(
            {
                "signal_date": signal_date,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "ticker": tk,
                "ret": exit_ret,
                "label": int(exit_ret > 0),
                "days_held": int((rets.index.get_loc(exit_date))),
                **dict(zip(FEATURE_NAMES, x)),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("signal_date").reset_index(drop=True)
    return df

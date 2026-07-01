from __future__ import annotations

import numpy as np
import pandas as pd


def run_backtest(
    prices: pd.DataFrame,
    trades: pd.DataFrame,
    cost_bps: float = 10.0,
    max_gross_exposure: float = 1.0,
    max_positions: int = 10,
) -> dict:
    """Turn sized trades into a daily portfolio equity curve.

    A trade holds its position from the entry close until the exit close,
    so it earns returns on the days (entry, exit]. Weights are capped at
    `max_positions` names and scaled so gross exposure never exceeds
    `max_gross_exposure`. Costs are charged on daily turnover.
    """
    rets = prices.pct_change().fillna(0.0)
    raw = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    for t in trades.itertuples():
        # weight active from entry close until the bar before exit close
        idx = prices.index
        start = idx.searchsorted(t.entry_date)
        end = idx.searchsorted(t.exit_date)  # exclusive: last held bar is exit-1
        if end <= start:
            continue
        raw.iloc[start:end, raw.columns.get_loc(t.ticker)] += t.size

    raw = raw.clip(upper=1.0)  # cap any single name at 100% of one unit

    # keep only the top-N conviction names each day
    if max_positions and max_positions < raw.shape[1]:
        rank = raw.rank(axis=1, ascending=False, method="first")
        raw = raw.where(rank <= max_positions, 0.0)

    gross = raw.sum(axis=1)
    scale = (max_gross_exposure / gross.replace(0, np.nan)).clip(upper=1.0).fillna(0.0)
    weights = raw.mul(scale, axis=0)

    port_gross = (weights.shift(1) * rets).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    costs = turnover.shift(1).fillna(0.0) * cost_bps / 1e4
    port_net = port_gross - costs

    equity = (1.0 + port_net).cumprod()
    return {
        "weights": weights,
        "returns_gross": port_gross,
        "returns_net": port_net,
        "turnover": turnover,
        "equity": equity,
        "exposure": weights.sum(axis=1),
    }

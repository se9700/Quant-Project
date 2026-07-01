"""First-class regularized-linear return forecaster.

On a small universe this consistently beats the LSTM (it regularizes instead of
overfitting), so it is a real model here, not just a baseline. It uses the same
pooled feature sequences as the LSTM but reads only the last timestep -- a
cross-sectional linear model on the current feature vector.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge

from .forecast_eval import standardize
from .forecast_lstm import current_sequences


def _make_estimator(alpha: float, model: str, l1_ratio: float):
    if model == "elasticnet":
        return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
    return Ridge(alpha=alpha)


def make_linear_fn(alpha: float = 1.0, model: str = "ridge", l1_ratio: float = 0.1):
    """Build a model fn for forecast_eval.walk_forward."""
    def fn(Xtr, ytr_s, Xte, seed):
        est = _make_estimator(alpha, model, l1_ratio)
        est.fit(Xtr[:, -1, :], ytr_s)
        return est.predict(Xte[:, -1, :]).astype(np.float32)
    return fn


def fit_full_and_forecast(
    data: dict,
    prices: pd.DataFrame,
    feats: dict[str, pd.DataFrame],
    holdings: list[str],
    seq_len: int = 30,
    alpha: float = 1.0,
    model: str = "ridge",
    l1_ratio: float = 0.1,
) -> pd.Series:
    """Train on ALL history, predict each holding's next h-day return."""
    X, y = data["X"], data["y"]
    mu, sd = standardize(X)
    ymu, ysd = float(y.mean()), float(y.std() or 1.0)
    est = _make_estimator(alpha, model, l1_ratio)
    est.fit(((X - mu) / sd)[:, -1, :], (y - ymu) / ysd)

    Xc, ok = current_sequences(prices, feats, holdings, seq_len)
    if len(Xc) == 0:
        return pd.Series(dtype=float)
    pred = est.predict(((Xc - mu) / sd)[:, -1, :]) * ysd + ymu
    return pd.Series(pred, index=ok)


def coefficients(
    data: dict,
    feature_names: list[str],
    alpha: float = 1.0,
    model: str = "ridge",
    l1_ratio: float = 0.1,
) -> pd.Series:
    """Standardized linear coefficients (which features the model leans on)."""
    X, y = data["X"], data["y"]
    mu, sd = standardize(X)
    ymu, ysd = float(y.mean()), float(y.std() or 1.0)
    est = _make_estimator(alpha, model, l1_ratio)
    est.fit(((X - mu) / sd)[:, -1, :], (y - ymu) / ysd)
    return pd.Series(est.coef_, index=feature_names).sort_values(key=np.abs, ascending=False)

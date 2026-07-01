"""Shared evaluation harness for the return forecasters.

A single walk-forward driver trains every model on the *same* folds with the
*same* standardization, so their out-of-sample predictions are aligned and the
information-coefficient comparison is apples-to-apples. Each model is supplied
as a callable; the driver owns the leakage-control (expanding folds, scaler fit
on train only) so no model can cheat.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# a model fn maps (Xtr_std, ytr_scaled, Xte_std, seed) -> test predictions in
# the SAME (target-scaled) space; the driver unscales them.
ModelFn = Callable[[np.ndarray, np.ndarray, np.ndarray, int], np.ndarray]


def standardize(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature mean/std over a (N, L, F) training block."""
    flat = X_train.reshape(-1, X_train.shape[-1])
    mu = flat.mean(0)
    sd = flat.std(0)
    sd[sd == 0] = 1.0
    return mu.astype(np.float32), sd.astype(np.float32)


def cross_sectional_ic(pred, true, dates) -> float:
    """Average daily Spearman rank-IC across names (can the model RANK stocks
    on a given day -- the metric a PM trades on)."""
    df = pd.DataFrame({"p": pred, "y": true, "d": dates})
    ics = []
    for _, g in df.groupby("d"):
        if len(g) >= 5 and g["p"].nunique() > 1:
            ics.append(spearmanr(g["p"], g["y"]).correlation)
    return float(np.nanmean(ics)) if ics else float("nan")


def metrics_table(y_true, preds: dict, dates) -> pd.DataFrame:
    rows = []
    for name, p in preds.items():
        if len(p) == 0:
            continue
        err = p - y_true
        rows.append({
            "model": name,
            "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "DirAcc": float(np.mean(np.sign(p) == np.sign(y_true))),
            "IC_pooled": float(spearmanr(p, y_true).correlation),
            "IC_xsection": cross_sectional_ic(p, y_true, dates),
            "n": len(p),
        })
    return pd.DataFrame(rows).set_index("model")


def walk_forward(
    data: dict,
    model_fns: dict[str, ModelFn],
    n_folds: int = 3,
    test_years: float = 3.0,
    min_train: int = 2000,
    seed: int = 42,
) -> dict:
    """Expanding walk-forward over the last `test_years`, retraining every fold.

    Returns aligned out-of-sample predictions for every model plus a constant
    (train-mean) baseline, and a metrics table.
    """
    X, y, date = data["X"], data["y"], data["date"]
    order = np.argsort(date, kind="stable")
    X, y, date = X[order], y[order], date[order]
    daypos, ticker = data["daypos"][order], data["ticker"][order]

    all_dates = np.sort(np.unique(date))
    last = all_dates[-1]
    test_start0 = last - pd.Timedelta(days=int(test_years * 365.25))
    fold_edges = pd.date_range(test_start0, last, periods=n_folds + 1)

    preds: dict[str, list] = {name: [] for name in model_fns}
    preds["const"] = []
    oos = {k: [] for k in ("y", "date", "ticker", "daypos")}

    for f in range(n_folds):
        lo = np.datetime64(fold_edges[f])
        hi = np.datetime64(fold_edges[f + 1])
        train_mask = date < lo
        if f < n_folds - 1:
            test_mask = (date >= lo) & (date < hi)
        else:
            test_mask = (date >= lo) & (date <= hi)
        if train_mask.sum() < min_train or test_mask.sum() == 0:
            continue

        mu, sd = standardize(X[train_mask])
        Xtr = (X[train_mask] - mu) / sd
        Xte = (X[test_mask] - mu) / sd
        ytr = y[train_mask]
        ymu, ysd = float(ytr.mean()), float(ytr.std() or 1.0)
        ytr_s = ((ytr - ymu) / ysd).astype(np.float32)

        for name, fn in model_fns.items():
            preds[name].append(fn(Xtr, ytr_s, Xte, seed) * ysd + ymu)
        preds["const"].append(np.full(int(test_mask.sum()), ymu, dtype=np.float32))
        oos["y"].append(y[test_mask])
        oos["date"].append(date[test_mask])
        oos["ticker"].append(ticker[test_mask])
        oos["daypos"].append(daypos[test_mask])

    for k in preds:
        preds[k] = np.concatenate(preds[k]) if preds[k] else np.empty(0)
    for k in oos:
        oos[k] = np.concatenate(oos[k]) if oos[k] else np.empty(0)

    return {"preds": preds, "oos": oos, "metrics": metrics_table(oos["y"], preds, oos["date"])}

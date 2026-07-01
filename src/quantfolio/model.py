from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .features import FEATURE_NAMES


def walk_forward_probabilities(
    events: pd.DataFrame,
    trading_days: pd.DatetimeIndex,
    n_estimators: int = 400,
    max_depth: int = 6,
    retrain_freq: int = 63,
    min_train_events: int = 200,
    random_state: int = 42,
) -> pd.Series:
    """Meta-model success probability for each event, without leakage.

    The model is refit every `retrain_freq` trading days. For a prediction
    block starting at t, the training set contains only events whose EXIT
    date is strictly before t — i.e. labels that were fully resolved before
    the model could have been trained. This purges the overlapping-label
    leakage that plagues naive cross-validation of triple-barrier labels.
    """
    probs = pd.Series(np.nan, index=events.index)
    if events.empty:
        return probs
    X_all = events[FEATURE_NAMES].to_numpy()
    y_all = events["label"].to_numpy()

    block_starts = trading_days[::retrain_freq]
    for i, t0 in enumerate(block_starts):
        t1 = block_starts[i + 1] if i + 1 < len(block_starts) else trading_days[-1] + pd.Timedelta(days=1)
        test_mask = (events["signal_date"] >= t0) & (events["signal_date"] < t1)
        if not test_mask.any():
            continue
        train_mask = events["exit_date"] < t0
        if train_mask.sum() < min_train_events:
            continue
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=20,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )
        clf.fit(X_all[train_mask.to_numpy()], y_all[train_mask.to_numpy()])
        probs.loc[test_mask] = clf.predict_proba(X_all[test_mask.to_numpy()])[:, 1]
    return probs


def bet_size(prob: pd.Series, min_prob: float = 0.55) -> pd.Series:
    """Linear sizing above the threshold: 0 at min_prob, 1 at prob=1."""
    size = (prob - min_prob) / (1.0 - min_prob)
    size = size.clip(lower=0.0, upper=1.0)
    return size.where(prob >= min_prob, 0.0).fillna(0.0)


def final_feature_importance(events: pd.DataFrame, **rf_kwargs) -> pd.Series:
    """Fit once on all events purely for reporting which features matter."""
    if events.empty:
        return pd.Series(dtype=float)
    clf = RandomForestClassifier(
        n_estimators=rf_kwargs.get("n_estimators", 400),
        max_depth=rf_kwargs.get("max_depth", 6),
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(events[FEATURE_NAMES], events["label"])
    return pd.Series(clf.feature_importances_, index=FEATURE_NAMES).sort_values(ascending=False)

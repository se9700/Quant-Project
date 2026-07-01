"""Pooled (panel) LSTM that forecasts forward h-day returns.

Design choices that matter for a credible quant result:
  * ONE model trained across all tickers (panel), not 30 fragile per-stock
    models. The ticker's recent feature sequence is the input.
  * Feature/target standardization is fit on TRAINING rows only (handled by
    forecast_eval) -> no lookahead through the scaler.
  * Evaluated by the shared walk-forward driver against the linear and constant
    models on identical folds.

GPU notes (this is also a GPU-optimization practice surface -- see GPU.md):
  * `get_device()` auto-selects CUDA when available.
  * Mixed precision (AMP) is enabled on CUDA via torch.amp; it is a no-op on
    CPU. On a small model the win is modest; it matters once the model and
    universe are scaled up.
Daily-return signal-to-noise is tiny; on a small universe this LSTM tends to
overfit and lose to the linear model -- scale the universe (S&P 500) to give
the panel enough cross-section to learn from.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import FEATURE_NAMES

try:
    import torch
    from torch import nn
except ImportError as e:  # pragma: no cover
    raise ImportError("forecast_lstm needs torch: pip install torch") from e


def get_device(prefer_gpu: bool = True) -> "torch.device":
    if prefer_gpu and torch.cuda.is_available():
        # fixed input shapes -> let cuDNN autotune; allow TF32 on Ampere+
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return torch.device("cuda")
    return torch.device("cpu")


# --------------------------------------------------------------------------
# data assembly
# --------------------------------------------------------------------------
def build_sequences(
    prices: pd.DataFrame,
    feats: dict[str, pd.DataFrame],
    horizon: int = 5,
    seq_len: int = 30,
) -> dict:
    """Pool (seq_len, n_features) windows across all tickers."""
    F = len(FEATURE_NAMES)
    day_pos = {d: i for i, d in enumerate(prices.index)}
    Xs, ys, dates, tickers, dpos = [], [], [], [], []

    for tk in prices.columns:
        px = prices[tk]
        fwd = px.shift(-horizon) / px - 1.0
        M = np.column_stack([feats[name][tk].to_numpy() for name in FEATURE_NAMES])
        fwd_v = fwd.to_numpy()
        idx = prices.index
        for i in range(seq_len - 1, len(idx)):
            tgt = fwd_v[i]
            if not np.isfinite(tgt):
                continue
            window = M[i - seq_len + 1 : i + 1]
            if not np.isfinite(window).all():
                continue
            Xs.append(window)
            ys.append(tgt)
            dates.append(idx[i])
            tickers.append(tk)
            dpos.append(day_pos[idx[i]])

    return {
        "X": np.asarray(Xs, dtype=np.float32),
        "y": np.asarray(ys, dtype=np.float32),
        "date": np.asarray(dates),
        "ticker": np.asarray(tickers),
        "daypos": np.asarray(dpos),
        "n_features": F,
    }


def current_sequences(
    prices: pd.DataFrame,
    feats: dict[str, pd.DataFrame],
    tickers: list[str],
    seq_len: int = 30,
) -> tuple[np.ndarray, list[str]]:
    """Latest (seq_len, F) window per requested ticker for live prediction."""
    rows, ok = [], []
    for tk in tickers:
        if tk not in prices.columns:
            continue
        M = np.column_stack([feats[name][tk].to_numpy() for name in FEATURE_NAMES])
        window = M[-seq_len:]
        if window.shape[0] == seq_len and np.isfinite(window).all():
            rows.append(window)
            ok.append(tk)
    X = np.asarray(rows, dtype=np.float32) if rows else np.empty((0, seq_len, len(FEATURE_NAMES)), np.float32)
    return X, ok


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
class LSTMForecaster(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features, hidden, num_layers=layers, batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def _train_one(
    X_tr, y_tr, n_features, hidden, layers, dropout,
    epochs, batch, lr, patience, seed, device, use_amp,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    n = len(X_tr)
    cut = int(n * 0.85)
    Xt, yt = X_tr[:cut], y_tr[:cut]
    Xv, yv = X_tr[cut:], y_tr[cut:]

    model = LSTMForecaster(n_features, hidden, layers, dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    amp = bool(use_amp and device.type == "cuda")
    dev_type = "cuda" if device.type == "cuda" else "cpu"
    scaler = torch.amp.GradScaler(dev_type, enabled=amp)

    Xt_t = torch.from_numpy(Xt).to(device)
    yt_t = torch.from_numpy(yt).to(device)
    Xv_t = torch.from_numpy(Xv).to(device)
    yv_t = torch.from_numpy(yv).to(device)

    best_val, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt_t), device=device)
        for b in range(0, len(perm), batch):
            sel = perm[b : b + batch]
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(dev_type, enabled=amp):
                loss = loss_fn(model(Xt_t[sel]), yt_t[sel])
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        model.eval()
        with torch.no_grad(), torch.amp.autocast(dev_type, enabled=amp):
            v = loss_fn(model(Xv_t), yv_t).item() if len(Xv_t) else 0.0
        if v < best_val - 1e-9:
            best_val = v
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _predict(model, X, device, batch=4096):
    model.eval()
    out = []
    dev_type = "cuda" if device.type == "cuda" else "cpu"
    with torch.no_grad(), torch.amp.autocast(dev_type, enabled=(device.type == "cuda")):
        for b in range(0, len(X), batch):
            xb = torch.from_numpy(X[b : b + batch]).to(device)
            out.append(model(xb).float().cpu().numpy())
    return np.concatenate(out) if out else np.empty(0, np.float32)


def make_lstm_fn(
    n_features: int,
    hidden: int = 64,
    layers: int = 2,
    dropout: float = 0.1,
    epochs: int = 25,
    batch: int = 256,
    lr: float = 1e-3,
    patience: int = 4,
    device=None,
    use_amp: bool = True,
):
    """Build a model fn for forecast_eval.walk_forward."""
    device = device or get_device()

    def fn(Xtr, ytr_s, Xte, seed):
        model = _train_one(Xtr, ytr_s, n_features, hidden, layers, dropout,
                           epochs, batch, lr, patience, seed, device, use_amp)
        return _predict(model, Xte, device)

    return fn


def fit_full_and_forecast(
    data: dict,
    prices: pd.DataFrame,
    feats: dict[str, pd.DataFrame],
    holdings: list[str],
    seq_len: int = 30,
    hidden: int = 64,
    layers: int = 2,
    dropout: float = 0.1,
    epochs: int = 25,
    batch: int = 256,
    lr: float = 1e-3,
    patience: int = 4,
    seed: int = 42,
    device=None,
    use_amp: bool = True,
) -> pd.Series:
    """Train on ALL history, predict each holding's next h-day return."""
    from .forecast_eval import standardize

    device = device or get_device()
    X, y = data["X"], data["y"]
    mu, sd = standardize(X)
    Xs = (X - mu) / sd
    ymu, ysd = float(y.mean()), float(y.std() or 1.0)
    model = _train_one(Xs, ((y - ymu) / ysd).astype(np.float32), data["n_features"],
                       hidden, layers, dropout, epochs, batch, lr, patience, seed,
                       device, use_amp)
    Xc, ok = current_sequences(prices, feats, holdings, seq_len)
    if len(Xc) == 0:
        return pd.Series(dtype=float)
    Xc = (Xc - mu) / sd
    pred = _predict(model, Xc, device) * ysd + ymu
    return pd.Series(pred, index=ok)

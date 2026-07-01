# Quantfolio — Handoff

A working notebook for picking the project back up. For *what it is* and *how
to run it*, see [README.md](README.md); this file is the working state, the
decisions behind it, the honest findings, and what to do next.

Last updated: 2026-07-01.

---

## 1. What this project is (one paragraph)

A portfolio-management research system that pairs **triple-barrier
meta-labeling** (López de Prado) with a **fundamental valuation** overlay, plus
forward-looking tools: a Monte-Carlo portfolio outlook, a panel-LSTM return
forecaster, and a VAR dividend forecaster. Built as a career-showcase quant
project. Daily bars, ~30 large-cap US equities (config universe) unioned with
whatever is in `portfolio.csv`. Data is yfinance, cached locally. Python 3.10.

---

## 2. How to run (quick reference)

```
pip install -r requirements.txt
#   torch is CPU-only: pip install torch --index-url https://download.pytorch.org/whl/cpu

python run_backtest.py          # full backtest + meta-model, writes reports/
python portfolio_outlook.py     # Monte-Carlo + model view of portfolio.csv
python forecast.py              # LSTM return forecast + VAR dividend forecast
```

Add `--refresh-data` to any of them to re-download instead of using the cache.
`portfolio.csv` holds current holdings (`ticker` + `shares` or `weight`).
All parameters live in `config.yaml`.

Environment notes:
- Console is **CP949 (Korean locale)** → keep all `print()` output ASCII-only.
- `lxml` is required (yfinance `get_earnings_dates`); it is in requirements.
- IDE may show "matplotlib/PyYAML not installed" hints — that's a different
  interpreter than the one that runs the code; ignore.

---

## 3. Architecture / file map

```
config.yaml                      all knobs (universe, barriers, model, costs)
portfolio.csv                    current holdings
run_backtest.py                  CLI: data -> labels -> meta-model -> backtest
portfolio_outlook.py             CLI: forward Monte-Carlo + per-holding view
forecast.py                      CLI: LSTM returns + VAR dividends
src/quantfolio/
  config.py                      yaml loader, makes data/ and reports/ dirs
  data.py                        yfinance loaders (prices, fundamentals,
                                 earnings dates), per-ticker CSV cache
  valuation.py                   cross-sectional value score/rank (E/P + B/P)
  fundamentals.py                point-in-time statement factors + earnings
                                 surprise (PEAD) factor
  features.py                    assembles the feature panel; FEATURE_NAMES is
                                 THE contract every model reads
  labeling.py                    primary signal (SMA cross + value filter) and
                                 triple-barrier labeling
  model.py                       random-forest meta-model, purged walk-forward
  backtest.py                    sized trades -> daily equity curve
  metrics.py                     performance stats + plots
  outlook.py                     model_view + block-bootstrap simulate_portfolio
  forecast_eval.py               shared walk-forward driver + IC/RMSE metrics
                                 (trains all models on identical folds)
  forecast_lstm.py               pooled panel LSTM (PyTorch), GPU/AMP-aware,
                                 exposes make_lstm_fn for the shared driver
  forecast_linear.py             regularized linear forecaster (ridge/elasticnet)
  forecast_dividends.py          VAR on log TTM-DPS + Lintner fallback
reports/                         generated CSVs and PNGs
data/cache/                      *_prices.csv, *_fundamentals.csv,
                                 *_earnings.csv, *_dividends.csv
```

**Key design rule:** `features.FEATURE_NAMES` is the single source of truth for
the feature vector. Add a feature there + produce its panel in `build_features`
and it automatically flows to the meta-model, the LSTM, the outlook model view,
and importance reporting. Don't hardcode feature lists anywhere else.

---

## 4. Methodology safeguards (don't break these)

- **No same-bar execution**: signal fires on a close, entry is the next close.
- **Reporting lag**: fundamentals usable only 60 days after period end
  (`valuation.py`, `fundamentals.py`).
- **Earnings surprise** is anchored to the announcement date + 1 day (no
  same-day lookahead) and decays over ~90 days.
- **Purged walk-forward** (`model.py`): the meta-model trains only on events
  whose triple-barrier exit date precedes the prediction window — removes
  overlapping-label leakage.
- **LSTM** (`forecast_lstm.py`): feature/target scalers fit on training rows
  only; expanding folds retrain from scratch; reported against constant +
  ridge-linear baselines via RMSE / directional accuracy / information
  coefficient (the script prints a blunt verdict).

If you add a feature or model, keep it point-in-time. The whole credibility of
the project rests on no lookahead.

---

## 5. Current results (as of last run)

Meta-model backtest, live window 2016-07 .. 2026-06, net of 10bps costs:
- Strategy Sharpe ~0.51 vs SPY 0.88; MaxDD ~-19% vs -34%; AnnVol ~6.5%.
- Low-exposure, defensive. Meta-model hit rate ~51% (base rate 45%).

LSTM forward 5d-return forecast (37-name panel, 3 folds):
- LSTM IC_xsection ~0.01; ridge-linear ~0.05; constant baseline ~0.
- **The linear model beats the LSTM.** This is real, not a bug.

VAR dividends (portfolio.csv): forward portfolio yield ~0.20%, ~$23/yr on
~$11.7k — appropriately tiny for an AI-heavy, mostly-non-paying book.

---

## 6. Honest findings to preserve (these ARE the value of the project)

1. **The LSTM does not beat a regularized linear model** on this universe, and
   adding richer features made the LSTM *worse* (0.029 -> 0.008 IC) while the
   linear model held ~0.05. Diagnosis: overfitting a 37-name panel. The fix is
   data breadth (S&P 500), not model complexity.
2. **Feature importance** (RF meta-model): price momentum/vol dominate;
   `earn_surprise` (~0.07) and `rel_mom_63` (~0.09) are genuinely useful;
   statement-level fundamentals (margin/ROE/earnings growth) ~0.00.
3. **Data constraint**: yfinance free tier returns only ~5 quarterly
   statements per ticker — too short for YoY-of-TTM growth (needs 8) over a
   multi-year backtest. `get_earnings_dates` gives ~49 quarters (to 2014), so
   the earnings-surprise feature carries the fundamentals history.
4. **VAR is a forced fit for dividends** (sticky/Lintner; lumpy ETF payers blow
   it up). It is implemented honestly with TTM smoothing + a +-60% clip and a
   Lintner fallback, and flags which method produced each number.

When presenting this project, these findings are strengths — they show
judgment. Don't bury them.

---

## 7. Known limitations / tech debt

- **Survivorship bias**: universe is today's large caps applied to history.
- **Fundamentals depth**: see finding #3; share counts fall back to current
  count when history is missing (`data.py`).
- **Daily closes only**: barrier touches detected on close, not intraday.
- yfinance is unofficial and occasionally flaky — the cache mitigates this.
- The three CLIs duplicate the load->value->features block. If it grows, factor
  it into one `assemble_features()` helper (kept out so far to avoid IO
  coupling in features.py).

---

## 8. Next steps (ordered by value)

1. **[DONE 2026-07-01] Regularized linear forecaster + S&P 500 universe +
   GPU support.** `forecast_linear.py` is first-class and wins on the small
   universe (IC ~0.056 vs LSTM ~0.018). `forecast_eval.py` trains both on
   identical folds. `forecast.py` flags: `--model`, `--universe config|sp500`,
   `--device auto|cpu|cuda`, `--no-amp`. GPU/RunPod guide in `GPU.md`. STILL TO
   DO: actually run `--universe sp500` on a GPU and see whether the breadth
   lets the LSTM close the gap (cache the S&P 500 data first — it's slow).
2. **Rebalancing suggestor** (user has asked for this twice). Combine
   meta-model probabilities + risk decomposition (`outlook.simulate_portfolio`
   returns `risk_contrib`) + return/dividend forecasts to propose weight
   changes, then backtest the overlay. New module + CLI.
3. **Deepen dividends**: use declared/announced forward dividends where
   available; model payout-ratio x forecasted earnings instead of a raw VAR;
   separate special vs regular and ETF distributions; add a dividend-safety
   (cut-probability) read.
4. **LSTM improvements** (only after #1): longer horizon, sector-neutral /
   excess-return targets, dropout/regularization, sequential-bootstrap weights.

---

## 9. Where context lives

- Persistent project memory:
  `C:\Users\se970\.claude\projects\c--Users-se970-PremiumProject\memory\quantfolio-project.md`
  (decisions, results, findings — kept in sync with this file).
- This HANDOFF.md is the human-readable counterpart; update both when the
  project's state changes materially.

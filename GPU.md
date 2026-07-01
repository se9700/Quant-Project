# Running on a Cloud GPU (RunPod) + GPU-Optimization Playbook

This project's LSTM is deliberately GPU-aware so you can practice GPU
optimization on a real workload. Honest expectation up front: **on the small
30-name universe the GPU barely helps** — the model is tiny and kernel-launch
overhead dominates. The GPU becomes worth it once you scale up, which is
exactly why we added `--universe sp500` and a bigger model. Do your
optimization practice on the scaled-up configuration.

---

## 0. What in the code is already GPU-ready

- `forecast_lstm.get_device()` auto-selects CUDA and turns on cuDNN autotune
  (`benchmark=True`) and TF32 matmul (Ampere+).
- `_train_one` uses mixed precision (`torch.amp` autocast + GradScaler) on CUDA;
  it is a no-op on CPU.
- Training tensors are moved to the device once (no per-batch host↔device copy),
  and shuffling is done on-device.
- CLI: `--device auto|cpu|cuda` and `--no-amp`. Hyperparameters live under
  `forecast:` in `config.yaml`.

---

## 1. Spin up a RunPod pod

1. Create a RunPod account, add credit.
2. **Deploy → Pods → GPU Cloud.** Pick a GPU:
   - RTX 3090 / 4090 (24 GB) — cheapest sensible choice for this workload.
   - A100 only if you scale the model a lot; overkill otherwise.
   - Prefer a **Community Cloud / Spot** instance to save money.
3. Template: **RunPod PyTorch** (ships CUDA + PyTorch). Confirm the CUDA-enabled
   PyTorch base so you don't fight driver/toolkit versions.
4. **Attach a persistent volume** mounted at `/workspace` (e.g. 20–50 GB) so the
   yfinance cache and reports survive pod restarts — re-downloading 500 tickers
   each session is slow and rate-limited.
5. Expose **SSH** (and optionally Jupyter). Start the pod.

```bash
ssh root@<pod-ip> -p <port> -i ~/.ssh/<key>      # connection string is in the pod UI
nvidia-smi                                        # confirm the GPU is visible
```

## 2. Get the project onto the pod

```bash
cd /workspace
git clone <your-repo-url> quantfolio   # or: rsync -avz ./PremiumProject root@pod:/workspace/quantfolio
cd quantfolio

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# the requirements torch line is CPU; on the pod install the CUDA build instead:
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 3. Pre-fetch data, then run

Downloading prices/fundamentals/earnings for the S&P 500 is the slow part and is
**CPU/network-bound, not GPU work** — do it once, it caches to `/workspace`.

```bash
# warm the cache (run once; safe to re-run, it skips cached tickers)
python -c "import sys; sys.path.insert(0,'src'); \
from quantfolio.config import load_config; \
from quantfolio.data import load_sp500_universe, load_prices, load_fundamentals, load_earnings; \
c=load_config(); u=load_sp500_universe(c['data_cache']); \
p=load_prices(u,c['start'],c['end'],c['data_cache']); \
load_fundamentals(list(p.columns),c['data_cache']); \
load_earnings(list(p.columns),c['data_cache'])"

# then the GPU run
python forecast.py --universe sp500 --device cuda
```

To compare CPU vs GPU on the same job:

```bash
python forecast.py --universe sp500 --device cpu --epochs 10
python forecast.py --universe sp500 --device cuda --epochs 10
```

---

## 4. GPU-optimization playbook (the practice part)

Work top-to-bottom; **measure after every change** — optimization without
measurement is guessing.

**Establish the baseline / find the bottleneck**
- `watch -n0.5 nvidia-smi` in a second SSH session while training. The number
  that matters is **GPU-Util %**. Low util = the GPU is starved (Python loop,
  tiny batches, data prep), not compute-bound.
- Time one epoch and compute **samples/sec**. That's your throughput metric.
- Rule of thumb here: if GPU-Util is low, the fix is *bigger batches / bigger
  model*, not a faster GPU.

**Levers, roughly in order of payoff**
1. **Batch size.** Raise `forecast.batch` (256 → 1024 → 4096…) until VRAM is
   ~80% full (`nvidia-smi`) or throughput stops improving. Bigger batches keep
   the GPU busy and cut Python/launch overhead. Watch that IC doesn't degrade.
2. **Mixed precision (AMP).** Already on by default on CUDA. Toggle `--no-amp`
   and compare samples/sec and VRAM. Expect a throughput win and lower memory
   on Ampere+.
3. **Model size = utilization.** A bigger `hidden`/`layers`/`seq_len` raises
   arithmetic intensity, so the GPU does more work per launch and util climbs.
   This is the honest way to make the GPU "worth it."
4. **`torch.compile`.** Add `model = torch.compile(model)` right after the model
   is built in `forecast_lstm._train_one` (guard with `device.type=="cuda"`).
   First step pays a compile cost; steady-state is faster. Good experiment.
5. **TF32 / cuDNN autotune.** Already enabled in `get_device()`. Verify the
   effect by toggling `torch.backends.cudnn.benchmark` off and comparing.
6. **Keep data off the host path.** Tensors already live on-device. If a scaled
   dataset no longer fits in VRAM, switch to a `DataLoader` with
   `pin_memory=True`, `num_workers>0`, and `non_blocking=True` transfers — then
   the bottleneck shifts to the input pipeline and the above changes.

**Profiling (go deeper)**
- `torch.profiler` with the `tensorboard_trace_handler` to see kernel-level time
  and the CPU/GPU timeline (look for gaps = stalls).
- `torch.cuda.max_memory_allocated()` to size memory headroom for batch tuning.
- `nvidia-smi dmon` / Nsight Systems for a system view.

**Cost discipline**
- **Stop the pod** when you're done — billing is per-second while running.
- Keep the cache on the persistent volume; don't re-download each session.
- Prefer spot/community instances; checkpoint if you run long jobs.

---

## 5. What to expect (so results don't surprise you)

- Small universe: GPU ≈ CPU or worse (overhead). Don't conclude the GPU is
  "broken" — the workload is just too small.
- Scaled universe + bigger model: GPU should show a clear samples/sec win and
  high util; AMP and batch tuning then matter.
- Modeling-wise, more cross-sectional breadth (S&P 500) is the main hope for the
  LSTM to finally rival the linear model; track `IC_xsection` in the run output
  to see whether scale actually closes the gap.

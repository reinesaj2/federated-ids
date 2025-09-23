# Federated IDS Demo (Flower + PyTorch)

Federated learning demo using [Flower](https://flower.dev) for orchestration and PyTorch for local training.
Supports synthetic and real IDS datasets (CIC-IDS2017, UNSW-NB15) with preprocessing (scaling and one‑hot encoding)
and non‑IID partitioning (IID, Dirichlet, protocol). Includes robust aggregation implementations (Median, Krum, simplified Bulyan).

## Table of Contents

1. Prerequisites (what you need installed)
2. One‑command verification (recommended first run)
3. Manual Quickstart (server + two clients)
4. Expected output (so you know it worked)
5. Reproducibility & logging (seeds, logs, plots)
6. Real datasets (UNSW‑NB15, CIC‑IDS2017)
7. Troubleshooting (common errors and fixes)
8. Project structure
9. Notes on privacy/robustness scaffolding

---

## 1) Prerequisites

- macOS or Linux (Windows works via WSL2).
- Python 3.10–3.12 recommended (CPU‑only is fine). Check with:
  ```bash
  python3 --version
  ```
- Enough disk for datasets (optional demos use a 10% UNSW sample).

Clone or open the project folder. In what follows, replace <ABS_PATH> with your absolute path:
`/Users/you/Documents/Thesis/federated-ids`.

---

## 2) One‑command verification (recommended)

This runs the server and two synthetic clients twice on two ports, and checks reproducibility. It creates `.verify_logs/`.

```bash
cd <ABS_PATH>
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt

export PORT_MAIN=8099 PORT_ALT=8100 ROUNDS=2 TIMEOUT_SECS=30 SEED=42
bash scripts/verify_readme.sh
```

You should see “All checks passed”. If this completes, you’re ready to demo.

---

## 3) Manual Quickstart (server + two clients)

Run everything from the project root. Use three terminals (one for server, two for clients).

### 3.1 Create and activate a virtual environment (if not already done)
```bash
cd <ABS_PATH>
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
```

### 3.2 Start the Flower server (FedAvg, 2 rounds)
```bash
export SEED=42
python server.py --rounds 2 --aggregation fedavg --server_address 127.0.0.1:8099
```

Notes:
- Deprecation warnings about `start_server`/`start_client` are expected on flwr==1.21.0.
- If port 8099 is busy, choose another (e.g., 8100) and use it for both server and clients.

### 3.3 Start two synthetic clients (in two new terminals)

Terminal B:
```bash
cd <ABS_PATH>
source .venv/bin/activate
python client.py --server_address 127.0.0.1:8099 --dataset synthetic --samples 2000 --features 20 --seed 42 --client_id 0 --num_clients 2
```

Terminal C:
```bash
cd <ABS_PATH>
source .venv/bin/activate
python client.py --server_address 127.0.0.1:8099 --dataset synthetic --samples 2000 --features 20 --seed 42 --client_id 1 --num_clients 2
```

The server will run 2 rounds and then print a summary and exit.

---

## 4) Expected output

On the server terminal, a successful 2‑round run ends with output similar to:

```64:71:/Users/abrahamreines/Documents/Thesis/federated-ids/.verify_logs/example_server_output.txt
INFO :      [SUMMARY]
INFO :      Run finished 2 round(s) in ~80s
INFO :          History (loss, distributed):
INFO :                  round 1: 0.05...
INFO :                  round 2: 0.04...
INFO :          History (metrics, distributed, evaluate):
INFO :          {'accuracy': [(1, ~0.98), (2, ~0.978)]}
```

On each client terminal, you’ll see lines such as:

```1:6:/Users/abrahamreines/Documents/Thesis/federated-ids/runs/smoke_metrics/README_example_client.txt
[Client X] Logging metrics to: ./logs/client_X_metrics.csv
[Data] Train samples=1600, class_counts={0: 800, 1: 800}; Test samples=400, class_counts={0: 200, 1: 200}
[Client X] Model validation passed: out_features=2, num_classes_global=2
```

---

## 5) Reproducibility, logs, and plots

- Reproducibility: set `SEED` on the server and `--seed` on clients. Example:
  ```bash
  export SEED=42
  python server.py --rounds 2
  python client.py --seed 42 ...
  ```
- Logs: CSV files are written to `./logs/` (e.g., `metrics.csv`, `client_0_metrics.csv`).
- Plots: generate figures from any run directory that contains CSVs:
  ```bash
  # Server + client plots → saves PNGs next to the CSVs by default
  python scripts/plot_metrics.py --run_dir ./logs --output_dir ./runs/smoke_metrics

  # JSON summary of client metrics
  python scripts/summarize_metrics.py --run_dir ./logs --output ./runs/smoke_metrics/summary.json
  ```

---

## 6) Real datasets (UNSW‑NB15, CIC‑IDS2017)

Important rule: all clients connected to the same server must use the same dataset and preprocessing settings.
Do not mix synthetic with UNSW/CIC (or different feature configs) on the same server run, or you will get a
“state_dict size mismatch” error.

### 6.1 UNSW‑NB15 (Dirichlet non‑IID; 3 clients)

Prepare a fast 10% sample for demos (adjust the input path to your file if needed):
```bash
cd <ABS_PATH>
mkdir -p data/unsw
python scripts/prepare_unsw_sample.py \
  --input data/unsw/UNSW_NB15_training-set.csv \
  --output data/unsw/UNSW_NB15_training-set.sample.csv \
  --frac 0.10 --seed 42
```

---

## 7) Troubleshooting

- Deprecation warnings (Flower): you might see messages about `start_server`/`start_client` being deprecated.
  This demo targets flwr==1.21.0 and is known to work despite the warnings.

- Address already in use: change the port (e.g., to 8100) and pass the same port to the clients.
  ```bash
  # Find what is using the port 8099 (macOS/Linux)
  lsof -i :8099
  ```

- State dict size mismatch: all clients in a given run must use the same dataset and preprocessing
  (do not mix synthetic with UNSW/CIC in the same server run).

- File not found for dataset: verify your `--data_path` exists. If your file is `.gz`, decompress or
  pass the correct path.

- Plots not showing: this script saves `.png` files; no GUI needed. If you run headless and see backend errors,
  try: `export MPLBACKEND=Agg` before running plotting scripts.

- Permissions for scripts: if `verify_readme.sh` is not executable, run `chmod +x scripts/verify_readme.sh`.

- CPU vs GPU: no GPU required. Torch CPU build is sufficient for the demo.

---

## 8) Project structure

- `server.py` – Flower server with FedAvg and robust aggregation options (`median`, `krum`, simplified `bulyan`).
  - For `fedavg`, aggregation is sample‑size weighted; robust methods are intentionally unweighted.
- `client.py` – PyTorch `NumPyClient` with a small MLP; supports synthetic, UNSW‑NB15, and CIC‑IDS2017 datasets
  with IID/Dirichlet/protocol partitions.
- `data_preprocessing.py` – CSV loaders, preprocessing (StandardScaler + OneHotEncoder), partitioning
  (iid/dirichlet/protocol), and DataLoader builders.
- `robust_aggregation.py` – Aggregation method enum and robust implementations (Median, Krum, simplified Bulyan).
- `scripts/verify_readme.sh` – Non‑interactive verification for automated demo sanity checks.
- `scripts/plot_metrics.py` – Generate server/client metric plots from CSV logs.
- `scripts/summarize_metrics.py` – Emit a compact JSON summary of client metrics.
- `requirements.txt` – Python dependencies.

---

## 9) Privacy & robustness disclosure (D2 scope)

- Differential Privacy (scaffold): client‑side clipping with Gaussian noise applied to the model update
  before sending. This is not DP‑SGD and does not include privacy accounting.
- Secure Aggregation (stub): toggle provided and status logged, but updates are not cryptographically masked.
  Integration of secure summation/masking is planned for a later milestone.
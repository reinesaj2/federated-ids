# Federated IDS Demo (Flower + PyTorch)

Federated learning setup using [Flower](https://flower.dev) for orchestration and PyTorch for local training. Supports synthetic and real IDS datasets (CIC-IDS2017, UNSW-NB15) with preprocessing (scaling and one‑hot encoding) and non‑IID partitioning (IID, Dirichlet, protocol). Includes robust aggregation implementations (median, Krum, simplified Bulyan).

## Quickstart

```bash
# 1) Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2) Install dependencies
pip install -U pip setuptools wheel
pip install -r requirements.txt

# 3) Start the Flower server (choose aggregation, optionally set byzantine_f)
python server.py --rounds 2 --aggregation fedavg --server_address 127.0.0.1:8080
# Examples:
#   Median:
#   python server.py --rounds 2 --aggregation median
#   Krum with explicit f=1:
#   python server.py --rounds 2 --aggregation krum --byzantine_f 1

# 4) Start clients (synthetic or real datasets)
# Synthetic (two clients):
python client.py --server_address 127.0.0.1:8080 --dataset synthetic --samples 2000 --features 20 --seed 42 --client_id 0 --num_clients 2
python client.py --server_address 127.0.0.1:8080 --dataset synthetic --samples 2000 --features 20 --seed 42 --client_id 1 --num_clients 2

# UNSW-NB15 (CSV), Dirichlet non-IID partitions across 3 clients:
# Assume data at /path/to/UNSW-NB15.csv
python client.py --dataset unsw --data_path /path/to/UNSW-NB15.csv --partition_strategy dirichlet --num_clients 3 --client_id 0 --alpha 0.1 --batch_size 64 --seed 42
python client.py --dataset unsw --data_path /path/to/UNSW-NB15.csv --partition_strategy dirichlet --num_clients 3 --client_id 1 --alpha 0.1 --batch_size 64 --seed 42
python client.py --dataset unsw --data_path /path/to/UNSW-NB15.csv --partition_strategy dirichlet --num_clients 3 --client_id 2 --alpha 0.1 --batch_size 64 --seed 42

# CIC-IDS2017 with protocol-based partitioning (if protocol column available):
python client.py --dataset cic --data_path /path/to/CIC-IDS2017.csv --partition_strategy protocol --num_clients 3 --client_id 0 --protocol_col Protocol --batch_size 64 --seed 42
python client.py --dataset cic --data_path /path/to/CIC-IDS2017.csv --partition_strategy protocol --num_clients 3 --client_id 1 --protocol_col Protocol --batch_size 64 --seed 42
python client.py --dataset cic --data_path /path/to/CIC-IDS2017.csv --partition_strategy protocol --num_clients 3 --client_id 2 --protocol_col Protocol --batch_size 64 --seed 42
```

You should see two clients connect, local training per round, and the server complete two rounds.

Note:
- Flower may print deprecation warnings about start_server/start_numpy_client. These commands are tested with flwr 1.21.0 and work; migration to SuperLink/SuperNode CLI will be handled later.
- If port 8080 is in use, choose another (e.g., 8081) and pass it to both server and clients.

## Reproducibility & Logging

Set a global seed to reproduce results:

```bash
export SEED=42
python server.py --rounds 2
python client.py --seed 42
```

Server logs include the chosen aggregation method and per-round aggregated accuracy.

## Project Structure

- `server.py` – Flower server with FedAvg and optional robust aggregation (`median`, `krum`, simplified `bulyan`).
  - For `fedavg`, aggregation is sample-size weighted. For robust methods, aggregation is unweighted by design.
- `client.py` – PyTorch `NumPyClient` with a small MLP; supports synthetic, UNSW-NB15, and CIC-IDS2017 datasets with IID/Dirichlet/protocol partitions.
- `data_preprocessing.py` – CSV loaders, preprocessing (StandardScaler + OneHotEncoder), partitioning (iid/dirichlet/protocol), and DataLoader builders.
- `robust_aggregation.py` – Aggregation method enum and robust implementations (median, Krum, simplified Bulyan).
- `requirements.txt` – Python dependencies.

## Algorithm Comparison: FedProx vs FedAvg

This implementation supports both FedAvg (standard federated averaging) and FedProx (proximal federated optimization) for handling non-IID data heterogeneity.

### FedProx Theory

FedProx mitigates client drift in non-IID scenarios by adding a proximal regularization term `μ/2 * ||w - w_global||²` to the local client loss function. This constrains local updates to stay close to the global model, reducing divergence.

### Usage

```bash
# Standard FedAvg (μ = 0)
python client.py --server_address 127.0.0.1:8080 --dataset synthetic --fedprox_mu 0.0

# FedProx with mild regularization (recommended starting point)
python client.py --server_address 127.0.0.1:8080 --dataset synthetic --fedprox_mu 0.01

# FedProx with stronger regularization for highly non-IID data
python client.py --server_address 127.0.0.1:8080 --dataset synthetic --fedprox_mu 0.1
```

### Automated Comparison

Run side-by-side experiments on non-IID synthetic data (Dirichlet α=0.05):

```bash
# Compare algorithms with 5 rounds, non-IID partitioning
scripts/compare_fedprox_fedavg.sh

# Generate comparison plots
python scripts/plot_metrics.py --fedprox_comparison --logdir ./comparison_logs
```

### Trade-offs Summary

| Aspect | FedAvg | FedProx |
|--------|--------|---------|
| **Convergence on IID** | Fast, optimal | Fast, minimal overhead |
| **Convergence on Non-IID** | Slow, client drift issues | Better stability, reduced drift |
| **Computational Cost** | Lower | ~5-10% overhead per round |
| **Hyperparameter Tuning** | None needed | Requires μ selection (0.001-0.1) |
| **Best Use Case** | IID data, homogeneous clients | Non-IID data, heterogeneous systems |

**Recommendation**: Start with FedAvg for baseline comparison. Use FedProx with μ=0.01 when experiencing convergence issues on non-IID data.

## Next Steps (D2 scope)

- Harden and evaluate robust aggregation (Krum/Bulyan), add unit tests.
- Migrate to Flower SuperLink/SuperNode CLI.
- Extend metrics and logging; pin dependency versions for reproducibility.

## Verification

A non-interactive script validates the README steps with timeouts to avoid hanging terminals, checks an alternate port, and verifies reproducibility:

```bash
# Optional: override defaults
export PORT_MAIN=8099 PORT_ALT=8100 ROUNDS=2 TIMEOUT_SECS=30 SEED=42

# Run verification (creates .verify_logs/)
scripts/verify_readme.sh
```

What it checks:
- Server + two clients complete ROUNDS on `127.0.0.1:$PORT_MAIN` within `TIMEOUT_SECS`.
- Same on `127.0.0.1:$PORT_ALT`.
- Two same-seed runs produce matching histories (basic reproducibility).

## Dataset Prep (UNSW-NB15)

Download the official training/testing CSVs from `https://research.unsw.edu.au/projects/unsw-nb15-dataset` and place them under `data/unsw/`, then create a 10% sample for fast demos:

```bash
mkdir -p data/unsw

# Example: sample the training set
python scripts/prepare_unsw_sample.py \
  --input data/unsw/UNSW_NB15_training-set.csv \
  --output data/unsw/UNSW_NB15_training-set.sample.csv \
  --frac 0.10 --seed 42
```

Use the sampled CSV in client commands via `--data_path`.

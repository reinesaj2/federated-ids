# Federated IDS Demo (Flower + PyTorch)

Baseline federated learning setup using [Flower](https://flower.dev) for orchestration and PyTorch for local training. Includes hooks for robust aggregation (median, Krum, Bulyan stubs) and synthetic-data preprocessing utilities. Real IDS datasets (CIC-IDS2017, UNSW-NB15) will be integrated next.

## Quickstart

```bash
# 1) Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2) Install dependencies
pip install -U pip setuptools wheel
pip install -r requirements.txt

# 3) Start the Flower server (FedAvg baseline)
python server.py --rounds 2 --aggregation fedavg --server_address 127.0.0.1:8080

# 4) In two separate terminals, start clients
python client.py --server_address 127.0.0.1:8080 --samples 2000 --features 20 --seed 42
python client.py --server_address 127.0.0.1:8080 --samples 2000 --features 20 --seed 42
```

You should see two clients connect, local training per round, and the server complete two rounds.

Note:
- Flower may print deprecation warnings about start_server/start_numpy_client. These commands are tested with flwr 1.21.0 and work; migration to SuperLink/SuperNode CLI will be handled later.
- If port 8080 is in use, choose another (e.g., 8081) and pass it to both server and clients.

## Reproducibility

Set a global seed to reproduce results:

```bash
export SEED=42
python server.py --rounds 2
python client.py --seed 42
```

## Project Structure

- `server.py` – Flower server using FedAvg; robust aggregation hooks are planned.
- `client.py` – PyTorch `NumPyClient` with a small MLP and synthetic data.
- `data_preprocessing.py` – Synthetic data generation and Dirichlet partitioning utility.
- `robust_aggregation.py` – Aggregation method enum and placeholder implementations.
- `requirements.txt` – Python dependencies.

## Next Steps (D2 scope)

- Integrate real IDS datasets (CIC-IDS2017, UNSW-NB15) with preprocessing (encode, scale, standardize).
- Enable non-IID client partitions via Dirichlet or protocol-based splits.
- Implement robust aggregation methods (Krum, Bulyan) and add unit tests.

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

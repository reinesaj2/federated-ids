# Federated IDS

Federated IDS is a research platform for studying federated learning in intrusion detection settings. The repository combines Flower for orchestration, PyTorch for local training, real and synthetic IDS datasets, Byzantine-robust aggregation, heterogeneity mitigation, personalization, privacy accounting, and a large supporting analysis toolchain for plots, census reports, and publication bundles.

This repository is intended for experimentation, evaluation, and thesis-grade reporting. It is not packaged or hardened as production IDS software.

## Highlights

- Flower-based federated training with configurable server and client entry points in [`server.py`](server.py) and [`client.py`](client.py)
- Dataset support for `synthetic`, `unsw`, `cic`, `edge-iiotset-quick`, `edge-iiotset-nightly`, `edge-iiotset-full`, and `hybrid`
- Partitioning strategies for `iid`, `dirichlet`, `protocol`, and `source` splits
- Aggregation methods `fedavg`, `median`, `krum`, and `bulyan`
- Heterogeneity mitigation through FedProx (`--fedprox_mu`)
- Optional post-round client personalization (`--personalization_epochs`)
- Client-side differential privacy controls with epsilon accounting support
- Secure aggregation research scaffold based on deterministic additive masking
- Experiment automation, plotting, runs census generation, and publication-oriented reporting under [`scripts/`](scripts)
- Slurm batch workflows for large experiment campaigns under [`scripts/slurm`](scripts/slurm)

## Scope And Caveats

- This is research code. Correctness, reproducibility, and observability matter more here than production deployment ergonomics.
- Secure aggregation is not a production cryptographic protocol. The current implementation is a deterministic masking scaffold for experimentation.
- Differential privacy support includes clipping, noise injection, and accounting. When Opacus is unavailable, the accountant falls back to an analytic approximation.
- Robust aggregation implementations are intended for comparative study and may evolve as the experiment design changes.
- Generated artifacts such as `runs/`, `logs/`, `results/`, `reports/publication/`, and derived census CSVs are local outputs and are intentionally ignored by Git.

## System Overview

| Component | Purpose |
| --- | --- |
| [`server.py`](server.py) | Runs the Flower server, coordinates rounds, applies aggregation, and logs server metrics |
| [`client.py`](client.py) | Loads local data, trains locally, applies optional adversarial/DP/personalization logic, and logs client metrics |
| [`data_preprocessing.py`](data_preprocessing.py) | Dataset loading, preprocessing, partitioning, and temporal validation utilities |
| [`robust_aggregation.py`](robust_aggregation.py) | FedAvg, coordinate-wise median, Krum, and Bulyan implementations |
| [`privacy_accounting.py`](privacy_accounting.py) | DP epsilon accounting with Opacus-backed and analytic fallback paths |
| [`secure_aggregation.py`](secure_aggregation.py) | Additive masking primitives used by the secure aggregation scaffold |
| [`scripts/`](scripts) | Experiment orchestration, plotting, reporting, dataset preparation, and cluster automation |
| [`tests/`](tests) | Unit and integration coverage for training, analysis, plotting, and infrastructure tooling |

## Requirements

- macOS or Linux
- Python 3.13 recommended
- `nc` available on the shell path if you plan to use [`scripts/verify_readme.sh`](scripts/verify_readme.sh)
- Optional: Slurm for cluster execution

## Quick Start

### 1. Create the environment

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
make setup
```

`make setup` installs runtime and development dependencies and registers `pre-commit`.

### 2. Run the smoke verification script

```bash
bash scripts/verify_readme.sh
```

The script creates a local virtual environment if needed, launches a short synthetic federated run on two ports, and checks reproducibility.

### 3. Run a minimal manual demo

Start the server in one terminal:

```bash
rm -rf logs/smoke && mkdir -p logs/smoke
SEED=42 python server.py \
  --rounds 2 \
  --aggregation fedavg \
  --server_address 127.0.0.1:8099 \
  --logdir ./logs/smoke
```

Start two clients in separate terminals:

```bash
python client.py \
  --server_address 127.0.0.1:8099 \
  --dataset synthetic \
  --samples 1000 \
  --features 10 \
  --seed 42 \
  --client_id 0 \
  --num_clients 2 \
  --logdir ./logs/smoke
```

```bash
python client.py \
  --server_address 127.0.0.1:8099 \
  --dataset synthetic \
  --samples 1000 \
  --features 10 \
  --seed 42 \
  --client_id 1 \
  --num_clients 2 \
  --logdir ./logs/smoke
```

The server exits after the configured number of rounds and writes metrics to `./logs/smoke`.

## Datasets

The client currently supports the following dataset selectors:

| Selector | Backing data |
| --- | --- |
| `synthetic` | Generated in-memory classification data |
| `unsw` | UNSW-NB15 CSV under `data/unsw/` |
| `cic` | CIC-IDS2017 multiclass CSV under `data/cic/` |
| `edge-iiotset-quick` | Small processed Edge-IIoTset sample |
| `edge-iiotset-nightly` | Medium processed Edge-IIoTset sample |
| `edge-iiotset-full` | Large processed Edge-IIoTset sample |
| `hybrid` | Combined dataset pipeline described in [`docs/HYBRID_DATASET.md`](docs/HYBRID_DATASET.md) |

To materialize the local dataset layout expected by the codebase:

```bash
python scripts/setup_real_datasets.py
```

What the setup script does:

- extracts compressed UNSW-NB15 and CIC-IDS2017 archives from `datasets/real/` into `data/`
- links or prepares processed Edge-IIoTset tiers under `data/edge-iiotset/`
- validates that required full datasets are present

If you only want the required full datasets:

```bash
python scripts/setup_real_datasets.py --full-only
```

## Core Experiment Controls

### Server

Useful server flags:

- `--aggregation {fedavg,median,krum,bulyan}`
- `--rounds`
- `--byzantine_f`
- `--secure_aggregation`
- `--fedprox_mu`
- `--fraction_fit`, `--fraction_eval`

Inspect the full interface with:

```bash
python server.py --help
```

### Client

Useful client flags:

- `--dataset`
- `--partition_strategy {iid,dirichlet,protocol,source}`
- `--alpha`
- `--adversary_mode {none,label_flip,grad_ascent,sign_flip_topk,targeted_label}`
- `--fedprox_mu`
- `--personalization_epochs`
- `--dp_enabled`, `--dp_clip`, `--dp_noise_multiplier`, `--dp_delta`, `--dp_sample_rate`
- `--binary_classification`
- `--temporal_validation`
- `--model_arch {auto,simple,per_dataset_encoder}`

Inspect the full interface with:

```bash
python client.py --help
```

## Running Comparative Campaigns

The primary orchestration entry point is [`scripts/comparative_analysis.py`](scripts/comparative_analysis.py). It expands experiment matrices, launches the server and clients as subprocesses, and stores run outputs under a generated analysis directory.

Example: heterogeneity and FedProx comparison on CIC-IDS2017.

```bash
python scripts/comparative_analysis.py \
  --dimension heterogeneity_fedprox \
  --dataset cic \
  --aggregation-methods fedavg,median \
  --alpha-values 0.02,0.5 \
  --fedprox-mu-values 0.0,0.01 \
  --seeds 42,43 \
  --num_clients 6 \
  --num_rounds 5
```

Example: robust aggregation under adversarial participation.

```bash
python scripts/comparative_analysis.py \
  --dimension attack \
  --dataset unsw \
  --aggregation-methods fedavg,median,krum,bulyan \
  --adversary-fractions 0.0,0.2 \
  --seeds 42,43 \
  --num_clients 6 \
  --num_rounds 5
```

Review the full matrix options with:

```bash
python scripts/comparative_analysis.py --help
```

## Plotting, Census, And Reporting

Representative analysis entry points:

```bash
python scripts/plot_metrics.py \
  --run_dir ./logs/smoke \
  --output_dir ./plots/smoke
```

```bash
python scripts/runs_census.py \
  --runs-dir ./runs \
  --output-dir ./reports/runs_census
```

```bash
python scripts/generate_publication_bundle.py \
  --output-dir ./reports/publication
```

```bash
python scripts/heterogeneity_claim_report.py
```

Additional plot generators and analysis scripts are organized under [`scripts/`](scripts), including dataset-scoped thesis plot generation, caption table generation, FedProx analysis, and publication summaries.

## Cluster Execution

Slurm helpers live under [`scripts/slurm`](scripts/slurm). Examples include:

- [`scripts/slurm/submit_paper_bulyan_adv20_confirmatory.sh`](scripts/slurm/submit_paper_bulyan_adv20_confirmatory.sh)
- [`scripts/slurm/submit_iiot_full_heterogeneity.sh`](scripts/slurm/submit_iiot_full_heterogeneity.sh)
- [`scripts/slurm/submit_neurips_p0.sh`](scripts/slurm/submit_neurips_p0.sh)

For cluster-specific procedures, start with:

- [`docs/JMU_CS470_CLUSTER_RUNBOOK.md`](docs/JMU_CS470_CLUSTER_RUNBOOK.md)
- [`docs/CLUSTER_ARCHITECTURE.md`](docs/CLUSTER_ARCHITECTURE.md)
- [`docs/NEURIPS_FINAL_RUNBOOK.md`](docs/NEURIPS_FINAL_RUNBOOK.md)

## Development Workflow

Common local commands:

```bash
make format
make lint
make type
make test
make coverage
pre-commit run --all-files
```

The repository also contains `package.json` and `turbo.json` so CI can invoke a uniform wrapper command, but the JavaScript tasks are intentionally no-ops because the project is Python-first:

```bash
npx turbo run typecheck lint
```

Pytest configuration lives in [`pytest.ini`](pytest.ini). Test discovery includes both `test_*.py` and `*_spec.py`.

Contributor guidance lives in [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Repository Layout

- [`client.py`](client.py): federated client, local training, adversarial modes, DP, personalization
- [`server.py`](server.py): server strategy, aggregation, round coordination, metrics
- [`data_preprocessing.py`](data_preprocessing.py): dataset ingestion, preprocessing, partitioning, temporal validation
- [`models/`](models): network architectures including per-dataset encoders and focal loss
- [`scripts/`](scripts): orchestration, plots, reports, dataset setup, diagnostics, Slurm tooling
- [`tests/`](tests): automated coverage
- [`docs/`](docs): experiment notes, runbooks, design rationale, and publication support material
- `data/`: local extracted datasets
- `datasets/`: compressed archives and raw source material
- `logs/`, `runs/`, `results/`, `reports/`: local generated outputs

## Documentation Map

Recommended starting points:

- [`docs/README.md`](docs/README.md): documentation index
- [`docs/threat_model.md`](docs/threat_model.md): threat assumptions and defense scope
- [`docs/HYBRID_DATASET.md`](docs/HYBRID_DATASET.md): hybrid dataset design
- [`docs/TEMPORAL_VALIDATION_PROTOCOL.md`](docs/TEMPORAL_VALIDATION_PROTOCOL.md): temporal evaluation protocol
- [`SECURITY.md`](SECURITY.md): vulnerability disclosure policy

## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff).

## License

This project is released under the MIT License. See [`LICENSE`](LICENSE).

# Edge-IIoTset Experiment Plan

[WARNING] **DEPRECATED - November 14, 2025**

This document describes the original full-scale experiment strategy which **failed due to memory constraints** (exit code 143).

**See:** [compute_constraints_and_solutions.md](./compute_constraints_and_solutions.md) for:

- Root cause analysis of exit code 143 failures
- Solutions explored (larger runners, AWS, etc.)
- Final temporal distribution architecture (6 workflows, 1 per day)

---

## Original Plan (DEPRECATED)

This plan details how we will run the Edge-IIoTset experiments locally to fulfill the thesis objectives in `deliverable1/FL.txt`. It explains the dataset layout, staged execution strategy, run parameters for every comparison dimension, and the reporting/plotting workflow that culminates in a cross-dataset comparison versus CIC-IDS2017 and UNSW-NB15.

## Dataset Layout

- **Raw archive**: `datasets/edge-iiotset/Edge-IIoTset dataset/` keeps every attack/normal CSV+PCAP pair exactly as released. We never modify this directory.
- **Processed tiers**: `datasets/edge-iiotset/processed/edge_iiotset_{quick,nightly,full}.csv` are stratified samples created via `scripts/prepare_edge_iiotset_samples.py`. A symlink at `data/edge-iiotset/` points back to this processed directory so existing tooling reads the newest samples automatically.
- **Tier definitions**:
  - `quick`: 50k rows, 3 clients, 5 rounds, ~10 min per experiment (sanity grid).
  - `nightly`: 500k rows, 6 clients, 20 rounds, ~45 min per experiment (core grid).
  - `full`: 2M rows, 10 clients, 50 rounds, ~2 h per experiment (publication-quality).

All tiers preserve the full 14 attack classes + BENIGN distribution so every experiment sees realistic IIoT traffic.

## Execution Phases

| Phase   | Datasets                 | Purpose                                                          | Entry Criteria                                  | Exit Criteria                                                                       |
| ------- | ------------------------ | ---------------------------------------------------------------- | ----------------------------------------------- | ----------------------------------------------------------------------------------- |
| Phase 1 | Edge-IIoTset Full        | Generate publication-quality metrics/plots across all dimensions | Dataset assets verified, optimized runner ready | All manifests complete, plots stored under `results/thesis_plots/edge_iiotset_full` |
| Phase 2 | CIC/UNSW Baselines       | Reuse/refresh CIC & UNSW runs for matched presets                | Equivalent configs identified                   | CIC/UNSW manifests complete                                                         |
| Phase 3 | Cross-Dataset Comparison | Plot IIoT vs CIC/UNSW per objective                              | Phase 1+2 completion                            | Combined figures/tables archived                                                    |

> **Note:** Previous quick/nightly staging steps are optional now that we’re prioritizing the full 2M-sample tier. If time permits, they can still be executed for smoke-testing, but the core plan assumes direct focus on the full dataset.

## Comparison Dimensions & Parameters

We re-use `scripts/comparative_analysis.py` presets (Issue #44) with dataset overrides. Unless noted, alpha defaults to IID (1.0) and we run seeds `{42, 43, 44, 45, 46}` on nightly/full tiers; quick tier uses `{42}` to accelerate smoke tests.

### Aggregation

- Aggregators: `fedavg`, `krum`, `bulyan`, `median`.
- Clients/Rounds: quick (3/5), nightly (6/20), full (10/50).
- Commands:
  - Quick: `python scripts/comparative_analysis.py --dimension aggregation --dataset edge-iiotset-quick --num_clients 3 --num_rounds 5 --seeds 42`
  - Nightly: `python scripts/comparative_analysis.py --dimension aggregation --dataset edge-iiotset-nightly --num_clients 6 --num_rounds 20 --seeds 42,43,44,45,46`
  - Full: `python scripts/run_experiments_optimized.py --dimension aggregation --dataset-type full --workers 2`

### Heterogeneity

- Alpha sweep: `[0.02, 0.05, 0.1, 0.2, 0.5, 1.0, inf]`.
- Commands mirror aggregation with `--dimension heterogeneity`.
- FedProx variant (`--dimension heterogeneity_fedprox`) also sweeps `mu` over `[0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.08, 0.1, 0.2]` to map heterogeneity mitigation.

### Attack Resilience

- Aggregators: `fedavg`, `krum`, `bulyan`, `median`.
- Adversary fractions: `[0.0, 0.1, 0.3]`.
- Clients: 11 to satisfy Bulyan’s `n >= 4f + 3`.
- Alpha fixed at 0.5 for moderate non-IID.
- Commands: `python scripts/comparative_analysis.py --dimension attack --dataset edge-iiotset-{tier} --num_clients 11 --num_rounds {5|20|50}`.

### Privacy

- DP configs: `{noise ∈ [0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]}`, DP enabled if noise > 0.
- Alpha = 0.5 to reflect heterogeneous clients.
- Commands: `python scripts/comparative_analysis.py --dimension privacy --dataset edge-iiotset-{tier} ...`.
- Post-run we compute ε via `scripts/privacy_accounting.py` and include DP utility plots.

### Personalization

- Personalization epochs sweep: `[0, 3, 5]`.
- Alpha = 0.5; personalization delta F1 tracked via `macro_f1_global` vs `macro_f1_personalized`.
- Commands: `python scripts/comparative_analysis.py --dimension personalization --dataset edge-iiotset-{tier} ...`.

## Optimized Runner for Full Tier

For Phase 3 we run:

```bash
export MPLCONFIGDIR=$PWD/tmp/mplconfig
python scripts/run_experiments_optimized.py \
  --dimension all \
  --dataset-type full \
  --workers 2
```

This covers aggregation, heterogeneity, attack, privacy, and personalization in one pass with stateful retries and manifest logging. We re-run with `--dimension heterogeneity_fedprox` to finish the FedProx sweep.

## Validation & Plotting

After each phase:

1. `python scripts/validate_experiment_matrix.py --runs_dir runs`
2. `python scripts/check_regressions.py --runs_dir runs`
3. `pytest test_edge_iiotset_preprocessing.py test_workflow_integration.py`

Visualization steps (repeat with `edge-iiotset-quick`, `edge-iiotset-nightly`, `edge-iiotset-full` outputs):

```bash
python scripts/generate_thesis_plots.py --dimension all --runs_dir runs --output_dir results/thesis_plots/edge_iiotset_full
python scripts/plot_metrics.py --runs_dir runs --output_dir results/metrics_server
python scripts/plot_metrics_client.py --runs_dir runs --output_dir results/metrics_client
python scripts/plot_personalization_gains.py --runs_dir runs --output_dir results/personalization
python scripts/commit_plots.py --source results/thesis_plots --destination plots/$(date +%F)
```

Confusion matrices and FedProx scatter plots are emitted automatically by `generate_thesis_plots.py`; caption/performance tables come from `scripts/caption_tables.py` and `scripts/generate_performance_tables.py`.

## Cross-Dataset Comparison

Once IIoT plots are complete, we run the identical presets for CIC (`--dataset cic`) and UNSW (`--dataset unsw`) or reuse existing manifests if the timestamps match. Final comparison workflow:

1. `python scripts/generate_thesis_plots.py --dimension all --runs_dir runs_cic --output_dir results/thesis_plots/cic`
2. `python scripts/generate_thesis_plots.py --dimension all --runs_dir runs_unsw --output_dir results/thesis_plots/unsw`
3. Use `scripts/consolidate_thesis_reporting.py` and bespoke notebooks (if needed) to overlay IIoT vs CIC/UNSW macro-F1, attack resilience curves, DP privacy-utility fronts, FedProx mu sweeps, and personalization gains.

Deliverables for Phase 5 include combined PDF/PNG figures, statistical summaries, and narrative text referencing how IIoT results stack against legacy datasets. This plan ensures every thesis objective—robust aggregation, heterogeneity mitigation, attack resilience, privacy, personalization, and comparative evaluation—is addressed with the full Edge-IIoTset dataset.

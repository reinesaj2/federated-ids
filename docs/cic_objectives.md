## CIC-IDS2017 Objective Matrix

The CIC campaign evaluates all five thesis objectives on the multi-class dataset.
The experiment planner (`scripts/plan_cic_experiments.py`) emits the seed/parameter
manifest at `analysis/cic_experiments/cic_manifest.json`.

### Experiment Dimensions

| Dimension                       | Command                                                                                                                            |
| ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| Aggregation vs Byzantine rates  | `python scripts/comparative_analysis.py --dimension aggregation --dataset cic --data_path data/cic/cic_ids2017_multiclass.csv`     |
| Robust aggregation under attack | `python scripts/comparative_analysis.py --dimension attack --dataset cic --data_path data/cic/cic_ids2017_multiclass.csv`          |
| FedProx / heterogeneity sweep   | `python scripts/comparative_analysis.py --dimension heterogeneity --dataset cic --data_path data/cic/cic_ids2017_multiclass.csv`   |
| Privacy / Secure Agg sweeps     | `python scripts/comparative_analysis.py --dimension privacy --dataset cic --data_path data/cic/cic_ids2017_multiclass.csv`         |
| Personalization rounds          | `python scripts/comparative_analysis.py --dimension personalization --dataset cic --data_path data/cic/cic_ids2017_multiclass.csv` |

- Use `--dry_run` to list presets without execution (helpful while local networking is restricted).
- Each dimension uses five seeds and the expanded α/μ grids required by Issue 44.
- Runs store metrics under `runs/comp_*` and plots/summaries under `results/comparative_analysis`.

### Analysis Pipeline

1. `python scripts/generate_thesis_plots.py --dimension heterogeneity --dataset cic --runs_dir runs --output_dir analysis/cic_experiments`
   - Produces CI ribbons for the μ-grid plus `fedprox_client_scatter.png`.
   - Exports CSV summaries (`server_ci_*`, `client_ci_macro_f1.csv`).
2. Repeat with `--dimension aggregation`, `attack`, `privacy`, and `personalization` for the remaining comparisons.
3. Commit aggregated LaTeX tables from `analysis/cic_experiments/` into the thesis document.

### Runtime Guidance

- Use the full CIC dataset unless a single configuration exceeds ~5 minutes locally. If a run exceeds the limit, fall back to `--num_clients 3 --num_rounds 10` smoke values to validate wiring before resuming the full job.
- GPU is not required; CPU runs are seed reproducible.
- gRPC bindings may be blocked in restricted sandboxes. If `server.py` cannot bind (StatusCode.UNAVAILABLE), run in dry-run mode and execute on the CI runner or a workstation with local networking enabled.

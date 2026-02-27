# Runs Census Data Dictionary

## runs_registry
- `run_id`: directory name under `runs/`.
- `family`: naming family (`ds_prefixed`, `simple`, `comp_short`, `comp_other`, `other`).
- `is_partial`: whether run name contains `__partial_`.
- `has_config`, `has_metrics`: core file existence flags.
- `config_signature_id`, `metrics_signature_id`: schema signature identifiers.
- `dataset`, `aggregation`, `alpha`, `adversary_fraction`, `adv_percent`, `fedprox_mu`, `seed`: normalized config fields.
- `quality_state`: terminal quality state.
- `max_round`, `parsed_row_count`: parsed metrics integrity indicators.
- `final_metric`: final-round macro-F1-like metric from prioritized columns.

## schema_drift_config / schema_drift_metrics
- `signature_id`: schema fingerprint.
- `key_count`: number of keys/columns.
- `keys_json`: ordered key list.
- `runs_count`: number of runs with this schema.

## runs_dedup_map
- `canonical_key_id`: dedup key fingerprint.
- `canonical_run_id`: selected run for key.
- `is_canonical`: whether row is selected representative.
- `duplicate_group_size`: size of duplicate cluster.

## coverage_confirmatory / coverage_exploratory
- `slice`: analysis slice identifier.
- `n_runs`, `n_seeds`: evidence counts.
- `metric_mean`, `ci_low`, `ci_high`: summary statistics.
- `reliability_grade`: grade from seed count (`A>=10`, `B>=5`, `C>=3`, `D<3`).

## gap_inventory
- `cell_status`: `claim_eligible`, `exploratory_only`, or `missing`.

## claim_ledger
- `support_level`: `supported`, `directional`, `exploratory`, `unsupported`.

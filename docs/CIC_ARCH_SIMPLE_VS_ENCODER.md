# CIC Architecture Comparison: SimpleNet vs PerDatasetEncoderNet

## Scope

- **Dataset:** CIC-IDS2017 (multi-class).
- **New results:** `/scratch/reinesaj/federated-ids/runs/` with run names prefixed `cic_simple_` (explicit `model_arch: simple` in `config.json`).
- **Archived results:** `/scratch/reinesaj/federated-ids/archive/runs_cic_iiot_20260107_185922/` (CIC runs default to `PerDatasetEncoderNet` when `model_arch` is not specified).
- **Metrics compared (per class):**
  - `f1_per_class_after`
  - `precision_per_class`
  - `recall_per_class`
  - `f1_per_class_holdout`
- **Objective:** Determine whether architecture differences are negligible or meaningful, and whether the advantage flips under specific `(aggregation, alpha, attack_mode)` conditions.

## Architecture Context

**SimpleNet** (CIC, `model_arch: simple`):

```
Input -> Linear(?, 64) -> ReLU -> Linear(64, 32) -> ReLU -> Linear(32, num_classes)
```

**PerDatasetEncoderNet** (CIC default; `model_arch: auto` resolves to encoder for dataset `cic`):

```
Input
  -> Linear(80, 768) -> BatchNorm -> ReLU -> Dropout(0.25)
  -> Linear(768, 384) -> BatchNorm -> ReLU -> Dropout(0.25)
  -> Linear(384, 192) -> BatchNorm -> ReLU -> Dropout(0.25)
  -> Linear(192, 256) -> BatchNorm -> ReLU
  -> Linear(256, 128) -> ReLU -> Dropout(0.25)
  -> Linear(128, 64) -> ReLU -> Dropout(0.25)
  -> Linear(64, num_classes)
```

Reference defaults are encoded in `models/per_dataset_encoder.py` and selected via `client.py` (`ENCODER_DATASETS` includes `cic`).

## Data Inventory Snapshot

- **New runs found:** 2,178
- **Archived runs found:** 3,692
- **Matched runs (config parity across architectures):** 1,060

## Matching Criteria (Per Run)

Runs were matched on these fields from `config.json`:

- `aggregation`
- `alpha`
- `adversary_fraction`
- `dp_enabled`
- `dp_noise_multiplier`
- `personalization_epochs`
- `num_clients`
- `num_rounds`
- `seed`
- `fedprox_mu`
- `dataset`
- `data_path`
- `attack_mode` (fallback to `adversary_mode`, else `none`)

Runs without a config match across architectures were excluded from the paired comparison set.

## Metric Extraction and Aggregation

For each run:

1. Read all `client_*_metrics.csv`.
2. For each client, select the row with the highest `round`.
3. Parse per-class JSON fields:
   - `f1_per_class_after`
   - `precision_per_class`
   - `recall_per_class`
   - `f1_per_class_holdout`
4. Average each metric per class across clients in the run.

For the paired comparison:

5. For each matched run, compute `diff = simple - encoder` for each class.
6. Aggregate diffs across all matched runs.

Interpretation:

- **Positive diff:** SimpleNet better
- **Negative diff:** PerDatasetEncoderNet better

## Global Per-Class Results (Mean Across Matched Runs)

### f1_per_class_after

| Class | Simple | Encoder | Diff (Simple - Encoder) | n |
| --- | --- | --- | --- | --- |
| BENIGN | 0.6977 | 0.6818 | 0.0158 | 1049 |
| BOT | 0.0766 | 0.3214 | -0.2448 | 1049 |
| DDOS | 0.6187 | 0.5324 | 0.0863 | 1049 |
| DOS GOLDENEYE | 0.4317 | 0.1540 | 0.2777 | 673 |
| DOS HULK | 0.5910 | 0.4186 | 0.1723 | 673 |
| DOS SLOWHTTPTEST | 0.3273 | 0.1256 | 0.2018 | 673 |
| DOS SLOWLORIS | 0.2962 | 0.0897 | 0.2064 | 673 |
| FTP-PATATOR | 0.2931 | 0.2083 | 0.0848 | 1049 |
| HEARTBLEED | 0.0021 | 0.0000 | 0.0021 | 673 |
| INFILTRATION | 0.0183 | 0.3242 | -0.3059 | 1049 |
| PORTSCAN | 0.5599 | 0.5716 | -0.0117 | 1049 |
| SSH-PATATOR | 0.2403 | 0.0627 | 0.1776 | 1049 |
| WEB ATTACK - BRUTE FORCE | 0.0886 | 0.3427 | -0.2541 | 1049 |
| WEB ATTACK - SQL INJECTION | 0.0006 | 0.0586 | -0.0580 | 1049 |
| WEB ATTACK - XSS | 0.0582 | 0.3436 | -0.2854 | 1049 |

### precision_per_class

| Class | Simple | Encoder | Diff (Simple - Encoder) | n |
| --- | --- | --- | --- | --- |
| BENIGN | 0.7034 | 0.6946 | 0.0088 | 1049 |
| BOT | 0.0900 | 0.3146 | -0.2246 | 1049 |
| DDOS | 0.6447 | 0.5486 | 0.0961 | 1049 |
| DOS GOLDENEYE | 0.4965 | 0.1832 | 0.3133 | 673 |
| DOS HULK | 0.6242 | 0.4570 | 0.1672 | 673 |
| DOS SLOWHTTPTEST | 0.3643 | 0.1373 | 0.2269 | 673 |
| DOS SLOWLORIS | 0.3688 | 0.1032 | 0.2656 | 673 |
| FTP-PATATOR | 0.3219 | 0.2041 | 0.1179 | 1049 |
| HEARTBLEED | 0.0021 | 0.0000 | 0.0021 | 673 |
| INFILTRATION | 0.0252 | 0.3248 | -0.2996 | 1049 |
| PORTSCAN | 0.5911 | 0.5705 | 0.0206 | 1049 |
| SSH-PATATOR | 0.2689 | 0.0618 | 0.2071 | 1049 |
| WEB ATTACK - BRUTE FORCE | 0.0916 | 0.3375 | -0.2460 | 1049 |
| WEB ATTACK - SQL INJECTION | 0.0006 | 0.0435 | -0.0428 | 1049 |
| WEB ATTACK - XSS | 0.0562 | 0.3403 | -0.2842 | 1049 |

### recall_per_class

| Class | Simple | Encoder | Diff (Simple - Encoder) | n |
| --- | --- | --- | --- | --- |
| BENIGN | 0.7082 | 0.6841 | 0.0241 | 1049 |
| BOT | 0.0738 | 0.3354 | -0.2616 | 1049 |
| DDOS | 0.6153 | 0.5397 | 0.0756 | 1049 |
| DOS GOLDENEYE | 0.4151 | 0.1521 | 0.2630 | 673 |
| DOS HULK | 0.5814 | 0.4097 | 0.1717 | 673 |
| DOS SLOWHTTPTEST | 0.3240 | 0.1274 | 0.1966 | 673 |
| DOS SLOWLORIS | 0.2722 | 0.0888 | 0.1835 | 673 |
| FTP-PATATOR | 0.2934 | 0.2537 | 0.0396 | 1049 |
| HEARTBLEED | 0.0022 | 0.0000 | 0.0022 | 673 |
| INFILTRATION | 0.0170 | 0.3283 | -0.3114 | 1049 |
| PORTSCAN | 0.5688 | 0.5935 | -0.0246 | 1049 |
| SSH-PATATOR | 0.2377 | 0.0667 | 0.1710 | 1049 |
| WEB ATTACK - BRUTE FORCE | 0.0956 | 0.3517 | -0.2561 | 1049 |
| WEB ATTACK - SQL INJECTION | 0.0006 | 0.0975 | -0.0969 | 1049 |
| WEB ATTACK - XSS | 0.0656 | 0.3508 | -0.2852 | 1049 |

### f1_per_class_holdout

| Class | Simple | Encoder | Diff (Simple - Encoder) | n |
| --- | --- | --- | --- | --- |
| BENIGN | 0.6450 | 0.5622 | 0.0828 | 673 |
| BOT | 0.0745 | 0.0327 | 0.0418 | 673 |
| DDOS | 0.5641 | 0.3375 | 0.2265 | 673 |
| DOS GOLDENEYE | 0.4317 | 0.1540 | 0.2777 | 673 |
| DOS HULK | 0.5910 | 0.4186 | 0.1723 | 673 |
| DOS SLOWHTTPTEST | 0.3273 | 0.1256 | 0.2018 | 673 |
| DOS SLOWLORIS | 0.2962 | 0.0897 | 0.2064 | 673 |
| FTP-PATATOR | 0.2709 | 0.0965 | 0.1744 | 673 |
| HEARTBLEED | 0.0021 | 0.0001 | 0.0021 | 673 |
| INFILTRATION | 0.0230 | 0.0022 | 0.0208 | 673 |
| PORTSCAN | 0.5277 | 0.3830 | 0.1447 | 673 |
| SSH-PATATOR | 0.2177 | 0.0724 | 0.1453 | 673 |
| WEB ATTACK - BRUTE FORCE | 0.0888 | 0.0359 | 0.0529 | 673 |
| WEB ATTACK - SQL INJECTION | 0.0009 | 0.0001 | 0.0008 | 673 |
| WEB ATTACK - XSS | 0.0644 | 0.0339 | 0.0304 | 673 |

## Notable Global Differences (|mean diff| >= 0.05)

### f1_per_class_after

- INFILTRATION: -0.3059
- WEB ATTACK - XSS: -0.2854
- DOS GOLDENEYE: 0.2777
- WEB ATTACK - BRUTE FORCE: -0.2541
- BOT: -0.2448
- DOS SLOWLORIS: 0.2064
- DOS SLOWHTTPTEST: 0.2018
- SSH-PATATOR: 0.1776
- DOS HULK: 0.1723
- DDOS: 0.0863
- FTP-PATATOR: 0.0848
- WEB ATTACK - SQL INJECTION: -0.0580

### precision_per_class

- DOS GOLDENEYE: 0.3133
- INFILTRATION: -0.2996
- WEB ATTACK - XSS: -0.2842
- DOS SLOWLORIS: 0.2656
- WEB ATTACK - BRUTE FORCE: -0.2460
- DOS SLOWHTTPTEST: 0.2269
- BOT: -0.2246
- SSH-PATATOR: 0.2071
- DOS HULK: 0.1672
- FTP-PATATOR: 0.1179
- DDOS: 0.0961

### recall_per_class

- INFILTRATION: -0.3114
- WEB ATTACK - XSS: -0.2852
- DOS GOLDENEYE: 0.2630
- BOT: -0.2616
- WEB ATTACK - BRUTE FORCE: -0.2561
- DOS SLOWHTTPTEST: 0.1966
- DOS SLOWLORIS: 0.1835
- DOS HULK: 0.1717
- SSH-PATATOR: 0.1710
- WEB ATTACK - SQL INJECTION: -0.0969
- DDOS: 0.0756

### f1_per_class_holdout

- DOS GOLDENEYE: 0.2777
- DDOS: 0.2265
- DOS SLOWLORIS: 0.2064
- DOS SLOWHTTPTEST: 0.2018
- FTP-PATATOR: 0.1744
- DOS HULK: 0.1723
- SSH-PATATOR: 0.1453
- PORTSCAN: 0.1447
- BENIGN: 0.0828
- WEB ATTACK - BRUTE FORCE: 0.0529

## Flip Analysis: Does the Advantage Reverse Under Specific Conditions?

### Grouping and Thresholds

Groups are defined by:

- `aggregation`
- `alpha`
- `attack_mode` (fallback to `adversary_mode` or `none`)

Flip detection rules:

- **Global sign threshold:** absolute mean diff < 0.02 treated as neutral.
- **Flip threshold:** `|group mean diff| >= 0.05`
- **Minimum group size:** `n >= 3` matched runs

### Summary of Flips (Only mode=none)

Across all metrics, flips meeting the threshold were observed **only in `attack_mode=none`** groups. Adversarial modes (`grad_ascent`, `label_flip`, `targeted_label`, `sign_flip_topk`) did not produce flips at the threshold used.

#### f1_per_class_after

- BOT (global favors Encoder):
  - bulyan alpha=0.2: mean_diff=0.1125 (n=20)
  - median alpha=0.1: mean_diff=0.0897 (n=20)
  - bulyan alpha=0.1: mean_diff=0.0860 (n=20)
- DDOS (global favors Simple):
  - fedavg alpha=0.05: mean_diff=-0.5853 (n=45)
  - fedavg alpha=0.02: mean_diff=-0.4519 (n=66)
  - fedavg alpha=0.1: mean_diff=-0.3411 (n=45)
- FTP-PATATOR (global favors Simple):
  - fedavg alpha=0.05: mean_diff=-0.2747 (n=45)
  - fedavg alpha=0.02: mean_diff=-0.2336 (n=66)
  - fedavg alpha=0.1: mean_diff=-0.1411 (n=45)
- INFILTRATION (global favors Encoder):
  - median alpha=0.05: mean_diff=0.0695 (n=20)
  - median alpha=0.02: mean_diff=0.0649 (n=20)
  - bulyan alpha=0.05: mean_diff=0.0618 (n=20)
- WEB ATTACK - BRUTE FORCE (global favors Encoder):
  - bulyan alpha=0.2: mean_diff=0.1117 (n=20)
  - median alpha=0.1: mean_diff=0.1003 (n=20)
  - bulyan alpha=0.1: mean_diff=0.0847 (n=20)
- WEB ATTACK - XSS (global favors Encoder):
  - median alpha=0.1: mean_diff=0.0982 (n=20)
  - bulyan alpha=0.2: mean_diff=0.0893 (n=20)
  - krum alpha=0.1: mean_diff=0.0812 (n=20)

#### precision_per_class

- BOT (global favors Encoder):
  - bulyan alpha=0.2: mean_diff=0.1259 (n=20)
  - median alpha=0.1: mean_diff=0.1092 (n=20)
  - bulyan alpha=0.1: mean_diff=0.0954 (n=20)
- DDOS (global favors Simple):
  - fedavg alpha=0.05: mean_diff=-0.5525 (n=45)
  - fedavg alpha=0.02: mean_diff=-0.4271 (n=66)
  - fedavg alpha=0.1: mean_diff=-0.3038 (n=45)
- FTP-PATATOR (global favors Simple):
  - fedavg alpha=0.02: mean_diff=-0.2369 (n=66)
  - fedavg alpha=0.05: mean_diff=-0.2080 (n=45)
  - fedavg alpha=0.1: mean_diff=-0.1083 (n=45)
- INFILTRATION (global favors Encoder):
  - median alpha=0.05: mean_diff=0.0880 (n=20)
  - median alpha=0.02: mean_diff=0.0734 (n=20)
  - bulyan alpha=0.02: mean_diff=0.0719 (n=20)
- PORTSCAN (global near neutral):
  - fedavg alpha=0.05: mean_diff=-0.6244 (n=45)
  - fedavg alpha=0.02: mean_diff=-0.4966 (n=66)
  - fedavg alpha=0.1: mean_diff=-0.4757 (n=45)
- WEB ATTACK - BRUTE FORCE (global favors Encoder):
  - bulyan alpha=0.2: mean_diff=0.1138 (n=20)
  - median alpha=0.1: mean_diff=0.1004 (n=20)
  - median alpha=0.05: mean_diff=0.0933 (n=20)
- WEB ATTACK - XSS (global favors Encoder):
  - median alpha=0.1: mean_diff=0.0932 (n=20)
  - bulyan alpha=0.2: mean_diff=0.0865 (n=20)
  - krum alpha=0.1: mean_diff=0.0801 (n=20)

#### recall_per_class

- BENIGN (global near neutral):
  - fedavg alpha=0.02: mean_diff=-0.4015 (n=66)
  - fedavg alpha=0.05: mean_diff=-0.3835 (n=45)
  - fedavg alpha=0.1: mean_diff=-0.2466 (n=45)
- BOT (global favors Encoder):
  - bulyan alpha=0.2: mean_diff=0.1027 (n=20)
  - median alpha=0.1: mean_diff=0.0786 (n=20)
  - bulyan alpha=0.1: mean_diff=0.0767 (n=20)
- DDOS (global favors Simple):
  - fedavg alpha=0.05: mean_diff=-0.6102 (n=45)
  - fedavg alpha=0.02: mean_diff=-0.4655 (n=66)
  - fedavg alpha=0.1: mean_diff=-0.3632 (n=45)
- FTP-PATATOR (global near neutral):
  - fedavg alpha=0.05: mean_diff=-0.4203 (n=45)
  - fedavg alpha=0.02: mean_diff=-0.3258 (n=66)
  - fedavg alpha=0.1: mean_diff=-0.2743 (n=45)
- INFILTRATION (global favors Encoder):
  - median alpha=0.05: mean_diff=0.0659 (n=20)
  - median alpha=0.02: mean_diff=0.0600 (n=20)
  - bulyan alpha=0.05: mean_diff=0.0587 (n=20)
- PORTSCAN (global near neutral, many flips):
  - krum alpha=0.2: mean_diff=0.2161 (n=20)
  - bulyan alpha=0.2: mean_diff=0.1886 (n=20)
  - fedprox alpha=0.2: mean_diff=0.1728 (n=51)
- WEB ATTACK - BRUTE FORCE (global favors Encoder):
  - bulyan alpha=0.2: mean_diff=0.1181 (n=20)
  - median alpha=0.1: mean_diff=0.1074 (n=20)
  - median alpha=0.05: mean_diff=0.0819 (n=20)
- WEB ATTACK - XSS (global favors Encoder):
  - median alpha=0.1: mean_diff=0.1042 (n=20)
  - bulyan alpha=0.2: mean_diff=0.1038 (n=20)
  - krum alpha=0.1: mean_diff=0.0912 (n=20)

#### f1_per_class_holdout

No flips meeting the thresholds were detected for holdout F1.

### Groups With the Most Flips (mode=none)

- bulyan alpha=0.1: 11 classes
- median alpha=0.1: 11 classes
- bulyan alpha=0.2: 10 classes
- krum alpha=0.1: 10 classes
- krum alpha=0.2: 10 classes
- median alpha=0.05: 9 classes
- fedavg alpha=0.02: 8 classes
- fedavg alpha=0.05: 8 classes
- fedavg alpha=0.1: 8 classes
- bulyan alpha=0.05: 7 classes
- fedprox alpha=0.2: 7 classes
- krum alpha=0.02: 7 classes
- fedavg alpha=0.2: 5 classes
- median alpha=0.02: 4 classes
- bulyan alpha=0.02: 3 classes

## Interpretation

- The difference between architectures is **not negligible** at the per-class level. Several classes show large shifts (0.20–0.31) in F1/precision/recall.
- The **encoder strongly improves** BOT, INFILTRATION, and all WEB ATTACK classes globally.
- **SimpleNet strongly improves** the DoS family (GoldeneEye/Hulk/SlowHTTPTest/Slowloris), plus SSH/FTP-Patator and often DDOS.
- Architecture advantage **does flip** under specific conditions, especially:
  - **FedAvg with low alpha** (encoder gains on DDOS/FTP‑PATATOR and even BENIGN recall).
  - **Bulyan/Median with alpha 0.1–0.2** (SimpleNet gains on BOT/INFILTRATION/WEB ATTACK classes).
- Holdout F1 differences are **stable** (no flips at threshold).

## Notes and Limitations

- This comparison is **paired** only where configs match exactly; unmatched runs are excluded.
- Class counts (`n`) vary by metric/class depending on which runs recorded those class metrics.
- `alpha` is stored as string `"inf"` in config for alpha=inf and grouped as a string.
- Flips were not observed in adversarial modes at the thresholds used; this could be due to fewer matched runs or smaller effect sizes in those settings.


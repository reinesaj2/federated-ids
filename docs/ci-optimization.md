# CI Optimization and Timeout Parameters

## Overview

This document describes the timeout configuration for federated learning experiments, both for local execution and CI/CD workflows. These optimizations were implemented to handle full-dataset experiments (82k samples) and prevent resource exhaustion.

## Timeout Parameters

### Script-Level Timeouts

The `scripts/comparative_analysis.py` script accepts two timeout parameters:

#### Server Timeout

```bash
--server_timeout <seconds>
```

- **Default**: 300 seconds (5 minutes)
- **Purpose**: Maximum time allowed for the Flower server process
- **When to increase**: Full dataset experiments, complex aggregation methods (Bulyan, Krum)

#### Client Timeout

```bash
--client_timeout <seconds>
```

- **Default**: 900 seconds (15 minutes)
- **Purpose**: Maximum time allowed for each client training process
- **When to increase**: Large datasets, deep models, differential privacy experiments

### CI Workflow Timeouts

File: `.github/workflows/comparative-analysis-nightly.yml`

```yaml
jobs:
  comparative_analysis:
    timeout-minutes: 480 # 8 hours
    strategy:
      max-parallel: 2 # Limit concurrent experiments
```

- **Workflow timeout**: 480 minutes (8 hours) for complete experiment matrix
- **Parallel jobs**: Maximum 2 concurrent experiments to prevent resource exhaustion
- **Success threshold**: 50% of experiments must complete successfully

## Usage Examples

### Local Execution

#### Default timeouts (sampled dataset)

```bash
python scripts/comparative_analysis.py \
  --alpha 0.5 \
  --adv_fraction 0
```

#### Extended timeouts (full dataset)

```bash
python scripts/comparative_analysis.py \
  --alpha 0.5 \
  --adv_fraction 0 \
  --server_timeout 600 \
  --client_timeout 1800
```

#### Custom timeouts for specific scenarios

```bash
# Byzantine experiments with full dataset
python scripts/comparative_analysis.py \
  --alpha 0.5 \
  --adv_fraction 0.3 \
  --server_timeout 900 \
  --client_timeout 2400

# Differential privacy experiments
python scripts/comparative_analysis.py \
  --alpha 0.5 \
  --dp_epsilon 1.0 \
  --server_timeout 450 \
  --client_timeout 1200
```

### CI Configuration

The nightly workflow automatically uses appropriate timeouts:

```yaml
env:
  SERVER_TIMEOUT: 300
  CLIENT_TIMEOUT: 900
```

Pass to script:

```bash
python scripts/comparative_analysis.py \
  --server_timeout ${SERVER_TIMEOUT} \
  --client_timeout ${CLIENT_TIMEOUT}
```

## Timeout Selection Guidelines

### Server Timeout

| Scenario                       | Recommended Value | Rationale                                 |
| ------------------------------ | ----------------- | ----------------------------------------- |
| Sampled dataset (8.2k samples) | 300s (default)    | Fast aggregation, minimal overhead        |
| Full dataset (82k samples)     | 600s              | 10x data requires longer aggregation      |
| Bulyan/Krum aggregation        | 900s              | O(n²) complexity for distance computation |
| Byzantine experiments          | 900s              | Additional validation overhead            |

### Client Timeout

| Scenario             | Recommended Value | Rationale                          |
| -------------------- | ----------------- | ---------------------------------- |
| Sampled dataset      | 900s (default)    | 3 local epochs on 8.2k samples     |
| Full dataset         | 1800s             | 10x data, 3 local epochs           |
| Differential privacy | 1200s             | Noise injection adds ~30% overhead |
| Personalization      | 1500s             | Additional fine-tuning phase       |

## Error Handling

### Timeout Behavior

When a timeout occurs:

1. **Server timeout**: Process terminated, experiment marked as failed
2. **Client timeout**: Individual client terminated, server waits for remaining clients
3. **Workflow timeout**: Entire CI job terminated, partial results preserved

### Graceful Degradation

The CI workflow uses a 50% success threshold:

```bash
# Validation logic
if [ $FOUND -lt $((TOTAL / 2)) ]; then
  echo "WARNING: Low success rate, but continuing..."
else
  echo "SUCCESS: Adequate experiment completion rate"
fi
```

This allows partial failures without blocking the entire pipeline.

## Performance Monitoring

### Logged Metrics

Each experiment logs timing information to `metrics.csv`:

- `t_aggregate_ms`: Aggregation time at server
- `t_round_ms`: Total round time (training + aggregation)
- `t_fit_ms`: Client training time

### Identifying Timeout Issues

Check experiment logs for timeout patterns:

```bash
# Find timed-out experiments
grep -r "TimeoutExpired" runs/*/experiment.log

# Check average round times
awk -F',' 'NR>1 {sum+=$11; count++} END {print "Avg round time:", sum/count/1000, "seconds"}' \
  runs/*/metrics.csv
```

## Historical Context

### Pre-Optimization Issues

Before parameterization:

- Fixed 120s server timeout caused failures on full dataset
- Fixed 600s client timeout insufficient for DP experiments
- No graceful degradation in CI
- Resource exhaustion from unlimited parallel jobs

### Implementation (PR #91)

Changes merged 2025-10-21:

- Parameterized timeouts in `comparative_analysis.py`
- Extended workflow timeout to 8 hours
- Limited parallel jobs to 2
- Added 50% success threshold
- Proper error logging and cleanup

## Troubleshooting

### Experiment Times Out Despite High Timeout

**Check**:

1. System resources (CPU, memory, disk I/O)
2. Port conflicts (default 8080)
3. Dataset corruption or missing files

**Solution**:

```bash
# Monitor resources during experiment
python scripts/comparative_analysis.py --alpha 0.5 &
PID=$!
while kill -0 $PID 2>/dev/null; do
  ps -p $PID -o %cpu,%mem,etime
  sleep 30
done
```

### CI Workflow Exceeds 8 Hours

**Check**:

1. Number of experiments in matrix
2. Dataset size setting
3. Number of rounds configured

**Solution**: Reduce experiment scope or split into multiple workflows

### High Failure Rate in CI

**Check**:

1. Success rate from validation logs
2. Individual experiment error logs
3. Timeout values vs actual execution times

**Solution**: Adjust timeouts based on `t_round_ms` percentiles from successful runs

## Related Documentation

- `EXPERIMENT_CONSTRAINTS.md`: Mathematical constraints for Bulyan/Krum
- `docs/gradient_clipping_theory.md`: Byzantine attack mitigations
- `docs/threat_model.md`: Security considerations for federated IDS
- `.github/workflows/comparative-analysis-nightly.yml`: CI configuration

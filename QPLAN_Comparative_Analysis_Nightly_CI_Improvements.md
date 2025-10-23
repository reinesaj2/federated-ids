# QPLAN: Comparative Analysis Nightly CI Improvements (Issue #44)

## Executive Summary

Comparative Analysis Nightly Run #29 reveals systemic CI/infrastructure issues in the federated IDS experimental framework. This QPLAN addresses three critical areas:

1. **Workflow Reliability** - 50% success threshold allows incomplete thesis datasets
2. **Resource Efficiency** - 15-hour timeout with sequential jobs; unnecessary parallelization constraints
3. **Observability & Debugging** - Limited logging, no real-time failure diagnosis, opaque aggregation logic

**Recommendation**: Implement tiered approach with immediate quick-wins and longer-term architectural improvements.

---

## 1. Issue Analysis

### 1.1 Current State (Baseline)

**Workflow**: `.github/workflows/comparative-analysis-nightly.yml`

- **Execution Model**: Sequential job matrix (max-parallel: 1)
- **Timeout**: 900 minutes (15 hours) per workflow run
- **Success Criteria**: 50% of experiments must complete (allows ~27 failures out of 54 viable experiments)
- **Dimensions Tested**: 6 (aggregation, heterogeneity, heterogeneity_fedprox, attack, privacy, personalization)
- **Datasets**: UNSW-NB15 + CIC-IDS2017 (optional)
- **Total Viable Experiments**: 54 (3 Bulyan+30% combinations are mathematically impossible per EXPERIMENT_CONSTRAINTS.md)

**Key Metrics**:
- Nightly run starts: Every Sunday 3 AM UTC (cron trigger)
- Manual triggers: Supported via `workflow_dispatch`
- Artifacts: 90-day retention
- Plot generation: Post-processing step with Git push to main

### 1.2 Identified Problems

#### Problem 1: Insufficient Success Threshold (Risk Level: HIGH)

**Current Policy**: Accept 50% success (≥27/54 experiments)

**Impact**:
- Thesis data may be incomplete for key dimensions (e.g., only 13/18 attack experiments)
- Inconsistent coverage across random seeds (42-46) creates reproducibility gaps
- Missing data breaks comparative analysis (e.g., "robustness across adversary fractions")

**Root Cause**: 
- No retry logic for failed experiments
- No partial-failure triage to identify systematic vs. transient issues
- Workflow cascades to plot generation even with incomplete metrics

**Example Failure Scenario**:
```
Expected: 54 experiments across 6 dimensions
Actual: 27 experiments complete → workflow considers this "success"
Missing: All CIC+attack+DP combinations (18 critical experiments)
Result: Thesis comparative tables have gaps, conclusions lack statistical rigor
```

#### Problem 2: Inadequate Observability (Risk Level: MEDIUM)

**Current Logging**:
- Console echo of system resources (memory, CPU, disk) at job start
- Validation step counts completed experiments but provides no failure diagnosis
- No per-experiment timing breakdown
- No real-time progress tracking
- Aggregation failures are buried in experiment logs without surfacing root causes

**Impact**:
- Operator cannot determine WHY an experiment failed (timeout, resource exhaustion, data corruption, bug)
- Post-mortem analysis requires manual log examination across 54+ files
- Nightly failures go undetected until morning review

**Current Validation Step** (lines 138-170):
```bash
for dir in runs/comp_*/; do
  if [ -f "$dir/metrics.csv" ]; then
    FOUND=$((FOUND + 1))
  else
    FAILED=$((FAILED + 1))
  fi
done
```
→ **Provides only counts, no diagnosis**

#### Problem 3: Inefficient Parallelization Strategy (Risk Level: MEDIUM)

**Current Configuration**:
- `max-parallel: 1` enforces sequential execution
- `timeout-minutes: 900` (15 hours) for complete matrix
- Rationale: "Sequential execution for expanded grids" (line 45 comment)

**Impact**:
- Ubuntu runner idles 80%+ of runtime while single dimension completes
- Job time could be reduced from 15 hours to 4-5 hours with `max-parallel: 2-3`
- Resource headroom unused (CI environment has 4 vCPU, 16GB RAM available)

**Opportunity**:
```
Dimensions that DON'T conflict (can run in parallel):
- aggregation + privacy (different experimental concerns)
- heterogeneity + personalization (orthogonal factors)
- heterogeneity_fedprox + attack (different aggregation/attack modes)

Bottleneck: Only require 1 shared port pool (8080+) per dimension
```

#### Problem 4: Weak Byzantine Constraint Handling (Risk Level: MEDIUM)

**Current State** (per EXPERIMENT_CONSTRAINTS.md):
- Bulyan requires n ≥ 4f + 3
- With 6 clients and 30% adversaries (f=2): n ≥ 11, actual=6 → VIOLATES constraint
- Workflow attempts all 3 Bulyan+30%+seed(42,43,44) experiments
- All 3 fail immediately with cryptic error
- Validation logic still marks them as "failed" and reduces success count

**Current Workaround**: Document in EXPERIMENT_CONSTRAINTS.md that 3/57 experiments are impossible

**Better Approach**: 
- Validate constraints upfront (pre-flight check)
- Skip impossible experiments programmatically
- Adjust experiment matrix to respect Byzantine limits
- Make 54 viable experiments (54/54 target, not 54/57)

---

## 2. Root Cause Analysis

### 2.1 Architectural Issues

| Issue | Root Cause | Evidence |
|-------|-----------|----------|
| Low success threshold | Overly pessimistic assumptions about FL stability | Hard-coded 50% in validation; no tuning over time |
| Sequential execution | Insufficient test of parallelization; fear of resource exhaustion | Comment on line 45; no metrics backing sequential choice |
| Weak observability | Early-stage CI implementation; focused on "does it complete?" not "why did it fail?" | Validation step only counts, no structured logging |
| Byzantine constraint conflicts | Experiment matrix generation doesn't validate feasibility | comparative_analysis.py generates 57, 3 are impossible |
| Single dataset limitation | CIC experiments tied to UNSW throughput (--run_cic optional) | Line 92: run_cic conditional; no parallel dataset runs |

### 2.2 Systemic Patterns

**Pattern 1: Fire-and-Forget Experiments**
- Script launches subprocess, waits for completion, checks metrics.csv exists
- No intermediate checkpoints or health indicators
- Failure is "no metrics.csv" not "process crashed at round 5 of 20"

**Pattern 2: Weak Upstream Validation**
- Experiment matrix (54 theoretical) includes 3 mathematically impossible cases
- Client count (6) known to violate Bulyan's safety threshold for 30% attacks
- No feasibility check before job matrix expansion

**Pattern 3: Siloed Dimensions**
- 6 separate jobs run sequentially, each unaware of siblings' progress
- Consolidated summary (job 2) tries to aggregate, but depends on all job 1 completions
- Cascading dependency: delayed aggregation if any dimension fails

---

## 3. QPLAN: Immediate Improvements (Scrum Analysis)

### 3.1 Recommended Branch & Scope

**Current Branch**: `exp/issue-44-comprehensive-experiments` (worktree active)

**Scope**: This work is scoped to **Phase 1: CI Infrastructure** of Issue #44.

**Recommendation**: Create sub-branch for CI improvements
```
Branch: exp/issue-44-ci-improvements (cut from exp/issue-44-comprehensive-experiments)
Scope: Fix observable, high-ROI CI issues
Duration: 1-2 sprints
Outcomes: 
  - 100% experiment completion rate (54/54 viable)
  - Real-time failure diagnosis
  - 50% reduction in runtime
```

### 3.2 Agile Prioritization (Moscow Method)

#### MUST (Blocking Thesis Completeness)

1. **M1: Increase Success Threshold to 95%** ⏱️ 30 min
   - Require 51/54 experiments (allow 3 transient failures)
   - Rationale: Only 5.6% failure tolerance; identifies systematic issues immediately
   - Implementation: Change line 162 in workflow from `FOUND -lt $((TOTAL / 2))` → `FOUND -lt $((TOTAL * 95 / 100))`
   - Acceptance: Workflow fails if <51 experiments complete

2. **M2: Pre-Flight Byzantine Constraint Validation** ⏱️ 1-2 hours
   - Add `scripts/validate_experiment_matrix.py` to check feasibility
   - Validate: client count, Bulyan f ≤ 2, Krum f ≤ 4
   - Remove mathematically impossible experiments from matrix
   - Add to workflow as first step before experiments run
   - Acceptance: Only 54 viable experiments in matrix; 0 constraint violations

3. **M3: Structured Failure Diagnosis Logging** ⏱️ 2-3 hours
   - Instrument `comparative_analysis.py` to emit per-experiment logs with:
     - Start time, end time, wall-clock duration
     - Process exit code, stderr capture
     - Metric CSV row count (early indicator of early termination)
     - Stage reached (setup, training, aggregation, validation)
   - Add validation step to parse logs and categorize failures (timeout vs. error)
   - Generate summary: "X experiments timed out, Y experiments failed with error Z"
   - Acceptance: Each failed experiment has categorized root cause in logs

#### SHOULD (High-Impact Efficiency)

4. **S1: Enable Safe Parallelization** ⏱️ 1-2 hours
   - Change `max-parallel: 1` → `max-parallel: 2`
   - Use separate ports per job (8080 for agg, 8081 for het, 8082 for privacy, etc.)
   - Reduce timeout from 900 → 600 minutes (worst case: 2 slow dimensions run simultaneously)
   - Expected time savings: 15 hours → 7-8 hours per run
   - Testing: Run on `main` or test workflow first
   - Acceptance: 2 jobs complete in parallel; no port conflicts; logs show proper isolation

5. **S2: Independent Dataset Execution** ⏱️ 1-2 hours
   - Remove `--run_cic` conditional coupling
   - Run UNSW and CIC in separate parallel jobs (don't cascade)
   - Allows thesis to have CIC data even if UNSW times out (or vice versa)
   - Acceptance: CIC and UNSW experiments run on separate runners

6. **S3: Implement Retry Logic for Transient Failures** ⏱️ 2-3 hours
   - Wrap each experiment in retry loop: `for attempt in 1 2 3`
   - Retry only on timeout exit code (124), not on validation errors
   - Max 3 attempts per experiment
   - Log "RETRY" attempt number
   - Acceptance: Transient network/resource issues don't block reproducible theses

#### COULD (Medium-Term UX)

7. **C1: Real-Time Progress Dashboard** ⏱️ 4-6 hours
   - Create GitHub status page or artifact-based dashboard
   - Show per-job completion %, memory usage, ETA
   - Update on each experiment completion
   - Link from job summary to detailed metrics

8. **C2: Drift-Detection for Metric Regressions** ⏱️ 3-4 hours
   - Compare latest run metrics to baseline (commit <N>)
   - Alert if accuracy drops >5% or convergence slows >20%
   - Helps catch introduced bugs early

---

## 4. Implementation Sequence (Scrum Timeline)

### Sprint 1: Foundation (Days 1-2)

**Goal**: Fix high-risk, fast-win issues → 95% success rate

| Task | Owner | Est. | Inputs | Outputs |
|------|-------|------|--------|---------|
| **M1**: Increase success threshold | TBD | 30 min | `.github/workflows/comparative-analysis-nightly.yml` | Updated threshold to 95% |
| **M2**: Add constraint validator | TBD | 2 hrs | `comparative_analysis.py`, math spec | `scripts/validate_experiment_matrix.py` + integrated into workflow |
| **M3**: Add per-experiment logging | TBD | 3 hrs | `comparative_analysis.py` | Structured logs with failure categorization |
| **Test on branch**: Run manual workflow trigger | TBD | 1-2 hrs | Branched workflow | Green run with 54/54 experiments, clear diagnostics |

**Acceptance Criteria**:
- [ ] Workflow rejects runs with <51/54 experiments
- [ ] No mathematically impossible experiments in matrix
- [ ] Each failed experiment has categorized root cause
- [ ] Manual trigger on branch shows improvements

### Sprint 2: Optimization (Days 3-4)

**Goal**: Reduce runtime 50%, decouple dataset constraints

| Task | Owner | Est. | Inputs | Outputs |
|------|-------|------|--------|---------|
| **S1**: Enable safe parallelization | TBD | 2 hrs | Workflow, port isolation plan | Updated workflow, port mapping strategy |
| **S2**: Independent dataset execution | TBD | 2 hrs | Workflow job structure | UNSW + CIC run in parallel jobs |
| **S3**: Add retry logic | TBD | 3 hrs | Error handling, timeout detection | Transient failure resilience |
| **Integration test**: Run full matrix | TBD | 2-3 hrs | All improvements | Verify 2 dimensions run in parallel; metrics complete |

**Acceptance Criteria**:
- [ ] Workflow completes in 7-8 hours (vs. 15 hours)
- [ ] CIC data available even if UNSW times out
- [ ] Transient timeouts retried automatically
- [ ] All 54 experiments consistently complete

### Sprint 3: Observability (Days 5-6) [OPTIONAL, if time]

**Goal**: Improve debugging UX for operators

| Task | Owner | Est. | Inputs | Outputs |
|------|-------|------|--------|---------|
| **C1**: Progress dashboard | TBD | 5 hrs | Job summaries, artifact API | Real-time status view |
| **C2**: Drift detection | TBD | 4 hrs | Metric history, baseline | Regression alert system |

---

## 5. Technical Details (Implementation Guide)

### 5.1 M1: Update Success Threshold

**File**: `.github/workflows/comparative-analysis-nightly.yml` line 162

**Current**:
```bash
if [ $FOUND -lt $((TOTAL / 2)) ]; then
  echo "WARNING: Low success rate, but continuing..."
```

**Proposed**:
```bash
THRESHOLD=$((TOTAL * 95 / 100))  # 95% threshold
if [ $FOUND -lt $THRESHOLD ]; then
  echo "ERROR: Success rate $((FOUND * 100 / TOTAL))% below required 95% ($FOUND/$TOTAL)"
  exit 1
else
  echo "SUCCESS: $FOUND/$TOTAL experiments completed ($(((FOUND * 100) / TOTAL))%)"
fi
```

**Rationale**: Forces attention to systematic issues; only allows 3 transient failures across 54 experiments.

---

### 5.2 M2: Pre-Flight Byzantine Constraint Validation

**New Script**: `scripts/validate_experiment_matrix.py`

```python
"""Validate experiment matrix feasibility before execution."""

def validate_bulyan_safety(n_clients: int, adversary_fraction: float) -> bool:
    """Check Bulyan requirement: n >= 4f + 3"""
    f = int(n_clients * adversary_fraction)
    required = 4 * f + 3
    return n_clients >= required

def validate_krum_safety(n_clients: int, adversary_fraction: float) -> bool:
    """Check Krum requirement: n >= 2f + 3"""
    f = int(n_clients * adversary_fraction)
    required = 2 * f + 3
    return n_clients >= required

def main():
    # Load experiment matrix
    matrix = ComparisonMatrix()
    
    viable_count = 0
    impossible_count = 0
    
    for preset in generate_experiment_presets(matrix):
        if preset['aggregation'] == 'bulyan':
            if not validate_bulyan_safety(matrix.num_clients, preset['adv_fraction']):
                print(f"SKIP (impossible): {preset['name']}")
                impossible_count += 1
                continue
        
        viable_count += 1
    
    print(f"Viable experiments: {viable_count}")
    print(f"Impossible: {impossible_count}")
    print(f"Target: {viable_count} experiments will run")
```

**Integration in Workflow** (new step before experiments):
```yaml
- name: Validate experiment matrix
  run: |
    python scripts/validate_experiment_matrix.py > matrix_validation.txt
    echo "Validated experiment matrix:"
    cat matrix_validation.txt
```

---

### 5.3 M3: Structured Failure Diagnosis

**Enhanced `comparative_analysis.py`** (add per-experiment wrapper):

```python
@contextmanager
def run_experiment_with_diagnostics(config, output_dir):
    """Run experiment with structured logging and failure categorization."""
    
    import time
    import subprocess
    
    log_file = output_dir / f"{config.to_preset_name()}.log"
    metrics_file = output_dir / f"{config.to_preset_name()}_metrics.csv"
    
    start_time = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Starting: {config.to_preset_name()}")
    
    try:
        # Launch experiment
        proc = subprocess.Popen(
            ["python", "server.py", ...],
            stdout=open(log_file, "w"),
            stderr=subprocess.STDOUT,
            timeout=config.server_timeout
        )
        exit_code = proc.wait()
        
    except subprocess.TimeoutExpired:
        proc.kill()
        print(f"TIMEOUT: {config.to_preset_name()} exceeded {config.server_timeout}s")
        emit_diagnostic("timeout", config)
        return
    except Exception as e:
        print(f"ERROR: {config.to_preset_name()}: {e}")
        emit_diagnostic("error", config)
        return
    
    elapsed = time.time() - start_time
    
    # Check for metrics
    if not metrics_file.exists():
        print(f"FAILED: No metrics for {config.to_preset_name()}")
        emit_diagnostic("no_metrics", config)
    else:
        row_count = len(open(metrics_file).readlines())
        print(f"SUCCESS: {config.to_preset_name()} ({row_count} rows, {elapsed:.1f}s)")
```

**Diagnostic Output Format** (structured JSON):
```json
{
  "experiment": "comp_bulyan_alpha0.5_adv30_dp0_pers0_seed42",
  "status": "failed",
  "failure_reason": "constraint_violation",
  "error_message": "Bulyan requires n >= 4f + 3; got n=6, f=2, required=11",
  "duration_seconds": 2.3,
  "timestamp": "2025-10-23T14:22:00Z"
}
```

**Validation Step** (enhanced; parse diagnostics):
```bash
echo "=== FAILURE ANALYSIS ==="
grep '"failure_reason"' runs/comp_*/*.diagnostic.json | \
  awk -F':' '{print $NF}' | sort | uniq -c | sort -rn

# Output example:
#   15 "timeout"
#   3 "constraint_violation"
#   0 "error"
```

---

### 5.4 S1: Safe Parallelization Strategy

**Port Assignment** (prevent collisions):
```bash
# Workflow matrix job: ${{ matrix.dimension }}
# Port assignment:
# - aggregation: 8080
# - heterogeneity: 8081
# - heterogeneity_fedprox: 8082
# - attack: 8083
# - privacy: 8084
# - personalization: 8085
```

**Updated Workflow**:
```yaml
strategy:
  max-parallel: 2
  fail-fast: false
  matrix:
    dimension: [aggregation, heterogeneity, heterogeneity_fedprox, attack, privacy, personalization]

steps:
  - name: Run comparative analysis
    env:
      PORT_OFFSET: ${{ matrix.port_offset }}
    run: |
      BASE_PORT=8080
      PORT=$((BASE_PORT + $PORT_OFFSET))
      python scripts/comparative_analysis.py \
        --dimension ${{ matrix.dimension }} \
        --port $PORT
```

**Expected Runtime Reduction**:
- Sequential (current): 6 dimensions × 1.5-2.5 hrs each = ~15 hours
- Parallel (2 jobs): 6 dimensions / 2 = 3 batches × 2.5 hrs = ~7.5 hours
- **Savings**: ~47%

---

### 5.5 S2: Decouple Datasets

**Current Workflow** (cascading):
```yaml
- name: Run UNSW experiments
  run: python scripts/comparative_analysis.py --dataset unsw

- name: Run CIC experiments
  if: github.event.inputs.run_cic != 'false'
  run: python scripts/comparative_analysis.py --dataset cic
```

**Problem**: If UNSW times out at 7 hours, CIC never runs. Both datasets missing from thesis.

**Proposed** (parallel datasets):
```yaml
jobs:
  comparative_analysis_unsw:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        dimension: [aggregation, heterogeneity, attack, privacy, personalization]
    steps:
      - run: python scripts/comparative_analysis.py --dataset unsw --dimension ${{ matrix.dimension }}

  comparative_analysis_cic:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        dimension: [aggregation, heterogeneity, attack, privacy, personalization]
    steps:
      - run: python scripts/comparative_analysis.py --dataset cic --dimension ${{ matrix.dimension }}
```

**Benefit**: Independent failure domains; CIC data available even if UNSW fails.

---

## 6. Risk Assessment & Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| Parallelization causes port conflicts | Medium | High (workflow hangs) | Use port offset strategy; test locally first |
| Retry logic masks real bugs | Low | Medium (delays discovery) | Log all retries; alert if same experiment fails 3x |
| Byzantine constraint validation misses edge case | Low | Medium (impossible experiments run) | Unit test validator against known cases |
| 95% threshold too strict (fails on transient glitches) | Medium | Low (temporary; retry helps) | Monitor first week; adjust to 90-93% if needed |

---

## 7. Acceptance Criteria (Definition of Done)

### Phase 1: Success Threshold & Constraints (M1, M2)
- [ ] Workflow enforces 95% success rate (≥51/54 experiments)
- [ ] No mathematically impossible experiments in matrix
- [ ] Manual trigger produces 54/54 viable experiments
- [ ] All 54 experiments complete consistently (10 consecutive runs)

### Phase 2: Observability & Diagnosis (M3)
- [ ] Each experiment generates `*.diagnostic.json` with failure reason
- [ ] Validation step categorizes failures: timeout vs. error vs. constraint vs. success
- [ ] Failure summary is human-readable (e.g., "3 timeouts, 0 errors, 51 success")
- [ ] Logs allow post-mortem of specific experiments in <5 minutes

### Phase 3: Runtime Optimization (S1, S2)
- [ ] Parallelization reduces 15-hour runtime to <8 hours
- [ ] 2 jobs (e.g., aggregation + privacy) run simultaneously with no conflicts
- [ ] UNSW and CIC run in parallel; results available independently
- [ ] Retry logic automatically recovers from transient failures
- [ ] 95% success rate maintained under parallelization

---

## 8. Related Documentation

- `EXPERIMENT_CONSTRAINTS.md` – Byzantine resilience math (Bulyan n ≥ 4f+3)
- `docs/ci-optimization.md` – Timeout parameter tuning
- `scripts/comparative_analysis.py` – Experiment matrix generation
- `.github/workflows/comparative-analysis-nightly.yml` – Current CI config

---

## 9. Sign-Off & Next Steps

**Scrum Master Assessment** (Agile Analysis):

| Criterion | Status | Rationale |
|-----------|--------|-----------|
| **Consistency with Codebase** | ✅ | Improvements align with existing patterns (port mgmt, error categorization) |
| **Minimal Changes** | ✅ | M1-M3 are isolated, low-touch changes; S1-S2 are modular |
| **Reuses Existing Code** | ✅ | Leverage existing validation step structure; wrap existing experiment logic |
| **Branch Strategy** | ✅ | Create sub-branch `exp/issue-44-ci-improvements` from current worktree |
| **Estimated Velocity** | ✅ | M-level items: 5-6 hours; S-level: 4-6 hours; total Sprint 1-2: ~1-2 weeks |
| **Risk Level** | ✅ | LOW-MEDIUM; changes are backward-compatible; can be reverted if issues arise |

**Recommended Action**:
1. Review this QPLAN for alignment with thesis objectives
2. Approve scoping & prioritization (MUST vs. SHOULD vs. COULD)
3. Create branch: `exp/issue-44-ci-improvements`
4. Proceed with Sprint 1 (M1-M3) to establish foundation
5. Evaluate results after first manual test run; proceed to Sprint 2 if successful

---

**Date Prepared**: 2025-10-23
**Status**: DRAFT (awaiting user review & approval)


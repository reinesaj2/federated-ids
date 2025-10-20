# Experiment Constraints and Mathematical Impossibilities

## Summary

**Total unique experiments in comparative analysis**: 57
**Achievable experiments**: 54
**Mathematically impossible**: 3

## Bulyan Byzantine Constraint Violation

### The Problem

Three experiments violate the fundamental Byzantine resilience constraint for Bulyan aggregation:

- `comp_bulyan_alpha0.5_adv30_dp0_pers0_seed42`
- `comp_bulyan_alpha0.5_adv30_dp0_pers0_seed43`
- `comp_bulyan_alpha0.5_adv30_dp0_pers0_seed44`

All three were attempted but immediately failed with the same constraint violation error.

### Mathematical Explanation

**Bulyan's Byzantine Resilience Requirement**: n ≥ 4f + 3

Where:
- n = total number of clients
- f = number of Byzantine (adversarial) clients

**Our Configuration**:
- n = 11 clients
- 30% adversaries = 3 Byzantine clients (f = 3)

**Constraint Check**:
- Required: n ≥ 4(3) + 3 = **15 clients**
- Actual: **11 clients**
- Result: **VIOLATION** (11 < 15)

### Why This Constraint Exists

Bulyan combines Krum with coordinate-wise median aggregation to achieve Byzantine resilience. The algorithm works by:

1. Computing n-f-2 Krum scores
2. Selecting the closest clients
3. Aggregating using coordinate-wise median

For this to work safely, the algorithm requires enough honest clients to:
- Outnumber Byzantine clients by a sufficient margin
- Ensure the median computation is not dominated by adversarial values
- Maintain statistical guarantees about convergence

The 4f+3 bound ensures that after removing the 2f most extreme values, at least f+1 honest values remain for the median.

### Maximum Safe Adversary Fraction

With n=11 clients, the maximum safe Byzantine tolerance is:

```
n ≥ 4f + 3
11 ≥ 4f + 3
8 ≥ 4f
f ≤ 2
```

**Maximum safe adversary fraction**: 2/11 = **18.2%**

Our 30% adversary setting (f=3) exceeds this limit.

## Impact on Experiment Matrix

### Attack Dimension Experiments

The attack dimension explores Byzantine robustness with:
- Aggregation methods: FedAvg, Krum, Bulyan, Median
- Adversary fractions: 0%, 10%, 30%
- Seeds: 42, 43, 44

**Total attack experiments**: 36 (4 aggregations × 3 adversary fractions × 3 seeds)

**Bulyan+30% subset**: 3 experiments (impossible)
**Achievable attack experiments**: 33

### Complete Experiment Breakdown

| Dimension       | Unique Experiments |
|-----------------|-------------------|
| Aggregation     | 12                |
| Heterogeneity   | 9                 |
| Attack          | 36                |
| Privacy         | 6                 |
| Personalization | 6                 |

**Note**: Some experiments appear in multiple dimensions but are counted once by unique preset name.

**Total unique experiments**: 57
**Bulyan+30% impossible**: 3 (all seeds: 42, 43, 44)
**Achievable unique experiments**: 54

## Why These Experiments Failed

All three Bulyan+30% experiments were included in the original experiment matrix but immediately failed when executed with:

```
ValueError: Bulyan requires n >= 4f + 3 for Byzantine resilience.
Got n=11, f=3, but need n >= 15
```

This error is correctly raised by the Bulyan implementation in `robust_aggregation.py:221` and represents proper defensive programming - the algorithm refuses to run in unsafe configurations rather than producing invalid results.

## Recommendations

### For Current Thesis

Accept 54 achievable experiments as the complete dataset:
- 50 already completed (92.6%)
- 4 currently running (IID + DP experiments)
- 3 documented as mathematically impossible

Upon completion of the 4 running experiments, the thesis will have **100% of achievable experiments** (54/54).

### For Future Work

**Option 1: Increase Client Count**
- Use n=15 clients to safely support f=3 (30% adversaries)
- Allows full 0%/10%/30% adversary exploration

**Option 2: Reduce Adversary Fraction**
- Replace 30% with 20% adversaries (f=2, within safe limit)
- Maintains n=11 clients

**Option 3: Remove Bulyan from High-Adversary Scenarios**
- Keep Bulyan for 0% and 10% adversary experiments
- Use Krum/Median for 30% adversary scenarios

## References

- Bulyan algorithm: El Mhamdi et al. "The Hidden Vulnerability of Distributed Learning in Byzantium" (ICML 2018)
- Implementation: `robust_aggregation.py:221` enforces n ≥ 4f+3 constraint
- Attack dimension config: `comparative_analysis.py:122-138`

## Date Documented

2025-10-20

## Related Issues

- Issue #82: Missing byzantine_f field (FIXED)
- Issue #83: DP cosine=1.0 measurement artifact (FIXED)
- TODO: Create issue for Bulyan constraint documentation

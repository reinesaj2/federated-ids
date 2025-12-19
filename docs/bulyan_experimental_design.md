# Bulyan Experimental Design Considerations

## Issue: Client Count Inconsistency

### Background

The Bulyan algorithm requires n ≥ 4f + 3 clients to guarantee Byzantine resilience. This constraint affects experimental design across comparison dimensions.

### Current State

**Attack Resilience Experiments (n=11)**:

- FedAvg: 11 clients
- Krum: 11 clients
- Bulyan: 11 clients (MEETS n >= 4f + 3 for f=2)
- Median: 11 clients

**Other Comparison Dimensions (n=6)**:

- Aggregation comparison: 6 clients
- Heterogeneity comparison: 6 clients
- Privacy comparison: 6 clients
- Personalization comparison: 6 clients

### Impact on Comparability

**Problem**: Cannot directly compare attack resilience results with other dimensions due to different client counts.

**Why This Matters**:

1. Training dynamics differ with client count (more clients = more diverse data distributions)
2. Aggregation variance changes with n
3. Statistical significance harder to establish across different n

### Solution Options

#### Option 1: Accept Inconsistency (CURRENT)

- **Pros**:
  - Experiments already running
  - Each dimension optimized for its specific research question
  - Attack resilience specifically tests Byzantine robustness where n=11 is required
- **Cons**:
  - Cannot directly compare metrics across dimensions
  - Thesis requires careful explanation

#### Option 2: Increase All Dimensions to n=11

- **Pros**:
  - Full comparability across dimensions
  - Consistent experimental setup
- **Cons**:
  - Requires re-running ALL existing experiments (expensive)
  - Not necessary for non-attack dimensions

#### Option 3: Supplementary Experiments (RECOMMENDED)

- **Approach**:
  - Keep attack dimension at n=11 (required for Bulyan)
  - Add supplementary n=6 attack experiments for FedAvg, Krum, Median
  - This provides BOTH:
    - Bulyan results at n=11 (where it can work)
    - Cross-dimension comparison at n=6 (excluding Bulyan)
- **Pros**:
  - Maintains Bulyan Byzantine resilience guarantees
  - Enables comparison with other dimensions
  - Documents Bulyan's higher client requirement
- **Cons**:
  - Requires additional 27 experiments (3 aggregations × 3 adversary % × 3 seeds)
  - Additional computation time (~45 min)

### Recommendation

**Proceed with Option 3**: Run supplementary n=6 attack experiments after n=11 experiments complete.

**Rationale**:

1. Demonstrates Bulyan's stricter requirements (thesis contribution)
2. Allows comparison with other dimensions at n=6
3. Shows how all methods perform under consistent conditions
4. Relatively low additional cost (27 experiments)

### Implementation Plan

1. [DONE] Complete n=11 attack experiments (in progress)
2. [TODO] Generate n=6 supplementary attack configs (exclude Bulyan)
3. [TODO] Run n=6 experiments
4. [TODO] Analysis: Compare both n=6 and n=11 results
5. [TODO] Visualization: Show both sets with clear labeling
6. [TODO] Thesis: Discuss Bulyan's higher client requirement as design consideration

### Statistical Considerations

When comparing across client counts:

- Report confidence intervals for all metrics
- Use appropriate statistical tests (e.g., bootstrap, permutation tests)
- Acknowledge client count as confounding variable
- Focus on relative performance within each n-group

### Documentation for Thesis

**Key Points to Emphasize**:

1. Bulyan's n ≥ 4f + 3 requirement is not a limitation but a mathematical necessity
2. The constraint ensures provable Byzantine resilience
3. In practice, federated systems often have n >> 11 clients
4. Results show tradeoff: Bulyan needs more clients but provides stronger guarantees

# QPLAN - Scrum Analysis: Fix Byzantine Attack Strength & Aggregation Validation

**Epic:** Fix Byzantine Attack Realism & Validation  
**Priority:** P0 (Blocks thesis completion)  
**Story Points:** 13  
**Branch:** `fix/issue-XX-bounded-adversarial-attacks` (to be created)

---

## Executive Summary

**Problem Discovered:**
- Adversarial gradients are **106x larger** than normal (15,384 vs 144)
- All Byzantine-robust methods fail catastrophically (22-33% accuracy)
- Thesis claims about robustness are currently **invalidated**

**Root Cause:**
- `client.py` line 352: `loss = -criterion(preds, yb)` with NO gradient clipping
- Aggregation methods (Krum, Median, Bulyan) are **correctly implemented**
- Attack is unrealistically powerful, overwhelming all defenses

**Solution:**
- Add gradient clipping to adversarial attack (4 lines of code)
- Validate aggregation methods with unit tests
- Re-run experiments with bounded attack

---

##  [READY] Sprint 1: Core Fixes (Days 1-3) - 8 Story Points

### User Story 1.1: Implement Gradient Clipping for Adversaries ⭐
**Story Points:** 3  
**File:** `client.py` lines 347-364

**Implementation:**
```python
# After loss.backward() at line 363, add:
clip_factor = float(self.runtime_config.get("adversary_clip_factor", 2.0))
if clip_factor > 0:
    torch.nn.utils.clip_grad_norm_(
        self.model.parameters(), 
        max_norm=clip_factor * 100.0,
        norm_type=2.0
    )
```

**Acceptance Criteria:**
- [ ] Gradient norm clipped to configurable max
- [ ] `adversary_clip_factor` in runtime_config
- [ ] Verified with unit test
- [ ] Type checking passes

---

### User Story 1.2: Add Unit Tests for Aggregation Methods ⭐
**Story Points:** 3  
**File:** `test_robust_aggregation_spec.py`

**Tests Needed:**
1. `test_krum_selects_honest_with_adversaries()` - 9 honest + 2 outliers
2. `test_median_ignores_outliers()` - Coordinate-wise median verification
3. `test_bulyan_byzantine_resilience()` - n=11, f=2 validation

**Acceptance Criteria:**
- [ ] All tests pass
- [ ] Property-based tests with hypothesis
- [ ] Integration with existing test suite

---

### User Story 1.3: Add Aggregation Debug Logging
**Story Points:** 2  
**Files:** `robust_aggregation.py`, `server.py`

**Features:**
- Log selected gradient indices (Krum)
- Log gradient norm statistics (min/median/max)
- Debug mode flag (environment variable)

---

##  [READY] Sprint 2: Experimental Validation (Days 4-7) - 5 Story Points

### User Story 2.1: Run Gradient Magnitude Analysis
**Story Points:** 2  
**File:** NEW `scripts/analyze_gradient_norms.py`

**Analysis:**
- Measure honest gradient norms (baseline)
- Test clip factors: [1.5x, 2x, 3x, 5x, 10x]
- Select optimal factor (likely 2.0-3.0)

---

### User Story 2.2: Re-run Core Experiments with Fixed Attack  
**Story Points:** 3  
**File:** `scripts/run_experiments_optimized.py`

**Experiments:**
- 4 methods × [0%, 10%, 20%, 30%] adversaries × 3 seeds = **48 experiments**
- Expected: Byzantine methods > FedAvg at 10%+
- Expected: Gradual degradation, not catastrophic failure

---

## 🌳 Branching Strategy

**Recommended:**
```bash
git checkout -b fix/issue-XX-bounded-adversarial-attacks
```

**Why:**
- Follows existing convention (`fix/issue-XX-*`)
- Isolates fix from `feat/issue-28`
- Can be merged independently
- CI validation possible

---

## Definition of Done

### For Epic:
- [ ] All user stories complete
- [ ] 48 core experiments run successfully
- [ ] Byzantine methods outperform FedAvg (statistical significance p < 0.05)
- [ ] Thesis plots generated
- [ ] Documentation updated
- [ ] Code merged to main

---

## Immediate Actions (Priority Order)

### TODAY:
1. Stop remaining experiments
2. Create branch `fix/issue-XX-bounded-adversarial-attacks`
3. Implement gradient clipping (15 min)
4. Write validation unit test (30 min)

### TOMORROW:
5. Complete unit test suite (2 hours)
6. Add debug logging (1 hour)
7. Run gradient analysis (2 hours)

### DAY 3+:
8. Run corrected experiments
9. Generate thesis plots
10. Statistical analysis

---

## Risk Mitigation

### Risk 1: Clipping Too Aggressive
**Mitigation:** Run gradient analysis FIRST to find optimal factor

### Risk 2: Aggregation Still Broken
**Mitigation:** Unit tests validate BEFORE experiments

### Risk 3: Timeline Slippage
**Mitigation:** Run subset first (12 experiments, 1 seed) for validation

---

## 🔗 References

- **Kanban Board:** https://github.com/users/reinesaj2/projects/2
- **Thesis:** `/Users/abrahamreines/Documents/Thesis/deliverable1/FL.txt`
- **Current Results:** `/Users/abrahamreines/Documents/Thesis/federated-ids/runs/`
- **Evidence:** Gradient norm analysis showing 106x amplification

---

**Status:** Ready to implement  
**Owner:** @reinesaj2  
**Created:** 2025-10-19  
**Target Completion:** 7 days from start


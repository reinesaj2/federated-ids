## Issue: Stabilize `main` with milestone merges and passing tests

**Summary**

`main` currently lacks functionality and test coverage that already landed in the milestone branches (secure aggregation logging, FedProx heterogeneity grid, ExperimentConfig dataset helpers, visualization utilities, DP validation, etc.). Tests fail because those features were effectively dropped. We need to restore the milestone work, confirm all tests pass, and keep `main` as the single source of truth going forward.

**Recommended Plan (Scrum Master)**

1. **Snapshot & Restore Baseline**
   - Tag the current `main` (for rollback) and restore milestone-verified versions of `client.py`, `scripts/generate_thesis_plots.py`, and `scripts/ci_checks.py` from the corresponding milestone branches.

2. **Sequential Milestone Merges**
   - Merge milestones oldest → newest (M0 → M4), resolving conflicts immediately:
     - M0 CI foundation (CI helpers, adaptive L2 thresholds)
     - M1 experiments core (FedProx µ grid, `ExperimentConfig.with_dataset`)
     - M2 privacy/security (secure aggregation logging, DP schema updates)
     - M3 visualization/reporting (thesis plots + confusion/personalization helpers)
     - M4 automation (workflow automation hooks)

3. **Gate with Targeted Tests**
   - After each merge, run the scoped tests tied to that milestone (e.g., CI helper tests, comparative-analysis suite, client secure-aggregation tests, visualization tests).

4. **Full Regression Pass**
   - Once all milestones are merged, run the entire pytest suite (`pytest -n auto` if possible). Record the passing commit (tag or release note).

5. **Worktree Hygiene**
   - Rebase or recreate the open worktrees (automation/confusion/roadmap) on the refreshed `main` to avoid reintroducing the drift.

6. **CI Follow-up**
   - Ensure GH workflows reference the restored scripts and run a dry-run on the actual CI environment (the local sandbox blocks some tests like secure aggregation integration).

**Acceptance Criteria**

- All milestone functionality (secure aggregation logging, DP validation, FedProx scatter/personalization plots, etc.) present on `main`.
- `pytest` succeeds end-to-end in the supported environment.
- Automation worktrees updated or documented with rebase plans.
- Issue closed only after CI pipelines confirm green status.

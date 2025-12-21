# Cluster Runs Decision: Use Unified Schema + Binary Labeling (Issues #134 + #135)

## Decision

For cluster experiments aligned to deliverable1/FL.txt objectives, use the unified schema adapters (Issue #134) and the binary classification toggle (Issue #135) instead of the hybrid dataset as the primary path.

## Why This Meets Objectives Better

1. **Objective alignment**: deliverable1/FL.txt evaluates Objective 1 (attack resilience) and Objective 2 (heterogeneity) on CIC-IDS2017, UNSW-NB15, and Edge-IIoTset. Unified schema adapters preserve each dataset's native feature space while enabling cross-dataset federation when needed.
2. **Comparability**: results remain directly comparable to prior runs and literature because each dataset keeps its original semantics and preprocessing.
3. **Mixed federation support**: Issue #134 provides compatible model parameters across datasets via adapter projections into a shared latent space, which is required for mixed-client federated training.
4. **Label consistency when required**: Issue #135 adds an explicit `--binary_classification` switch so that binary attack detection runs are consistent across datasets without altering multiclass experiments.
5. **Reduced confounds**: the hybrid dataset performs semantic feature extraction and zero-filling, which introduces an extra transformation layer that can blur heterogeneity and attack effects. That makes objective-specific attribution harder.

## Why Not Use the Hybrid Dataset as the Primary Path

- **Lossy transformation**: semantic feature extraction drops or zero-fills many dataset-specific signals, which can change the attack-resilience and heterogeneity behavior.
- **Different task definition**: the hybrid dataset introduces a new label taxonomy and feature space; results are no longer directly tied to the original datasets in deliverable1/FL.txt.
- **Interpretability risk**: it becomes unclear whether results reflect federated dynamics or the hybrid feature engineering layer.

## Recommended Usage for Cluster Runs

1. **Per-dataset runs**: use native datasets with default model behavior; keep multiclass where appropriate.
2. **Binary consistency runs**: add `--binary_classification` when the comparison requires a shared binary label space.
3. **Mixed-client runs**: set `--model_arch unified_schema` so CIC/UNSW/Edge-IIoTset clients share compatible parameters.

## When the Hybrid Dataset Is Still Useful

- Supplementary experiments to study cross-domain generalization or unified taxonomy effects.
- Exploratory analysis where a single consolidated dataset is explicitly the target.

## Summary

Issues #134 and #135 preserve dataset fidelity and satisfy the deliverable1/FL.txt objectives while enabling mixed-dataset federation and consistent binary labeling. The hybrid dataset is valuable as a secondary experiment, but should not replace the primary cluster runs.

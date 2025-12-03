# Edge-IIoTset Dataset Availability in CI

## Overview

The “Edge-IIoTset Full-Scale Weekly” workflow started failing immediately after the
`Verify full dataset` step even though the 2M-row CSV existed locally and was tracked via Git LFS. GitHub Actions runners never materialized the large file, so every job halted before running any experiments.

## Symptoms

- Workflow run IDs: `19346320653`, `19348259128`, `19348692935`
- Step: `full_scale_experiments › Verify full dataset`
- Log excerpt:
  ```
  ERROR: Full-scale dataset not found at data/edge-iiotset/edge_iiotset_full.csv
  ```
- Preceding `setup_real_datasets.py` output showed the symlink target (`datasets/edge-iiotset/processed/edge_iiotset_full.csv`) was missing on the runner.

## Root Cause

`actions/checkout` with `lfs: true` only fetches pointers; it does not automatically download large LFS objects unless they are referenced by files in the checkout working tree. Because the workflow never explicitly ran `git lfs fetch`/`git lfs checkout`, the 900 MB CSV was absent at runtime, causing the `-f` check to fail.

## Fix

Commit `a3546c7` added a dedicated step after each checkout to fetch and check out the dataset explicitly:

```yaml
- name: Fetch LFS datasets
  run: |
    git lfs fetch --all
    git lfs checkout datasets/edge-iiotset/processed/edge_iiotset_full.csv
```

This guarantees `datasets/edge-iiotset/processed/edge_iiotset_full.csv` exists before `scripts/setup_real_datasets.py` attempts to link it into `data/edge-iiotset/`.

## Verification

After the fix, workflow run `19348987144` shows the new “Fetch LFS datasets” step executing in every job. The `Verify full dataset` step now reports the file and size instead of failing, allowing experiments to proceed.

## Recommendations

- Keep the explicit `git lfs fetch/checkout` block in any workflow that relies on large datasets, especially when new LFS files are added.
- If additional Edge-IIoTset tiers (quick/nightly) are committed via LFS later, extend the step with the relevant paths so they are also available in CI.


import pandas as pd
from pathlib import Path

runs_dir = Path("runs")
target_adv = 30
target_alpha = "0.5"

run_paths = list(runs_dir.glob(f"dsedge-iiotset-nightly_*_alpha{target_alpha}_*adv{target_adv}*"))

for run_path in run_paths[:2]:
    print(f"\nChecking {run_path.name}")
    client_files = list(run_path.glob("client_*_metrics.csv"))
    if not client_files: continue
    
    try:
        df = pd.read_csv(client_files[0])
        if "macro_f1_before" in df.columns:
            valid_count = df['macro_f1_before'].notna().sum()
            print(f"  [OK] macro_f1_before present. Valid Rows: {valid_count}/{len(df)}. Mean: {df['macro_f1_before'].mean():.4f}")
        else:
            print("  [MISSING] macro_f1_before MISSING!")

        if "macro_f1_after" in df.columns:
            valid_count = df['macro_f1_after'].notna().sum()
            print(f"  [OK] macro_f1_after present. Valid Rows: {valid_count}/{len(df)}. Mean: {df['macro_f1_after'].mean():.4f}")
             
    except Exception as e:
        print(f"  Error: {e}")

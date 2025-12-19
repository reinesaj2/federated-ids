import pandas as pd
from pathlib import Path
import re

runs_dir = Path("runs")
target_adv = 30
target_alpha = "0.5"

print(f"DEBUGGING F1 for Alpha={target_alpha}, Adv={target_adv}")

# 1. Find Runs
run_paths = list(runs_dir.glob(f"dsedge-iiotset-nightly_*_alpha{target_alpha}_*adv{target_adv}*"))
print(f"Found {len(run_paths)} runs matching criteria.")

# 2. Check Client Data
for run_path in run_paths[:2]:  # Check first 2
    print(f"\nChecking {run_path.name}")
    client_files = list(run_path.glob("client_*_metrics.csv"))
    if not client_files:
        print("  NO CLIENT FILES FOUND!")
        continue

    print(f"  Found {len(client_files)} client files.")
    # Check headers of first client file
    try:
        df = pd.read_csv(client_files[0])
        if "macro_f1_global" in df.columns:
            print(f"  [OK] macro_f1_global present. Mean: {df['macro_f1_global'].mean():.4f}")
        else:
            print("  [MISSING] macro_f1_global MISSING!")
            print(f"  Columns: {df.columns.tolist()}")

        if "macro_f1_personalized" in df.columns:
            print(f"  [OK] macro_f1_personalized present. Mean: {df['macro_f1_personalized'].mean():.4f}")
        else:
            print("  [MISSING] macro_f1_personalized MISSING!")

    except Exception as e:
        print(f"  Error reading csv: {e}")

# 3. Check Merge Logic Keys
print("\nVerifying Merge Keys...")


# Mock data loading
def parse_config(name):
    config = {}
    if "comp_fedavg" in name:
        config["aggregation"] = "FedAvg"
    elif "comp_krum" in name:
        config["aggregation"] = "Krum"
    elif "comp_bulyan" in name:
        config["aggregation"] = "Bulyan"
    elif "comp_median" in name:
        config["aggregation"] = "Median"
    seed_match = re.search(r"seed(\d+)", name)
    config["seed"] = int(seed_match.group(1)) if seed_match else 0
    config["alpha"] = "0.5"  # Force matching for test
    config["adversary"] = 30  # Force matching for test
    return config


server_rows = []
client_rows = []

for run_path in run_paths[:5]:
    config = parse_config(run_path.name)

    # Server
    s_path = run_path / "metrics.csv"
    if s_path.exists():
        sdf = pd.read_csv(s_path)
        for k, v in config.items():
            sdf[k] = v
        server_rows.append(sdf)

    # Client
    c_files = list(run_path.glob("client_*_metrics.csv"))
    if c_files:
        cdfs = [pd.read_csv(f) for f in c_files]
        cdf = pd.concat(cdfs, ignore_index=True)
        for k, v in config.items():
            cdf[k] = v
        client_rows.append(cdf)

if server_rows and client_rows:
    server_df = pd.concat(server_rows)
    client_df = pd.concat(client_rows)

    print(f"Server Cols: {server_df.columns.tolist()}")
    print(f"Client Cols: {client_df.columns.tolist()}")

    group_cols = ["aggregation", "alpha", "seed", "round", "adversary"]
    print(f"Merge Keys: {group_cols}")

    try:
        client_agg = client_df.groupby(group_cols)[["macro_f1_global"]].mean().reset_index()
        print(f"Aggregated Client Data Rows: {len(client_agg)}")

        merged = pd.merge(server_df, client_agg, on=group_cols, how="left")
        print(f"Merged Rows: {len(merged)}")
        print(f"F1 NaNs after merge: {merged['macro_f1_global'].isna().sum()}")
        print(merged[['aggregation', 'round', 'macro_f1_global']].head())
    except Exception as e:
        print(f"Merge Failed: {e}")

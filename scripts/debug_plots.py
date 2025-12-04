
import pandas as pd
from pathlib import Path
import re
import sys

# Helper functions from the main script
def parse_run_config(run_dir: Path) -> dict:
    name = run_dir.name
    config = {}
    if "comp_fedavg" in name:
        config["aggregation"] = "FedAvg"
    elif "comp_krum" in name:
        config["aggregation"] = "Krum"
    elif "comp_bulyan" in name:
        config["aggregation"] = "Bulyan"
    elif "comp_median" in name:
        config["aggregation"] = "Median"
    elif "fedprox" in name:
        config["aggregation"] = "FedProx"
    else:
        config["aggregation"] = "Unknown"
        
    seed_match = re.search(r"seed(\d+)", name)
    config["seed"] = int(seed_match.group(1)) if seed_match else 0
    
    alpha_match = re.search(r"alpha([0-9\.]+|inf)", name)
    config["alpha"] = alpha_match.group(1) if alpha_match else "Unknown"
    return config

def load_server_metrics(run_dir: Path) -> pd.DataFrame | None:
    path = run_dir / "metrics.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)

def load_client_metrics(run_dir: Path) -> pd.DataFrame | None:
    client_files = list(run_dir.glob("client_*_metrics.csv"))
    if not client_files:
        return None
    dfs = []
    for cf in client_files:
        try:
            df = pd.read_csv(cf)
            client_id = int(re.search(r"client_(\d+)", cf.name).group(1))
            df["client_id"] = client_id
            dfs.append(df)
        except Exception:
            continue
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)

runs_dir = Path("/Users/abrahamreines/Documents/Thesis/worktrees/iiot-experiments/runs")
all_server_data = []
all_client_data = []

print("Scanning runs...")
count = 0
for run_path in runs_dir.glob("dsedge-iiotset-nightly*"):
    if not run_path.is_dir():
        continue
    count += 1
    if count > 5: break # just check a few

    print(f"Checking {run_path.name}")
    config = parse_run_config(run_path)
    print(f"Config: {config}")

    sdf = load_server_metrics(run_path)
    if sdf is not None:
        for k, v in config.items():
            sdf[k] = v
        all_server_data.append(sdf)
        print(f"Server columns: {sdf.columns.tolist()}")
        print(f"Server head: {sdf.head(1).to_dict()}")
    
    cdf = load_client_metrics(run_path)
    if cdf is not None:
        for k, v in config.items():
            cdf[k] = v
        all_client_data.append(cdf)
        print(f"Client columns: {cdf.columns.tolist()}")

if all_server_data:
    server_df = pd.concat(all_server_data, ignore_index=True)
    print(f"\nTotal Server Rows: {len(server_df)}")
    print("Server Aggregation Counts:")
    print(server_df["aggregation"].value_counts())
    
    print("\nL2 Dispersion Mean Stats:")
    if "l2_dispersion_mean" in server_df.columns:
        print(server_df["l2_dispersion_mean"].describe())
    else:
        print("Column not found!")

if all_client_data:
    client_df = pd.concat(all_client_data, ignore_index=True)
    print(f"\nTotal Client Rows: {len(client_df)}")

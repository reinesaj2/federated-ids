from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a sampled UNSW-NB15 CSV for fast demos"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to UNSW CSV (training or testing set)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output CSV path; default appends .sample.csv next to input",
    )
    parser.add_argument(
        "--frac", type=float, default=0.10, help="Sample fraction in (0,1]"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        raise SystemExit(f"Input not found: {inp}")
    out = Path(args.output) if args.output else inp.with_suffix("")
    if out.suffix.lower() != ".csv":
        out = out.with_name(out.name + ".sample.csv")

    df = pd.read_csv(inp)
    if not (0.0 < args.frac <= 1.0):
        raise SystemExit("--frac must be in (0,1]")
    rng = np.random.default_rng(args.seed)
    mask = rng.random(len(df)) <= args.frac
    sampled = df.loc[mask].reset_index(drop=True)
    sampled.to_csv(out, index=False)
    print(f"Wrote {len(sampled)} rows to {out}")


if __name__ == "__main__":
    main()

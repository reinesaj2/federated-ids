#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import yaml


def main() -> int:
    p = argparse.ArgumentParser(description="Validate run config.yaml for D2")
    p.add_argument("--path", type=str, required=True)
    args = p.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"config file not found: {path}")
        return 2

    cfg = yaml.safe_load(path.read_text()) or {}

    required = [
        ("dataset", str),
        ("data_path", str),
        ("partition_strategy", str),
        ("num_clients", int),
        ("rounds", int),
        ("alpha", (int, float)),
        ("seed", int),
        ("logdir", str),
    ]

    ok = True
    for key, typ in required:
        if key not in cfg:
            print(f"missing required key: {key}")
            ok = False
        else:
            if not isinstance(cfg[key], typ):
                print(f"invalid type for {key}: got {type(cfg[key]).__name__}, expected {typ}")
                ok = False

    if ok:
        print("config.yaml validation: OK")
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())



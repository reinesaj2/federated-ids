#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

SOURCE_DATASETS = {"cic", "edge-iiotset-full"}


def normalize_key_value(value: object) -> object:
    if isinstance(value, float) and math.isinf(value):
        return "inf"
    return value


def is_bulyan_feasible(num_clients: int, adversary_fraction: float) -> bool:
    byzantine_f = int(adversary_fraction * num_clients)
    return num_clients >= (4 * byzantine_f + 3)


def build_run_name(config: dict, digest: str) -> str:
    aggregation = config.get("aggregation")
    alpha = config.get("alpha")
    adversary_fraction = config.get("adversary_fraction", 0.0)
    dp_enabled = bool(config.get("dp_enabled", False))
    dp_noise_multiplier = config.get("dp_noise_multiplier", 0.0)
    personalization_epochs = config.get("personalization_epochs", 0)
    fedprox_mu = config.get("fedprox_mu", 0.0)
    seed = config.get("seed")

    alpha_value = "inf" if (isinstance(alpha, float) and math.isinf(alpha)) else str(alpha)
    adv_percent = int(adversary_fraction * 100)

    parts = [
        f"unsw_simple_{digest}",
        f"comp_{aggregation}",
        f"alpha{alpha_value}",
        f"adv{adv_percent}",
        f"dp{int(dp_enabled)}",
    ]
    if dp_enabled:
        parts.append(f"dpnoise{dp_noise_multiplier}")
    parts.extend(
        [
            f"pers{personalization_epochs}",
            f"mu{fedprox_mu}",
            f"seed{seed}",
        ]
    )
    return "_".join(parts)


def manifest_key(config: dict) -> tuple:
    return (
        normalize_key_value(config.get("aggregation")),
        normalize_key_value(config.get("alpha")),
        normalize_key_value(config.get("adversary_fraction")),
        normalize_key_value(config.get("dp_enabled")),
        normalize_key_value(config.get("dp_noise_multiplier")),
        normalize_key_value(config.get("personalization_epochs")),
        normalize_key_value(config.get("num_clients")),
        normalize_key_value(config.get("num_rounds")),
        normalize_key_value(config.get("seed")),
        normalize_key_value(config.get("fedprox_mu")),
        normalize_key_value(config.get("attack_mode")),
        normalize_key_value(config.get("temporal_validation")),
    )


def hash_manifest_key(key: tuple) -> str:
    payload = repr(key).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:10]


def build_unsw_simple_mirror_manifest(
    source_configs: Iterable[dict],
    data_path: str,
    model_arch: str,
) -> list[dict]:
    entries: dict[tuple, dict] = {}

    for config in source_configs:
        aggregation = config.get("aggregation")
        num_clients = int(config.get("num_clients", 0))
        adversary_fraction = float(config.get("adversary_fraction", 0.0))

        if aggregation == "bulyan" and not is_bulyan_feasible(num_clients, adversary_fraction):
            continue

        key = manifest_key(config)
        entry = entries.get(key)
        source_dataset = config.get("dataset")

        if entry is None:
            entry = dict(config)
            entry.pop("run_name", None)
            entry["dataset"] = "unsw"
            entry["data_path"] = data_path
            entry["model_arch"] = model_arch
            entry["source_datasets"] = sorted({source_dataset} if source_dataset else [])
            entries[key] = entry
        elif source_dataset:
            sources = set(entry.get("source_datasets", []))
            sources.add(source_dataset)
            entry["source_datasets"] = sorted(sources)

    manifest_entries = []
    for key, entry in entries.items():
        digest = hash_manifest_key(key)
        entry["run_name"] = build_run_name(entry, digest)
        manifest_entries.append(entry)

    manifest_entries.sort(key=lambda item: item["run_name"])
    return manifest_entries


def load_source_configs(source_dir: Path) -> list[dict]:
    configs = []
    for entry in source_dir.iterdir():
        if not entry.is_dir():
            continue
        config_path = entry / "config.json"
        if not config_path.exists():
            continue
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        if config.get("dataset") in SOURCE_DATASETS:
            configs.append(config)
    return configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate UNSW SimpleNet manifest from CIC+IIOT archive.")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--model-arch", type=str, default="simple")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_configs = load_source_configs(args.source_dir)
    entries = build_unsw_simple_mirror_manifest(source_configs, args.data_path, args.model_arch)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")
    print(f"Wrote {len(entries)} entries to {args.output}")


if __name__ == "__main__":
    main()

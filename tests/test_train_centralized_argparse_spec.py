import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "scripts"))

import train_centralized

DATASET_NAME = "cic"
DATA_PATH = "data.csv"
MODEL_NAME = "encoder"
WEIGHT_DECAY = 1e-4

def test_build_arg_parser_accepts_encoder_model() -> None:
    parser = train_centralized.build_arg_parser()
    args = parser.parse_args(["--dataset", DATASET_NAME, "--data_path", DATA_PATH, "--model", MODEL_NAME])

    assert args.model == MODEL_NAME


def test_build_arg_parser_accepts_weight_decay() -> None:
    parser = train_centralized.build_arg_parser()
    args = parser.parse_args(["--dataset", DATASET_NAME, "--data_path", DATA_PATH, "--weight_decay", str(WEIGHT_DECAY)])

    assert args.weight_decay == WEIGHT_DECAY

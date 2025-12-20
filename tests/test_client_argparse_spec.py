import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import client


@pytest.mark.parametrize(
    "attack_mode",
    [
        "none",
        "label_flip",
        "grad_ascent",
        "sign_flip_topk",
        "targeted_label",
    ],
)
def test_build_arg_parser_accepts_adversary_modes(attack_mode: str) -> None:
    parser = client.build_arg_parser()
    args = parser.parse_args(["--adversary_mode", attack_mode])

    assert args.adversary_mode == attack_mode

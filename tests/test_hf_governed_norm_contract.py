import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CARD_DIR = ROOT / "hf-kernels" / "szl-governed-norm"
NON_KERNEL_REVISION = "27faddd262c6ee36d08aad9ae234595d75a999f1"


def _contract():
    return json.loads((CARD_DIR / "contract.json").read_text(encoding="utf-8"))


def test_quickstart_pins_the_first_class_kernel_revision():
    contract = _contract()
    card = (CARD_DIR / "README.md").read_text(encoding="utf-8")
    revision = contract["observed_revision"]

    assert re.fullmatch(r"[0-9a-f]{40}", revision)
    assert revision == "fe16433d44be03177167e8355c43a4bfdc63e03e"
    assert NON_KERNEL_REVISION not in card
    assert f'revision="{revision}"' in card
    assert "trust_remote_code=True" in card
    assert 'revision="main"' not in card


def test_contract_points_to_the_active_successor_package():
    contract = _contract()

    assert contract["canonical_source"] == "szl-holdings/szl-lambda-gate"
    assert contract["compatibility_package"] == "szl_lambda_gate.governed_norm"


def test_contract_declares_kernel_hub_license_metadata():
    contract = _contract()

    assert contract["repo_type"] == "kernel"
    assert contract["license"] == "apache-2.0"

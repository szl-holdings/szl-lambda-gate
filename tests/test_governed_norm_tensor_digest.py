# SPDX-License-Identifier: Apache-2.0
# © 2026 SZL Holdings
"""Versioned tensor-digest contract for the folded governed-norm kernel.

Donor behavior is szl-governed-norm commit
3ef27eb7ebf491b0a6ce69be170ecef4c37885a2, file
torch-ext/szl_governed_norm/_receipt.py at
c68d06d35058542ea77dc6bad4c8bde2361cc16a.
Target baseline is szl-lambda-gate 8035af3e190a2e23ca1345bdc07736935634f6f4.

rounded-int64-v1 is the historical encoding. class-byte-v1 prefixes a class
byte per element, which changes finite digests too. Stored hashes are not
rewritten from one encoding to the other. ReceiptChain.verify still hashes
the original body keys.

The frozen hexes are SHA3-256 over the little-endian numpy buffers this
kernel already uses. They are pins for these exact tensors, not a claim
about another device endianness.
"""
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "torch-ext"))
import szl_lambda_gate.governed_norm as gn  # noqa: E402
from szl_lambda_gate.governed_norm._receipt import (  # noqa: E402
    TENSOR_DIGEST_CLASS_BYTE,
    TENSOR_DIGEST_LEGACY,
    _tensor_digest,
)

_HAS_SZL_RECEIPT = importlib.util.find_spec("szl_receipt") is not None

# torch.tensor([1.0, -2.5, 0.0], dtype=torch.float32)
FINITE_LEGACY = "c77fb7384bfdcf5612bacf2b39f9029dfe8f4129e097c26d0484be806fbacdbd"
FINITE_CLASS = "e6140d675abe3220089a5d6f26ee0a555f0bf1c6537e6153cb9513a299fc7431"
# torch.full((3,), +inf / -inf / nan) and torch.zeros(3), class-byte-v1
PINF_CLASS = "1bef230fe5811c46860111ca4ee1d2ebd19c4b6ba7e8cf86bd0fed4d8f6f2c96"
NINF_CLASS = "003e857d2bcfd6e58a08e00c448e0ceeda05e246a844858f11e4f483c5dd9f1a"
NAN_CLASS = "706af01a99c708eb93037517ee74fb6ed9abecda4940a795ee70d58a1d490fcf"
ZERO_CLASS = "028744e3ccc696c129422b79f8175118c0a1b32e490a0e7f757820fb534f8a45"
# Historical saturation: +Inf, -Inf, and NaN share one legacy digest.
NONFINITE_LEGACY_COLLISION = "0229f269bef3c8e806ed347322c1d5dc1cf8316d86084abca95fab353d68243d"


def _finite():
    return torch.tensor([1.0, -2.5, 0.0], dtype=torch.float32)


def test_frozen_legacy_and_class_byte_vectors_differ_for_finite_values():
    finite = _finite()
    legacy = _tensor_digest(finite, encoding=TENSOR_DIGEST_LEGACY)
    current = _tensor_digest(finite, encoding=TENSOR_DIGEST_CLASS_BYTE)
    assert legacy == FINITE_LEGACY
    assert current == FINITE_CLASS
    assert legacy != current
    # The default is the new encoding, and calling it does not change the legacy pin.
    assert _tensor_digest(finite) == FINITE_CLASS
    assert _tensor_digest(finite, encoding=TENSOR_DIGEST_LEGACY) == FINITE_LEGACY


def test_legacy_nonfinite_collision_stays_interpretable():
    """The historical encoding still collides. It is not rewritten into class-byte-v1."""
    pinf = torch.full((3,), float("inf"), dtype=torch.float32)
    ninf = torch.full((3,), float("-inf"), dtype=torch.float32)
    nan = torch.full((3,), float("nan"), dtype=torch.float32)
    assert _tensor_digest(pinf, encoding=TENSOR_DIGEST_LEGACY) == NONFINITE_LEGACY_COLLISION
    assert _tensor_digest(ninf, encoding=TENSOR_DIGEST_LEGACY) == NONFINITE_LEGACY_COLLISION
    assert _tensor_digest(nan, encoding=TENSOR_DIGEST_LEGACY) == NONFINITE_LEGACY_COLLISION
    assert _tensor_digest(pinf) == PINF_CLASS
    assert _tensor_digest(ninf) == NINF_CLASS
    assert _tensor_digest(nan) == NAN_CLASS
    assert len({PINF_CLASS, NINF_CLASS, NAN_CLASS, ZERO_CLASS}) == 4


def test_tensor_digest_distinguishes_nonfinite_classes():
    pinf = torch.full((3,), float("inf"), dtype=torch.float32)
    ninf = torch.full((3,), float("-inf"), dtype=torch.float32)
    nan = torch.full((3,), float("nan"), dtype=torch.float32)
    zero = torch.zeros(3, dtype=torch.float32)
    assert _tensor_digest(pinf) == PINF_CLASS
    assert _tensor_digest(ninf) == NINF_CLASS
    assert _tensor_digest(nan) == NAN_CLASS
    assert _tensor_digest(zero) == ZERO_CLASS
    assert len({PINF_CLASS, NINF_CLASS, NAN_CLASS, ZERO_CLASS}) == 4


def test_tensor_digest_nonfinite_is_position_sensitive():
    leading_inf = torch.tensor([float("inf"), 1.0, 2.0], dtype=torch.float32)
    middle_inf = torch.tensor([1.0, float("inf"), 2.0], dtype=torch.float32)
    leading_nan = torch.tensor([float("nan"), 1.0, 2.0], dtype=torch.float32)
    assert _tensor_digest(leading_inf) != _tensor_digest(middle_inf)
    assert _tensor_digest(leading_inf) != _tensor_digest(leading_nan)


def test_tensor_digest_nonfinite_is_deterministic():
    left = torch.tensor([float("inf"), float("nan"), -1.5], dtype=torch.float32)
    right = left.clone()
    assert _tensor_digest(left) == _tensor_digest(right)
    assert _tensor_digest(left, encoding=TENSOR_DIGEST_LEGACY) == _tensor_digest(
        right, encoding=TENSOR_DIGEST_LEGACY
    )


def test_tensor_digest_signed_zero_still_collides():
    positive = torch.tensor([0.0, 0.0], dtype=torch.float32)
    negative = torch.tensor([-0.0, -0.0], dtype=torch.float32)
    assert _tensor_digest(positive) == _tensor_digest(negative)
    assert _tensor_digest(positive) == "d6c230fa96cd2a2aadb99196a84aaf736960dbb723ed091e9ef9a45b4c788c96"
    # The class-byte prefix still moves finite zero off the legacy digest.
    assert _tensor_digest(positive) != _tensor_digest(positive, encoding=TENSOR_DIGEST_LEGACY)


def test_unknown_encoding_is_rejected_and_records_nothing():
    chain = gn.ReceiptChain()
    finite = _finite()
    with pytest.raises(ValueError, match="unknown tensor digest encoding"):
        chain.emit("rms_norm", finite, finite, 1e-6, digest_encoding="rewritten")
    assert chain.count() == 0


def test_later_class_byte_receipt_does_not_rewrite_a_legacy_hash():
    chain = gn.ReceiptChain()
    finite = _finite()
    legacy = chain.emit(
        "rms_norm", finite, finite, 1e-6, digest_encoding=TENSOR_DIGEST_LEGACY
    )
    current = chain.emit("rms_norm", finite, finite, 1e-6)
    assert legacy["out_digest"] == FINITE_LEGACY
    assert legacy["tensor_digest_encoding"] == TENSOR_DIGEST_LEGACY
    assert current["out_digest"] == FINITE_CLASS
    assert current["tensor_digest_encoding"] == TENSOR_DIGEST_CLASS_BYTE
    assert chain.tail(2)[0]["out_digest"] == FINITE_LEGACY
    ok, depth, first_break = chain.verify()
    assert ok is True and depth == 2 and first_break == -1
    # The label is outside the hashed body. Removing it leaves the stored hash.
    del chain.tail(2)[0]["tensor_digest_encoding"]
    assert chain.tail(2)[0]["out_digest"] == FINITE_LEGACY
    ok, depth, first_break = chain.verify()
    assert ok is True and depth == 2 and first_break == -1


def test_governed_receipt_digest_distinguishes_inf_sign():
    pinf = torch.full((2, 4), float("inf"), dtype=torch.float32)
    ninf = torch.full((2, 4), float("-inf"), dtype=torch.float32)
    positive = gn.ReceiptChain()
    negative = gn.ReceiptChain()
    left = positive.emit("rms_norm", pinf, pinf, 1e-6)
    right = negative.emit("rms_norm", ninf, ninf, 1e-6)
    assert left["out_digest"] != right["out_digest"]
    assert left["tensor_digest_encoding"] == TENSOR_DIGEST_CLASS_BYTE
    assert positive.verify()[0] is True and negative.verify()[0] is True


def test_public_governed_call_records_the_class_byte_digest():
    chain = gn.ReceiptChain()
    source = torch.tensor([[1.0, -2.5, 0.0]], dtype=torch.float32)
    output = gn.rms_norm(source, eps=1e-6, chain=chain)
    assert chain.count() == 1
    record = chain.tail(1)[0]
    assert record["op"] == "rms_norm"
    assert record["tensor_digest_encoding"] == TENSOR_DIGEST_CLASS_BYTE
    assert record["out_digest"] == _tensor_digest(output, encoding=TENSOR_DIGEST_CLASS_BYTE)
    assert record["out_digest"] != _tensor_digest(output, encoding=TENSOR_DIGEST_LEGACY)
    ok, depth, first_break = chain.verify()
    assert ok is True and depth == 1 and first_break == -1


@pytest.mark.skipif(
    not _HAS_SZL_RECEIPT,
    reason="szl-receipt is not provisioned; signing and binding stay optional",
)
def test_provisioned_szl_receipt_binding_uses_the_stored_digest():
    chain = gn.ReceiptChain()
    source = torch.tensor([[1.0, -2.5, 0.0]], dtype=torch.float32)
    output = gn.rms_norm(source, eps=1e-6)
    record = chain.emit("rms_norm", source, output, 1e-6)
    binding = record["receipt"]
    assert binding["energy"] == "UNAVAILABLE"
    assert binding["output_digest"] == record["out_digest"]
    assert binding["output_digest"] == _tensor_digest(output, encoding=TENSOR_DIGEST_CLASS_BYTE)
    assert binding["signature"]["signed"] is False
    assert chain.verify()[0] is True

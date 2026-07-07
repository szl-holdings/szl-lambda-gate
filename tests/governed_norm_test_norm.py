# SPDX-License-Identifier: Apache-2.0
"""Correctness + governance tests for szl_governed_norm.

Run: python -m pytest tests/ -q   (or: python tests/test_norm.py)
Verifies numerical equivalence to PyTorch references and that governed
receipts hash-chain and verify. CPU-only so it runs anywhere.
"""
import sys
from pathlib import Path

import torch

# Import the built universal kernel package directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "torch-ext"))
import szl_lambda_gate.governed_norm as gn  # noqa: E402  (folded-in governed-norm kernels)


def test_rms_norm_matches_reference():
    torch.manual_seed(0)
    x = torch.randn(8, 512, dtype=torch.float32)
    w = torch.randn(512, dtype=torch.float32)
    eps = 1e-6
    # Reference RMSNorm (Llama-style, float32 compute).
    xf = x.to(torch.float32)
    ref = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    ref = ref.to(x.dtype) * w
    out = gn.rms_norm(x, weight=w, eps=eps)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_layer_norm_matches_torch():
    torch.manual_seed(1)
    x = torch.randn(8, 256, dtype=torch.float32)
    w = torch.randn(256, dtype=torch.float32)
    b = torch.randn(256, dtype=torch.float32)
    eps = 1e-5
    ref = torch.nn.functional.layer_norm(x, (256,), weight=w, bias=b, eps=eps)
    out = gn.layer_norm(x, weight=w, bias=b, eps=eps)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_governed_receipts_chain_and_verify():
    x = torch.randn(4, 128)
    start = gn.receipt_count()
    gn.rms_norm(x, eps=1e-6, governed=True)
    gn.layer_norm(x, eps=1e-5, governed=True)
    assert gn.receipt_count() == start + 2
    v = gn.receipt_verify()
    assert v["ok"] is True
    assert v["depth"] == gn.receipt_count()
    assert len(gn.receipt_head()) == 64  # SHA3-256 hex


def test_governed_off_by_default_records_nothing():
    before = gn.receipt_count()
    gn.rms_norm(torch.randn(2, 32), eps=1e-6)  # governed defaults False
    assert gn.receipt_count() == before


if __name__ == "__main__":
    test_rms_norm_matches_reference()
    test_layer_norm_matches_torch()
    test_governed_receipts_chain_and_verify()
    test_governed_off_by_default_records_nothing()
    print("ALL TESTS PASSED")

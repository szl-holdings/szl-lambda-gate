# SPDX-License-Identifier: Apache-2.0
"""Smoke hook for hf-kernel-publish: SZLHOLDINGS/szl-lambda-gate.

`smoke(kernel)` receives the module that the `kernels` loader imported, from a local
build (check-local) or from the Hub (verify). Raise, or return False, to fail the run.
Keep it CPU-only, deterministic and fast. The result's repr is written to the receipt.
"""
from __future__ import annotations


def smoke(kernel):
    import torch

    axes = torch.tensor([0.9, 0.8, 0.95], dtype=torch.float64)
    weights = torch.tensor([0.5, 0.25, 0.25], dtype=torch.float64)  # Lambda v1 requires fsum(weights) == 1
    score = kernel.lambda_v1(axes, weights)
    if not isinstance(score, torch.Tensor) or score.dtype != torch.float64 or score.shape != torch.Size([]):
        raise AssertionError("lambda_v1 must return a scalar float64 tensor")
    gate = kernel.lambda_v1_gate(axes, weights, 0.5)
    if gate.verdict != "GO":
        raise AssertionError(f"lambda_v1_gate(tau=0.5) expected GO, got {gate.verdict} ({gate.code})")
    verdict = kernel.selfcheck()
    if verdict.get("all_axioms_hold") is not True:
        raise AssertionError(f"selfcheck reported a violated axiom: {verdict}")
    norm = kernel.governed_norm.selfcheck()
    if norm.get("ok") is not True:
        raise AssertionError(f"governed_norm.selfcheck failed: {norm}")
    return {"lambda_v1": float(score), "verdict": gate.verdict, "code": gate.code,
            "all_axioms_hold": True, "governed_norm_receipt_head": norm["receipt_head"]}

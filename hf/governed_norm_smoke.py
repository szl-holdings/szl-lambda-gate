# SPDX-License-Identifier: Apache-2.0
"""Smoke hook for hf-kernel-publish: SZLHOLDINGS/szl-governed-norm.

The kernel is the standalone build of szl_lambda_gate.governed_norm, so it must
work without szl_lambda_gate being importable. `smoke(kernel)` receives the module
the `kernels` loader imported (check-local or verify); raise to fail the run.
"""
from __future__ import annotations


def smoke(kernel):
    import torch

    x = torch.arange(1.0, 9.0, dtype=torch.float32).reshape(2, 4)
    y = kernel.rms_norm(x)
    ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    if not torch.allclose(y, ref, rtol=1e-5, atol=1e-6):
        raise AssertionError("rms_norm differs from the float32 reference")
    verdict = kernel.selfcheck()
    if verdict.get("ok") is not True:
        raise AssertionError(f"selfcheck failed: {verdict}")
    return {"version": verdict["version"], "checks": verdict["checks"],
            "receipt_head": verdict["receipt_head"]}

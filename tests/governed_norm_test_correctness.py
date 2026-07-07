# SPDX-License-Identifier: Apache-2.0
# © 2026 SZL Holdings
"""Thorough correctness tests for szl_governed_norm vs PyTorch references.

Covers:
  * rms_norm vs a Llama-style float32 reference
  * layer_norm vs torch.nn.functional.layer_norm
  * dtypes fp32 / fp16 / bf16 (fp16 & bf16 use looser tolerances)
  * shapes: 1D edge case, batched, large last-dim
  * with and without affine weight / bias

All comparisons use torch.testing.assert_close. CPU-only so it runs anywhere;
this is a *universal* (pure-Python) kernel, validated as a correctness
reference, not a tuned-CUDA speed claim.

Run:  python -m pytest tests/test_correctness.py -q
"""
import sys
from pathlib import Path

import pytest
import torch

# Import the built universal kernel package directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "torch-ext"))
import szl_lambda_gate.governed_norm as gn  # noqa: E402  (folded-in governed-norm kernels)


# --- dtype / tolerance matrix --------------------------------------------------
# fp16 and bf16 accumulate rounding error; loosen tolerances accordingly. The
# kernel computes in float32 internally then casts back, so the dominant error
# is the final cast to the low-precision dtype.
_DTYPES = [torch.float32]
if hasattr(torch, "float16"):
    _DTYPES.append(torch.float16)
if hasattr(torch, "bfloat16"):
    _DTYPES.append(torch.bfloat16)

_TOL = {
    torch.float32: dict(rtol=1e-5, atol=1e-5),
    torch.float16: dict(rtol=3e-3, atol=3e-3),
    torch.bfloat16: dict(rtol=2e-2, atol=2e-2),
}

# (label, shape) — 1D edge case, batched 2D, batched 3D, large last dim.
_SHAPES = [
    ("1d_edge", (1024,)),
    ("batched_2d", (8, 512)),
    ("batched_3d", (2, 4, 256)),
    ("large_lastdim", (2, 8192)),
    ("small_lastdim", (16, 3)),
]


def _ids(prefix, items):
    return [f"{prefix}-{lbl}" for lbl, _ in items]


# --- reference implementations -------------------------------------------------
def _ref_rms_norm(x, weight, eps):
    """Llama-style RMSNorm reference: float32 compute, cast back, then scale."""
    in_dtype = x.dtype
    xf = x.to(torch.float32)
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    out = xf.to(in_dtype)
    if weight is not None:
        out = out * weight
    return out


def _make(shape, dtype):
    torch.manual_seed(1234)
    # Build in fp32 then cast so all dtypes see the same logical values.
    x = torch.randn(*shape, dtype=torch.float32).to(dtype)
    last = shape[-1]
    w = torch.randn(last, dtype=torch.float32).to(dtype)
    b = torch.randn(last, dtype=torch.float32).to(dtype)
    return x, w, b


# --- rms_norm ------------------------------------------------------------------
@pytest.mark.parametrize("dtype", _DTYPES, ids=[str(d).replace("torch.", "") for d in _DTYPES])
@pytest.mark.parametrize("shape", [s for _, s in _SHAPES], ids=_ids("rms", _SHAPES))
@pytest.mark.parametrize("with_weight", [True, False], ids=["w", "now"])
def test_rms_norm_matches_reference(dtype, shape, with_weight):
    x, w, _ = _make(shape, dtype)
    eps = 1e-6
    weight = w if with_weight else None
    ref = _ref_rms_norm(x, weight, eps)
    out = gn.rms_norm(x, weight=weight, eps=eps)
    assert out.shape == x.shape
    assert out.dtype == dtype
    torch.testing.assert_close(out, ref, **_TOL[dtype])


# --- layer_norm ----------------------------------------------------------------
@pytest.mark.parametrize("dtype", _DTYPES, ids=[str(d).replace("torch.", "") for d in _DTYPES])
@pytest.mark.parametrize("shape", [s for _, s in _SHAPES], ids=_ids("ln", _SHAPES))
@pytest.mark.parametrize(
    "with_weight,with_bias",
    [(True, True), (True, False), (False, True), (False, False)],
    ids=["wb", "w", "b", "none"],
)
def test_layer_norm_matches_torch(dtype, shape, with_weight, with_bias):
    x, w, b = _make(shape, dtype)
    eps = 1e-5
    weight = w if with_weight else None
    bias = b if with_bias else None
    normalized_shape = (shape[-1],)
    # torch's F.layer_norm requires weight/bias dtype match; it computes the
    # affine in the input dtype, matching the kernel's cast-then-scale path.
    ref = torch.nn.functional.layer_norm(
        x, normalized_shape, weight=weight, bias=bias, eps=eps
    )
    out = gn.layer_norm(x, weight=weight, bias=bias, eps=eps)
    assert out.shape == x.shape
    assert out.dtype == dtype
    torch.testing.assert_close(out, ref, **_TOL[dtype])


# --- statistical sanity checks (fp32) -----------------------------------------
def test_layer_norm_zero_mean_unit_var_fp32():
    """Without affine, LayerNorm output has ~0 mean and ~1 var on last dim."""
    torch.manual_seed(7)
    x = torch.randn(64, 512, dtype=torch.float32)
    out = gn.layer_norm(x, eps=1e-5)
    torch.testing.assert_close(
        out.mean(-1), torch.zeros(64), rtol=0, atol=1e-4
    )
    torch.testing.assert_close(
        out.var(-1, unbiased=False), torch.ones(64), rtol=1e-3, atol=1e-3
    )


def test_rms_norm_unit_rms_fp32():
    """Without weight, RMSNorm output has ~unit root-mean-square on last dim."""
    torch.manual_seed(8)
    x = torch.randn(64, 512, dtype=torch.float32)
    out = gn.rms_norm(x, eps=1e-8)
    rms = out.pow(2).mean(-1).sqrt()
    torch.testing.assert_close(rms, torch.ones(64), rtol=1e-3, atol=1e-3)


def test_governed_path_returns_same_values_fp32():
    """governed=True must not change numerical output, only emit a receipt."""
    torch.manual_seed(9)
    x = torch.randn(4, 256, dtype=torch.float32)
    w = torch.randn(256, dtype=torch.float32)
    plain = gn.rms_norm(x, weight=w, eps=1e-6, governed=False)
    governed = gn.rms_norm(x, weight=w, eps=1e-6, governed=True)
    torch.testing.assert_close(plain, governed, rtol=0, atol=0)

    plain_ln = gn.layer_norm(x, weight=w, eps=1e-5, governed=False)
    gov_ln = gn.layer_norm(x, weight=w, eps=1e-5, governed=True)
    torch.testing.assert_close(plain_ln, gov_ln, rtol=0, atol=0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

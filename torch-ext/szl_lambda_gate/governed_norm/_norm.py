# SPDX-License-Identifier: Apache-2.0
# © 2026 SZL Holdings · Stephen P. Lutar · ORCID 0009-0001-0110-4173
"""Pure-PyTorch normalization primitives for the SZL governed-norm kernel.

These are correctness-verified reference implementations (RMSNorm, LayerNorm,
and the residual-fused RMSNorm pattern used by transformer blocks) written in
pure PyTorch. They run on CPU and CUDA, are torch.compile-friendly, and depend
ONLY on torch + the Python standard library (a Kernel Hub requirement for
universal kernels).

HONESTY: this is a *universal* (pure-Python) kernel. It does NOT ship a
hand-tuned CUDA/Triton binary, so it is a correctness reference, not a
speed record. We make no fabricated benchmark claims. Where it adds value
is the optional *governed* path (see _receipt.py): every normalization call
can emit a content-addressed, hash-chained receipt of its inputs/outputs so
the operation is auditable — SZL Holdings' provenance doctrine applied at
the kernel layer.

Numerical convention (all ops): reductions and the normalization math are
computed in float32 for stability, then the result is cast back to the input
dtype. This is the standard Llama-style convention and is what makes
float16 / bfloat16 inputs numerically well-behaved.

Validation convention: guards below are cheap, branch-only checks on metadata
(dtype / ndim / shape / device) — they allocate nothing on the happy path and
constant-fold away under torch.compile, so they do not perturb traced graphs.
They exist to turn silent broadcasting / device-mismatch bugs into clear,
early errors. A zero-size normalized last dimension is rejected (normalizing
over zero elements is undefined); a single-element last dimension is allowed
(RMSNorm yields sign(x); LayerNorm yields 0, matching F.layer_norm).

Non-finite convention (NaN / Inf inputs): these ops do NOT sanitize their
input. A NaN or Inf in the input propagates through the reduction and appears
in the output, exactly as it would in torch.nn.functional.layer_norm / a
hand-written kernel. We deliberately do NOT silently replace non-finite values
(that would hide upstream numerical bugs); detecting/handling them is the
caller's responsibility. This propagation behavior is covered by regression
tests so it cannot change unnoticed.
"""
from typing import Optional

import torch

# Floating dtypes this kernel supports. Integer / complex inputs are rejected
# early with a clear message rather than silently producing garbage.
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


def _compute_dtype(in_dtype: torch.dtype) -> torch.dtype:
    """Reduction/normalization compute dtype.

    Low-precision inputs (fp16/bf16) are upcast to float32 for stability — the
    standard Llama-style convention. float64 inputs are NOT downcast: doing so
    would silently lose precision (and break gradcheck), so we keep float64.
    """
    return torch.float32 if in_dtype in (torch.float16, torch.bfloat16) else in_dtype


def _check_input(x: torch.Tensor, name: str = "x") -> None:
    """Cheap, allocation-free guards on the primary input tensor.

    Only inspects metadata (type / dtype / ndim), so it is constant-folded by
    torch.compile and adds no runtime tensor work on the happy path.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(x).__name__}")
    if x.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(
            f"{name} has unsupported dtype {x.dtype}; "
            f"expected one of {tuple(str(d) for d in _SUPPORTED_DTYPES)}"
        )
    if x.dim() < 1:
        raise ValueError(
            f"{name} must have at least 1 dimension (the normalized dim); "
            f"got a {x.dim()}-d tensor"
        )
    # A zero-size normalized (last) dimension is mathematically undefined:
    # mean/RMS over zero elements is NaN, so normalization has no meaning.
    # Reject it early with a clear message instead of silently returning an
    # empty/NaN tensor (the classic shape-bug-masquerading-as-success case).
    if x.shape[-1] == 0:
        raise ValueError(
            f"{name} has a zero-size normalized last dimension {tuple(x.shape)}; "
            f"normalization over zero elements is undefined"
        )


def _check_affine(
    x: torch.Tensor,
    param: Optional[torch.Tensor],
    name: str,
) -> None:
    """Validate an optional affine parameter (weight/bias/residual peer).

    Enforces that the parameter is 1-D and matches the normalized (last)
    dimension, and lives on the same device as ``x``. This catches the
    classic silent-broadcast bug where a mis-shaped weight would broadcast
    instead of erroring. Metadata-only: no allocations.
    """
    if param is None:
        return
    if not isinstance(param, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor or None, got {type(param).__name__}")
    if param.device != x.device:
        raise ValueError(
            f"{name} is on device {param.device} but x is on {x.device}; "
            f"move them to the same device"
        )
    last = x.shape[-1]
    if param.dim() != 1 or param.shape[0] != last:
        raise ValueError(
            f"{name} must be 1-D with shape ({last},) to match the normalized "
            f"last dimension of x; got shape {tuple(param.shape)}"
        )


def _check_eps(eps: float) -> None:
    """eps must be a positive, finite scalar (rsqrt(var+eps) must be safe)."""
    e = float(eps)
    if not (e > 0.0) or e != e or e == float("inf"):
        raise ValueError(f"eps must be a positive finite float, got {eps!r}")


def rms_norm(
    x: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Root-mean-square layer normalization over the last dimension.

    y = x / sqrt(mean(x^2, dim=-1) + eps) * weight

    Computed in float32 for numerical stability, then cast back to the input
    dtype (the standard, correctness-preserving convention used by Llama-style
    RMSNorm). `weight` is optional; when omitted, no affine scale is applied.

    Raises clear TypeError/ValueError on bad dtype, rank, eps, or a weight
    whose shape/device does not match x's normalized dimension.
    """
    _check_input(x)
    _check_eps(eps)
    _check_affine(x, weight, "weight")

    in_dtype = x.dtype
    xf = x.to(_compute_dtype(in_dtype))
    variance = xf.pow(2).mean(dim=-1, keepdim=True)
    xf = xf * torch.rsqrt(variance + eps)
    out = xf.to(in_dtype)
    if weight is not None:
        out = out * weight
    return out


def layer_norm(
    x: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Standard layer normalization over the last dimension.

    Mean/variance computed in float32 for stability, then cast back. Matches
    torch.nn.functional.layer_norm semantics for the normalized-shape = last
    dim case; verified against it in the test suite.

    Raises clear TypeError/ValueError on bad dtype, rank, eps, or a
    weight/bias whose shape/device does not match x's normalized dimension.
    """
    _check_input(x)
    _check_eps(eps)
    _check_affine(x, weight, "weight")
    _check_affine(x, bias, "bias")

    in_dtype = x.dtype
    xf = x.to(_compute_dtype(in_dtype))
    mean = xf.mean(dim=-1, keepdim=True)
    # Biased (population) variance = mean of squared deviations. We compute it
    # directly rather than via Tensor.var(unbiased=False): torch's .var emits a
    # "degrees of freedom <= 0" UserWarning when the normalized dim has a single
    # element, even though unbiased=False is well-defined there (variance 0).
    # Computing it ourselves matches F.layer_norm exactly and stays silent and
    # torch.compile(fullgraph=True)-clean for the single-element edge case.
    centered = xf - mean
    var = centered.pow(2).mean(dim=-1, keepdim=True)
    xf = centered * torch.rsqrt(var + eps)
    out = xf.to(in_dtype)
    if weight is not None:
        out = out * weight
    if bias is not None:
        out = out + bias
    return out


def fused_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
):
    """Residual-add followed by RMSNorm — the canonical transformer block pattern.

        h = x + residual          # updated residual stream
        y = rms_norm(h, weight, eps)
        return y, h

    This mirrors the `fused_add_rms_norm` used in real LLM inference stacks
    (e.g. the pre-norm transformer block: the normalized output `y` feeds the
    sublayer, while the un-normalized sum `h` is carried forward as the next
    residual). We return BOTH so callers can thread the residual stream, which
    is exactly why the fused form exists.

    HONESTY: "fused" here means *logically* fused (one Python op, one float32
    cast path, the add done in float32 alongside the norm) — it is a correct,
    allocation-conscious pure-PyTorch reference, not a hand-written fused CUDA
    kernel. No speed claims are made.

    The add is performed in float32 so that, for float16/bfloat16 inputs, the
    residual accumulation does not lose precision before normalization — this
    matches high-quality reference implementations.
    """
    _check_input(x, "x")
    _check_input(residual, "residual")
    _check_eps(eps)
    if residual.shape != x.shape:
        raise ValueError(
            f"residual shape {tuple(residual.shape)} must equal x shape "
            f"{tuple(x.shape)} for the residual add"
        )
    if residual.device != x.device:
        raise ValueError(
            f"residual is on device {residual.device} but x is on {x.device}; "
            f"move them to the same device"
        )
    _check_affine(x, weight, "weight")

    in_dtype = x.dtype
    cdt = _compute_dtype(in_dtype)
    # Add in compute dtype, keep both the normalized output and the residual.
    hf = x.to(cdt) + residual.to(cdt)
    new_residual = hf.to(in_dtype)
    variance = hf.pow(2).mean(dim=-1, keepdim=True)
    yf = hf * torch.rsqrt(variance + eps)
    out = yf.to(in_dtype)
    if weight is not None:
        out = out * weight
    return out, new_residual

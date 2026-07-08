# SPDX-License-Identifier: Apache-2.0
# © 2026 SZL Holdings · Stephen P. Lutar · ORCID 0009-0001-0110-4173
"""szl_lambda_gate — the Lambda-Spine aggregator (Λ) as a universal kernel.

A pure-PyTorch (universal) kernel from SZL Holdings for the Hugging Face
Kernel Hub. It ports the canonical Λ aggregator into a differentiable,
torch.compile-friendly torch op:

    Λ(x) = ∏ xᵢ^{wᵢ},  Σwᵢ = 1,  wᵢ > 0,  xᵢ ∈ [0,1]   (weighted geometric mean)

plus an ADVISORY governance gate (Λ vs threshold), the four carried axioms as
real runtime self-checks, and pure nn.Module layers.

Load from the Hub:

    import torch
    from kernels import get_kernel

    lg = get_kernel("SZLHOLDINGS/szl-lambda-gate")
    axes = torch.tensor([0.9, 0.8, 0.95])        # axis scores in [0,1]
    score = lg.lambda_aggregate(axes)            # Λ(x) ∈ [0,1]
    res = lg.lambda_gate(axes, threshold=0.5)    # ADVISORY pass/fail
    print(res.score, res.passed, res.advisory)

WHAT Λ IS / IS NOT (HONESTY — SZL Holdings doctrine v11):
  Λ is the weighted-geometric-mean aggregator — a non-compensatory, ADVISORY
  way to roll axis scores in [0,1] into one number (any zeroed axis zeroes the
  aggregate). It is NOT "proven trust" and NOT a closed theorem: Λ-uniqueness
  remains Conjecture 1 (OPEN — an unresolved CAUCHY_ND step plus a missing
  symmetry axiom). Label it honestly everywhere; a gate "pass" is advisory.

PROVENANCE: backed by the Lean 4 formalization szl-holdings/lutar-lean
  (749 declarations / 14 axioms / 163 tracked sorries),
  DOI 10.5281/zenodo.20434308 (lutar-lean). Λ uniqueness = Conjecture 1 (open).
"""
from typing import Optional

import torch

from . import layers  # noqa: F401  (must be importable for Hub layer mapping)
# CONSOLIDATION (Wave D): the governed-norm universal kernel is folded in here
# as a subpackage so szl-lambda-gate is the ONE canonical kernels package. The
# source repo szl-holdings/szl-governed-norm is DEPRECATED and points here;
# nothing was deleted (additive, reversible copy). Λ stays Conjecture 1.
from . import governed_norm  # noqa: F401  (folded-in governed normalization kernels)
from ._lambda import YUYAY_AXES, YUYAY_FLOORS, LambdaGateResult
from ._lambda import find_axiom_violation as _find_axiom_violation
from ._lambda import is_bounded_by_max as _is_bounded_by_max
from ._lambda import is_egyptian_exact as _is_egyptian_exact
from ._lambda import is_homogeneous as _is_homogeneous
from ._lambda import is_monotone as _is_monotone
from ._lambda import lambda_aggregate as _lambda_aggregate
from ._lambda import lambda_gate as _lambda_gate
from ._lambda import lambda_gate_batch as _lambda_gate_batch
from ._lambda import selfcheck as _selfcheck
from ._lambda import yuyay_weights as _yuyay_weights

__all__ = [
    "lambda_aggregate",
    "lambda_gate",
    "lambda_gate_batch",
    "LambdaGateResult",
    "is_monotone",
    "is_egyptian_exact",
    "is_bounded_by_max",
    "is_homogeneous",
    "find_axiom_violation",
    "selfcheck",
    "yuyay_weights",
    "YUYAY_AXES",
    "YUYAY_FLOORS",
    "layers",
    "DOCTRINE_FOOTER",
    "PROVENANCE",
    "__version__",
    # ---- folded-in governed-norm kernels (Wave D consolidation) ----
    "governed_norm",
    "rms_norm",
    "layer_norm",
    "fused_add_rms_norm",
]

# ---- folded-in governed-norm surface (Wave D consolidation) ---------------- #
# Convenience top-level re-exports of the governed normalization kernels that
# were absorbed from szl-governed-norm. The full surface (ReceiptChain,
# emit_receipt, receipt_* helpers, selfcheck, layers) lives under
# ``szl_lambda_gate.governed_norm``. These are a DIFFERENT kernel family from Λ
# (normalization, not the Λ aggregator); Λ itself remains Conjecture 1
# (advisory, uniqueness OPEN) and is never described as proven trust.
rms_norm = governed_norm.rms_norm
layer_norm = governed_norm.layer_norm
fused_add_rms_norm = governed_norm.fused_add_rms_norm

__version__ = "0.2.0"
DOCTRINE_FOOTER = (
    "SZL Holdings · Λ = Conjecture 1 (ADVISORY, weighted geometric mean) · "
    "uniqueness OPEN · NOT proven trust · honesty over checklist"
)
PROVENANCE = {
    "lean_repo": "szl-holdings/lutar-lean",
    "lean_declarations": 749,
    "lean_axioms": 14,
    "lean_tracked_sorries": 163,
    "doi_lutar_lean": "10.5281/zenodo.20434308",
    "lambda_status": "Conjecture 1 (open) — uniqueness unproven; advisory only",
}


def lambda_aggregate(
    axes: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Λ(x) = ∏ xᵢ^{wᵢ}, the weighted geometric mean over the last dim of axes.

    See ``szl_lambda_gate._lambda.lambda_aggregate``. Axis scores in [0,1],
    uniform weights when ``weights`` is None. Differentiable, batched, and
    torch.compile-friendly. ADVISORY — NOT proven trust.
    """
    return _lambda_aggregate(axes, weights=weights)


def lambda_gate(
    axes: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    threshold: float = 0.5,
) -> LambdaGateResult:
    """ADVISORY Λ governance gate: returns LambdaGateResult(score, passed,
    threshold, advisory). ``passed`` = Λ(axes) >= threshold. ``threshold`` must
    lie within Λ's range [0,1] (a value outside it is a misconfiguration — a
    negative threshold would advisory-pass a fully-failing Λ=0 candidate — and
    is rejected). A pass is an advisory, non-compensatory signal — NOT proven
    trust (Λ = Conjecture 1).
    """
    return _lambda_gate(axes, weights=weights, threshold=threshold)


def lambda_gate_batch(
    candidates: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    threshold: float = 0.5,
) -> LambdaGateResult:
    """ADVISORY batch gate over many candidate action-vectors (shape (..., N, k)).

    The realistic per-inference-step call: score all N candidates at once and
    return the advisory pass mask. Returns LambdaGateResult(score, passed,
    threshold, advisory) with score/passed of shape (..., N). ``threshold``
    must lie within Λ's range [0,1] (same domain guard as ``lambda_gate``).
    NOT proven trust.
    """
    return _lambda_gate_batch(candidates, weights=weights, threshold=threshold)


def yuyay_weights(dtype: torch.dtype = torch.float64, device=None) -> torch.Tensor:
    """Canonical 13-axis Yuyay Λ weight vector (uniform 1/13), ADVISORY only.

    Use as ``weights`` over the 13 ``YUYAY_AXES``. The yuyay_v3 gate is a
    conjunctive AND with per-axis floors (``YUYAY_FLOORS``); this Λ roll-up is
    the weighted geometric mean and is ADVISORY — NOT proven trust.
    """
    return _yuyay_weights(dtype=dtype, device=device)


def find_axiom_violation(k=5, trials=200, weights=None, seed=0, tol=1e-6):
    """Random-search for any A1–A4 violation; returns (axiom, axes, weights) or
    None. An honest falsification attempt — finding nothing is evidence, not a
    proof (Λ-uniqueness is Conjecture 1, open).
    """
    return _find_axiom_violation(k=k, trials=trials, weights=weights, seed=seed, tol=tol)


def selfcheck(k=5, trials=64, seed=0) -> dict:
    """Expose the A1–A4 empirical self-checks + version as a single verdict dict.

    Callable as get_kernel(...).selfcheck(). EMPIRICAL checks on sampled inputs,
    NOT a proof of Λ-uniqueness (Conjecture 1, open). Advisory only.
    """
    return _selfcheck(k=k, trials=trials, seed=seed)


# ---- axiom runtime self-checks (real, verifiable; NOT a uniqueness proof) -- #
def is_monotone(axes, weights=None, delta=0.05, tol=1e-7) -> bool:
    """A1 IsMonotone self-check: Λ is non-decreasing in each axis (on this data)."""
    return _is_monotone(axes, weights=weights, delta=delta, tol=tol)


def is_egyptian_exact(c, k=3, weights=None, tol=1e-5) -> bool:
    """A3 IsEgyptianExact self-check: Λ(c, …, c) = c."""
    return _is_egyptian_exact(c, k=k, weights=weights, tol=tol)


def is_bounded_by_max(axes, weights=None, tol=1e-6) -> bool:
    """A4 IsBounded self-check: Λ(x) ≤ maxᵢ xᵢ."""
    return _is_bounded_by_max(axes, weights=weights, tol=tol)


def is_homogeneous(axes, t, weights=None, tol=1e-5) -> bool:
    """A2 IsHomogeneous(degree 1) self-check: Λ(t·x) = t·Λ(x)."""
    return _is_homogeneous(axes, t, weights=weights, tol=tol)

# SPDX-License-Identifier: Apache-2.0
# © 2026 SZL Holdings · Stephen P. Lutar · ORCID 0009-0001-0110-4173
"""Pure-PyTorch Lambda-Spine aggregator (Λ) for the szl-lambda-gate kernel.

Λ(x) = ∏ xᵢ^{wᵢ},  Σwᵢ = 1,  wᵢ > 0,  xᵢ ∈ [0,1]   (weighted geometric mean)

This is a TORCH port of the canonical pure-Python reference
(packages/puriq-os/puriq_os/lambda_aggregator.py — saved alongside this kernel
as lambda_aggregator_source.py). It is a correctness reference, computed via
logs in float32 for stability, differentiable (autograd works), and
torch.compile-friendly. Depends ONLY on torch + the Python standard library
(a Kernel Hub requirement for universal kernels).

WHAT Λ IS / IS NOT (HONESTY — SZL Holdings doctrine v11):
  Λ is the *weighted-geometric-mean aggregator*: a non-compensatory way to
  combine axis scores in [0,1] into one number. It is ADVISORY governance
  signal — a conservative roll-up where any single zeroed axis drives the
  aggregate to 0. It is NOT "proven trust" and NOT a closed theorem. Its
  *uniqueness* (that the weighted geometric mean is the only aggregator
  satisfying the carried axioms) remains Conjecture 1 — OPEN (an unresolved
  CAUCHY_ND step plus a missing symmetry axiom in the Lean development). Do
  not describe Λ as proven trust anywhere.

PRIOR ART (honest attribution): the weighted geometric mean as a *less-
  compensatory* composite-indicator aggregator is established practice — the
  UN HDI (arithmetic→geometric switch, 2010), the OECD Handbook on
  Constructing Composite Indicators (2008), and the UNECE well-being
  guidelines all use it "to limit the compensation effect". The veto / cut-off
  idea (a single failing criterion blocks a pass regardless of the others) is
  the ELECTRE veto threshold / "satisficing" minimum-threshold screen. The
  13-axis conjunctive form exposed by :func:`yuyay_weights` is SZL's own
  yuyay_v3 "Heart" gate. None of this makes Λ "proven trust"; the gate is
  ADVISORY (a11oy: "the advisory Λ trust score is a research conjecture, not a
  pass/fail oracle").

PROVENANCE: backed by the Lean 4 formalization szl-holdings/lutar-lean
  (749 declarations / 14 axioms / 163 tracked sorries),
  DOI 10.5281/zenodo.20434308 (lutar-lean).
  Λ uniqueness = Conjecture 1 (open).

Axioms carried (Lutar/Axioms.lean), available below as runtime self-checks:
  A1 IsMonotone        — Λ is non-decreasing in each axis
  A2 IsHomogeneous     — Λ(t·x) = t·Λ(x)  (degree 1)
  A3 IsEgyptianExact   — Λ(c,…,c) = c       (the uniform-diagonal fixpoint)
  A4 IsBounded(by max) — Λ(x) ≤ maxᵢ xᵢ
"""
from typing import Optional

import torch

# Compute reductions/log-sum in float32 for stability when inputs are low
# precision; keep float64 inputs in float64 (downcasting would break gradcheck
# and silently lose precision).
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


def _compute_dtype(in_dtype: torch.dtype) -> torch.dtype:
    return torch.float32 if in_dtype in (torch.float16, torch.bfloat16) else in_dtype


def _check_axes(axes: torch.Tensor) -> None:
    """Cheap, allocation-free metadata guards on the axis-score tensor.

    Inspects only type / dtype / rank / last-dim, so it constant-folds under
    torch.compile and adds no tensor work on the happy path.
    """
    if not isinstance(axes, torch.Tensor):
        raise TypeError(f"axes must be a torch.Tensor, got {type(axes).__name__}")
    if axes.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(
            f"axes has unsupported dtype {axes.dtype}; "
            f"expected one of {tuple(str(d) for d in _SUPPORTED_DTYPES)}"
        )
    if axes.dim() < 1:
        raise ValueError(
            "axes must have at least 1 dimension (the k axis scores live on "
            f"the last dim); got a {axes.dim()}-d tensor"
        )
    if axes.shape[-1] < 1:
        raise ValueError("axes last dimension (k = number of axes) must be >= 1")


def _resolve_weights(
    axes: torch.Tensor,
    weights: Optional[torch.Tensor],
    cdt: torch.dtype,
) -> torch.Tensor:
    """Return a normalized (Σw = 1) weight vector of shape (k,) in compute dtype.

    ``weights=None`` -> uniform 1/k (the Egyptian-exact diagonal). Otherwise the
    weights must be 1-D of length k, strictly positive, with a positive sum;
    they are normalized so Σwᵢ = 1.
    """
    k = axes.shape[-1]
    if weights is None:
        return torch.full((k,), 1.0 / k, dtype=cdt, device=axes.device)
    if not isinstance(weights, torch.Tensor):
        raise TypeError(f"weights must be a torch.Tensor or None, got {type(weights).__name__}")
    if weights.device != axes.device:
        raise ValueError(
            f"weights is on device {weights.device} but axes is on {axes.device}; "
            "move them to the same device"
        )
    if weights.dim() != 1 or weights.shape[0] != k:
        raise ValueError(
            f"weights must be 1-D with shape ({k},) to match the last dim of axes; "
            f"got shape {tuple(weights.shape)}"
        )
    wf = weights.to(cdt)
    # Reject non-finite weights up front: a NaN/Inf weight is meaningless for a
    # governance roll-up and would silently poison the normalization.
    if not bool(torch.all(torch.isfinite(wf))):
        raise ValueError("weights must all be finite (no NaN/Inf)")
    # Positivity / sum guards mirror the pure-Python reference (wᵢ>0, Σw>0).
    if bool(torch.any(wf <= 0.0)):
        raise ValueError("weights must be strictly positive (wᵢ > 0)")
    sw = wf.sum()
    if not bool(sw > 0.0):
        raise ValueError("weights must sum to a positive value")
    return wf / sw


def lambda_aggregate(
    axes: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Weighted geometric mean Λ(x) = ∏ xᵢ^{wᵢ} over the last dim of ``axes``.

    Λ is the (ADVISORY) Lambda-Spine aggregator. Axis scores are expected in
    [0,1] and are clamped into [0,1]; uniform weights (1/k) are used when
    ``weights`` is None — the Egyptian-exact diagonal. Computed via logs in
    float32 (or float64 for float64 inputs) for numerical stability:

        Λ(x) = exp( Σᵢ wᵢ · log(clamp(xᵢ, 0, 1)) )

    Non-compensatory zero-routing (A4-consistent): any axis that is zero, OR
    that is NON-FINITE (NaN / ±Inf), is treated as a FAILING axis and drives
    the whole aggregate to exactly 0. This is the conservative governance
    choice — a garbage/invalid axis must never silently pass as a "perfect"
    (clamped-to-1) axis, and the output (and its gradient) stay finite and in
    [0,1] for every input. Zeros/non-finite axes are routed explicitly so
    log(0) = -inf and log(NaN) = NaN never produce a NaN value or gradient.

    Args:
        axes:    tensor of shape (..., k) of axis scores in [0,1]. Batched:
                 the reduction is over the last dim, leading dims are batch.
        weights: optional 1-D tensor of shape (k,); None -> uniform. Normalized
                 internally so Σwᵢ = 1.

    Returns:
        tensor of shape (...) — Λ(x) ∈ [0,1] per batch row. Differentiable
        w.r.t. ``axes`` (and ``weights``).

    HONESTY: this is a non-compensatory governance roll-up, NOT proven trust.
    Λ-uniqueness is Conjecture 1 (open).
    """
    _check_axes(axes)
    in_dtype = axes.dtype
    cdt = _compute_dtype(in_dtype)
    xf = axes.to(cdt)
    w = _resolve_weights(axes, weights, cdt)  # (k,), Σw=1

    # A "bad" axis is one that fails non-compensatorily: a non-positive score
    # OR a non-finite value (NaN / ±Inf). clamp(+inf)=1 would otherwise count a
    # garbage axis as perfect, and clamp(NaN)=NaN would poison the product — we
    # treat BOTH as failing (zeroing) axes. Detect non-finite on the RAW input.
    finite_mask = torch.isfinite(xf)
    xc = xf.clamp(0.0, 1.0)
    bad_mask = (~finite_mask) | (xc <= 0.0)
    any_bad = torch.any(bad_mask, dim=-1)  # (...)

    # Replace bad axes with 1.0 before the log purely to keep log finite and the
    # gradient well-defined; the bad-axis contribution is reinstated via any_bad.
    safe = torch.where(bad_mask, torch.ones_like(xc), xc)
    logx = torch.log(safe)                    # (..., k)
    acc = (logx * w).sum(dim=-1)              # (...)  weighted log-sum
    val = torch.exp(acc)                      # (...)  Λ before zero-routing

    out = torch.where(any_bad, torch.zeros_like(val), val)
    out = out.clamp(0.0, 1.0)
    return out.to(in_dtype)


def lambda_gate(
    axes: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    threshold: float = 0.5,
):
    """ADVISORY governance gate over Λ(x): score plus a pass/fail vs threshold.

    Computes Λ(x) (see :func:`lambda_aggregate`) and compares it to
    ``threshold``: pass := Λ(x) >= threshold.

    Returns a :class:`LambdaGateResult` namedtuple with fields:
        score     — Λ(x) tensor of shape (...), in [0,1]
        passed    — boolean tensor of shape (...), Λ(x) >= threshold
        threshold — the float threshold used
        advisory  — always True; a STANDING reminder that this is a
                    non-compensatory governance signal, NOT proven trust.

    HONESTY: a "pass" is an ADVISORY signal only. Λ is the weighted-geometric-
    mean aggregator; its uniqueness is Conjecture 1 (open). Do not treat a
    pass as proven trust or a closed theorem.
    """
    t = float(threshold)
    if t != t or t == float("inf") or t == float("-inf"):
        raise ValueError(f"threshold must be a finite float, got {threshold!r}")
    score = lambda_aggregate(axes, weights)
    passed = score >= t
    return LambdaGateResult(score=score, passed=passed, threshold=t, advisory=True)


def lambda_gate_batch(
    candidates: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    threshold: float = 0.5,
):
    """ADVISORY batch gate: score MANY candidate action-vectors in one call.

    This is the realistic way a model/agent uses the gate — one call per
    inference step that scores every proposed action-vector at once and returns
    the advisory pass mask (which candidates clear the threshold).

    ``candidates`` is a tensor of shape (..., N, k): the last dim ``k`` holds
    the per-axis scores of a single candidate, and the second-to-last dim ``N``
    enumerates the candidates (any leading dims are extra batch). Equivalent to
    calling :func:`lambda_gate` on the whole tensor — the reduction is over the
    last dim — but named to make the agent-loop intent explicit.

    Returns a :class:`LambdaGateResult` with:
        score     — Λ tensor of shape (..., N), one score per candidate
        passed    — boolean mask of shape (..., N): score >= threshold
        threshold — the float threshold used
        advisory  — always True (NOT proven trust)

    HONESTY: the pass mask is an ADVISORY, non-compensatory signal. A "pass"
    is not proven trust; Λ-uniqueness is Conjecture 1 (open).
    """
    _check_axes(candidates)
    if candidates.dim() < 2:
        raise ValueError(
            "candidates must be at least 2-D, shape (..., N, k): the last dim is "
            f"the k axis scores and the one before it enumerates the N candidates; "
            f"got a {candidates.dim()}-d tensor"
        )
    # Reuse the single-call gate — its reduction over the last dim already gives
    # one score per candidate, so the (..., N) layout falls out for free.
    return lambda_gate(candidates, weights=weights, threshold=threshold)


# ---- A1..A4 axiom RUNTIME self-checks (real, verifiable) ------------------- #
# These are honest empirical checks callers can run on concrete inputs. They
# verify the carried axioms hold for THIS implementation on the given data —
# they are NOT a proof of Λ-uniqueness (that is Conjecture 1, open).

def is_egyptian_exact(
    c: float,
    k: int = 3,
    weights: Optional[torch.Tensor] = None,
    tol: float = 1e-5,
) -> bool:
    """A3 IsEgyptianExact: Λ(c, …, c) = c for a constant axis vector of length k.

    Builds the uniform vector (c repeated k times) and checks Λ equals c within
    ``tol``. ``c`` is clamped into [0,1] to match the aggregator's domain.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    cc = min(max(float(c), 0.0), 1.0)
    axes = torch.full((k,), cc, dtype=torch.float64)
    val = lambda_aggregate(axes, weights)
    return bool(torch.abs(val - cc) <= tol)


def is_bounded_by_max(
    axes: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    tol: float = 1e-6,
) -> bool:
    """A4 IsBounded: Λ(x) ≤ maxᵢ xᵢ (over the last dim), within ``tol``.

    Returns True iff the bound holds for every batch row. Non-finite axis
    values are clamped/zero-routed the same way the aggregator treats them, so
    the bound is checked on the conservative (finite) domain.
    """
    _check_axes(axes)
    val = lambda_aggregate(axes, weights)                       # (...)
    xf = axes.to(_compute_dtype(axes.dtype))
    # Mirror the aggregator: non-finite axes are failing (treated as 0) for the
    # purposes of the max bound, so the check matches the routed semantics.
    xf = torch.where(torch.isfinite(xf), xf, torch.zeros_like(xf))
    mx = xf.clamp(0.0, 1.0).amax(dim=-1)                        # (...)
    return bool(torch.all(val.to(mx.dtype) <= mx + tol))


def is_homogeneous(
    axes: torch.Tensor,
    t: float,
    weights: Optional[torch.Tensor] = None,
    tol: float = 1e-5,
) -> bool:
    """A2 IsHomogeneous (degree 1): Λ(t·x) = t·Λ(x) for scalar t in [0,1].

    Verified on the clamped domain: both ``axes`` and ``t*axes`` must remain in
    [0,1] for the identity to be meaningful, so ``axes`` is clamped to [0,1] and
    ``t`` to [0,1] before the comparison.
    """
    _check_axes(axes)
    tt = min(max(float(t), 0.0), 1.0)
    x = axes.to(torch.float64).clamp(0.0, 1.0)
    lhs = lambda_aggregate(x * tt, weights)
    rhs = tt * lambda_aggregate(x, weights)
    return bool(torch.all(torch.abs(lhs - rhs) <= tol))


def is_monotone(
    axes: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    delta: float = 0.05,
    tol: float = 1e-7,
) -> bool:
    """A1 IsMonotone: Λ is non-decreasing in each axis.

    For each axis j, nudges that axis UP by ``delta`` (clamped to stay ≤ 1) on
    every batch row and checks Λ does not decrease (within ``tol``). Rows that
    cannot move (already at 1) are skipped for that axis. A real check on the
    given data — not a symbolic proof.
    """
    _check_axes(axes)
    x = axes.to(torch.float64).clamp(0.0, 1.0)
    base = lambda_aggregate(x, weights)
    k = x.shape[-1]
    ok = True
    for j in range(k):
        bumped = x.clone()
        bumped[..., j] = (bumped[..., j] + float(delta)).clamp(0.0, 1.0)
        bumped_val = lambda_aggregate(bumped, weights)
        # Λ must not go DOWN when an axis goes UP.
        ok = ok and bool(torch.all(bumped_val - base >= -tol))
    return ok


# ---- Adversarial axiom search (honest: a falsification attempt) ------------ #
def find_axiom_violation(
    k: int = 5,
    trials: int = 200,
    weights: Optional[torch.Tensor] = None,
    seed: Optional[int] = 0,
    tol: float = 1e-6,
):
    """Random-search for ANY A1–A4 violation on random axis/weight draws.

    Returns the first ``(axiom, axes, weights)`` triple that violates a carried
    axiom within ``tol``, or ``None`` if none is found in ``trials`` draws. This
    is an honest FALSIFICATION attempt on this implementation — finding nothing
    is empirical evidence, NOT a proof (Λ-uniqueness is Conjecture 1, open).
    """
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(int(seed))
    for _ in range(int(trials)):
        x = torch.rand(k, generator=gen, dtype=torch.float64)
        w = weights
        if w is None:
            w = torch.rand(k, generator=gen, dtype=torch.float64) + 1e-3
        # A3 on a constant draw
        c = float(torch.rand(1, generator=gen).item())
        if not is_egyptian_exact(c, k=k, weights=w, tol=max(tol, 1e-5)):
            return ("A3_IsEgyptianExact", torch.full((k,), c, dtype=torch.float64), w)
        # A4 bounded-by-max
        if not is_bounded_by_max(x, w, tol=max(tol, 1e-6)):
            return ("A4_IsBounded", x, w)
        # A2 homogeneous at a random t
        t = float(torch.rand(1, generator=gen).item())
        if not is_homogeneous(x, t, weights=w, tol=max(tol, 1e-5)):
            return ("A2_IsHomogeneous", x, w)
        # A1 monotone (leave headroom so an up-bump stays in range)
        if not is_monotone(x * 0.9, w, tol=max(tol, 1e-7)):
            return ("A1_IsMonotone", x * 0.9, w)
    return None


# ---- Canonical 13-axis Yuyay preset (ADVISORY ONLY) ------------------------ #
# SZL's own yuyay_v3 "Heart" gate is a 13-axis CONJUNCTIVE-AND screen (each axis
# independently clears its floor — no compensation). We expose its published
# axis NAMES and per-axis FLOORS as advisory metadata, and a uniform Λ weight
# vector over the 13 axes. This is ADVISORY: Λ here is still the weighted
# geometric mean, and a "pass" is a research-conjecture signal, NOT proven
# trust. Source: yuyay_v3 spec (Lutar, 2026).
YUYAY_AXES = (
    "moralGrounding",
    "measurabilityHonesty",
    "empiricalGrounding",
    "logicalConsistency",
    "sourceTransparency",
    "reproducibility",
    "licenseHygiene",
    "scopeDiscipline",
    "claimCalibration",
    "evalAwareness",
    "deceptionKeywords",
    "conflictingDirectives",
    "reversalDirective",
)
# Published per-axis advisory floors for the CONJUNCTIVE screen: two "sacred"
# axes at 0.95, seven "structural" at 0.90, four "introspection" at 0.90.
YUYAY_FLOORS = (
    0.95, 0.95,                          # sacred
    0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90,  # structural (7)
    0.90, 0.90, 0.90, 0.90,              # introspection (4)
)


def yuyay_weights(
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Canonical 13-axis Yuyay Λ weight vector (uniform 1/13), ADVISORY only.

    Returns a length-13 weight tensor for use as the ``weights`` argument to
    :func:`lambda_aggregate` / :func:`lambda_gate` over the 13 :data:`YUYAY_AXES`.
    Uniform by default (the Egyptian-exact diagonal). The published yuyay_v3
    gate is a conjunctive AND with per-axis floors (:data:`YUYAY_FLOORS`); the
    Λ roll-up here is the weighted geometric mean and is ADVISORY — NOT proven
    trust (Λ-uniqueness is Conjecture 1, open).
    """
    k = len(YUYAY_AXES)
    return torch.full((k,), 1.0 / k, dtype=dtype, device=device)


# ---- Kernel self-check surface --------------------------------------------- #
def selfcheck(
    k: int = 5,
    trials: int = 64,
    seed: Optional[int] = 0,
) -> dict:
    """Run the A1–A4 empirical self-checks and report a verdict + version.

    Returns a dict:
        version          — kernel version string
        axioms           — {A1..A4: bool} empirical pass on sampled inputs
        all_axioms_hold  — bool, every sampled axiom check passed
        adversarial      — {trials, violation} from a random falsification search
                           (violation is None when no violation was found)
        advisory         — always True
        lambda_status    — Conjecture 1 (open) honesty string

    HONESTY: these are EMPIRICAL checks on sampled inputs, NOT a proof of
    Λ-uniqueness (Conjecture 1, open). A clean run is evidence, not proof.
    """
    x = torch.rand(k, dtype=torch.float64) * 0.9  # headroom for the A1 up-bump
    w = torch.rand(k, dtype=torch.float64) + 1e-3
    axioms = {
        "A1_IsMonotone": is_monotone(x, w),
        "A2_IsHomogeneous": is_homogeneous(x, float(torch.rand(1).item()), weights=w),
        "A3_IsEgyptianExact": is_egyptian_exact(float(torch.rand(1).item()), k=k, weights=w),
        "A4_IsBounded": is_bounded_by_max(x, w),
    }
    violation = find_axiom_violation(k=k, trials=trials, seed=seed)
    return {
        "version": __version__,
        "axioms": axioms,
        "all_axioms_hold": all(axioms.values()) and violation is None,
        "adversarial": {"trials": int(trials), "violation": violation},
        "advisory": True,
        "lambda_status": "Conjecture 1 (open) — uniqueness unproven; advisory only",
    }


# Kept in sync with the package __version__ (single source of truth lives in
# __init__; duplicated here so _lambda is importable/selfcheck-able standalone).
__version__ = "0.2.0"


# Namedtuple result type for the gate. Defined after functions so docstrings
# above can reference it; imported by __init__ and layers.
from collections import namedtuple  # noqa: E402

LambdaGateResult = namedtuple(
    "LambdaGateResult", ["score", "passed", "threshold", "advisory"]
)

# SPDX-License-Identifier: Apache-2.0
# © 2026 SZL Holdings · Stephen P. Lutar · ORCID 0009-0001-0110-4173
"""szl.lambda/v1 strict kernel entry: Λ and its gate on tensors, with no clamping or defaults.

This is the torch port of ``reference/szl_lambda_v1.py`` and follows
``spec/szl.lambda.v1.json``. An input outside the contract makes ``lambda_v1``
raise ``LambdaV1Error(code)`` and makes ``lambda_v1_gate`` return verdict BLOCK
with that code. Nothing is clamped, renormalised, defaulted or rounded before
the compare::

    lambda_v1(axes, weights)            -> 0-d float64 tensor, Λ in [0, 1]
    lambda_v1_gate(axes, weights, tau)  -> LambdaV1GateResult(verdict, code, score, tau, advisory)

The legacy ``lambda_aggregate`` / ``lambda_gate`` are unchanged. They clamp
x > 1 to 1, route NaN and ±Inf to 0 and renormalise weights, and
``lambda_gate`` falls back to a threshold of 0.5, which is now deprecated.
Use this module when a verdict must follow szl.lambda/v1.

Inputs
    ``axes`` and ``weights`` are 1-D tensors of a real dtype (floating or
    integer). bool and complex are not numbers. A Python list, None, or a 0-d
    or 2-D tensor is LAMBDA_TYPE_INVALID; build the tensor with
    ``torch.tensor(values, dtype=torch.float64)``. ``weights`` is moved to the
    device of ``axes``. Inputs are read and never modified.

    ``tau`` is a Python int or float (not bool), finite, with 0 < tau <= 1. It
    is required and has no default. The policy value is ``policy_tau`` in
    frontier/model_admit_contract.v1.json.

Checks run in phases, in the reference's order, so the reported code does not
depend on axis order::

    LAMBDA_TYPE_INVALID (container) > LAMBDA_EMPTY > LAMBDA_LENGTH_MISMATCH
    > LAMBDA_TYPE_INVALID (dtype) > LAMBDA_NONFINITE_AXIS
    > LAMBDA_AXIS_OUT_OF_RANGE > LAMBDA_NONFINITE_WEIGHT
    > LAMBDA_WEIGHT_NONPOSITIVE > LAMBDA_WEIGHT_SUM

The gate checks tau first (LAMBDA_TAU_INVALID). The element checks and the
weight sum (``math.fsum``) run on the float64 values exactly as the reference
runs them, so every error code agrees with the reference.

How the numbers are computed
    * Λ, which is ``lambda_v1`` and the gate's ``score``, comes from
      ``_lambda.lambda_aggregate`` in float64, so it stays a differentiable
      tensor. On validated input that function's clamp and NaN routing change
      nothing, and a zero axis gives exactly 0. Its renormalisation divides by
      a sum within 1e-12 of 1, which moves Λ by at most Λ·|log Λ|·1e-12
      (<= 3.7e-13) plus float64 rounding. The vectors' value_tol is 1e-12.
    * The verdict comes from log Λ = fsum(w_k · log x_k) over the same float64
      values, as in the reference, so it never depends on how Λ rounds (a
      subnormal Λ has little relative precision). A zero axis gives
      log Λ = -inf and the verdict NO_GO / ZERO_VETO, a veto rather than an
      error. |log Λ - log tau| <= TIE_EPS gives ABSTAIN / NUMERIC_TIE. Above
      the band the verdict is GO, and below it NO_GO / BELOW_TAU.

The checks depend on the data and the verdict is a Python string, so this
entry is not meant to run inside ``torch.compile``. The legacy compiled weight
path in ``_lambda._resolve_weights``, which clamps weights to ``finfo.tiny``,
is outside szl.lambda/v1.

Λ is advisory. Λ uniqueness is Conjecture 1 (open), and nothing here depends on it.
"""
from __future__ import annotations

import math
from collections import namedtuple
from typing import Any, List, Optional, Tuple

import torch

from ._lambda import lambda_aggregate

SCHEMA = "szl.lambda/v1"
UNIQUENESS = "CONJECTURE_1_NOT_USED"
WEIGHT_SUM_TOL = 1e-12
TIE_EPS = 1e-9

TYPE_INVALID = "LAMBDA_TYPE_INVALID"
EMPTY = "LAMBDA_EMPTY"
LENGTH_MISMATCH = "LAMBDA_LENGTH_MISMATCH"
NONFINITE_AXIS = "LAMBDA_NONFINITE_AXIS"
AXIS_OUT_OF_RANGE = "LAMBDA_AXIS_OUT_OF_RANGE"
NONFINITE_WEIGHT = "LAMBDA_NONFINITE_WEIGHT"
WEIGHT_NONPOSITIVE = "LAMBDA_WEIGHT_NONPOSITIVE"
WEIGHT_SUM = "LAMBDA_WEIGHT_SUM"
TAU_INVALID = "LAMBDA_TAU_INVALID"

#: Every error code, in precedence order (tau is checked first by the gate).
ERROR_CODES = (
    TYPE_INVALID,
    EMPTY,
    LENGTH_MISMATCH,
    NONFINITE_AXIS,
    AXIS_OUT_OF_RANGE,
    NONFINITE_WEIGHT,
    WEIGHT_NONPOSITIVE,
    WEIGHT_SUM,
    TAU_INVALID,
)

GO = "GO"
NO_GO = "NO_GO"
ABSTAIN = "ABSTAIN"
BLOCK = "BLOCK"
VERDICTS = (GO, NO_GO, ABSTAIN, BLOCK)

ZERO_VETO = "ZERO_VETO"
BELOW_TAU = "BELOW_TAU"
NUMERIC_TIE = "NUMERIC_TIE"

_INTEGER_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


class LambdaV1Error(ValueError):
    """An input outside the szl.lambda/v1 contract. ``code`` is one of ERROR_CODES."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}" if detail else code)


#: verdict in VERDICTS; code as in the spec's gate rules; score is Λ as a 0-d
#: float64 tensor (None on BLOCK); tau is the validated float (None when tau
#: itself is invalid); advisory is always True.
LambdaV1GateResult = namedtuple(
    "LambdaV1GateResult", ["verdict", "code", "score", "tau", "advisory"]
)


def _is_real(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_real_dtype(dtype: torch.dtype) -> bool:
    return dtype.is_floating_point or dtype in _INTEGER_DTYPES


def _validate(
    axes: Any, weights: Any
) -> Tuple[torch.Tensor, torch.Tensor, List[float], List[float]]:
    """Float64 tensors and their values, or LambdaV1Error in the reference's phase order."""
    for name, t in (("axes", axes), ("weights", weights)):
        if not isinstance(t, torch.Tensor):
            raise LambdaV1Error(TYPE_INVALID, f"{name} must be a 1-D torch.Tensor, got {type(t).__name__}")
        if t.dim() != 1:
            raise LambdaV1Error(TYPE_INVALID, f"{name} must be a 1-D tensor, got {t.dim()}-D")
    k, m = axes.shape[0], weights.shape[0]
    if k == 0 or m == 0:
        raise LambdaV1Error(EMPTY, f"len(axes)={k}, len(weights)={m}")
    if k != m:
        raise LambdaV1Error(LENGTH_MISMATCH, f"len(axes)={k} != len(weights)={m}")
    for name, t in (("axes", axes), ("weights", weights)):
        if not _is_real_dtype(t.dtype):
            raise LambdaV1Error(TYPE_INVALID, f"{name} has dtype {t.dtype}, not a real number type")
    x = axes.to(torch.float64)
    w = weights.to(device=x.device, dtype=torch.float64)
    xs, ws = x.tolist(), w.tolist()
    for i, v in enumerate(xs):
        if not math.isfinite(v):
            raise LambdaV1Error(NONFINITE_AXIS, f"axes[{i}]={v!r}")
    for i, v in enumerate(xs):
        if not 0.0 <= v <= 1.0:
            raise LambdaV1Error(AXIS_OUT_OF_RANGE, f"axes[{i}]={v!r} is outside [0, 1]")
    for i, v in enumerate(ws):
        if not math.isfinite(v):
            raise LambdaV1Error(NONFINITE_WEIGHT, f"weights[{i}]={v!r}")
    for i, v in enumerate(ws):
        if not v > 0.0:
            raise LambdaV1Error(WEIGHT_NONPOSITIVE, f"weights[{i}]={v!r} is not > 0")
    try:
        total = math.fsum(ws)
    except OverflowError:
        raise LambdaV1Error(WEIGHT_SUM, "sum of weights overflows a float") from None
    if not abs(total - 1.0) <= WEIGHT_SUM_TOL:
        raise LambdaV1Error(WEIGHT_SUM, f"fsum(weights)={total!r} is not within {WEIGHT_SUM_TOL} of 1")
    return x, w, xs, ws


def lambda_v1(axes: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Λ_w(axes) as a 0-d float64 tensor in [0, 1]; exactly 0 iff some axis is 0.

    Raises ``LambdaV1Error(code)`` for any input outside szl.lambda/v1.
    Differentiable with respect to ``axes``.
    """
    x, w, _, _ = _validate(axes, weights)
    return lambda_aggregate(x, w)


def _check_tau(tau: Any) -> float:
    if not _is_real(tau):
        raise LambdaV1Error(TAU_INVALID, f"tau is {type(tau).__name__}, not a real number")
    if isinstance(tau, float) and not math.isfinite(tau):
        raise LambdaV1Error(TAU_INVALID, f"tau={tau!r} is not finite")
    if not 0 < tau <= 1:
        raise LambdaV1Error(TAU_INVALID, f"tau={tau!r} is outside (0, 1]")
    return float(tau)


def lambda_v1_gate(axes: torch.Tensor, weights: torch.Tensor, tau: float) -> LambdaV1GateResult:
    """The szl.lambda/v1 gate. It never raises on bad input; it returns BLOCK with the code.

    tau is required. The compare is in log space on the unrounded value, and a
    tie within TIE_EPS is ABSTAIN. A verdict is ADVISORY.
    """
    t: Optional[float] = None
    try:
        t = _check_tau(tau)
        x, w, xs, ws = _validate(axes, weights)
    except LambdaV1Error as err:
        return LambdaV1GateResult(BLOCK, err.code, None, t, True)
    score = lambda_aggregate(x, w)
    if any(v == 0.0 for v in xs):
        return LambdaV1GateResult(NO_GO, ZERO_VETO, score, t, True)
    log_lam = math.fsum(wk * math.log(xk) for xk, wk in zip(xs, ws))
    delta = log_lam - math.log(t)
    if abs(delta) <= TIE_EPS:
        return LambdaV1GateResult(ABSTAIN, NUMERIC_TIE, score, t, True)
    if delta > 0:
        return LambdaV1GateResult(GO, None, score, t, True)
    return LambdaV1GateResult(NO_GO, BELOW_TAU, score, t, True)

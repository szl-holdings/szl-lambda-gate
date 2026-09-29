# SPDX-License-Identifier: Apache-2.0
# © 2026 SZL Holdings · Stephen P. Lutar · ORCID 0009-0001-0110-4173
"""szl.lambda/v1 stdlib reference: Λ and its gate, exactly as spec/szl.lambda.v1.json says.

    Λ_w(x) = ∏_k x_k^{w_k} = exp(fsum_k(w_k · log x_k)),   Λ = 0 if some x_k == 0

Contract (every violation raises ``LambdaV1Error(code)``; nothing is clamped,
renormalised, defaulted or rounded):

* ``axes`` and ``weights`` are lists or tuples of real numbers (``bool`` is not
  a number), of equal, non-zero length.
* every axis is finite and 0 <= x_k <= 1;
* every weight is finite and w_k > 0, and |fsum(w) - 1| <= 1e-12.

Checks run in phases, each over every element, so the code reported for an
input with several faults does not depend on axis order::

    LAMBDA_TYPE_INVALID (container) > LAMBDA_EMPTY > LAMBDA_LENGTH_MISMATCH
    > LAMBDA_TYPE_INVALID (element) > LAMBDA_NONFINITE_AXIS
    > LAMBDA_AXIS_OUT_OF_RANGE > LAMBDA_NONFINITE_WEIGHT
    > LAMBDA_WEIGHT_NONPOSITIVE > LAMBDA_WEIGHT_SUM

``gate_v1(axes, weights, tau)`` takes tau as a required argument (no implicit
default; the policy value lives in frontier/model_admit_contract.v1.json) and
returns ``(verdict, code)``:

* BLOCK with the error code if tau is invalid (checked first: real, finite,
  0 < tau <= 1) or Λ raises;
* NO_GO / ZERO_VETO if some axis is 0 (log Λ = -inf; a veto, not an error);
* ABSTAIN / NUMERIC_TIE if |log Λ - log tau| <= TIE_EPS (1e-9);
* GO / None if log Λ > log tau, otherwise NO_GO / BELOW_TAU.

The compare is in log space on the unrounded value. Λ is advisory. Λ
uniqueness is Conjecture 1 (open) and nothing here depends on it.

Stdlib only. Other implementations port this file; they must agree on every
error code and verdict, and on values within each vector's ``value_tol``.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
from typing import Any, Optional, Tuple

SCHEMA = "szl.lambda/v1"
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


class LambdaV1Error(ValueError):
    """An input outside the szl.lambda/v1 contract. ``code`` is one of ERROR_CODES."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}" if detail else code)


def _is_real(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_nonfinite(value: Any) -> bool:
    # ints are always finite; only floats can be NaN or ±Inf.
    return isinstance(value, float) and not math.isfinite(value)


def _validate(axes: Any, weights: Any) -> None:
    for name, seq in (("axes", axes), ("weights", weights)):
        if not isinstance(seq, (list, tuple)):
            raise LambdaV1Error(TYPE_INVALID, f"{name} must be a list or tuple, got {type(seq).__name__}")
    if len(axes) == 0 or len(weights) == 0:
        raise LambdaV1Error(EMPTY, f"len(axes)={len(axes)}, len(weights)={len(weights)}")
    if len(axes) != len(weights):
        raise LambdaV1Error(LENGTH_MISMATCH, f"len(axes)={len(axes)} != len(weights)={len(weights)}")
    for name, seq in (("axes", axes), ("weights", weights)):
        for i, value in enumerate(seq):
            if not _is_real(value):
                raise LambdaV1Error(TYPE_INVALID, f"{name}[{i}] is {type(value).__name__}, not a real number")
    for i, x in enumerate(axes):
        if _is_nonfinite(x):
            raise LambdaV1Error(NONFINITE_AXIS, f"axes[{i}]={x!r}")
    for i, x in enumerate(axes):
        if not 0 <= x <= 1:
            raise LambdaV1Error(AXIS_OUT_OF_RANGE, f"axes[{i}]={x!r} is outside [0, 1]")
    for i, w in enumerate(weights):
        if _is_nonfinite(w):
            raise LambdaV1Error(NONFINITE_WEIGHT, f"weights[{i}]={w!r}")
    for i, w in enumerate(weights):
        if not w > 0:
            raise LambdaV1Error(WEIGHT_NONPOSITIVE, f"weights[{i}]={w!r} is not > 0")
    try:
        total = math.fsum(weights)
    except OverflowError:
        raise LambdaV1Error(WEIGHT_SUM, "sum of weights overflows a float") from None
    if not abs(total - 1.0) <= WEIGHT_SUM_TOL:
        raise LambdaV1Error(WEIGHT_SUM, f"fsum(weights)={total!r} is not within {WEIGHT_SUM_TOL} of 1")


def log_lambda_v1(axes: Any, weights: Any) -> float:
    """log Λ_w(x) = fsum(w_k · log x_k); -inf if some axis is 0. Raises LambdaV1Error."""
    _validate(axes, weights)
    if any(x == 0 for x in axes):
        return -math.inf
    return math.fsum(float(w) * math.log(float(x)) for x, w in zip(axes, weights))


def lambda_v1(axes: Any, weights: Any) -> float:
    """Λ_w(x) in [0, 1]; exactly 0.0 iff some axis is 0. Raises LambdaV1Error."""
    log_lam = log_lambda_v1(axes, weights)
    if log_lam == -math.inf:
        return 0.0
    return math.exp(log_lam)


def _check_tau(tau: Any) -> float:
    if not _is_real(tau):
        raise LambdaV1Error(TAU_INVALID, f"tau is {type(tau).__name__}, not a real number")
    if _is_nonfinite(tau):
        raise LambdaV1Error(TAU_INVALID, f"tau={tau!r} is not finite")
    if not 0 < tau <= 1:
        raise LambdaV1Error(TAU_INVALID, f"tau={tau!r} is outside (0, 1]")
    return float(tau)


def gate_v1(axes: Any, weights: Any, tau: Any) -> Tuple[str, Optional[str]]:
    """(verdict, code) for Λ_w(axes) against tau. Never raises on bad input: it BLOCKs."""
    try:
        t = _check_tau(tau)
        log_lam = log_lambda_v1(axes, weights)
    except LambdaV1Error as err:
        return BLOCK, err.code
    if log_lam == -math.inf:
        return NO_GO, ZERO_VETO
    delta = log_lam - math.log(t)
    if abs(delta) <= TIE_EPS:
        return ABSTAIN, NUMERIC_TIE
    if delta > 0:
        return GO, None
    return NO_GO, BELOW_TAU


# ------------------------------------------------ canonical numbers and bytes --


def encode_f64(x: float) -> str:
    """'f64:' + the 16 lowercase hex digits of the IEEE-754 binary64 bits (big-endian)."""
    return "f64:" + struct.pack(">d", float(x)).hex()


def decode_f64(text: str) -> float:
    """Inverse of encode_f64. Rejects anything that is not exactly 'f64:' + 16 lowercase hex."""
    if not isinstance(text, str) or len(text) != 20 or not text.startswith("f64:"):
        raise ValueError(f"not an f64 literal: {text!r}")
    digits = text[4:]
    if any(c not in "0123456789abcdef" for c in digits):
        raise ValueError(f"not an f64 literal: {text!r}")
    return struct.unpack(">d", bytes.fromhex(digits))[0]


def canonical_json_bytes(obj: Any) -> bytes:
    """Canonical JSON: sorted keys, compact separators, UTF-8, no NaN/Infinity."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def canonical_sha256(obj: Any) -> str:
    """SHA-256 hex over canonical_json_bytes(obj): stable across CRLF/LF checkouts."""
    return hashlib.sha256(canonical_json_bytes(obj)).hexdigest()

# SPDX-License-Identifier: Apache-2.0
# © 2026 Lutar, Stephen P. — SZL Holdings · ORCID 0009-0001-0110-4173
# CANONICAL SOURCE pulled from szl-holdings/platform packages/puriq-os/puriq_os/lambda_aggregator.py
"""
lambda_aggregator.py — Λ(x), the Lambda-Spine aggregator (canonical D2, v11 §12).

Λ(x) = ∏ xᵢ^{wᵢ},  Σwᵢ = 1,  wᵢ > 0,  xᵢ ∈ [0,1]   (weighted geometric mean)

Properties carried (Lutar/Axioms.lean):
  A1 IsMonotone · A2 IsHomogeneous (degree 1) · A3 IsEgyptianExact (Λ(c,…,c)=c) · A4 IsBounded (Λ ≤ max xᵢ)
Λ-uniqueness remains Conjecture 1 (open CAUCHY_ND sorry + missing symmetry axiom).
"""
from __future__ import annotations

import math
from typing import Sequence


def lambda_aggregate(axes: Sequence[float], weights: Sequence[float] | None = None) -> float:
    """Weighted geometric mean over axis scores in [0,1]. Uniform weights by default
    (the Egyptian-exact diagonal). Returns Λ(x) ∈ [0,1]."""
    n = len(axes)
    if n == 0:
        return 0.0
    if weights is None:
        weights = [1.0 / n] * n
    if len(weights) != n:
        raise ValueError("axes and weights length mismatch")
    sw = sum(weights)
    if sw <= 0:
        raise ValueError("weights must be positive and sum > 0")
    weights = [w / sw for w in weights]  # normalize Σw=1
    acc = 0.0
    for x, w in zip(axes, weights):
        x = min(max(float(x), 0.0), 1.0)
        if x <= 0.0:
            return 0.0  # any zero axis zeroes the product (A4-consistent)
        acc += w * math.log(x)
    val = math.exp(acc)
    return min(max(val, 0.0), 1.0)


def is_bounded_by_max(axes: Sequence[float], weights: Sequence[float] | None = None) -> bool:
    """A4 IsBounded check: Λ(x) ≤ max_i x_i (used by the loop's self-test)."""
    if not axes:
        return True
    return lambda_aggregate(axes, weights) <= max(axes) + 1e-12

# SPDX-License-Identifier: Apache-2.0
# © 2026 SZL Holdings · Stephen P. Lutar · ORCID 0009-0001-0110-4173
"""Hub-compliant kernel layer for the szl-lambda-gate kernel.

Per the Kernel Hub `kernel-requirements`, layers exposed for extension must be
PURE torch.nn.Module subclasses:
  - no custom __init__,
  - no class variables,
  - only a `forward` method.

The layer therefore reads its parameters (weights / threshold) off the module
instance it is bound to (set by the host model) and only defines `forward`.

HONESTY: `LambdaGate` emits an ADVISORY governance signal (the weighted
geometric mean Λ plus a pass/fail vs threshold). Λ is NOT proven trust; its
uniqueness is Conjecture 1 (open).
"""
import torch
from torch import nn

from ._lambda import lambda_aggregate, lambda_gate


class LambdaGate(nn.Module):
    """Pure Λ-gate layer.

    Reads optional ``self.weights`` (1-D, length k) and ``self.threshold``
    (float, default 0.5) off the bound module instance.

    forward(axes) -> LambdaGateResult(score, passed, threshold, advisory) where
    ``score`` = Λ(axes) over the last dim and ``passed`` = score >= threshold.
    Differentiable in ``score`` w.r.t. ``axes``.
    """

    def forward(self, axes: torch.Tensor):
        weights = getattr(self, "weights", None)
        threshold = getattr(self, "threshold", 0.5)
        return lambda_gate(axes, weights=weights, threshold=float(threshold))


class LambdaAggregate(nn.Module):
    """Pure Λ-aggregator layer: forward(axes) -> Λ(axes) tensor in [0,1].

    Reads optional ``self.weights`` (1-D, length k) off the bound module
    instance; uniform weights when absent. Returns just the score (no gate),
    fully differentiable w.r.t. ``axes``.
    """

    def forward(self, axes: torch.Tensor) -> torch.Tensor:
        weights = getattr(self, "weights", None)
        return lambda_aggregate(axes, weights=weights)

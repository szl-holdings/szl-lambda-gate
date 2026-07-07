# SPDX-License-Identifier: Apache-2.0
# © 2026 SZL Holdings · Stephen P. Lutar · ORCID 0009-0001-0110-4173
"""Hub-compliant kernel layers.

Per the Kernel Hub `kernel-requirements`, layers exposed for extension must
be PURE torch.nn.Module subclasses:
  - no custom __init__,
  - no class variables,
  - only a `forward` method,
  - forward signature compatible with the module it extends.

These layers therefore read their parameters (weight/bias/eps) off the
module instance they are bound to (set by the host model), and only define
`forward`. They are drop-in replacements for an existing RMSNorm/LayerNorm
module via the `kernels` layer-mapping mechanism.
"""
import torch
from torch import nn

from ._norm import fused_add_rms_norm, layer_norm, rms_norm


class RMSNorm(nn.Module):
    """Pure RMSNorm layer. Expects the host module to provide `self.weight`
    (optional) and `self.variance_epsilon` or `self.eps`."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        weight = getattr(self, "weight", None)
        eps = getattr(self, "variance_epsilon", None)
        if eps is None:
            eps = getattr(self, "eps", 1e-6)
        return rms_norm(hidden_states, weight=weight, eps=float(eps))


class LayerNorm(nn.Module):
    """Pure LayerNorm layer. Expects the host module to provide `self.weight`
    (optional), `self.bias` (optional), and `self.eps`."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        weight = getattr(self, "weight", None)
        bias = getattr(self, "bias", None)
        eps = getattr(self, "eps", 1e-5)
        return layer_norm(hidden_states, weight=weight, bias=bias, eps=float(eps))


class FusedAddRMSNorm(nn.Module):
    """Pure residual-add + RMSNorm layer for pre-norm transformer blocks.

    Expects the host module to provide `self.weight` (optional) and
    `self.variance_epsilon` or `self.eps`. Returns `(normalized, new_residual)`
    where `new_residual = hidden_states + residual` is carried forward as the
    next block's residual stream.
    """

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor):
        weight = getattr(self, "weight", None)
        eps = getattr(self, "variance_epsilon", None)
        if eps is None:
            eps = getattr(self, "eps", 1e-6)
        return fused_add_rms_norm(hidden_states, residual, weight=weight, eps=float(eps))

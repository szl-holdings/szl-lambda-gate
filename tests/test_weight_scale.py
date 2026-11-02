"""Finite weight scale must not change the advisory aggregate or gate."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "torch-ext"))
from szl_lambda_gate._lambda import lambda_aggregate, lambda_gate


@pytest.mark.parametrize("dtype,scale", [(torch.float32, 2e38), (torch.float64, 1e308)])
@pytest.mark.parametrize("compiled", [False, True])
def test_overflowing_weight_sum_does_not_turn_low_scores_into_pass(dtype, scale, compiled):
    gate = torch.compile(lambda_gate, backend="eager", fullgraph=True) if compiled else lambda_gate
    axes = torch.tensor([0.01, 0.04], dtype=dtype)
    weights = torch.tensor([scale, scale], dtype=dtype)
    assert torch.isfinite(weights).all()
    result = gate(axes, weights, threshold=0.8)
    assert result.score.item() == pytest.approx(0.02, rel=1e-6)
    assert not result.passed.item()
    assert result.advisory is True


@pytest.mark.parametrize("dtype,scale", [(torch.float32, 1e38), (torch.float64, 5e307)])
@pytest.mark.parametrize("compiled", [False, True])
def test_weight_rescaling_preserves_batched_scores(dtype, scale, compiled):
    aggregate = torch.compile(lambda_aggregate, backend="eager", fullgraph=True) if compiled else lambda_aggregate
    axes = torch.tensor([[0.04, 0.81, 0.25], [0.2, 0.3, 0.9]], dtype=dtype)
    weights = torch.tensor([1.0, 2.0, 3.0], dtype=dtype)
    expected = aggregate(axes, weights)
    actual = aggregate(axes, weights * scale)
    torch.testing.assert_close(actual, expected)
    assert torch.all(actual <= axes.max(dim=-1).values)


def test_large_weight_aggregate_retains_finite_gradients():
    axes = torch.tensor([0.01, 0.04], dtype=torch.float64, requires_grad=True)
    weights = torch.tensor([1e308, 1e308], dtype=torch.float64, requires_grad=True)
    score = lambda_aggregate(axes, weights)
    score.backward()
    assert torch.isfinite(axes.grad).all()
    assert torch.isfinite(weights.grad).all()
    assert score.item() == pytest.approx(0.02)

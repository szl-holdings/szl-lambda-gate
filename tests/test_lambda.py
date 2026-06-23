# SPDX-License-Identifier: Apache-2.0
# © 2026 SZL Holdings · Stephen P. Lutar · ORCID 0009-0001-0110-4173
"""Tests for the szl-lambda-gate universal kernel.

Covers:
  * correctness of the torch Λ aggregator vs the pure-Python reference
    (lambda_aggregator_source.py) across random inputs and weights,
  * the four carried axioms A1..A4 as property tests,
  * an autograd finite-difference gradient test,
  * batched-input behaviour,
  * the ADVISORY gate surface and honesty metadata.

Run: python -m pytest tests/ -q   (CPU-only; runs anywhere.)
"""
import math
import random
import sys
from pathlib import Path

import pytest
import torch

# Import the BUILT universal kernel package directly (the artifact get_kernel
# would load), exactly like the sibling kernel's tests do.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "build" / "torch-universal"))
import szl_lambda_gate as lg  # noqa: E402

# Import the canonical pure-Python reference for cross-checks.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lambda_aggregator_source import lambda_aggregate as ref_lambda_aggregate  # noqa: E402

torch.manual_seed(0)
random.seed(0)


# --------------------------------------------------------------------------- #
# Correctness vs the pure-Python reference                                    #
# --------------------------------------------------------------------------- #
def test_matches_reference_uniform_weights():
    """Λ(torch) == Λ(reference) for random inputs, uniform weights."""
    for _ in range(200):
        k = random.randint(1, 8)
        vals = [random.random() for _ in range(k)]   # in [0,1)
        ref = ref_lambda_aggregate(vals)
        got = lg.lambda_aggregate(torch.tensor(vals, dtype=torch.float64)).item()
        assert math.isclose(got, ref, rel_tol=1e-9, abs_tol=1e-9), (vals, got, ref)


def test_matches_reference_custom_weights():
    """Λ(torch) == Λ(reference) for random inputs AND random positive weights."""
    for _ in range(200):
        k = random.randint(1, 8)
        vals = [random.random() for _ in range(k)]
        ws = [random.random() + 1e-3 for _ in range(k)]
        ref = ref_lambda_aggregate(vals, ws)
        got = lg.lambda_aggregate(
            torch.tensor(vals, dtype=torch.float64),
            torch.tensor(ws, dtype=torch.float64),
        ).item()
        assert math.isclose(got, ref, rel_tol=1e-9, abs_tol=1e-9), (vals, ws, got, ref)


def test_zero_axis_zeroes_product():
    """Any zero axis drives Λ to exactly 0 (A4-consistent, matches reference)."""
    vals = [0.9, 0.0, 0.7]
    assert ref_lambda_aggregate(vals) == 0.0
    assert lg.lambda_aggregate(torch.tensor(vals, dtype=torch.float64)).item() == 0.0


def test_clamps_out_of_range():
    """Inputs outside [0,1] are clamped, matching the reference."""
    vals = [1.5, 0.8, -0.2]  # ref clamps to [1.0, 0.8, 0.0] -> 0 (a zero axis)
    ref = ref_lambda_aggregate(vals)
    got = lg.lambda_aggregate(torch.tensor(vals, dtype=torch.float64)).item()
    assert math.isclose(got, ref, abs_tol=1e-12)
    # And a no-zero clamp case.
    vals2 = [1.5, 0.8, 0.5]
    ref2 = ref_lambda_aggregate(vals2)
    got2 = lg.lambda_aggregate(torch.tensor(vals2, dtype=torch.float64)).item()
    assert math.isclose(got2, ref2, rel_tol=1e-9, abs_tol=1e-9)


# --------------------------------------------------------------------------- #
# Axiom property tests  A1..A4                                                #
# --------------------------------------------------------------------------- #
def test_A3_egyptian_exact():
    """A3: Λ(c,…,c) = c for many c and k (via the kernel self-check + direct)."""
    for _ in range(50):
        c = random.random()
        k = random.randint(1, 7)
        assert lg.is_egyptian_exact(c, k=k)
        val = lg.lambda_aggregate(torch.full((k,), c, dtype=torch.float64)).item()
        assert math.isclose(val, c, rel_tol=1e-9, abs_tol=1e-9)


def test_A4_bounded_by_max():
    """A4: Λ(x) ≤ max_i x_i for random inputs and weights."""
    for _ in range(200):
        k = random.randint(1, 8)
        x = torch.rand(k, dtype=torch.float64)
        ws = torch.rand(k, dtype=torch.float64) + 1e-3
        assert lg.is_bounded_by_max(x, ws)
        assert lg.lambda_aggregate(x, ws).item() <= x.max().item() + 1e-9


def test_A2_homogeneous_degree1():
    """A2: Λ(t·x) = t·Λ(x) for t in [0,1] (homogeneity of degree 1)."""
    for _ in range(100):
        k = random.randint(1, 6)
        x = torch.rand(k, dtype=torch.float64)
        t = random.random()
        assert lg.is_homogeneous(x, t)
        lhs = lg.lambda_aggregate(x * t).item()
        rhs = t * lg.lambda_aggregate(x).item()
        assert math.isclose(lhs, rhs, rel_tol=1e-7, abs_tol=1e-9)


def test_A1_monotone():
    """A1: Λ is non-decreasing in each axis (checked on random data)."""
    for _ in range(100):
        k = random.randint(1, 6)
        x = torch.rand(k, dtype=torch.float64) * 0.9  # leave room to bump up
        ws = torch.rand(k, dtype=torch.float64) + 1e-3
        assert lg.is_monotone(x, ws)


def test_A1_monotone_direct_pairwise():
    """A1 directly: raising one axis never lowers Λ."""
    x = torch.tensor([0.4, 0.6, 0.3], dtype=torch.float64)
    base = lg.lambda_aggregate(x).item()
    for j in range(x.shape[-1]):
        up = x.clone()
        up[j] = min(up[j].item() + 0.2, 1.0)
        assert lg.lambda_aggregate(up).item() >= base - 1e-12


# --------------------------------------------------------------------------- #
# Autograd                                                                     #
# --------------------------------------------------------------------------- #
def test_autograd_finite_difference():
    """Analytic gradient of Λ matches a finite-difference estimate.

    For the weighted geometric mean, ∂Λ/∂xᵢ = wᵢ · Λ / xᵢ (interior point).
    We compare autograd to a central finite difference.
    """
    x = torch.tensor([0.5, 0.7, 0.3], dtype=torch.float64, requires_grad=True)
    ws = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    y = lg.lambda_aggregate(x, ws)
    y.backward()
    g = x.grad.clone()

    eps = 1e-6
    for i in range(x.shape[-1]):
        xp = x.detach().clone(); xp[i] += eps
        xm = x.detach().clone(); xm[i] -= eps
        fd = (lg.lambda_aggregate(xp, ws).item() - lg.lambda_aggregate(xm, ws).item()) / (2 * eps)
        assert math.isclose(g[i].item(), fd, rel_tol=1e-5, abs_tol=1e-7), (i, g[i].item(), fd)


def test_autograd_gradcheck():
    """torch.autograd.gradcheck on Λ (double precision, interior point)."""
    x = torch.rand(4, 3, dtype=torch.float64).clamp(0.1, 0.95).requires_grad_(True)
    ws = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda z: lg.lambda_aggregate(z, ws), (x,), eps=1e-6, atol=1e-5)


def test_grad_is_finite():
    """Gradients stay finite even when some axes are zero (zero-routing)."""
    x = torch.tensor([0.0, 0.8, 0.5], dtype=torch.float64, requires_grad=True)
    y = lg.lambda_aggregate(x)
    y.backward()
    assert torch.all(torch.isfinite(x.grad))


# --------------------------------------------------------------------------- #
# Batched input                                                                #
# --------------------------------------------------------------------------- #
def test_batched_matches_rowwise():
    """Batched Λ over (..., k) equals per-row scalar Λ from the reference."""
    B, k = 16, 5
    X = torch.rand(B, k, dtype=torch.float64)
    ws = torch.rand(k, dtype=torch.float64) + 1e-3
    batched = lg.lambda_aggregate(X, ws)              # shape (B,)
    assert batched.shape == (B,)
    for b in range(B):
        ref = ref_lambda_aggregate(X[b].tolist(), ws.tolist())
        assert math.isclose(batched[b].item(), ref, rel_tol=1e-9, abs_tol=1e-9)


def test_batched_multidim():
    """Λ reduces only the last dim for higher-rank batches."""
    X = torch.rand(3, 4, 6, dtype=torch.float64)
    out = lg.lambda_aggregate(X)
    assert out.shape == (3, 4)
    # spot-check one element against the reference
    ref = ref_lambda_aggregate(X[1, 2].tolist())
    assert math.isclose(out[1, 2].item(), ref, rel_tol=1e-9, abs_tol=1e-9)


# --------------------------------------------------------------------------- #
# Gate surface + honesty                                                       #
# --------------------------------------------------------------------------- #
def test_gate_pass_fail():
    """lambda_gate returns score + boolean pass/fail vs threshold, labeled advisory."""
    axes = torch.tensor([0.9, 0.8, 0.95], dtype=torch.float64)
    res = lg.lambda_gate(axes, threshold=0.5)
    assert math.isclose(res.score.item(), lg.lambda_aggregate(axes).item())
    assert bool(res.passed) is True
    assert res.threshold == 0.5
    assert res.advisory is True

    res2 = lg.lambda_gate(axes, threshold=0.99)
    assert bool(res2.passed) is False


def test_gate_batched():
    """Gate works batched: score and passed are per-row."""
    X = torch.tensor([[0.9, 0.9, 0.9], [0.1, 0.9, 0.9]], dtype=torch.float64)
    res = lg.lambda_gate(X, threshold=0.5)
    assert res.score.shape == (2,)
    assert res.passed.shape == (2,)
    assert bool(res.passed[0]) is True
    assert bool(res.passed[1]) is False  # low first axis drags Λ down


def test_gate_rejects_nonfinite_threshold():
    with pytest.raises(ValueError):
        lg.lambda_gate(torch.tensor([0.5, 0.5]), threshold=float("inf"))


def test_layer_lambda_gate():
    """Pure nn.Module LambdaGate layer reads weights/threshold off the instance."""
    layer = lg.layers.LambdaGate()
    axes = torch.tensor([0.8, 0.8, 0.8], dtype=torch.float64)
    res = layer(axes)
    assert math.isclose(res.score.item(), 0.8, rel_tol=1e-9, abs_tol=1e-9)
    assert bool(res.passed) is True
    # bind a threshold + weights onto the instance (host-model style)
    layer.threshold = 0.9
    layer.weights = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
    res2 = layer(axes)
    assert bool(res2.passed) is False  # 0.8 < 0.9


def test_layer_lambda_aggregate_module():
    layer = lg.layers.LambdaAggregate()
    axes = torch.tensor([0.5, 0.5], dtype=torch.float64)
    assert math.isclose(layer(axes).item(), 0.5, rel_tol=1e-9, abs_tol=1e-9)


def test_provenance_and_honesty_metadata():
    """Provenance is baked in and Λ is labeled advisory / Conjecture 1 — NOT proven."""
    assert lg.PROVENANCE["lean_repo"] == "szl-holdings/lutar-lean"
    assert lg.PROVENANCE["lean_declarations"] == 749
    assert lg.PROVENANCE["lean_axioms"] == 14
    assert lg.PROVENANCE["lean_tracked_sorries"] == 163
    assert lg.PROVENANCE["doi_lutar_lean"] == "10.5281/zenodo.20434308"
    assert "Conjecture 1" in lg.PROVENANCE["lambda_status"]
    assert "ADVISORY" in lg.DOCTRINE_FOOTER
    assert "NOT proven trust" in lg.DOCTRINE_FOOTER


def test_torch_compile_friendly():
    """Λ traces/compiles without graph breaks blowing up (smoke test)."""
    fn = torch.compile(lambda z: lg.lambda_aggregate(z), fullgraph=False)
    x = torch.rand(8, 4, dtype=torch.float32)
    out = fn(x)
    ref = lg.lambda_aggregate(x)
    assert torch.allclose(out, ref, atol=1e-6)


# =========================================================================== #
# Upgrade2 (Dev 2): stress-test fixes, adversarial axioms, new surface.       #
# =========================================================================== #

# --------------------------------------------------------------------------- #
# REGRESSION: BUG1 — a NaN axis used to make Λ = NaN and the gradient          #
# non-finite ([0, nan, nan]). It must now be a conservative FAILING axis:      #
# score == 0, gradient finite, gate does NOT pass.                             #
# --------------------------------------------------------------------------- #
def test_regression_nan_axis_is_zero_not_nan():
    vals = torch.tensor([float("nan"), 0.8, 0.5], dtype=torch.float64)
    out = lg.lambda_aggregate(vals)
    assert torch.isfinite(out).all()
    assert out.item() == 0.0


def test_regression_nan_axis_gradient_finite():
    x = torch.tensor([float("nan"), 0.8, 0.5], dtype=torch.float64, requires_grad=True)
    y = lg.lambda_aggregate(x)
    y.backward()
    assert torch.all(torch.isfinite(x.grad))


def test_regression_nan_axis_gate_does_not_pass():
    res = lg.lambda_gate(torch.tensor([float("nan"), 0.9, 0.9], dtype=torch.float64), threshold=0.5)
    assert torch.isfinite(res.score).all()
    assert res.score.item() == 0.0
    assert bool(res.passed) is False


# --------------------------------------------------------------------------- #
# REGRESSION: BUG2 — a +Inf axis used to clamp to 1.0 and silently count as a  #
# PERFECT axis, letting the gate PASS. A garbage/invalid axis must never make  #
# a conservative non-compensatory gate pass: +Inf (and -Inf) now zero Λ.        #
# --------------------------------------------------------------------------- #
def test_regression_posinf_axis_zeroes_and_fails():
    vals = torch.tensor([float("inf"), 0.8, 0.5], dtype=torch.float64)
    assert lg.lambda_aggregate(vals).item() == 0.0
    res = lg.lambda_gate(vals, threshold=0.5)
    assert bool(res.passed) is False


def test_regression_neginf_axis_zeroes():
    vals = torch.tensor([float("-inf"), 0.8, 0.5], dtype=torch.float64)
    assert lg.lambda_aggregate(vals).item() == 0.0


def test_regression_posinf_axis_gradient_finite():
    x = torch.tensor([float("inf"), 0.8, 0.5], dtype=torch.float64, requires_grad=True)
    y = lg.lambda_aggregate(x)
    y.backward()
    assert torch.all(torch.isfinite(x.grad))


def test_finite_out_of_range_still_clamps():
    """Guard: the BUG2 fix must NOT change finite out-of-range clamping."""
    # 1.5 is finite -> clamps to 1.0 (not a failing axis).
    got = lg.lambda_aggregate(torch.tensor([1.5, 0.8, 0.5], dtype=torch.float64)).item()
    ref = ref_lambda_aggregate([1.5, 0.8, 0.5])
    assert math.isclose(got, ref, rel_tol=1e-9, abs_tol=1e-9)
    # all-large finite -> all clamp to 1 -> Λ == 1.
    assert lg.lambda_aggregate(torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64)).item() == 1.0


def test_batch_with_one_nan_row_isolated():
    """A NaN in one batch row must not poison the other rows' scores."""
    X = torch.tensor([[0.9, 0.9, 0.9], [float("nan"), 0.9, 0.9]], dtype=torch.float64)
    out = lg.lambda_aggregate(X)
    assert torch.isfinite(out).all()
    assert math.isclose(out[0].item(), 0.9, rel_tol=1e-9, abs_tol=1e-9)
    assert out[1].item() == 0.0


# --------------------------------------------------------------------------- #
# Weights edge cases (auto-normalize, reject zero/neg/non-finite)             #
# --------------------------------------------------------------------------- #
def test_weights_auto_normalize():
    """Non-normalized positive weights are auto-normalized (Σw=1)."""
    x = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float64)
    raw = torch.tensor([2.0, 3.0, 5.0], dtype=torch.float64)
    a = lg.lambda_aggregate(x, raw).item()
    b = lg.lambda_aggregate(x, raw / raw.sum()).item()
    assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)


def test_weights_reject_zero_negative():
    x = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float64)
    for ws in ([0.0, 0.5, 0.5], [-0.1, 0.6, 0.5], [0.0, 0.0, 0.0]):
        with pytest.raises(ValueError):
            lg.lambda_aggregate(x, torch.tensor(ws, dtype=torch.float64))


def test_weights_reject_nonfinite():
    x = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float64)
    for ws in ([float("nan"), 0.5, 0.5], [float("inf"), 0.5, 0.5]):
        with pytest.raises(ValueError):
            lg.lambda_aggregate(x, torch.tensor(ws, dtype=torch.float64))


# --------------------------------------------------------------------------- #
# Shape edge cases: single axis, many axes (1000+), 3D/4D batches             #
# --------------------------------------------------------------------------- #
def test_single_axis():
    assert math.isclose(lg.lambda_aggregate(torch.tensor([0.42], dtype=torch.float64)).item(), 0.42,
                        rel_tol=1e-12, abs_tol=1e-12)


def test_many_axes_1000plus():
    big = torch.rand(2000, dtype=torch.float64).clamp(0.01, 1.0)
    out = lg.lambda_aggregate(big)
    assert torch.isfinite(out).all()
    ref = ref_lambda_aggregate(big.tolist())
    assert math.isclose(out.item(), ref, rel_tol=1e-7, abs_tol=1e-9)


def test_batched_3d_4d_shapes():
    assert lg.lambda_aggregate(torch.rand(2, 3, 4, dtype=torch.float64)).shape == (2, 3)
    assert lg.lambda_aggregate(torch.rand(2, 3, 4, 5, dtype=torch.float64)).shape == (2, 3, 4)


# --------------------------------------------------------------------------- #
# dtype coverage fp16/bf16/fp32/fp64                                          #
# --------------------------------------------------------------------------- #
def test_all_dtypes_finite_and_close():
    vals = [0.9, 0.8, 0.7]
    ref = ref_lambda_aggregate(vals)
    tols = {torch.float16: 2e-2, torch.bfloat16: 3e-2, torch.float32: 1e-5, torch.float64: 1e-9}
    for dt, tol in tols.items():
        out = lg.lambda_aggregate(torch.tensor(vals, dtype=dt))
        assert out.dtype == dt
        assert torch.isfinite(out).all()
        assert abs(out.item() - ref) <= tol


# --------------------------------------------------------------------------- #
# torch.compile FULLGRAPH on lambda_aggregate AND lambda_gate (+ batch)        #
# --------------------------------------------------------------------------- #
def test_fullgraph_lambda_aggregate():
    fn = torch.compile(lambda z: lg.lambda_aggregate(z), fullgraph=True)
    x = torch.rand(8, 4, dtype=torch.float32)
    assert torch.allclose(fn(x), lg.lambda_aggregate(x), atol=1e-6)


def test_fullgraph_lambda_gate_score():
    fn = torch.compile(lambda z: lg.lambda_gate(z, threshold=0.5).score, fullgraph=True)
    x = torch.rand(8, 4, dtype=torch.float32)
    assert torch.allclose(fn(x), lg.lambda_aggregate(x), atol=1e-6)


def test_fullgraph_lambda_gate_batch_score():
    fn = torch.compile(lambda z: lg.lambda_gate_batch(z, threshold=0.5).score, fullgraph=True)
    C = torch.rand(5, 3, 4, dtype=torch.float32)
    out = fn(C)
    assert out.shape == (5, 3)
    assert torch.allclose(out, lg.lambda_aggregate(C), atol=1e-6)


# --------------------------------------------------------------------------- #
# ADVERSARIAL axiom property tests: random-search for an A1–A4 violation;      #
# assert none is found within tolerance (honest falsification, NOT a proof).   #
# --------------------------------------------------------------------------- #
def test_adversarial_no_axiom_violation_uniform():
    for k in (1, 2, 3, 5, 8, 13):
        assert lg.find_axiom_violation(k=k, trials=300, seed=k) is None


def test_adversarial_no_axiom_violation_many_seeds():
    for seed in range(10):
        assert lg.find_axiom_violation(k=7, trials=200, seed=seed) is None


def test_adversarial_axioms_hold_batched_random():
    """A2/A4 hold on large random batches (extra adversarial coverage)."""
    for _ in range(50):
        k = random.randint(1, 12)
        X = torch.rand(32, k, dtype=torch.float64)
        ws = torch.rand(k, dtype=torch.float64) + 1e-3
        assert lg.is_bounded_by_max(X, ws)
        assert lg.is_homogeneous(X, random.random(), weights=ws)


# --------------------------------------------------------------------------- #
# lambda_gate_batch (the realistic per-inference-step agent call)             #
# --------------------------------------------------------------------------- #
def test_gate_batch_scores_many_candidates():
    C = torch.tensor([[0.9, 0.9, 0.9], [0.1, 0.9, 0.9], [0.6, 0.6, 0.6]], dtype=torch.float64)
    res = lg.lambda_gate_batch(C, threshold=0.5)
    assert res.score.shape == (3,)
    assert res.passed.shape == (3,)
    assert res.passed.tolist() == [True, False, True]
    assert res.advisory is True
    # each row equals the single-call gate on that row
    for i in range(3):
        assert math.isclose(res.score[i].item(), lg.lambda_aggregate(C[i]).item(), abs_tol=1e-12)


def test_gate_batch_higher_rank():
    C = torch.rand(4, 6, 3, dtype=torch.float64)  # (batch=4, N=6 candidates, k=3)
    res = lg.lambda_gate_batch(C, threshold=0.5)
    assert res.score.shape == (4, 6)
    assert res.passed.shape == (4, 6)


def test_gate_batch_requires_2d_min():
    with pytest.raises(ValueError):
        lg.lambda_gate_batch(torch.tensor([0.5, 0.5, 0.5], dtype=torch.float64))


# --------------------------------------------------------------------------- #
# selfcheck() — A1–A4 verification surface + version                           #
# --------------------------------------------------------------------------- #
def test_selfcheck_passes_and_reports_version():
    sc = lg.selfcheck()
    assert sc["version"] == lg.__version__
    assert set(sc["axioms"]) == {"A1_IsMonotone", "A2_IsHomogeneous", "A3_IsEgyptianExact", "A4_IsBounded"}
    assert all(sc["axioms"].values())
    assert sc["all_axioms_hold"] is True
    assert sc["adversarial"]["violation"] is None
    assert sc["advisory"] is True
    assert "Conjecture 1" in sc["lambda_status"]


# --------------------------------------------------------------------------- #
# Yuyay 13-axis preset (ADVISORY only)                                        #
# --------------------------------------------------------------------------- #
def test_yuyay_preset_shape_and_normalization():
    assert len(lg.YUYAY_AXES) == 13
    assert len(lg.YUYAY_FLOORS) == 13
    w = lg.yuyay_weights()
    assert w.numel() == 13
    assert math.isclose(float(w.sum()), 1.0, rel_tol=1e-9, abs_tol=1e-9)


def test_yuyay_floors_published_values():
    # Two sacred @ 0.95, the remaining eleven @ 0.90 (yuyay_v3 spec).
    assert lg.YUYAY_FLOORS[0] == 0.95 and lg.YUYAY_FLOORS[1] == 0.95
    assert all(f == 0.90 for f in lg.YUYAY_FLOORS[2:])
    assert lg.YUYAY_AXES[0] == "moralGrounding"
    assert lg.YUYAY_AXES[1] == "measurabilityHonesty"


def test_yuyay_weights_usable_in_gate():
    axes = torch.full((13,), 0.95, dtype=torch.float64)
    res = lg.lambda_gate(axes, weights=lg.yuyay_weights(), threshold=0.9)
    assert math.isclose(res.score.item(), 0.95, rel_tol=1e-9, abs_tol=1e-9)
    assert bool(res.passed) is True
    assert res.advisory is True  # advisory, NOT proven trust


def test_version_bumped_to_0_2_0():
    assert lg.__version__ == "0.2.0"

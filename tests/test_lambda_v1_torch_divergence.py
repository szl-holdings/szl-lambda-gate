# SPDX-License-Identifier: Apache-2.0
# © 2026 SZL Holdings · Stephen P. Lutar · ORCID 0009-0001-0110-4173
"""The torch kernel against the szl.lambda/v1 vectors: a pinned divergence table.

``torch-ext/szl_lambda_gate/_lambda.py::lambda_aggregate`` predates the v1
contract. It clamps x > 1 to 1, routes NaN and ±Inf to 0, renormalises
weights, and raises an uncoded ``ValueError`` where v1 raises a coded error.
Those behaviours (FUSION_BRIEF E5) are asserted here exactly, row by row, so a
silent change in either the kernel or the vectors fails this test. The slice
that aligns the kernel (FF-02) edits this table in the same change.

torch is this repo's one declared dependency and CI installs it, so it is
imported directly.
"""
from __future__ import annotations

import importlib.util
import json
import struct
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
VECTORS_PATH = ROOT / "spec" / "lambda_v1_vectors.json"
REFERENCE_PATH = ROOT / "reference" / "szl_lambda_v1.py"

for _candidate in (ROOT / "build" / "torch-universal", ROOT / "torch-ext"):
    if (_candidate / "szl_lambda_gate").is_dir():
        sys.path.insert(0, str(_candidate))
        break
else:
    raise ImportError("szl_lambda_gate missing from build/torch-universal and torch-ext")
from szl_lambda_gate import lambda_aggregate  # noqa: E402


def _load_reference():
    name = "szl_lambda_v1_reference"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REFERENCE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _decode_scalar(value):
    if isinstance(value, str) and value.startswith("f64:"):
        return struct.unpack(">d", bytes.fromhex(value[4:]))[0]
    return value


def _decode(value):
    return [_decode_scalar(v) for v in value] if isinstance(value, list) else _decode_scalar(value)


def _tensorable(values) -> bool:
    return isinstance(values, list) and all(
        isinstance(x, (int, float)) and not isinstance(x, bool) for x in values
    )


ref = _load_reference()
VECTORS = json.loads(VECTORS_PATH.read_text(encoding="utf-8"))["vectors"]

# vector id -> what torch lambda_aggregate does today (float64 inputs).
#   ("value", v): returns v, compared at 6 dp; the reference raises a coded error.
#   ("raises", ValueError): raises an uncoded ValueError; the reference raises a coded error.
TORCH_DIVERGENCE_TABLE = {
    # E5: clamps x > 1 to 1, so [1.5, 0.9] reads as [1, 0.9].
    "x_gt_1": ("value", 0.948683),
    "precedence_axis_before_weight": ("value", 0.948683),
    "tau_precedes_axis_error": ("value", 0.948683),
    # E5: a NaN or ±Inf axis, or a negative one, is routed to 0 rather than refused.
    "nan_axis": ("value", 0.0),
    "pos_inf_axis": ("value", 0.0),
    "neg_inf_axis": ("value", 0.0),
    "negative_axis": ("value", 0.0),
    "precedence_nonfinite_before_range_a": ("value", 0.0),
    "precedence_nonfinite_before_range_b": ("value", 0.0),
    # E5: weights are renormalised instead of refused.
    "w_unnormalised_2_2": ("value", 0.72),
    "weight_sum_outside_tol": ("value", 0.67082),
    # Refused, but without a v1 code.
    "empty": ("raises", ValueError),
    "empty_axes_nonempty_weights": ("raises", ValueError),
    "e5_negative_weight": ("raises", ValueError),
    "w_zero_weight": ("raises", ValueError),
    "e5_dropped_axis_renormalised": ("raises", ValueError),
    "length_mismatch_extra_weight": ("raises", ValueError),
    "weight_nan": ("raises", ValueError),
    "weight_pos_inf": ("raises", ValueError),
    "weight_neg_inf": ("raises", ValueError),
}


def _in_domain(vector) -> bool:
    return _tensorable(_decode(vector["axes"])) and _tensorable(_decode(vector["weights"]))


def _torch_lambda(vector) -> float:
    axes = torch.tensor(_decode(vector["axes"]), dtype=torch.float64)
    weights = torch.tensor(_decode(vector["weights"]), dtype=torch.float64)
    return float(lambda_aggregate(axes, weights).item())


def test_only_type_invalid_rows_have_no_tensor_form():
    excluded = {v["id"] for v in VECTORS if not _in_domain(v)}
    type_invalid = {v["id"] for v in VECTORS if v["expect"].get("error") == "LAMBDA_TYPE_INVALID"}
    assert excluded == type_invalid
    assert excluded, "the vectors must exercise the type contract"


def test_table_names_only_real_divergences():
    in_domain = {v["id"]: v for v in VECTORS if _in_domain(v)}
    assert set(TORCH_DIVERGENCE_TABLE) <= set(in_domain)
    reference_errors = {vid for vid, v in in_domain.items() if "error" in v["expect"]}
    assert set(TORCH_DIVERGENCE_TABLE) == reference_errors


def test_acceptance_rows_are_pinned_exactly():
    assert TORCH_DIVERGENCE_TABLE["x_gt_1"] == ("value", 0.948683)
    assert TORCH_DIVERGENCE_TABLE["nan_axis"] == ("value", 0.0)
    assert TORCH_DIVERGENCE_TABLE["pos_inf_axis"] == ("value", 0.0)
    assert TORCH_DIVERGENCE_TABLE["w_unnormalised_2_2"] == ("value", 0.72)
    assert TORCH_DIVERGENCE_TABLE["empty"] == ("raises", ValueError)
    assert TORCH_DIVERGENCE_TABLE["e5_negative_weight"] == ("raises", ValueError)


def test_renormalisation_of_w_2_2_equals_the_half_half_value():
    axes = torch.tensor([0.81, 0.64], dtype=torch.float64)
    unnormalised = lambda_aggregate(axes, torch.tensor([2.0, 2.0], dtype=torch.float64)).item()
    halves = lambda_aggregate(axes, torch.tensor([0.5, 0.5], dtype=torch.float64)).item()
    assert unnormalised == halves


@pytest.mark.parametrize(
    "vector", [v for v in VECTORS if _in_domain(v)], ids=[v["id"] for v in VECTORS if _in_domain(v)]
)
def test_torch_kernel_against_vector(vector):
    vid, expect = vector["id"], vector["expect"]
    axes, weights = _decode(vector["axes"]), _decode(vector["weights"])
    if vid in TORCH_DIVERGENCE_TABLE:
        kind, observed = TORCH_DIVERGENCE_TABLE[vid]
        with pytest.raises(ref.LambdaV1Error) as info:
            ref.lambda_v1(axes, weights)
        assert info.value.code == expect["error"]
        if kind == "value":
            assert round(_torch_lambda(vector), 6) == observed
        else:
            with pytest.raises(observed):
                _torch_lambda(vector)
        return
    expected = _decode(expect["value_f64"])
    assert abs(_torch_lambda(vector) - expected) <= vector["value_tol"]

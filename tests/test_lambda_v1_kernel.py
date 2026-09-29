# SPDX-License-Identifier: Apache-2.0
# © 2026 SZL Holdings · Stephen P. Lutar · ORCID 0009-0001-0110-4173
"""FF-02: the strict szl.lambda/v1 kernel entry, and the deprecated implicit 0.5 threshold.

Covers:
  * ``lambda_v1`` / ``lambda_v1_gate`` (torch-ext/szl_lambda_gate/_v1.py) against every
    golden vector and against the stdlib reference, error codes included; values within
    each vector's ``value_tol``;
  * the tensor-side type contract and the phase precedence;
  * a seeded differential run against the reference;
  * the ``None`` threshold sentinel: every entry point that defaulted to 0.5 warns once
    when the threshold is omitted and otherwise behaves exactly as before;
  * trace safety of that path under ``torch.compile(fullgraph=True)``;
  * ``selfcheck()`` still reports Conjecture 1 (open).

Vectors are mapped into the kernel's input form: a list of real numbers becomes a
float64 tensor. Anything else (None, or a list holding a bool, None or a string) is
passed through unchanged; the kernel reports LAMBDA_TYPE_INVALID for it, as the
reference does for the raw value.
"""
from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import math
import random
import re
import struct
import sys
import warnings
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "spec" / "szl.lambda.v1.json"
VECTORS_PATH = ROOT / "spec" / "lambda_v1_vectors.json"
REFERENCE_PATH = ROOT / "reference" / "szl_lambda_v1.py"
CONTRACT_PATH = ROOT / "frontier" / "model_admit_contract.v1.json"

for _candidate in (ROOT / "build" / "torch-universal", ROOT / "torch-ext"):
    if (_candidate / "szl_lambda_gate").is_dir():
        sys.path.insert(0, str(_candidate))
        break
else:
    raise ImportError("szl_lambda_gate missing from build/torch-universal and torch-ext")
import szl_lambda_gate as lg  # noqa: E402
from szl_lambda_gate import _lambda as lam  # noqa: E402
from szl_lambda_gate import _v1  # noqa: E402


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


def _is_real(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _kernel_form(value):
    """A list of real numbers -> float64 tensor; anything else unchanged."""
    if isinstance(value, list) and all(_is_real(x) for x in value):
        return torch.tensor(value, dtype=torch.float64)
    return value


def _t(values, dtype=torch.float64):
    return torch.tensor(values, dtype=dtype)


ref = _load_reference()
SPEC = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
VECTORS = json.loads(VECTORS_PATH.read_text(encoding="utf-8"))["vectors"]
POLICY_TAU = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))["policy_tau"]
SPEC_CODES = tuple(e["code"] for e in SPEC["error_codes"])


# ------------------------------------------------------ constants and surface --


def test_v1_constants_match_the_spec_and_the_reference():
    assert _v1.SCHEMA == SPEC["schema"] == ref.SCHEMA
    assert _v1.WEIGHT_SUM_TOL == SPEC["weight_sum_tol"] == ref.WEIGHT_SUM_TOL
    assert _v1.TIE_EPS == SPEC["gate"]["tie_eps"] == ref.TIE_EPS
    assert tuple(_v1.ERROR_CODES) == SPEC_CODES == tuple(ref.ERROR_CODES)
    assert tuple(_v1.VERDICTS) == tuple(SPEC["gate"]["verdicts"]) == tuple(ref.VERDICTS)
    assert (_v1.ZERO_VETO, _v1.BELOW_TAU, _v1.NUMERIC_TIE) == (ref.ZERO_VETO, ref.BELOW_TAU, ref.NUMERIC_TIE)
    assert _v1.UNIQUENESS == SPEC["uniqueness"] == "CONJECTURE_1_NOT_USED"


def test_v1_module_imports_only_torch_the_stdlib_and_the_kernel():
    tree = ast.parse(Path(_v1.__file__).read_text(encoding="utf-8"))
    absolute, relative = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            absolute.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                relative.add(node.module)
            else:
                absolute.add((node.module or "").split(".")[0])
    assert absolute <= {"__future__", "collections", "math", "typing", "torch"}, absolute
    assert relative <= {"_lambda"}, relative


def test_package_exports_the_strict_entry():
    for name in ("lambda_v1", "lambda_v1_gate", "LambdaV1Error", "LambdaV1GateResult"):
        assert name in lg.__all__, name
        assert getattr(lg, name) is getattr(_v1, name), name
    assert lg.LambdaV1GateResult._fields == ("verdict", "code", "score", "tau", "advisory")


def test_error_type_carries_its_code():
    assert issubclass(_v1.LambdaV1Error, ValueError)
    with pytest.raises(ValueError) as info:
        _v1.lambda_v1(_t([]), _t([]))
    assert info.value.code == "LAMBDA_EMPTY"
    assert "LAMBDA_EMPTY" in str(info.value)


def test_tau_is_a_required_argument_with_no_default():
    params = inspect.signature(_v1.lambda_v1_gate).parameters
    assert list(params) == ["axes", "weights", "tau"]
    assert all(p.default is inspect.Parameter.empty for p in params.values())
    with pytest.raises(TypeError):
        _v1.lambda_v1_gate(_t([0.9]), _t([1.0]))  # noqa: the missing tau is the point


# ----------------------------------------------------------------- vectors --


@pytest.mark.parametrize("vector", VECTORS, ids=[v["id"] for v in VECTORS])
def test_kernel_matches_vector_and_reference(vector):
    axes, weights, tau = _decode(vector["axes"]), _decode(vector["weights"]), _decode(vector["tau"])
    k_axes, k_weights = _kernel_form(axes), _kernel_form(weights)
    expect, tol = vector["expect"], vector["value_tol"]

    if "value_f64" in expect:
        got = _v1.lambda_v1(k_axes, k_weights)
        assert isinstance(got, torch.Tensor) and got.dtype == torch.float64 and got.dim() == 0
        assert abs(float(got) - _decode(expect["value_f64"])) <= tol
    else:
        with pytest.raises(_v1.LambdaV1Error) as info:
            _v1.lambda_v1(k_axes, k_weights)
        assert info.value.code == expect["error"]

    res = _v1.lambda_v1_gate(k_axes, k_weights, tau)
    assert isinstance(res, _v1.LambdaV1GateResult)
    assert (res.verdict, res.code) == (expect["verdict"], expect["code"])
    assert (res.verdict, res.code) == ref.gate_v1(axes, weights, tau)
    assert res.advisory is True
    if res.verdict == "BLOCK":
        assert res.score is None
    else:
        assert abs(float(res.score) - _decode(expect["value_f64"])) <= tol
        assert res.tau == float(tau)


def test_every_vector_code_and_verdict_is_reached_through_the_kernel():
    codes, verdicts = set(), set()
    for v in VECTORS:
        res = _v1.lambda_v1_gate(_kernel_form(_decode(v["axes"])), _kernel_form(_decode(v["weights"])), _decode(v["tau"]))
        verdicts.add(res.verdict)
        if res.verdict == "BLOCK":
            codes.add(res.code)
    assert len(VECTORS) == SPEC["vectors"]["count"] == 60
    assert codes == set(SPEC_CODES)
    assert verdicts == {"GO", "NO_GO", "ABSTAIN", "BLOCK"}


def test_only_type_invalid_rows_reach_the_kernel_as_raw_values():
    raw = {v["id"] for v in VECTORS
           if not isinstance(_kernel_form(_decode(v["axes"])), torch.Tensor)
           or not isinstance(_kernel_form(_decode(v["weights"])), torch.Tensor)}
    type_invalid = {v["id"] for v in VECTORS if v["expect"].get("error") == "LAMBDA_TYPE_INVALID"}
    assert raw == type_invalid and raw


def _by_id(vid):
    return next(v for v in VECTORS if v["id"] == vid)


def test_strict_entry_refuses_what_the_legacy_aggregate_accepts():
    """The FF-01 divergence rows: legacy lambda_aggregate is unchanged; the strict entry refuses."""
    rows = {
        "x_gt_1": ("LAMBDA_AXIS_OUT_OF_RANGE", 0.948683),
        "nan_axis": ("LAMBDA_NONFINITE_AXIS", 0.0),
        "pos_inf_axis": ("LAMBDA_NONFINITE_AXIS", 0.0),
        "w_unnormalised_2_2": ("LAMBDA_WEIGHT_SUM", 0.72),
    }
    for vid, (code, legacy_value) in rows.items():
        v = _by_id(vid)
        axes, weights = _kernel_form(_decode(v["axes"])), _kernel_form(_decode(v["weights"]))
        assert round(float(lg.lambda_aggregate(axes, weights)), 6) == legacy_value, vid
        with pytest.raises(_v1.LambdaV1Error) as info:
            _v1.lambda_v1(axes, weights)
        assert info.value.code == code, vid
        assert _v1.lambda_v1_gate(axes, weights, 0.8)[:2] == ("BLOCK", code), vid
    with pytest.raises(ValueError) as legacy:
        lg.lambda_aggregate(_t([]), _t([]))
    assert not hasattr(legacy.value, "code")
    assert _v1.lambda_v1_gate(_t([]), _t([]), 0.8)[:2] == ("BLOCK", "LAMBDA_EMPTY")


# ----------------------------------------------------- tensor-side contract --


def _code(axes, weights):
    with pytest.raises(_v1.LambdaV1Error) as info:
        _v1.lambda_v1(axes, weights)
    return info.value.code


def test_tensor_type_contract():
    w2 = _t([0.5, 0.5])
    assert _code([0.5, 0.5], w2) == "LAMBDA_TYPE_INVALID"            # a list is not a tensor
    assert _code(None, w2) == "LAMBDA_TYPE_INVALID"
    assert _code(_t([0.5, 0.5]), [0.5, 0.5]) == "LAMBDA_TYPE_INVALID"
    assert _code(torch.tensor(0.5, dtype=torch.float64), _t([1.0])) == "LAMBDA_TYPE_INVALID"   # 0-d
    assert _code(_t([[0.5, 0.5]]), w2) == "LAMBDA_TYPE_INVALID"      # 2-D
    assert _code(torch.tensor([True, True]), w2) == "LAMBDA_TYPE_INVALID"
    assert _code(_t([0.5, 0.5]), torch.tensor([True, False])) == "LAMBDA_TYPE_INVALID"
    assert _code(torch.tensor([0.5 + 0j, 0.5 + 0j]), w2) == "LAMBDA_TYPE_INVALID"


def test_real_dtypes_are_accepted_and_computed_in_float64():
    ints = _v1.lambda_v1(torch.tensor([1, 1]), _t([0.5, 0.5]))
    assert ints.dtype == torch.float64 and float(ints) == 1.0
    assert _v1.lambda_v1_gate(torch.tensor([1, 1], dtype=torch.int32), _t([0.5, 0.5]), 0.8)[:2] == ("GO", None)
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        axes = torch.tensor([0.5, 0.25], dtype=dtype)   # exact in every float dtype
        got = _v1.lambda_v1(axes, _t([0.5, 0.5]).to(dtype))
        assert got.dtype == torch.float64
        assert abs(float(got) - ref.lambda_v1([0.5, 0.25], [0.5, 0.5])) <= 1e-12, dtype
    assert _code(torch.tensor([2, 1]), _t([0.5, 0.5])) == "LAMBDA_AXIS_OUT_OF_RANGE"
    assert _code(_t([0.5, 0.5]), torch.tensor([1, 1])) == "LAMBDA_WEIGHT_SUM"


def test_phase_precedence_matches_the_reference_on_tensors():
    nan = float("nan")
    assert _code(torch.tensor([], dtype=torch.bool), _t([])) == "LAMBDA_EMPTY"          # empty before dtype
    assert _code(torch.tensor([True]), _t([0.5, 0.5])) == "LAMBDA_LENGTH_MISMATCH"      # length before dtype
    assert _code(_t([1.5, nan]), _t([0.5, 0.5])) == "LAMBDA_NONFINITE_AXIS"
    assert _code(_t([nan, 1.5]), _t([0.5, 0.5])) == "LAMBDA_NONFINITE_AXIS"
    assert _code(_t([1.5, 0.9]), _t([nan, -1.0])) == "LAMBDA_AXIS_OUT_OF_RANGE"          # axes before weights
    assert _code(_t([0.5, 0.9]), _t([-1.0, nan])) == "LAMBDA_NONFINITE_WEIGHT"
    assert _code(_t([0.5, 0.9]), _t([2.0, -1.0])) == "LAMBDA_WEIGHT_NONPOSITIVE"         # sum is 1, still refused
    assert _code(_t([0.5, 0.9]), _t([1.7e308, 1.7e308])) == "LAMBDA_WEIGHT_SUM"          # the sum overflows
    assert _v1.lambda_v1_gate(_t([1.5, 0.9]), _t([0.5, 0.5]), nan)[:2] == ("BLOCK", "LAMBDA_TAU_INVALID")


def test_tau_contract():
    axes, weights = _t([0.95, 0.92, 0.88, 0.9]), _t([0.25] * 4)
    for bad in (True, False, None, "0.8", float("nan"), math.inf, -math.inf, 0, 0.0, -0.0,
                -0.1, 1.5, 1 + 1e-15, 10 ** 400, torch.tensor(0.8), object()):
        res = _v1.lambda_v1_gate(axes, weights, bad)
        assert (res.verdict, res.code, res.score, res.tau) == ("BLOCK", "LAMBDA_TAU_INVALID", None, None), bad
        assert (res.verdict, res.code) == ref.gate_v1([0.95, 0.92, 0.88, 0.9], [0.25] * 4, bad), bad
    assert _v1.lambda_v1_gate(axes, weights, 1)[:2] == ("NO_GO", "BELOW_TAU")
    assert _v1.lambda_v1_gate(axes, weights, 1).tau == 1.0
    assert _v1.lambda_v1_gate(axes, weights, 5e-324)[:2] == ("GO", None)
    assert _v1.lambda_v1_gate(_t([1.5]), _t([1.0]), 0.8).tau == 0.8   # valid tau is kept on a BLOCK


def test_gate_never_raises_on_garbage():
    garbage = [
        (object(), _t([1.0]), 0.8),
        (_t([0.5]), object(), 0.8),
        ("0.5", "1", 0.8),
        (None, None, None),
        ([0.5], [1.0], 0.8),
        (torch.tensor([True]), _t([1.0]), 0.8),
        (_t([0.5]), _t([1.0]), object()),
        (_t([float("nan")]), _t([1.0]), float("nan")),
        (_t([0.5]), _t([float("inf")]), 0.8),
        (_t([0.5, 0.5]), _t([1.0, 1e308]), 0.8),
    ]
    for axes, weights, tau in garbage:
        res = _v1.lambda_v1_gate(axes, weights, tau)
        assert res.verdict == "BLOCK" and res.code in SPEC_CODES and res.score is None, (axes, weights, tau)


def test_zero_axis_is_a_veto_not_an_error():
    for axes in (_t([0.0, 0.9]), _t([-0.0, 0.9]), torch.tensor([0, 1])):
        lam0 = _v1.lambda_v1(axes, _t([0.5, 0.5]))
        assert float(lam0) == 0.0
        for tau in (5e-324, 1e-300, 0.5, 0.8, 1.0, 1):
            res = _v1.lambda_v1_gate(axes, _t([0.5, 0.5]), tau)
            assert (res.verdict, res.code) == ("NO_GO", "ZERO_VETO"), (axes, tau)
            assert float(res.score) == 0.0


def test_no_rounding_before_the_compare():
    res = _v1.lambda_v1_gate(_t([0.649951]), _t([1.0]), 0.65)
    assert round(float(res.score), 4) == 0.65 and float(res.score) < 0.65
    assert (res.verdict, res.code) == ("NO_GO", "BELOW_TAU")


def test_verdict_does_not_depend_on_the_rounding_of_a_subnormal_lambda():
    # Λ ≈ 2e-322 is a subnormal with about 2% relative precision, so log(Λ as a float)
    # is not log Λ. With tau set to that float the reference does not see a tie; a gate
    # that compared log(score) would. The kernel compares log Λ, as the reference does.
    axes, weights = [5e-324, 1e-320], [0.5, 0.5]
    tau = ref.lambda_v1(axes, weights)
    assert 0.0 < tau < 1e-321
    expected = ref.gate_v1(axes, weights, tau)
    assert expected[0] != "ABSTAIN"
    assert _v1.lambda_v1_gate(_t(axes), _t(weights), tau)[:2] == expected


def test_inputs_are_read_never_modified_and_the_score_is_differentiable():
    axes = _t([0.9, 0.8, 0.95]).requires_grad_(True)
    weights = _t([0.4, 0.3, 0.3])
    before_a, before_w = axes.detach().clone(), weights.clone()
    res = _v1.lambda_v1_gate(axes, weights, 0.8)
    assert res.verdict == "GO"
    res.score.backward()
    assert torch.equal(axes.detach(), before_a) and torch.equal(weights, before_w)
    assert axes.grad is not None and bool(torch.all(torch.isfinite(axes.grad)))


# ------------------------------------------------- differential vs reference --


def _fuzz_cases(n=600, seed=20260929):
    rng = random.Random(seed)
    axis_specials = (0.0, -0.0, 1.0, 5e-324, 1e-320, 1.5, -0.1, 1.0000000000000002,
                     math.nan, math.inf, -math.inf)
    weight_faults = (0.0, -0.0, -0.1, math.nan, math.inf, -math.inf)
    cases = []
    for _ in range(n):
        k = rng.randint(1, 13)
        axes = [rng.random() for _ in range(k)]
        raw = [rng.random() + 1e-3 for _ in range(k)]
        total = math.fsum(raw)
        weights = [w / total for w in raw]
        if rng.random() < 0.3:
            axes[rng.randrange(k)] = rng.choice(axis_specials)
        roll = rng.random()
        if roll < 0.12:
            weights[rng.randrange(k)] += rng.choice((4e-13, -4e-13, 4e-12, -4e-12))
        elif roll < 0.2:
            weights[rng.randrange(k)] = rng.choice(weight_faults)
        elif roll < 0.24:
            weights.append(0.1)
        tau_roll = rng.random()
        if tau_roll < 0.35:
            try:
                log_lam = ref.log_lambda_v1(axes, weights)
            except ref.LambdaV1Error:
                log_lam = math.log(0.8)
            tau = 0.5 if log_lam == -math.inf else min(1.0, math.exp(log_lam + rng.uniform(-3e-9, 3e-9)))
        elif tau_roll < 0.9:
            tau = rng.random() or 0.5
        else:
            tau = rng.choice((0.0, -0.1, 1.5, math.nan, math.inf, 1, 5e-324))
        cases.append((axes, weights, tau))
    return cases


def test_differential_agreement_with_the_reference():
    seen_codes, seen_verdicts = set(), set()
    for axes, weights, tau in _fuzz_cases():
        k_axes, k_weights = _t(axes), _t(weights)
        try:
            expected, ref_code = ref.lambda_v1(axes, weights), None
        except ref.LambdaV1Error as err:
            expected, ref_code = None, err.code
        if ref_code is None:
            assert abs(float(_v1.lambda_v1(k_axes, k_weights)) - expected) <= 1e-12, (axes, weights)
        else:
            with pytest.raises(_v1.LambdaV1Error) as info:
                _v1.lambda_v1(k_axes, k_weights)
            assert info.value.code == ref_code, (axes, weights)
        res = _v1.lambda_v1_gate(k_axes, k_weights, tau)
        assert (res.verdict, res.code) == ref.gate_v1(axes, weights, tau), (axes, weights, tau)
        seen_verdicts.add(res.verdict)
        seen_codes.add(res.code)
    assert seen_verdicts == {"GO", "NO_GO", "ABSTAIN", "BLOCK"}
    assert set(SPEC_CODES) - {"LAMBDA_TYPE_INVALID", "LAMBDA_EMPTY"} <= seen_codes, seen_codes
    assert {"ZERO_VETO", "BELOW_TAU", "NUMERIC_TIE", None} <= seen_codes


# ------------------------------------------- deprecated implicit 0.5 threshold --

KEY_PHRASE = "default threshold 0.5 differs from policy_tau 0.8; pass tau"
AXES = _t([0.9, 0.8, 0.95])
CANDIDATES = _t([[0.9, 0.9, 0.9], [0.1, 0.9, 0.9], [0.6, 0.6, 0.6]])


def _bound_layer(threshold):
    layer = lg.layers.LambdaGate()
    layer.threshold = threshold
    return layer


OMITTED = {
    "_lambda.lambda_gate": lambda: lam.lambda_gate(AXES),
    "_lambda.lambda_gate_batch": lambda: lam.lambda_gate_batch(CANDIDATES),
    "szl_lambda_gate.lambda_gate": lambda: lg.lambda_gate(AXES),
    "szl_lambda_gate.lambda_gate_batch": lambda: lg.lambda_gate_batch(CANDIDATES),
    "layers.LambdaGate": lambda: lg.layers.LambdaGate()(AXES),
}
EXPLICIT = {
    "_lambda.lambda_gate": lambda t: lam.lambda_gate(AXES, threshold=t),
    "_lambda.lambda_gate_batch": lambda t: lam.lambda_gate_batch(CANDIDATES, threshold=t),
    "szl_lambda_gate.lambda_gate": lambda t: lg.lambda_gate(AXES, threshold=t),
    "szl_lambda_gate.lambda_gate_batch": lambda t: lg.lambda_gate_batch(CANDIDATES, threshold=t),
    "layers.LambdaGate": lambda t: _bound_layer(t)(AXES),
}


def _threshold_warnings(call):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        result = call()
    return result, [w for w in rec if "default threshold" in str(w.message)]


def test_threshold_default_is_the_none_sentinel():
    for fn in (lam.lambda_gate, lam.lambda_gate_batch, lg.lambda_gate, lg.lambda_gate_batch):
        assert inspect.signature(fn).parameters["threshold"].default is None, fn


@pytest.mark.parametrize("entry", sorted(OMITTED))
def test_omitted_threshold_warns_exactly_once(entry):
    with pytest.warns(DeprecationWarning, match=re.escape(KEY_PHRASE)):
        OMITTED[entry]()
    _, caught = _threshold_warnings(OMITTED[entry])
    assert len(caught) == 1, [str(w.message) for w in caught]
    assert caught[0].category is DeprecationWarning


@pytest.mark.parametrize("entry", sorted(set(OMITTED) - {"layers.LambdaGate"}))
def test_warning_points_at_the_caller(entry):
    _, caught = _threshold_warnings(OMITTED[entry])
    assert Path(caught[0].filename).resolve() == Path(__file__).resolve()


def test_layer_warning_points_at_the_layer():
    _, caught = _threshold_warnings(OMITTED["layers.LambdaGate"])
    assert Path(caught[0].filename).name == "layers.py"


@pytest.mark.parametrize("entry", sorted(EXPLICIT))
def test_explicit_threshold_never_warns(entry):
    for t in (0.5, 0.8, 0.0, 1.0):
        _, caught = _threshold_warnings(lambda: EXPLICIT[entry](t))
        assert caught == [], (entry, t)


@pytest.mark.parametrize("entry", sorted(OMITTED))
def test_omitted_threshold_behaves_exactly_like_0_5(entry):
    omitted, _ = _threshold_warnings(OMITTED[entry])
    explicit = EXPLICIT[entry](0.5)
    assert torch.equal(omitted.score, explicit.score)
    assert torch.equal(omitted.passed, explicit.passed)
    assert omitted.threshold == explicit.threshold == 0.5
    assert omitted.advisory is True


def test_warning_names_the_admit_contract_policy_tau_and_the_strict_entry():
    _, caught = _threshold_warnings(OMITTED["szl_lambda_gate.lambda_gate"])
    message = str(caught[0].message)
    assert f"policy_tau {POLICY_TAU}" in message and POLICY_TAU == 0.8
    assert "frontier/model_admit_contract.v1.json" in message
    assert "lambda_v1_gate(axes, weights, tau)" in message


def test_invalid_explicit_threshold_still_raises_without_the_deprecation():
    for bad in (-0.5, 1.5, float("nan")):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            with pytest.raises(ValueError):
                lg.lambda_gate(AXES, threshold=bad)
        assert not [w for w in rec if "default threshold" in str(w.message)]


TRACED = {
    "szl_lambda_gate.lambda_gate": (lambda z: lg.lambda_gate(z), (8, 4)),
    "szl_lambda_gate.lambda_gate_batch": (lambda z: lg.lambda_gate_batch(z), (5, 3, 4)),
    "_lambda.lambda_gate": (lambda z: lam.lambda_gate(z), (8, 4)),
    "_lambda.lambda_gate_batch": (lambda z: lam.lambda_gate_batch(z), (5, 3, 4)),
    "layers.LambdaGate": (lg.layers.LambdaGate(), (8, 4)),
    "explicit 0.5 (tests/test_lambda.py:443)": (lambda z: lg.lambda_gate(z, threshold=0.5), (8, 4)),
}


@pytest.mark.parametrize("entry", sorted(TRACED))
def test_threshold_path_is_trace_safe_under_fullgraph(entry):
    """Dynamo cannot trace warnings.warn, so the sentinel skips it while compiling.

    backend="eager" runs dynamo's fullgraph trace without the inductor C++
    toolchain, so this runs on every CI host; the inductor variants in
    tests/test_lambda.py still run where a compiler exists.
    """
    fn, shape = TRACED[entry]
    compiled = torch.compile(lambda z: tuple(fn(z)[:2]), fullgraph=True, backend="eager")
    x = torch.rand(*shape, dtype=torch.float32)
    score, passed = compiled(x)
    assert torch.allclose(score, lg.lambda_aggregate(x), atol=1e-6)
    assert torch.equal(passed, lg.lambda_aggregate(x) >= 0.5)


# ------------------------------------------------------------------ honesty --


def test_selfcheck_still_reports_conjecture_1_open():
    sc = lg.selfcheck()
    assert "Conjecture 1 (open)" in sc["lambda_status"]
    assert sc["all_axioms_hold"] is True and sc["advisory"] is True
    assert "Conjecture 1 (open)" in lg.PROVENANCE["lambda_status"]


STRICT_PROOF = re.compile(r"\bprov(en|ed|able|ably)\b|\btheorem\b", re.IGNORECASE)
STRICT_UNIQUE = re.compile(r"uniqu", re.IGNORECASE)
STRICT_CONJECTURE = re.compile(r"conjecture[ _-]?1", re.IGNORECASE)


def test_strict_entry_makes_no_proof_or_uniqueness_claim():
    for n, line in enumerate(Path(_v1.__file__).read_text(encoding="utf-8").splitlines(), 1):
        assert not STRICT_PROOF.search(line), f"_v1.py:{n}: {line}"
        if STRICT_UNIQUE.search(line):
            assert STRICT_CONJECTURE.search(line), f"_v1.py:{n}: {line}"

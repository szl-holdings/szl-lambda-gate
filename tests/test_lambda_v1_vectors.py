# SPDX-License-Identifier: Apache-2.0
# © 2026 SZL Holdings · Stephen P. Lutar · ORCID 0009-0001-0110-4173
"""szl.lambda/v1: the spec as data, the golden vectors, and the stdlib reference.

Stdlib only; no torch import. Run on its own with:

    python -m pytest -q -p no:cacheprovider tests/test_lambda_v1_vectors.py

The reference lives in ``reference/``, which is not on CI's PYTHONPATH, so it
is loaded by file path. Floats in the vectors are ``f64:<16 hex digits>``
(IEEE-754 binary64 bits); this file decodes them itself rather than trusting
the module under test to decode its own expectations.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
import math
import random
import re
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "spec" / "szl.lambda.v1.json"
VECTORS_PATH = ROOT / "spec" / "lambda_v1_vectors.json"
REFERENCE_PATH = ROOT / "reference" / "szl_lambda_v1.py"
CONTRACT_PATH = ROOT / "frontier" / "model_admit_contract.v1.json"
LEGACY_SOURCE_PATH = ROOT / "tests" / "lambda_aggregator_source.py"
README_PATH = ROOT / "README.md"

ERROR_CODES = (
    "LAMBDA_TYPE_INVALID",
    "LAMBDA_EMPTY",
    "LAMBDA_LENGTH_MISMATCH",
    "LAMBDA_NONFINITE_AXIS",
    "LAMBDA_AXIS_OUT_OF_RANGE",
    "LAMBDA_NONFINITE_WEIGHT",
    "LAMBDA_WEIGHT_NONPOSITIVE",
    "LAMBDA_WEIGHT_SUM",
    "LAMBDA_TAU_INVALID",
)
VERDICTS = ("GO", "NO_GO", "ABSTAIN", "BLOCK")
OUTCOME_CODES = {"GO": {None}, "NO_GO": {"ZERO_VETO", "BELOW_TAU"}, "ABSTAIN": {"NUMERIC_TIE"}}

# Rows the vectors must carry: szl-math-core.md §3, FUSION_BRIEF E5 and the tie band.
REQUIRED_IDS = {
    "nominal", "hidden_weak", "x_gt_1", "nan_axis", "pos_inf_axis", "neg_inf_axis",
    "negative_axis", "zero_axis", "w_unnormalised_2_2", "w_zero_weight", "empty",
    "e5_round_4dp_not_a_pass", "e5_dropped_axis_present", "e5_dropped_axis_renormalised",
    "e5_negative_weight", "tie_inside_above", "tie_inside_below", "tie_outside_above",
    "tie_outside_below",
}

F64_RE = re.compile(r"^f64:[0-9a-f]{16}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


# ----------------------------------------------------------------- helpers --


def _load_reference():
    name = "szl_lambda_v1_reference"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REFERENCE_PATH)
    assert spec is not None and spec.loader is not None, REFERENCE_PATH
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_bytes(obj) -> bytes:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _f64(x: float) -> str:
    return "f64:" + struct.pack(">d", x).hex()


def _decode_scalar(value):
    if isinstance(value, str) and value.startswith("f64:"):
        assert F64_RE.match(value), value
        return struct.unpack(">d", bytes.fromhex(value[4:]))[0]
    return value


def _decode(value):
    if isinstance(value, list):
        return [_decode_scalar(v) for v in value]
    return _decode_scalar(value)


ref = _load_reference()
SPEC = _load_json(SPEC_PATH)
VECTORS_DOC = _load_json(VECTORS_PATH)
VECTORS = VECTORS_DOC["vectors"]


# -------------------------------------------------------------- spec as data --


def test_spec_declares_the_contract():
    assert SPEC["schema"] == "szl.lambda/v1"
    assert SPEC["weight_sum_tol"] == 1e-12
    assert SPEC["gate"]["tie_eps"] == 1e-9
    assert [e["code"] for e in SPEC["error_codes"]] == list(ERROR_CODES)
    assert SPEC["gate"]["verdicts"] == list(VERDICTS)
    assert SPEC["uniqueness"] == "CONJECTURE_1_NOT_USED"
    assert SPEC["domain"]["renormalisation"].startswith("never")
    assert SPEC["domain"]["clamping"].startswith("never")
    assert SPEC["domain"]["rounding"].startswith("never")


def test_spec_tau_is_required_and_sourced_from_the_admit_contract():
    tau = SPEC["gate"]["tau"]
    assert tau["required"] is True
    assert tau["implicit_default"] is None
    path, _, pointer = tau["default_source"].partition("#/")
    assert path == "frontier/model_admit_contract.v1.json"
    contract = _load_json(ROOT / path)
    assert contract[pointer] == tau["policy_tau_at_authoring"] == 0.8


def test_reference_constants_match_the_spec():
    assert ref.SCHEMA == SPEC["schema"]
    assert ref.WEIGHT_SUM_TOL == SPEC["weight_sum_tol"]
    assert ref.TIE_EPS == SPEC["gate"]["tie_eps"]
    assert tuple(ref.ERROR_CODES) == ERROR_CODES
    assert tuple(ref.VERDICTS) == VERDICTS


def test_reference_imports_only_the_standard_library():
    allowed = {"__future__", "hashlib", "json", "math", "struct", "typing"}
    tree = ast.parse(REFERENCE_PATH.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert imported <= allowed, imported - allowed


# ------------------------------------------------------------------ digest --


def test_vectors_digest_matches_spec():
    recorded = SPEC["vectors"]
    assert recorded["path"] == "spec/lambda_v1_vectors.json"
    assert recorded["digest"] == {
        "algorithm": "sha256",
        "over": "canonical_json",
        "sort_keys": True,
        "separators": [",", ":"],
        "ensure_ascii": False,
        "allow_nan": False,
        "encoding": "utf-8",
    }
    assert SHA256_RE.match(recorded["sha256"])
    assert hashlib.sha256(_canonical_bytes(VECTORS_DOC)).hexdigest() == recorded["sha256"]
    assert ref.canonical_sha256(VECTORS_DOC) == recorded["sha256"]
    assert recorded["count"] == len(VECTORS)


def test_vectors_digest_survives_crlf_checkout():
    raw = VECTORS_PATH.read_bytes().replace(b"\r\n", b"\n")
    crlf = raw.replace(b"\n", b"\r\n")
    assert crlf != raw
    assert hashlib.sha256(crlf).hexdigest() != hashlib.sha256(raw).hexdigest()
    lf_digest = hashlib.sha256(_canonical_bytes(json.loads(raw.decode("utf-8")))).hexdigest()
    crlf_digest = hashlib.sha256(_canonical_bytes(json.loads(crlf.decode("utf-8")))).hexdigest()
    assert lf_digest == crlf_digest == SPEC["vectors"]["sha256"]


# ----------------------------------------------------------------- vectors --


def test_vectors_are_well_formed():
    assert VECTORS_DOC["schema"] == "szl.lambda/v1.vectors"
    assert VECTORS_DOC["spec"] == "spec/szl.lambda.v1.json"
    ids = [v["id"] for v in VECTORS]
    assert len(ids) == len(set(ids)), "vector ids must be unique"
    for v in VECTORS:
        vid = v["id"]
        assert isinstance(v["source"], str) and v["source"], vid
        tol = v["value_tol"]
        assert isinstance(tol, float) and 0.0 < tol <= 1e-9, vid
        for key in ("axes", "weights", "tau"):
            assert key in v, (vid, key)
        expect = v["expect"]
        assert ("value_f64" in expect) != ("error" in expect), vid
        if "value_f64" in expect:
            assert F64_RE.match(expect["value_f64"]), vid
            value = _decode(expect["value_f64"])
            assert 0.0 <= value <= 1.0, vid
            assert expect["value_decimal"] == repr(value), vid
        else:
            assert expect["error"] in ERROR_CODES and expect["error"] != "LAMBDA_TAU_INVALID", vid
        verdict, code = expect["verdict"], expect["code"]
        assert verdict in VERDICTS, vid
        if verdict == "BLOCK":
            assert code in ERROR_CODES, vid
        else:
            assert code in OUTCOME_CODES[verdict], vid
            assert "value_f64" in expect, vid
        for key in ("axes", "weights"):
            if isinstance(v[key], list) and all(isinstance(e, str) and e.startswith("f64:") for e in v[key]):
                decimals = [repr(x) for x in _decode(v[key])]
                assert v[key + "_decimal"] == decimals, (vid, key)
        if isinstance(v["tau"], str) and v["tau"].startswith("f64:"):
            assert v["tau_decimal"] == repr(_decode(v["tau"])), vid


def test_vectors_cover_the_required_rows_codes_and_verdicts():
    ids = {v["id"] for v in VECTORS}
    assert REQUIRED_IDS <= ids, REQUIRED_IDS - ids
    blocked_codes = {v["expect"]["code"] for v in VECTORS if v["expect"]["verdict"] == "BLOCK"}
    assert blocked_codes == set(ERROR_CODES)
    assert {v["expect"]["verdict"] for v in VECTORS} == set(VERDICTS)
    outcomes = {v["expect"]["code"] for v in VECTORS if v["expect"]["verdict"] != "BLOCK"}
    assert outcomes == {None, "ZERO_VETO", "BELOW_TAU", "NUMERIC_TIE"}


def _by_id(vid):
    for v in VECTORS:
        if v["id"] == vid:
            return v
    raise KeyError(vid)


def test_named_rows_have_the_planned_outcomes():
    """The E5 and math-core rows say what the plan says, not just what the reference says."""
    def value(vid):
        return _decode(_by_id(vid)["expect"]["value_f64"])

    assert round(value("nominal"), 4) == 0.9121
    assert round(value("hidden_weak"), 4) == 0.7653
    assert _by_id("hidden_weak")["expect"]["verdict"] == "NO_GO"
    assert value("zero_axis") == 0.0
    assert _by_id("zero_axis")["expect"]["code"] == "ZERO_VETO"
    expected_errors = {
        "x_gt_1": "LAMBDA_AXIS_OUT_OF_RANGE",
        "nan_axis": "LAMBDA_NONFINITE_AXIS",
        "pos_inf_axis": "LAMBDA_NONFINITE_AXIS",
        "neg_inf_axis": "LAMBDA_NONFINITE_AXIS",
        "negative_axis": "LAMBDA_AXIS_OUT_OF_RANGE",
        "w_unnormalised_2_2": "LAMBDA_WEIGHT_SUM",
        "w_zero_weight": "LAMBDA_WEIGHT_NONPOSITIVE",
        "empty": "LAMBDA_EMPTY",
        "e5_dropped_axis_renormalised": "LAMBDA_LENGTH_MISMATCH",
        "e5_negative_weight": "LAMBDA_WEIGHT_NONPOSITIVE",
    }
    for vid, code in expected_errors.items():
        assert _by_id(vid)["expect"]["error"] == code, vid
        assert _by_id(vid)["expect"]["verdict"] == "BLOCK", vid
    # 0.649951 rounds to 0.65 at 4 dp; unrounded it is below tau 0.65.
    row = _by_id("e5_round_4dp_not_a_pass")
    assert _decode(row["tau"]) == 0.65
    assert round(value("e5_round_4dp_not_a_pass"), 4) == 0.65
    assert (row["expect"]["verdict"], row["expect"]["code"]) == ("NO_GO", "BELOW_TAU")
    assert round(value("e5_dropped_axis_present"), 4) == 0.5533
    for vid in ("tie_inside_above", "tie_inside_below"):
        assert _by_id(vid)["expect"]["verdict"] == "ABSTAIN", vid
    assert _by_id("tie_outside_above")["expect"]["verdict"] == "GO"
    assert _by_id("tie_outside_below")["expect"]["verdict"] == "NO_GO"


@pytest.mark.parametrize("vector", VECTORS, ids=[v["id"] for v in VECTORS])
def test_reference_matches_vector(vector):
    axes, weights, tau = _decode(vector["axes"]), _decode(vector["weights"]), _decode(vector["tau"])
    expect = vector["expect"]
    if "value_f64" in expect:
        got = ref.lambda_v1(axes, weights)
        assert isinstance(got, float)
        assert _f64(got) == expect["value_f64"], (got, expect["value_decimal"])
    else:
        with pytest.raises(ref.LambdaV1Error) as info:
            ref.lambda_v1(axes, weights)
        assert info.value.code == expect["error"]
    assert ref.gate_v1(axes, weights, tau) == (expect["verdict"], expect["code"])


def test_log_lambda_agrees_with_lambda_on_every_value_vector():
    for v in VECTORS:
        if "value_f64" not in v["expect"]:
            continue
        axes, weights = _decode(v["axes"]), _decode(v["weights"])
        log_lam = ref.log_lambda_v1(axes, weights)
        lam = ref.lambda_v1(axes, weights)
        if lam == 0.0:
            assert log_lam == -math.inf, v["id"]
        else:
            assert log_lam <= 0.0 and _f64(math.exp(log_lam)) == _f64(lam), v["id"]


# --------------------------------------------------------------- semantics --


def test_zero_axis_is_no_go_through_log_minus_inf():
    axes, weights = [0.0, 0.9], [0.5, 0.5]
    assert ref.log_lambda_v1(axes, weights) == -math.inf
    lam = ref.lambda_v1(axes, weights)
    assert lam == 0.0 and _f64(lam) == "f64:0000000000000000"
    for tau in (5e-324, 1e-300, 0.5, 0.8, 1.0, 1):
        assert ref.gate_v1(axes, weights, tau) == ("NO_GO", "ZERO_VETO"), tau


def test_tau_is_a_required_argument_with_no_default():
    params = inspect.signature(ref.gate_v1).parameters
    assert list(params) == ["axes", "weights", "tau"]
    assert all(p.default is inspect.Parameter.empty for p in params.values())
    lam_params = inspect.signature(ref.lambda_v1).parameters
    assert all(p.default is inspect.Parameter.empty for p in lam_params.values())
    with pytest.raises(TypeError):
        ref.gate_v1([0.9], [1.0])  # noqa: the missing tau is the point


def test_no_rounding_before_the_compare():
    lam = ref.lambda_v1([0.649951], [1.0])
    assert round(lam, 4) == 0.65 and lam < 0.65
    assert ref.gate_v1([0.649951], [1.0], 0.65) == ("NO_GO", "BELOW_TAU")


def test_gate_fails_closed_and_never_raises_on_garbage():
    garbage = [
        (object(), [1.0], 0.8),
        ([0.5], object(), 0.8),
        ("0.5", "1", 0.8),
        (b"\x01", [1.0], 0.8),
        ({"a": 0.5}, {"a": 1.0}, 0.8),
        ((x for x in [0.5]), [1.0], 0.8),
        (None, None, None),
        ([0.5], [1.0], object()),
        ([0.5], [1.0], "0.8"),
        ([0.5], [1.0], 10 ** 400),
        ([0.5], [10 ** 400], 0.8),
        ([10 ** 400], [1.0], 0.8),
        ([0.5, 0.5], [1, 10 ** 400], 0.8),
        ([float("nan")], [1.0], float("nan")),
        ([0.5], [1.0], -math.inf),
        ([0.5], [1.0], 0),
        ([0.5], [1.0], True),
    ]
    for axes, weights, tau in garbage:
        verdict, code = ref.gate_v1(axes, weights, tau)
        assert verdict == "BLOCK" and code in ERROR_CODES, (axes, weights, tau)
    # Rows 0-6 and 10-13 are bad Λ inputs; the others only have a bad tau.
    for axes, weights, _ in garbage[:7] + garbage[10:14]:
        with pytest.raises(ref.LambdaV1Error) as info:
            ref.lambda_v1(axes, weights)
        assert info.value.code in ERROR_CODES


def test_error_type_carries_its_code():
    assert issubclass(ref.LambdaV1Error, ValueError)
    with pytest.raises(ValueError) as info:
        ref.lambda_v1([], [])
    assert info.value.code == "LAMBDA_EMPTY"
    assert "LAMBDA_EMPTY" in str(info.value)


def test_error_precedence_does_not_depend_on_axis_order():
    nan = float("nan")
    for axes in ([1.5, nan], [nan, 1.5]):
        with pytest.raises(ref.LambdaV1Error) as info:
            ref.lambda_v1(axes, [0.5, 0.5])
        assert info.value.code == "LAMBDA_NONFINITE_AXIS"


def test_value_is_bitwise_invariant_under_permutation():
    rng = random.Random(20260929)
    for _ in range(200):
        k = rng.randint(1, 13)
        axes = [rng.random() for _ in range(k)]
        raw = [rng.random() + 1e-3 for _ in range(k)]
        weights = [w / math.fsum(raw) for w in raw]
        if abs(math.fsum(weights) - 1.0) > 1e-12:
            continue
        base = _f64(ref.lambda_v1(axes, weights))
        order = list(range(k))
        rng.shuffle(order)
        permuted = _f64(ref.lambda_v1([axes[i] for i in order], [weights[i] for i in order]))
        assert permuted == base


def test_output_is_in_the_unit_interval_and_zero_only_on_a_zero_axis():
    rng = random.Random(7)
    for _ in range(500):
        k = rng.randint(1, 8)
        axes = [rng.choice([0.0, 1.0, 5e-324, rng.random()]) for _ in range(k)]
        weights = [1.0 / k] * k
        if abs(math.fsum(weights) - 1.0) > 1e-12:
            continue
        lam = ref.lambda_v1(axes, weights)
        assert 0.0 <= lam <= 1.0
        assert (lam == 0.0) == any(x == 0.0 for x in axes), axes


def test_f64_helpers_round_trip():
    for x in (0.0, -0.0, 1.0, 0.8, 5e-324, math.inf, -math.inf):
        assert ref.encode_f64(x) == _f64(x)
        assert _f64(ref.decode_f64(ref.encode_f64(x))) == _f64(x)
    assert math.isnan(ref.decode_f64(ref.encode_f64(float("nan"))))
    for bad in ("f64:123", "F64:3fe999999999999a", "0.8", "f64:3FE999999999999A"):
        with pytest.raises(ValueError):
            ref.decode_f64(bad)


# ------------------------------------------------------------------ honesty --

# The two rules of szl-holdings/.github reusable-overclaim-guard.yml (pinned in
# .github/workflows/overclaim-guard.yml), applied here to more files than the
# org guard scans.
ORG_RULE_A = re.compile(
    r"(Λ|lambda)[^.|]{0,80}(is|are|was)[^.|]{0,40}uniqu"
    r"|uniqu[a-z]*[^.|]{0,40}(of|for)[^.|]{0,20}(Λ|lambda)[^.|]{0,40}(is |are )?"
    r"(proven|proved|established|a theorem)",
    re.IGNORECASE,
)
ORG_RULE_A_SAFE = re.compile(
    r"conjecture|conditional|modulo|≈Λ|≉Λ|Theorem U|TheoremU|U₁|U₂|under IA|Identifiab"
    r"|Anchored|Normalized|statement-only|machine-checked|false|not a theorem|never|stays"
    r"|open|bounty|lambda_unique_|unique aggregator|min-gate|impostor",
    re.IGNORECASE,
)
ORG_RULE_B = re.compile(
    r"conjecture[ _-]?1\b[^.|]{0,25}(is |was |now |been |=|: )?"
    r"(proven|proved|closed|resolved|solved|a theorem|holds unconditionally)",
    re.IGNORECASE,
)
ORG_RULE_B_SAFE = re.compile(
    r"statement-only|machine-checked|false|open|not |never|stays|bounty|non-claim|unqualified|overclaim",
    re.IGNORECASE,
)
# Stricter rule for the surfaces this contract adds: no proof vocabulary at
# all, and uniqueness is only ever named together with Conjecture 1.
STRICT_PROOF = re.compile(r"\bprov(en|ed|able|ably)\b|\btheorem\b", re.IGNORECASE)
STRICT_UNIQUE = re.compile(r"uniqu", re.IGNORECASE)
STRICT_CONJECTURE = re.compile(r"conjecture[ _-]?1", re.IGNORECASE)


def _readme_v1_section() -> str:
    text = README_PATH.read_text(encoding="utf-8")
    start = text.index("\n## szl.lambda/v1")
    end = text.find("\n## ", start + 1)
    return text[start: end if end != -1 else len(text)]


def test_no_file_says_lambda_is_proven_or_unique():
    surfaces = {
        "README.md": README_PATH.read_text(encoding="utf-8"),
        "spec/szl.lambda.v1.json": SPEC_PATH.read_text(encoding="utf-8"),
        "spec/lambda_v1_vectors.json": VECTORS_PATH.read_text(encoding="utf-8"),
        "reference/szl_lambda_v1.py": REFERENCE_PATH.read_text(encoding="utf-8"),
        "tests/lambda_aggregator_source.py": LEGACY_SOURCE_PATH.read_text(encoding="utf-8"),
    }
    for name, text in surfaces.items():
        for n, line in enumerate(text.splitlines(), 1):
            if ORG_RULE_A.search(line):
                assert ORG_RULE_A_SAFE.search(line), f"{name}:{n}: {line}"
            if ORG_RULE_B.search(line):
                assert ORG_RULE_B_SAFE.search(line), f"{name}:{n}: {line}"
    strict = {
        "README.md#szl.lambda/v1": _readme_v1_section(),
        "spec/szl.lambda.v1.json": surfaces["spec/szl.lambda.v1.json"],
        "spec/lambda_v1_vectors.json": surfaces["spec/lambda_v1_vectors.json"],
        "reference/szl_lambda_v1.py": surfaces["reference/szl_lambda_v1.py"],
    }
    for name, text in strict.items():
        for n, line in enumerate(text.splitlines(), 1):
            assert not STRICT_PROOF.search(line), f"{name}:{n}: {line}"
            if STRICT_UNIQUE.search(line):
                assert STRICT_CONJECTURE.search(line), f"{name}:{n}: {line}"


def test_readme_points_at_the_spec_vectors_and_digest():
    section = _readme_v1_section()
    for needle in (
        "spec/szl.lambda.v1.json",
        "spec/lambda_v1_vectors.json",
        "reference/szl_lambda_v1.py",
        SPEC["vectors"]["sha256"],
        "Conjecture 1",
    ):
        assert needle in section, needle


def test_legacy_python_source_is_marked_not_canonical_and_unchanged():
    header = "\n".join(LEGACY_SOURCE_PATH.read_text(encoding="utf-8").splitlines()[:8])
    assert "NOT canonical" in header
    assert "spec/szl.lambda.v1.json" in header
    spec = importlib.util.spec_from_file_location("lambda_aggregator_source_ff01", LEGACY_SOURCE_PATH)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    # What the header says about it is still true: +Inf counts as 1, NaN leaks, empty is 0.
    assert round(legacy.lambda_aggregate([math.inf, 0.9]), 6) == 0.948683
    assert math.isnan(legacy.lambda_aggregate([float("nan"), 0.9]))
    assert legacy.lambda_aggregate([]) == 0.0

"""Replay the frozen szl.lambda/v1 vectors and synthetic envelope faults.

The numerical result comes only from the existing reference/szl_lambda_v1.py.
This script orchestrates source checks and reporting, not another formula.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "spec" / "lambda_v1_vectors.json"
SPEC = ROOT / "spec" / "szl.lambda.v1.json"
REFERENCE = ROOT / "reference" / "szl_lambda_v1.py"
FIXTURE_ORIGIN_COMMIT = "6a874e11ab948a47e982be3651b8021ba6b82e19"
VECTOR_BLOB = "d99cc31298b8fad424d6783bdd2295ca9d7adab2"
REFERENCE_BLOB = "8396d8bee64f8aa4431e215dba203eae25225fc2"
CONTRACT_BLOB = "57c12a3c9233a4a1fe788e9cb2329e941f4e6be5"
CANONICAL_SHA256 = "61bfb0410b9f0eaab0eb9f22f29cb7cb13cfde8c083fe308d895565d6ba9ebd4"
UNIT = "dimensionless"  # Replay envelope label; the frozen fixture has no unit field.


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_blob(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def repository_head() -> str | None:
    """Record the checkout revision when Git is available; bytes remain bound below."""
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    head = result.stdout.strip()
    return head if result.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", head) else None


def load_reference() -> Any:
    spec = importlib.util.spec_from_file_location("szl_lambda_v1_pinned", REFERENCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("Pinned reference cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def decode(value: Any, reference: Any) -> Any:
    if isinstance(value, str) and value.startswith("f64:"):
        return reference.decode_f64(value)
    if isinstance(value, list):
        return [decode(item, reference) for item in value]
    return value


def check_row_shape(row: dict[str, Any]) -> str | None:
    for field in ("id", "axes", "weights", "tau", "expect", "value_tol"):
        if field not in row:
            return f"MISSING_INPUT:{field}"
    return None


def evaluate_row(row: dict[str, Any], reference: Any) -> dict[str, Any]:
    axes = decode(row["axes"], reference)
    weights = decode(row["weights"], reference)
    tau = decode(row["tau"], reference)
    expected = row["expect"]
    value: float | None = None
    error: str | None = None
    try:
        value = reference.lambda_v1(axes, weights)
    except reference.LambdaV1Error as exc:
        error = exc.code
    verdict, code = reference.gate_v1(axes, weights, tau)
    value_f64 = reference.encode_f64(value) if value is not None else None
    expected_f64 = expected.get("value_f64")
    expected_error = expected.get("error")
    absolute_error = (
        abs(value - reference.decode_f64(expected_f64))
        if value is not None and expected_f64 is not None
        else None
    )
    # The pinned stdlib reference must match its own vectors bit for bit.
    numerical_match = (
        error == expected_error
        and (value_f64 == expected_f64 if expected_f64 is not None else value is None)
    )
    gate_match = (verdict, code) == (expected["verdict"], expected["code"])
    passed = numerical_match and gate_match
    return {
        "id": row["id"],
        "expected": {
            "value_f64": expected_f64,
            "error": expected_error,
            "verdict": expected["verdict"],
            "code": expected["code"],
            "value_tol": row["value_tol"],
        },
        "observed": {
            "value_f64": value_f64,
            "error": error,
            "verdict": verdict,
            "code": code,
            "absolute_error": absolute_error,
        },
        "pass": passed,
    }


def preflight(
    document: dict[str, Any], reference: Any, claimed_hash: str, unit: str
) -> tuple[str, str | None, str]:
    actual_hash = reference.canonical_sha256(document)
    if unit != UNIT:
        return "ABSTAIN", "UNIT_LABEL_MISMATCH", actual_hash
    if not isinstance(document.get("vectors"), list):
        return "ABSTAIN", "MISSING_INPUT:vectors", actual_hash
    for row in document["vectors"]:
        missing = check_row_shape(row)
        if missing:
            return "ABSTAIN", missing, actual_hash
    if actual_hash != CANONICAL_SHA256:
        return "FAIL", "FIXTURE_CHANGED", actual_hash
    if claimed_hash != actual_hash:
        return "ABSTAIN", "STALE_HASH", actual_hash
    return "PASS", None, actual_hash


def build_report() -> dict[str, Any]:
    """Build a deterministic report without writing into the source checkout."""
    reference_bytes = REFERENCE.read_bytes()
    fixture_bytes = FIXTURE.read_bytes()
    contract_bytes = SPEC.read_bytes()
    if git_blob(reference_bytes) != REFERENCE_BLOB:
        raise RuntimeError("Reference is not the pinned Git blob")
    if git_blob(fixture_bytes) != VECTOR_BLOB:
        raise RuntimeError("Fixture is not the pinned Git blob")
    if git_blob(contract_bytes) != CONTRACT_BLOB:
        raise RuntimeError("Contract is not the pinned Git blob")
    spec_doc = json.loads(contract_bytes.decode("utf-8"))
    recorded = spec_doc["vectors"]
    if (
        spec_doc["schema"] != "szl.lambda/v1"
        or recorded["path"] != "spec/lambda_v1_vectors.json"
        or recorded["sha256"] != CANONICAL_SHA256
        or recorded["count"] != 60
    ):
        raise RuntimeError("Contract does not bind the pinned 60-vector fixture")
    reference = load_reference()
    document = json.loads(fixture_bytes.decode("utf-8"))
    clean_status, clean_reason, actual_hash = preflight(
        document, reference, CANONICAL_SHA256, UNIT
    )
    ids = [row["id"] for row in document["vectors"]]
    if len(ids) != 60 or len(set(ids)) != 60 or actual_hash != CANONICAL_SHA256:
        raise RuntimeError("Fixture count, uniqueness, or canonical identity failed")
    rows = [evaluate_row(row, reference) for row in document["vectors"]] if clean_status == "PASS" else []
    passed = sum(row["pass"] for row in rows)
    clean_observed = "PASS" if clean_status == "PASS" and passed == 60 else "FAIL"
    clean = {
        "expected_status": "PASS",
        "observed_status": clean_observed,
        "preflight": {"status": clean_status, "reason": clean_reason},
        "expected_vector_count": 60,
        "observed_vector_count": len(rows),
        "observed_pass_count": passed,
        "rows": rows,
    }

    # All changes below are newly authored in memory. The pinned bytes stay intact.
    changed_number = copy.deepcopy(document)
    changed_number_row = next(row for row in changed_number["vectors"] if row["id"] == "nominal")
    changed_number_row["expect"]["value_f64"] = "f64:3fe0000000000000"  # 0.5
    number_status, number_reason, number_hash = preflight(
        changed_number, reference, CANONICAL_SHA256, UNIT
    )
    # A diagnostic comparison still shows the altered number is wrong. It is not
    # promoted as an authoritative result after preflight failure.
    number_diagnostic = evaluate_row(changed_number_row, reference)

    unit_status, unit_reason, unit_hash = preflight(
        document, reference, CANONICAL_SHA256, "joule"
    )
    missing = copy.deepcopy(document)
    missing_row = next(row for row in missing["vectors"] if row["id"] == "nominal")
    del missing_row["axes"]
    missing_status, missing_reason, missing_hash = preflight(
        missing, reference, CANONICAL_SHA256, UNIT
    )
    stale_claim = "0" * 64
    stale_status, stale_reason, stale_actual_hash = preflight(
        document, reference, stale_claim, UNIT
    )
    corruptions = [
        {
            "case": "changed_expected_number",
            "expected_status": "FAIL",
            "observed_status": number_status,
            "expected_reason": "FIXTURE_CHANGED",
            "observed_reason": number_reason,
            "claimed_source_sha256": CANONICAL_SHA256,
            "observed_payload_sha256": number_hash,
            "authoritative_evaluated_rows": 0,
            "diagnostic_evaluated_rows": 1,
            "diagnostic_only": number_diagnostic,
        },
        {
            "case": "changed_unit_label",
            "expected_status": "ABSTAIN",
            "observed_status": unit_status,
            "expected_reason": "UNIT_LABEL_MISMATCH",
            "observed_reason": unit_reason,
            "expected_unit": UNIT,
            "observed_unit": "joule",
            "observed_payload_sha256": unit_hash,
            "evaluated_rows": 0,
        },
        {
            "case": "missing_input",
            "expected_status": "ABSTAIN",
            "observed_status": missing_status,
            "expected_reason": "MISSING_INPUT:axes",
            "observed_reason": missing_reason,
            "missing_field": "axes",
            "observed_payload_sha256": missing_hash,
            "evaluated_rows": 0,
        },
        {
            "case": "stale_hash",
            "expected_status": "ABSTAIN",
            "observed_status": stale_status,
            "expected_reason": "STALE_HASH",
            "observed_reason": stale_reason,
            "claimed_source_sha256": stale_claim,
            "observed_payload_sha256": stale_actual_hash,
            "evaluated_rows": 0,
        },
    ]
    probes_match = all(
        case["observed_status"] == case["expected_status"]
        and case["observed_reason"] == case["expected_reason"]
        for case in corruptions
    ) and not number_diagnostic["pass"]

    report = {
        "schema": "szl.lambda/v1.offline-replay/1",
        "claim_scope": "Frozen numerical formula conformance only; no empirical result, model inference, or proof of uniqueness.",
        "overall_status": "PASS" if clean_observed == "PASS" and probes_match else "FAIL",
        "source_identity": {
            "repository": "szl-holdings/szl-lambda-gate",
            "repository_head": repository_head(),
            "declared_fixture_origin_commit": FIXTURE_ORIGIN_COMMIT,
            "fixture_git_blob_sha1": git_blob(fixture_bytes),
            "fixture_raw_sha256": sha256(fixture_bytes),
            "fixture_canonical_sha256": actual_hash,
            "contract_git_blob_sha1": git_blob(contract_bytes),
            "contract_sha256": sha256(contract_bytes),
            "reference_git_blob_sha1": git_blob(reference_bytes),
            "reference_sha256": sha256(reference_bytes),
            "driver_sha256": sha256(Path(__file__).read_bytes()),
        },
        "environment_identity": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "python_executable_name": Path(sys.executable).name,
            "python_compiler": platform.python_compiler(),
            "os": platform.system(),
            "os_release": platform.release(),
            "machine": platform.machine(),
            "byteorder": sys.byteorder,
            "dependencies": "Python standard library only",
        },
        "clean_case": clean,
        "synthetic_corruptions": corruptions,
        "limitations": [
            "The vector fixture is a frozen committed source artifact; its scratch generator was not available.",
            "The unit label exists only in this replay envelope, not in the upstream vector schema.",
            "Corruption diagnostics do not promote altered fixtures to authoritative evidence.",
            "This runner does not execute the Torch kernel, the Atelier NumPy port, or the repository's separate pytest suite.",
            "Numerical conformance does not establish empirical scientific truth or the open Lambda uniqueness conjecture.",
        ],
    }
    return report


def render_report(report: dict[str, Any]) -> bytes:
    """Stable bytes for evidence hashing and regression checks."""
    return (json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Explicit report path outside the source tree")
    args = parser.parse_args()
    report = build_report()
    output = args.output.resolve()
    if output == ROOT or ROOT in output.parents:
        parser.error("--output must be outside the source checkout")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = render_report(report)
    output.write_bytes(payload)
    output.with_suffix(output.suffix + ".sha256").write_text(sha256(payload) + "\n", encoding="ascii")
    clean = report["clean_case"]
    print(f"Clean: {clean['observed_pass_count']}/60; corruption probes: {len(report['synthetic_corruptions'])}; overall: {report['overall_status']}")
    print(f"Report: {output}")
    return 0 if report["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

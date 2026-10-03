# SPDX-License-Identifier: Apache-2.0
"""End-to-end checks for the offline replay report and its refusal states."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "replay_lambda_v1.py"
SPEC = importlib.util.spec_from_file_location("lambda_replay_stdlib", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Cannot load replay script")
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


class OfflineReplayTests(unittest.TestCase):
    def test_clean_replay_binds_60_outcomes_to_source(self) -> None:
        report = replay.build_report()
        self.assertEqual(report["overall_status"], "PASS")
        clean = report["clean_case"]
        self.assertEqual((clean["observed_vector_count"], clean["observed_pass_count"]), (60, 60))
        self.assertTrue(all(row["pass"] for row in clean["rows"]))
        self.assertEqual(report["source_identity"]["fixture_canonical_sha256"], replay.CANONICAL_SHA256)
        self.assertEqual(report["source_identity"]["reference_git_blob_sha1"], replay.REFERENCE_BLOB)
        rows = {row["id"]: row for row in clean["rows"]}
        self.assertEqual(rows["nominal"]["observed"]["value_f64"], "f64:3fed3035e27f23fe")
        self.assertEqual(rows["hidden_weak"]["observed"]["verdict"], "NO_GO")
        self.assertEqual(rows["hidden_weak"]["observed"]["code"], "BELOW_TAU")
        self.assertEqual(rows["x_gt_1"]["observed"]["code"], "LAMBDA_AXIS_OUT_OF_RANGE")

    def test_independent_corruptions_fail_or_abstain_before_promotion(self) -> None:
        reference = replay.load_reference()
        original = json.loads(replay.FIXTURE.read_text(encoding="utf-8"))

        changed = copy.deepcopy(original)
        changed["vectors"][0]["expect"]["value_f64"] = "f64:3fe0000000000000"
        self.assertEqual(replay.preflight(changed, reference, replay.CANONICAL_SHA256, replay.UNIT)[:2],
                         ("FAIL", "FIXTURE_CHANGED"))
        self.assertEqual(replay.preflight(original, reference, replay.CANONICAL_SHA256, "joule")[:2],
                         ("ABSTAIN", "UNIT_LABEL_MISMATCH"))
        missing = copy.deepcopy(original)
        del missing["vectors"][0]["axes"]
        self.assertEqual(replay.preflight(missing, reference, replay.CANONICAL_SHA256, replay.UNIT)[:2],
                         ("ABSTAIN", "MISSING_INPUT:axes"))
        self.assertEqual(replay.preflight(original, reference, "0" * 64, replay.UNIT)[:2],
                         ("ABSTAIN", "STALE_HASH"))

        reported = {case["case"]: case for case in replay.build_report()["synthetic_corruptions"]}
        self.assertFalse(reported["changed_expected_number"]["diagnostic_only"]["pass"])
        self.assertEqual(reported["changed_expected_number"]["authoritative_evaluated_rows"], 0)
        for name in ("changed_unit_label", "missing_input", "stale_hash"):
            self.assertEqual(reported[name]["evaluated_rows"], 0)

    def test_semantic_contract_change_cannot_pass_with_same_fixture_metadata(self) -> None:
        contract = json.loads(replay.SPEC.read_text(encoding="utf-8"))
        contract["gate"]["tie_eps"] = 1e-8
        with tempfile.TemporaryDirectory(prefix="lambda-contract-") as scratch:
            changed_path = Path(scratch) / "szl.lambda.v1.json"
            changed_path.write_text(json.dumps(contract), encoding="utf-8")
            with mock.patch.object(replay, "SPEC", changed_path):
                with self.assertRaisesRegex(RuntimeError, "Contract is not the pinned Git blob"):
                    replay.build_report()

    def test_cli_writes_identical_reports_and_matching_digest(self) -> None:
        with tempfile.TemporaryDirectory(prefix="lambda-replay-") as scratch:
            output = Path(scratch) / "report.json"
            command = [sys.executable, "-B", "-I", str(SCRIPT), "--output", str(output)]
            first = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(first.returncode, 0, first.stderr)
            first_bytes = output.read_bytes()
            self.assertEqual(json.loads(first_bytes)["overall_status"], "PASS")
            digest = hashlib.sha256(first_bytes).hexdigest()
            self.assertEqual(output.with_suffix(".json.sha256").read_text(encoding="ascii").strip(), digest)
            second = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(output.read_bytes(), first_bytes)


if __name__ == "__main__":
    unittest.main()

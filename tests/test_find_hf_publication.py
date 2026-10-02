"""Offline checks for locating the original, measured Hub publication."""

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "find_hf_publication.py"
spec = importlib.util.spec_from_file_location("find_hf_publication", SCRIPT)
assert spec and spec.loader
finder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(finder)

REPO = "szl-holdings/szl-lambda-gate"
TARGET = "SZLHOLDINGS/szl-lambda-gate"
TAG = "v0.2.1"
NAME = f"hf-mirror-receipt-lambda-gate-{TAG}"


def artifact(run_id: int, *, expired: bool = False, name: str = NAME) -> dict:
    return {"name": name, "expired": expired, "created_at": f"2026-10-02T{run_id:02d}:00:00Z",
            "workflow_run": {"id": run_id, "head_branch": "main", "head_sha": "a" * 40}}


def run(run_id: int) -> dict:
    return {"id": run_id, "path": ".github/workflows/hf-mirror.yml",
            "event": "workflow_dispatch", "conclusion": "success", "head_branch": "main",
            "head_sha": "a" * 40, "repository": {"full_name": REPO},
            "head_repository": {"full_name": REPO}}


def receipt(operation: str) -> dict:
    return {"state": "MEASURED", "operation": operation, "github_repo": REPO,
            "github_tag": TAG, "github_sha": "b" * 40, "github_workflow_sha": "a" * 40,
            "hf_repo_id": TARGET, "hf_revision": "c" * 40, "auth": "pat",
            "verified_file_count": 1, "file_sha256": {"README.md": "d" * 64}}


def test_newer_verification_does_not_displace_original_publication() -> None:
    receipts = {1: receipt("publish"), 2: receipt("verify-existing")}
    selected = finder.select_publication(
        [artifact(1), artifact(2)], lambda ident: run(ident), lambda ident: receipts[ident],
        repo=REPO, target=TARGET, tag=TAG, slug="lambda-gate", branch="main",
    )
    assert selected == 1


@pytest.mark.parametrize("tamper", [
    lambda data: data.update(github_workflow_sha="e" * 40),
    lambda data: data.update(verified_file_count=2),
    lambda data: data.update(state="PENDING_HUB_PR_REVIEW", operation="propose"),
    lambda data: data.update(hf_repo_id="someone-else/model"),
])
def test_unbound_or_proposal_artifact_cannot_be_selected(tamper) -> None:
    bad = receipt("publish")
    tamper(bad)
    with pytest.raises(RuntimeError, match="no measured publication receipt"):
        finder.select_publication(
            [artifact(1)], lambda ident: run(ident), lambda ident: bad,
            repo=REPO, target=TARGET, tag=TAG, slug="lambda-gate", branch="main",
        )


def test_other_workflow_and_expired_artifacts_are_ignored() -> None:
    wrong_run = run(2)
    wrong_run["path"] = ".github/workflows/other.yml"
    with pytest.raises(RuntimeError, match="no measured publication receipt"):
        finder.select_publication(
            [artifact(1, expired=True), artifact(2), artifact(3, name="other")],
            lambda ident: wrong_run, lambda ident: receipt("publish"),
            repo=REPO, target=TARGET, tag=TAG, slug="lambda-gate", branch="main",
        )

# SPDX-License-Identifier: Apache-2.0
"""Offline contract for finalizing a reviewed and merged Hub proposal."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "hf_mirror_finalize_test", ROOT / "scripts" / "hf_mirror_finalize.py",
)
assert spec and spec.loader
finalizer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(finalizer)

GITHUB_REPO = "szl-holdings/szl-lambda-gate"
HUB_REPO = "SZLHOLDINGS/szl-lambda-gate"
TAG = "v0.3.0"
SOURCE_SHA = "1" * 40
BASELINE_SHA = "2" * 40
MERGE_SHA = "3" * 40
PROPOSAL_SHA = "4" * 40
WORKFLOW_SHA = "5" * 40
FINALIZER_SHA = "6" * 40
OTHER_SHA = "7" * 40
PR_NUMBER = 17
RUN_ID = "123456789"
METADATA = {
    "library_name": "kernels", "license": "apache-2.0",
    "tags": ["doi:10.5281/zenodo.19944926"],
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class RevisionNotFoundError(Exception):
    pass


class Hub:
    def __init__(self, store: Path, before: dict[str, bytes], files: dict[str, bytes]) -> None:
        self.store = store
        self.before = before
        self.files = files
        self.main = MERGE_SHA
        self.tag: str | None = None
        self.main_answers: list[str] = []
        self.tag_answers: list[str] = []
        self.mutations: list[tuple[str, str]] = []
        self.status = "merged"
        self.is_pull_request = True
        self.target_branch = "refs/heads/main"
        self.merge_commit_oid = MERGE_SHA
        self.events = [types.SimpleNamespace(oid=PROPOSAL_SHA)]

    def repo_info(self, repo: str, *, repo_type: str, revision: str) -> types.SimpleNamespace:
        assert (repo, repo_type) == (HUB_REPO, "model")
        if revision == "main":
            return types.SimpleNamespace(sha=self.main_answers.pop(0) if self.main_answers else self.main)
        if revision == TAG:
            if self.tag is None:
                raise RevisionNotFoundError("missing tag")
            return types.SimpleNamespace(sha=self.tag_answers.pop(0) if self.tag_answers else self.tag)
        assert revision in (BASELINE_SHA, MERGE_SHA)
        return types.SimpleNamespace(sha=revision)

    def get_discussion_details(self, repo: str, discussion_num: int, *, repo_type: str) -> types.SimpleNamespace:
        assert (repo, repo_type, discussion_num) == (HUB_REPO, "model", PR_NUMBER)
        return types.SimpleNamespace(
            num=PR_NUMBER, repo_id=HUB_REPO, repo_type="model",
            status=self.status, is_pull_request=self.is_pull_request,
            target_branch=self.target_branch, merge_commit_oid=self.merge_commit_oid,
            events=self.events,
        )

    def list_repo_files(self, repo: str, *, repo_type: str, revision: str) -> list[str]:
        assert (repo, repo_type) == (HUB_REPO, "model")
        assert revision in ("main", BASELINE_SHA, MERGE_SHA, TAG)
        return sorted(self.before if revision == BASELINE_SHA else self.files)

    def download(self, repo: str, name: str, *, repo_type: str, revision: str, token: str) -> str:
        assert (repo, repo_type, token) == (HUB_REPO, "model", "test-token")
        path = self.store / revision / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((self.before if revision == BASELINE_SHA else self.files)[name])
        return str(path)

    def create_tag(self, repo: str, *, tag: str, repo_type: str, revision: str, token: str,
                   exist_ok: bool) -> None:
        assert (repo, tag, repo_type, revision, token, exist_ok) == (
            HUB_REPO, TAG, "model", MERGE_SHA, "test-token", False,
        )
        assert self.tag is None
        self.tag = revision
        self.mutations.append(("create_tag", revision))


@pytest.fixture
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> types.SimpleNamespace:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".github").mkdir()
    for key, value in {
        "GITHUB_REPOSITORY": GITHUB_REPO, "GITHUB_SHA": FINALIZER_SHA,
        "SOURCE_GITHUB_SHA": SOURCE_SHA, "RELEASE_TAG": TAG,
        "HF_REPO_ID": HUB_REPO, "HF_REPO_TYPE": "model",
        "HF_MIRROR_OIDC_RESOURCE": HUB_REPO, "HF_AUTH_MODE": "pat",
        "HF_TOKEN": "test-token", "HF_PROPOSAL_RUN": RUN_ID,
        "DEFAULT_BRANCH": "main",
    }.items():
        monkeypatch.setenv(key, value)
    item = {
        "hf_repo_id": HUB_REPO, "repo_type": "model", "oidc_resource": HUB_REPO,
        "hub_pr_required": True, "preserve_hub_card": True,
        "required_card_metadata": {"library_name": "kernels", "license": "apache-2.0"},
        "required_card_tags": METADATA["tags"],
        "required_release_assets": ["bom.json"],
        "replace_hub_paths": ["kernel.py"], "preserve_hub_paths": ["LICENSE"],
    }
    (tmp_path / ".github" / "hf-mirror.json").write_text(
        json.dumps({"targets": [item]}), encoding="utf-8",
    )
    finalizer.mirror.RELEASE_FILE.write_text(json.dumps({
        "tagName": TAG, "isDraft": False,
        "assets": [{"name": "bom.json", "size": 3, "digest": "sha256:" + digest(b"bom")}],
    }), encoding="utf-8")
    files = {
        "README.md": (
            "Curated model card.\n" + finalizer.mirror.BEGIN +
            f"\nRelease {TAG} from {SOURCE_SHA}.\n" + finalizer.mirror.END + "\n"
        ).encode(),
        "LICENSE": b"Hub license\n", "kernel.py": b"released source\n", "bom.json": b"bom",
    }
    before = {
        "README.md": b"Curated model card.\n", "LICENSE": files["LICENSE"],
        "kernel.py": b"previous source\n",
    }
    hub = Hub(tmp_path / "hub", before, files)
    proposal = {
        "state": "PENDING_HUB_PR_REVIEW", "operation": "propose",
        "github_repo": GITHUB_REPO, "github_sha": SOURCE_SHA,
        "github_workflow_sha": WORKFLOW_SHA, "github_tag": TAG,
        "hf_repo_id": HUB_REPO, "hf_repo_type": "model",
        "hf_oidc_resource": HUB_REPO, "hf_main_before": BASELINE_SHA,
        "hf_pr_url": f"https://huggingface.co/{HUB_REPO}/discussions/{PR_NUMBER}",
        "hf_pr_revision": f"refs/pr/{PR_NUMBER}", "hf_pr_commit": PROPOSAL_SHA,
        "file_count_before": len(before), "file_count": len(files),
        "preserved_hub_files": 1, "replaced_hub_files": ["kernel.py"],
        "staged_files": ["README.md", "bom.json", "kernel.py"],
        "release_assets": ["bom.json"],
        "release_asset_manifest": [{"name": "bom.json", "size": 3,
                                    "digest": "sha256:" + digest(b"bom")}],
        "file_sha256": {name: digest(data) for name, data in files.items()},
        "auth": "oidc", "hub_main_published": False, "hub_tag_created": False,
    }
    finalizer.PROPOSAL_FILE.write_text(json.dumps(proposal), encoding="utf-8")
    run = {
        "id": int(RUN_ID), "head_sha": WORKFLOW_SHA,
        "repository": {"full_name": GITHUB_REPO},
        "head_repository": {"full_name": GITHUB_REPO},
        "path": ".github/workflows/hf-mirror.yml", "event": "workflow_dispatch",
        "conclusion": "failure", "head_branch": "main",
    }
    finalizer.PROPOSAL_RUN_FILE.write_text(json.dumps(run), encoding="utf-8")
    module = types.ModuleType("huggingface_hub")
    module.HfApi = lambda *, token: hub if token == "test-token" else None

    class ModelCard:
        def __init__(self, text: str) -> None:
            self.text = text
            self.data = types.SimpleNamespace(to_dict=lambda: dict(METADATA))

        @classmethod
        def load(cls, path: Path) -> "ModelCard":
            return cls(Path(path).read_text(encoding="utf-8"))

    module.ModelCard, module.hf_hub_download = ModelCard, hub.download
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    errors = types.ModuleType("huggingface_hub.errors")
    errors.RevisionNotFoundError = RevisionNotFoundError
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors)
    return types.SimpleNamespace(hub=hub, proposal=proposal, run=run)


def assert_no_receipt_or_tag(setup: types.SimpleNamespace) -> None:
    assert setup.hub.mutations == []
    assert setup.hub.tag is None
    assert not finalizer.mirror.RECEIPT_FILE.exists()


def test_merged_proposal_creates_tag_and_measured_receipt(setup: types.SimpleNamespace) -> None:
    assert finalizer.finalize() == "MEASURED"
    receipt = json.loads(finalizer.mirror.RECEIPT_FILE.read_text(encoding="utf-8"))
    assert setup.hub.mutations == [("create_tag", MERGE_SHA)]
    assert receipt["hf_revision"] == receipt["hf_main_observed"] == MERGE_SHA
    assert receipt["github_sha"] == SOURCE_SHA
    assert receipt["github_workflow_sha"] == FINALIZER_SHA
    assert receipt["source_proposal_run"] == RUN_ID
    assert receipt["source_proposal_sha256"] == digest(finalizer.PROPOSAL_FILE.read_bytes())
    assert receipt["file_sha256"] == setup.proposal["file_sha256"]
    assert receipt["operation"] == "publish" and receipt["state"] == "MEASURED"
    assert receipt["auth"] == receipt["publication_auth"] == "pat"


def test_matching_existing_tag_is_safe_to_retry(setup: types.SimpleNamespace) -> None:
    setup.hub.tag = MERGE_SHA
    assert finalizer.finalize() == "MEASURED"
    assert setup.hub.mutations == []


@pytest.mark.parametrize(("field", "value", "message"), [
    ("state", "PENDING_HUB_PR_UNVERIFIED", "state"),
    ("github_sha", OTHER_SHA, "github_sha"),
    ("github_workflow_sha", OTHER_SHA, "workflow"),
    ("hf_pr_revision", "refs/pr/99", "PR revision"),
    ("file_count", 99, "file_count"),
    ("file_sha256", {"README.md": "0" * 64}, "file_count"),
    ("release_asset_manifest", [], "asset manifest"),
])
def test_tampered_proposal_fails_before_tag(
    setup: types.SimpleNamespace, field: str, value: object, message: str,
) -> None:
    setup.proposal[field] = value
    finalizer.PROPOSAL_FILE.write_text(json.dumps(setup.proposal), encoding="utf-8")
    with pytest.raises(RuntimeError, match=message):
        finalizer.finalize()
    assert_no_receipt_or_tag(setup)


@pytest.mark.parametrize(("field", "value", "message"), [
    ("status", "open", "merged"),
    ("is_pull_request", False, "pull request"),
    ("target_branch", "refs/heads/dev", "target branch"),
    ("merge_commit_oid", OTHER_SHA, "Hub main"),
    ("events", [], "proposal commit"),
])
def test_hub_pr_or_main_mismatch_fails_before_tag(
    setup: types.SimpleNamespace, field: str, value: object, message: str,
) -> None:
    setattr(setup.hub, field, value)
    with pytest.raises(RuntimeError, match=message):
        finalizer.finalize()
    assert_no_receipt_or_tag(setup)


def test_hub_byte_tamper_fails_before_tag(setup: types.SimpleNamespace) -> None:
    setup.hub.files["LICENSE"] = b"changed\n"
    with pytest.raises(RuntimeError, match="byte mismatch: LICENSE"):
        finalizer.finalize()
    assert_no_receipt_or_tag(setup)


def test_existing_other_tag_never_moves(setup: types.SimpleNamespace) -> None:
    setup.hub.tag = OTHER_SHA
    with pytest.raises(RuntimeError, match="tag differs"):
        finalizer.finalize()
    assert setup.hub.tag == OTHER_SHA
    assert setup.hub.mutations == []
    assert not finalizer.mirror.RECEIPT_FILE.exists()


def test_main_movement_after_tag_creation_emits_no_receipt(setup: types.SimpleNamespace) -> None:
    setup.hub.main_answers = [MERGE_SHA, MERGE_SHA, OTHER_SHA]
    with pytest.raises(RuntimeError, match="main moved during finalization"):
        finalizer.finalize()
    assert setup.hub.tag == MERGE_SHA
    assert setup.hub.mutations == [("create_tag", MERGE_SHA)]
    assert not finalizer.mirror.RECEIPT_FILE.exists()


def test_tag_movement_during_readback_emits_no_receipt(setup: types.SimpleNamespace) -> None:
    setup.hub.tag_answers = [MERGE_SHA, OTHER_SHA]
    with pytest.raises(RuntimeError, match="moved during verification"):
        finalizer.finalize()
    assert not finalizer.mirror.RECEIPT_FILE.exists()

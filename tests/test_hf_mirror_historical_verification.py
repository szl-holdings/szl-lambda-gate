# SPDX-License-Identifier: Apache-2.0
"""Exercise historical verification and protected Hub proposals offline.

The mock holds separate immutable baseline, release, and advanced-main trees.
Historical verification rejects every write. New release tests permit only an
explicit Hub PR proposal and keep main, tags, and measured receipts untouched.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "hf_mirror_release_historical", ROOT / "scripts" / "hf_mirror_release.py",
)
assert spec and spec.loader
mirror = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mirror)

GITHUB_REPO = "szl-holdings/szl-lambda-gate"
HUB_REPO = "SZLHOLDINGS/szl-lambda-gate"
TAG = "v0.2.0"
SOURCE_SHA = "1" * 40
BASELINE_SHA = "2" * 40
RELEASE_SHA = "3" * 40
MAIN_SHA = "4" * 40
WORKFLOW_SHA = "5" * 40
OTHER_SHA = "6" * 40
PR_SHA = "8" * 40
PR_REF = "refs/pr/17"
PR_URL = f"https://huggingface.co/{HUB_REPO}/discussions/17"
METADATA = {
    "library_name": "kernels", "license": "apache-2.0",
    "tags": ["doi:10.5281/zenodo.19944926"],
}


class RevisionNotFoundError(Exception):
    """Offline stand-in for the Hub's missing-revision response."""


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class HistoricalHub:
    """Revision-aware Hub stand-in; only an explicit test can propose a PR."""

    def __init__(self, store: Path, trees: dict[str, dict[str, bytes]]) -> None:
        self.store = store
        self.trees = trees
        self.main = MAIN_SHA
        self.tag = RELEASE_SHA
        self.tag_exists = True
        self.mutations: list[str] = []
        self.info_calls: list[str] = []
        self.main_answers: list[str] = []
        self.tag_answers: list[str] = []
        self.refs_denied = False
        self.tag_error: Exception | None = None
        self.allow_proposal = False
        self.proposal_error: Exception | None = None
        self.corrupt_pr_file: str | None = None

    def repo_info(self, repo: str, *, repo_type: str, revision: str) -> types.SimpleNamespace:
        assert (repo, repo_type) == (HUB_REPO, "model")
        self.info_calls.append(revision)
        if revision == "main":
            resolved = self.main_answers.pop(0) if self.main_answers else self.main
        elif revision == TAG:
            if self.tag_error is not None:
                raise self.tag_error
            if not self.tag_exists:
                raise RevisionNotFoundError("tag does not exist")
            resolved = self.tag_answers.pop(0) if self.tag_answers else self.tag
        elif revision == PR_REF:
            resolved = PR_SHA
        else:
            assert revision in self.trees, "unknown immutable revision"
            resolved = revision
        return types.SimpleNamespace(sha=resolved)

    def tree(self, revision: str) -> dict[str, bytes]:
        resolved = self.main if revision == "main" else self.tag if revision == TAG else PR_SHA if revision == PR_REF else revision
        return self.trees[resolved]

    def list_repo_files(self, repo: str, *, repo_type: str, revision: str) -> list[str]:
        assert (repo, repo_type) == (HUB_REPO, "model")
        return sorted(self.tree(revision))

    def list_repo_refs(self, repo: str, *, repo_type: str) -> types.SimpleNamespace:
        assert (repo, repo_type) == (HUB_REPO, "model")
        if self.refs_denied:
            raise PermissionError("gated repo refuses the refs endpoint")
        tags = [types.SimpleNamespace(name=TAG)] if self.tag_exists else []
        return types.SimpleNamespace(tags=tags)

    def download(self, repo: str, name: str, *, repo_type: str, revision: str, token: str) -> str:
        assert (repo, repo_type, token) == (HUB_REPO, "model", "test-token")
        path = self.store / revision / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.tree(revision)[name])
        return str(path)

    def reject_write(self, operation: str) -> None:
        self.mutations.append(operation)
        raise AssertionError(f"historical verification attempted {operation}")

    def upload_folder(self, *args: object, **kwargs: object) -> types.SimpleNamespace:
        if not self.allow_proposal:
            self.reject_write("upload_folder")
        assert kwargs["repo_id"] == HUB_REPO and kwargs["repo_type"] == "model"
        assert kwargs["token"] == "test-token"
        assert kwargs["parent_commit"] == MAIN_SHA
        assert kwargs["create_pr"] is True
        assert kwargs["folder_path"] == str(mirror.STAGE)
        self.mutations.append("upload_folder_pr")
        if self.proposal_error is not None:
            raise self.proposal_error
        staged = {path.relative_to(mirror.STAGE).as_posix(): path.read_bytes()
                  for path in mirror.STAGE.rglob("*") if path.is_file()}
        self.trees[PR_SHA] = {**self.trees[MAIN_SHA], **staged}
        if self.corrupt_pr_file is not None:
            self.trees[PR_SHA][self.corrupt_pr_file] += b"tampered\n"
        return types.SimpleNamespace(oid=PR_SHA, pr_url=PR_URL, pr_revision=PR_REF)

    def create_tag(self, *args: object, **kwargs: object) -> None:
        self.reject_write("create_tag")

    def delete_tag(self, *args: object, **kwargs: object) -> None:
        self.reject_write("delete_tag")

    def delete_file(self, *args: object, **kwargs: object) -> None:
        self.reject_write("delete_file")


@pytest.fixture
def historical(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> types.SimpleNamespace:
    work = tmp_path / "work"
    (work / ".github").mkdir(parents=True)
    monkeypatch.chdir(work)
    for key, value in {
        "GITHUB_REPOSITORY": GITHUB_REPO, "GITHUB_SHA": WORKFLOW_SHA,
        "SOURCE_GITHUB_SHA": SOURCE_SHA, "RELEASE_TAG": TAG,
        "HF_REPO_ID": HUB_REPO, "HF_REPO_TYPE": "model", "HF_TOKEN": "test-token",
        "HF_MIRROR_OIDC_RESOURCE": HUB_REPO, "HF_AUTH_MODE": "oidc",
        "HF_RECEIPT_RUN": "123456789",
    }.items():
        monkeypatch.setenv(key, value)

    item = {
        "hf_repo_id": HUB_REPO, "repo_type": "model", "oidc_resource": HUB_REPO,
        "hub_pr_required": True,
        "preserve_hub_card": True, "required_card_metadata": {
            "library_name": "kernels", "license": "apache-2.0",
        },
        "required_card_tags": METADATA["tags"], "preserve_hub_paths": ["LICENSE"],
        "replace_hub_paths": ["kernel.py"], "hub_only_paths": ["lambda_gate.py"],
    }
    (work / ".github" / "hf-mirror.json").write_text(
        json.dumps({"targets": [item]}), encoding="utf-8",
    )
    curated = "Curated correction and kernel quickstart.\n"
    baseline_card = (
        curated + "\n" + mirror.BEGIN + "\nPrevious v0.1.0 release.\n" + mirror.END + "\n"
    ).encode()
    rendered = (
        curated + "\n" + mirror.BEGIN + f"\nRelease {TAG} from {SOURCE_SHA}.\n"
        f"| GitHub source | https://github.com/{GITHUB_REPO} |\n"
        f"| Source commit | `{SOURCE_SHA}` |\n"
        f"| Hub revision | `{TAG}` |\n" + mirror.END + "\n"
    ).encode()
    baseline = {
        "README.md": baseline_card, "LICENSE": b"Hub license notice\n",
        "lambda_gate.py": b"Hub-only package\n", "kernel.py": b"previous kernel\n",
    }
    staged = {
        "README.md": b"source README\n", "kernel.py": b"released kernel\n",
        "contract.json": b'{"version": 1}\n', "bom.json": b'{"components": []}\n',
    }
    pinned = {**baseline, **staged, "README.md": rendered}
    advanced = {
        **pinned, "kernel.py": b"later kernel\n", "new-main-only.txt": b"later file\n",
        "README.md": curated.encode() + mirror.BEGIN.encode() + b"\nLater release.\n" + mirror.END.encode(),
    }
    mirror.STAGE.mkdir()
    for name, contents in staged.items():
        (mirror.STAGE / name).write_bytes(contents)
    mirror.RELEASE_FILE.write_text(json.dumps({
        "tagName": TAG, "isDraft": False, "body": "release notes",
        "assets": [{"name": "bom.json", "size": len(staged["bom.json"]),
                    "digest": "sha256:" + digest(staged["bom.json"])}],
    }), encoding="utf-8")
    receipt = {
        "state": "MEASURED", "github_repo": GITHUB_REPO, "github_sha": SOURCE_SHA,
        "github_workflow_sha": "7" * 40, "github_tag": TAG, "auth": "pat",
        "hf_repo_id": HUB_REPO, "hf_repo_type": "model", "hf_oidc_resource": HUB_REPO,
        "hf_main_before": BASELINE_SHA, "hf_revision": RELEASE_SHA,
        "file_count_before": len(baseline), "file_count": len(pinned),
        "verified_file_count": len(pinned), "preserved_hub_files": 2,
        "replaced_hub_files": ["kernel.py"], "release_assets": ["bom.json"],
    }
    mirror.PRIOR_RECEIPT_FILE.parent.mkdir()
    mirror.PRIOR_RECEIPT_FILE.write_text(json.dumps(receipt), encoding="utf-8")
    run = {
        "id": 123456789, "head_sha": receipt["github_workflow_sha"],
        "repository": {"full_name": GITHUB_REPO},
        "head_repository": {"full_name": GITHUB_REPO},
        "path": ".github/workflows/hf-mirror.yml", "event": "workflow_dispatch",
        "conclusion": "success", "head_branch": "main",
    }
    mirror.PRIOR_RUN_FILE.write_text(json.dumps(run), encoding="utf-8")
    hub = HistoricalHub(tmp_path / "hub", {
        BASELINE_SHA: baseline, RELEASE_SHA: pinned, MAIN_SHA: advanced,
    })
    module = types.ModuleType("huggingface_hub")

    def api(*, token: str) -> HistoricalHub:
        assert token == "test-token"
        return hub

    class ModelCard:
        def __init__(self, text: str) -> None:
            self.text = text
            self.data = types.SimpleNamespace(to_dict=lambda: dict(METADATA))

        @classmethod
        def load(cls, path: Path) -> "ModelCard":
            return cls(Path(path).read_text(encoding="utf-8"))

    module.HfApi, module.ModelCard, module.hf_hub_download = api, ModelCard, hub.download
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    errors = types.ModuleType("huggingface_hub.errors")
    errors.RevisionNotFoundError = RevisionNotFoundError
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors)
    return types.SimpleNamespace(hub=hub, item=item, receipt=receipt, run=run, rendered=rendered,
                                 staged=staged, pinned=pinned)


def prepare(historical: types.SimpleNamespace) -> None:
    mirror.preflight()
    if historical.hub.info_calls.count(TAG):
        assert (mirror.STAGE / "README.md").read_bytes() == historical.rendered
    else:
        # A normal publish renders the source card; existing tags still require
        # the explicit historical receipt lane, even with a valid rendered card.
        (mirror.STAGE / "README.md").write_bytes(historical.rendered)


def assert_no_publication(historical: types.SimpleNamespace) -> None:
    assert historical.hub.mutations == []
    assert historical.hub.main == MAIN_SHA
    assert historical.hub.tag == RELEASE_SHA
    assert not mirror.RECEIPT_FILE.exists()


@pytest.mark.parametrize(("key", "value", "message"), [
    ("state", "UNAVAILABLE", "state mismatch"),
    ("github_repo", "other/repository", "github_repo mismatch"),
    ("github_sha", OTHER_SHA, "github_sha mismatch"),
    ("github_workflow_sha", OTHER_SHA, "workflow commit differs from publisher run"),
    ("github_tag", "v9.9.9", "github_tag mismatch"),
    ("hf_repo_id", "OTHER/repository", "hf_repo_id mismatch"),
    ("hf_repo_type", "space", "hf_repo_type mismatch"),
    ("hf_oidc_resource", "OTHER/repository", "hf_oidc_resource mismatch"),
    ("operation", "verify-existing", "must record publication"),
    ("auth", "unknown", "authentication missing"),
    ("hf_main_before", "not-a-sha", "hf_main_before invalid"),
    ("hf_revision", "not-a-sha", "hf_revision invalid"),
    ("release_assets", ["other.json"], "release asset set mismatch"),
])
def test_prior_receipt_mismatches_fail_closed(
    historical: types.SimpleNamespace, key: str, value: object, message: str,
) -> None:
    historical.receipt[key] = value
    mirror.PRIOR_RECEIPT_FILE.write_text(json.dumps(historical.receipt), encoding="utf-8")
    with pytest.raises(RuntimeError, match=message):
        mirror.prior_receipt(historical.item)
    assert historical.hub.info_calls == []
    assert_no_publication(historical)


@pytest.mark.parametrize("run_id", ["", "0", "-1", "1x"])
def test_prior_receipt_requires_valid_run_id(
    historical: types.SimpleNamespace, monkeypatch: pytest.MonkeyPatch, run_id: str,
) -> None:
    monkeypatch.setenv("HF_RECEIPT_RUN", run_id)
    with pytest.raises(RuntimeError, match="successful publisher run ID"):
        mirror.prior_receipt(historical.item)
    assert_no_publication(historical)


@pytest.mark.parametrize(("key", "value", "message"), [
    ("id", 999, "run ID mismatch"),
    ("repository", {"full_name": "other/repository"}, "run repository mismatch"),
    ("head_repository", {"full_name": "other/fork"}, "run repository mismatch"),
    ("path", ".github/workflows/other.yml", "identity or outcome mismatch"),
    ("event", "pull_request", "identity or outcome mismatch"),
    ("conclusion", "failure", "identity or outcome mismatch"),
    ("head_sha", "not-a-sha", "workflow commit differs from publisher run"),
])
def test_retained_receipt_requires_original_successful_publisher_identity(
    historical: types.SimpleNamespace, key: str, value: object, message: str,
) -> None:
    historical.run[key] = value
    mirror.PRIOR_RUN_FILE.write_text(json.dumps(historical.run), encoding="utf-8")
    with pytest.raises(RuntimeError, match=message):
        mirror.prior_receipt(historical.item)
    assert historical.hub.info_calls == []
    assert_no_publication(historical)


def test_existing_historical_tag_verifies_after_main_advanced_without_writes(
    historical: types.SimpleNamespace,
) -> None:
    prepare(historical)
    baseline = json.loads(mirror.BASELINE_FILE.read_text(encoding="utf-8"))
    assert baseline["sha"] == BASELINE_SHA
    assert baseline["main_sha"] == MAIN_SHA
    assert "new-main-only.txt" not in baseline["files"]
    mirror.publish()
    observed = json.loads(mirror.RECEIPT_FILE.read_text(encoding="utf-8"))
    assert observed["operation"] == "verify-existing"
    assert observed["publication_auth"] == "pat"
    assert observed["verification_auth"] == observed["auth"] == "oidc"
    assert observed["hf_revision"] == RELEASE_SHA
    assert observed["hf_main_before"] == BASELINE_SHA
    assert observed["hf_main_observed"] == MAIN_SHA
    assert observed["github_workflow_sha"] == WORKFLOW_SHA
    assert observed["source_receipt_run"] == "123456789"
    assert observed["source_receipt_sha256"] == mirror.sha256(mirror.PRIOR_RECEIPT_FILE)
    assert observed["file_sha256"] == {
        name: digest(data) for name, data in historical.pinned.items()
    }
    assert historical.hub.mutations == []
    assert historical.hub.main == MAIN_SHA
    assert historical.hub.tag == RELEASE_SHA


def test_gated_refs_endpoint_is_not_needed_to_verify_existing_tag(
    historical: types.SimpleNamespace,
) -> None:
    prepare(historical)
    historical.hub.refs_denied = True
    mirror.publish()
    assert json.loads(mirror.RECEIPT_FILE.read_text(encoding="utf-8"))["operation"] == "verify-existing"
    assert historical.hub.mutations == []


def test_gated_refs_endpoint_does_not_block_protected_hub_pr_proposal(
    historical: types.SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_RECEIPT_RUN")
    historical.hub.tag_exists = False
    prepare(historical)
    historical.hub.refs_denied = True
    historical.hub.allow_proposal = True
    assert mirror.publish() == "PENDING_HUB_PR_REVIEW"
    proposal = json.loads(mirror.PROPOSAL_FILE.read_text(encoding="utf-8"))
    assert proposal["state"] == "PENDING_HUB_PR_REVIEW"
    assert proposal["hf_pr_url"] == PR_URL and proposal["hf_pr_revision"] == PR_REF
    assert proposal["hf_pr_commit"] == PR_SHA
    assert proposal["hf_main_before"] == MAIN_SHA
    assert proposal["file_sha256"] == {name: digest(data) for name, data in historical.hub.trees[PR_SHA].items()}
    assert proposal["hub_main_published"] is False and proposal["hub_tag_created"] is False
    assert historical.hub.mutations == ["upload_folder_pr"]
    assert historical.hub.main == MAIN_SHA and historical.hub.tag_exists is False
    assert not mirror.RECEIPT_FILE.exists()


def test_new_release_refuses_direct_upload_without_explicit_hub_pr_policy(
    historical: types.SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_RECEIPT_RUN")
    historical.hub.tag_exists = False
    historical.item.pop("hub_pr_required")
    (Path(".github") / "hf-mirror.json").write_text(
        json.dumps({"targets": [historical.item]}), encoding="utf-8")
    prepare(historical)
    with pytest.raises(RuntimeError, match="Hub PR publication policy is required"):
        mirror.publish()
    assert_no_publication(historical)


def test_unverified_hub_proposal_cannot_emit_publication_receipt(
    historical: types.SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_RECEIPT_RUN")
    historical.hub.tag_exists = False
    prepare(historical)
    historical.hub.allow_proposal = True
    historical.hub.corrupt_pr_file = "lambda_gate.py"
    with pytest.raises(RuntimeError, match="byte mismatch: lambda_gate.py"):
        mirror.publish()
    assert json.loads(mirror.PROPOSAL_FILE.read_text(encoding="utf-8"))["state"] == "PENDING_HUB_PR_UNVERIFIED"
    assert historical.hub.main == MAIN_SHA and historical.hub.tag_exists is False
    assert not mirror.RECEIPT_FILE.exists()


def test_hub_pr_api_failure_cannot_fall_back_to_direct_main_commit(
    historical: types.SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_RECEIPT_RUN")
    historical.hub.tag_exists = False
    prepare(historical)
    historical.hub.allow_proposal = True
    historical.hub.proposal_error = PermissionError("Hub PR denied")
    with pytest.raises(PermissionError, match="Hub PR denied"):
        mirror.publish()
    assert historical.hub.mutations == ["upload_folder_pr"]
    assert historical.hub.main == MAIN_SHA and historical.hub.tag_exists is False
    assert not mirror.PROPOSAL_FILE.exists() and not mirror.RECEIPT_FILE.exists()


def test_tag_lookup_denial_fails_before_upload(
    historical: types.SimpleNamespace,
) -> None:
    prepare(historical)
    historical.hub.tag_error = PermissionError("tag lookup denied")
    with pytest.raises(PermissionError, match="tag lookup denied"):
        mirror.publish()
    assert_no_publication(historical)


@pytest.mark.parametrize("row", [
    f"| GitHub source | https://github.com/{GITHUB_REPO} |",
    f"| Source commit | `{SOURCE_SHA}` |",
    f"| Hub revision | `{TAG}` |",
])
def test_original_card_requires_exact_source_commit_and_tag_binding(
    historical: types.SimpleNamespace, row: str,
) -> None:
    prepare(historical)
    # The introductory sentence still contains the expected SHA and tag. Their
    # appearance elsewhere cannot substitute for the explicit provenance rows.
    card = (mirror.STAGE / "README.md").read_text(encoding="utf-8")
    (mirror.STAGE / "README.md").write_text(card.replace(row, "| modified binding |"), encoding="utf-8")
    with pytest.raises(RuntimeError, match="historical release card source binding differs"):
        mirror.publish()
    assert_no_publication(historical)


def test_wrong_tag_revision_fails_preflight(historical: types.SimpleNamespace) -> None:
    historical.hub.tag_answers = [OTHER_SHA]
    with pytest.raises(RuntimeError, match="tag differs from publication receipt"):
        mirror.preflight()
    assert not mirror.BASELINE_FILE.exists()
    assert_no_publication(historical)


def test_tag_changed_after_preflight_fails(historical: types.SimpleNamespace) -> None:
    prepare(historical)
    historical.hub.tag_answers = [OTHER_SHA]
    with pytest.raises(RuntimeError, match="tag differs from publication receipt"):
        mirror.publish()
    assert_no_publication(historical)


def test_tag_movement_during_byte_verification_fails(historical: types.SimpleNamespace) -> None:
    prepare(historical)
    historical.hub.tag_answers = [RELEASE_SHA, RELEASE_SHA, OTHER_SHA]
    with pytest.raises(RuntimeError, match="moved during verification"):
        mirror.publish()
    assert_no_publication(historical)


def test_main_changed_since_preflight_fails(historical: types.SimpleNamespace) -> None:
    prepare(historical)
    historical.hub.main_answers = [OTHER_SHA]
    with pytest.raises(RuntimeError, match="main changed since baseline"):
        mirror.publish()
    assert_no_publication(historical)


def test_main_movement_during_verification_fails(historical: types.SimpleNamespace) -> None:
    prepare(historical)
    historical.hub.main_answers = [MAIN_SHA, OTHER_SHA]
    with pytest.raises(RuntimeError, match="main moved during release"):
        mirror.publish()
    assert_no_publication(historical)


@pytest.mark.parametrize("name", ["kernel.py", "LICENSE", "README.md", "bom.json"])
def test_historical_release_byte_tamper_fails(
    historical: types.SimpleNamespace, name: str,
) -> None:
    prepare(historical)
    historical.hub.trees[RELEASE_SHA][name] += b"tampered\n"
    with pytest.raises(RuntimeError, match=f"byte mismatch: {name}"):
        mirror.publish()
    assert_no_publication(historical)


def test_historical_release_extra_file_fails(historical: types.SimpleNamespace) -> None:
    prepare(historical)
    historical.hub.trees[RELEASE_SHA]["unexpected.txt"] = b"unexpected\n"
    with pytest.raises(RuntimeError, match="file set differs"):
        mirror.publish()
    assert_no_publication(historical)


@pytest.mark.parametrize("key", [
    "file_count_before", "file_count", "verified_file_count", "preserved_hub_files",
])
def test_original_publication_counts_must_match_reconstructed_union(
    historical: types.SimpleNamespace, key: str,
) -> None:
    historical.receipt[key] += 1
    mirror.PRIOR_RECEIPT_FILE.write_text(json.dumps(historical.receipt), encoding="utf-8")
    prepare(historical)
    with pytest.raises(RuntimeError, match=f"historical publication {key} mismatch"):
        mirror.publish()
    assert_no_publication(historical)


def test_receipt_changed_after_preflight_fails(historical: types.SimpleNamespace) -> None:
    prepare(historical)
    historical.receipt["file_count"] += 1
    mirror.PRIOR_RECEIPT_FILE.write_text(json.dumps(historical.receipt), encoding="utf-8")
    with pytest.raises(RuntimeError, match="receipt changed since baseline"):
        mirror.publish()
    assert_no_publication(historical)


def test_missing_historical_tag_cannot_trigger_upload(historical: types.SimpleNamespace) -> None:
    prepare(historical)
    historical.hub.tag_exists = False
    with pytest.raises(RuntimeError, match="historical Hub tag missing; verification cannot upload"):
        mirror.publish()
    assert_no_publication(historical)


def test_existing_tag_without_publication_receipt_cannot_be_republished(
    historical: types.SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_RECEIPT_RUN")
    prepare(historical)
    with pytest.raises(RuntimeError, match="existing Hub tag requires receipt_run"):
        mirror.publish()
    assert_no_publication(historical)


def test_missing_retained_receipt_fails_preflight(historical: types.SimpleNamespace) -> None:
    mirror.PRIOR_RECEIPT_FILE.unlink()
    with pytest.raises(RuntimeError, match="retained publication receipt missing"):
        mirror.preflight()
    assert not mirror.BASELINE_FILE.exists()
    assert_no_publication(historical)

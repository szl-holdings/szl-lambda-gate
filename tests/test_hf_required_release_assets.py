# SPDX-License-Identifier: Apache-2.0
"""Require complete release evidence before staging a new Hub publication."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "hf_mirror_required_release_assets", ROOT / "scripts" / "hf_mirror_release.py",
)
assert spec and spec.loader
mirror = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mirror)

HUB_REPO = "SZLHOLDINGS/szl-lambda-gate"
REQUIRED = ["szl-lambda-gate-sbom-2.spdx.json", "szl-lambda-gate-sbom.cyclonedx.json"]
MISSING = object()


@pytest.fixture
def release_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HF_REPO_ID", HUB_REPO)
    monkeypatch.setenv("HF_REPO_TYPE", "model")
    monkeypatch.setenv("RELEASE_TAG", "v0.2.1")
    monkeypatch.delenv("HF_RECEIPT_RUN", raising=False)
    (tmp_path / ".github").mkdir()
    mirror.STAGE.mkdir()
    mirror.ASSETS_DIR.mkdir()

    def configure(required: object = REQUIRED, assets: dict[str, bytes] | None = None) -> dict:
        item = {"hf_repo_id": HUB_REPO, "repo_type": "model", "oidc_resource": HUB_REPO}
        if required is not MISSING:
            item["required_release_assets"] = required
        (tmp_path / ".github" / "hf-mirror.json").write_text(
            json.dumps({"targets": [item]}), encoding="utf-8",
        )
        rows = []
        for name, data in (assets or {}).items():
            (mirror.ASSETS_DIR / name).write_bytes(data)
            rows.append({"name": name, "size": len(data),
                         "digest": "sha256:" + hashlib.sha256(data).hexdigest()})
        mirror.RELEASE_FILE.write_text(json.dumps({
            "tagName": "v0.2.1", "isDraft": False, "assets": rows,
        }), encoding="utf-8")
        return item

    return configure


@pytest.mark.parametrize("available", [[], REQUIRED[:1]])
def test_missing_required_assets_fail_before_any_copy(release_workspace, available: list[str]) -> None:
    release_workspace(assets={name: b"release evidence" for name in available})
    with pytest.raises(RuntimeError, match="required release assets missing"):
        mirror.assets()
    assert list(mirror.STAGE.iterdir()) == []


def test_complete_required_asset_set_is_verified_and_staged(release_workspace) -> None:
    assets = {REQUIRED[0]: b"SPDX evidence", REQUIRED[1]: b"CycloneDX evidence", "notes.txt": b"notes"}
    release_workspace(assets=assets)
    mirror.assets()
    assert {path.name: path.read_bytes() for path in mirror.STAGE.iterdir()} == assets


@pytest.mark.parametrize("required", [MISSING, []])
def test_other_target_maps_may_omit_asset_policy(release_workspace, required: object) -> None:
    release_workspace(required=required, assets={"old-evidence.json": b"original evidence"})
    mirror.assets()
    assert (mirror.STAGE / "old-evidence.json").read_bytes() == b"original evidence"


@pytest.mark.parametrize("required", [
    None, "evidence.json", [1], ["evidence.json", "evidence.json"],
    ["../evidence.json"], ["folder/evidence.json"], ["/evidence.json"],
    ["folder\\evidence.json"], ["C:evidence.json"], [""], ["bad\x00name"],
])
def test_invalid_required_asset_policy_fails_closed(release_workspace, required: object) -> None:
    release_workspace(required=required)
    with pytest.raises(RuntimeError, match="required_release_assets|unsafe|duplicate required"):
        mirror.assets()
    assert list(mirror.STAGE.iterdir()) == []


def test_asset_policy_comes_from_exact_target_map(
    release_workspace, monkeypatch: pytest.MonkeyPatch,
) -> None:
    release_workspace(assets={name: b"evidence" for name in REQUIRED})
    monkeypatch.setenv("HF_REPO_ID", "OTHER/repository")
    with pytest.raises(RuntimeError, match="exactly one target"):
        mirror.assets()
    assert list(mirror.STAGE.iterdir()) == []


def test_historical_verification_uses_original_receipt_asset_set(
    release_workspace, monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = release_workspace(assets={"original-bom.json": b"original evidence"})
    monkeypatch.setenv("HF_RECEIPT_RUN", "12345")
    monkeypatch.setenv("GITHUB_REPOSITORY", "szl-holdings/szl-lambda-gate")
    monkeypatch.setenv("SOURCE_GITHUB_SHA", "1" * 40)
    receipt = {
        "state": "MEASURED", "github_repo": "szl-holdings/szl-lambda-gate",
        "github_sha": "1" * 40, "github_workflow_sha": "2" * 40,
        "github_tag": "v0.2.1", "hf_repo_id": HUB_REPO, "hf_repo_type": "model",
        "hf_oidc_resource": HUB_REPO, "auth": "pat", "hf_main_before": "3" * 40,
        "hf_revision": "4" * 40, "release_assets": ["original-bom.json"],
    }
    run = {
        "id": 12345, "head_sha": "2" * 40, "repository": {"full_name": receipt["github_repo"]},
        "head_repository": {"full_name": receipt["github_repo"]},
        "path": ".github/workflows/hf-mirror.yml", "event": "workflow_dispatch", "conclusion": "success",
    }
    mirror.PRIOR_RECEIPT_FILE.parent.mkdir()
    mirror.PRIOR_RECEIPT_FILE.write_text(json.dumps(receipt), encoding="utf-8")
    mirror.PRIOR_RUN_FILE.write_text(json.dumps(run), encoding="utf-8")
    mirror.assets()
    assert (mirror.STAGE / "original-bom.json").read_bytes() == b"original evidence"
    assert mirror.prior_receipt(item) == receipt

    receipt["release_assets"] = REQUIRED
    mirror.PRIOR_RECEIPT_FILE.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(RuntimeError, match="prior publication release asset set mismatch"):
        mirror.prior_receipt(item)


def test_historical_verification_still_rejects_invalid_asset_policy(
    release_workspace, monkeypatch: pytest.MonkeyPatch,
) -> None:
    release_workspace(required=["../bom.json"], assets={"original-bom.json": b"evidence"})
    monkeypatch.setenv("HF_RECEIPT_RUN", "12345")
    with pytest.raises(RuntimeError, match="unsafe"):
        mirror.assets()
    assert list(mirror.STAGE.iterdir()) == []

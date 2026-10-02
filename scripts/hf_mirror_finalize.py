#!/usr/bin/env python3
"""Finalize one reviewed Hub PR without moving an existing release tag.

The workflow runs this from reviewed GitHub main. The proposal artifact and
release-tag checkout are evidence and payload only, never executable code.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re

import hf_mirror_release as mirror


PROPOSAL_FILE = Path(".hfmirror-proposal.json")
PROPOSAL_RUN_FILE = Path(".hfmirror-proposal-run.json")
SHA = re.compile(r"[0-9a-f]{40}\Z")
DIGEST = re.compile(r"[0-9a-f]{64}\Z")
RUN_ID = re.compile(r"[1-9][0-9]*\Z")


def field(obj: object, key: str) -> object:
    """Read the SDK's discussion/event objects without assuming dict or class."""
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def proposal_evidence(item: dict) -> dict:
    mirror.require(item.get("hub_pr_required") is True, "protected Hub PR policy is required")
    mirror.require(os.environ.get("HF_AUTH_MODE") == "pat", "finalization requires explicit owner PAT")
    run_id = os.environ.get("HF_PROPOSAL_RUN", "")
    mirror.require(RUN_ID.fullmatch(run_id) is not None, "valid proposal run ID required")
    proposal = json.loads(PROPOSAL_FILE.read_text(encoding="utf-8"))
    run = json.loads(PROPOSAL_RUN_FILE.read_text(encoding="utf-8"))
    mirror.require(str(run.get("id")) == run_id, "proposal run ID mismatch")
    mirror.require(
        run.get("repository", {}).get("full_name") == os.environ["GITHUB_REPOSITORY"]
        and run.get("head_repository", {}).get("full_name") == os.environ["GITHUB_REPOSITORY"],
        "proposal run repository mismatch",
    )
    mirror.require(
        run.get("path") == ".github/workflows/hf-mirror.yml"
        and run.get("event") == "workflow_dispatch"
        and run.get("head_branch") == os.environ["DEFAULT_BRANCH"]
        and run.get("conclusion") == "failure",
        "proposal run identity or outcome mismatch",
    )
    mirror.require(SHA.fullmatch(run.get("head_sha", "")) is not None,
                   "proposal workflow commit invalid")
    expected = {
        "state": "PENDING_HUB_PR_REVIEW", "operation": "propose",
        "github_repo": os.environ["GITHUB_REPOSITORY"],
        "github_sha": os.environ["SOURCE_GITHUB_SHA"],
        "github_workflow_sha": run["head_sha"],
        "github_tag": os.environ["RELEASE_TAG"],
        "hf_repo_id": os.environ["HF_REPO_ID"],
        "hf_repo_type": os.environ["HF_REPO_TYPE"],
        "hf_oidc_resource": item["oidc_resource"],
        "hub_main_published": False, "hub_tag_created": False,
    }
    for key, value in expected.items():
        mirror.require(proposal.get(key) == value, f"proposal {key} mismatch")
    mirror.require(proposal.get("auth") in ("oidc", "pat"),
                   "proposal authentication mode invalid")
    mirror.require(SHA.fullmatch(proposal.get("hf_main_before", "")) is not None,
                   "proposal Hub baseline invalid")
    mirror.require(SHA.fullmatch(proposal.get("hf_pr_commit", "")) is not None,
                   "proposal commit invalid")
    match = re.fullmatch(
        rf"https://huggingface\.co/{re.escape(os.environ['HF_REPO_ID'])}/discussions/([1-9][0-9]*)",
        proposal.get("hf_pr_url", ""),
    )
    mirror.require(match is not None, "proposal PR URL invalid")
    mirror.require(proposal.get("hf_pr_revision") == f"refs/pr/{match.group(1)}",
                   "proposal PR revision mismatch")
    proposal["pr_number"] = int(match.group(1))
    hashes = proposal.get("file_sha256")
    mirror.require(isinstance(hashes, dict) and bool(hashes), "proposal file hashes missing")
    mirror.require(proposal.get("file_count") == len(hashes), "proposal file_count mismatch")
    for name, digest in hashes.items():
        mirror.require(isinstance(name, str) and isinstance(digest, str),
                       "proposal file hash entry invalid")
        mirror.safe_relative(name)
        mirror.require(DIGEST.fullmatch(digest) is not None,
                       f"proposal file hash invalid: {name}")
    mirror.require("README.md" in hashes, "proposal card missing")
    staged = proposal.get("staged_files")
    mirror.require(isinstance(staged, list) and all(isinstance(name, str) for name in staged)
                   and staged == sorted(set(staged)) and all(name in hashes for name in staged),
                   "proposal staged file set invalid")
    mirror.require("README.md" in staged, "proposal rendered card missing")
    replaced = proposal.get("replaced_hub_files")
    mirror.require(isinstance(replaced, list) and all(isinstance(name, str) for name in replaced)
                   and replaced == sorted(set(replaced)) and all(name in staged for name in replaced),
                   "proposal replacement list invalid")
    mirror.require(set(replaced).issubset(item.get("replace_hub_paths", [])),
                   "proposal replacement exceeds reviewed paths")
    release_assets = mirror.release()["assets"]
    assets = sorted(asset["name"] for asset in release_assets)
    mirror.require(proposal.get("release_assets") == assets,
                   "proposal release assets mismatch")
    manifest = sorted(
        ({"name": asset["name"], "size": asset["size"], "digest": asset["digest"]}
         for asset in release_assets), key=lambda asset: asset["name"]
    )
    mirror.require(proposal.get("release_asset_manifest") == manifest,
                   "proposal release asset manifest mismatch")
    mirror.require(len(assets) == len(set(assets)) and all(
        isinstance(asset["size"], int) and asset["size"] >= 0
        and re.fullmatch(r"sha256:[0-9a-f]{64}", asset["digest"]) is not None
        for asset in manifest
    ), "release asset manifest invalid")
    mirror.require(set(item.get("required_release_assets", [])).issubset(assets),
                   "required release assets missing")
    mirror.require(set(assets).issubset(staged), "proposal release assets not staged")
    return proposal


def finalize() -> str:
    from huggingface_hub import HfApi
    from huggingface_hub.errors import RevisionNotFoundError

    item = mirror.target()
    proposal = proposal_evidence(item)
    repo, repo_type, tag, token = (
        os.environ[key] for key in ("HF_REPO_ID", "HF_REPO_TYPE", "RELEASE_TAG", "HF_TOKEN")
    )
    api = HfApi(token=token)
    details = api.get_discussion_details(repo, proposal["pr_number"], repo_type=repo_type)
    mirror.require(field(details, "num") == proposal["pr_number"], "Hub PR number mismatch")
    mirror.require(field(details, "repo_id") == repo and field(details, "repo_type") == repo_type,
                   "Hub PR repository mismatch")
    mirror.require(field(details, "is_pull_request") is True, "Hub discussion is not a pull request")
    mirror.require(field(details, "status") == "merged", "Hub PR is not merged")
    mirror.require(field(details, "target_branch") == "refs/heads/main", "Hub PR target branch mismatch")
    merged_oid = field(details, "merge_commit_oid")
    mirror.require(isinstance(merged_oid, str) and SHA.fullmatch(merged_oid) is not None,
                   "Hub PR merge commit invalid")
    events = field(details, "events") or []
    mirror.require(any(field(event, "oid") == proposal["hf_pr_commit"] for event in events),
                   "Hub PR does not contain proposal commit")
    mirror.require(api.repo_info(repo, repo_type=repo_type, revision="main").sha == merged_oid,
                   "Hub main differs from merged PR commit")
    mirror.require(merged_oid != proposal["hf_main_before"], "Hub PR did not advance main")

    before = proposal["hf_main_before"]
    mirror.require(api.repo_info(repo, repo_type=repo_type, revision=before).sha == before,
                   "proposal Hub baseline revision unavailable")
    before_files = set(api.list_repo_files(repo, repo_type=repo_type, revision=before))
    staged = set(proposal["staged_files"])
    hashes = proposal["file_sha256"]
    mirror.require(proposal.get("file_count_before") == len(before_files),
                   "proposal file_count_before mismatch")
    mirror.require(set(hashes) == before_files | staged, "proposal file set differs from additive union")
    mirror.require(proposal.get("preserved_hub_files") == len(before_files - staged),
                   "proposal preserved_hub_files mismatch")
    mirror.require(not (set(item.get("preserve_hub_paths", [])) & staged),
                   "proposal staged a preserved Hub path")
    changed = sorted(
        name for name in before_files & staged if name != "README.md"
        and mirror.sha256(mirror.hub_file(repo, repo_type, before, name, token)) != hashes[name]
    )
    mirror.require(changed == proposal["replaced_hub_files"],
                   "proposal replacement list differs from Hub baseline")
    for name in before_files - staged:
        mirror.require(mirror.sha256(mirror.hub_file(repo, repo_type, before, name, token)) == hashes[name],
                       f"preserved Hub file changed: {name}")

    mirror.verify_revision(api, item, merged_oid, merged_oid, set(hashes), hashes, token)
    mirror.require(api.repo_info(repo, repo_type=repo_type, revision="main").sha == merged_oid,
                   "Hub main moved before release tagging")
    try:
        tagged = api.repo_info(repo, repo_type=repo_type, revision=tag)
    except RevisionNotFoundError:
        tagged = None
    if tagged is None:
        api.create_tag(repo, tag=tag, repo_type=repo_type, revision=merged_oid,
                       token=token, exist_ok=False)
    else:
        mirror.require(tagged.sha == merged_oid, "existing Hub tag differs from merged PR")
    mirror.verify_revision(api, item, tag, merged_oid, set(hashes), hashes, token)
    mirror.require(api.repo_info(repo, repo_type=repo_type, revision="main").sha == merged_oid,
                   "Hub main moved during finalization")

    receipt = {
        "state": "MEASURED", "operation": "publish",
        "github_repo": os.environ["GITHUB_REPOSITORY"],
        "github_sha": os.environ["SOURCE_GITHUB_SHA"],
        "github_workflow_sha": os.environ["GITHUB_SHA"],
        "github_tag": tag, "hf_repo_id": repo, "hf_repo_type": repo_type,
        "hf_oidc_resource": item["oidc_resource"],
        "hf_main_before": before, "hf_revision": merged_oid,
        "hf_main_observed": merged_oid,
        "file_count_before": len(before_files), "file_count": len(hashes),
        "verified_file_count": len(hashes),
        "release_assets": proposal["release_assets"],
        "preserved_hub_files": proposal["preserved_hub_files"],
        "replaced_hub_files": proposal["replaced_hub_files"],
        "file_sha256": hashes,
        "auth": "pat", "publication_auth": "pat", "verification_auth": "pat",
        "source_proposal_run": os.environ["HF_PROPOSAL_RUN"],
        "source_proposal_sha256": mirror.sha256(PROPOSAL_FILE),
        "hf_pr_url": proposal["hf_pr_url"],
        "hf_pr_commit": proposal["hf_pr_commit"],
        "hf_pr_merge_commit": merged_oid,
    }
    mirror.RECEIPT_FILE.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt["state"]


if __name__ == "__main__":
    finalize()

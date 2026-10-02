# SPDX-License-Identifier: Apache-2.0
"""Wait for the release's SBOM producer before dispatching its Hub publisher.

Asset existence is insufficient: the SBOM release job replaces existing assets.
This gate first requires the exact release-event workflow run to succeed, then
checks required asset metadata and confirms that the producer has not restarted.
The publisher separately checks downloaded bytes against those asset digests.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import time
from typing import Any, Callable
from urllib.parse import quote

WORKFLOW_PATH = ".github/workflows/sbom.yml"


class CoordinationError(RuntimeError):
    """A release cannot safely proceed to the publisher."""


def timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise CoordinationError(f"{label} is absent or invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise CoordinationError(f"{label} is invalid") from None
    if parsed.tzinfo is None:
        raise CoordinationError(f"{label} must include a timezone")
    return parsed


def required_asset_names(config: dict[str, Any]) -> list[str]:
    targets = config.get("targets") if isinstance(config, dict) else None
    if not isinstance(targets, list) or not targets:
        raise CoordinationError("mirror targets are absent or invalid")
    required: set[str] = set()
    for target in targets:
        names = target.get("required_release_assets") if isinstance(target, dict) else None
        if not isinstance(names, list) or not names:
            raise CoordinationError("required_release_assets must be a nonempty list for every target")
        if any(not isinstance(name, str) or not name or PurePosixPath(name).name != name
               or name in (".", "..")
               or any(char in "\\:" or ord(char) < 32 or ord(char) == 127 for char in name)
               for name in names):
            raise CoordinationError("required release asset name is unsafe")
        if len(names) != len(set(names)):
            raise CoordinationError("required release asset names are duplicated")
        required.update(names)
    return sorted(required)


def latest_producer(payload: Any, *, repo: str, tag: str, source_sha: str,
                    published_at: datetime) -> dict[str, Any] | None:
    runs = payload.get("workflow_runs") if isinstance(payload, dict) else None
    if not isinstance(runs, list):
        raise CoordinationError("SBOM run listing is invalid")
    candidates = []
    for run in runs:
        if not isinstance(run, dict):
            raise CoordinationError("SBOM run listing contains an invalid run")
        if (run.get("event"), run.get("head_branch"), run.get("path")) != (
                "release", tag, WORKFLOW_PATH):
            continue
        created = timestamp(run.get("created_at"), "SBOM run created_at")
        if created < published_at:
            continue
        repository = run.get("repository")
        head_repository = run.get("head_repository")
        if (not isinstance(repository, dict) or repository.get("full_name") != repo
                or not isinstance(head_repository, dict) or head_repository.get("full_name") != repo
                or run.get("head_sha") != source_sha):
            raise CoordinationError("SBOM producer repository or source SHA mismatch")
        if (type(run.get("id")) is not int or run["id"] <= 0
                or type(run.get("run_attempt")) is not int or run["run_attempt"] <= 0):
            raise CoordinationError("SBOM producer run identity is invalid")
        if run.get("status") not in ("queued", "in_progress", "waiting", "pending", "requested", "completed"):
            raise CoordinationError("SBOM producer status is invalid")
        candidates.append((created, run["id"], run))
    return max(candidates, key=lambda item: item[:2])[2] if candidates else None


def checked_release_assets(release: Any, *, tag: str, published_at: datetime,
                           required: list[str]) -> dict[str, dict[str, Any]]:
    if (not isinstance(release, dict) or release.get("tag_name") != tag
            or release.get("draft") is not False
            or timestamp(release.get("published_at"), "release published_at") != published_at):
        raise CoordinationError("published release identity mismatch")
    assets = release.get("assets")
    if not isinstance(assets, list):
        raise CoordinationError("release asset listing is invalid")
    selected: dict[str, dict[str, Any]] = {}
    for name in required:
        matching = [asset for asset in assets if isinstance(asset, dict) and asset.get("name") == name]
        if len(matching) != 1:
            raise CoordinationError(f"required release asset is missing or duplicated: {name}")
        asset = matching[0]
        if (asset.get("state") != "uploaded" or type(asset.get("size")) is not int or asset["size"] <= 0
                or not isinstance(asset.get("digest"), str)
                or re.fullmatch(r"sha256:[0-9a-f]{64}", asset["digest"]) is None):
            raise CoordinationError(f"required release asset lacks uploaded bytes and SHA256 digest: {name}")
        selected[name] = {key: asset.get(key) for key in ("id", "name", "size", "digest")}
    return selected


def gh_json(endpoint: str, timeout_seconds: float) -> Any:
    # Only gh's GitHub credential is needed. Never forward a Hub credential.
    env = dict(os.environ)
    for key in ("HF_TOKEN", "HF_FALLBACK_TOKEN", "HF_OIDC_RESOURCE"):
        env.pop(key, None)
    try:
        result = subprocess.run(["gh", "api", endpoint], shell=False, capture_output=True,
                                encoding="utf-8", timeout=timeout_seconds, env=env, check=False)
    except (OSError, subprocess.TimeoutExpired):
        raise CoordinationError("GitHub API request failed or timed out") from None
    if result.returncode != 0:
        # Do not echo arbitrary CLI stderr or credentials into Actions logs.
        raise CoordinationError("GitHub API request failed")
    try:
        return json.loads(result.stdout)
    except (TypeError, ValueError):
        raise CoordinationError("GitHub API response is not JSON") from None


def wait_for_sbom(*, repo: str, tag: str, source_sha: str, published_at: str,
                  required: list[str], timeout_seconds: float = 900, poll_seconds: float = 10,
                  fetch: Callable[[str, float], Any] = gh_json,
                  monotonic: Callable[[], float] = time.monotonic,
                  sleep: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    if re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo) is None:
        raise CoordinationError("GitHub repository is invalid")
    if re.fullmatch(r"[0-9a-f]{40}", source_sha) is None:
        raise CoordinationError("release source SHA is invalid")
    if not tag or any(ord(char) < 33 or char == "\\" for char in tag):
        raise CoordinationError("release tag is invalid")
    if (not math.isfinite(timeout_seconds) or not math.isfinite(poll_seconds)
            or timeout_seconds <= 0 or poll_seconds <= 0 or not required):
        raise CoordinationError("wait bounds or required assets are invalid")
    lower_bound = timestamp(published_at, "release published_at")
    identity = dict(repo=repo, tag=tag, source_sha=source_sha, published_at=lower_bound)
    runs_endpoint = (f"repos/{repo}/actions/workflows/sbom.yml/runs"
                     f"?event=release&branch={quote(tag, safe='')}&per_page=100")
    deadline = monotonic() + timeout_seconds

    def get(endpoint: str) -> Any:
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise CoordinationError("timed out waiting for the exact successful release SBOM producer")
        return fetch(endpoint, min(remaining, 30))

    while True:
        producer = latest_producer(get(runs_endpoint), **identity)
        if producer is not None and producer["status"] == "completed":
            if producer.get("conclusion") != "success":
                raise CoordinationError(f"release SBOM producer {producer['id']} did not succeed")
            release = get(f"repos/{repo}/releases/tags/{quote(tag, safe='')}")
            assets = checked_release_assets(release, tag=tag, published_at=lower_bound, required=required)
            after = latest_producer(get(runs_endpoint), **identity)
            if (after is None or after["id"] != producer["id"]
                    or after["run_attempt"] != producer["run_attempt"]
                    or after["status"] != "completed" or after.get("conclusion") != "success"):
                raise CoordinationError("release SBOM producer changed during the asset snapshot")
            return {
                "state": "READY", "github_repo": repo, "github_tag": tag,
                "github_sha": source_sha, "release_published_at": published_at,
                "sbom_workflow": WORKFLOW_PATH, "sbom_run_id": producer["id"],
                "sbom_run_attempt": producer["run_attempt"], "assets": assets,
            }
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise CoordinationError("timed out waiting for the exact successful release SBOM producer")
        sleep(min(poll_seconds, remaining))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout-seconds", type=float, default=900)
    parser.add_argument("--poll-seconds", type=float, default=10)
    args = parser.parse_args()
    try:
        if not (os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")):
            raise CoordinationError("GitHub workflow credential is absent")
        config = json.loads(Path(".github/hf-mirror.json").read_text(encoding="utf-8"))
        receipt = wait_for_sbom(repo=os.environ.get("GITHUB_REPOSITORY", ""),
                                tag=os.environ.get("RELEASE_TAG", ""),
                                source_sha=os.environ.get("SOURCE_RELEASE_SHA", ""),
                                published_at=os.environ.get("RELEASE_PUBLISHED_AT", ""),
                                required=required_asset_names(config),
                                timeout_seconds=args.timeout_seconds, poll_seconds=args.poll_seconds)
        Path(".hfmirror-sbom-ready.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        print(f"release SBOM producer {receipt['sbom_run_id']} succeeded; "
              f"required assets verified: {', '.join(receipt['assets'])}")
    except (CoordinationError, OSError, ValueError) as exc:
        # Configuration/IO diagnostics contain no subprocess stderr or credentials.
        raise SystemExit(f"UNAVAILABLE - {exc}") from None


if __name__ == "__main__":
    main()

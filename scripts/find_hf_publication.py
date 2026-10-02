#!/usr/bin/env python3
"""Find a measured Hub publication receipt for the latest release audit.

The selected run is only a candidate. The dispatched mirror workflow rechecks
its GitHub run, release assets, receipt, Hub tag, and every Hub file byte.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Callable
from urllib.parse import quote


def gh(*args: str) -> str:
    result = subprocess.run(["gh", *args], text=True, capture_output=True)
    if result.returncode:
        reason = "API rate limit" if "API rate limit exceeded" in result.stderr else f"exit {result.returncode}"
        raise RuntimeError(f"GitHub {args[0]} unavailable: {reason}")
    return result.stdout


def api(path: str) -> dict:
    return json.loads(gh("api", path))


def artifacts(repo: str, name: str) -> list[dict]:
    found: list[dict] = []
    for page in range(1, 11):
        path = f"repos/{repo}/actions/artifacts?name={quote(name, safe='')}&per_page=100&page={page}"
        data = api(path)
        rows = data.get("artifacts")
        if not isinstance(rows, list) or not isinstance(data.get("total_count"), int):
            raise RuntimeError("GitHub artifact listing is malformed")
        found.extend(rows)
        if len(found) >= data["total_count"]:
            return found
    raise RuntimeError("more than 1000 matching artifacts; receipt discovery is incomplete")


def valid_run(data: dict, repo: str, branch: str, ident: int) -> bool:
    return (
        data.get("id") == ident
        and data.get("path") == ".github/workflows/hf-mirror.yml"
        and data.get("event") == "workflow_dispatch"
        and data.get("conclusion") == "success"
        and data.get("head_branch") == branch
        and re.fullmatch(r"[0-9a-f]{40}", data.get("head_sha", "")) is not None
        and data.get("repository", {}).get("full_name") == repo
        and data.get("head_repository", {}).get("full_name") == repo
    )


def valid_receipt(data: dict, run: dict, repo: str, target: str, tag: str) -> bool:
    hashes = data.get("file_sha256")
    return (
        data.get("state") == "MEASURED"
        and data.get("operation") == "publish"
        and data.get("github_repo") == repo
        and data.get("github_tag") == tag
        and data.get("hf_repo_id") == target
        and data.get("github_workflow_sha") == run["head_sha"]
        and re.fullmatch(r"[0-9a-f]{40}", data.get("github_sha", "")) is not None
        and re.fullmatch(r"[0-9a-f]{40}", data.get("hf_revision", "")) is not None
        and data.get("auth") in ("oidc", "pat")
        and isinstance(hashes, dict) and bool(hashes)
        and data.get("verified_file_count") == len(hashes)
        and all(isinstance(name, str) and re.fullmatch(r"[0-9a-f]{64}", digest or "")
                for name, digest in hashes.items())
    )


def select_publication(
    rows: list[dict], get_run: Callable[[int], dict], get_receipt: Callable[[int], dict],
    *, repo: str, target: str, tag: str, slug: str, branch: str,
) -> int:
    name = f"hf-mirror-receipt-{slug}-{tag}"
    for row in sorted(rows, key=lambda item: item.get("created_at", ""), reverse=True):
        workflow = row.get("workflow_run") or {}
        ident = workflow.get("id")
        if row.get("name") != name or row.get("expired") is not False:
            continue
        if not isinstance(ident, int) or ident <= 0 or workflow.get("head_branch") != branch:
            continue
        run = get_run(ident)
        if not valid_run(run, repo, branch, ident) or workflow.get("head_sha") != run["head_sha"]:
            continue
        receipt = get_receipt(ident)
        if valid_receipt(receipt, run, repo, target, tag):
            return ident
    raise RuntimeError(f"no measured publication receipt for {target}@{tag}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--branch", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.repo):
        raise SystemExit("invalid GitHub repository")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.target):
        raise SystemExit("invalid Hub target")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.tag) or not re.fullmatch(r"[a-z0-9-]+", args.slug):
        raise SystemExit("invalid tag or target slug")
    name = f"hf-mirror-receipt-{args.slug}-{args.tag}"

    def download_receipt(ident: int) -> dict:
        with tempfile.TemporaryDirectory(prefix="hf-mirror-audit-") as directory:
            gh("run", "download", str(ident), "--repo", args.repo, "--name", name,
               "--dir", directory)
            path = Path(directory) / "hf-mirror-receipt.json"
            return json.loads(path.read_text(encoding="utf-8"))

    ident = select_publication(
        artifacts(args.repo, name),
        lambda run_id: api(f"repos/{args.repo}/actions/runs/{run_id}"),
        download_receipt,
        repo=args.repo, target=args.target, tag=args.tag, slug=args.slug, branch=args.branch,
    )
    print(json.dumps({"tag": args.tag, "receipt_run": ident, "target": args.target}))


if __name__ == "__main__":
    main()

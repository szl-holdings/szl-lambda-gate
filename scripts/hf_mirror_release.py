#!/usr/bin/env python3
"""Publish an additive, source-bound Hugging Face release with a verified receipt.

This script runs from the reviewed default-branch checkout. The release-tag
checkout is payload data only; its scripts are never executed by the publisher.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys

STAGE = Path(".hfstage")
BASELINE_DIR = Path(".hfmirror-baseline")
BASELINE_FILE = BASELINE_DIR / "evidence.json"
RELEASE_FILE = Path(".hfmirror-release.json")
ASSETS_DIR = Path(".hfmirror-assets")
RECEIPT_FILE = Path("hf-mirror-receipt.json")
BEGIN = "<!-- SZL-HF-MIRROR:START -->"
END = "<!-- SZL-HF-MIRROR:END -->"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"UNAVAILABLE: {message}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def target() -> dict:
    mapping = json.loads(Path(".github/hf-mirror.json").read_text(encoding="utf-8"))
    matching = [item for item in mapping["targets"] if item["hf_repo_id"] == os.environ["HF_REPO_ID"]]
    require(len(matching) == 1, "exactly one target must match HF_REPO_ID")
    item = matching[0]
    require(item["repo_type"] == os.environ["HF_REPO_TYPE"], "target repository type mismatch")
    return item


def safe_relative(name: str) -> Path:
    parsed = PurePosixPath(name)
    require(bool(parsed.parts) and parsed.as_posix() == name and not parsed.is_absolute()
            and "\\" not in name, f"unsafe path: {name!r}")
    require(all(part not in ("", ".", "..") for part in parsed.parts), f"unsafe path: {name!r}")
    return Path(*parsed.parts)


def stage_files() -> dict[str, Path]:
    require(STAGE.is_dir(), "staged payload missing")
    files: dict[str, Path] = {}
    for path in STAGE.rglob("*"):
        require(not path.is_symlink(), f"symlink in staged payload: {path}")
        if path.is_file():
            rel = path.relative_to(STAGE).as_posix()
            safe_relative(rel)
            files[rel] = path
    return files


def release() -> dict:
    data = json.loads(RELEASE_FILE.read_text(encoding="utf-8"))
    require(data["tagName"] == os.environ["RELEASE_TAG"] and not data["isDraft"], "release/tag mismatch")
    return data


def stage() -> None:
    item = target()
    for name in item.get("preserve_hub_paths", []):
        rel = safe_relative(name)
        path = STAGE / rel
        if path.exists() or path.is_symlink():
            require(path.is_file() and not path.is_symlink(), f"preserve path is not a regular file: {name}")
            path.unlink()
    files = stage_files()
    require("README.md" in files, "release source README.md missing")
    print(f"staged {len(files)} source files; preserved Hub paths excluded")


def assets() -> None:
    expected = release()["assets"]
    names: set[str] = set()
    for asset in expected:
        name = asset["name"]
        require(safe_relative(name).name == name and name not in names, f"unsafe or duplicate asset: {name!r}")
        names.add(name)
        digest = asset.get("digest") or ""
        require(re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is not None, f"asset digest missing: {name}")
        source = ASSETS_DIR / name
        require(source.is_file() and not source.is_symlink(), f"release asset download missing: {name}")
        require(source.stat().st_size == asset["size"], f"release asset size mismatch: {name}")
        require(sha256(source) == digest.split(":", 1)[1], f"release asset digest mismatch: {name}")
        destination = STAGE / name
        require(not destination.exists(), f"release asset collides with source payload: {name}")
        shutil.copy2(source, destination)
    actual = {path.name for path in ASSETS_DIR.iterdir() if path.is_file()} if ASSETS_DIR.is_dir() else set()
    require(actual == names, "downloaded release asset set differs from GitHub release")
    print(f"verified and staged {len(names)} release assets")


def hub_file(repo: str, repo_type: str, revision: str, name: str, token: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo, name, repo_type=repo_type, revision=revision, token=token))


def assert_metadata(card: ModelCard, item: dict) -> dict:
    metadata = dict(card.data.to_dict())
    for key, value in item.get("required_card_metadata", {}).items():
        require(metadata.get(key) == value, f"card metadata {key} changed or missing")
    tags = metadata.get("tags") or []
    require(set(item.get("required_card_tags", [])).issubset(tags), "required card tag missing")
    return metadata


def preflight() -> None:
    from huggingface_hub import HfApi, ModelCard

    item = target()
    require(item.get("preserve_hub_card") is True, "curated Hub card policy is required")
    require(os.environ["HF_REPO_TYPE"] == "model", "model-card verification supports model repositories only")
    repo, repo_type, token = os.environ["HF_REPO_ID"], os.environ["HF_REPO_TYPE"], os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    info = api.repo_info(repo, repo_type=repo_type, revision="main")
    require(bool(info.sha), "Hub main revision missing")
    files = set(api.list_repo_files(repo, repo_type=repo_type, revision=info.sha))
    require("README.md" in files, "Hub curated card missing")
    staged = stage_files()
    for name in item.get("hub_only_paths", []):
        require(any(path.startswith(name) for path in files) if name.endswith("/") else name in files,
                f"Hub-only path missing: {name}")
    for name in item.get("preserve_hub_paths", []):
        require(name in files and name not in staged, f"Hub preservation path unavailable or staged: {name}")

    hashes = {name: sha256(hub_file(repo, repo_type, info.sha, name, token)) for name in sorted(files)}
    for name, path in staged.items():
        if name in hashes and name != "README.md":
            require(sha256(path) == hashes[name], f"source/Hub collision differs: {name}")

    card_bytes = hub_file(repo, repo_type, info.sha, "README.md", token).read_bytes()
    BASELINE_DIR.mkdir(exist_ok=True)
    (BASELINE_DIR / "README.md").write_bytes(card_bytes)
    card = ModelCard.load(BASELINE_DIR / "README.md")
    metadata = assert_metadata(card, item)
    evidence = {"sha": info.sha, "files": sorted(files), "hashes": hashes, "card_metadata": metadata}
    BASELINE_FILE.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Hub baseline {info.sha}: {len(files)} files; curated metadata and collisions verified")


def without_release_block(body: str) -> str:
    require(body.count(BEGIN) == body.count(END), "unbalanced mirror block markers")
    require(body.count(BEGIN) <= 1, "duplicate mirror release blocks")
    return re.sub(re.escape(BEGIN) + r".*?" + re.escape(END), "", body, flags=re.S).rstrip()


def release_auth(item: dict) -> str:
    auth = os.environ["HF_AUTH_MODE"]
    require(auth in ("oidc", "pat"), "release publisher requires verified OIDC or PAT auth")
    require(os.environ["HF_MIRROR_OIDC_RESOURCE"] == item["oidc_resource"],
            "OIDC resource differs from target map")
    return auth


def verify_revision(api: HfApi, item: dict, revision: str, expected_sha: str,
                    expected_files: set[str], expected_hashes: dict[str, str], token: str) -> None:
    from huggingface_hub import ModelCard

    repo, repo_type = os.environ["HF_REPO_ID"], os.environ["HF_REPO_TYPE"]
    info = api.repo_info(repo, repo_type=repo_type, revision=revision)
    require(info.sha == expected_sha, f"Hub {revision} resolves to unexpected commit")
    files = set(api.list_repo_files(repo, repo_type=repo_type, revision=revision))
    require(files == expected_files, f"Hub {revision} file set differs from additive union")
    for name, expected in expected_hashes.items():
        actual = sha256(hub_file(repo, repo_type, revision, name, token))
        require(actual == expected, f"Hub {revision} byte mismatch: {name}")
    card = ModelCard.load(hub_file(repo, repo_type, revision, "README.md", token))
    assert_metadata(card, item)
    require(card.text.count(BEGIN) == card.text.count(END) == 1, "mirror block missing or duplicate")
    require(api.repo_info(repo, repo_type=repo_type, revision=revision).sha == expected_sha,
            f"Hub {revision} moved during verification")


def publish() -> None:
    from huggingface_hub import HfApi, ModelCard

    item = target()
    base = json.loads(BASELINE_FILE.read_text(encoding="utf-8"))
    staged = stage_files()
    require("README.md" in staged, "rendered card missing")
    source_sha = os.environ["SOURCE_GITHUB_SHA"]
    require(re.fullmatch(r"[0-9a-f]{40}", source_sha) is not None, "release source SHA invalid")
    auth = release_auth(item)
    before_card = ModelCard.load(BASELINE_DIR / "README.md")
    after_card = ModelCard.load(staged["README.md"])
    require(assert_metadata(after_card, item) == base["card_metadata"], "curated card metadata changed")
    require(without_release_block(after_card.text) == without_release_block(before_card.text),
            "curated card body changed outside release block")
    require(after_card.text.count(BEGIN) == after_card.text.count(END) == 1,
            "rendered release block missing or duplicate")
    block = after_card.text.split(BEGIN, 1)[1].split(END, 1)[0]
    require(source_sha in block and os.environ["RELEASE_TAG"] in block, "release block source binding missing")

    repo, repo_type, tag, token = (os.environ[key] for key in ("HF_REPO_ID", "HF_REPO_TYPE", "RELEASE_TAG", "HF_TOKEN"))
    api = HfApi(token=token)
    require(api.repo_info(repo, repo_type=repo_type, revision="main").sha == base["sha"],
            "Hub main changed since baseline")
    stage_hashes = {name: sha256(path) for name, path in staged.items()}
    expected_files = set(base["files"]) | set(staged)
    expected_hashes = dict(base["hashes"])
    expected_hashes.update(stage_hashes)
    existing_tags = {ref.name for ref in api.list_repo_refs(repo, repo_type=repo_type).tags}
    if tag in existing_tags:
        # A prior run may have created the tag before receipt/artifact delivery
        # failed. A retry may witness it, but may never move it.
        oid = api.repo_info(repo, repo_type=repo_type, revision=tag).sha
        require(oid == base["sha"], "existing Hub tag does not point at current main")
    else:
        commit = api.upload_folder(
            repo_id=repo, repo_type=repo_type, folder_path=str(STAGE), token=token,
            parent_commit=base["sha"],
            commit_message=f"mirror {tag} from {os.environ['GITHUB_REPOSITORY']}@{source_sha}",
        )
        require(re.fullmatch(r"[0-9a-f]{40}", commit.oid or "") is not None, "Hub upload commit missing")
        oid = commit.oid
        verify_revision(api, item, oid, oid, expected_files, expected_hashes, token)
        notes = (release().get("body") or f"Mirror of {os.environ['GITHUB_REPOSITORY']} {tag}")[:4000]
        api.create_tag(repo, tag=tag, tag_message=notes, revision=oid, repo_type=repo_type,
                       token=token, exist_ok=False)
    verify_revision(api, item, tag, oid, expected_files, expected_hashes, token)
    require(api.repo_info(repo, repo_type=repo_type, revision="main").sha == oid,
            "Hub main moved after release upload")

    receipt = {
        "state": "MEASURED",
        "github_repo": os.environ["GITHUB_REPOSITORY"],
        "github_sha": source_sha,
        "github_workflow_sha": os.environ["GITHUB_SHA"],
        "github_tag": tag,
        "hf_repo_id": repo,
        "hf_repo_type": repo_type,
        "hf_oidc_resource": os.environ["HF_MIRROR_OIDC_RESOURCE"],
        "hf_main_before": base["sha"],
        "hf_revision": oid,
        "file_count_before": len(base["files"]),
        "file_count": len(expected_files),
        "verified_file_count": len(expected_hashes),
        "release_assets": sorted(asset["name"] for asset in release()["assets"]),
        "preserved_hub_files": len(set(base["files"]) - set(staged)),
        "auth": auth,
    }
    RECEIPT_FILE.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    commands = {"stage": stage, "assets": assets, "preflight": preflight, "publish": publish}
    require(len(sys.argv) == 2 and sys.argv[1] in commands, "expected stage, assets, preflight, or publish")
    commands[sys.argv[1]]()

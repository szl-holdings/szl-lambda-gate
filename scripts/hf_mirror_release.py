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
PRIOR_RECEIPT_FILE = Path(".hfmirror-prior/hf-mirror-receipt.json")
PRIOR_RUN_FILE = Path(".hfmirror-prior/run.json")
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


def collision_plan(staged_hashes: dict[str, str], hub_hashes: dict[str, str], item: dict) -> dict[str, dict[str, str]]:
    """Classify staged/Hub overlaps and return the reviewed replacements.

    README.md is rendered from the Hub card and verified separately. Every other
    overlap must be byte-identical unless the target lists the exact path in
    replace_hub_paths (GitHub-owned source a release may update). An absent or
    empty list keeps the strict additive-only rule unchanged.
    """
    replace = list(item.get("replace_hub_paths", []))
    preserve = set(item.get("preserve_hub_paths", []))
    require(len(set(replace)) == len(replace), "duplicate replace_hub_paths entry")
    for name in replace:
        safe_relative(name)
        require(name != "README.md", "README.md is rendered, never replaced")
        require(name not in preserve, f"path is both preserved and replaceable: {name}")
    replacements: dict[str, dict[str, str]] = {}
    for name, digest in sorted(staged_hashes.items()):
        if name == "README.md" or name not in hub_hashes or hub_hashes[name] == digest:
            continue
        require(name in replace, f"source/Hub collision differs: {name}")
        replacements[name] = {"hub_before": hub_hashes[name], "source": digest}
    return replacements


def assert_metadata(card: ModelCard, item: dict) -> dict:
    metadata = dict(card.data.to_dict())
    for key, value in item.get("required_card_metadata", {}).items():
        require(metadata.get(key) == value, f"card metadata {key} changed or missing")
    tags = metadata.get("tags") or []
    require(set(item.get("required_card_tags", [])).issubset(tags), "required card tag missing")
    return metadata


def prior_receipt(item: dict) -> dict:
    """Bind a retained publication receipt to this exact release and target."""
    require(re.fullmatch(r"[1-9][0-9]*", os.environ.get("HF_RECEIPT_RUN", "")) is not None,
            "historical verification requires a successful publisher run ID")
    data = json.loads(PRIOR_RECEIPT_FILE.read_text(encoding="utf-8"))
    run = json.loads(PRIOR_RUN_FILE.read_text(encoding="utf-8"))
    require(str(run.get("id")) == os.environ["HF_RECEIPT_RUN"], "prior publisher run ID mismatch")
    require(run.get("repository", {}).get("full_name") == os.environ["GITHUB_REPOSITORY"]
            and run.get("head_repository", {}).get("full_name") == os.environ["GITHUB_REPOSITORY"],
            "prior publisher run repository mismatch")
    require(run.get("path") == ".github/workflows/hf-mirror.yml"
            and run.get("event") == "workflow_dispatch" and run.get("conclusion") == "success",
            "prior publisher run identity or outcome mismatch")
    require(re.fullmatch(r"[0-9a-f]{40}", run.get("head_sha", "")) is not None
            and data.get("github_workflow_sha") == run["head_sha"],
            "prior receipt workflow commit differs from publisher run")
    expected = {
        "state": "MEASURED", "github_repo": os.environ["GITHUB_REPOSITORY"],
        "github_sha": os.environ["SOURCE_GITHUB_SHA"], "github_tag": os.environ["RELEASE_TAG"],
        "hf_repo_id": os.environ["HF_REPO_ID"], "hf_repo_type": os.environ["HF_REPO_TYPE"],
        "hf_oidc_resource": item["oidc_resource"],
    }
    for key, value in expected.items():
        require(data.get(key) == value, f"prior publication receipt {key} mismatch")
    require(data.get("operation", "publish") == "publish", "prior receipt must record publication")
    require(data.get("auth") in ("pat", "oidc"), "prior publication authentication missing")
    for key in ("hf_main_before", "hf_revision"):
        require(re.fullmatch(r"[0-9a-f]{40}", data.get(key, "")) is not None,
                f"prior publication receipt {key} invalid")
    require(data.get("release_assets") == sorted(asset["name"] for asset in release()["assets"]),
            "prior publication release asset set mismatch")
    return data


def preflight() -> None:
    from huggingface_hub import HfApi, ModelCard

    item = target()
    require(item.get("preserve_hub_card") is True, "curated Hub card policy is required")
    require(os.environ["HF_REPO_TYPE"] == "model", "model-card verification supports model repositories only")
    repo, repo_type, token = os.environ["HF_REPO_ID"], os.environ["HF_REPO_TYPE"], os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    info = api.repo_info(repo, repo_type=repo_type, revision="main")
    require(bool(info.sha), "Hub main revision missing")
    main_sha = info.sha
    prior = None
    if os.environ.get("HF_RECEIPT_RUN"):
        require(PRIOR_RECEIPT_FILE.is_file(), "retained publication receipt missing")
        prior = prior_receipt(item)
        require(api.repo_info(repo, repo_type=repo_type, revision=os.environ["RELEASE_TAG"]).sha
                == prior["hf_revision"], "existing Hub tag differs from publication receipt")
        info = api.repo_info(repo, repo_type=repo_type, revision=prior["hf_main_before"])
        require(info.sha == prior["hf_main_before"], "historical Hub baseline mismatch")
    files = set(api.list_repo_files(repo, repo_type=repo_type, revision=info.sha))
    require("README.md" in files, "Hub curated card missing")
    staged = stage_files()
    for name in item.get("hub_only_paths", []):
        require(any(path.startswith(name) for path in files) if name.endswith("/") else name in files,
                f"Hub-only path missing: {name}")
    for name in item.get("preserve_hub_paths", []):
        require(name in files and name not in staged, f"Hub preservation path unavailable or staged: {name}")

    hashes = {name: sha256(hub_file(repo, repo_type, info.sha, name, token)) for name in sorted(files)}
    replacements = collision_plan({name: sha256(path) for name, path in staged.items()}, hashes, item)
    for name, change in replacements.items():
        print(f"reviewed replacement {name}: hub {change['hub_before']} -> source {change['source']}")

    card_bytes = hub_file(repo, repo_type, info.sha, "README.md", token).read_bytes()
    BASELINE_DIR.mkdir(exist_ok=True)
    (BASELINE_DIR / "README.md").write_bytes(card_bytes)
    card = ModelCard.load(BASELINE_DIR / "README.md")
    metadata = assert_metadata(card, item)
    if prior:
        # Historical cards retain their original release block. Today's renderer
        # and edited release notes must not rewrite an immutable published card.
        published_card = hub_file(repo, repo_type, prior["hf_revision"], "README.md", token)
        staged["README.md"].write_bytes(published_card.read_bytes())
    evidence = {"sha": info.sha, "files": sorted(files), "hashes": hashes, "card_metadata": metadata,
                "replacements": replacements, "main_sha": main_sha, "prior_receipt": prior}
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
    if base.get("prior_receipt"):
        for row in (
            f"| GitHub source | https://github.com/{os.environ['GITHUB_REPOSITORY']} |",
            f"| Source commit | `{source_sha}` |",
            f"| Hub revision | `{os.environ['RELEASE_TAG']}` |",
        ):
            require(row in block.splitlines(), "historical release card source binding differs")

    repo, repo_type, tag, token = (os.environ[key] for key in ("HF_REPO_ID", "HF_REPO_TYPE", "RELEASE_TAG", "HF_TOKEN"))
    api = HfApi(token=token)
    main_before = base.get("main_sha", base["sha"])
    require(api.repo_info(repo, repo_type=repo_type, revision="main").sha == main_before,
            "Hub main changed since baseline")
    stage_hashes = {name: sha256(path) for name, path in staged.items()}
    expected_files = set(base["files"]) | set(staged)
    expected_hashes = dict(base["hashes"])
    expected_hashes.update(stage_hashes)
    prior = base.get("prior_receipt")
    if prior:
        require(prior == prior_receipt(item), "prior publication receipt changed since baseline")
        for key, value in {
            "file_count_before": len(base["files"]), "file_count": len(expected_files),
            "verified_file_count": len(expected_hashes),
            "preserved_hub_files": len(set(base["files"]) - set(staged)),
            "replaced_hub_files": sorted(base.get("replacements", {})),
        }.items():
            require(prior.get(key, [] if key == "replaced_hub_files" else None) == value,
                    f"historical publication {key} mismatch")
    existing_tags = {ref.name for ref in api.list_repo_refs(repo, repo_type=repo_type).tags}
    require(not prior or tag in existing_tags, "historical Hub tag missing; verification cannot upload")
    if tag in existing_tags:
        # A prior run may have created the tag before receipt/artifact delivery
        # failed. A retry may witness it, but may never move it.
        oid = api.repo_info(repo, repo_type=repo_type, revision=tag).sha
        require(prior is not None,
                "existing Hub tag requires receipt_run for read-only historical verification")
        require(oid == prior["hf_revision"], "existing Hub tag differs from publication receipt")
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
    require(api.repo_info(repo, repo_type=repo_type, revision="main").sha == (main_before if prior else oid),
            "Hub main moved during release publication or verification")

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
        "replaced_hub_files": sorted(base.get("replacements", {})),
        "auth": auth,
        "operation": "verify-existing" if prior else "publish",
        "publication_auth": prior["auth"] if prior else auth,
        "verification_auth": auth,
        "hf_main_observed": main_before,
        "file_sha256": expected_hashes,
    }
    if prior:
        receipt["source_receipt_run"] = os.environ["HF_RECEIPT_RUN"]
        receipt["source_receipt_sha256"] = sha256(PRIOR_RECEIPT_FILE)
    RECEIPT_FILE.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    commands = {"stage": stage, "assets": assets, "preflight": preflight, "publish": publish}
    require(len(sys.argv) == 2 and sys.argv[1] in commands, "expected stage, assets, preflight, or publish")
    commands[sys.argv[1]]()

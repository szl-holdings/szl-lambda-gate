#!/usr/bin/env python3
"""Package immutable Git objects for the universal Kernel Hub loader.

This is a stdlib-only packager for this pure-Python kernel, not kernel-builder
and not a signer. No working-tree bytes or wall-clock timestamps enter a build.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "https://github.com/szl-holdings/szl-lambda-gate"
PACKAGE = "szl_lambda_gate"
VARIANT = "build/torch-universal"
VARIANTS = (VARIANT, "build/torch-cpu")
CONTRACT_FILES = (
    "spec/szl.lambda.v1.json", "spec/lambda_v1_vectors.json",
    "reference/szl_lambda_v1.py", "LICENSE", "build.toml",
)


def canonical_bytes(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def git(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repo), *args])


def source_snapshot(repo: Path, revision: str) -> tuple[str, dict[str, bytes]]:
    commit = git(repo, "rev-parse", "--verify", "--end-of-options", f"{revision}^{{commit}}").decode().strip()
    object_ids = {}
    for row in git(repo, "ls-tree", "-r", "-z", commit).split(b"\0"):
        if not row:
            continue
        meta, raw_path = row.split(b"\t", 1)
        path = raw_path.decode("utf-8")
        if path not in CONTRACT_FILES and not path.startswith(f"torch-ext/{PACKAGE}/"):
            continue
        mode, kind, oid = meta.decode().split()
        if mode != "100644" or kind != "blob":
            raise ValueError(f"source must be a regular Git blob: {path}")
        object_ids[path] = oid
    # One process avoids repeatedly starting Git (especially costly on Windows).
    batch = subprocess.check_output(
        ["git", "-C", str(repo), "cat-file", "--batch"],
        input="".join(oid + "\n" for oid in object_ids.values()).encode(),
    )
    objects = {}
    offset = 0
    for path, oid in object_ids.items():
        end = batch.index(b"\n", offset)
        actual_oid, kind, size = batch[offset:end].decode().split()
        if actual_oid != oid or kind != "blob":
            raise ValueError(f"Git object mismatch: {path}")
        start = end + 1
        stop = start + int(size)
        objects[path] = batch[start:stop]
        offset = stop + 1
    for required in (*CONTRACT_FILES, f"torch-ext/{PACKAGE}/__init__.py",
                     f"torch-ext/{PACKAGE}/_v1.py"):
        if required not in objects:
            raise ValueError(f"revision lacks required v1 source: {required}")
    return commit, objects


def validate_vectors(spec: dict, vectors: dict) -> str:
    digest = hashlib.sha256(canonical_bytes(vectors)).hexdigest()
    if spec["schema"] != "szl.lambda/v1" or vectors["schema"] != "szl.lambda/v1.vectors":
        raise ValueError("unexpected v1 schema")
    if spec["vectors"]["sha256"] != digest or spec["vectors"]["count"] != len(vectors["vectors"]):
        raise ValueError("vector count or canonical SHA-256 differs from spec")
    ids = [row["id"] for row in vectors["vectors"]]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("empty vectors or duplicate vector id")
    return digest


def distribution_payload(repo: Path, revision: str) -> tuple[dict, dict[str, bytes]]:
    commit, source = source_snapshot(repo, revision)
    spec = json.loads(source["spec/szl.lambda.v1.json"])
    vectors = json.loads(source["spec/lambda_v1_vectors.json"])
    vector_digest = validate_vectors(spec, vectors)
    payload = {name: source[name] for name in CONTRACT_FILES if name != "LICENSE"}
    # Keep the curated Hub root license intact; retain exact source license bytes.
    payload["source/LICENSE"] = source["LICENSE"]
    # Update the CPU noarch variant too: it outranks the universal variant in
    # the consumer loader and must not keep pointing to an older implementation.
    for variant in VARIANTS:
        for name, data in source.items():
            if name.startswith("torch-ext/"):
                if not name.endswith((".py", ".pyi")):
                    raise ValueError(f"unexpected non-Python kernel source: {name}")
                payload[f"{variant}/{name.removeprefix('torch-ext/')}"] = data
        # Modern loaders enter the variant root; older loaders enter its package.
        payload[f"{variant}/__init__.py"] = (
            f"from .{PACKAGE} import *  # noqa: F401,F403\n"
            f"from .{PACKAGE} import __all__\n"
        ).encode()
        files = {
            path.removeprefix(f"{variant}/"): base64.b64encode(hashlib.sha256(data).digest()).decode()
            for path, data in sorted(payload.items()) if path.startswith(f"{variant}/")
        }
        suffix = variant.split("-")[-1]
        metadata = {
            "name": "szl-lambda-gate", "id": f"_szl_lambda_gate_{suffix}_{commit}",
            "version": 1, "license": "Apache-2.0", "universal": variant == VARIANT,
            "python-depends": [], "backend": {"type": "cpu"}, "source": REPOSITORY,
            "digest": {"algorithm": "sha256", "files": files},
        }
        payload[f"{variant}/metadata.json"] = canonical_bytes(metadata) + b"\n"
    receipt = {
        "schema": "szl.lambda/distribution.v1", "source_repository": REPOSITORY,
        "source_commit": commit, "packager": "scripts/build_lambda_distribution.py",
        "vectors_count": len(vectors["vectors"]), "vectors_sha256": vector_digest,
        "source_files": {path: hashlib.sha256(data).hexdigest() for path, data in sorted(source.items())},
        "files": {path: hashlib.sha256(data).hexdigest() for path, data in sorted(payload.items())},
    }
    payload["distribution.json"] = canonical_bytes(receipt) + b"\n"
    return receipt, payload


def build(repo: Path, revision: str, output: Path) -> dict:
    # Refuse overwrite rather than deleting or mixing a previous build.
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    receipt, payload = distribution_payload(repo, revision)
    for path, data in sorted(payload.items()):
        destination = output / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True, help="Git commit or tag to package")
    parser.add_argument("--output", required=True, type=Path, help="new output directory")
    args = parser.parse_args()
    receipt = build(ROOT, args.revision, args.output)
    print(json.dumps({key: receipt[key] for key in
                      ("source_commit", "vectors_count", "vectors_sha256")}, indent=2))


if __name__ == "__main__":
    main()

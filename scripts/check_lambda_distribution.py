#!/usr/bin/env python3
"""Verify a source-bound package before importing it, then run every v1 vector."""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import struct
import sys
import tempfile

from build_lambda_distribution import ROOT, REPOSITORY, VARIANTS, distribution_payload, validate_vectors


def verify_files(root: Path, expected_commit: str, source_repo: Path = ROOT) -> dict:
    receipt = json.loads((root / "distribution.json").read_bytes())
    if receipt["schema"] != "szl.lambda/distribution.v1" or receipt["source_repository"] != REPOSITORY:
        raise ValueError("distribution provenance schema or repository mismatch")
    if receipt["source_commit"] != expected_commit:
        raise ValueError("distribution source commit differs from expected commit")
    paths = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"symlink in distribution: {path}")
        if path.is_file():
            paths[path.relative_to(root).as_posix()] = path
    actual = set(paths) - {"distribution.json"}
    if actual != set(receipt["files"]):
        raise ValueError("distribution file set differs from receipt")
    for name, digest in receipt["files"].items():
        if hashlib.sha256(paths[name].read_bytes()).hexdigest() != digest:
            raise ValueError(f"distribution file hash mismatch: {name}")
    # A receipt cannot authenticate itself. Reconstruct the expected bytes from
    # trusted Git objects, so editing a file AND rehashing the receipt still fails.
    trusted_receipt, trusted_payload = distribution_payload(source_repo, expected_commit)
    if receipt != trusted_receipt:
        raise ValueError("distribution receipt differs from trusted Git source")
    for name, data in trusted_payload.items():
        if paths[name].read_bytes() != data:
            raise ValueError(f"distribution bytes differ from trusted Git source: {name}")
    spec = json.loads((root / "spec/szl.lambda.v1.json").read_bytes())
    vectors = json.loads((root / "spec/lambda_v1_vectors.json").read_bytes())
    digest = validate_vectors(spec, vectors)
    if receipt["vectors_sha256"] != digest or receipt["vectors_count"] != len(vectors["vectors"]):
        raise ValueError("distribution vector provenance differs from spec")
    for variant in VARIANTS:
        metadata = json.loads((root / variant / "metadata.json").read_bytes())
        suffix = variant.split("-")[-1]
        if metadata["source"] != REPOSITORY or metadata["id"] != f"_szl_lambda_gate_{suffix}_{expected_commit}":
            raise ValueError("kernel metadata source differs from receipt")
        actual_kernel = {name.removeprefix(f"{variant}/") for name in actual
                         if name.startswith(f"{variant}/") and name != f"{variant}/metadata.json"}
        if metadata["digest"]["algorithm"] != "sha256" or actual_kernel != set(metadata["digest"]["files"]):
            raise ValueError("kernel metadata file set differs from distribution")
        for name, expected in metadata["digest"]["files"].items():
            data = (root / variant / name).read_bytes()
            if base64.b64encode(hashlib.sha256(data).digest()).decode() != expected:
                raise ValueError(f"kernel metadata digest mismatch: {name}")
    return receipt


def decode(value):
    if isinstance(value, list):
        return [decode(v) for v in value]
    if isinstance(value, str) and value.startswith("f64:"):
        return struct.unpack(">d", bytes.fromhex(value[4:]))[0]
    return value


def check_vectors(kernel, vectors: list[dict], device: str = "cpu") -> list[dict]:
    import torch

    def form(value):
        value = decode(value)
        if isinstance(value, list) and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
            return torch.tensor(value, dtype=torch.float64, device=device)
        return value

    failures = []
    for row in vectors:
        try:
            axes, weights = form(row["axes"]), form(row["weights"])
            expect = row["expect"]
            try:
                score = kernel.lambda_v1(axes, weights)
            except kernel.LambdaV1Error as err:
                if expect.get("error") != err.code:
                    raise AssertionError(f"error {err.code}, expected {expect}")
            else:
                if "error" in expect:
                    raise AssertionError(f"returned a score, expected {expect['error']}")
                if not isinstance(score, torch.Tensor) or score.shape != torch.Size([]) or score.dtype != torch.float64:
                    raise AssertionError("score must be a scalar float64 tensor")
                value = score.item()
                if not math.isfinite(value) or abs(value - decode(expect["value_f64"])) > row["value_tol"]:
                    raise AssertionError(f"score {value} differs beyond value_tol")
            gate = kernel.lambda_v1_gate(axes, weights, decode(row["tau"]))
            if (gate.verdict, gate.code) != (expect["verdict"], expect["code"]):
                raise AssertionError(f"gate {(gate.verdict, gate.code)} differs from {expect}")
        except Exception as err:
            failures.append({"id": row["id"], "reason": f"{type(err).__name__}: {err}"})
    return failures


def load_local(root: Path):
    import kernels

    major, minor = map(int, importlib.metadata.version("kernels").split(".")[:2])
    if (major, minor) < (0, 14):
        return kernels.get_local_kernel(root, "szl_lambda_gate")
    return kernels.get_local_kernel(root, backend="cpu")


def download_remote(repo: str, revision: str, expected_commit: str, source_repo: Path, root: Path) -> str:
    """Download trusted expected files and reject unknown executable variant files."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    resolved = api.repo_info(repo, repo_type="kernel", revision=revision).sha
    if not resolved:
        raise ValueError("Hub revision did not resolve to a commit")
    _, expected = distribution_payload(source_repo, expected_commit)
    remote = set(api.list_repo_files(repo, repo_type="kernel", revision=resolved))
    missing = set(expected) - remote
    if missing:
        raise ValueError(f"remote distribution files missing: {sorted(missing)}")
    # Curated Hub root files may coexist, but executable variants must be exact.
    for variant in VARIANTS:
        if {p for p in remote if p.startswith(variant + "/")} != {p for p in expected if p.startswith(variant + "/")}:
            raise ValueError(f"remote variant file set differs: {variant}")
    for name in sorted(expected):
        cached = hf_hub_download(repo, filename=name, repo_type="kernel", revision=resolved)
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(Path(cached).read_bytes())
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    origin = parser.add_mutually_exclusive_group(required=True)
    origin.add_argument("--local", type=Path)
    origin.add_argument("--remote-revision", help="Hub kernel commit or review ref to verify")
    parser.add_argument("--remote-repo", default="SZLHOLDINGS/szl-lambda-gate")
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--source-repo", type=Path, default=ROOT)
    args = parser.parse_args()
    # Avoid writing Python caches into an integrity-checked payload.
    sys.dont_write_bytecode = True
    scratch = tempfile.TemporaryDirectory(prefix="lambda-distribution-") if args.remote_revision else None
    root = Path(scratch.name) if scratch else args.local
    hub_revision = None
    if args.remote_revision:
        hub_revision = download_remote(args.remote_repo, args.remote_revision, args.expected_commit, args.source_repo, root)
    receipt = verify_files(root, args.expected_commit, args.source_repo)
    if hub_revision:
        import kernels
        if tuple(map(int, importlib.metadata.version("kernels").split(".")[:2])) < (0, 14):
            raise ValueError("remote kernel verification requires kernels >= 0.14")
        kernel = kernels.get_kernel(args.remote_repo, revision=hub_revision, backend="cpu",
                                    trust_remote_code=[args.remote_repo])
    else:
        kernel = load_local(root)
    rows = json.loads((root / "spec/lambda_v1_vectors.json").read_bytes())["vectors"]
    failures = check_vectors(kernel, rows)
    # Exercise both entry variants, including the fallback that is normally
    # hidden by the CPU variant's higher priority.
    variant_results = {}
    for variant in VARIANTS:
        selected = load_local(root / variant)
        result = check_vectors(selected, rows)
        variant_results[variant] = {"passed": len(rows) - len(result), "failures": result}
        failures.extend({**failure, "variant": variant} for failure in result)
    # Import must not mutate the package.
    verify_files(root, args.expected_commit, args.source_repo)
    report = {
        "state": "PASS" if not failures else "FAIL", "source_commit": receipt["source_commit"],
        "vectors_sha256": receipt["vectors_sha256"], "vectors_count": len(rows),
        "passed": len(rows) - len({failure["id"] for failure in failures}), "failures": failures,
        "variants": variant_results,
        "kernels_version": importlib.metadata.version("kernels"),
        "torch_version": importlib.metadata.version("torch"), "device": "cpu",
        "signature_verified": False,
        "hub_repository": args.remote_repo if hub_revision else None,
        "hub_revision": hub_revision,
    }
    if args.receipt:
        args.receipt.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)
    if scratch:
        scratch.cleanup()


if __name__ == "__main__":
    main()

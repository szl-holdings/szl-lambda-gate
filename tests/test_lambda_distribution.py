"""Distribution integrity: Git objects, reproducibility, and hostile receipts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_lambda_distribution import build, canonical_bytes, source_snapshot  # noqa: E402
from check_lambda_distribution import load_remote, verify_files  # noqa: E402


@pytest.fixture
def committed_source(tmp_path):
    _, source = source_snapshot(ROOT, "HEAD")
    repo = tmp_path / "source"
    repo.mkdir()
    for name, data in source.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args])
    git("init", "-q")
    git("config", "user.name", "Distribution test")
    git("config", "user.email", "distribution-test@example.invalid")
    git("add", ".")
    git("-c", "commit.gpgsign=false", "commit", "-qm",
        "Test source fixture\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>")
    return repo, git("rev-parse", "HEAD").decode().strip()


def test_build_uses_committed_bytes_and_is_reproducible(committed_source, tmp_path):
    repo, commit = committed_source
    first, second = tmp_path / "first", tmp_path / "second"
    build(repo, commit, first)
    (repo / "torch-ext/szl_lambda_gate/_v1.py").write_text("raise RuntimeError('dirty')\n")
    (repo / "torch-ext/szl_lambda_gate/rogue.py").write_text("raise RuntimeError('untracked')\n")
    build(repo, commit, second)
    a = {p.relative_to(first): p.read_bytes() for p in first.rglob("*") if p.is_file()}
    b = {p.relative_to(second): p.read_bytes() for p in second.rglob("*") if p.is_file()}
    assert a == b
    assert verify_files(first, commit, repo)["source_commit"] == commit


def test_build_never_overwrites_an_existing_destination(committed_source, tmp_path):
    repo, commit = committed_source
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_bytes(b"keep")
    with pytest.raises(ValueError, match="already exists"):
        build(repo, commit, output)
    assert sentinel.read_bytes() == b"keep"


def test_verifier_rejects_file_corruption_before_import(committed_source, tmp_path):
    repo, commit = committed_source
    output = tmp_path / "payload"
    build(repo, commit, output)
    path = output / "build/torch-universal/szl_lambda_gate/_v1.py"
    path.write_bytes(b"raise RuntimeError('must not execute')\n")
    with pytest.raises(ValueError, match="file hash mismatch"):
        verify_files(output, commit, repo)


def test_rehashing_a_corrupt_payload_does_not_authenticate_it(committed_source, tmp_path):
    repo, commit = committed_source
    output = tmp_path / "payload"
    build(repo, commit, output)
    name = "build/torch-universal/szl_lambda_gate/_v1.py"
    (output / name).write_bytes(b"raise RuntimeError('must not execute')\n")
    manifest = output / "distribution.json"
    receipt = json.loads(manifest.read_bytes())
    receipt["files"][name] = hashlib.sha256((output / name).read_bytes()).hexdigest()
    manifest.write_bytes(canonical_bytes(receipt) + b"\n")
    with pytest.raises(ValueError, match="trusted Git source"):
        verify_files(output, commit, repo)


def test_verifier_rejects_unlisted_files(committed_source, tmp_path):
    repo, commit = committed_source
    output = tmp_path / "payload"
    build(repo, commit, output)
    (output / "unexpected.py").write_bytes(b"print('extra code')\n")
    with pytest.raises(ValueError, match="file set differs"):
        verify_files(output, commit, repo)


def test_verifier_requires_the_expected_source_commit(committed_source, tmp_path):
    repo, commit = committed_source
    output = tmp_path / "payload"
    build(repo, commit, output)
    with pytest.raises(ValueError, match="expected commit"):
        verify_files(output, "0" * 40, repo)


@pytest.mark.parametrize("loader_version,repo_type", [("0.12.3", "kernel"), ("0.17.1", "model")])
def test_remote_verifier_refuses_wrong_repository_type_before_loading(monkeypatch, loader_version, repo_type):
    def must_not_load(*args, **kwargs):
        raise AssertionError("unsupported repository type must never be loaded")
    monkeypatch.setitem(sys.modules, "kernels", SimpleNamespace(get_kernel=must_not_load))
    monkeypatch.setattr("check_lambda_distribution.importlib.metadata.version", lambda _: loader_version)
    with pytest.raises(ValueError, match="repository verification requires"):
        load_remote("SZLHOLDINGS/szl-lambda-gate", "a" * 40, repo_type)

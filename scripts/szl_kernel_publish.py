#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""SZL canonical kernel publisher: GitHub torch-ext/ -> Hugging Face Kernel Hub.

One writer per Hub kernel repository. Everything the Hub serves to
``kernels.get_kernel`` is derived from Git objects at one GitHub commit:

* ``build/torch-cpu/<pkg>/...`` and ``build/torch-universal/<pkg>/...`` are byte
  copies (CRLF normalised to LF) of the configured package sources;
* ``build/<variant>/metadata.json`` is generated per variant (name, id, version,
  license, source, python-depends, backend, sha256 digest);
* optional README.md (kernel card), LICENSE and extra files come from the same
  commit; ``SZL_KERNEL_SOURCE.json`` binds the Hub tree to that commit.

The publisher owns ``build/`` on the kernel repo completely (stale variants and
stale modules are removed) and writes the same managed tree to ``main`` and to
the version branch ``v<major>`` (created when absent). Other Hub paths are kept
unless the repository config lists them in ``prune_paths``.

Transport is plain Git over HTTPS (``huggingface_hub`` cannot commit to
``repo_type="kernel"``). Pushes are fast-forward only: no force, no repository
creation, no settings changes. A moved remote branch makes the push fail; a
re-run recomputes from the new head, so repeated runs converge.

Subcommands (see ``--help``): resolve-source, plan, build, check-local,
publish, verify, twin. ``build``/``publish``/``twin`` need only the standard
library and git. ``check-local`` and ``verify`` need ``torch`` and
``kernels==0.17.2`` (and the ``huggingface_hub`` that kernels installs).
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

CONFIG_PATH = ".github/hf-kernels.json"
CONFIG_SCHEMA = "szl.hf-kernels/v1"
BINDING_PATH = "SZL_KERNEL_SOURCE.json"
BINDING_SCHEMA = "szl.kernel-source-binding/v1"
PUBLISHER = "scripts/szl_kernel_publish.py"
VARIANTS = ("torch-cpu", "torch-universal")
HUB = "https://huggingface.co"
KERNEL_NAME = re.compile(r"^[a-z][-a-z0-9]*[a-z0-9]$")  # kernels-common KernelName
PACKAGE_NAME = re.compile(r"^[a-z_][a-z0-9_]*$")
REPO_ID = re.compile(r"^SZLHOLDINGS/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
GH_REPO = re.compile(r"^szl-holdings/[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
SHA40 = re.compile(r"^[0-9a-f]{40}$")
MAX_FILE_BYTES = 5_000_000
CARD_TOKENS = ("REPO_ID", "KERNEL_NAME", "KERNEL_VERSION", "KERNEL_URL",
               "MODEL_TWIN_URL", "GITHUB_REPOSITORY")
CARD_FORBIDDEN = (
    (re.compile(r"revision\s*=\s*['\"]main['\"]"),
     'floating revision="main" (use version=<major> or an immutable kernel-repo commit)'),
    (re.compile(r"get_kernel\(\s*['\"][^'\"]+['\"]\s*\)"),
     "bare get_kernel(repo_id): kernels>=0.17 requires version= or revision="),
    (re.compile(r"REPLACE_WITH|<40-hex|TODO_PIN", re.IGNORECASE), "unfilled placeholder"),
)


class PublishError(RuntimeError):
    """Any failure is a failed publication; nothing is reported as published."""


# --------------------------------------------------------------------------- utils

def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_b64(data: bytes) -> str:
    return base64.b64encode(hashlib.sha256(data).digest()).decode("ascii")


def scrub(text: str) -> str:
    token = os.environ.get("HF_TOKEN") or ""
    if token:
        text = text.replace(token, "[REDACTED]")
    return re.sub(r"hf_[A-Za-z0-9]{20,}", "[REDACTED]", text)


def git(repo: Path, *args: str, data: bytes | None = None,
        env: dict[str, str] | None = None, check: bool = True) -> bytes:
    result = subprocess.run(
        ["git", "-c", "credential.helper=", "-c", "core.hooksPath=/dev/null",
         "-c", "commit.gpgsign=false", *args],
        cwd=repo, input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=env, timeout=600, check=False,
    )
    if check and result.returncode:
        raise PublishError(f"git {args[0]} failed: "
                           f"{scrub(result.stderr.decode('utf-8', 'replace'))[-2000:]}")
    return result.stdout


def normalize_text(path: str, data: bytes) -> bytes:
    """Kernel sources must be UTF-8 text; CRLF is normalised to LF.

    The Hub rejects plain-Git binary pushes (binaries need Xet/LFS), and a kernel
    package that ships binaries is out of scope for this noarch publisher.
    """
    if len(data) > MAX_FILE_BYTES:
        raise PublishError(f"file too large for the noarch publisher: {path}")
    if b"\0" in data:
        raise PublishError(f"binary file in kernel payload (not supported): {path}")
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        raise PublishError(f"non-UTF-8 file in kernel payload (not supported): {path}") from None
    return data.replace(b"\r\n", b"\n")


def write_output(name: str, value: str) -> None:
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a", encoding="utf-8") as handle:
            handle.write(f"{name}={value}\n")
    print(f"{name}={value}")


# -------------------------------------------------------------------------- config

@dataclass(frozen=True)
class Package:
    src: str   # directory in the GitHub repository, e.g. torch-ext/szl_x
    name: str  # package directory name on the Hub, e.g. szl_x


@dataclass(frozen=True)
class Twin:
    repo_id: str
    card: str
    prune_paths: tuple[str, ...]


@dataclass(frozen=True)
class KernelConfig:
    repo_id: str
    name: str
    version: int
    license: str
    source_repository: str
    packages: tuple[Package, ...]
    card: str | None = None
    license_file: str | None = None
    extra_files: tuple[tuple[str, str], ...] = ()
    prune_paths: tuple[str, ...] = ()
    python_depends: tuple[str, ...] = ()
    smoke: str | None = None
    twin: Twin | None = None
    raw: dict = field(default_factory=dict, compare=False, repr=False)

    @property
    def python_name(self) -> str:
        return self.name.replace("-", "_")

    @property
    def slug(self) -> str:
        return self.repo_id.split("/", 1)[1].lower()

    @property
    def version_branch(self) -> str:
        return f"v{self.version}"

    @property
    def kernel_url(self) -> str:
        return f"{HUB}/kernels/{self.repo_id}"

    def managed_files(self) -> set[str]:
        files = {BINDING_PATH, *(dst for _, dst in self.extra_files)}
        if self.card:
            files.add("README.md")
        if self.license_file:
            files.add("LICENSE")
        return files


def _rel_path(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value or value.startswith(("/", "\\")) \
            or "\\" in value or any(part in ("", ".", "..") for part in value.rstrip("/").split("/")):
        raise PublishError(f"{what} must be a relative POSIX path: {value!r}")
    return value


def parse_kernel(entry: dict, index: int) -> KernelConfig:
    where = f"kernels[{index}]"
    repo_id = entry.get("repo_id")
    if not isinstance(repo_id, str) or not REPO_ID.fullmatch(repo_id):
        raise PublishError(f"{where}.repo_id must be SZLHOLDINGS/<name>: {repo_id!r}")
    name = entry.get("name")
    if not isinstance(name, str) or not KERNEL_NAME.fullmatch(name):
        raise PublishError(f"{where}.name must match {KERNEL_NAME.pattern} (kernels KernelName): {name!r}")
    version = entry.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise PublishError(f"{where}.version must be an integer >= 1 (the v<major> branch)")
    license_id = entry.get("license")
    if not isinstance(license_id, str) or not license_id:
        raise PublishError(f"{where}.license is required (SPDX id, e.g. Apache-2.0)")
    source_repository = entry.get("source_repository")
    if not isinstance(source_repository, str) or not GH_REPO.fullmatch(source_repository):
        raise PublishError(f"{where}.source_repository must be szl-holdings/<repo>")
    packages_raw = entry.get("packages")
    if not isinstance(packages_raw, list) or not packages_raw:
        raise PublishError(f"{where}.packages must be a non-empty list")
    packages = []
    for row in packages_raw:
        if not isinstance(row, dict):
            raise PublishError(f"{where}.packages entries are objects {{src, name}}")
        src = _rel_path(row.get("src"), f"{where}.packages[].src").rstrip("/")
        pkg = row.get("name")
        if not isinstance(pkg, str) or not PACKAGE_NAME.fullmatch(pkg) or pkg.startswith("torch"):
            raise PublishError(f"{where}.packages[].name must be a lowercase Python package name: {pkg!r}")
        packages.append(Package(src, pkg))
    if len({p.name for p in packages}) != len(packages):
        raise PublishError(f"{where}.packages names must be unique")
    if packages[0].name != name.replace("-", "_"):
        raise PublishError(
            f"{where}: the first package must be named {name.replace('-', '_')!r} "
            "(kernels imports build/<variant>/<python_name>/__init__.py)")
    extra = entry.get("extra_files") or {}
    if not isinstance(extra, dict):
        raise PublishError(f"{where}.extra_files must map source path -> Hub path")
    extra_files = tuple(sorted((_rel_path(s, "extra_files source"), _rel_path(d, "extra_files target"))
                               for s, d in extra.items()))
    card = entry.get("card")
    license_file = entry.get("license_file")
    if card is not None:
        _rel_path(card, f"{where}.card")
    if license_file is not None:
        _rel_path(license_file, f"{where}.license_file")
    prune = tuple(_rel_path(p, f"{where}.prune_paths[]") for p in entry.get("prune_paths") or [])
    depends = entry.get("python_depends") or []
    if not isinstance(depends, list) or not all(isinstance(d, str) and d for d in depends):
        raise PublishError(f"{where}.python_depends must be a list of kernels dependency names")
    smoke = entry.get("smoke")
    if smoke is not None:
        _rel_path(smoke, f"{where}.smoke")
    twin = None
    twin_raw = entry.get("model_twin")
    if twin_raw is not None:
        if not isinstance(twin_raw, dict) or twin_raw.get("mode") != "pointer":
            raise PublishError(f"{where}.model_twin.mode must be \"pointer\" "
                               "(the model twin is never loaded by kernels>=0.16; it only points to the kernel repo)")
        twin_id = twin_raw.get("repo_id")
        if not isinstance(twin_id, str) or not REPO_ID.fullmatch(twin_id):
            raise PublishError(f"{where}.model_twin.repo_id must be SZLHOLDINGS/<name>")
        twin_card = _rel_path(twin_raw.get("card"), f"{where}.model_twin.card")
        twin_prune = tuple(_rel_path(p, "model_twin.prune_paths[]") for p in twin_raw.get("prune_paths") or [])
        if any(p in (".gitattributes", "README.md") for p in twin_prune):
            raise PublishError(f"{where}.model_twin.prune_paths may not remove .gitattributes or README.md")
        twin = Twin(twin_id, twin_card, twin_prune)
    cfg = KernelConfig(
        repo_id=repo_id, name=name, version=version, license=license_id,
        source_repository=source_repository, packages=tuple(packages), card=card,
        license_file=license_file, extra_files=extra_files, prune_paths=prune,
        python_depends=tuple(depends), smoke=smoke, twin=twin, raw=entry,
    )
    managed = cfg.managed_files()
    for _, dst in extra_files:
        if dst.startswith("build/") or dst == BINDING_PATH:
            raise PublishError(f"{where}.extra_files target is reserved: {dst}")
        if (dst == "README.md" and card) or (dst == "LICENSE" and license_file):
            raise PublishError(f"{where}.extra_files target duplicates card/license_file: {dst}")
    if len({dst for _, dst in extra_files}) != len(extra_files):
        raise PublishError(f"{where}.extra_files targets must be unique")
    for path in prune:
        covers_managed = any(m.startswith(path) for m in managed) if path.endswith("/") else path in managed
        if path == ".gitattributes" or path.startswith("build/") or covers_managed:
            raise PublishError(f"{where}.prune_paths may not touch build/, .gitattributes or managed files: {path}")
    return cfg


def load_config(repo_root: Path, only: str = "") -> list[KernelConfig]:
    path = repo_root / CONFIG_PATH
    if not path.is_file():
        raise PublishError(f"{CONFIG_PATH} missing")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != CONFIG_SCHEMA:
        raise PublishError(f"{CONFIG_PATH} schema must be {CONFIG_SCHEMA!r}")
    entries = data.get("kernels")
    if not isinstance(entries, list) or not entries:
        raise PublishError(f"{CONFIG_PATH}.kernels must be a non-empty list")
    configs = [parse_kernel(entry, i) for i, entry in enumerate(entries)]
    if len({c.repo_id for c in configs}) != len(configs):
        raise PublishError("duplicate kernel repo_id in config")
    if only:
        configs = [c for c in configs if c.repo_id == only]
        if not configs:
            raise PublishError(f"no configured kernel matches {only!r}")
    return configs


def find_kernel(repo_root: Path, repo_id: str) -> KernelConfig:
    return load_config(repo_root, repo_id)[0]


# ------------------------------------------------------------------- source/build

def resolve_commit(repo_root: Path, ref: str) -> str:
    sha = git(repo_root, "rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}").decode().strip()
    if not SHA40.fullmatch(sha):
        raise PublishError(f"cannot resolve {ref!r} to a commit")
    return sha


def resolve_ref(repo_root: Path, ref: str) -> str:
    """HEAD and hex SHAs resolve directly; else a tag wins over origin branches, then any rev."""
    if ref == "HEAD" or re.fullmatch(r"[0-9a-f]{7,40}", ref):
        return resolve_commit(repo_root, ref)
    for candidate in (f"refs/tags/{ref}", f"refs/remotes/origin/{ref}", ref):
        probe = subprocess.run(["git", "rev-parse", "--verify", "--quiet", "--end-of-options",
                                f"{candidate}^{{commit}}"], cwd=repo_root,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        if probe.returncode == 0:
            return resolve_commit(repo_root, candidate)
    raise PublishError(f"cannot resolve {ref!r} to a commit")


def is_ancestor(repo_root: Path, commit: str, branch_ref: str) -> bool:
    return subprocess.run(["git", "merge-base", "--is-ancestor", commit, branch_ref], cwd=repo_root,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False).returncode == 0


def read_blob(repo_root: Path, commit: str, path: str) -> bytes:
    kind = git(repo_root, "cat-file", "-t", f"{commit}:{path}", check=False).decode().strip()
    if kind != "blob":
        raise PublishError(f"{path} is not a file at {commit[:12]}")
    mode_line = git(repo_root, "ls-tree", "-z", commit, "--", path).split(b"\0")[0].decode()
    if not mode_line.startswith(("100644 ", "100755 ")):
        raise PublishError(f"{path} must be a regular file at {commit[:12]} (no symlinks)")
    return git(repo_root, "cat-file", "blob", f"{commit}:{path}")


def read_tree_files(repo_root: Path, commit: str, prefix: str) -> dict[str, bytes]:
    """Regular files under ``prefix`` at ``commit``, keyed by path relative to it."""
    kind = git(repo_root, "cat-file", "-t", f"{commit}:{prefix}", check=False).decode().strip()
    if kind != "tree":
        raise PublishError(f"package source {prefix}/ is not a directory at {commit[:12]}")
    rows = git(repo_root, "ls-tree", "-r", "-z", "--full-tree", commit, "--", f"{prefix}/")
    entries: dict[str, str] = {}
    for row in rows.split(b"\0"):
        if not row:
            continue
        meta, raw_path = row.split(b"\t", 1)
        mode, kind, oid = meta.decode().split()
        path = raw_path.decode("utf-8")
        rel = path[len(prefix) + 1:]
        if "__pycache__" in rel.split("/") or rel.endswith((".pyc", ".pyo")):
            raise PublishError(f"bytecode is tracked in Git; remove it: {path}")
        if mode not in ("100644", "100755") or kind != "blob":
            raise PublishError(f"only regular files are publishable (mode {mode}): {path}")
        entries[rel] = oid
    if not entries:
        raise PublishError(f"package source {prefix}/ is empty at {commit[:12]}")
    batch = git(repo_root, "cat-file", "--batch",
                data="".join(f"{oid}\n" for oid in entries.values()).encode())
    files: dict[str, bytes] = {}
    offset = 0
    for rel, oid in entries.items():
        end = batch.index(b"\n", offset)
        got_oid, kind, size = batch[offset:end].decode().split()
        if got_oid != oid or kind != "blob":
            raise PublishError(f"git object mismatch for {prefix}/{rel}")
        start = end + 1
        files[rel] = batch[start:start + int(size)]
        offset = start + int(size) + 1
    return files


def render_card(cfg: KernelConfig, text: str, where: str) -> str:
    values = {
        "REPO_ID": cfg.repo_id, "KERNEL_NAME": cfg.name, "KERNEL_VERSION": str(cfg.version),
        "KERNEL_URL": cfg.kernel_url, "GITHUB_REPOSITORY": cfg.source_repository,
        "MODEL_TWIN_URL": f"{HUB}/{cfg.twin.repo_id}" if cfg.twin else f"{HUB}/{cfg.repo_id}",
    }
    for key in CARD_TOKENS:
        text = text.replace("{{" + key + "}}", values[key])
    left = re.findall(r"\{\{[A-Z_]+\}\}", text)
    if left:
        raise PublishError(f"{where}: unknown card tokens {sorted(set(left))}")
    return text


def lint_card(cfg: KernelConfig, text: str, where: str, require_usage: bool) -> None:
    problems = [msg for pattern, msg in CARD_FORBIDDEN if pattern.search(text)]
    if require_usage:
        if not re.search(rf"get_kernel\(\s*['\"]{re.escape(cfg.repo_id)}['\"]", text):
            problems.append(f'card must show get_kernel("{cfg.repo_id}", ...)')
        if not re.search(rf"version\s*=\s*{cfg.version}\b", text):
            problems.append(f"card must show version={cfg.version}")
    if problems:
        raise PublishError(f"{where} fails the card lint: " + "; ".join(problems))


def variant_metadata(cfg: KernelConfig, variant: str, commit: str,
                     variant_files: dict[str, bytes]) -> bytes:
    """metadata.json in the shape kernels-common's ``Metadata`` deserialises."""
    suffix = variant.split("-", 1)[1]
    return canonical_json({
        "name": cfg.name,
        "id": f"_{cfg.python_name}_{suffix}_{commit[:12]}",
        "version": cfg.version,
        "license": cfg.license,
        "source": f"https://github.com/{cfg.source_repository}",
        "python-depends": list(cfg.python_depends),
        "kernel-depends": [],
        # noarch Python: no archs, so kernels' ArchValidator never rejects a device.
        # Variant choice is by directory: torch-cpu on CPU hosts, torch-universal elsewhere.
        "backend": {"type": "cpu"},
        "digest": {"algorithm": "sha256",
                   "files": {rel: sha256_b64(data) for rel, data in sorted(variant_files.items())}},
    })


def content_key(payload: dict[str, bytes]) -> str:
    """Digest of everything except commit-bound files (metadata ids, binding)."""
    rows = {path: sha256_hex(data) for path, data in sorted(payload.items())
            if path != BINDING_PATH and not (path.startswith("build/") and path.endswith("/metadata.json")
                                             and path.count("/") == 2)}
    return sha256_hex(canonical_json(rows))


def build_payload(cfg: KernelConfig, repo_root: Path, commit: str) -> dict[str, bytes]:
    """The exact managed Hub tree for ``cfg`` at GitHub ``commit`` (deterministic)."""
    if not SHA40.fullmatch(commit):
        raise PublishError("source must be a 40-hex commit")
    payload: dict[str, bytes] = {}
    per_variant: dict[str, dict[str, bytes]] = {v: {} for v in VARIANTS}
    for pkg in cfg.packages:
        files = read_tree_files(repo_root, commit, pkg.src)
        if pkg is cfg.packages[0] and "__init__.py" not in files:
            raise PublishError(f"{pkg.src}/__init__.py missing: the kernel module is the first package")
        for rel, data in sorted(files.items()):
            content = normalize_text(f"{pkg.src}/{rel}", data)
            for variant in VARIANTS:
                per_variant[variant][f"{pkg.name}/{rel}"] = content
    for variant in VARIANTS:
        for rel, data in per_variant[variant].items():
            payload[f"build/{variant}/{rel}"] = data
        payload[f"build/{variant}/metadata.json"] = variant_metadata(cfg, variant, commit, per_variant[variant])
    if cfg.card:
        text = normalize_text(cfg.card, read_blob(repo_root, commit, cfg.card)).decode("utf-8")
        text = render_card(cfg, text, cfg.card)
        lint_card(cfg, text, cfg.card, require_usage=True)
        payload["README.md"] = text.encode("utf-8")
    if cfg.license_file:
        payload["LICENSE"] = normalize_text(cfg.license_file, read_blob(repo_root, commit, cfg.license_file))
    for src, dst in cfg.extra_files:
        payload[dst] = normalize_text(src, read_blob(repo_root, commit, src))
    payload[BINDING_PATH] = canonical_json({
        "schema": BINDING_SCHEMA,
        "repo_id": cfg.repo_id,
        "repo_type": "kernel",
        "kernel_name": cfg.name,
        "kernel_version": cfg.version,
        "source_repository": cfg.source_repository,
        "source_commit": commit,
        "publisher": PUBLISHER,
        "packages": [{"src": p.src, "name": p.name} for p in cfg.packages],
        "content_sha256": content_key(payload),
        "files": {path: sha256_hex(data) for path, data in sorted(payload.items())},
        "load": {"call": f'get_kernel("{cfg.repo_id}", version={cfg.version}, trust_remote_code=True)',
                 "requires": "kernels>=0.17.2"},
    })
    return payload


def write_dist(payload: dict[str, bytes], out: Path) -> None:
    if out.exists() and any(out.iterdir()):
        raise PublishError(f"output directory is not empty: {out}")
    for path, data in sorted(payload.items()):
        target = out / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def is_pruned(path: str, prune: Iterable[str]) -> bool:
    return any(path.startswith(p) if p.endswith("/") else path == p for p in prune)


# ------------------------------------------------------------------ git transport

class HubGit:
    """Fast-forward-only Git writer for one Hub repository (index-only, no checkout)."""

    def __init__(self, url: str, root: Path, token: str | None):
        self.url = url
        self.root = root
        root.mkdir(parents=True, exist_ok=False)
        self.env = dict(os.environ)
        self.env.update(
            GIT_TERMINAL_PROMPT="0", GIT_LFS_SKIP_SMUDGE="1",
            GIT_AUTHOR_NAME="SZL kernel publisher", GIT_COMMITTER_NAME="SZL kernel publisher",
            GIT_AUTHOR_EMAIL="kernel-publisher@szl-holdings.invalid",
            GIT_COMMITTER_EMAIL="kernel-publisher@szl-holdings.invalid",
        )
        self.env.pop("GIT_INDEX_FILE", None)
        if token:
            askpass = root.parent / f"{root.name}-askpass.sh"
            askpass.write_text('#!/bin/sh\ncase "$1" in\n*sername*) printf "%s\\n" "szl-kernel-publisher";;\n'
                               '*) printf "%s\\n" "$SZL_HUB_TOKEN";;\nesac\n', encoding="utf-8")
            askpass.chmod(0o700)
            self.env.update(GIT_ASKPASS=str(askpass), SZL_HUB_TOKEN=token)
        git(root, "init", "-q", env=self.env)
        git(root, "remote", "add", "origin", url, env=self.env)

    def run(self, *args: str, data: bytes | None = None, index: str | None = None) -> bytes:
        env = self.env if index is None else {**self.env, "GIT_INDEX_FILE": str(self.root / ".git" / index)}
        return git(self.root, *args, data=data, env=env)

    def heads(self) -> dict[str, str]:
        rows = self.run("ls-remote", "--heads", "origin").decode().splitlines()
        return {ref.removeprefix("refs/heads/"): sha for sha, ref in (row.split("\t") for row in rows if row)}

    def fetch(self, branches: Iterable[str]) -> None:
        specs = [f"+refs/heads/{b}:refs/remotes/origin/{b}" for b in branches]
        if specs:
            self.run("fetch", "-q", "--no-tags", "--depth=1", "origin", *specs)

    def tree(self, rev: str) -> dict[str, tuple[str, str, str]]:
        entries = {}
        for item in self.run("ls-tree", "-r", "-z", "--full-tree", rev).split(b"\0"):
            if item:
                meta, path = item.split(b"\t", 1)
                mode, kind, oid = meta.decode().split()
                entries[path.decode("utf-8")] = (mode, kind, oid)
        return entries

    def candidate(self, branch: str, parent: str, payload: dict[str, bytes], owned_prefixes: tuple[str, ...],
                  prune: tuple[str, ...], message: str) -> tuple[str, list[dict]]:
        before = self.tree(parent)
        lines, changes = [], []
        for path, (mode, kind, oid) in sorted(before.items()):
            if path in payload:
                continue
            if any(path.startswith(p) for p in owned_prefixes) or is_pruned(path, prune):
                lines.append(f"0 {'0' * 40}\t{path}")
                changes.append({"path": path, "op": "delete", "before_blob": oid})
        for path, data in sorted(payload.items()):
            old = before.get(path)
            if old is not None and old[1] != "blob":
                raise PublishError(f"managed path is not a regular file on the Hub: {path}")
            oid = self.run("hash-object", "-w", "--stdin", data=data).decode().strip()
            if old is not None and old[2] == oid and old[0] in ("100644", "100755"):
                continue
            lines.append(f"100644 {oid}\t{path}")
            changes.append({"path": path, "op": "replace" if old else "add",
                            "before_blob": old[2] if old else None, "after_blob": oid,
                            "after_sha256": sha256_hex(data)})
        if not changes:
            return parent, []
        index = f"index-{branch}"
        self.run("read-tree", parent, index=index)
        self.run("update-index", "--index-info", data=("\n".join(lines) + "\n").encode("utf-8"), index=index)
        tree = self.run("write-tree", index=index).decode().strip()
        commit = self.run("commit-tree", tree, "-p", parent, data=message.encode("utf-8")).decode().strip()
        return commit, changes

    def push(self, updates: dict[str, str], dry_run: bool) -> None:
        refspecs = [f"{sha}:refs/heads/{branch}" for branch, sha in sorted(updates.items())]
        if refspecs:
            self.run("push", "--porcelain", *(["--dry-run"] if dry_run else []), "origin", *refspecs)

    def readback(self, branch: str, expected_commit: str, payload: dict[str, bytes],
                 owned_prefixes: tuple[str, ...], prune: tuple[str, ...],
                 unmanaged_before: dict[str, tuple[str, str, str]]) -> dict[str, str]:
        heads = self.heads()
        if heads.get(branch) != expected_commit:
            raise PublishError(f"Hub {branch} is {heads.get(branch)} after push, expected {expected_commit}")
        self.fetch([branch])
        observed = self.run("rev-parse", f"refs/remotes/origin/{branch}").decode().strip()
        if observed != expected_commit:
            raise PublishError(f"fetched {branch} differs from the published commit")
        after = self.tree(observed)
        owned = {p for p in after if any(p.startswith(o) for o in owned_prefixes)}
        expected_owned = {p for p in payload if any(p.startswith(o) for o in owned_prefixes)}
        if owned != expected_owned:
            raise PublishError(f"{branch}: owned paths differ on readback: "
                               f"extra={sorted(owned - expected_owned)} missing={sorted(expected_owned - owned)}")
        if any(is_pruned(p, prune) for p in after):
            raise PublishError(f"{branch}: pruned paths still present on readback")
        unmanaged_after = {p: v for p, v in after.items() if p not in payload and p not in owned}
        if unmanaged_after != unmanaged_before:
            raise PublishError(f"{branch}: unmanaged Hub files changed during publication")
        digests = {}
        for path, data in sorted(payload.items()):
            got = self.run("cat-file", "blob", f"{observed}:{path}")
            if got != data:
                raise PublishError(f"{branch}: readback bytes differ: {path}")
            digests[path] = sha256_hex(got)
        return digests


def check_credential(token: str | None) -> None:
    if not token:
        raise PublishError("UNAVAILABLE - secrets.HF_TOKEN is empty for this workflow")
    request = urllib.request.Request(f"{HUB}/api/whoami-v2", headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            info = json.load(response)
    except urllib.error.HTTPError as exc:
        raise PublishError(
            f"UNAVAILABLE - HF_TOKEN rejected by whoami (HTTP {exc.code}). A repository-level HF_TOKEN "
            "secret shadows the org secret; check `gh secret list` for this repository.") from None
    auth = (info.get("auth") or {}).get("accessToken") or {}
    print(f"credential: valid, type={info.get('type')}, token_role={auth.get('role', 'unknown')}")


def publish_kernel(cfg: KernelConfig, repo_root: Path, commit: str, *, dry_run: bool,
                   token: str | None, url: str | None = None, workdir: Path | None = None) -> dict:
    payload = build_payload(cfg, repo_root, commit)
    receipt: dict[str, Any] = {
        "schema": "szl.kernel-publication/v1", "repo_id": cfg.repo_id, "repo_type": "kernel",
        "source_repository": cfg.source_repository, "source_commit": commit,
        "kernel_version": cfg.version, "dry_run": dry_run, "state": "HOLD", "branches": {},
        "payload_sha256": {p: sha256_hex(d) for p, d in sorted(payload.items())},
    }
    run_url = (f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/"
               f"{os.environ.get('GITHUB_REPOSITORY', cfg.source_repository)}/actions/runs/"
               f"{os.environ.get('GITHUB_RUN_ID', 'local')}")
    message = (f"Publish {cfg.name} v{cfg.version} from {cfg.source_repository}@{commit}\n\n"
               f"Source: https://github.com/{cfg.source_repository}/tree/{commit}\n"
               f"Publisher: {PUBLISHER} ({run_url})\n")
    owned = ("build/",)
    with tempfile.TemporaryDirectory(prefix="szl-kernel-") as scratch:
        hub = HubGit(url or cfg.kernel_url + ".git", Path(workdir or scratch) / "kernel", token)
        heads = hub.heads()
        if "main" not in heads:
            raise PublishError(f"{cfg.repo_id}: kernel repo or its main branch is missing "
                               "(create the kernel repo once by hand; CI never creates repositories)")
        vb = cfg.version_branch
        hub.fetch(["main"] + ([vb] if vb in heads else []))
        plan: dict[str, tuple[str | None, str, list[dict]]] = {}
        main_new, main_changes = hub.candidate("main", heads["main"], payload, owned, cfg.prune_paths, message)
        plan["main"] = (heads["main"], main_new, main_changes)
        if vb in heads:
            v_new, v_changes = hub.candidate(vb, heads[vb], payload, owned, cfg.prune_paths, message)
            plan[vb] = (heads[vb], v_new, v_changes)
        else:
            plan[vb] = (None, main_new, [{"op": "create-branch", "from": "main candidate"}])
        updates = {b: new for b, (old, new, _) in plan.items() if new != old}
        unmanaged = {}
        for branch, (old, _, _) in plan.items():
            base = old or heads["main"]
            unmanaged[branch] = {p: v for p, v in hub.tree(base).items()
                                 if p not in payload and not p.startswith(owned) and not is_pruned(p, cfg.prune_paths)}
        for branch, (old, new, changes) in plan.items():
            receipt["branches"][branch] = {"parent": old, "candidate": new, "changes": changes}
        if updates:
            hub.push(updates, dry_run=True)  # authenticated write + fast-forward preflight
            receipt["write_preflight"] = "passed"
        if dry_run:
            receipt["state"] = "PLAN_ONLY"
            return receipt
        hub.push(updates, dry_run=False)
        for branch, (_, new, _) in plan.items():
            receipt["branches"][branch]["readback_sha256"] = hub.readback(
                branch, new, payload, owned, cfg.prune_paths, unmanaged[branch])
            receipt["branches"][branch]["published"] = new
        receipt["state"] = "PUBLISHED" if updates else "NO_CHANGES"
    return receipt


def publish_twin(cfg: KernelConfig, repo_root: Path, commit: str, *, dry_run: bool,
                 token: str | None, url: str | None = None) -> dict:
    if cfg.twin is None:
        return {"state": "NOT_CONFIGURED", "repo_id": cfg.repo_id}
    text = normalize_text(cfg.twin.card, read_blob(repo_root, commit, cfg.twin.card)).decode("utf-8")
    text = render_card(cfg, text, cfg.twin.card)
    lint_card(cfg, text, cfg.twin.card, require_usage=True)
    payload = {"README.md": text.encode("utf-8")}
    receipt: dict[str, Any] = {"schema": "szl.kernel-twin-pointer/v1", "repo_id": cfg.twin.repo_id,
                               "repo_type": "model", "points_to": cfg.kernel_url, "dry_run": dry_run,
                               "prune_paths": list(cfg.twin.prune_paths), "state": "HOLD"}
    message = (f"Point model twin at the kernel repo {cfg.kernel_url}\n\n"
               "The kernels library (>=0.16) loads only the kernel repo; this model repo carries no build.\n"
               f"Source: https://github.com/{cfg.source_repository}/tree/{commit}\n")
    with tempfile.TemporaryDirectory(prefix="szl-twin-") as scratch:
        hub = HubGit(url or f"{HUB}/{cfg.twin.repo_id}.git", Path(scratch) / "twin", token)
        heads = hub.heads()
        if "main" not in heads:
            raise PublishError(f"model twin {cfg.twin.repo_id} has no main branch")
        hub.fetch(["main"])
        new, changes = hub.candidate("main", heads["main"], payload, (), cfg.twin.prune_paths, message)
        receipt.update(parent=heads["main"], candidate=new, changes=changes)
        unmanaged = {p: v for p, v in hub.tree(heads["main"]).items()
                     if p not in payload and not is_pruned(p, cfg.twin.prune_paths)}
        if new != heads["main"]:
            hub.push({"main": new}, dry_run=True)
        if dry_run:
            receipt["state"] = "PLAN_ONLY"
            return receipt
        if new != heads["main"]:
            hub.push({"main": new}, dry_run=False)
        receipt["readback_sha256"] = hub.readback("main", new, payload, (), cfg.twin.prune_paths, unmanaged)
        receipt["state"] = "PUBLISHED" if new != heads["main"] else "NO_CHANGES"
    return receipt


# ------------------------------------------------------------------ load + verify

def files_under(root: Path) -> dict[str, bytes]:
    out = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if "__pycache__" in rel.split("/") or rel.endswith(".pyc"):
            continue
        if path.is_file():
            out[rel] = path.read_bytes()
    return out


def compare_variant(served_root: Path, variant: str, payload: dict[str, bytes]) -> None:
    prefix = f"build/{variant}/"
    expected = {p[len(prefix):]: d for p, d in payload.items() if p.startswith(prefix)}
    served = files_under(served_root)
    if set(served) != set(expected):
        raise PublishError(f"{variant}: served file set differs from GitHub source: "
                           f"extra={sorted(set(served) - set(expected))} "
                           f"missing={sorted(set(expected) - set(served))}")
    for rel, data in expected.items():
        got = served[rel]
        if rel == "metadata.json":
            if json.loads(got) != json.loads(data):
                raise PublishError(f"{variant}: served metadata.json differs from the expected build")
        elif got.replace(b"\r\n", b"\n") != data:
            raise PublishError(f"{variant}: served bytes differ from GitHub source (normalised): {rel}")


def kernels_digest_check(variant_root: Path) -> str:
    """Cross-check the digest with the kernels library's own hasher, when exposed."""
    try:
        from kernels._rust import Digest, Metadata  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - private module moved
        return "SKIPPED (kernels._rust not importable)"
    try:
        metadata = Metadata.read_from_file(variant_root / "metadata.json")
    except Exception as exc:
        raise PublishError(f"{variant_root.name}: kernels cannot parse metadata.json: {exc}") from None
    if metadata.digest is None:
        raise PublishError(f"{variant_root.name}: metadata.json has no digest")
    try:
        metadata.digest.validate(Digest.hash_variant(metadata.digest.algorithm, variant_root))
    except Exception as exc:
        raise PublishError(f"{variant_root.name}: kernels digest validation failed: {exc}") from None
    return "PASS"


def load_smoke(cfg: KernelConfig, repo_root: Path, commit: str):
    if not cfg.smoke:
        return None
    namespace: dict[str, Any] = {"__name__": "szl_kernel_smoke"}
    code = read_blob(repo_root, commit, cfg.smoke).decode("utf-8")
    exec(compile(code, cfg.smoke, "exec"), namespace)  # repo-owned file at the published commit
    smoke = namespace.get("smoke")
    if not callable(smoke):
        raise PublishError(f"{cfg.smoke} must define smoke(kernel)")
    return smoke


def run_smoke(module, smoke, label: str) -> dict:
    report: dict[str, Any] = {"module": getattr(module, "__name__", None)}
    names = getattr(module, "__all__", None)
    if names is not None:
        missing = [n for n in names if not hasattr(module, n)]
        if missing:
            raise PublishError(f"{label}: __all__ names missing on the loaded kernel: {missing}")
        report["exports"] = len(names)
    if smoke is not None:
        result = smoke(module)
        if result is False:
            raise PublishError(f"{label}: smoke(kernel) returned False")
        report["smoke"] = repr(result)[:500]
    return report


def check_local(cfg: KernelConfig, repo_root: Path, commit: str, dist: Path) -> dict:
    import kernels  # noqa: F401 - imported for its loader
    from kernels import get_local_kernel

    payload = build_payload(cfg, repo_root, commit)
    smoke = load_smoke(cfg, repo_root, commit)
    report: dict[str, Any] = {"repo_id": cfg.repo_id, "source_commit": commit, "variants": {}}
    for variant in VARIANTS:
        root = dist / "build" / variant
        compare_variant(root, variant, payload)
        module = get_local_kernel(root)
        served = Path(module.__file__).resolve()
        if served.parent.parent != root.resolve() or served.parent.name != cfg.packages[0].name:
            raise PublishError(f"{variant}: loader imported {served}, not the built package")
        report["variants"][variant] = {"digest": kernels_digest_check(root),
                                       **run_smoke(module, smoke, f"local {variant}")}
    chosen = Path(get_local_kernel(dist).__file__).resolve().parent.parent.name
    report["preferred_variant_here"] = chosen
    report["state"] = "PASS"
    return report


def verify_hub(cfg: KernelConfig, repo_root: Path, source: str, ref: str, default_branch: str,
               release_sha: str | None) -> dict:
    sys.dont_write_bytecode = True
    from huggingface_hub import HfApi
    import kernels

    api = HfApi()
    branch = "main" if ref == "main" else cfg.version_branch
    refs = api.list_repo_refs(cfg.repo_id, repo_type="kernel")
    heads = {b.name: b.target_commit for b in refs.branches}
    if branch not in heads:
        raise PublishError(f"{cfg.repo_id}: branch {branch} does not exist on the kernel repo")
    hub_commit = heads[branch]
    report: dict[str, Any] = {"schema": "szl.kernel-hub-verification/v1", "repo_id": cfg.repo_id,
                              "repo_type": "kernel", "ref": branch, "hub_commit": hub_commit,
                              "kernels_version": kernels.__version__}
    if source == "auto":
        binding_path = api.hf_hub_download(cfg.repo_id, BINDING_PATH, repo_type="kernel", revision=hub_commit)
        binding = json.loads(Path(binding_path).read_bytes())
        source = binding.get("source_commit", "")
        if binding.get("schema") != BINDING_SCHEMA or not SHA40.fullmatch(source):
            raise PublishError(f"{branch}: {BINDING_PATH} missing or invalid; this ref was not published by "
                               f"{PUBLISHER}")
    source = resolve_commit(repo_root, source)
    if not is_ancestor(repo_root, source, f"origin/{default_branch}"):
        raise PublishError(f"served source {source} is not on origin/{default_branch}")
    report["source_commit"] = source
    payload = build_payload(cfg, repo_root, source)

    # 1. The call consumers make, in a fresh cache.
    kwargs = {"revision": "main"} if branch == "main" else {"version": cfg.version}
    module = kernels.get_kernel(cfg.repo_id, trust_remote_code=True, **kwargs)
    served = Path(module.__file__)  # snapshot path; do not resolve the blob symlinks
    variant_root = served.parent.parent
    if (served.parent.name != cfg.packages[0].name or variant_root.parent.name != "build"
            or variant_root.parent.parent.name != hub_commit):
        raise PublishError(f"get_kernel loaded {served}, not build/<variant>/{cfg.packages[0].name} "
                           f"of {branch}@{hub_commit}")
    loaded_variant = variant_root.name
    compare_variant(variant_root, loaded_variant, payload)
    smoke = load_smoke(cfg, repo_root, source)
    report["get_kernel"] = {"call": f"get_kernel({cfg.repo_id!r}, {', '.join(f'{k}={v!r}' for k, v in kwargs.items())}, "
                                    "trust_remote_code=True)",
                            "variant": loaded_variant, **run_smoke(module, smoke, f"get_kernel {branch}")}
    allow = kernels.get_kernel(cfg.repo_id, trust_remote_code=[cfg.repo_id], **kwargs)
    report["get_kernel"]["allowlist_form"] = "PASS" if allow is not None else "FAIL"

    # 2. Every managed Hub path, both variants, and nothing stale under build/.
    remote = set(api.list_repo_files(cfg.repo_id, repo_type="kernel", revision=hub_commit))
    build_remote = {p for p in remote if p.startswith("build/")}
    build_expected = {p for p in payload if p.startswith("build/")}
    if build_remote != build_expected:
        raise PublishError(f"{branch}: build/ differs from GitHub source: extra={sorted(build_remote - build_expected)} "
                           f"missing={sorted(build_expected - build_remote)}")
    stale = sorted(p for p in remote if is_pruned(p, cfg.prune_paths))
    if stale:
        raise PublishError(f"{branch}: paths that config prunes are still served: {stale}")
    snapshot = Path(api.snapshot_download(cfg.repo_id, repo_type="kernel", revision=hub_commit,
                                          allow_patterns=["build/*", *sorted(cfg.managed_files())],
                                          cache_dir=os.environ.get("KERNELS_CACHE")))
    report["variants"] = {}
    for variant in VARIANTS:
        compare_variant(snapshot / "build" / variant, variant, payload)
        report["variants"][variant] = {"bytes": "MATCH", "digest": kernels_digest_check(snapshot / "build" / variant)}
    universal = kernels.get_local_kernel(snapshot / "build" / "torch-universal")
    report["variants"]["torch-universal"].update(run_smoke(universal, smoke, f"{branch} torch-universal"))
    for path in sorted(cfg.managed_files()):
        got = (snapshot / path).read_bytes() if (snapshot / path).is_file() else None
        if got is None:
            raise PublishError(f"{branch}: managed file missing on the Hub: {path}")
        if path == BINDING_PATH:
            if json.loads(got) != json.loads(payload[path]):
                raise PublishError(f"{branch}: {BINDING_PATH} differs from the expected binding")
        elif got.replace(b"\r\n", b"\n") != payload[path]:
            raise PublishError(f"{branch}: {path} differs from GitHub source")
    if "README.md" in payload:
        lint_card(cfg, payload["README.md"].decode("utf-8"), "kernel README.md", require_usage=True)

    # 3. What the resolver decides on this host (CPU runner: torch-cpu preferred, universal compatible).
    decisions = kernels.get_kernel_variants(cfg.repo_id, **kwargs)
    report["variant_decisions"] = [
        {"variant": d.variant.variant_str, "accepted": type(d).__name__ == "VariantAccepted",
         "reason": getattr(d, "reason", None)} for d in decisions]
    accepted = {d["variant"] for d in report["variant_decisions"] if d["accepted"]}
    if "torch-universal" not in accepted:
        raise PublishError(f"{branch}: torch-universal is not accepted by the resolver: {report['variant_decisions']}")

    # 4. Drift guard: a newer release whose kernel content was never published.
    if release_sha:
        release_sha = resolve_commit(repo_root, release_sha)
        if content_key(build_payload(cfg, repo_root, release_sha)) != content_key(payload):
            raise PublishError(f"SOURCE_AHEAD: release {release_sha[:12]} changes the kernel but {branch} serves "
                               f"{source[:12]}; dispatch hf-kernel-publish to publish it")
        report["release_checked"] = release_sha
    report["state"] = "PASS"
    return report


# ------------------------------------------------------------------- source/plan

def latest_release(repo: str) -> str:
    result = subprocess.run(
        ["gh", "release", "list", "--repo", repo, "--exclude-drafts", "--exclude-pre-releases",
         "--limit", "1", "--json", "tagName", "--jq", '.[0].tagName // ""'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120, check=False)
    if result.returncode:
        raise PublishError(f"cannot list releases: {result.stderr.strip()[-500:]}")
    return result.stdout.strip()


def cmd_resolve_source(args, repo_root: Path) -> int:
    event = os.environ.get("GITHUB_EVENT_NAME", "")
    default = os.environ.get("DEFAULT_BRANCH") or "main"
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    mode, ref = "skip", ""
    if event == "pull_request":
        mode, ref = "check", "HEAD"
    elif event == "release":
        if os.environ.get("RELEASE_PRERELEASE") == "true":
            print("::notice::prerelease: nothing is published")
        else:
            mode, ref = "publish", os.environ.get("RELEASE_TAG", "")
    elif event == "workflow_dispatch":
        if os.environ.get("GITHUB_REF") != f"refs/heads/{default}":
            raise PublishError(f"dispatch from {default} only (the reviewed workflow and config publish)")
        ref = os.environ.get("INPUT_SOURCE_REF") or latest_release(repo)
        if not ref:
            raise PublishError("no published release: pass source_ref (a tag, branch or commit on the default branch)")
        mode = "plan" if os.environ.get("INPUT_DRY_RUN") == "true" else "publish"
    elif event == "schedule":
        ref = latest_release(repo)
        mode = "drift" if ref else "skip"
        if not ref:
            print("::notice::no release yet: drift guard skipped")
    else:
        raise PublishError(f"unsupported event {event!r}")
    sha = ""
    if mode != "skip":
        sha = resolve_commit(repo_root, "HEAD") if ref == "HEAD" else resolve_ref(repo_root, ref)
        if mode in ("publish", "plan") and not is_ancestor(repo_root, sha, f"origin/{default}"):
            raise PublishError(f"{ref} ({sha[:12]}) is not on origin/{default}; only reviewed commits are published")
    write_output("mode", mode)
    write_output("source_ref", ref)
    write_output("source_sha", sha)
    return 0


def cmd_plan(args, repo_root: Path) -> int:
    configs = load_config(repo_root, args.only or "")
    for cfg in configs:
        expected_repo = os.environ.get("GITHUB_REPOSITORY")
        if expected_repo and expected_repo.lower() != cfg.source_repository.lower():
            raise PublishError(f"{cfg.repo_id}: source_repository {cfg.source_repository} != {expected_repo}")
    matrix = [{"repo_id": c.repo_id, "slug": c.slug, "version": c.version, "twin": c.twin is not None}
              for c in configs]
    write_output("kernels", json.dumps(matrix, separators=(",", ":")))
    return 0


def _receipt(path: str | None, data: dict) -> None:
    text = json.dumps(data, indent=2, sort_keys=True) + "\n"
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text, encoding="utf-8")
    print(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("resolve-source", help="decide mode and source commit from the GitHub event")
    p = sub.add_parser("plan", help="emit the kernel matrix")
    p.add_argument("--only", default="")
    for name in ("build", "check-local", "publish", "verify", "twin"):
        p = sub.add_parser(name)
        p.add_argument("--kernel", required=True, help="kernel repo_id from the config")
        p.add_argument("--source", required=True, help="GitHub commit/ref (verify: 'auto' reads the Hub binding)")
        p.add_argument("--receipt")
        if name in ("build", "check-local"):
            p.add_argument("--out" if name == "build" else "--dist", required=True, type=Path)
        if name in ("publish", "twin"):
            p.add_argument("--dry-run", action="store_true")
        if name == "verify":
            p.add_argument("--ref", choices=("version", "main"), required=True)
            p.add_argument("--default-branch", default=os.environ.get("DEFAULT_BRANCH") or "main")
            p.add_argument("--release-sha", default="")
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    try:
        if args.command == "resolve-source":
            return cmd_resolve_source(args, root)
        if args.command == "plan":
            return cmd_plan(args, root)
        cfg = find_kernel(root, args.kernel)
        if args.source == "auto" and args.command != "verify":
            raise PublishError("--source auto is only valid for verify")
        source = args.source if args.source == "auto" else resolve_ref(root, args.source)
        if args.command == "build":
            payload = build_payload(cfg, root, source)
            write_dist(payload, args.out)
            _receipt(args.receipt, {"repo_id": cfg.repo_id, "source_commit": source, "state": "BUILT",
                                    "content_sha256": content_key(payload),
                                    "files": {p: sha256_hex(d) for p, d in sorted(payload.items())}})
        elif args.command == "check-local":
            _receipt(args.receipt, check_local(cfg, root, source, args.dist))
        elif args.command == "publish":
            token = os.environ.get("HF_TOKEN") or None
            check_credential(token)
            _receipt(args.receipt, publish_kernel(cfg, root, source, dry_run=args.dry_run, token=token))
        elif args.command == "twin":
            token = os.environ.get("HF_TOKEN") or None
            if cfg.twin is None:
                print(f"::notice::{cfg.repo_id}: no model_twin configured; twin left untouched")
                return 0
            check_credential(token)
            _receipt(args.receipt, publish_twin(cfg, root, source, dry_run=args.dry_run, token=token))
        elif args.command == "verify":
            _receipt(args.receipt, verify_hub(cfg, root, source, args.ref, args.default_branch,
                                              args.release_sha or None))
        return 0
    except Exception as exc:  # every failure is reported, with secrets scrubbed, and fails the step
        detail = scrub(f"{type(exc).__name__}: {exc}" if not isinstance(exc, PublishError) else str(exc))
        print(f"::error::{detail}", file=sys.stderr)
        if getattr(args, "receipt", None):
            _receipt(args.receipt, {"state": "FAIL", "error": detail})
        if not isinstance(exc, PublishError):
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

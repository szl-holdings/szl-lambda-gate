#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify Sigstore bundles on SZL kernel build variants (consumer + CI check).

For every ``build/<variant>/metadata.json`` under ``--root`` (a local tree, or a
fresh anonymous clone of ``--hub REPO`` at ``--revision``), checks that
``metadata.json.sigstore`` is a valid Sigstore bundle over the exact
metadata.json bytes, issued by GitHub Actions OIDC for ``--identity-repo``,
from the workflow ``--workflow`` on refs/heads/main or a refs/tags/v* release.
The file digests inside metadata.json are checked by the kernels loader.

Exit 0 only when every variant is signed and verified, or, without
``--require``, when unsigned variants are the only finding (reported as
UNSIGNED). Any present-but-invalid signature always fails closed.
Needs ``sigstore`` (pip install "kernels[sigstore]"). No credentials used.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

GITHUB_ISSUER = "https://token.actions.githubusercontent.com"
BUNDLE = "metadata.json.sigstore"


def san_pattern(identity_repo: str, workflow: str) -> re.Pattern:
    return re.compile(rf"^https://github\.com/{re.escape(identity_repo)}/\.github/workflows/"
                      rf"{re.escape(workflow)}@refs/(heads/main|tags/v[0-9][0-9A-Za-z.\-]*)$")


def verify_variant(meta: Path, identity_repo: str, workflow: str) -> dict:
    bundle_path = meta.with_name(BUNDLE)
    if not bundle_path.exists():
        return {"state": "UNSIGNED"}
    from cryptography.x509 import SubjectAlternativeName, UniformResourceIdentifier
    from sigstore.models import Bundle
    from sigstore.verify import Verifier
    from sigstore.verify.policy import AllOf, GitHubWorkflowRepository, OIDCIssuer

    bundle = Bundle.from_json(bundle_path.read_bytes())
    policy = AllOf([OIDCIssuer(GITHUB_ISSUER), GitHubWorkflowRepository(identity_repo)])
    Verifier.production().verify_artifact(meta.read_bytes(), bundle, policy)  # raises on failure
    sans = bundle.signing_certificate.extensions.get_extension_for_class(SubjectAlternativeName)
    uris = sans.value.get_values_for_type(UniformResourceIdentifier)
    pattern = san_pattern(identity_repo, workflow)
    if not any(pattern.fullmatch(u) for u in uris):
        raise ValueError(f"signing identity {uris} is not {identity_repo} {workflow} on main or a v* tag")
    return {"state": "VERIFIED", "identity": uris[0]}


def clone(hub: str, revision: str, into: Path) -> Path:
    url = f"https://huggingface.co/kernels/{hub}"
    subprocess.run(["git", "-c", "credential.helper=", "clone", "-q", "--depth", "1", "--branch", revision,
                    "--filter=blob:none", "--sparse", url, str(into)], check=True,
                   env={"GIT_TERMINAL_PROMPT": "0", "GIT_LFS_SKIP_SMUDGE": "1", "PATH": "/usr/bin:/bin"})
    subprocess.run(["git", "-C", str(into), "sparse-checkout", "set", "build"], check=True)
    return into


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--root", type=Path, help="tree that contains build/<variant>/metadata.json")
    src.add_argument("--hub", help="kernel repo id, e.g. SZLHOLDINGS/szl-block-kv (anonymous clone)")
    ap.add_argument("--revision", default="v1")
    ap.add_argument("--identity-repo", required=True, help="GitHub owner/repo whose workflow must have signed")
    ap.add_argument("--workflow", default="hf-kernel-publish.yml")
    ap.add_argument("--require", action="store_true", help="fail when any variant is unsigned")
    ap.add_argument("--report", type=Path)
    a = ap.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="szl-sig-") as tmp:
        root = a.root if a.root else clone(a.hub, a.revision, Path(tmp) / "k")
        metas = sorted((root / "build").glob("*/metadata.json"))
        report = {"schema": "szl.kernel-signature-check/v1", "identity_repo": a.identity_repo,
                  "source": a.hub or str(a.root), "revision": a.revision if a.hub else None, "variants": {}}
        failed = False
        if not metas:
            report["error"] = "no build/<variant>/metadata.json found"
            failed = True
        for meta in metas:
            name = meta.parent.name
            try:
                row = verify_variant(meta, a.identity_repo, a.workflow)
            except Exception as exc:  # any present-but-bad signature fails closed
                row = {"state": "INVALID", "error": f"{type(exc).__name__}: {exc}"[:400]}
            report["variants"][name] = row
            if row["state"] == "INVALID" or (row["state"] == "UNSIGNED" and a.require):
                failed = True
        report["state"] = "FAIL" if failed else "PASS"
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if a.report:
        a.report.parent.mkdir(parents=True, exist_ok=True)
        a.report.write_text(text + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

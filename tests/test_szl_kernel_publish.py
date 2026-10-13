# SPDX-License-Identifier: Apache-2.0
"""Offline tests for scripts/szl_kernel_publish.py (stdlib + git only).

A temporary Git repository plays the GitHub source and local bare repositories
play the Hub kernel repo and model twin, so build, publish, prune, version-branch
creation, idempotency and the twin pointer run end to end without network.
"""
from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "szl_kernel_publish.py"
spec = importlib.util.spec_from_file_location("szl_kernel_publish", SCRIPT)
kp = importlib.util.module_from_spec(spec)
sys.modules["szl_kernel_publish"] = kp
spec.loader.exec_module(kp)

ENV = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
       "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid"}
ENV.pop("GIT_INDEX_FILE", None)

CARD = """---
license: apache-2.0
library_name: kernels
---
```python
kernel = get_kernel("{{REPO_ID}}", version={{KERNEL_VERSION}}, trust_remote_code=True)
```
"""


def g(repo: Path, *args: str, data: bytes | None = None) -> str:
    return kp.git(repo, *args, data=data, env=ENV).decode().strip()


def config(**overrides) -> dict:
    entry = {
        "repo_id": "SZLHOLDINGS/szl-demo", "name": "szl-demo", "version": 1,
        "license": "Apache-2.0", "source_repository": "szl-holdings/szl-demo",
        "packages": [{"src": "torch-ext/szl_demo", "name": "szl_demo"}],
        "card": "hf/kernel-card.md", "license_file": "LICENSE",
        "extra_files": {"spec/demo.json": "spec/demo.json"},
        "prune_paths": ["OPERATIONAL.json"],
        "model_twin": {"repo_id": "SZLHOLDINGS/szl-demo", "mode": "pointer",
                       "card": "hf/kernel-card.md", "prune_paths": ["build/", "torch-ext/"]},
    }
    entry.update(overrides)
    return {"schema": kp.CONFIG_SCHEMA, "kernels": [entry]}


class Fixture(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.src = self.tmp / "src"
        self.src.mkdir()
        g(self.src, "init", "-q", "-b", "main")
        self.write({
            "torch-ext/szl_demo/__init__.py": b"from ._ops import add\r\n__all__ = ['add']\r\n",
            "torch-ext/szl_demo/_ops.py": b"def add(a, b):\n    return a + b\n",
            "torch-ext/szl_demo/data/table.json": b'{"k": 1}\n',
            "LICENSE": b"Apache License 2.0\n",
            "spec/demo.json": b'{"spec": true}\n',
            "hf/kernel-card.md": CARD.encode(),
            ".github/hf-kernels.json": json.dumps(config()).encode(),
        })
        self.commit = self.commit_all("source v1")

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, files: dict[str, bytes]) -> None:
        for path, data in files.items():
            target = self.src / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)

    def commit_all(self, message: str) -> str:
        g(self.src, "-c", "core.autocrlf=false", "add", "-A")
        g(self.src, "commit", "-q", "-m", message)
        return g(self.src, "rev-parse", "HEAD")

    def cfg(self):
        return kp.find_kernel(self.src, "SZLHOLDINGS/szl-demo")

    def bare(self, name: str, files: dict[str, bytes], branches: tuple[str, ...] = ("main",)) -> str:
        bare = self.tmp / f"{name}.git"
        g(self.tmp, "init", "-q", "--bare", "-b", "main", str(bare))
        work = self.tmp / f"{name}-seed"
        work.mkdir()
        g(work, "init", "-q", "-b", "main")
        for path, data in files.items():
            (work / path).parent.mkdir(parents=True, exist_ok=True)
            (work / path).write_bytes(data)
        g(work, "add", "-A")
        g(work, "commit", "-q", "-m", "seed")
        for branch in branches:
            g(work, "push", "-q", bare.as_uri(), f"HEAD:refs/heads/{branch}")
        return bare.as_uri()

    def hub_files(self, url: str, branch: str) -> dict[str, bytes]:
        work = self.tmp / f"read-{len(list(self.tmp.iterdir()))}"
        work.mkdir()
        g(work, "init", "-q")
        g(work, "fetch", "-q", url, f"refs/heads/{branch}")
        rows = g(work, "ls-tree", "-r", "--name-only", "FETCH_HEAD").splitlines()
        return {p: kp.git(work, "cat-file", "blob", f"FETCH_HEAD:{p}") for p in rows}


class BuildTests(Fixture):
    def test_layout_metadata_and_normalisation(self):
        payload = kp.build_payload(self.cfg(), self.src, self.commit)
        for variant in kp.VARIANTS:
            self.assertEqual(payload[f"build/{variant}/szl_demo/__init__.py"],
                             b"from ._ops import add\n__all__ = ['add']\n")
            self.assertIn(f"build/{variant}/szl_demo/data/table.json", payload)
            self.assertNotIn(f"build/{variant}/__init__.py", payload)  # no shim: the module IS the package
            meta = json.loads(payload[f"build/{variant}/metadata.json"])
            self.assertEqual(meta["name"], "szl-demo")
            self.assertEqual(meta["version"], 1)
            self.assertEqual(meta["backend"], {"type": "cpu"})
            self.assertEqual(meta["python-depends"], [])
            self.assertTrue(meta["id"].startswith(f"_szl_demo_{variant.split('-')[1]}_"))
            files = {p[len(f"build/{variant}/"):]: d for p, d in payload.items()
                     if p.startswith(f"build/{variant}/") and not p.endswith("/metadata.json")}
            self.assertEqual(meta["digest"]["files"],
                             {k: base64.b64encode(hashlib.sha256(v).digest()).decode() for k, v in files.items()})
        cpu = {p.split("/", 2)[2]: d for p, d in payload.items() if p.startswith("build/torch-cpu/")
               and not p.endswith("metadata.json")}
        uni = {p.split("/", 2)[2]: d for p, d in payload.items() if p.startswith("build/torch-universal/")
               and not p.endswith("metadata.json")}
        self.assertEqual(cpu, uni)
        binding = json.loads(payload[kp.BINDING_PATH])
        self.assertEqual(binding["source_commit"], self.commit)
        self.assertIn('version=1', payload["README.md"].decode())
        self.assertEqual(payload["spec/demo.json"], b'{"spec": true}\n')

    def test_deterministic_and_content_key_ignores_commit(self):
        a = kp.build_payload(self.cfg(), self.src, self.commit)
        self.assertEqual(a, kp.build_payload(self.cfg(), self.src, self.commit))
        self.write({"README.md": b"unrelated\n"})
        b = kp.build_payload(self.cfg(), self.src, self.commit_all("unrelated change"))
        self.assertNotEqual(a[kp.BINDING_PATH], b[kp.BINDING_PATH])
        self.assertEqual(kp.content_key(a), kp.content_key(b))

    def test_binary_file_is_refused(self):
        self.write({"torch-ext/szl_demo/weights.bin": b"\x00\x01\x02"})
        commit = self.commit_all("binary")
        with self.assertRaisesRegex(kp.PublishError, "binary file"):
            kp.build_payload(self.cfg(), self.src, commit)

    def test_symlink_is_refused(self):
        oid = g(self.src, "hash-object", "-w", "--stdin", data=b"_ops.py")
        g(self.src, "update-index", "--add", "--cacheinfo", f"120000,{oid},torch-ext/szl_demo/link.py")
        g(self.src, "commit", "-q", "-m", "symlink")
        with self.assertRaisesRegex(kp.PublishError, "regular files"):
            kp.build_payload(self.cfg(), self.src, g(self.src, "rev-parse", "HEAD"))

    def test_card_lint(self):
        for bad in ('get_kernel("{{REPO_ID}}", revision="main", trust_remote_code=True) version=1',
                    'get_kernel("{{REPO_ID}}") version=1',
                    'get_kernel("{{REPO_ID}}", revision="REPLACE_WITH_COMMIT") version=1'):
            self.write({"hf/kernel-card.md": bad.encode()})
            commit = self.commit_all("bad card")
            with self.assertRaisesRegex(kp.PublishError, "card lint"):
                kp.build_payload(self.cfg(), self.src, commit)

    def test_unknown_card_token(self):
        self.write({"hf/kernel-card.md": (CARD + "{{NOPE}}").encode()})
        with self.assertRaisesRegex(kp.PublishError, "unknown card tokens"):
            kp.build_payload(self.cfg(), self.src, self.commit_all("token"))

    def test_config_validation(self):
        bad = [
            {"name": "SZL-Demo"},
            {"packages": [{"src": "torch-ext/szl_demo", "name": "other"}]},
            {"prune_paths": ["build/"]},
            {"prune_paths": ["LICENSE"]},
            {"version": 0},
            {"model_twin": {"repo_id": "SZLHOLDINGS/szl-demo", "mode": "mirror", "card": "x.md"}},
            {"extra_files": {"spec/demo.json": "build/x.json"}},
            {"source_repository": "someone/else"},
        ]
        for override in bad:
            with self.subTest(override=override):
                with self.assertRaises(kp.PublishError):
                    kp.parse_kernel(config(**override)["kernels"][0], 0)


class PublishTests(Fixture):
    def setUp(self):
        super().setUp()
        self.hub = self.bare("kernel", {
            ".gitattributes": b"*.bin filter=lfs diff=lfs merge=lfs -text\n",
            "README.md": b"old card revision main\n",
            "OPERATIONAL.json": b'{"load_path": "main"}\n',
            "BENCH.json": b"{}\n",
            "build/torch-cpu/szl_demo/__init__.py": b"stale\n",
            "build/torch-cpu/szl_demo/superseded.py": b"stale\n",
            "build/torch-cuda/szl_demo/__init__.py": b"would win on CUDA\n",
        })

    def publish(self, commit: str, dry_run: bool = False) -> dict:
        return kp.publish_kernel(self.cfg(), self.src, commit, dry_run=dry_run, token=None, url=self.hub)

    def test_publish_main_and_new_version_branch(self):
        receipt = self.publish(self.commit)
        self.assertEqual(receipt["state"], "PUBLISHED")
        payload = kp.build_payload(self.cfg(), self.src, self.commit)
        for branch in ("main", "v1"):
            files = self.hub_files(self.hub, branch)
            build = {p for p in files if p.startswith("build/")}
            self.assertEqual(build, {p for p in payload if p.startswith("build/")}, branch)
            for path, data in payload.items():
                self.assertEqual(files[path], data, f"{branch}:{path}")
            self.assertNotIn("OPERATIONAL.json", files)          # pruned by config
            self.assertEqual(files["BENCH.json"], b"{}\n")       # unmanaged, preserved
            self.assertIn(".gitattributes", files)
        self.assertEqual(receipt["branches"]["v1"]["parent"], None)

    def test_idempotent_and_dry_run(self):
        self.publish(self.commit)
        again = self.publish(self.commit)
        self.assertEqual(again["state"], "NO_CHANGES")
        self.write({"torch-ext/szl_demo/_ops.py": b"def add(a, b):\n    return b + a\n"})
        commit2 = self.commit_all("fix")
        before = (self.hub_files(self.hub, "main"), self.hub_files(self.hub, "v1"))
        plan = self.publish(commit2, dry_run=True)
        self.assertEqual(plan["state"], "PLAN_ONLY")
        self.assertEqual(plan["write_preflight"], "passed")
        self.assertEqual((self.hub_files(self.hub, "main"), self.hub_files(self.hub, "v1")), before)
        receipt = self.publish(commit2)
        self.assertEqual(receipt["state"], "PUBLISHED")
        self.assertIsNotNone(receipt["branches"]["v1"]["parent"])  # existing v1 fast-forwarded
        for branch in ("main", "v1"):
            self.assertEqual(self.hub_files(self.hub, branch)["build/torch-universal/szl_demo/_ops.py"],
                             b"def add(a, b):\n    return b + a\n")

    def test_version_bump_leaves_previous_branch(self):
        self.publish(self.commit)
        v1 = self.hub_files(self.hub, "v1")
        cfg2 = json.loads((self.src / ".github/hf-kernels.json").read_text())
        cfg2["kernels"][0]["version"] = 2
        self.write({".github/hf-kernels.json": json.dumps(cfg2).encode(),
                    "hf/kernel-card.md": CARD.encode()})
        commit2 = self.commit_all("v2 api")
        kp.publish_kernel(self.cfg(), self.src, commit2, dry_run=False, token=None, url=self.hub)
        self.assertEqual(self.hub_files(self.hub, "v1"), v1)
        self.assertEqual(json.loads(self.hub_files(self.hub, "v2")["build/torch-cpu/metadata.json"])["version"], 2)

    def test_missing_main_refuses(self):
        empty = self.tmp / "empty.git"
        g(self.tmp, "init", "-q", "--bare", "-b", "main", str(empty))
        with self.assertRaisesRegex(kp.PublishError, "main branch is missing"):
            kp.publish_kernel(self.cfg(), self.src, self.commit, dry_run=False, token=None, url=empty.as_uri())


class TwinTests(Fixture):
    def test_pointer(self):
        twin = self.bare("twin", {
            "README.md": b"stale card\n",
            "build/torch-universal/szl_demo/__init__.py": b"stale\n",
            "torch-ext/szl_demo/__init__.py": b"stale\n",
            "bom/model-bom.cdx.json": b"{}\n",
        })
        receipt = kp.publish_twin(self.cfg(), self.src, self.commit, dry_run=False, token=None, url=twin)
        self.assertEqual(receipt["state"], "PUBLISHED")
        files = self.hub_files(twin, "main")
        self.assertEqual(set(files), {"README.md", "bom/model-bom.cdx.json"})
        self.assertIn(b'get_kernel("SZLHOLDINGS/szl-demo", version=1', files["README.md"])
        again = kp.publish_twin(self.cfg(), self.src, self.commit, dry_run=False, token=None, url=twin)
        self.assertEqual(again["state"], "NO_CHANGES")


if __name__ == "__main__":
    unittest.main()

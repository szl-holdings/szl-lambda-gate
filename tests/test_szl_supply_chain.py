# SPDX-License-Identifier: Apache-2.0
"""Offline tests: metadata provenance, signature attach (fail closed), allowlisted load call."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_szl_kernel_publish import Fixture, PublishTests, kp  # noqa: E402

BUNDLE = json.dumps({"mediaType": "application/vnd.dev.sigstore.bundle.v0.3+json",
                     "verificationMaterial": {}, "messageSignature": {}}).encode()


def signed_dir(tc, payload, bundle=BUNDLE, tamper=None):
    out = tc.tmp / "signed"
    kp.write_dist(payload, out)
    for v in kp.VARIANTS:
        (out / f"build/{v}/{kp.SIGNATURE_NAME}").write_bytes(bundle)
    if tamper:
        (out / f"build/{tamper}/metadata.json").write_bytes(b"{}")
    return out


class ProvenanceTests(Fixture):
    def test_metadata_has_no_partial_provenance(self):
        # kernels 0.18.3 refuses a provenance block without kernel-builder, and this is
        # not a kernel-builder build; the commit is bound by id + SZL_KERNEL_SOURCE.json.
        payload = kp.build_payload(self.cfg(), self.src, self.commit)
        for v in kp.VARIANTS:
            meta = json.loads(payload[f"build/{v}/metadata.json"])
            self.assertNotIn("provenance", meta)
            self.assertTrue(meta["id"].endswith(self.commit[:12]))
        binding = json.loads(payload[kp.BINDING_PATH])
        self.assertEqual(binding["source_commit"], self.commit)

    def test_binding_load_call_uses_repo_allowlist(self):
        cfg = self.cfg()
        binding = json.loads(kp.build_payload(cfg, self.src, self.commit)[kp.BINDING_PATH])
        self.assertEqual(binding["load"]["call"],
                         f'get_kernel("{cfg.repo_id}", version=1, trust_remote_code=["{cfg.repo_id}"])')
        self.assertNotIn("trust_remote_code=True", binding["load"]["call"])

    def test_kernels_parses_provenance(self):
        try:
            from kernels._rust import Metadata  # type: ignore[attr-defined]
        except Exception:
            self.skipTest("kernels not installed")
        out = self.tmp / "dist"
        kp.write_dist(kp.build_payload(self.cfg(), self.src, self.commit), out)
        for v in kp.VARIANTS:
            self.assertIsNotNone(Metadata.read_from_file(out / f"build/{v}/metadata.json").digest)
            self.assertEqual(kp.kernels_digest_check(out / f"build/{v}"), "PASS")


class SignatureAttachTests(Fixture):
    def test_attach_adds_bundles(self):
        payload = kp.build_payload(self.cfg(), self.src, self.commit)
        added = kp.attach_signatures(payload, signed_dir(self, dict(payload)))
        self.assertEqual(set(added), set(kp.VARIANTS))
        for v in kp.VARIANTS:
            self.assertEqual(payload[f"build/{v}/{kp.SIGNATURE_NAME}"], BUNDLE)

    def test_attach_refuses_mismatched_metadata(self):
        payload = kp.build_payload(self.cfg(), self.src, self.commit)
        with self.assertRaisesRegex(kp.PublishError, "differs"):
            kp.attach_signatures(payload, signed_dir(self, dict(payload), tamper="torch-cpu"))

    def test_attach_refuses_missing_or_non_bundle(self):
        payload = kp.build_payload(self.cfg(), self.src, self.commit)
        out = signed_dir(self, dict(payload))
        (out / f"build/torch-universal/{kp.SIGNATURE_NAME}").unlink()
        with self.assertRaisesRegex(kp.PublishError, "missing"):
            kp.attach_signatures(dict(payload), out)
        (out / f"build/torch-universal/{kp.SIGNATURE_NAME}").write_bytes(b'{"mediaType": "text/plain"}')
        with self.assertRaisesRegex(kp.PublishError, "not a Sigstore bundle"):
            kp.attach_signatures(dict(payload), out)

    def test_compare_variant_ignores_bundle_only(self):
        payload = kp.build_payload(self.cfg(), self.src, self.commit)
        out = signed_dir(self, dict(payload))
        kp.compare_variant(out / "build/torch-cpu", "torch-cpu", payload)  # bundle tolerated
        (out / "build/torch-cpu/szl_demo/_ops.py").write_bytes(b"evil\n")
        with self.assertRaises(kp.PublishError):
            kp.compare_variant(out / "build/torch-cpu", "torch-cpu", payload)


class SignedPublishTests(PublishTests):
    def test_signed_publish_ships_bundles_and_is_idempotent(self):
        payload = kp.build_payload(self.cfg(), self.src, self.commit)
        out = signed_dir(self, dict(payload))
        r = kp.publish_kernel(self.cfg(), self.src, self.commit, dry_run=False, token=None, url=self.hub,
                              signatures=out)
        self.assertEqual(r["state"], "PUBLISHED")
        self.assertEqual(set(r["signatures"]), set(kp.VARIANTS))
        for branch in ("main", "v1"):
            files = self.hub_files(self.hub, branch)
            for v in kp.VARIANTS:
                self.assertEqual(files[f"build/{v}/{kp.SIGNATURE_NAME}"], BUNDLE)
        again = kp.publish_kernel(self.cfg(), self.src, self.commit, dry_run=False, token=None, url=self.hub,
                                  signatures=out)
        self.assertEqual(again["state"], "NO_CHANGES")

    def test_unsigned_publish_still_works(self):
        r = kp.publish_kernel(self.cfg(), self.src, self.commit, dry_run=False, token=None, url=self.hub)
        self.assertEqual(r["signatures"], "NONE")

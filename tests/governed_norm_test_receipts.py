# SPDX-License-Identifier: Apache-2.0
# © 2026 SZL Holdings
"""Governance-receipt tests for szl_governed_norm.

Verifies the provenance doctrine actually holds:
  * A freshly built ReceiptChain hash-chains and verify()s ok.
  * Tampering with a record's out_digest is detected: verify() returns
    ok=False with the correct first_break_seq.
  * The tensor digest is deterministic for identical logical inputs.
  * The module-level governed surface (receipt_count/head/tail/verify and
    governed=True) behaves consistently.

All tests build their own ReceiptChain objects for tamper experiments so the
process-wide default chain is never corrupted. CPU-only.

Run:  python -m pytest tests/test_receipts.py -q
"""
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "torch-ext"))
import szl_lambda_gate.governed_norm as gn  # noqa: E402  (folded-in governed-norm kernels)

# ReceiptChain is part of the public surface; guard in case the API shifts.
ReceiptChain = getattr(gn, "ReceiptChain", None)
_HAS_CHAIN = ReceiptChain is not None
# The digest helper is internal; reach it defensively for the determinism test.
try:
    from szl_lambda_gate.governed_norm._receipt import _tensor_digest  # type: ignore
    _HAS_DIGEST = True
except Exception:  # pragma: no cover - only if internal layout changes
    _tensor_digest = None
    _HAS_DIGEST = False

requires_chain = pytest.mark.skipif(not _HAS_CHAIN, reason="ReceiptChain not exported")


def _build_chain(n=3):
    """Emit n receipts into a fresh chain via its public emit() method."""
    chain = ReceiptChain()
    torch.manual_seed(0)
    for i in range(n):
        x = torch.randn(2, 64, dtype=torch.float32)
        out = gn.rms_norm(x, eps=1e-6)  # plain compute; chain.emit records it
        chain.emit("rms_norm", x, out, 1e-6)
    return chain


# --- chain integrity -----------------------------------------------------------
@requires_chain
def test_fresh_chain_verifies():
    chain = _build_chain(4)
    assert chain.count() == 4
    ok, depth, brk = chain.verify()
    assert ok is True
    assert depth == 4
    assert brk == -1
    assert len(chain.head()) == 64  # SHA3-256 hex digest


@requires_chain
def test_empty_chain_is_genesis():
    chain = ReceiptChain()
    assert chain.count() == 0
    assert chain.head() == "0" * 64
    ok, depth, brk = chain.verify()
    assert ok is True
    assert depth == 0
    assert brk == -1


# --- tamper detection ----------------------------------------------------------
@requires_chain
def test_tamper_out_digest_detected_at_right_seq():
    """Mutate a middle record's out_digest; verify() must flag that seq."""
    chain = _build_chain(5)
    assert chain.verify()[0] is True

    target_seq = 2
    # Reach into the append-only store and corrupt one field. The stored
    # body digest no longer matches the recomputed digest, so verify() should
    # report the FIRST broken sequence index.
    records = chain._records  # internal list; deliberate tamper for the test
    original = records[target_seq]["out_digest"]
    records[target_seq]["out_digest"] = "f" * 64
    assert records[target_seq]["out_digest"] != original

    ok, depth, first_break = chain.verify()
    assert ok is False
    assert depth == 5
    assert first_break == target_seq


@requires_chain
def test_tamper_prev_link_detected():
    """Breaking the prev-hash linkage is also caught."""
    chain = _build_chain(4)
    target_seq = 1
    chain._records[target_seq]["prev"] = "a" * 64
    ok, _, first_break = chain.verify()
    assert ok is False
    assert first_break == target_seq


# --- digest determinism --------------------------------------------------------
@pytest.mark.skipif(not _HAS_DIGEST, reason="_tensor_digest not available")
def test_tensor_digest_deterministic_for_identical_inputs():
    torch.manual_seed(42)
    a = torch.randn(3, 128, dtype=torch.float32)
    b = a.clone()
    assert _tensor_digest(a) == _tensor_digest(b)
    # A genuinely different tensor yields a different digest.
    c = a.clone()
    c[0, 0] += 1.0
    assert _tensor_digest(a) != _tensor_digest(c)


@pytest.mark.skipif(not _HAS_DIGEST, reason="_tensor_digest not available")
def test_tensor_digest_stable_across_equal_dtypes():
    """Same logical values via different dtype routes -> same rounded digest."""
    torch.manual_seed(11)
    base = torch.randn(4, 32, dtype=torch.float32)
    # float64 then back to float32 preserves these values exactly enough that
    # rounding to the digest's fixed decimals lands on the same integers.
    same = base.to(torch.float64).to(torch.float32)
    assert _tensor_digest(base) == _tensor_digest(same)


@requires_chain
def test_identical_calls_produce_identical_out_digest():
    """Two fresh chains with identical inputs record identical out_digests."""
    torch.manual_seed(5)
    x = torch.randn(2, 64, dtype=torch.float32)
    out = gn.rms_norm(x, eps=1e-6)

    c1, c2 = ReceiptChain(), ReceiptChain()
    r1 = c1.emit("rms_norm", x, out, 1e-6)
    r2 = c2.emit("rms_norm", x, out, 1e-6)
    assert r1["out_digest"] == r2["out_digest"]
    # First-record digest only depends on the body (seq/op/shape/.../prev),
    # which is identical, so the chain digests match too.
    assert r1["digest"] == r2["digest"]


# --- module-level governed surface --------------------------------------------
def test_module_governed_surface_consistent():
    start = gn.receipt_count()
    x = torch.randn(4, 128, dtype=torch.float32)
    gn.rms_norm(x, eps=1e-6, governed=True)
    gn.layer_norm(x, eps=1e-5, governed=True)
    assert gn.receipt_count() == start + 2

    v = gn.receipt_verify()
    assert v["ok"] is True
    assert v["depth"] == gn.receipt_count()
    assert v["first_break_seq"] == -1
    assert len(gn.receipt_head()) == 64

    tail = gn.receipt_tail(2)
    assert len(tail) == 2
    assert tail[-1]["op"] == "layer_norm"
    assert tail[0]["op"] == "rms_norm"


def test_governed_off_records_nothing():
    before = gn.receipt_count()
    gn.rms_norm(torch.randn(2, 32, dtype=torch.float32), eps=1e-6)  # governed default False
    gn.layer_norm(torch.randn(2, 32, dtype=torch.float32), eps=1e-5)
    assert gn.receipt_count() == before


# --- canonical szl-receipt v0.2.0 evidence binding (emit_receipt) --------------
_HAS_SZL_RECEIPT = importlib.util.find_spec("szl_receipt") is not None
requires_szl_receipt = pytest.mark.skipif(
    not _HAS_SZL_RECEIPT, reason="szl-receipt not installed")


@requires_szl_receipt
def test_emit_receipt_binds_subject_input_output_policy_energy():
    """The binding carries subject + input digest + output digest + policy id +
    energy=='UNAVAILABLE', and is UNSIGNED-honest (no fabricated signature)."""
    torch.manual_seed(7)
    x = torch.randn(2, 64, dtype=torch.float32)
    out = gn.rms_norm(x, eps=1e-6)
    binding = gn.emit_receipt(
        "rms_norm", x, out, 1e-6,
        subject="szl-governed-norm/rms_norm#0",
        policy_id="szl-governed-norm/provenance@v1",
    )
    assert binding is not None
    assert binding["subject"] == "szl-governed-norm/rms_norm#0"
    assert len(binding["input_digest"]) == 64   # SHA3-256 hex
    assert len(binding["output_digest"]) == 64  # SHA3-256 hex
    assert binding["policy_id"] == "szl-governed-norm/provenance@v1"
    # HONESTY: no joules are measured by this kernel -> literal "UNAVAILABLE",
    # never a fabricated number.
    assert binding["energy"] == "UNAVAILABLE"
    # HONESTY: keyless -> UNSIGNED-honest envelope; a signature is never faked.
    env = binding["signature"]
    assert env["signed"] is False
    assert env["signature"] == ""
    assert "UNSIGNED-honest" in env["note"]


@requires_szl_receipt
def test_emit_receipt_input_digest_binds_shape_dtype_eps():
    """Changing eps (part of the input spec) changes the input digest."""
    torch.manual_seed(1)
    x = torch.randn(2, 64, dtype=torch.float32)
    out = gn.rms_norm(x, eps=1e-6)
    b1 = gn.emit_receipt("rms_norm", x, out, 1e-6)
    b2 = gn.emit_receipt("rms_norm", x, out, 1e-5)
    assert b1["input_digest"] != b2["input_digest"]
    # Identical spec -> identical, deterministic input digest.
    b3 = gn.emit_receipt("rms_norm", x, out, 1e-6)
    assert b1["input_digest"] == b3["input_digest"]


@requires_szl_receipt
def test_governed_chain_carries_canonical_binding():
    """A governed emit additively carries the canonical binding; the output
    digest reuses the EXISTING SHA3-256 rounded-tensor digest and the additive
    binding does not disturb the SHA3-256 hash chain."""
    chain = ReceiptChain()
    torch.manual_seed(3)
    x = torch.randn(2, 64, dtype=torch.float32)
    out = gn.rms_norm(x, eps=1e-6)
    rec = chain.emit("rms_norm", x, out, 1e-6)
    binding = rec["receipt"]
    assert binding["energy"] == "UNAVAILABLE"
    assert binding["output_digest"] == rec["out_digest"]  # reuses existing digest
    assert binding["subject"].endswith("/rms_norm#0")
    assert binding["signature"]["signed"] is False  # keyless => no fake signature
    ok, _, brk = chain.verify()
    assert ok and brk == -1


def test_emit_receipt_import_guarded_returns_none(monkeypatch):
    """With szl-receipt ABSENT the canonical binding is skipped (returns None).

    The universal kernel must import and run on stdlib+torch+numpy alone, so the
    binding is import-guarded. Simulate absence by making ``import szl_receipt``
    fail, then confirm emit_receipt returns None and the chain still records
    (just without the optional binding), verifying cleanly.
    """
    import szl_lambda_gate.governed_norm._receipt as rc
    monkeypatch.setitem(sys.modules, "szl_receipt", None)  # -> ImportError on import
    torch.manual_seed(9)
    x = torch.randn(2, 32, dtype=torch.float32)
    out = gn.rms_norm(x, eps=1e-6)
    assert rc.emit_receipt("rms_norm", x, out, 1e-6) is None
    chain = ReceiptChain()
    rec = chain.emit("rms_norm", x, out, 1e-6)
    assert "receipt" not in rec  # guarded path skipped the optional binding
    ok, _, brk = chain.verify()
    assert ok and brk == -1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

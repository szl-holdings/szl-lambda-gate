# SPDX-License-Identifier: Apache-2.0
# © 2026 SZL Holdings · Stephen P. Lutar · ORCID 0009-0001-0110-4173
"""Content-addressed governance receipts for normalization calls.

SZL Holdings' provenance doctrine applied at the kernel layer: when a
normalization runs in *governed* mode, it emits a small, deterministic
receipt describing the call — input shape/dtype, eps, and a SHA3-256 digest
of the (quantized) output tensor — and hash-chains it to the previous
receipt. This makes a sequence of kernel calls independently auditable
without trusting the caller.

HONESTY:
- The digest is a real SHA3-256 over the output bytes (rounded to a fixed
  decimal precision so it is reproducible across runs/devices). It is an
  integrity fingerprint, NOT a cryptographic signature — we never claim
  it proves authorship. DSSE signing is a separate, out-of-band concern.
- Receipts are kept in an in-process, append-only chain. Nothing is written
  to disk or the network from inside the kernel.
- Stdlib + torch (+ numpy for the output digest) only — Kernel Hub
  universal-kernel requirement.
- The canonical szl-receipt v0.2.0 evidence binding (``emit_receipt``) is
  ADDITIVE and IMPORT-GUARDED: with szl-receipt absent it returns ``None`` and
  the kernel runs unchanged. It binds subject / input-digest / output-digest /
  policy-id / energy; energy is the literal string "UNAVAILABLE" because this
  kernel measures NO joules — a value is never fabricated. Like the SHA3-256
  chain, it is an EVIDENCE trail, NOT a proof of correctness.
"""
import hashlib
import json
import threading
import time
from typing import Any, Dict, List, Optional, Union

import torch

_GENESIS = "0" * 64

# Logical signing-authority label stamped onto signature envelopes.
_ORGAN = "szl-governed-norm"

# Governing policy id bound into every canonical szl-receipt evidence binding.
_POLICY_ID = "szl-governed-norm/provenance@v1"

# This universal kernel measures NO joules. The honesty doctrine forbids
# fabricating an energy value, so the canonical binding records the literal
# string "UNAVAILABLE" rather than a placeholder number.
_ENERGY_UNAVAILABLE = "UNAVAILABLE"


def _maybe_sign(
    body: Dict[str, Any],
    sign_key: Optional[Union[str, bytes]],
    organ: str,
) -> Optional[Dict[str, Any]]:
    """ADDITIVE szl-receipt signature layer over the receipt *body*.

    Returns a DSSE envelope (from ``szl_receipt.sign_receipt``) covering the
    exact canonical body, or ``None`` when szl-receipt is not installed (the
    kernel then behaves exactly as before). Doctrine: with no *sign_key* the
    envelope is UNSIGNED-honest (``signed=False``); a signature is NEVER
    fabricated. This is distinct from and additive to the SHA3-256 chain
    integrity hash (``digest``) — szl-receipt's envelope carries its own
    SHA-256 ``digest``/``algo`` so the two integrity hashes are explicit.
    """
    try:
        from szl_receipt import Receipt, sign_receipt
    except Exception:  # noqa: BLE001 - signing is optional; absence is honest
        return None
    env = sign_receipt(Receipt(kind="governed-norm", body=body),
                       sign_key, organ=organ)
    return env


def _tensor_digest(t: torch.Tensor, decimals: int = 6) -> str:
    """Deterministic SHA3-256 over a tensor's rounded float32 contents.

    Rounding to a fixed number of decimals makes the digest stable across
    devices/dtypes for the same logical values (tiny FP noise won't change
    it). This is an integrity fingerprint, not a signature.
    """
    flat = t.detach().to(torch.float32).reshape(-1)
    # Round to `decimals` places, integerize, hash the raw bytes. CPU move is
    # required to read bytes; kept O(n) and allocation-light.
    scaled = torch.round(flat * (10 ** decimals)).to(torch.int64).cpu().numpy().tobytes()
    h = hashlib.sha3_256()
    h.update(scaled)
    return h.hexdigest()


def _input_digest(x: torch.Tensor, eps: float) -> str:
    """SHA3-256 over the canonical JSON of a call's input spec.

    Binds {input shape, dtype, eps} — the *shape* of the call, not the input
    bytes — so the receipt is a compact fingerprint of what produced the
    output. Deterministic and stdlib-only (json + hashlib).
    """
    spec = {
        "in_shape": list(x.shape),
        "in_dtype": str(x.dtype).replace("torch.", ""),
        "eps": float(eps),
    }
    raw = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha3_256(raw).hexdigest()


def emit_receipt(
    op: str,
    x: torch.Tensor,
    out: torch.Tensor,
    eps: float,
    subject: Optional[str] = None,
    policy_id: str = _POLICY_ID,
    sign_key: Optional[Union[str, bytes]] = None,
    organ: str = _ORGAN,
) -> Optional[Dict[str, Any]]:
    """Canonical szl-receipt v0.2.0 evidence binding for a governed-norm call.

    ADDITIVE and IMPORT-GUARDED: returns ``None`` when szl-receipt is not
    installed, so this universal Kernel-Hub kernel still imports and runs on
    stdlib + torch + numpy alone. When szl-receipt is present it binds an
    EVIDENCE trail and wraps it in a DSSE envelope via ``sign_receipt``:

        subject        organ / norm-call id (who/what emitted this)
        input_digest   SHA3-256 over canonical {input shape, dtype, eps}
        output_digest  the EXISTING SHA3-256 rounded-tensor digest of ``out``
        policy_id      the governing policy id
        energy         the literal string "UNAVAILABLE"

    Doctrine (non-negotiable):
      * A receipt is an integrity/EVIDENCE trail, NOT a proof of correctness.
      * ``energy == "UNAVAILABLE"`` — this kernel measures NO joules; a joule is
        NEVER fabricated.
      * Keyless => UNSIGNED-honest (``signature["signed"] is False``); a
        signature is NEVER fabricated. A real ``sign_key`` yields a real DSSE
        signature over the exact canonical binding.

    Returns the binding dict (subject/input_digest/output_digest/policy_id/
    energy) with the DSSE envelope under ``signature``, or ``None`` when
    szl-receipt is absent.
    """
    try:
        from szl_receipt import Receipt, sign_receipt
    except Exception:  # noqa: BLE001 - canonical binding is optional; absence is honest
        return None
    body = {
        "subject": subject if subject is not None else f"{organ}/{op}",
        "input_digest": _input_digest(x, eps),
        "output_digest": _tensor_digest(out),
        "policy_id": policy_id,
        "energy": _ENERGY_UNAVAILABLE,
    }
    env = sign_receipt(Receipt(kind="governed-norm", body=body), sign_key, organ=organ)
    return dict(body, signature=env)


class ReceiptChain:
    """Append-only, SHA3-256 hash-chained log of normalization receipts.

    Each receipt: {seq, op, in_shape, in_dtype, eps, out_digest, prev, digest, ts}
    digest = SHA3-256 over the canonical JSON body (excluding digest/ts).
    verify() re-walks the chain and returns (ok, depth, first_break_seq).
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._records: List[Dict[str, Any]] = []

    @staticmethod
    def _digest_body(body: Dict[str, Any]) -> str:
        raw = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha3_256(raw).hexdigest()

    def emit(
        self,
        op: str,
        x: torch.Tensor,
        out: torch.Tensor,
        eps: float,
        sign_key: Optional[Union[str, bytes]] = None,
        organ: str = _ORGAN,
        policy_id: str = _POLICY_ID,
    ) -> Dict[str, Any]:
        with self._lock:
            prev = self._records[-1]["digest"] if self._records else _GENESIS
            seq = len(self._records)
            body = {
                "seq": seq,
                "op": op,
                "in_shape": list(x.shape),
                "in_dtype": str(x.dtype).replace("torch.", ""),
                "eps": float(eps),
                "out_digest": _tensor_digest(out),
                "prev": prev,
            }
            digest = self._digest_body(body)
            rec = dict(body, digest=digest, ts=time.time())
            sig = _maybe_sign(body, sign_key, organ)
            if sig is not None:
                rec["signature"] = sig
            # ADDITIVE canonical szl-receipt v0.2.0 evidence binding. Import-
            # guarded: None when szl-receipt is absent, so the universal kernel
            # keeps working on stdlib + torch + numpy only. Binds subject /
            # input-digest / output-digest / policy-id / energy; energy is
            # "UNAVAILABLE" (no joules measured here). It does NOT enter the
            # SHA3-256 chain body, so verify() is unaffected.
            binding = emit_receipt(
                op, x, out, eps,
                subject=f"{organ}/{op}#{seq}",
                policy_id=policy_id,
                sign_key=sign_key,
                organ=organ,
            )
            if binding is not None:
                rec["receipt"] = binding
            self._records.append(rec)
            return rec

    def head(self) -> str:
        with self._lock:
            return self._records[-1]["digest"] if self._records else _GENESIS

    def count(self) -> int:
        with self._lock:
            return len(self._records)

    def tail(self, n: int = 10) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._records[-n:])

    def verify(self):
        """Re-walk the chain. Returns (ok: bool, depth: int, first_break: int)."""
        with self._lock:
            prev = _GENESIS
            for i, rec in enumerate(self._records):
                body = {k: rec[k] for k in
                        ("seq", "op", "in_shape", "in_dtype", "eps", "out_digest", "prev")}
                if rec["prev"] != prev or rec["digest"] != self._digest_body(body):
                    return (False, len(self._records), i)
                prev = rec["digest"]
            return (True, len(self._records), -1)


# Module-level default chain (opt-in: only written when governed=True is used).
_DEFAULT_CHAIN: Optional[ReceiptChain] = None
_chain_lock = threading.Lock()


def default_chain() -> ReceiptChain:
    global _DEFAULT_CHAIN
    with _chain_lock:
        if _DEFAULT_CHAIN is None:
            _DEFAULT_CHAIN = ReceiptChain()
        return _DEFAULT_CHAIN

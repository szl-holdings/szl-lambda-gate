# SPDX-License-Identifier: Apache-2.0
# © 2026 SZL Holdings · Stephen P. Lutar · ORCID 0009-0001-0110-4173
"""szl_lambda_gate.governed_norm — governed normalization kernels (folded in).

CONSOLIDATION (Wave D): this subpackage is the ``szl-governed-norm`` universal
kernel folded into the canonical ``szl-lambda-gate`` kernels package so the two
duplicate micro-repos become ONE canonical home. The source repo
``szl-holdings/szl-governed-norm`` is DEPRECATED (see its DEPRECATED.md) and
points here; nothing was deleted — this is the additive, reversible copy.

It provides correctness-verified RMSNorm and LayerNorm that run on CPU and CUDA
and are torch.compile-friendly, plus an optional *governed* path that emits
content-addressed, SHA3-256 hash-chained receipts of each call — provenance at
the kernel layer, in the spirit of the a11oy governed-AI platform
(https://a-11-oy.com).

Usage (as a subpackage of the canonical kernel)::

    import torch
    from szl_lambda_gate import governed_norm as gn

    print(gn.selfcheck())                 # one-shot correctness + receipt check
    x = torch.randn(4, 1024, dtype=torch.float16)
    y = gn.rms_norm(x, eps=1e-6)          # plain path
    y2 = gn.rms_norm(x, eps=1e-6, governed=True)   # records to the default chain
    chain = gn.ReceiptChain()
    y3 = gn.rms_norm(x, eps=1e-6, chain=chain)     # records into YOUR chain only
    print(chain.verify())                 # (ok, depth, first_break_seq)

Honesty: this is a universal (pure-Python) kernel — a correctness reference,
not a hand-tuned CUDA speed record. No fabricated benchmarks. Its
differentiator is verifiable governance, not raw FLOPs. Λ = Conjecture 1
(advisory, uniqueness OPEN) — never described as proven trust anywhere.

Note on torch.compile: every op is torch.compile(fullgraph=True)-compatible.
Receipt emission is an eager-only side effect (it hashes materialized tensor
bytes), so when a *governed* call is captured into a compiled graph the
numerics are unchanged but NO receipt is recorded — govern at the eager audit
boundary. This is documented honestly and covered by tests.
"""
from typing import Any, Dict, List, Optional, Tuple

import torch

from . import layers  # noqa: F401  (must be importable for Hub layer mapping)
from ._norm import fused_add_rms_norm as _fused_add_rms_norm
from ._norm import layer_norm as _layer_norm
from ._norm import rms_norm as _rms_norm
from ._receipt import _GENESIS as _GENESIS_HEAD
from ._receipt import ReceiptChain, default_chain, emit_receipt

__all__ = [
    "rms_norm",
    "layer_norm",
    "fused_add_rms_norm",
    "layers",
    "ReceiptChain",
    "emit_receipt",
    "receipt_head",
    "receipt_count",
    "receipt_tail",
    "receipt_verify",
    "selfcheck",
    "DOCTRINE_FOOTER",
    "__version__",
]

__version__ = "0.2.0"
DOCTRINE_FOOTER = (
    "SZL Holdings · governed normalization · provenance at the kernel layer · "
    "Lambda = Conjecture 1 (advisory) · honesty over checklist"
)


def _is_tracing() -> bool:
    """True while torch.compile / Dynamo is tracing this code.

    Receipt emission reads materialized tensor bytes (hashing on CPU), which is
    an inherently eager, side-effecting host operation that cannot live inside
    a traced FX graph — so under torch.compile we skip the emit. This keeps
    EVERY op torch.compile(fullgraph=True)-compatible while remaining honest:
    when a governed call is captured into a compiled graph, NO receipt is
    recorded (the numerics are unchanged and identical to the eager path).
    Governance is intended for the eager audit boundary; record receipts there.
    """
    is_compiling = getattr(torch.compiler, "is_compiling", None)
    return bool(is_compiling()) if is_compiling is not None else False


def _emit(
    chain: Optional[ReceiptChain],
    op: str,
    x: torch.Tensor,
    out: torch.Tensor,
    eps: float,
    sign_key: Optional[Any] = None,
    organ: str = "szl-governed-norm",
) -> None:
    """Append a receipt to ``chain`` (or the process default chain if None).

    No-op while torch.compile is tracing (see ``_is_tracing``). When
    ``sign_key`` (a PEM ECDSA-P256 private key) is supplied and szl-receipt is
    installed, the receipt carries an additive DSSE ``signature`` envelope;
    keyless is UNSIGNED-honest.
    """
    if _is_tracing():
        return
    target = chain if chain is not None else default_chain()
    target.emit(op, x, out, eps, sign_key=sign_key, organ=organ)


def rms_norm(
    x: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    governed: bool = False,
    chain: Optional[ReceiptChain] = None,
    sign_key: Optional[Any] = None,
    organ: str = "szl-governed-norm",
) -> torch.Tensor:
    """RMSNorm over the last dim.

    If ``governed=True``, append an audit receipt. By default the receipt goes
    to the process-wide default chain (convenient). Pass your own ``chain`` (a
    ``ReceiptChain`` instance) to record into a caller-owned chain instead —
    this avoids global-state contention when many threads/requests govern
    independently. Passing ``chain`` implies governance even if
    ``governed=False`` is left at its default. Pass ``sign_key`` (PEM
    ECDSA-P256) to additively sign the receipt via szl-receipt.
    """
    out = _rms_norm(x, weight=weight, eps=eps)
    if governed or chain is not None:
        _emit(chain, "rms_norm", x, out, eps, sign_key=sign_key, organ=organ)
    return out


def layer_norm(
    x: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    governed: bool = False,
    chain: Optional[ReceiptChain] = None,
    sign_key: Optional[Any] = None,
    organ: str = "szl-governed-norm",
) -> torch.Tensor:
    """LayerNorm over the last dim.

    If ``governed=True`` (or a ``chain`` is supplied), append an audit receipt
    to ``chain`` when given, otherwise to the process default chain. See
    ``rms_norm`` for the per-call ``chain`` rationale and ``sign_key``.
    """
    out = _layer_norm(x, weight=weight, bias=bias, eps=eps)
    if governed or chain is not None:
        _emit(chain, "layer_norm", x, out, eps, sign_key=sign_key, organ=organ)
    return out


def fused_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    governed: bool = False,
    chain: Optional[ReceiptChain] = None,
    sign_key: Optional[Any] = None,
    organ: str = "szl-governed-norm",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Residual-add + RMSNorm (transformer block pattern).

    Returns ``(y, new_residual)`` where ``new_residual = x + residual`` and
    ``y = rms_norm(new_residual, weight, eps)``. If ``governed=True`` (or a
    ``chain`` is supplied), append an audit receipt over the normalized output
    to ``chain`` when given, otherwise to the process default chain. Pass
    ``sign_key`` to additively sign the receipt via szl-receipt.
    """
    out, new_residual = _fused_add_rms_norm(x, residual, weight=weight, eps=eps)
    if governed or chain is not None:
        _emit(chain, "fused_add_rms_norm", x, out, eps, sign_key=sign_key, organ=organ)
    return out, new_residual


# ---- governance receipt surface (operates on the default in-process chain) --
def receipt_head() -> str:
    """SHA3-256 head of the governed-call receipt chain ('0'*64 if empty)."""
    return default_chain().head()


def receipt_count() -> int:
    """Number of governed calls recorded."""
    return default_chain().count()


def receipt_tail(n: int = 10) -> List[Dict[str, Any]]:
    """Last n receipts."""
    return default_chain().tail(n)


def receipt_verify() -> Dict[str, Any]:
    """Re-walk the receipt chain. Returns {ok, depth, first_break_seq}."""
    ok, depth, brk = default_chain().verify()
    return {"ok": ok, "depth": depth, "first_break_seq": brk, "head": default_chain().head()}


# ---- one-shot self-verification --------------------------------------------
def selfcheck() -> Dict[str, Any]:
    """Verify correctness + governance in a single call; never raises.

    Runs a tiny, self-contained, CPU-only smoke test against PyTorch references
    so downstream code (and SZL's own a11oy / hatun-mcp) can confirm the loaded
    kernel is the real, working article before trusting it.

    Checks (all on a *private, throwaway* ReceiptChain so the process default
    chain is never touched):
      * ``rms_norm`` matches a Llama-style float32 reference,
      * ``layer_norm`` matches ``torch.nn.functional.layer_norm``,
      * ``fused_add_rms_norm`` matches the unfused add-then-norm path,
      * a governed call emits exactly one receipt and the chain verifies.

    Returns a JSON-able dict:
      ``{ok, version, checks: {name: bool}, receipt_ok, receipt_head, error}``
    ``ok`` is True iff every check passed. On unexpected failure ``ok`` is
    False and ``error`` carries the message — this function is designed to be
    safe to call in a health probe and will not raise.
    """
    checks: Dict[str, bool] = {}
    receipt_ok = False
    receipt_head = _GENESIS_HEAD
    error = None
    try:
        torch.manual_seed(0)
        x = torch.randn(4, 64, dtype=torch.float32)
        w = torch.randn(64, dtype=torch.float32)
        b = torch.randn(64, dtype=torch.float32)
        res = torch.randn(4, 64, dtype=torch.float32)
        eps_r, eps_l = 1e-6, 1e-5

        # rms_norm vs Llama-style fp32 reference
        xf = x.to(torch.float32)
        ref_rms = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps_r)) * w
        checks["rms_norm"] = bool(
            torch.allclose(rms_norm(x, weight=w, eps=eps_r), ref_rms, rtol=1e-5, atol=1e-5)
        )

        # layer_norm vs torch reference
        ref_ln = torch.nn.functional.layer_norm(x, (64,), weight=w, bias=b, eps=eps_l)
        checks["layer_norm"] = bool(
            torch.allclose(layer_norm(x, weight=w, bias=b, eps=eps_l), ref_ln,
                           rtol=1e-5, atol=1e-5)
        )

        # fused_add_rms_norm vs unfused path
        y_f, new_res = fused_add_rms_norm(x, res, weight=w, eps=eps_r)
        h = x.to(torch.float32) + res.to(torch.float32)
        ref_y = rms_norm(h.to(x.dtype), weight=w, eps=eps_r)
        checks["fused_add_rms_norm"] = bool(
            torch.allclose(y_f, ref_y, rtol=1e-5, atol=1e-5)
            and torch.allclose(new_res, h.to(x.dtype), rtol=1e-6, atol=1e-6)
        )

        # governance on a private chain: one emit, chain verifies
        probe_chain = ReceiptChain()
        rms_norm(x, weight=w, eps=eps_r, chain=probe_chain)
        ok, depth, brk = probe_chain.verify()
        receipt_ok = bool(ok and depth == 1 and brk == -1)
        receipt_head = probe_chain.head()
        checks["governance"] = receipt_ok
    except Exception as exc:  # never raise from a health probe
        error = f"{type(exc).__name__}: {exc}"

    ok = bool(checks) and all(checks.values()) and error is None
    return {
        "ok": ok,
        "version": __version__,
        "checks": checks,
        "receipt_ok": receipt_ok,
        "receipt_head": receipt_head,
        "error": error,
    }

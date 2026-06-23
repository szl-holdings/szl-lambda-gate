---
tags:
- kernel
license: apache-2.0
---

# szl-lambda-gate

> **GitHub mirror** of the Kernel Hub kernel published at **[huggingface.co/SZLHOLDINGS/szl-lambda-gate](https://huggingface.co/SZLHOLDINGS/szl-lambda-gate)**. The Hugging Face repo is the canonical `get_kernel` source; this repository mirrors the same source of truth.

**The Λ (Lambda-Spine) aggregator as a universal kernel — a non-compensatory, *advisory* governance roll-up for the Hugging Face Kernel Hub.**

> Λ is the weighted geometric mean of axis scores in `[0,1]`. Any single zeroed axis drives the whole aggregate to `0`: a conservative, non-compensatory signal. It is **advisory** — *not* proven trust, *not* a closed theorem.

A universal (pure-PyTorch) kernel from [SZL Holdings](https://huggingface.co/SZLHOLDINGS). It ports the canonical Λ aggregator into a differentiable, `torch.compile`-friendly torch op, plus an advisory governance gate (Λ vs a threshold), the four carried axioms as real runtime self-checks, and pure `nn.Module` layers.

---

## What it is

`szl-lambda-gate` is a [Kernel Hub](https://huggingface.co/docs/kernels) kernel that computes:

\[
\Lambda(x) = \prod_i x_i^{\,w_i}, \quad \sum_i w_i = 1, \quad w_i > 0, \quad x_i \in [0,1]
\]

the **weighted geometric mean** over the last dim of an axis-score tensor. It gives you:

1. **A correctness reference you can trust.** Λ is implemented in pure PyTorch, computed via logs in float32 (float64 for float64 inputs) for numerical stability, differentiable (autograd works), and verified against a pure-Python reference in the test suite.
2. **An advisory governance gate.** `lambda_gate(axes, threshold=...)` returns a score plus a pass/fail mask (`Λ ≥ threshold`). `lambda_gate_batch` scores many candidate action-vectors in one call — the realistic per-inference-step agent usage.
3. **The four carried axioms as runtime self-checks.** A1–A4 (monotone, homogeneous, Egyptian-exact, bounded-by-max) are exposed as honest empirical checks plus a `selfcheck()` verdict and an adversarial falsification search.

This is a **universal kernel**: it ships no hand-tuned CUDA/Triton binary. Its differentiator is a verifiable, honestly-labeled governance aggregator — not raw FLOPs.

---

## Quickstart

```python
import torch
from kernels import get_kernel

lg = get_kernel("SZLHOLDINGS/szl-lambda-gate")

axes = torch.tensor([0.9, 0.8, 0.95])         # axis scores in [0,1]
score = lg.lambda_aggregate(axes)             # Λ(x) ∈ [0,1]

res = lg.lambda_gate(axes, threshold=0.5)     # ADVISORY pass/fail
print(res.score, res.passed, res.advisory)    # advisory is always True
```

### Batch gate (one call per inference step)

```python
# candidates: (..., N, k) — N candidate action-vectors, each with k axis scores
candidates = torch.rand(8, 13)
res = lg.lambda_gate_batch(candidates, threshold=0.5)
print(res.score.shape, res.passed.shape)      # (8,) (8,)
```

### Axiom self-checks

```python
verdict = lg.selfcheck()
print(verdict["all_axioms_hold"], verdict["lambda_status"])
# empirical checks on sampled inputs — NOT a proof of Λ-uniqueness
```

---

## API reference

### Functional API

| Function | Signature | Notes |
|---|---|---|
| `lambda_aggregate` | `lambda_aggregate(axes, weights=None)` | Λ over the last dim. Uniform weights when `None`. Differentiable, batched. |
| `lambda_gate` | `lambda_gate(axes, weights=None, threshold=0.5)` | ADVISORY gate → `LambdaGateResult(score, passed, threshold, advisory)`. |
| `lambda_gate_batch` | `lambda_gate_batch(candidates, weights=None, threshold=0.5)` | Score `(..., N, k)` candidates at once; advisory pass mask. |
| `yuyay_weights` | `yuyay_weights(dtype=..., device=None)` | Canonical 13-axis uniform Λ weight vector (advisory). |

### Axiom runtime self-checks (real, verifiable — NOT a uniqueness proof)

| Function | Axiom | Description |
|---|---|---|
| `is_monotone` | A1 | Λ is non-decreasing in each axis (on the given data). |
| `is_homogeneous` | A2 | Λ(t·x) = t·Λ(x) (degree 1). |
| `is_egyptian_exact` | A3 | Λ(c,…,c) = c. |
| `is_bounded_by_max` | A4 | Λ(x) ≤ maxᵢ xᵢ. |
| `find_axiom_violation` | — | Random-search falsification attempt; returns a violating triple or `None`. |
| `selfcheck` | — | Runs A1–A4 + adversarial search, returns a verdict dict. |

### `nn.Module` layers

Pure `torch.nn.Module` subclasses (only `forward`, no custom `__init__`, no class variables) for the `kernels` layer-mapping mechanism:

| Layer | Reads from host module | Description |
|---|---|---|
| `LambdaGate` | `self.weights` (optional), `self.threshold` (default 0.5) | forward(axes) → `LambdaGateResult`. |
| `LambdaAggregate` | `self.weights` (optional) | forward(axes) → Λ(axes) tensor. |

---

## Non-compensatory zero-routing (the conservative choice)

Any axis that is zero, **or** non-finite (NaN / ±Inf), is treated as a **failing** axis and drives the whole aggregate to exactly `0`. A garbage or invalid axis must never silently pass as a "perfect" (clamped-to-1) axis, and both the output and its gradient stay finite and in `[0,1]` for every input. This is the conservative governance semantics, A4-consistent (`Λ(x) ≤ maxᵢ xᵢ`).

---

## What Λ IS / IS NOT — honesty (SZL Holdings doctrine)

We hold this kernel to a plain-spoken standard:

- **Λ is the weighted-geometric-mean aggregator** — a non-compensatory, **advisory** way to roll axis scores in `[0,1]` into one number.
- **Λ is NOT "proven trust" and NOT a closed theorem.** Its *uniqueness* (that the weighted geometric mean is the only aggregator satisfying the carried axioms) remains **Conjecture 1 — OPEN** (an unresolved `CAUCHY_ND` step plus a missing symmetry axiom in the Lean development). A gate "pass" is an advisory signal, never proven trust.
- **The axiom self-checks are empirical, not a proof.** They verify the carried axioms hold for *this implementation on the given data*; finding no violation is evidence, not a uniqueness proof.
- **No fabricated benchmarks.** This is a universal kernel, not a hand-tuned binary — there are **no speedup claims** here.

### Prior art (honest attribution)

The weighted geometric mean as a *less-compensatory* composite-indicator aggregator is established practice: the UN HDI (arithmetic→geometric switch, 2010), the OECD *Handbook on Constructing Composite Indicators* (2008), and the UNECE well-being guidelines all use it "to limit the compensation effect". The veto / cut-off idea (a single failing criterion blocks a pass regardless of the others) is the ELECTRE veto threshold / satisficing minimum-threshold screen. The 13-axis conjunctive form exposed by `yuyay_weights` is SZL's own yuyay_v3 gate. None of this makes Λ proven trust; the gate is advisory.

---

## Provenance

Backed by the Lean 4 formalization [`szl-holdings/lutar-lean`](https://github.com/szl-holdings/lutar-lean) (749 declarations / 14 axioms / 163 tracked sorries), DOI [10.5281/zenodo.20434308](https://doi.org/10.5281/zenodo.20434308). **Λ uniqueness = Conjecture 1 (open).**

---

## Compatibility

| Requirement | Version |
|---|---|
| Python | 3.9+ |
| PyTorch | `torch>=2.5` |
| Dependencies | Python standard library + `torch` only |

The universal-kernel constraint (stdlib + torch only) is intentional: it keeps the kernel portable, easy to audit, and free of supply-chain surprises.

---

## About SZL Holdings

SZL Holdings, founded by **Stephen Lutar**, builds governed-AI infrastructure — provenance, observability, and security tooling for AI systems. This kernel applies that governance doctrine at the level of a single PyTorch operation: a conservative, honestly-labeled, advisory aggregator. See the [SZL Holdings Hugging Face org](https://huggingface.co/SZLHOLDINGS) and the [a11oy governed-AI platform](https://a11oy.net).

## Citation

If you use this kernel, please cite it via the included [`CITATION.cff`](./CITATION.cff). Authored by Stephen Lutar (ORCID [0009-0001-0110-4173](https://orcid.org/0009-0001-0110-4173)).

## License

Apache-2.0 — see [`LICENSE`](./LICENSE). Copyright 2026 SZL Holdings.

---

<sub>
<b>SZL Holdings</b> · Λ = Conjecture 1 (advisory, weighted geometric mean) · uniqueness OPEN · NOT proven trust ·
<a href="https://a11oy.net">a11oy.net</a> ·
<a href="https://github.com/szl-holdings/szl-lambda-gate">github.com/szl-holdings/szl-lambda-gate</a> ·
<a href="https://huggingface.co/SZLHOLDINGS/szl-lambda-gate">huggingface.co/SZLHOLDINGS/szl-lambda-gate</a>
</sub>

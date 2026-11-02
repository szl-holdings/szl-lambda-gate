---
license: apache-2.0
library_name: kernels
tags:
  - kernel
  - governance
  - lambda
  - gate
  - provenance
  - szl-holdings
  - doi:10.5281/zenodo.19944926
---

<p><img src="https://raw.githubusercontent.com/szl-holdings/szl-brand/27d12e77fc0f9f1a0e5fae8b34adf2380d51d070/kit/logos/orbit-v2/suite/szl-icon-32.svg" alt="SZL Holdings" width="40" height="40" /></p>


# {{KERNEL_NAME}}

**Λ, a governance aggregator, as a pure-PyTorch (noarch) kernel.** A differentiable,
`torch.compile`-friendly weighted-geometric-mean aggregator with an ADVISORY
non-compensatory gate, the strict `szl.lambda/v1` entry point and runtime axiom
self-checks, from [SZL Holdings](https://huggingface.co/SZLHOLDINGS).

This repository is published from GitHub
[{{GITHUB_REPOSITORY}}](https://github.com/{{GITHUB_REPOSITORY}}) by its
`hf-kernel-publish` workflow, which is its only writer. GitHub is the source of truth:
every file under `build/` is a byte copy of `torch-ext/szl_lambda_gate` at the commit
named in `SZL_KERNEL_SOURCE.json`.

## Load it

```python
# pip install "kernels>=0.17.2" torch
import torch
from kernels import get_kernel

lg = get_kernel("{{REPO_ID}}", version={{KERNEL_VERSION}}, trust_remote_code=True)

axes = torch.tensor([0.9, 0.8, 0.95], dtype=torch.float64)   # axis scores in [0,1]
weights = torch.tensor([0.5, 0.25, 0.25], dtype=torch.float64)
gate = lg.lambda_v1_gate(axes, weights, tau=0.8)              # GO / NO_GO / ABSTAIN / BLOCK
print(gate.verdict, gate.code)

print(lg.lambda_aggregate(axes))   # legacy Λ(x) in [0,1]
print(lg.selfcheck())              # empirical A1-A4 checks, not a proof
```

- `version={{KERNEL_VERSION}}` resolves the `v{{KERNEL_VERSION}}` branch, the stable API channel. It
  is republished on every release. The `kernels` library refuses a call that names neither
  a version nor a revision.
- `trust_remote_code` is required: SZLHOLDINGS is not a trusted kernel publisher on the Hub.
  The narrowest form is `trust_remote_code=["{{REPO_ID}}"]`.
- CPU hosts load `build/torch-cpu`; CUDA and other hosts load `build/torch-universal`.
  Both variants carry identical code, built from the same GitHub commit.

### Pin an exact build (production)

```python
from huggingface_hub import HfApi
from kernels import get_kernel

refs = HfApi().list_repo_refs("{{REPO_ID}}", repo_type="kernel")
pin = next(b.target_commit for b in refs.branches if b.name == "v{{KERNEL_VERSION}}")
lg = get_kernel("{{REPO_ID}}", revision=pin, trust_remote_code=["{{REPO_ID}}"])
```

Record `pin` (a commit of this kernel repository) in your lockfile. Commits of the
model repository [{{MODEL_TWIN_URL}}]({{MODEL_TWIN_URL}}) are not kernel revisions and
do not resolve here.

## What Λ is, and is not

Λ is the weighted geometric mean over axis scores in [0,1]:
Λ(x) = ∏ xᵢ^wᵢ with Σ wᵢ = 1, wᵢ > 0. It is a non-compensatory, ADVISORY roll-up: any
zero (or, in the legacy entry, non-finite) axis drives the aggregate to 0. A gate pass is
an advisory signal that the host must enforce; it is not proven trust. Λ uniqueness
remains **Conjecture 1, machine-checked false as stated**. The A1-A4 self-checks are empirical evidence on sampled
inputs, not a proof.

## API

| Entry | Notes |
|---|---|
| `lambda_v1(axes, weights)` / `lambda_v1_gate(axes, weights, tau)` | Strict `szl.lambda/v1` contract (`spec/szl.lambda.v1.json`): validated, never repaired; bad input returns `BLOCK` with a `LAMBDA_*` code; τ is required. |
| `lambda_aggregate(axes, weights=None)` | Legacy Λ over the last dim. Differentiable, batched. Clamps and renormalises (see the spec for the differences). |
| `lambda_gate(axes, weights=None, threshold=...)` / `lambda_gate_batch(...)` | Legacy advisory gate → `LambdaGateResult(score, passed, threshold, advisory)`. |
| `selfcheck()`, `is_monotone`, `is_homogeneous`, `is_egyptian_exact`, `is_bounded_by_max`, `find_axiom_violation` | Empirical A1-A4 checks and a falsification search. |
| `governed_norm` (`rms_norm`, `layer_norm`, `fused_add_rms_norm`) | Governed normalization with SHA3-256 hash-chained receipts. Also published standalone as [SZLHOLDINGS/szl-governed-norm](https://huggingface.co/kernels/SZLHOLDINGS/szl-governed-norm). |
| `layers.LambdaGate`, `layers.LambdaAggregate` | Pure `nn.Module` layers for kernel layer mapping. |

## Files

| Path | What it is |
|---|---|
| `build/torch-cpu/`, `build/torch-universal/` | The kernel package and its generated `metadata.json` (with a sha256 digest of every file). |
| `spec/szl.lambda.v1.json`, `spec/lambda_v1_vectors.json` | The `szl.lambda/v1` contract and its golden vectors. |
| `reference/szl_lambda_v1.py` | Stdlib-only reference implementation of the contract. |
| `SZL_KERNEL_SOURCE.json` | The GitHub commit and per-file sha256 this tree was built from. |
| `build.toml`, `LICENSE` | Kernel build manifest and Apache-2.0 license, copied from GitHub. |

No speedup, throughput or energy figure is claimed for this kernel. Test results live
in GitHub Actions for the source commit.

## License and citation

Apache-2.0. Copyright 2026 SZL Holdings. Concept DOI
[10.5281/zenodo.19944926](https://doi.org/10.5281/zenodo.19944926); see `CITATION.cff` in
the GitHub repository.

Artwork: SZL Orbit · © 2026 SZL Holdings / Stephen Lutar · [CC BY 4.0](https://github.com/szl-holdings/szl-brand/blob/27d12e77fc0f9f1a0e5fae8b34adf2380d51d070/kit/LICENSE) · [Trademark notice](https://github.com/szl-holdings/szl-brand/blob/27d12e77fc0f9f1a0e5fae8b34adf2380d51d070/NOTICE).

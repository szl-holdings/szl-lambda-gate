---
license: apache-2.0
library_name: kernels
tags:
- kernel
- szl-holdings
szl:
  source_repo: szl-holdings/szl-lambda-gate
  proof_url: https://github.com/szl-holdings/szl-lambda-gate
---

<p><a href="https://huggingface.co/spaces/SZLHOLDINGS/szl-command-lab"><img src="https://raw.githubusercontent.com/szl-holdings/szl-brand/27d12e77fc0f9f1a0e5fae8b34adf2380d51d070/kit/logos/orbit-v2/suite/szl-icon-32.svg" alt="SZL Holdings" width="40" height="40" /></a></p>

# SZL Governed Norm

Normalize tensors with governed RMSNorm and LayerNorm kernels that can record SHA3-256 hash-chained receipts of each call.

**Artifact:** Tensor-normalization software kernel · **Stage:** Compatibility kernel, published from its canonical source

[Explore in Command Lab](https://huggingface.co/spaces/SZLHOLDINGS/szl-command-lab) · [Source](https://github.com/szl-holdings/szl-lambda-gate/tree/main/torch-ext/szl_lambda_gate/governed_norm) · [Contract](https://github.com/szl-holdings/szl-lambda-gate/blob/main/hf-kernels/szl-governed-norm/contract.json)

## Load it

```python
# pip install "kernels>=0.17.2" torch
import torch
from kernels import get_kernel

gn = get_kernel("SZLHOLDINGS/szl-governed-norm", version=1, trust_remote_code=True)

x = torch.randn(4, 1024)
print(gn.rms_norm(x))
print(gn.selfcheck())   # one-shot correctness and receipt check
```

- `version=1` resolves the `v1` branch, the stable API channel. It is republished on every
  release of the source repository. The `kernels` library refuses a call that names neither
  a version nor a revision.
- Loading the kernel executes repository code, so `trust_remote_code` is required
  (SZLHOLDINGS is not a trusted kernel publisher). The narrowest form is
  `trust_remote_code=["SZLHOLDINGS/szl-governed-norm"]`.
- CPU hosts load `build/torch-cpu`; CUDA and other hosts load `build/torch-universal`.
  Both variants carry identical code, built from the same GitHub commit.

## Source and publication

The package is `szl_lambda_gate.governed_norm` from
[szl-holdings/szl-lambda-gate](https://github.com/szl-holdings/szl-lambda-gate), published
here as the standalone package `szl_governed_norm` by that repository's
`hf-kernel-publish` workflow. `SZL_KERNEL_SOURCE.json` in this repository names the GitHub
commit and the sha256 of every file. The former `szl-holdings/szl-governed-norm` GitHub
repository is archived.

To pin an exact build, take the current commit of the `v1` branch
(`HfApi().list_repo_refs("SZLHOLDINGS/szl-governed-norm", repo_type="kernel")`) and pass it
as `revision=`. Commits of the separate model repository are not kernel revisions.

Earlier cards pinned
`get_kernel("SZLHOLDINGS/szl-governed-norm", revision="fe16433d44be03177167e8355c43a4bfdc63e03e", trust_remote_code=True)`.
That July 2026 build ships `build/torch-cpu` only, so it does not load on CUDA hosts, and it
predates the receipt digest fixes. It still resolves for reproducibility; new code should
use `version=1`.

Artwork: SZL Orbit · © 2026 SZL Holdings / Stephen Lutar · [CC BY 4.0](https://github.com/szl-holdings/szl-brand/blob/27d12e77fc0f9f1a0e5fae8b34adf2380d51d070/kit/LICENSE).

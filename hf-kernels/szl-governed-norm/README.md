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

<p><a href="https://huggingface.co/spaces/SZLHOLDINGS/szl-command-lab"><img src="https://raw.githubusercontent.com/szl-holdings/.github/main/profile/assets/szl/logos/szl_mark_holographic.svg" alt="SZL Holdings" width="112" /></a></p>

# SZL Governed Norm

Normalize tensors with the retained RMSNorm kernel and follow ongoing development in the canonical Lambda Gate package.

**Artifact:** Tensor-normalization software kernel · **Stage:** Retained compatibility contract

[Explore in Command Lab](https://huggingface.co/spaces/SZLHOLDINGS/szl-command-lab) · [Build](https://github.com/szl-holdings/szl-lambda-gate) · [Evidence](https://github.com/szl-holdings/szl-lambda-gate/blob/ee213a148dede345bfd05396178dfba04f59bd82/hf-kernels/szl-governed-norm/contract.json)

## Before you use it

- Use the exact first-class Kernel Hub pin below. The native kernel and model repositories have separate revision histories.
- Loading the kernel executes repository code and requires explicit trust. Inspect the immutable revision before enabling `trust_remote_code=True`.
- The ABI, existing builds and qualification evidence stay unchanged; new source development lives in `szl_lambda_gate.governed_norm`.

<details>
<summary>Technical details and original loading contract</summary>

The complete original source card follows. Its immutable loading pin and explicit trust requirement remain in force.

<!-- SZL-PRESERVED-TECHNICAL-BODY:START -->
# SZL governed norm — immutable compatibility contract

The legacy Kernel Hub artifact remains available for compatibility. New source
development lives in `szl_lambda_gate.governed_norm`.

## Quickstart

```python
import torch
from kernels import get_kernel

gn = get_kernel(
    "SZLHOLDINGS/szl-governed-norm",
    revision="fe16433d44be03177167e8355c43a4bfdc63e03e",
    trust_remote_code=True,
)

x = torch.randn(4, 1024)
print(gn.rms_norm(x))
```

This pin names the current first-class kernel commit, not a commit from the
separate legacy model repository. Advance it only after publishing and verifying
a new Kernel Hub revision.

<!-- SZL-PRESERVED-TECHNICAL-BODY:END -->

</details>

---
license: apache-2.0
tags:
  - kernel
  - szl-holdings
---

<p><img src="https://raw.githubusercontent.com/szl-holdings/szl-brand/27d12e77fc0f9f1a0e5fae8b34adf2380d51d070/kit/logos/orbit-v2/suite/szl-icon-32.svg" alt="SZL Holdings" width="40" height="40" /></p>


# {{KERNEL_NAME}} has moved to the kernel repository

This model repository holds no loadable build. The `kernels` library (0.16 and later) loads
kernels only from the kernel repository:

**[{{KERNEL_URL}}]({{KERNEL_URL}})**

```python
# pip install "kernels>=0.17.2" torch
from kernels import get_kernel

kernel = get_kernel("{{REPO_ID}}", version={{KERNEL_VERSION}}, trust_remote_code=True)
```

The source of truth is [{{GITHUB_REPOSITORY}}](https://github.com/{{GITHUB_REPOSITORY}}).
Its `hf-kernel-publish` workflow builds the kernel, verifies it through `get_kernel`, and
then updates this pointer. Commits of this model repository are not kernel revisions.

Artwork: SZL Orbit · © 2026 SZL Holdings / Stephen Lutar · [CC BY 4.0](https://github.com/szl-holdings/szl-brand/blob/27d12e77fc0f9f1a0e5fae8b34adf2380d51d070/kit/LICENSE) · [Trademark notice](https://github.com/szl-holdings/szl-brand/blob/27d12e77fc0f9f1a0e5fae8b34adf2380d51d070/NOTICE).

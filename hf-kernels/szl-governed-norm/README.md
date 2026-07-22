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

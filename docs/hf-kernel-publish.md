# Hugging Face Kernel Hub publication

GitHub is the source of truth. `.github/workflows/hf-kernel-publish.yml` is the only writer
of the Hub repositories listed in `.github/hf-kernels.json`:

| Hub kernel repo | Built from | Hub package |
|---|---|---|
| `kernels/SZLHOLDINGS/szl-lambda-gate` | `torch-ext/szl_lambda_gate` | `szl_lambda_gate` |
| `kernels/SZLHOLDINGS/szl-governed-norm` | `torch-ext/szl_lambda_gate/governed_norm` | `szl_governed_norm` |

## Triggers

| Event | What happens |
|---|---|
| Pull request touching the kernel, cards or publisher | Offline publisher tests. Then both noarch variants (`build/torch-cpu`, `build/torch-universal`) are built from Git objects and loaded with `kernels==0.17.2` on CPU torch. No Hub access. |
| Release published (not a prerelease) | Build, publish fast-forward to Hub `main` and `v1`, verify anonymously through `get_kernel(version=1)` and `get_kernel(revision="main")`, then replace each model twin with a pointer card. |
| `workflow_dispatch` on `main` | The same for `source_ref` (default: latest release). `dry_run=true` builds the candidate commits and runs an authenticated `push --dry-run` only. |
| Weekly schedule | Drift guard. It re-verifies what `get_kernel` serves, and fails with `SOURCE_AHEAD` if the latest release changed a kernel that was never published. |

## Rules

- The publisher owns `build/` on each kernel repo. Stale variants and modules are deleted. Other Hub paths are kept unless `prune_paths` lists them.
- Pushes are fast-forward only. There is no force push and no repository creation. A moved branch fails the push; re-run to converge.
- `SZL_KERNEL_SOURCE.json` on the Hub names the GitHub commit and the sha256 of every published file.
- The credential is the org secret `HF_TOKEN`. It is exposed only to the two writing steps; verification runs anonymously.
- Bumping `version` in `.github/hf-kernels.json` starts `v<N+1>` and leaves `v<N>` unchanged.
- Model twins (`SZLHOLDINGS/szl-lambda-gate`, `SZLHOLDINGS/szl-governed-norm` model repos) carry a README pointer only. `kernels` 0.16 and later never load them.

## Loading

```python
from kernels import get_kernel

lg = get_kernel("SZLHOLDINGS/szl-lambda-gate", version=1, trust_remote_code=True)
gn = get_kernel("SZLHOLDINGS/szl-governed-norm", version=1, trust_remote_code=True)
```

To pin an exact build, pass a commit of the kernel repository's `v1` branch as `revision=`.

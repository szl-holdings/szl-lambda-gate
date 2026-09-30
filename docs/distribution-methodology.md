# Source-bound Lambda distribution

GitHub `szl-holdings/szl-lambda-gate` owns the Python implementation and the
`szl.lambda/v1` contract. Hugging Face model and kernel repositories distribute
those inputs; websites describe measured publication state. Stephen merges.

## What we adopt from upstream

Hugging Face's [kernel requirements](https://huggingface.co/docs/kernels/kernel-requirements)
and [kernel tests](https://huggingface.co/docs/kernels/builder/writing-kernels#kernel-tests)
require testing through the loader, versioned metadata, and portable packages.
We check both the older loader used in this estate and the current loader. A
variant root entry and the compatibility package export the same public API.
Both `torch-cpu` and `torch-universal` are rebuilt from the same commit and tested
individually, so a stale higher-priority CPU variant cannot hide an updated
universal variant.

PyTorch's [comparison contract](https://docs.pytorch.org/docs/stable/testing.html)
separates numerical tolerance from dtype, device, and shape. Our v1 vectors set
an absolute `value_tol`; error codes and gate verdicts must match exactly. A
scalar float64 result is required. NaN is an input error and cannot pass through
an approximate-value comparison.

[PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/) uses
short-lived publisher identity, while [SLSA provenance](https://slsa.dev/spec/v1.1/levels)
describes source and build evidence. We keep those claims separate: a file-hash
receipt proves neither publisher identity nor an attestation. This packager
creates no signature and claims no SLSA level. The check report explicitly says
`signature_verified: false`.

## Acceptance method

1. Package immutable Git objects from a reviewed commit. Dirty or untracked
   working-tree files cannot enter the output. Rebuilding the same commit yields
   identical bytes; a pre-existing output directory is refused.
   Source license bytes live at `source/LICENSE`, preserving the curated Hub
   root license during an additive upload.
2. Verify the complete output file set, its hashes, the canonical vector digest
   and count, and kernel metadata. Reconstruct all expected bytes from the
   trusted Git revision before importing any package code. Rehashing altered
   code into a new receipt must still fail.
3. Load the package through `kernels.get_local_kernel`, offline, and run every
   vector. Keep separate receipts for each loader version and tested device.
   A CPU result does not establish CUDA performance or GPU compatibility.
4. Publish only reviewed releases. Verify the remote package at its exact Hub
   commit through `get_kernel`, with the repository explicitly allowlisted for
   remote code. A successful upload alone does not establish conformance.
   Check each consumer reference separately: updating `main` does not advance
   the `v1` branch used by `get_kernel(..., version=1)`. Both channels must resolve
   to packages from the intended source commit. Verify each channel's resolved
   commit and all vectors after publication; a passing review commit does not
   prove the published branch was updated.
   The legacy `kernels==0.12.3` consumer loads the model repository, while the
   current loader uses the kernel repository. Mirroring Python source alone
   does not rebuild Hub-only `build/` files. Check that legacy distribution
   separately with `--remote-repo-type model`; the verifier rejects a repository
   type unsupported by the selected loader before importing remote code.
5. Record source commit, Hub commit, vector digest, loader, device, and test
   outcome in publication evidence. Refresh website inventory from that measured
   evidence. Models, Spaces, and datasets unrelated to Lambda need no new copy.

The application-specific improvement is checking zero-plus-invalid inputs in
both orders on the distributed package. The v1 spec requires validation phases
before any zero veto; these adversarial rows exercise that requirement. This is
a measurable acceptance criterion, not a claim of general superiority over
upstream projects.

## Local commands

```sh
python scripts/build_lambda_distribution.py --revision HEAD --output build/distribution
python scripts/check_lambda_distribution.py --local build/distribution \
  --expected-commit "$(git rev-parse HEAD)" --receipt build/conformance.json
# After publishing, verify the exact remote commit (or a review ref):
python scripts/check_lambda_distribution.py --remote-revision HUB_COMMIT \
  --expected-commit SOURCE_COMMIT --receipt build/remote-conformance.json
```

The new CI workflow runs this path with `kernels==0.12.3` and `kernels==0.17.1`.
It emits reviewable artifacts and has no upload credential or publication step.
The existing release mirror and another session's kernel-target work remain
responsible for publication. Review their target policy before replacing a
Hub-only build: a more specific stale variant can outrank `torch-universal`.

Lambda remains advisory; uniqueness is Conjecture 1 (open).

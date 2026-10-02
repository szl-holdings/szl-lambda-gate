# Protected Hugging Face release proposals

The canonical source for `SZLHOLDINGS/szl-lambda-gate` is the protected
`szl-holdings/szl-lambda-gate` GitHub repository. The Hub repository has a
protected `main` branch. On 2026-10-02, workflow run
[`37066200083`](https://github.com/szl-holdings/szl-lambda-gate/actions/runs/37066200083)
resolved OIDC, verified the release assets and Hub baseline, then received
HTTP 403 from the Hub when it tried to commit to `main`. The Hub response
required `create_pr=1`. That attempt did not publish `v0.2.1` to the Hub.

For a new release, `hf-mirror` now submits an additive Hub pull request with
`create_pr=True` and `parent_commit` bound to the captured Hub `main`. The
preflight compares every staged overlap with the Hub baseline. Only exact paths
in `replace_hub_paths` may change; the curated card is rendered from its Hub
copy. Hub-only files are retained in the proposed tree. The publisher verifies
the complete PR file set, SHA-256 of every file, required card metadata, and
stable PR head before recording `PENDING_HUB_PR_REVIEW` in
`hf-mirror-proposal.json`.

A proposal exits with code 3, so the workflow cannot be mistaken for a green
release. The proposal artifact records the source commit, workflow commit,
Hub baseline, PR URL and revision, proposed commit, expected file hashes,
reviewed replacements, and release assets. If verification fails after a PR is
created, the artifact instead remains `PENDING_HUB_PR_UNVERIFIED`. The workflow
never creates a tag or a `MEASURED` receipt for a proposal.

The Hub PR needs owner review and a protected merge. After merge, a separate
finalization implementation must bind the merged Hub main to the proposal and
source bytes, create the immutable release tag, and read back the tag and all
files before it can emit a `MEASURED` publication receipt. That finalization
is outside this proposal change. Re-running the new-release dispatch can
create another Hub PR, so use the retained proposal artifact to reconcile an
existing proposal before any retry.

The `receipt_run` workflow input still performs read-only verification of a
successful historical publication receipt. A pending proposal run cannot be
used as a publication receipt.

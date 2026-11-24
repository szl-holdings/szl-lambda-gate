<!-- PROPOSE-ONLY / DO-NOT-MERGE. Study document only. No code, config, or workflow change is proposed by this file. -->
# LEADERS-STUDY -- szl-lambda-gate (tranche 6, 2026-11-24)

> **PROPOSE-ONLY / DO-NOT-MERGE.** Per-repo-domain public-surface study card from the round-2 leader pack.
> Nothing here is approved for implementation. Merge requires a separate governing review plus a fresh owner GO naming repo, PR, and pin.

## 0. Scope gate

Study-only. This PR adds exactly one documentation file (this file). No kernel change, no CI change, no publish-workflow change. All absorb candidates are UNAPPROVED.

## 1. Current state (measured evidence, 2026-11-24 qualification)

- Repo: szl-holdings/szl-lambda-gate, main tip at qualification 1aebffa22b9cd79834d1dc844d176695d30b7552, 87 files, zero open PRs at scan time.
NaN
- README doctrine: Lambda = weighted geometric mean of axis scores in [0,1]; any zeroed axis drives the aggregate to 0; ADVISORY, never proven trust; Lambda uniqueness = Conjecture 1, machine-checked false as stated; Theorem U conditional.
- Wave D consolidation: former szl-governed-norm folded in as szl_lambda_gate.governed_norm subpackage with SHA3-256 hash-chained governance receipts; szl-governed-norm DEPRECATED, points here, not deleted.
- Kernel posture: universal (pure-PyTorch) port of the canonical Lambda aggregator; builds torch-cpu and torch-universal; get_kernel(..., version=1) semantics verified against this repo.

## 2. Field leaders (public-surface study, round-2 leader pack)

| # | Leader | Public source | Leads at |
|---|---|---|---|
| L1 | FlashAttention-4 | arXiv 2603.05451; tridao.me 2026 flash4 blog; pytorch.org low-precision FA4 post | Asymmetric-hardware co-design (CuTe-DSL, software-emulated exp/softmax) on Blackwell |
| L2 | ThunderKittens 2.0 (Hazy Research) | hazyresearch.stanford.edu 2026-02-19-tk-2; together.ai Vera Rubin NVL72 port post | Tile-level kernel DSL with clean portability across GPU generations |
| L3 | DeepGEMM (deepseek-ai) | github.com/deepseek-ai/DeepGEMM | Clean BLAS kernel-library reference; benchmark baseline discipline |
| L4 | HF Kernels Hub versioned serving | huggingface.co/kernels; get_kernel version pinning | Version-pinned kernel distribute + verify semantics (this repo already publishes here) |

## 3. Honest gaps (this repo vs the leaders)

- Gap A: single pure-PyTorch reference backend; leaders run hardware co-design (L1) or multi-generation tile portability (L2). No CUDA receipts exist and none are claimed.
- Gap B: no published signed benchmark receipts versus a DeepGEMM-style baseline (L3 posture): GPU/clocks/dtype/shape/tolerances columns are the leader norm.
- Gap C: speed is never attested as correctness here (good doctrine), but the complementary leader posture -- one kernel, two backends, identical numeric tolerances, one signed receipt per backend -- is unmet because there is no second backend.
- Honest non-gaps: fail-closed reference fallback, advisory-only Lambda, and Conjecture 1 bookkeeping are leader-grade honesty; kernel-publish SHA verification already mirrors L4.

## 4. Absorb candidates A1-A4 (ALL UNAPPROVED)

- A1: Signed benchmark receipts vs DeepGEMM-style baseline. Applicability test: receipts record GPU/clock/dtype/shape/tolerances; speed numbers never appear in correctness receipts.
- A2: TK-style tile abstraction as the portability reference for the second backend. Applicability test: one kernel, two backends, identical numeric tolerances, one signed receipt per backend; fail-closed to reference kernel on mismatch.
- A3: Formalize the fail-closed fallback into the benchmark receipt schema (which backend produced which number). Applicability test: consumers can never read a hardware result as universal.
- A4: Publish benchmark receipts alongside Kernel Hub version tags (v1/main branch posture unchanged). Applicability test: hub consumers can trace a version to its measured envelope.

## 5. Open questions A5-A6 (governing gates)

- A5 (owner GO): any hardware branch (consumer-GPU tier) requires a fresh owner GO naming repo + budget + hardware; round-1 posture holds until then.
- A6 (Kimi lap): do governance-receipt semantics (governed_norm fold-in, SHA3-256 chain) stay byte-stable across a second backend, or does each backend carry its own receipt family?

## 6. Non-goals

No implementation in this PR. No CUDA/Blackwell claims, no performance claims, no change to Lambda advisory status (still Conjecture 1; never described as proven). No merge without governing review + fresh owner GO naming repo, PR, and pin.

## 7. Markers and receipts

LEADERS_STUDY_TRANCH6_CARD_4_OF_6 | PROPOSE_ONLY_DNM | qualification tip 1aebffa2 | erf scan C:/SZL/edge-debug/sp9_census.txt | qualification receipt ft_h4_t6qual.json (2026-11-24T13:46:23Z) | leader pack round2-leader-pack.md.

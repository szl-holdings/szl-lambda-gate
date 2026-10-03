# Offline `szl.lambda/v1` replay

`scripts/replay_lambda_v1.py` reads this repository's frozen 60-vector fixture,
contract, and stdlib reference. It does not import Torch, load a model, call a
provider, or run training. The original reference formula, fixture, and spec are
unchanged. A clean replay requires bit-for-bit reference values and exact error,
verdict, and gate codes.

## Run and test

With an existing Python 3.9+ interpreter, choose a report path **outside this
checkout**:

```text
python -B -I scripts/replay_lambda_v1.py --output <external-path>/lambda-v1-report.json
python -B -I -m unittest discover -s tests -p test_lambda_replay_stdlib.py -v
```

The runner writes deterministic JSON and a neighboring `.sha256` file. It exits
nonzero unless all 60 clean rows and all four corruption probes produce their
expected states. The report records the Git checkout HEAD when available, the
declared frozen-fixture origin, fixture/reference/contract Git blob IDs and
SHA-256 hashes, the driver hash, and the Python/OS identity. The runner checks
the staged bytes against those blob IDs. A shallow checkout may not contain the
historical origin commit object; the declared origin needs source-history
verification separately. Its 60 clean rows retain
the exact expected and observed outcomes.

## Corruption probes

The probes are newly constructed in memory. They never modify the committed
fixture or promote altered data as source evidence.

| Probe | Replay-envelope result | Evaluated authoritative rows |
|---|---|---:|
| Change `nominal`'s expected value to 0.5 | `FAIL / FIXTURE_CHANGED`; a diagnostic comparison shows the numerical mismatch | 0 |
| Change the envelope unit from dimensionless to joule | `ABSTAIN / UNIT_LABEL_MISMATCH` | 0 |
| Remove `nominal.axes` | `ABSTAIN / MISSING_INPUT:axes` | 0 |
| Supply a stale claimed source hash | `ABSTAIN / STALE_HASH` | 0 |

These are **replay-envelope** states, separate from `gate_v1`'s `GO`, `NO_GO`,
`ABSTAIN`, and `BLOCK` verdicts. The unit label belongs to this replay envelope;
the upstream vector format has no unit field. A diagnostic may evaluate the
changed-value row after rejecting its fixture identity, but it is never an
authoritative row.

The committed vectors were generated with a scratch script that is not in the
repository. This tool replays the frozen source; it does not claim independent
vector regeneration. Passing the replay establishes numerical conformance for
the recorded code, vectors, and environment. It does not establish an empirical
scientific result, model quality, or proof of Lambda's open uniqueness
conjecture.

The existing pytest suite tests the reference and Torch boundaries separately.
Atelier's NumPy port should consume a reviewed, revision-bound report in a
separate change; this replay does not modify Atelier.

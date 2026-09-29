# SPDX-License-Identifier: Apache-2.0
"""Offline render of the Hub card's szl.lambda/v1 section (HF-01).

The release lane runs scripts/render_model_card.py in the step "Render and
validate model card" (hf-mirror.yml). The script reads three things written
by earlier steps: the captured Hub card (.hfmirror-baseline/README.md), the
GitHub release record (.hfmirror-release.json) and the staged payload
(.hfstage/). These tests build that workspace from fixtures in tmp_path and
run the real script.

Standard library and pytest only. The ci workflow installs torch and pytest,
but not huggingface_hub or PyYAML, so the script runs against small in-memory
stand-ins for both. No test touches the network.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import runpy
import sys
import types
from pathlib import Path

import pytest

from test_lambda_v1_vectors import (
    ORG_RULE_A,
    ORG_RULE_A_SAFE,
    ORG_RULE_B,
    ORG_RULE_B_SAFE,
    STRICT_CONJECTURE,
    STRICT_PROOF,
    STRICT_UNIQUE,
)

ROOT = Path(__file__).resolve().parents[1]
RENDERER = ROOT / "scripts" / "render_model_card.py"
PUBLISHER = ROOT / "scripts" / "hf_mirror_release.py"

spec = importlib.util.spec_from_file_location("hf_mirror_release_card", PUBLISHER)
assert spec and spec.loader
mirror = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mirror)
BEGIN, END = mirror.BEGIN, mirror.END

REPO_ID = "SZLHOLDINGS/szl-lambda-gate"
GH_REPO = "szl-holdings/szl-lambda-gate"
SOURCE_SHA = "0123456789abcdef0123456789abcdef01234567"
TAG = "v0.2.0"
SOURCE_README = b"source readme\n"
CONTRACT = "spec/szl.lambda.v1.json"
VECTORS = "spec/lambda_v1_vectors.json"
PAYLOAD = (CONTRACT, VECTORS, "reference/szl_lambda_v1.py", "frontier/model_admit_contract.v1.json")
RECORDED = json.loads((ROOT / CONTRACT).read_text(encoding="utf-8"))["vectors"]["sha256"]
HEADING = "### szl.lambda/v1"

# Front matter as curated on the Hub (MCP read of Hub main, 2026-09-29).
BASELINE_FRONT = {
    "library_name": "kernels",
    "license": "apache-2.0",
    "tags": ["kernel", "governance", "lambda", "doi:10.5281/zenodo.19944926"],
}
CURATED = (
    "> **SOFTWARE / ADVISORY KERNEL · NO PROMOTION**\n"
    ">\n"
    "> A curated correction the mirror must never touch.\n"
    "\n"
    "## Quickstart\n"
    "\n"
    "```python\n"
    "from kernels import get_kernel\n"
    "```\n"
)
PRIOR_BLOCK = (
    f"{BEGIN}\n## Mirrored release v0.1.0\n\nOld release notes.\n\n"
    "| Field | Value |\n|---|---|\n"
    f"| Source commit | `{'c' * 40}` |\n| Hub revision | `v0.1.0` |\n{END}"
)
BASELINES = {
    "block-at-end": CURATED + "\n" + PRIOR_BLOCK + "\n",
    "block-mid-card": CURATED + "\n" + PRIOR_BLOCK + "\n\n## Curated tail\n\nKept byte for byte.\n",
    "no-block": CURATED,
}


# --- stand-ins for huggingface_hub and yaml -----------------------------------


def dump_front(front: dict) -> str:
    """One `key: <json>` line per field. JSON scalars and lists are valid YAML."""
    return "\n".join(f"{key}: {json.dumps(value, ensure_ascii=False)}" for key, value in front.items())


def split_card(content: str) -> tuple[dict, str]:
    match = re.match(r"^---\n(.*?)\n---\n", content, re.S)
    assert match, "card has no front matter"
    front = {}
    for line in match.group(1).splitlines():
        key, value = line.split(": ", 1)
        front[key] = json.loads(value)
    return front, content[match.end():]


def install_stand_ins(monkeypatch: pytest.MonkeyPatch) -> None:
    hub = types.ModuleType("huggingface_hub")
    utils = types.ModuleType("huggingface_hub.utils")

    class HfHubHTTPError(Exception):
        pass

    class ModelCardData:
        def __init__(self, **fields: object) -> None:
            self._fields = dict(fields)

        def to_dict(self) -> dict:
            return dict(self._fields)

        def __str__(self) -> str:
            return dump_front(self._fields)

    class ModelCard:
        def __init__(self, content: str) -> None:
            self.content = content
            front, self.text = split_card(content)
            self.data = ModelCardData(**front)

        @classmethod
        def load(cls, source: object, repo_type: str | None = None) -> "ModelCard":
            path = Path(str(source))
            if not path.is_file():
                raise OSError(f"stand-in has no Hub access: {source}")
            return cls(path.read_text(encoding="utf-8"))

        def save(self, path: object) -> None:
            Path(str(path)).write_bytes(self.content.encode("utf-8"))

    def safe_load(_: str) -> None:
        raise AssertionError("the Hub-baseline path must not parse YAML")

    hub.ModelCard, hub.ModelCardData, hub.utils = ModelCard, ModelCardData, utils
    utils.HfHubHTTPError = HfHubHTTPError
    yaml = types.ModuleType("yaml")
    yaml.safe_load = safe_load
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.utils", utils)
    monkeypatch.setitem(sys.modules, "yaml", yaml)


# --- fixture workspace ----------------------------------------------------------


def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, baseline: str = BASELINES["block-at-end"],
              payload: tuple[str, ...] = PAYLOAD, subdir: str = ".") -> Path:
    """Lay out what the release lane has on disk when the card is rendered."""
    work = tmp_path / "work"
    stage = work / ".hfstage"
    stage.mkdir(parents=True)
    (stage / "README.md").write_bytes(SOURCE_README)
    for rel in payload:
        (stage / rel).parent.mkdir(parents=True, exist_ok=True)
        (stage / rel).write_bytes((ROOT / rel).read_bytes())
    (work / ".hfmirror-baseline").mkdir()
    card = "---\n" + dump_front(BASELINE_FRONT) + "\n---\n" + baseline
    (work / ".hfmirror-baseline" / "README.md").write_bytes(card.encode("utf-8"))
    (work / ".hfmirror-release.json").write_text(json.dumps({
        "tagName": TAG, "isDraft": False, "assets": [],
        "url": f"https://github.com/{GH_REPO}/releases/tag/{TAG}",
        "body": "Release notes for the fixture.",
    }), encoding="utf-8")
    env = {
        "HF_REPO_ID": REPO_ID, "HF_REPO_TYPE": "model", "RELEASE_TAG": TAG, "GITHUB_REPOSITORY": GH_REPO,
        "SOURCE_GITHUB_SHA": SOURCE_SHA, "SUBDIR": subdir, "HUB_CARD_PATH": ".hfmirror-baseline/README.md",
        "PRESERVE_HUB_CARD": "true", "CARD_TEMPLATE": "", "HF_TOKEN": "",
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.chdir(work)
    install_stand_ins(monkeypatch)
    return work


def render() -> tuple[dict, str]:
    runpy.run_path(str(RENDERER), run_name="__main__")
    return split_card((Path(".hfstage") / "README.md").read_text(encoding="utf-8"))


def block_of(body: str) -> str:
    assert body.count(BEGIN) == body.count(END) == 1
    return body.split(BEGIN, 1)[1].split(END, 1)[0]


def section_of(body: str) -> str:
    block = block_of(body)
    return block[block.index(HEADING):block.index("Mirror direction:")]


# --- A1: the section lives inside the mirror block ------------------------------


@pytest.mark.parametrize("shape", sorted(BASELINES))
def test_section_is_inside_the_mirror_block_and_the_rest_is_the_baseline(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shape: str) -> None:
    workspace(tmp_path, monkeypatch, baseline=BASELINES[shape])
    front, body = render()
    assert front == BASELINE_FRONT
    assert body.count(BEGIN) == body.count(END) == 1
    start, end = body.index(BEGIN), body.index(END)
    assert start < body.index(HEADING) < end
    assert "szl.lambda/v1" not in body[:start] + body[end:]
    # The same comparison publish() makes before it uploads anything.
    assert mirror.without_release_block(body) == mirror.without_release_block(BASELINES[shape])
    assert "v0.1.0" not in body


def test_section_points_at_the_v1_vectors_at_the_release_sha(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    work = workspace(tmp_path, monkeypatch)
    _, body = render()
    section = section_of(body)
    blob = f"https://github.com/{GH_REPO}/blob/{SOURCE_SHA}/"
    for rel in PAYLOAD:
        assert blob + rel in section, rel
    staged = (work / ".hfstage" / VECTORS).read_bytes()
    assert f"`{RECORDED}`" in section
    assert f"`{hashlib.sha256(staged).hexdigest()}`" in section
    assert "Λ uniqueness: Conjecture 1" in section
    assert "advisory kernel; the gate is floors ∧ Λ with τ from the admit contract" in section
    # publish() still finds the source binding it checks.
    block = block_of(body)
    assert SOURCE_SHA in block and f"## Mirrored release {TAG}" in block


def test_digests_come_from_the_staged_release_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    work = workspace(tmp_path, monkeypatch)
    vectors = work / ".hfstage" / VECTORS
    checkout = (ROOT / VECTORS).read_bytes()
    crlf = checkout.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    assert crlf != checkout
    vectors.write_bytes(crlf)
    _, body = render()
    section = section_of(body)
    assert f"`{hashlib.sha256(crlf).hexdigest()}`" in section
    assert hashlib.sha256(checkout).hexdigest() not in section
    # The parsed JSON is unchanged, so the canonical digest is too.
    assert f"`{RECORDED}`" in section


def test_links_carry_the_mirror_subdirectory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace(tmp_path, monkeypatch, subdir="kernel/")
    _, body = render()
    assert f"https://github.com/{GH_REPO}/blob/{SOURCE_SHA}/kernel/{VECTORS}" in section_of(body)


def test_re_rendering_replaces_the_block_instead_of_stacking(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    work = workspace(tmp_path, monkeypatch)
    render()
    first = (work / ".hfstage" / "README.md").read_bytes()
    (work / ".hfmirror-baseline" / "README.md").write_bytes(first)
    _, body = render()
    assert body.count(HEADING) == 1
    assert (work / ".hfstage" / "README.md").read_bytes() == first


# --- fail closed ------------------------------------------------------------------


def test_release_without_the_contract_renders_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace(tmp_path, monkeypatch, payload=())
    _, body = render()
    section = section_of(body)
    assert section.startswith(f"{HEADING}\n\nUNAVAILABLE:")
    assert "/blob/" not in block_of(body)
    assert re.search(r"[0-9a-f]{64}", block_of(body)) is None
    assert mirror.without_release_block(body) == mirror.without_release_block(BASELINES["block-at-end"])


def drop(rel: str):
    def mutate(stage: Path) -> None:
        (stage / rel).unlink()
    return mutate


def overwrite(rel: str, data: bytes):
    def mutate(stage: Path) -> None:
        (stage / rel).write_bytes(data)
    return mutate


def edit_contract(change):
    def mutate(stage: Path) -> None:
        path = stage / CONTRACT
        doc = json.loads(path.read_text(encoding="utf-8"))
        change(doc)
        path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return mutate


def edit_vectors(change):
    def mutate(stage: Path) -> None:
        path = stage / VECTORS
        before = path.read_text(encoding="utf-8")
        after = change(before)
        assert after != before, "fixture edit matched nothing"
        path.write_text(after, encoding="utf-8")
    return mutate


def set_schema(doc: dict) -> None:
    doc["schema"] = "szl.lambda/v2"


def set_honesty_field(doc: dict) -> None:
    doc["uniqueness"] = "ASSUMED"


def set_vectors_path(doc: dict) -> None:
    doc["vectors"]["path"] = "spec/other_vectors.json"


def set_digest_recipe(doc: dict) -> None:
    doc["vectors"]["digest"]["sort_keys"] = False


def set_bad_digest(doc: dict) -> None:
    doc["vectors"]["sha256"] = "not-a-digest"


def set_count(doc: dict) -> None:
    doc["vectors"]["count"] += 1


def set_reference(doc: dict) -> None:
    doc["reference"] = "reference/other.py"


def set_tau_source(doc: dict) -> None:
    doc["gate"]["tau"]["default_source"] = "frontier/other.json#/policy_tau"


def rename_first_vector(text: str) -> str:
    return text.replace('"id": "nominal"', '"id": "nominal-edited"', 1)


def re_record(stage: Path, *, count: object, allow_nan: bool = False) -> None:
    """Make the staged contract agree with the staged vectors.

    The digest recipe is the contract's, except that `allow_nan` can be relaxed
    to build a pair that only the renderer's NaN rule can reject.
    """
    doc = json.loads((stage / VECTORS).read_text(encoding="utf-8"))
    canonical = json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=allow_nan)
    contract = json.loads((stage / CONTRACT).read_text(encoding="utf-8"))
    contract["vectors"]["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    contract["vectors"]["count"] = count
    (stage / CONTRACT).write_text(json.dumps(contract, ensure_ascii=False), encoding="utf-8")


def keep_one_vector(stage: Path) -> None:
    doc = json.loads((stage / VECTORS).read_text(encoding="utf-8"))
    doc["vectors"] = doc["vectors"][:1]
    (stage / VECTORS).write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")


def nan_with_matching_digest(stage: Path) -> None:
    # The recorded digest matches the NaN-bearing bytes, so the only thing
    # that can reject this payload is the rule that NaN has no canonical form.
    text = (stage / VECTORS).read_text(encoding="utf-8")
    assert text.count('"value_tol": 1e-12') >= 1
    (stage / VECTORS).write_text(text.replace('"value_tol": 1e-12', '"value_tol": NaN', 1), encoding="utf-8")
    re_record(stage, count=50, allow_nan=True)


def bool_count_with_matching_length(stage: Path) -> None:
    # True == 1, so only the int type rule can reject a count of True here.
    keep_one_vector(stage)
    re_record(stage, count=True)


UNREADABLE = "FAIL: szl.lambda/v1 contract or vectors unreadable ({})"
DIFFERS = "FAIL: szl.lambda/v1 contract differs from what the card states: {}"
INCOMPLETE = "FAIL: szl.lambda/v1 payload incomplete: {}"
DIGEST = "FAIL: szl.lambda/v1 vectors digest differs from the contract"

BROKEN = {
    "vectors-missing": (drop(VECTORS), INCOMPLETE.format(VECTORS)),
    "reference-missing": (drop("reference/szl_lambda_v1.py"), INCOMPLETE.format("reference/szl_lambda_v1.py")),
    "admit-contract-missing": (drop("frontier/model_admit_contract.v1.json"),
                               INCOMPLETE.format("frontier/model_admit_contract.v1.json")),
    "contract-not-json": (overwrite(CONTRACT, b"["), UNREADABLE.format("JSONDecodeError")),
    "contract-not-an-object": (overwrite(CONTRACT, b"[]"), UNREADABLE.format("TypeError")),
    "vectors-not-json": (overwrite(VECTORS, b"{not json"), UNREADABLE.format("JSONDecodeError")),
    "vectors-not-utf8": (overwrite(VECTORS, b"\xff\xfe"), UNREADABLE.format("UnicodeDecodeError")),
    "vectors-edited": (edit_vectors(rename_first_vector), DIGEST),
    "vectors-carry-nan": (nan_with_matching_digest, UNREADABLE.format("ValueError")),
    "schema-changed": (edit_contract(set_schema), DIFFERS.format("schema")),
    "honesty-field-changed": (edit_contract(set_honesty_field), DIFFERS.format("uniqueness")),
    "vectors-path-redirected": (edit_contract(set_vectors_path), DIFFERS.format("vectors.path")),
    "digest-recipe-changed": (edit_contract(set_digest_recipe), DIFFERS.format("vectors.digest")),
    "recorded-digest-malformed": (edit_contract(set_bad_digest), DIFFERS.format("vectors.sha256")),
    "count-mismatch": (edit_contract(set_count), DIFFERS.format("vectors.count")),
    "count-is-bool": (bool_count_with_matching_length, DIFFERS.format("vectors.count")),
    "reference-redirected": (edit_contract(set_reference), DIFFERS.format("reference")),
    "tau-source-redirected": (edit_contract(set_tau_source), DIFFERS.format("gate.tau.default_source")),
}


@pytest.mark.parametrize("case", sorted(BROKEN))
def test_inconsistent_contract_payload_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str) -> None:
    work = workspace(tmp_path, monkeypatch)
    mutate, expected = BROKEN[case]
    mutate(work / ".hfstage")
    with pytest.raises(SystemExit) as failure:
        render()
    assert failure.value.code == expected
    assert (work / ".hfstage" / "README.md").read_bytes() == SOURCE_README


def test_re_recorded_payload_is_consistent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Control for the NaN and bool cases above: re_record() on its own yields a
    # payload the renderer accepts, so those cases fail only on the rule named.
    work = workspace(tmp_path, monkeypatch)
    keep_one_vector(work / ".hfstage")
    re_record(work / ".hfstage", count=1)
    _, body = render()
    assert "| Golden vectors (1) |" in section_of(body)


# --- honesty ------------------------------------------------------------------------


@pytest.mark.parametrize("payload", [PAYLOAD, ()], ids=["contract", "no-contract"])
def test_section_makes_no_proof_or_uniqueness_claim(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, payload: tuple[str, ...]) -> None:
    workspace(tmp_path, monkeypatch, payload=payload)
    _, body = render()
    for line in block_of(body).splitlines():
        assert not STRICT_PROOF.search(line), line
        if STRICT_UNIQUE.search(line):
            assert STRICT_CONJECTURE.search(line), line
        if ORG_RULE_A.search(line):
            assert ORG_RULE_A_SAFE.search(line), line
        if ORG_RULE_B.search(line):
            assert ORG_RULE_B_SAFE.search(line), line
    assert "MEASURED" not in section_of(body)

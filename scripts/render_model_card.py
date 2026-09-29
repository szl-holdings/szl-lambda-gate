#!/usr/bin/env python3
"""Render .hfstage/README.md for the Hugging Face mirror.

Card source precedence:
  1. captured Hub baseline (HUB_CARD_PATH) for curated cards
  2. local template (CARD_TEMPLATE), when the repo wants GitHub to own the card
  3. the live Hub card, or staged README only when preservation is not required

Front matter from the chosen source is preserved as-is. Nothing is defaulted
except `license`, which the Hub requires. The release section is written between
idempotent markers so re-mirroring a tag replaces the block instead of stacking
copies. Fail closed: invalid front matter or no card source exits non-zero.

The block also points at the szl.lambda/v1 contract and golden vectors staged
for this release. The pointer lives inside the markers, so the curated card
outside them stays byte-identical to the Hub baseline.
"""
import hashlib
import os
import pathlib
import re
import sys
import json

import yaml
from huggingface_hub import ModelCard, ModelCardData
from huggingface_hub.utils import HfHubHTTPError

STAGE = pathlib.Path(".hfstage")
BEGIN = "<!-- SZL-HF-MIRROR:START -->"
END = "<!-- SZL-HF-MIRROR:END -->"
# The files the szl.lambda/v1 section links. Each one must be in the staged
# payload, because a link must name bytes the Hub revision actually holds.
LAMBDA_V1 = {
    "contract": "spec/szl.lambda.v1.json",
    "vectors": "spec/lambda_v1_vectors.json",
    "reference": "reference/szl_lambda_v1.py",
    "admit": "frontier/model_admit_contract.v1.json",
}
LAMBDA_V1_DIGEST = {"algorithm": "sha256", "over": "canonical_json", "sort_keys": True,
                    "separators": [",", ":"], "ensure_ascii": False, "allow_nan": False,
                    "encoding": "utf-8"}

repo_id = os.environ["HF_REPO_ID"]
repo_type = os.environ.get("HF_REPO_TYPE", "model")
tag = os.environ.get("RELEASE_TAG", "untagged")
gh_repo = os.environ.get("GITHUB_REPOSITORY", "szl-holdings/unknown")
gh_sha = os.environ.get("SOURCE_GITHUB_SHA", "")
if not re.fullmatch(r"[0-9a-f]{40}", gh_sha):
    sys.exit("FAIL: SOURCE_GITHUB_SHA must be the verified release tag commit")
release = json.loads(pathlib.Path(".hfmirror-release.json").read_text(encoding="utf-8"))
if release.get("tagName") != tag or release.get("isDraft"):
    sys.exit("FAIL: published GitHub release/tag mismatch")
notes = (release.get("body") or "").strip()
release_url = release.get("url") or f"https://github.com/{gh_repo}/releases/tag/{tag}"


def split_front(raw):
    m = re.match(r"^---\n(.*?)\n---\n", raw, re.S)
    if m:
        return yaml.safe_load(m.group(1)) or {}, raw[m.end():]
    return {}, raw


def lambda_v1_section():
    """Point the card at the szl.lambda/v1 contract carried by this release.

    Every value comes from the staged payload: the bytes the Hub receives at the
    release commit. A release that predates the contract renders UNAVAILABLE and
    links nothing. A release that carries the contract, but not consistent
    vectors, reference and admit contract, fails closed.
    """
    staged = {name: STAGE / rel for name, rel in LAMBDA_V1.items()}
    if not staged["contract"].is_file():
        return (f"### szl.lambda/v1\n\nUNAVAILABLE: this release does not carry "
                f"`{LAMBDA_V1['contract']}`, so no vectors are linked.")
    missing = [LAMBDA_V1[name] for name, path in staged.items() if not path.is_file()]
    if missing:
        sys.exit(f"FAIL: szl.lambda/v1 payload incomplete: {', '.join(missing)}")
    try:
        spec = json.loads(staged["contract"].read_text(encoding="utf-8"))
        vectors = spec["vectors"]
        pinned = {
            "schema": spec["schema"] == "szl.lambda/v1",
            "uniqueness": spec["uniqueness"] == "CONJECTURE_1_NOT_USED",
            "vectors.path": vectors["path"] == LAMBDA_V1["vectors"],
            "vectors.digest": vectors["digest"] == LAMBDA_V1_DIGEST,
            "vectors.sha256": re.fullmatch(r"[0-9a-f]{64}", vectors["sha256"]) is not None,
            "reference": spec["reference"] == LAMBDA_V1["reference"],
            "gate.tau.default_source":
                spec["gate"]["tau"]["default_source"].split("#", 1)[0] == LAMBDA_V1["admit"],
        }
        raw = staged["vectors"].read_bytes()
        doc = json.loads(raw.decode("utf-8"))
        canonical = json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                               allow_nan=False).encode("utf-8")
        pinned["vectors.count"] = type(vectors["count"]) is int and len(doc["vectors"]) == vectors["count"]
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        sys.exit(f"FAIL: szl.lambda/v1 contract or vectors unreadable ({exc.__class__.__name__})")
    broken = sorted(name for name, ok in pinned.items() if not ok)
    if broken:
        sys.exit(f"FAIL: szl.lambda/v1 contract differs from what the card states: {', '.join(broken)}")
    if hashlib.sha256(canonical).hexdigest() != vectors["sha256"]:
        sys.exit("FAIL: szl.lambda/v1 vectors digest differs from the contract")
    sub = os.environ.get("SUBDIR", ".").strip("/")
    blob = f"https://github.com/{gh_repo}/blob/{gh_sha}/" + ("" if sub in ("", ".") else sub + "/")
    return f"""### szl.lambda/v1

| Field | Value |
|---|---|
| Contract | {blob}{LAMBDA_V1['contract']} |
| Golden vectors ({vectors['count']}) | {blob}{LAMBDA_V1['vectors']} |
| Vectors SHA-256, canonical JSON (recorded in the contract) | `{vectors['sha256']}` |
| Vectors SHA-256, file bytes in this revision | `{hashlib.sha256(raw).hexdigest()}` |
| Stdlib reference | {blob}{LAMBDA_V1['reference']} |
| Admit contract (τ) | {blob}{LAMBDA_V1['admit']} |

Λ uniqueness: Conjecture 1 (open); nothing in the contract depends on it.
This is an advisory kernel; the gate is floors ∧ Λ with τ from the admit contract."""


front, body, origin = {}, "", None

preserve = os.environ.get("PRESERVE_HUB_CARD", "").lower() == "true"
baseline_env = os.environ.get("HUB_CARD_PATH") or ""
baseline = pathlib.Path(baseline_env) if baseline_env else None
tpl_env = os.environ.get("CARD_TEMPLATE") or ""
tpl = pathlib.Path(tpl_env) if tpl_env else None
staged = STAGE / "README.md"

if preserve and (baseline is None or not baseline.is_file()):
    sys.exit("FAIL: preserved Hub card baseline is missing")

if baseline and baseline.is_file():
    hub_card = ModelCard.load(baseline)
    front = dict(hub_card.data.to_dict())
    body = hub_card.text
    origin = f"baseline:{baseline}"
elif tpl and tpl.is_file():
    front, body = split_front(tpl.read_text(encoding="utf-8"))
    origin = f"template:{tpl}"
else:
    try:
        hub_card = ModelCard.load(repo_id, repo_type=repo_type)
        front = dict(hub_card.data.to_dict())
        body = hub_card.text
        origin = f"hub:{repo_id}"
    except (HfHubHTTPError, OSError, ValueError) as exc:
        if preserve:
            sys.exit(f"FAIL: curated Hub card unavailable ({exc.__class__.__name__})")
        print(f"::notice::could not load Hub card ({exc.__class__.__name__}); falling back to payload")
        if staged.is_file():
            front, body = split_front(staged.read_text(encoding="utf-8"))
            origin = "payload:README.md"

if origin is None:
    sys.exit("FAIL: no card source - no template, no Hub card, no staged README")

# Only the Hub-required field is defaulted. Curated metadata is never overwritten.
front.setdefault("license", "apache-2.0")
if front["license"] != "other":
    front.pop("license_name", None)
    front.pop("license_link", None)

block = f"""{BEGIN}
## Mirrored release {tag}

{notes if notes else "No release notes were supplied for this tag."}

| Field | Value |
|---|---|
| GitHub source | https://github.com/{gh_repo} |
| GitHub release | {release_url} |
| Source commit | `{gh_sha}` |
| Hub revision | `{tag}` |

{lambda_v1_section()}

Mirror direction: GitHub is authoritative for code; Hub-only artifacts in this
repo are preserved and never deleted by the mirror. Truth state is MEASURED only
after the workflow re-reads this repo from the Hub.
{END}"""

if BEGIN in body and END in body:
    body = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END), block, body, flags=re.S)
else:
    body = body.rstrip() + "\n\n" + block + "\n"

# Hub card text and release notes are untrusted Markdown. Do not evaluate them
# as a Jinja template in the credentialed release job.
card = ModelCard("---\n" + str(ModelCardData(**front)).rstrip() + "\n---\n" + body)
STAGE.mkdir(parents=True, exist_ok=True)
card.save(staged)
print(f"rendered {staged} from {origin} | license={front['license']} | library_name={front.get('library_name', 'UNSET')}")

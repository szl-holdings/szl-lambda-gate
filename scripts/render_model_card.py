#!/usr/bin/env python3
"""Render .hfstage/README.md for the Hugging Face mirror.

Front matter is authoritative and validated. Release notes from the GitHub
release are appended verbatim under a versioned heading, then a provenance
stamp with a literal backlink to the GitHub source. Fail closed: an invalid
card exits non-zero before anything is pushed to the Hub.
"""
import os
import pathlib
import re
import sys

import yaml
from huggingface_hub import ModelCard, ModelCardData

STAGE = pathlib.Path(".hfstage")
REQUIRED = ("license", "library_name", "pipeline_tag")

repo_id = os.environ["HF_REPO_ID"]
repo_type = os.environ.get("HF_REPO_TYPE", "model")
tag = os.environ.get("RELEASE_TAG", "untagged")
gh_repo = os.environ.get("GITHUB_REPOSITORY", "szl-holdings/unknown")
gh_sha = os.environ.get("GITHUB_SHA", "")
notes = (os.environ.get("RELEASE_BODY") or "").strip()
release_url = os.environ.get("RELEASE_URL") or f"https://github.com/{gh_repo}/releases/tag/{tag}"

template = pathlib.Path(os.environ.get("CARD_TEMPLATE") or "docs/card_body.md")
existing = STAGE / "README.md"

body = ""
front = {}
if template.is_file():
    body = template.read_text(encoding="utf-8")
elif existing.is_file():
    raw = existing.read_text(encoding="utf-8")
    m = re.match(r"^---\n(.*?)\n---\n", raw, re.S)
    if m:
        front = yaml.safe_load(m.group(1)) or {}
        body = raw[m.end():]
    else:
        body = raw
else:
    sys.exit(f"FAIL: no card template at {template} and no existing README in payload")

front.setdefault("license", "apache-2.0")
front.setdefault("library_name", "peft")
front.setdefault("pipeline_tag", "text-generation")
front.setdefault("tags", [])
for t in ("szl-holdings", "governed-ai", "receipts", f"release-{tag}"):
    if t not in front["tags"]:
        front["tags"].append(t)

# license_name / license_link are only honored when license == "other".
if front["license"] != "other":
    front.pop("license_name", None)
    front.pop("license_link", None)

missing = [k for k in REQUIRED if k not in front]
if missing:
    sys.exit(f"FAIL: front matter missing {missing}")

body = body.rstrip() + f"""

## Release {tag}

{notes if notes else "No release notes were supplied for this tag."}

## Provenance

- GitHub source: https://github.com/{gh_repo}
- GitHub release: {release_url}
- Source commit: `{gh_sha}`
- Hub revision: `{tag}` (created by the hf-mirror workflow)
- Mirror direction: GitHub is authoritative; this Hub repo is a projection.
- Truth state: MEASURED only after the workflow re-reads this repo from the Hub.
"""

card = ModelCard.from_template(
    card_data=ModelCardData(**front),
    model_id=repo_id,
    template_str="---\n{{ card_data }}\n---\n" + body,
)
STAGE.mkdir(parents=True, exist_ok=True)
card.save(existing)
print(f"rendered {existing} for {repo_id} ({repo_type}) @ {tag}")

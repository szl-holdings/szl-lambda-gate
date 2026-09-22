#!/usr/bin/env python3
"""Render .hfstage/README.md for the Hugging Face mirror.

Card source precedence:
  1. local template (CARD_TEMPLATE), when the repo wants GitHub to own the card
  2. the LIVE Hub card (ModelCard.load), so a hand-curated Hub card is preserved
  3. the staged README from the payload

Front matter from the chosen source is preserved as-is. Nothing is defaulted
except `license`, which the Hub requires. The release section is written between
idempotent markers so re-mirroring a tag replaces the block instead of stacking
copies. Fail closed: invalid front matter or no card source exits non-zero.
"""
import os
import pathlib
import re
import sys

import yaml
from huggingface_hub import ModelCard, ModelCardData
from huggingface_hub.utils import HfHubHTTPError

STAGE = pathlib.Path(".hfstage")
BEGIN = "<!-- SZL-HF-MIRROR:START -->"
END = "<!-- SZL-HF-MIRROR:END -->"

repo_id = os.environ["HF_REPO_ID"]
repo_type = os.environ.get("HF_REPO_TYPE", "model")
tag = os.environ.get("RELEASE_TAG", "untagged")
gh_repo = os.environ.get("GITHUB_REPOSITORY", "szl-holdings/unknown")
gh_sha = os.environ.get("GITHUB_SHA", "")
notes = (os.environ.get("RELEASE_BODY") or "").strip()
release_url = os.environ.get("RELEASE_URL") or f"https://github.com/{gh_repo}/releases/tag/{tag}"


def split_front(raw):
    m = re.match(r"^---\n(.*?)\n---\n", raw, re.S)
    if m:
        return yaml.safe_load(m.group(1)) or {}, raw[m.end():]
    return {}, raw


front, body, origin = {}, "", None

tpl_env = os.environ.get("CARD_TEMPLATE") or ""
tpl = pathlib.Path(tpl_env) if tpl_env else None
staged = STAGE / "README.md"

if tpl and tpl.is_file():
    front, body = split_front(tpl.read_text(encoding="utf-8"))
    origin = f"template:{tpl}"
else:
    try:
        hub_card = ModelCard.load(repo_id, repo_type=repo_type)
        front = dict(hub_card.data.to_dict())
        body = hub_card.text
        origin = f"hub:{repo_id}"
    except (HfHubHTTPError, OSError, ValueError) as exc:
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

Mirror direction: GitHub is authoritative for code; Hub-only artifacts in this
repo are preserved and never deleted by the mirror. Truth state is MEASURED only
after the workflow re-reads this repo from the Hub.
{END}"""

if BEGIN in body and END in body:
    body = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END), block, body, flags=re.S)
else:
    body = body.rstrip() + "\n\n" + block + "\n"

card = ModelCard.from_template(
    card_data=ModelCardData(**front),
    model_id=repo_id,
    template_str="---\n{{ card_data }}\n---\n" + body,
)
STAGE.mkdir(parents=True, exist_ok=True)
card.save(staged)
print(f"rendered {staged} from {origin} | license={front['license']} | library_name={front.get('library_name', 'UNSET')}")

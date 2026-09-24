import json
from pathlib import Path

import pytest

from scripts import hf_mirror_release as mirror


WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "hf-mirror.yml"


def test_release_mirror_explicitly_exchanges_oidc_token() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "id-token: write" in text
    assert "HF_OIDC_RESOURCE:" in text
    assert "hf auth token" in text
    assert "hf auth whoami" not in text
    assert 'echo "::add-mask::$oidc_token"' in text
    assert "printf 'HF_TOKEN=%s\\n' \"$oidc_token\" >> \"$GITHUB_ENV\"" in text
    assert 'echo "HF_AUTH_MODE=oidc" >> "$GITHUB_ENV"' in text
    assert "HF_OIDC_RESOURCE: ''" in text


def test_release_mirror_requires_branch_bound_oidc() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "workflow_run_id" in text
    assert 'gh run watch "$run_id"' in text
    assert 'github.ref == format(\'refs/heads/{0}\', github.event.repository.default_branch)' in text
    assert "trusted-publisher exchange failed" in text
    assert "HF_FALLBACK_TOKEN" not in text


def test_release_lane_uses_reviewed_code_and_never_moves_hub_tags() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    publisher = Path(mirror.__file__).read_text(encoding="utf-8")

    assert "path: release-source" in workflow
    assert "ref: refs/tags/${{ inputs.tag }}" in workflow
    assert "SOURCE_GITHUB_SHA=$source_sha" in workflow
    assert "api.delete_tag" not in publisher
    assert "exist_ok=False" in publisher
    assert "parent_commit=base[\"sha\"]" in publisher
    assert "revision=oid" in publisher


def test_release_asset_digest_mismatch_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RELEASE_TAG", "v0.1.0")
    mirror.STAGE.mkdir()
    mirror.ASSETS_DIR.mkdir()
    (mirror.ASSETS_DIR / "receipt.json").write_bytes(b"different")
    mirror.RELEASE_FILE.write_text(json.dumps({
        "tagName": "v0.1.0", "isDraft": False,
        "assets": [{"name": "receipt.json", "size": 9, "digest": "sha256:" + "0" * 64}],
    }), encoding="utf-8")

    with pytest.raises(RuntimeError, match="digest mismatch"):
        mirror.assets()
    assert not (mirror.STAGE / "receipt.json").exists()


def test_release_block_preserves_surrounding_card_text() -> None:
    baseline = "Curated correction and quickstart.\n"
    rendered = baseline.rstrip() + "\n\n" + mirror.BEGIN + "\nrelease\n" + mirror.END + "\n"
    assert mirror.without_release_block(rendered) == mirror.without_release_block(baseline)
    with pytest.raises(RuntimeError, match="unbalanced"):
        mirror.without_release_block(mirror.BEGIN + "oops")


def test_card_renderer_does_not_evaluate_hub_markdown_as_template() -> None:
    renderer = (WORKFLOW.parents[2] / "scripts" / "render_model_card.py").read_text(encoding="utf-8")
    assert "ModelCard.from_template" not in renderer
    assert "ModelCard(\"---\\n\"" in renderer

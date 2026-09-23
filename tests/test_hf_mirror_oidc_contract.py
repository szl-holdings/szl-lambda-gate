from pathlib import Path


WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "hf-mirror.yml"


def test_release_mirror_explicitly_exchanges_oidc_token() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "id-token: write" in text
    assert "HF_OIDC_RESOURCE:" in text
    assert "hf auth token" in text
    assert "hf auth whoami" not in text
    assert 'echo "::add-mask::$oidc_token"' in text
    assert "printf 'HF_TOKEN=%s\\n' \"$oidc_token\" >> \"$GITHUB_ENV\"" in text
    assert 'echo "auth=oidc" >> "$GITHUB_ENV"' in text


def test_release_mirror_keeps_fail_closed_fallback_boundary() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "HF_FALLBACK_TOKEN: ${{ secrets.HF_TOKEN }}" in text
    assert "trusted-publisher exchange failed and no HF_TOKEN" in text
    assert 'echo "::add-mask::$HF_FALLBACK_TOKEN"' in text
    assert 'printf \'HF_TOKEN=%s\\n\' "$HF_FALLBACK_TOKEN" >> "$GITHUB_ENV"' in text

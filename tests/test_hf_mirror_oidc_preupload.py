"""Offline contract for the metadata-only OIDC PR permission probe."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from urllib.error import HTTPError

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "hf_mirror_oidc_probe.py"
spec = importlib.util.spec_from_file_location("hf_mirror_oidc_probe_test", SCRIPT)
assert spec and spec.loader
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_probe_negotiates_pr_upload_without_content_mutation(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    token = "hf_jwt_example-secret"
    monkeypatch.setenv("HF_REPO_ID", "SZLHOLDINGS/szl-lambda-gate")
    monkeypatch.setenv("HF_REPO_TYPE", "model")
    monkeypatch.setenv("HF_TOKEN", token)
    requests = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self) -> bytes:
            return b'{"files":[{"path":".szl-oidc-pr-probe.txt","uploadMode":"regular"}]}'

    def urlopen(request, *, timeout):
        requests.append((request, timeout))
        return Response()

    monkeypatch.setattr(probe.request, "urlopen", urlopen)
    assert probe.probe() == "OIDC_PR_PREUPLOAD_ALLOWED"
    assert len(requests) == 1
    outgoing, timeout = requests[0]
    assert timeout == 15
    assert outgoing.get_method() == "POST"
    assert outgoing.full_url == (
        "https://huggingface.co/api/models/SZLHOLDINGS/szl-lambda-gate/"
        "preupload/main?create_pr=1"
    )
    assert outgoing.get_header("Authorization") == f"Bearer {token}"
    assert json.loads(outgoing.data) == {"files": [
        {"path": ".szl-oidc-pr-probe.txt", "sample": "cHJvYmU=", "size": 5},
    ]}
    assert token not in capsys.readouterr().out


def test_probe_reports_only_status_on_denial(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    token = "hf_jwt_example-secret"
    monkeypatch.setenv("HF_REPO_ID", "SZLHOLDINGS/szl-lambda-gate")
    monkeypatch.setenv("HF_REPO_TYPE", "model")
    monkeypatch.setenv("HF_TOKEN", token)

    def denied(request, *, timeout):
        raise HTTPError(request.full_url, 403, "Forbidden", None, None)

    monkeypatch.setattr(probe.request, "urlopen", denied)
    with pytest.raises(RuntimeError, match="HTTP 403") as error:
        probe.probe()
    assert token not in str(error.value) + capsys.readouterr().out


def test_probe_rejects_wrong_target_before_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_REPO_ID", "someone/else")
    monkeypatch.setenv("HF_REPO_TYPE", "model")
    monkeypatch.setenv("HF_TOKEN", "hf_jwt_example-secret")
    monkeypatch.setattr(probe.request, "urlopen", lambda *_args, **_kwargs: pytest.fail("network called"))
    with pytest.raises(RuntimeError, match="canonical Hub target required"):
        probe.probe()


def test_probe_rejects_user_scoped_token_before_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_REPO_ID", "SZLHOLDINGS/szl-lambda-gate")
    monkeypatch.setenv("HF_REPO_TYPE", "model")
    monkeypatch.setenv("HF_TOKEN", "hf_oauth_read-only")
    monkeypatch.setattr(probe.request, "urlopen", lambda *_args, **_kwargs: pytest.fail("network called"))
    with pytest.raises(RuntimeError, match="repo-scoped OIDC token required"):
        probe.probe()

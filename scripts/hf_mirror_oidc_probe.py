#!/usr/bin/env python3
"""Check the Hub PR preupload permission without uploading or committing a file."""
from __future__ import annotations

import json
import os
from urllib import error, request


REPO = "SZLHOLDINGS/szl-lambda-gate"
PATH = ".szl-oidc-pr-probe.txt"
PREUPLOAD_URL = f"https://huggingface.co/api/models/{REPO}/preupload/main?create_pr=1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def probe() -> str:
    require(os.environ.get("HF_REPO_ID") == REPO and os.environ.get("HF_REPO_TYPE") == "model",
            "canonical Hub target required for OIDC PR probe")
    token = os.environ.get("HF_TOKEN", "")
    require(token.startswith("hf_jwt_"), "repo-scoped OIDC token required for PR probe")
    payload = {"files": [{"path": PATH, "sample": "cHJvYmU=", "size": 5}]}
    outgoing = request.Request(
        PREUPLOAD_URL,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "szl-lambda-gate-oidc-pr-probe",
        },
        method="POST",
    )
    try:
        with request.urlopen(outgoing, timeout=15) as response:
            result = json.load(response)
    except error.HTTPError as exc:
        raise RuntimeError(f"UNAVAILABLE - OIDC PR preupload rejected: HTTP {exc.code}") from None
    except error.URLError as exc:
        raise RuntimeError(f"UNAVAILABLE - OIDC PR preupload transport error: {type(exc).__name__}") from None
    except ValueError:
        raise RuntimeError("UNAVAILABLE - OIDC PR preupload returned invalid JSON") from None

    files = result.get("files") if isinstance(result, dict) else None
    require(isinstance(files, list) and len(files) == 1 and isinstance(files[0], dict) and
            files[0].get("path") == PATH and files[0].get("uploadMode") in ("regular", "lfs"),
            "UNAVAILABLE - OIDC PR preupload returned unexpected file metadata")
    print("OIDC PR preupload negotiation accepted; no file was uploaded or committed")
    return "OIDC_PR_PREUPLOAD_ALLOWED"


if __name__ == "__main__":
    probe()

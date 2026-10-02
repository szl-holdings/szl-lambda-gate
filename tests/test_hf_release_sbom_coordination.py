# SPDX-License-Identifier: Apache-2.0
"""Exact release producer coordination without network calls or real waits."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("wait_release_sbom", ROOT / "scripts" / "wait_release_sbom.py")
assert spec and spec.loader
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)

REPO = "szl-holdings/szl-lambda-gate"
TAG = "v0.2.1"
SOURCE = "a" * 40
PUBLISHED = "2026-10-02T01:00:00Z"
REQUIRED = ["szl-lambda-gate-sbom-2.spdx.json", "szl-lambda-gate-sbom.cyclonedx.json"]


def producer(**changes):
    run = {
        "id": 123, "run_attempt": 1, "repository": {"full_name": REPO},
        "head_repository": {"full_name": REPO}, "head_branch": TAG,
        "head_sha": SOURCE, "event": "release", "path": ".github/workflows/sbom.yml",
        "created_at": "2026-10-02T01:00:01Z", "status": "completed", "conclusion": "success",
    }
    run.update(changes)
    return run


def release():
    return {
        "tag_name": TAG, "draft": False, "published_at": PUBLISHED,
        "assets": [{"id": i + 1, "name": name, "size": 80 + i, "state": "uploaded",
                    "digest": "sha256:" + str(i + 1) * 64} for i, name in enumerate(REQUIRED)],
    }


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class FakeAPI:
    def __init__(self, runs, asset_release=None):
        self.runs = runs
        self.asset_release = asset_release or release()
        self.run_reads = 0
        self.calls = []

    def __call__(self, endpoint, timeout):
        self.calls.append((endpoint, timeout))
        if "/actions/workflows/sbom.yml/runs?" in endpoint:
            listing = self.runs[min(self.run_reads, len(self.runs) - 1)]
            self.run_reads += 1
            return {"workflow_runs": deepcopy(listing)}
        assert endpoint == f"repos/{REPO}/releases/tags/{TAG}"
        return deepcopy(self.asset_release)


def wait(api, clock=None, timeout=30):
    clock = clock or FakeClock()
    return gate.wait_for_sbom(repo=REPO, tag=TAG, source_sha=SOURCE, published_at=PUBLISHED,
                              required=REQUIRED, timeout_seconds=timeout, poll_seconds=10,
                              fetch=api, monotonic=clock.monotonic, sleep=clock.sleep)


def asset_reads(api):
    return [endpoint for endpoint, _ in api.calls if "/releases/tags/" in endpoint]


def test_queued_producer_and_asset_existence_do_not_allow_early_dispatch():
    api = FakeAPI([[producer(status="queued", conclusion=None)],
                   [producer(status="in_progress", conclusion=None)], [producer()]])
    clock = FakeClock()
    result = wait(api, clock)
    assert clock.sleeps == [10, 10]
    assert ["/releases/tags/" in endpoint for endpoint, _ in api.calls] == [False, False, False, True, False]
    assert result["sbom_run_id"] == 123 and result["github_sha"] == SOURCE
    assert sorted(result["assets"]) == REQUIRED


@pytest.mark.parametrize("conclusion", ["failure", "cancelled", "timed_out", "skipped", None])
def test_failed_completed_producer_fails_without_reading_assets(conclusion):
    api = FakeAPI([[producer(conclusion=conclusion)]])
    with pytest.raises(gate.CoordinationError, match="did not succeed"):
        wait(api)
    assert asset_reads(api) == []


@pytest.mark.parametrize("changes", [
    {"head_sha": "b" * 40}, {"repository": {"full_name": "other/repo"}},
    {"head_repository": {"full_name": "other/repo"}},
])
def test_same_release_candidate_with_different_identity_fails_closed(changes):
    api = FakeAPI([[producer(**changes)]])
    with pytest.raises(gate.CoordinationError, match="mismatch"):
        wait(api)
    assert asset_reads(api) == []


@pytest.mark.parametrize("changes", [
    {"created_at": "2026-10-02T00:59:59Z"}, {"head_branch": "v0.2.0"},
    {"event": "push"}, {"path": ".github/workflows/unrelated.yml"},
])
def test_old_or_unrelated_run_cannot_satisfy_gate(changes):
    clock = FakeClock()
    api = FakeAPI([[producer(**changes)]])
    with pytest.raises(gate.CoordinationError, match="timed out"):
        wait(api, clock, timeout=25)
    assert clock.sleeps == [10, 10, 5]
    assert asset_reads(api) == []
    assert all(timeout <= 25 for _, timeout in api.calls)


def test_newest_producer_must_finish_even_if_earlier_run_succeeded():
    latest = producer(id=124, created_at="2026-10-02T01:00:02Z", status="queued", conclusion=None)
    api = FakeAPI([[producer(), latest], [producer(), producer(id=124, created_at=latest["created_at"])]])
    clock = FakeClock()
    assert wait(api, clock)["sbom_run_id"] == 124
    assert clock.sleeps == [10]


@pytest.mark.parametrize("changed", [
    producer(run_attempt=2), producer(status="in_progress", conclusion=None),
    producer(id=124, created_at="2026-10-02T01:00:02Z"),
])
def test_producer_rerun_or_new_run_during_asset_snapshot_fails_closed(changed):
    api = FakeAPI([[producer()], [changed]])
    with pytest.raises(gate.CoordinationError, match="changed during"):
        wait(api)
    assert len(asset_reads(api)) == 1


@pytest.mark.parametrize("changes", [
    {"tag_name": "v0.2.0"}, {"draft": True}, {"published_at": "2026-10-02T01:00:01Z"},
])
def test_release_snapshot_identity_must_match_event(changes):
    item = release()
    item.update(changes)
    with pytest.raises(gate.CoordinationError, match="release identity mismatch"):
        wait(FakeAPI([[producer()]], item))


@pytest.mark.parametrize("changes", [
    {"size": 0}, {"size": -1}, {"size": True}, {"digest": None},
    {"digest": "sha512:" + "a" * 64}, {"digest": "sha256:" + "a" * 63},
    {"state": "starter"},
])
def test_required_assets_must_have_uploaded_nonempty_sha256_metadata(changes):
    item = release()
    item["assets"][0].update(changes)
    with pytest.raises(gate.CoordinationError, match="lacks uploaded bytes"):
        wait(FakeAPI([[producer()]], item))


def test_required_assets_missing_or_duplicated_fail_closed():
    for assets in (release()["assets"][1:], release()["assets"] + [release()["assets"][0]]):
        item = release()
        item["assets"] = assets
        with pytest.raises(gate.CoordinationError, match="missing or duplicated"):
            wait(FakeAPI([[producer()]], item))


@pytest.mark.parametrize("names", [None, [], ["../asset"], ["dir/asset"], ["dir\\asset"], ["."],
                                      ["asset:name"], ["asset\x00name"], ["asset\nname"], ["asset\x7fname"],
                                      ["asset", "asset"], [42]])
def test_config_requires_explicit_safe_unique_asset_names(names):
    with pytest.raises(gate.CoordinationError):
        gate.required_asset_names({"targets": [{"required_release_assets": names}]})


def test_required_asset_union_is_deterministic():
    assert gate.required_asset_names({"targets": [
        {"required_release_assets": ["b", "a"]}, {"required_release_assets": ["c", "a"]},
    ]}) == ["a", "b", "c"]


@pytest.mark.parametrize("bounds", [(float("inf"), 10), (30, float("nan")), (0, 10), (30, -1)])
def test_wait_bounds_are_finite_and_positive(bounds):
    with pytest.raises(gate.CoordinationError, match="wait bounds"):
        gate.wait_for_sbom(repo=REPO, tag=TAG, source_sha=SOURCE, published_at=PUBLISHED,
                          required=REQUIRED, timeout_seconds=bounds[0], poll_seconds=bounds[1])


def test_gh_runner_uses_argv_and_does_not_forward_hub_credentials(monkeypatch):
    monkeypatch.setenv("GH_TOKEN", "test-github-token")
    monkeypatch.setenv("HF_TOKEN", "test-hub-token")
    monkeypatch.setenv("HF_FALLBACK_TOKEN", "test-fallback")
    monkeypatch.setenv("HF_OIDC_RESOURCE", "test/resource")

    def run(argv, **kwargs):
        assert argv == ["gh", "api", "repos/example/repo/releases"]
        assert kwargs["shell"] is False and kwargs["timeout"] == 4
        assert kwargs["env"]["GH_TOKEN"] == "test-github-token"
        assert not {"HF_TOKEN", "HF_FALLBACK_TOKEN", "HF_OIDC_RESOURCE"} & kwargs["env"].keys()
        return SimpleNamespace(returncode=0, stdout='{"ok": true}')

    monkeypatch.setattr(gate.subprocess, "run", run)
    assert gate.gh_json("repos/example/repo/releases", 4) == {"ok": True}


def test_cli_failure_never_echoes_credential_bearing_stderr(monkeypatch):
    monkeypatch.setattr(gate.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        returncode=1, stdout="", stderr="sensitive credential"))
    with pytest.raises(gate.CoordinationError, match="^GitHub API request failed$"):
        gate.gh_json("repos/example/repo/releases", 4)


def test_cli_timeout_fails_closed(monkeypatch):
    def run(*args, **kwargs):
        raise subprocess.TimeoutExpired(["gh", "api"], 4, stderr="sensitive credential")

    monkeypatch.setattr(gate.subprocess, "run", run)
    with pytest.raises(gate.CoordinationError, match="^GitHub API request failed or timed out$"):
        gate.gh_json("repos/example/repo/releases", 4)


def test_release_dispatch_waits_before_publisher_and_keeps_main_oidc_identity():
    text = (ROOT / ".github" / "workflows" / "hf-mirror.yml").read_text(encoding="utf-8")
    dispatch = text[text.index("  release-dispatch:"):text.index("  plan:")]
    assert "SOURCE_RELEASE_SHA: ${{ github.sha }}" in dispatch
    assert "RELEASE_PUBLISHED_AT: ${{ github.event.release.published_at }}" in dispatch
    assert "ref: ${{ github.event.repository.default_branch }}" in dispatch
    assert dispatch.index("actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1") < dispatch.index("python3 scripts/wait_release_sbom.py")
    assert dispatch.index("python3 scripts/wait_release_sbom.py") < dispatch.index("Dispatch reviewed Hub proposal on main")
    assert '{ref: $ref, inputs: {tag: $tag, auth: "oidc"}}' in dispatch
    assert "--timeout-seconds 900" in dispatch

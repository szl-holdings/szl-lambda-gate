# SPDX-License-Identifier: Apache-2.0
"""Hardening contract for the release-only Hugging Face mirror.

Standard library and pytest only: the ci workflow installs torch and pytest,
not huggingface_hub or PyYAML. preflight() is exercised against an in-memory
stand-in for huggingface_hub, so no test touches the network.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "hf-mirror.yml"
CONFIG = ROOT / ".github" / "hf-mirror.json"
PUBLISHER = ROOT / "scripts" / "hf_mirror_release.py"
REPO_ID = "SZLHOLDINGS/szl-lambda-gate"
HUB_MAIN = "a" * 40

spec = importlib.util.spec_from_file_location("hf_mirror_release_hardening", PUBLISHER)
assert spec and spec.loader
mirror = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mirror)


def workflow() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def run_blocks(text: str) -> list[str]:
    """Return the shell body of every `run:` key in a workflow file."""
    lines = text.splitlines()
    blocks: list[str] = []
    for index, line in enumerate(lines):
        match = re.match(r"^(\s*)(?:- )?run:\s*(.*)$", line)
        if not match:
            continue
        indent, value = len(match.group(1)), match.group(2).strip()
        if value not in ("|", ">", "|-", ">-"):
            blocks.append(value)
            continue
        body = []
        for follow in lines[index + 1:]:
            if follow.strip() and len(follow) - len(follow.lstrip()) <= indent:
                break
            body.append(follow)
        blocks.append("\n".join(body))
    return blocks


def job_steps(text: str, job: str) -> list[str]:
    """Split one job's `steps:` list into raw step texts (fixed 6-space step indent)."""
    lines = text.splitlines()
    start = lines.index(f"  {job}:")
    end = next((i for i in range(start + 1, len(lines)) if re.match(r"^  \S", lines[i])), len(lines))
    body = lines[start:end]
    steps, current = [], None
    for line in body[body.index("    steps:") + 1:]:
        if line.startswith("      - "):
            if current is not None:
                steps.append("\n".join(current))
            current = [line]
        elif current is not None:
            current.append(line)
    if current is not None:
        steps.append("\n".join(current))
    return steps


def step(text: str, job: str, marker: str) -> str:
    matches = [item for item in job_steps(text, job) if marker in item]
    assert len(matches) == 1, marker
    return matches[0]


# --- (a) reviewed replacements -------------------------------------------------


def test_lambda_gate_default_replace_list_is_empty() -> None:
    targets = json.loads(CONFIG.read_text(encoding="utf-8"))["targets"]
    assert len(targets) == 1
    item = targets[0]
    assert item["hf_repo_id"] == REPO_ID
    assert item["replace_hub_paths"] == []
    with pytest.raises(RuntimeError, match="collision differs: build.toml"):
        mirror.collision_plan({"build.toml": "new"}, {"build.toml": "old"}, item)


def test_collision_plan_allows_identical_new_and_rendered_card() -> None:
    item = {"replace_hub_paths": [], "preserve_hub_paths": []}
    staged = {"README.md": "a", "same.py": "b", "new.py": "c"}
    hub = {"README.md": "z", "same.py": "b"}
    assert mirror.collision_plan(staged, hub, item) == {}
    assert mirror.collision_plan(staged, hub, {}) == {}


def test_collision_plan_fails_closed_on_unlisted_difference() -> None:
    item = {"replace_hub_paths": ["build.toml"], "preserve_hub_paths": []}
    with pytest.raises(RuntimeError, match="collision differs: lambda_gate.py"):
        mirror.collision_plan({"lambda_gate.py": "new"}, {"lambda_gate.py": "old"}, item)


def test_collision_plan_records_listed_replacement_only() -> None:
    item = {"replace_hub_paths": ["build.toml", "pyproject.toml"], "preserve_hub_paths": ["LICENSE"]}
    staged = {"build.toml": "new", "pyproject.toml": "same", "LICENSE": "x"}
    hub = {"build.toml": "old", "pyproject.toml": "same"}
    assert mirror.collision_plan(staged, hub, item) == {"build.toml": {"hub_before": "old", "source": "new"}}


@pytest.mark.parametrize(
    ("replace", "preserve", "message"),
    [
        (["README.md"], [], "never replaced"),
        (["LICENSE"], ["LICENSE"], "both preserved and replaceable"),
        (["build.toml", "build.toml"], [], "duplicate"),
        (["../escape"], [], "unsafe path"),
        (["/abs"], [], "unsafe path"),
        (["dir\\file"], [], "unsafe path"),
    ],
)
def test_collision_plan_rejects_bad_config(replace: list[str], preserve: list[str], message: str) -> None:
    item = {"replace_hub_paths": replace, "preserve_hub_paths": preserve}
    with pytest.raises(RuntimeError, match=message):
        mirror.collision_plan({}, {}, item)


def fake_hub(monkeypatch: pytest.MonkeyPatch, store: Path, hub_files: dict[str, bytes]) -> None:
    for name, data in hub_files.items():
        (store / name).parent.mkdir(parents=True, exist_ok=True)
        (store / name).write_bytes(data)
    module = types.ModuleType("huggingface_hub")

    class HfApi:
        def __init__(self, token: str | None = None) -> None:
            assert token == "test-token"

        def repo_info(self, repo: str, repo_type: str, revision: str) -> types.SimpleNamespace:
            assert (repo, repo_type, revision) == (REPO_ID, "model", "main")
            return types.SimpleNamespace(sha=HUB_MAIN)

        def list_repo_files(self, repo: str, repo_type: str, revision: str) -> list[str]:
            assert revision == HUB_MAIN
            return sorted(hub_files)

    def hf_hub_download(repo: str, name: str, repo_type: str, revision: str, token: str) -> str:
        assert revision == HUB_MAIN and token == "test-token"
        return str(store / name)

    class ModelCard:
        def __init__(self, text: str) -> None:
            self.text = text
            self.data = types.SimpleNamespace(to_dict=lambda: {"library_name": "kernels", "license": "apache-2.0"})

        @classmethod
        def load(cls, path: Path) -> "ModelCard":
            return cls(Path(path).read_text(encoding="utf-8"))

    module.HfApi, module.hf_hub_download, module.ModelCard = HfApi, hf_hub_download, ModelCard
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)


def preflight_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replace: list[str]) -> None:
    store = tmp_path / "hub"
    fake_hub(monkeypatch, store, {
        "README.md": b"curated card\n", "LICENSE": b"hub notice\n", "build.toml": b"[general]\nold\n",
        "lambda_gate.py": b"hub only\n", "pyproject.toml": b"same\n",
    })
    work = tmp_path / "work"
    (work / ".github").mkdir(parents=True)
    (work / ".github" / "hf-mirror.json").write_text(json.dumps({"targets": [{
        "hf_repo_id": REPO_ID, "repo_type": "model", "preserve_hub_card": True,
        "required_card_metadata": {"library_name": "kernels", "license": "apache-2.0"},
        "required_card_tags": [], "preserve_hub_paths": ["LICENSE"], "replace_hub_paths": replace,
        "hub_only_paths": ["lambda_gate.py"],
    }]}), encoding="utf-8")
    stage = work / ".hfstage"
    stage.mkdir()
    (stage / "README.md").write_bytes(b"source readme\n")
    (stage / "build.toml").write_bytes(b"[general]\nnew\n")
    (stage / "pyproject.toml").write_bytes(b"same\n")
    (stage / "torch-ext").mkdir()
    (stage / "torch-ext" / "new.py").write_bytes(b"new file\n")
    monkeypatch.chdir(work)
    for key, value in {"HF_REPO_ID": REPO_ID, "HF_REPO_TYPE": "model", "HF_TOKEN": "test-token"}.items():
        monkeypatch.setenv(key, value)


def test_preflight_keeps_strict_rule_when_replace_list_is_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    preflight_workspace(tmp_path, monkeypatch, replace=[])
    with pytest.raises(RuntimeError, match="collision differs: build.toml"):
        mirror.preflight()
    assert not mirror.BASELINE_FILE.exists()


def test_preflight_records_reviewed_replacement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                capsys: pytest.CaptureFixture[str]) -> None:
    preflight_workspace(tmp_path, monkeypatch, replace=["build.toml"])
    mirror.preflight()
    evidence = json.loads(mirror.BASELINE_FILE.read_text(encoding="utf-8"))
    old, new = mirror.sha256(tmp_path / "hub" / "build.toml"), mirror.sha256(mirror.STAGE / "build.toml")
    assert evidence["sha"] == HUB_MAIN
    assert evidence["replacements"] == {"build.toml": {"hub_before": old, "source": new}}
    assert evidence["hashes"]["build.toml"] == old
    assert f"reviewed replacement build.toml: hub {old} -> source {new}" in capsys.readouterr().out


def test_receipt_lists_replaced_files() -> None:
    publisher = PUBLISHER.read_text(encoding="utf-8")
    assert '"replaced_hub_files": sorted(base.get("replacements", {}))' in publisher
    assert "expected_hashes.update(stage_hashes)" in publisher


# --- (b) destructive push lane removed ----------------------------------------


def test_hub_sync_lane_and_its_secret_binding_are_gone() -> None:
    text = workflow()
    assert "hub-sync@" not in text
    assert "\n  sync:" not in text
    assert "push_targets" not in text
    assert "hf_token:" not in text
    assert text.count("secrets.HF_TOKEN") == 1
    assert "HF_FALLBACK_TOKEN: ${{ secrets.HF_TOKEN }}" in step(text, "release-mirror", "Resolve Hub credential (PAT)")


def test_plan_fails_closed_if_a_target_re_enables_push_mirroring() -> None:
    load = step(workflow(), "plan", "id: load")
    assert 'select(has("mirror_on_push") and .mirror_on_push != false)' in load
    assert "destructive lane removed" in load
    assert load.index("destructive lane removed") < load.index('--arg only "$ONLY"')
    for item in json.loads(CONFIG.read_text(encoding="utf-8"))["targets"]:
        assert item.get("mirror_on_push", False) is False


def test_trusted_publisher_identity_is_unchanged() -> None:
    text = workflow()
    assert WORKFLOW.name == "hf-mirror.yml"
    assert text.startswith("name: hf-mirror\n")
    triggers = text[text.index("\non:\n"):text.index("\npermissions:\n")]
    assert "  push:\n    branches: [main]\n" in triggers
    assert "  release:\n    types: [published]\n" in triggers
    for name in ("tag:", "only:", "auth:"):
        assert f"      {name}\n" in triggers
    assert "\npermissions:\n  contents: read\n" in text
    assert "\n  release-mirror:\n    needs: plan\n" in text
    assert "actions/workflows/hf-mirror.yml/dispatches" in text
    mirror_job = text[text.index("\n  release-mirror:\n"):]
    assert "    permissions:\n      contents: read\n      id-token: write\n" in mirror_job
    assert "if: github.event_name == 'workflow_dispatch' && inputs.tag != '' && github.ref == format(" in mirror_job


# --- (c) token exposure -------------------------------------------------------


@pytest.mark.parametrize("marker", [
    "name: Stage payload",
    "name: Attach release assets",
    "name: Render and validate model card",
    "uses: actions/upload-artifact@",
])
def test_steps_without_hub_calls_blank_the_token(marker: str) -> None:
    assert "\n          HF_TOKEN: ''\n" in step(workflow(), "release-mirror", marker) + "\n"


@pytest.mark.parametrize("marker", [
    "name: Capture Hub baseline and check preservation",
    "name: Publish, bind tag, verify bytes, and emit receipt",
])
def test_hub_steps_keep_the_resolved_token(marker: str) -> None:
    assert "HF_TOKEN" not in step(workflow(), "release-mirror", marker)


def test_first_tokenless_step_fails_closed_if_blanking_regresses() -> None:
    stage = step(workflow(), "release-mirror", "name: Stage payload")
    assert 'if [ -n "${HF_TOKEN:-}" ]; then' in stage
    names = [re.search(r"name: (.+)", item).group(1) for item in job_steps(workflow(), "release-mirror")
             if re.search(r"name: (.+)", item)]
    assert names.index("Stage payload") > names.index("Resolve Hub credential (OIDC)")


# --- (d) dependency pin and general workflow hygiene --------------------------


def test_huggingface_hub_is_pinned_to_the_proven_version() -> None:
    text = workflow()
    assert 'python -m pip install --upgrade "huggingface_hub[cli,hf_xet]==2.0.0" pyyaml' in text
    assert ">=1.32.0" not in text
    assert len(re.findall(r"huggingface_hub\[", text)) == 1


def test_workflow_run_scripts_never_interpolate_expressions() -> None:
    blocks = run_blocks(workflow())
    assert len(blocks) >= 10
    for block in blocks:
        assert "${{" not in block, block


def test_workflow_actions_keep_their_reviewed_sha_pins() -> None:
    uses = re.findall(r"uses:\s*(\S+)", workflow())
    assert sorted(set(uses)) == [
        "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1",
        "actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97",
        "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a",
    ]

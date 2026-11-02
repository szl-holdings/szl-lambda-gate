"""Guard the governed-norm tests against falling out of default pytest discovery."""

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GOVERNED_NORM_MODULES = {
    "tests/test_governed_norm.py",
    "tests/test_governed_norm_correctness.py",
    "tests/test_governed_norm_receipts.py",
}


def test_default_collection_includes_governed_norm_modules():
    """Collect as CI does, without naming the modules explicitly."""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "torch-ext"), str(ROOT / "tests"), env.get("PYTHONPATH", "")]
    )
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    collected_modules = {
        line.split("::", 1)[0].replace("\\", "/")
        for line in result.stdout.splitlines()
        if "::" in line
    }
    assert GOVERNED_NORM_MODULES <= collected_modules, (
        f"Not collected by default: {sorted(GOVERNED_NORM_MODULES - collected_modules)}"
    )

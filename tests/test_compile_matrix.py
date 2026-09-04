import json
from pathlib import Path

MATRIX = Path(__file__).resolve().parents[1] / "frontier" / "compile_matrix.json"


def test_compile_matrix_does_not_claim_acceleration() -> None:
    data = json.loads(MATRIX.read_text(encoding="utf-8"))
    assert data["schema"] == "szl.kernel-compile-matrix/v1"
    assert data["acceleration_claim"] is False
    assert data["hub_write"] == "DENIED_IN_THIS_CHANGE"
    by_id = {row["id"]: row for row in data["surfaces"]}
    assert by_id["torch.compile-fullgraph"]["status"] == "UNSUPPORTED_UNTIL_REPUBLISH"
    assert by_id["cuda-performance"]["status"] == "UNSUPPORTED"
    assert by_id["cpu-eager"]["status"] == "SUPPORTED"

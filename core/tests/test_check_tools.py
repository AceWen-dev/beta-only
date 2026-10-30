from __future__ import annotations

import io
import json
import os
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from core.tools import audit_data, check

ROOT = Path(__file__).resolve().parents[2]

try:
    import torch  # noqa: F401

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def run_main(module, argv: list[str]) -> tuple[int, str]:
    output = io.StringIO()
    previous = sys.argv
    sys.argv = [module.__name__, *argv]
    try:
        with redirect_stdout(output):
            code = module.main()
    finally:
        sys.argv = previous
    return code, output.getvalue()


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
class CheckToolTests(unittest.TestCase):
    def test_project_passes_initialization_gate(self) -> None:
        if not os.environ.get("DLEF_DATA_ROOT"):
            self.skipTest("DLEF_DATA_ROOT is not set")
        code, output = run_main(
            check,
            [
                "--project",
                str(ROOT / "PROJECT.yaml"),
                "--config",
                str(ROOT / "input" / "configs" / "experiments" / "v11" / "baseline.yaml"),
                "--json",
            ],
        )
        payload = json.loads(output[output.index("{"):])
        self.assertEqual(code, 0, payload["errors"])
        self.assertTrue(payload["ok"])


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
class AuditToolTests(unittest.TestCase):
    def test_synthetic_template_audit_runs(self) -> None:
        code, output = run_main(
            audit_data,
            [
                "--config",
                str(ROOT / "input" / "configs" / "base.yaml"),
                "--project",
                str(ROOT / "PROJECT.yaml"),
                "--limit",
                "1",
                "--json",
            ],
        )
        payload = json.loads(output)
        self.assertEqual(code, 0)
        self.assertTrue(payload["ok"])
        self.assertEqual(set(payload["splits"]), {"train", "val", "test"})


if __name__ == "__main__":
    unittest.main()

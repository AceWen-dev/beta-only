from __future__ import annotations

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from dlef.benchmark import BenchmarkSpec
from dlef.config import load_config

from core.config import ProjectConfig
from core.tools import new_benchmark, new_experiment, status

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


class NewExperimentTests(unittest.TestCase):
    def test_creates_valid_variant(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "demo.yaml"
            code, output = run_main(
                new_experiment,
                [
                    "--name",
                    "demo",
                    "--experiment",
                    "baseline",
                    "--out",
                    str(out),
                    "--set",
                    "optim.lr=0.005",
                ],
            )
            self.assertEqual(code, 0)
            self.assertIn(str(out), output)
            config = load_config(out, ProjectConfig)
            self.assertEqual(config.schema.experiment, "baseline-demo")
            self.assertEqual(config.schema.optim.lr, 0.005)


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
class NewBenchmarkTests(unittest.TestCase):
    def test_creates_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "demo-test-v1.json"
            code, output = run_main(
                new_benchmark,
                [
                    "--name",
                    "demo-test-v1",
                    "--config",
                    str(ROOT / "input" / "configs" / "base.yaml"),
                    "--out",
                    str(out),
                ],
            )
            self.assertEqual(code, 0)
            self.assertIn("demo-test-v1", output)
            spec = BenchmarkSpec.load(out)
            self.assertEqual(spec.name, "demo-test-v1")


class StatusTests(unittest.TestCase):
    def test_status_is_readable(self) -> None:
        code, output = run_main(status, [])
        self.assertEqual(code, 0)
        self.assertIn("project:", output)
        self.assertIn("phase:", output)
        self.assertIn("runs:", output)

    def test_handoff_adds_checklist(self) -> None:
        code, output = run_main(status, ["--handoff"])
        self.assertEqual(code, 0)
        self.assertIn("handoff checklist:", output)


if __name__ == "__main__":
    unittest.main()

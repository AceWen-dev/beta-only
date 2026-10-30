from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class ProjectMemoryTests(unittest.TestCase):
    def test_state_and_decisions_exist(self) -> None:
        for name in ("STATE.md", "DECISIONS.md"):
            path = ROOT / name
            self.assertTrue(path.is_file(), f"{name} is missing")
            text = path.read_text(encoding="utf-8")
            self.assertIn("##", text, f"{name} has no sections")
        self.assertTrue((ROOT / "PROJECT.yaml").is_file(), "PROJECT.yaml is missing")

    def test_agent_guide_points_to_project_memory(self) -> None:
        text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
        self.assertIn("PROJECT.yaml", text)
        self.assertIn("STATE.md", text)
        self.assertIn("DECISIONS.md", text)
        self.assertIn("Reading order", text)
        self.assertIn("0.2", text)


if __name__ == "__main__":
    unittest.main()

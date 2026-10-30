from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE_COMMANDS = {"train.py", "eval.py", "sweep.py"}


class CommandSurfaceTests(unittest.TestCase):
    def test_commands_only_contains_core_entries(self) -> None:
        actual = {
            path.name
            for path in (ROOT / "commands").glob("*.py")
            if path.is_file() and path.name != "__init__.py"
        }
        self.assertEqual(
            actual,
            CORE_COMMANDS,
            "core/commands/ is for train/eval/sweep only; put project scripts under scripts/",
        )


if __name__ == "__main__":
    unittest.main()

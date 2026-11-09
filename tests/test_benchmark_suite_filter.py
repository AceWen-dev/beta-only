from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

try:
    sys.path.insert(0, str(ROOT / "scripts"))
    from plot_experiment import filter_rows
    sys.path.pop(0)
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False


class BenchmarkSuiteFilterTests(unittest.TestCase):
    def test_r_member_excludes_1_2mm(self) -> None:
        for name in ("benchmark2.json", "benchmark2-big.json", "benchmark2-endo.json"):
            suite = json.loads((ROOT / "input" / "benchmarks" / name).read_text())
            self.assertEqual(
                suite["members"]["r"].get("min_truth_mm"), 2.0, name
            )

    def test_b_member_excludes_7_8mm(self) -> None:
        for name in ("benchmark2.json", "benchmark2-big.json", "benchmark2-endo.json"):
            suite = json.loads((ROOT / "input" / "benchmarks" / name).read_text())
            self.assertEqual(
                suite["members"]["b"].get("exclude_bins"), [7], name
            )

    @unittest.skipUnless(PLOT_AVAILABLE, "plot_experiment dependencies missing")
    def test_filter_rows_drops_1_2mm(self) -> None:
        rows = [
            {"split": "test_r", "truth_mm": 1.5},
            {"split": "test_r", "truth_mm": 2.0},
            {"split": "test_r", "truth_mm": 2.5},
            {"split": "test_b", "truth_mm": 1.5},
        ]
        kept = filter_rows(rows, "test_r", 2.0)
        self.assertEqual([r["truth_mm"] for r in kept], [2.0, 2.5])
        unfiltered = filter_rows(rows, "test_r")
        self.assertEqual([r["truth_mm"] for r in unfiltered], [1.5, 2.0, 2.5])

    @unittest.skipUnless(PLOT_AVAILABLE, "plot_experiment dependencies missing")
    def test_filter_rows_drops_7_8mm(self) -> None:
        rows = [
            {"split": "test_b", "truth_mm": 7.2},
            {"split": "test_b", "truth_mm": 7.48},
            {"split": "test_b", "truth_mm": 8.0},
            {"split": "test_b", "truth_mm": 8.5},
            {"split": "test_r", "truth_mm": 7.5},
        ]
        kept = filter_rows(rows, "test_b", exclude_bins=[7])
        self.assertEqual([r["truth_mm"] for r in kept], [8.0, 8.5])
        unfiltered = filter_rows(rows, "test_b")
        self.assertEqual(
            [r["truth_mm"] for r in unfiltered], [7.2, 7.48, 8.0, 8.5]
        )

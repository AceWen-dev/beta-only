from __future__ import annotations

import unittest
from types import SimpleNamespace

try:
    import torch

    from core.metrics.v11 import V11Metric

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
class V11MetricTests(unittest.TestCase):
    def test_case_weighting_and_min_cases(self) -> None:
        # 8 cases in 2–3mm (error 1.0) and 4 cases in 9–10mm (error 3.0).
        # robust worst must ignore 9–10 because it has < 8 cases.
        preds = [3.5] * 8 + [12.5] * 4
        targets = [2.5] * 8 + [9.5] * 4
        cases = [f"c{i}" for i in range(8)] + [f"d{i}" for i in range(4)]
        metric = V11Metric()
        metric.update(
            SimpleNamespace(predictions=torch.tensor(preds)),
            {"y": torch.tensor(targets), "case_id": cases},
        )
        values = metric.compute().values
        self.assertAlmostEqual(values["case_mae"], (8 * 1.0 + 4 * 3.0) / 12)
        self.assertAlmostEqual(values["case_mae_lt10"], (8 * 1.0 + 4 * 3.0) / 12)
        self.assertAlmostEqual(values["robust_worst_bin"], 1.0)
        self.assertEqual(values["robust_bin_count"], 1.0)

    def test_robust_bin_included_when_enough_cases(self) -> None:
        preds = [2.5] * 8 + [12.0] * 8
        targets = [2.5] * 8 + [9.5] * 8
        cases = [f"a{i}" for i in range(8)] + [f"b{i}" for i in range(8)]
        metric = V11Metric()
        metric.update(
            SimpleNamespace(predictions=torch.tensor(preds)),
            {"y": torch.tensor(targets), "case_id": cases},
        )
        values = metric.compute().values
        self.assertAlmostEqual(values["robust_worst_bin"], 2.5)
        self.assertEqual(values["robust_bin_count"], 2.0)


if __name__ == "__main__":
    unittest.main()

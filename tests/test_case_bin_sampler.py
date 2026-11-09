from __future__ import annotations

import unittest

from core.data.samplers import bin_of, case_bin_weights


def _row(case: str, truth: float) -> dict:
    return {"case_id": case, "truth_mm": truth}


class CaseBinSamplerTest(unittest.TestCase):
    def test_bin_boundary(self) -> None:
        self.assertEqual(bin_of(9.999), 9)
        self.assertEqual(bin_of(10.0), 10)
        self.assertEqual(bin_of(2.0), 2)

    def test_rare_bin_upweighted_and_case_normalised(self) -> None:
        rows = (
            [_row("company-rare", 2.5)] * 4
            + [_row("company-a", 4.2)] * 4
            + [_row("company-b", 4.4)] * 4
            + [_row("company-c", 5.1)] * 4
            + [_row("company-d", 5.6)] * 4
        )
        weights = case_bin_weights(rows)
        rare = weights[:4]
        common = weights[4:8]
        # 每例内部等权（除以帧数），稀有档整例权重是常见档的 base/count = 2/1 倍
        self.assertAlmostEqual(float(rare[0]), float(rare[1]))
        self.assertAlmostEqual(float(common[0]), float(common[1]))
        self.assertAlmostEqual(float(rare[0]) / float(common[0]), 2.0)
        # 每例的帧权重之和 = 该例的档位权重
        self.assertAlmostEqual(float(rare.sum()), 2.0)
        self.assertAlmostEqual(float(common.sum()), 1.0)

    def test_non_company_case_keeps_neutral_weight(self) -> None:
        rows = [_row("phantom-1", 2.5)] * 2 + [_row("company-a", 4.2)] * 4
        weights = case_bin_weights(rows)
        self.assertAlmostEqual(float(weights[0]), 0.5)
        self.assertAlmostEqual(float(weights[-1]), 0.25)


if __name__ == "__main__":
    unittest.main()

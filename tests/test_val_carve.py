"""carve_val_cases 的黄金测试：确定性、case 级、只取 4–6mm、无泄漏。"""

from __future__ import annotations

import unittest

from core.data.datasets.v11_frames import carve_val_cases


def make_rows() -> list[dict]:
    rows = []
    # 60 个 4-6mm train 病例（可被抽为 val）+ 稀缺档病例 + 1 个 val 病例
    for i in range(60):
        rows.append({
            "split": "train",
            "case_id": f"company-{1000 + i}",
            "truth_mm": 4.0 + (i % 20) * 0.1,
        })
    for i in range(10):  # 稀缺档：不应被抽走
        rows.append({"split": "train", "case_id": f"company-{2000 + i}", "truth_mm": 2.4})
    rows.append({"split": "val", "case_id": "company-3000", "truth_mm": 5.0})
    rows.append({"split": "test_b", "case_id": "polyp-size-01", "truth_mm": 7.0})
    return rows


class ValCarveTests(unittest.TestCase):
    def test_deterministic_and_case_level(self) -> None:
        rows = make_rows()
        a = carve_val_cases(rows, 10)
        b = carve_val_cases(rows, 10)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 10)
        self.assertTrue(all(c.startswith("company-1") for c in a))

    def test_only_well_stocked_bins_are_taken(self) -> None:
        chosen = carve_val_cases(make_rows(), 10)
        truth_by_case = {}
        for r in make_rows():
            if r["split"] == "train":
                truth_by_case.setdefault(r["case_id"], []).append(r["truth_mm"])
        for c in chosen:
            med = sorted(truth_by_case[c])[len(truth_by_case[c]) // 2]
            self.assertGreaterEqual(med, 4.0)
            self.assertLess(med, 6.0)
        self.assertNotIn("company-2000", chosen)  # 稀缺档不被抽走

    def test_never_touches_val_or_test_cases(self) -> None:
        chosen = carve_val_cases(make_rows(), 60)
        self.assertNotIn("company-3000", chosen)
        self.assertNotIn("polyp-size-01", chosen)

    def test_n_larger_than_eligible_takes_all(self) -> None:
        self.assertEqual(len(carve_val_cases(make_rows(), 500)), 60)


if __name__ == "__main__":
    unittest.main()

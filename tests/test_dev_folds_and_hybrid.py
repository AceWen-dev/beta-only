"""dev 三折分配与混合损失的黄金测试（D-0022）。"""

from __future__ import annotations

import collections
import unittest

import torch

from core.data.datasets.v11_frames import assign_dev_folds
from core.losses.mm_case_bin import mm_case_bin


def make_rows() -> list[dict]:
    rows = []
    # company 小档 / company 大档 / newdata 小档 三个层，各 9 个病例
    for i in range(9):
        rows.append({"split": "train", "case_id": f"company-{100 + i}", "truth_mm": 2.0 + i * 0.05})
        rows.append({"split": "train", "case_id": f"company-{200 + i}", "truth_mm": 8.0 + i * 0.05})
        rows.append({"split": "val", "case_id": f"newdata-{300 + i}", "truth_mm": 2.5 + i * 0.05})
    return rows


class DevFoldTests(unittest.TestCase):
    def test_deterministic_and_balanced(self) -> None:
        rows = make_rows()
        a = assign_dev_folds(rows, 3)
        b = assign_dev_folds(rows, 3)
        self.assertEqual(a, b)
        folds = collections.Counter(a.values())
        self.assertEqual(set(folds), {0, 1, 2})
        self.assertEqual(sorted(folds.values()), [9, 9, 9])  # 27 病例均分

    def test_every_fold_sees_every_stratum(self) -> None:
        rows = make_rows()
        fold_of = assign_dev_folds(rows, 3)
        for fold in range(3):
            srcs = {c.split("-")[0] for c, f in fold_of.items() if f == fold}
            self.assertEqual(srcs, {"company", "newdata"})
            sizes = [float(r["truth_mm"]) for r in rows if fold_of[str(r["case_id"])] == fold]
            self.assertTrue(min(sizes) < 3)  # 小档在每折 val 都出现
            self.assertTrue(max(sizes) > 7)  # 大档也在

    def test_case_level_disjointness(self) -> None:
        rows = make_rows()
        fold_of = assign_dev_folds(rows, 3)
        train_ids = {r["case_id"] for r in rows if fold_of[r["case_id"]] != 0}
        val_ids = {r["case_id"] for r in rows if fold_of[r["case_id"]] == 0}
        self.assertFalse(train_ids & val_ids)


class HybridLossTests(unittest.TestCase):
    def test_gradient_pressure_ratio_between_small_and_large(self) -> None:
        # 2mm 与 8mm 病例各错 1mm：纯相对梯度比 4:1，混合(w=0.25,m=5)约 1.5:1
        pred = torch.tensor([3.0, 9.0], requires_grad=True)
        truth = torch.tensor([2.0, 8.0])
        cases = ["a", "b"]
        for mode, expected_ratio in ((dict(relative=True), 4.0),
                                     (dict(hybrid_rel_weight=0.25, hybrid_rel_scale=5.0), 1.5)):
            p = pred.detach().clone().requires_grad_(True)
            loss = mm_case_bin(p, truth, cases, 1.0, **mode)
            loss.backward()
            g = p.grad.abs().tolist()
            ratio = g[0] / g[1]
            self.assertAlmostEqual(ratio, expected_ratio, places=1, msg=str(mode))

    def test_hybrid_reduces_to_absolute_shape_when_w_zero(self) -> None:
        pred = torch.tensor([3.0, 9.0])
        truth = torch.tensor([2.0, 8.0])
        off = mm_case_bin(pred, truth, ["a", "b"], 99.0, hybrid_rel_weight=0.0)
        # w=0 时走 SmoothL1 分支（β=99、err=1 → 二次段 0.5·err²/β，病例均值）
        self.assertAlmostEqual(off.item(), 0.5 * 1.0 / 99.0, places=6)


class LogSpaceLossTests(unittest.TestCase):
    """对数空间（乘性）损失：对整体缩放不变，且 mm_rank 正确透传。"""

    def test_scale_invariance(self) -> None:
        pred = torch.tensor([3.0, 9.0])
        truth = torch.tensor([2.0, 8.0])
        c = ["a", "b"]
        l1 = mm_case_bin(pred, truth, c, 1.0, log_space=True)
        l2 = mm_case_bin(pred * 10, truth * 10, c, 1.0, log_space=True)
        self.assertAlmostEqual(float(l1), float(l2), places=6)

    def test_log_space_differs_from_relative(self) -> None:
        pred = torch.tensor([3.0, 9.0])
        truth = torch.tensor([2.0, 8.0])
        c = ["a", "b"]
        self.assertNotAlmostEqual(
            float(mm_case_bin(pred, truth, c, 1.0, log_space=True)),
            float(mm_case_bin(pred, truth, c, 1.0, relative=True)),
            places=3,
        )

    def test_mm_rank_passes_log_space(self) -> None:
        from core.config import LossConfig
        from core.losses.mm_rank import build_mm_rank
        from dlef.contracts import ModelOutput

        fn = build_mm_rank(LossConfig(name="mm_rank", log_space=True, log_beta=0.1))
        pred = torch.tensor([3.0, 9.0, 5.0])
        y = torch.tensor([2.0, 8.0, 6.0])
        out = ModelOutput(predictions=pred, aux={"features": torch.randn(3, 8)})
        got = fn(out, {"y": y, "case_id": ["a", "b", "c"]})
        expect = mm_case_bin(pred, y, ["a", "b", "c"], 1.0, log_space=True)
        self.assertAlmostEqual(
            float(got.terms["mm_case_bin"]), float(expect), places=6
        )


if __name__ == "__main__":
    unittest.main()


class BranchSupervisionTests(unittest.TestCase):
    """D-0045：分支独立监督的加权与项名。"""

    def test_branch_terms_added_with_weights(self) -> None:
        from core.config import LossConfig
        from core.losses.mm_rank import build_mm_rank
        from dlef.contracts import ModelOutput

        pred = torch.tensor([3.0, 9.0, 5.0])
        y = torch.tensor([2.0, 8.0, 6.0])
        out = ModelOutput(
            predictions=pred,
            aux={
                "features": torch.randn(3, 8),
                "pred_beta": pred * 1.1,
                "pred_direct": pred * 0.9,
            },
        )
        cfg = LossConfig(name="mm_rank", log_space=True,
                         branch_beta_weight=0.25, branch_direct_weight=0.25)
        res = build_mm_rank(cfg)(out, {"y": y, "case_id": ["a", "b", "c"]})
        self.assertIn("loss_beta", res.terms)
        self.assertIn("loss_direct", res.terms)
        total = (float(res.terms["mm_case_bin"]) + float(res.terms["rank"])
                 + 0.25 * float(res.terms["loss_beta"])
                 + 0.25 * float(res.terms["loss_direct"]))
        self.assertAlmostEqual(float(res.total), total, places=5)

    def test_branch_weights_require_mm_rank(self) -> None:
        from core.config import LossConfig
        from dlef.config import ConfigError

        with self.assertRaises(ConfigError):
            LossConfig(name="mm_case_bin", branch_beta_weight=0.25).validate()

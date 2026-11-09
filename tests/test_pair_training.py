"""D-0050：真实远近成对采样 + β 分支配对一致性。"""

from __future__ import annotations

import unittest

import torch

from core.config import LossConfig
from core.data.samplers import MixedPairSampler, PairSampler
from core.losses.mm_rank import build_mm_rank
from dlef.contracts import ModelOutput


def _rows() -> list[dict]:
    return [
        {"case_id": "c1", "bbox_content": [0, 0, 10, 10]},
        {"case_id": "c1", "bbox_content": [0, 0, 40, 40]},
        {"case_id": "c1", "bbox_content": [0, 0, 25, 25]},
        {"case_id": "c2", "bbox_content": [0, 0, 10, 10]},
    ]


class PairSamplerTests(unittest.TestCase):
    def test_takes_extreme_spans_and_skips_single_frame_cases(self):
        sampler = PairSampler(_rows(), seed=0)
        self.assertEqual(len(sampler), 4)  # 默认与行数同量（步数可比）
        self.assertEqual(set(sampler), {0, 1})  # 跨度最小/最大的两帧
        self.assertEqual(list(sampler)[:2], list(sampler)[2:])  # 同一对重复出现

    def test_empty_when_no_case_has_two_frames(self):
        with self.assertRaises(ValueError):
            PairSampler([{"case_id": "a", "bbox_content": [0, 0, 1, 1]}])

    def test_epoch_changes_order_not_membership(self):
        sampler = PairSampler(_rows(), seed=0)
        first = list(sampler)
        sampler.set_epoch(1)
        second = list(sampler)
        self.assertEqual(set(first), set(second))


class MixedPairSamplerTests(unittest.TestCase):
    def test_first_pair_batches_then_full_coverage(self):
        rows = _rows() + [
            {"case_id": "c3", "bbox_content": [0, 0, 20, 20]},
            {"case_id": "c3", "bbox_content": [0, 0, 60, 60]},
        ]
        sampler = MixedPairSampler(
            rows, seed=0, batch_size=4, pair_batches=2, min_span_ratio=1.3
        )
        order = list(sampler)
        n_pairs = 2 * (4 // 2)
        self.assertEqual(len(sampler), 2 * n_pairs + len(rows))
        head = order[: 2 * n_pairs]
        self.assertTrue(all(rows[head[i]]["case_id"] == rows[head[i + 1]]["case_id"]
                            for i in range(0, len(head), 2)))
        tail = order[2 * n_pairs:]
        self.assertEqual(sorted(tail), list(range(len(rows))))  # 100% 覆盖且不重复


class PairBetaLossTests(unittest.TestCase):
    def _loss(self):
        config = LossConfig(
            name="mm_rank", log_space=True, pair_beta_weight=1.0, rank_gamma=0.0
        )
        return build_mm_rank(config)

    def _output(self, pred_beta):
        return ModelOutput(
            predictions=torch.tensor([5.0, 5.0]),
            aux={"features": torch.zeros(2, 4), "pred_beta": torch.as_tensor(pred_beta)},
        )

    def test_equal_branch_predictions_give_zero_pair_loss(self):
        batch = {"y": torch.tensor([5.0, 5.0]), "case_id": ["c", "c"]}
        out = self._loss()(self._output([5.0, 5.0]), batch)
        self.assertAlmostEqual(float(out.terms["loss_pair_beta"]), 0.0, places=6)

    def test_different_branch_predictions_are_penalised(self):
        batch = {"y": torch.tensor([5.0, 5.0]), "case_id": ["c", "c"]}
        out = self._loss()(self._output([4.0, 6.0]), batch)
        self.assertGreater(float(out.terms["loss_pair_beta"]), 0.0)

    def test_missing_branch_output_raises(self):
        batch = {"y": torch.tensor([5.0, 5.0]), "case_id": ["c", "c"]}
        output = ModelOutput(
            predictions=torch.tensor([5.0, 5.0]), aux={"features": torch.zeros(2, 4)}
        )
        with self.assertRaises(RuntimeError):
            self._loss()(output, batch)


if __name__ == "__main__":
    unittest.main()

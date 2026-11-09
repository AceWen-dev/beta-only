"""D-0069：分档头（ordinal head）+ `mm_ordinal` 损失的契约测试。

锁定四条：
1. `ordinal_head=False` 时模型与旧结构完全一致（不构造 head_ord/head_off）；
2. 输出 = 档左端 + 档内偏移，档位定义与 `mm_case_bin.bin_of` 一致（>=10mm 为第 11 档）；
3. 初值落在训练集中位数所在档的档中心（与旧 head_dir 的 bias 初值同一思路）；
4. 损失 = CE（病例/档位等权）+ 档内偏移 + RankSim，三项都能反传。
"""

from __future__ import annotations

import unittest

import torch

from core.config import ConfigError, LossConfig, ModelConfig, ProjectConfig
from core.losses.build import build_loss
from core.losses.mm_case_bin import bin_of
from core.models.v11 import ORDINAL_BINS, TAIL_BIN, V11Net, _forward_v11
from dlef.contracts import ModelOutput


def make_batch(n: int = 1) -> dict:
    batch = {
        "rgb_views": torch.zeros(n, 2, 3, 224, 224),
        "depth": torch.zeros(n, 2, 2, 128, 128),
        "scalars": torch.zeros(n, 10),
        "D": torch.full((n,), 0.4),
        "S": torch.full((n,), 120.0),
        "y": torch.full((n,), 5.0),
        "case_id": [f"c{i}" for i in range(n)],
    }
    batch["scalars"][:, 0] = 0.4
    batch["scalars"][:, 5] = 1.0
    return batch


def build(**kw) -> V11Net:
    return V11Net(ModelConfig(name="v11", views=2, **kw)).eval()


class BinConventionTests(unittest.TestCase):
    def test_bins_match_bin_of(self):
        self.assertEqual(ORDINAL_BINS, 11)
        self.assertEqual(TAIL_BIN, 10)
        self.assertEqual([bin_of(v) for v in (0.5, 5.0, 9.99, 10.0, 14.0)],
                         [0, 5, 9, 10, 10])


class StructureTests(unittest.TestCase):
    def test_flag_off_keeps_old_heads(self):
        keys = set(build().state_dict())
        self.assertIn("head_beta.0.weight", keys)
        self.assertIn("head_dir.0.weight", keys)
        self.assertIn("gate.weight", keys)
        self.assertNotIn("head_ord.weight", keys)

    def test_flag_on_replaces_fusion_heads(self):
        keys = set(build(ordinal_head=True).state_dict())
        self.assertIn("head_ord.weight", keys)
        self.assertIn("head_off.weight", keys)
        for old in ("head_beta.0.weight", "head_dir.0.weight", "gate.weight"):
            self.assertNotIn(old, keys)
        self.assertEqual(tuple(build(ordinal_head=True).head_ord.bias.shape), (ORDINAL_BINS,))

    def test_forward_shapes_and_aux(self):
        net = build(ordinal_head=True)
        with torch.no_grad():
            pred, pooled = net(make_batch(3))
        self.assertEqual(tuple(pred.shape), (3,))
        self.assertEqual(tuple(pooled.shape), (3, 64))
        out = _forward_v11(net, make_batch(3))
        self.assertEqual(tuple(out.aux["ord_logits"].shape), (3, ORDINAL_BINS))
        self.assertEqual(tuple(out.aux["ord_offset"].shape), (3,))
        self.assertIn("features", out.aux)


class ReadoutTests(unittest.TestCase):
    def test_init_predicts_median_bin_center(self):
        net = build(ordinal_head=True)  # truth_median=5.0 → 档 5，偏移 sigmoid(0)=0.5
        with torch.no_grad():
            pred, _ = net(make_batch(2))
        self.assertTrue(torch.allclose(pred, torch.full((2,), 5.5), atol=1e-6))

    def test_offset_is_added_to_bin_left_edge(self):
        net = build(ordinal_head=True)
        with torch.no_grad():
            net.head_ord.bias.zero_()
            net.head_ord.bias[9] = 3.0
            net.head_off.bias.fill_(1.0)  # sigmoid(1) = 0.7310586
            pred, _ = net(make_batch(1))
        self.assertAlmostEqual(float(pred[0]), 9.0 + torch.sigmoid(torch.tensor(1.0)).item(),
                               places=5)

    def test_tail_bin_is_unbounded_above_ten(self):
        net = build(ordinal_head=True)
        with torch.no_grad():
            net.head_ord.bias.zero_()
            net.head_ord.bias[TAIL_BIN] = 3.0
            net.head_off.bias.fill_(-2.0)
            pred, _ = net(make_batch(1))
        expect = 10.0 + float(torch.nn.functional.softplus(torch.tensor(-2.0)))
        self.assertAlmostEqual(float(pred[0]), expect, places=5)
        self.assertGreaterEqual(float(pred[0]), 10.0)


class LossTests(unittest.TestCase):
    def output(self, logits, raw_off, n_feat=6):
        n = logits.shape[0]
        left = logits.argmax(dim=1).to(raw_off.dtype)
        tail = logits.argmax(dim=1) == TAIL_BIN
        pred = torch.where(tail, left + torch.nn.functional.softplus(raw_off),
                           left + torch.sigmoid(raw_off))
        return ModelOutput(
            predictions=pred,
            aux={"features": torch.randn(n, n_feat), "ord_logits": logits,
                 "ord_offset": raw_off},
        )

    def loss(self, **kw):
        cfg = LossConfig(**{"name": "mm_ordinal", "rank_gamma": 2.0,
                            "rank_interp": 1.0, **kw})
        cfg.validate()
        return build_loss(cfg)

    def test_terms_and_backward(self):
        torch.manual_seed(0)
        logits = torch.randn(4, ORDINAL_BINS, requires_grad=True)
        raw_off = torch.randn(4, requires_grad=True)
        batch = make_batch(4)
        batch["y"] = torch.tensor([2.4, 5.1, 9.6, 12.0])
        out = self.loss()(self.output(logits, raw_off), batch)
        self.assertEqual(sorted(out.terms), ["ord_ce", "ord_offset", "rank"])
        self.assertTrue(torch.isfinite(out.total))
        out.total.backward()
        self.assertIsNotNone(logits.grad)
        self.assertIsNotNone(raw_off.grad)

    def test_case_and_bin_equal_weighting(self):
        # 两个病例各一帧、分属不同档 → CE 等于两条样本 CE 的简单平均
        logits = torch.zeros(2, ORDINAL_BINS)
        raw_off = torch.zeros(2)
        batch = make_batch(2)
        batch["y"] = torch.tensor([2.0, 9.0])
        out = self.loss(rank_gamma=0.0)(self.output(logits, raw_off), batch)
        per = torch.nn.functional.cross_entropy(
            logits, torch.tensor([bin_of(2.0), bin_of(9.0)]), reduction="none")
        self.assertAlmostEqual(float(out.terms["ord_ce"]), float(per.mean()), places=6)

    def test_offset_target_is_position_within_true_bin(self):
        logits = torch.zeros(1, ORDINAL_BINS)
        logits[0, 9] = 3.0
        raw_off = torch.full((1,), 10.0)  # sigmoid ≈ 1
        batch = make_batch(1)
        batch["y"] = torch.tensor([9.5])  # 档 9 内位置 0.5
        out = self.loss(rank_gamma=0.0, ordinal_offset_beta=0.1)(
            self.output(logits, raw_off), batch)
        expect = float(torch.nn.functional.smooth_l1_loss(
            torch.sigmoid(torch.tensor(10.0)), torch.tensor(0.5), beta=0.1))
        self.assertAlmostEqual(float(out.terms["ord_offset"]), expect, places=6)

    def test_tail_only_batch_has_zero_offset_term(self):
        logits = torch.zeros(2, ORDINAL_BINS)
        logits[:, TAIL_BIN] = 3.0
        batch = make_batch(2)
        batch["y"] = torch.tensor([11.0, 14.0])
        out = self.loss(rank_gamma=0.0)(self.output(logits, torch.zeros(2)), batch)
        self.assertEqual(float(out.terms["ord_offset"]), 0.0)
        self.assertGreater(float(out.terms["ord_ce"]), 0.0)


class ConfigTests(unittest.TestCase):
    def test_ordinal_requires_v11(self):
        with self.assertRaises(ConfigError):
            ModelConfig(name="mlp", ordinal_head=True).validate()

    def test_ordinal_is_single_variable(self):
        for kw in ({"gate_beta_floor": 0.5}, {"direct_scale_anchor": True}, {"size_input": True}):
            with self.subTest(**kw):
                with self.assertRaises(ConfigError):
                    ModelConfig(name="v11", ordinal_head=True, **kw).validate()

    def test_head_and_loss_must_be_paired(self):
        base = ProjectConfig()
        with self.assertRaises(ConfigError):
            ProjectConfig(model=ModelConfig(name="v11", ordinal_head=True),
                          loss=base.loss).validate()
        with self.assertRaises(ConfigError):
            ProjectConfig(model=base.model,
                          loss=LossConfig(name="mm_ordinal")).validate()
        ProjectConfig(model=ModelConfig(name="v11", ordinal_head=True),
                      loss=LossConfig(name="mm_ordinal")).validate()

    def test_ordinal_loss_rejects_scalar_loss_options(self):
        for kw in ({"log_space": True}, {"relative": True}, {"branch_beta_weight": 1.0},
                   {"pair_beta_weight": 1.0}):
            with self.subTest(**kw):
                with self.assertRaises(ConfigError):
                    LossConfig(name="mm_ordinal", **kw).validate()


if __name__ == "__main__":
    unittest.main()

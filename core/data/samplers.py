"""病例/尺寸档均衡采样权重。

动机：训练集里 6–7 mm 有几百帧、2–3 mm 只有几十帧。普通 shuffle 每轮看到的
小息肉样本数就是它的帧数，模型自然把小尺寸档当噪声忽略。这里按**病例**加权：

1. 每个病例取标签中位数落到哪个 1 mm 档；
2. 3–6 mm 档的病例权重 1.0，其余档按 ``min(5, 中位档病例数 / 该档病例数)``；
3. 每个病例内部再除以它的帧数 —— 一例 600 帧不能压过一例只有 2 帧。

最终每帧权重 = 病例档位权重 / 该病例帧数。与 PolypMeasure 侧既有最优配方一致。
"""

from __future__ import annotations

import random
from collections import defaultdict

import numpy as np
from torch.utils.data import Sampler


def bin_of(value: float) -> int:
    """1 mm 分档：9.7 → 9，10.0 → 10。"""
    return 10 if value >= 10.0 else int(value)


def case_bin_weights(rows: list[dict]) -> np.ndarray:
    """Return one sampling weight per row (unnormalised)."""
    by_case: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_case[str(row["case_id"])].append(row)

    case_bin = {
        case: bin_of(float(np.median([float(r["truth_mm"]) for r in rs])))
        for case, rs in by_case.items()
    }
    company_counts: dict[int, int] = defaultdict(int)
    for case, bucket in case_bin.items():
        if case.startswith("company-"):
            company_counts[bucket] += 1

    reference = [company_counts[b] for b in range(3, 7) if company_counts[b] > 0]
    base = float(np.median(reference)) if reference else 1.0

    weights = []
    for row in rows:
        case = str(row["case_id"])
        bucket = case_bin[case]
        if 3 <= bucket <= 6 or bucket not in company_counts:
            case_weight = 1.0
        else:
            case_weight = min(5.0, base / max(1, company_counts[bucket]))
        weights.append(case_weight / len(by_case[case]))
    return np.asarray(weights, dtype=np.float64)


class PairSampler(Sampler[int]):
    """真实远近成对采样（D-0050）。

    每个病例取表观跨度（bbox 最大边长）**最大/最小**的两帧组成一对，一对在
    批内相邻。病例顺序与对内先后每轮重新洗牌（由 ``set_epoch`` 驱动），因此
    "近在前/远在前"不会固定。只有 ≥2 帧的病例参与；单帧病例在本模式下不采样。

    配对本身不含标签信息，只改变 batch 组成；配合 ``loss.pair_beta_weight``
    才能约束 β 分支在同一息肉的不同取景下输出一致。

    ``num_samples`` 默认等于行数：一轮里把所有对按洗牌顺序循环若干遍，凑够与
    普通 shuffle 相同的样本量（步数/epoch 与基线可比）。同一对一轮会被看到多次，
    这是成对采样的固有代价；两臂（配对/配对+β一致性）完全一致。
    """

    def __init__(
        self,
        rows: list[dict],
        seed: int = 0,
        num_samples: int | None = None,
        min_span_ratio: float = 1.0,
    ) -> None:
        groups: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            groups[str(row["case_id"])].append(index)
        self.pairs: list[tuple[int, int]] = []
        for _case, idxs in sorted(groups.items()):
            if len(idxs) < 2:
                continue
            spans = {
                i: max(
                    float(rows[i]["bbox_content"][2]) - float(rows[i]["bbox_content"][0]),
                    float(rows[i]["bbox_content"][3]) - float(rows[i]["bbox_content"][1]),
                )
                for i in idxs
            }
            ordered = sorted(idxs, key=lambda i: spans[i])
            if spans[ordered[-1]] / max(spans[ordered[0]], 1e-6) < min_span_ratio:
                continue  # 取景几乎没变的病例不提供配对信号
            self.pairs.append((ordered[0], ordered[-1]))
        if not self.pairs:
            raise ValueError("PairSampler needs at least one case with >= 2 frames")
        self.seed = int(seed)
        self.num_samples = int(num_samples) if num_samples else len(rows)
        self.epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        yielded = 0
        while yielded < self.num_samples:
            order = list(range(len(self.pairs)))
            rng.shuffle(order)
            for k in order:
                if yielded >= self.num_samples:
                    break
                far, near = self.pairs[k]
                if rng.random() < 0.5:
                    far, near = near, far
                yield far
                yield near
                yielded += 2

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


class MixedPairSampler(Sampler[int]):
    """混合采样（D-0056）：配对批 + 全帧 shuffle。

    每个 epoch 先产出 ``pair_batches`` 个批，每批 ``batch_size//2`` 对
    （同一病例的远/近两帧相邻，有放回抽取），随后把**全部行**打乱各出现一次。
    这样既保留 D-0052 里有效的一致性信号，又不牺牲帧多样性。
    """

    def __init__(
        self,
        rows: list[dict],
        seed: int = 0,
        *,
        batch_size: int = 16,
        pair_batches: int = 16,
        min_span_ratio: float = 1.0,
    ) -> None:
        groups: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            groups[str(row["case_id"])].append(index)
        self.pairs: list[tuple[int, int]] = []
        for _case, idxs in sorted(groups.items()):
            if len(idxs) < 2:
                continue
            spans = {
                i: max(
                    float(rows[i]["bbox_content"][2]) - float(rows[i]["bbox_content"][0]),
                    float(rows[i]["bbox_content"][3]) - float(rows[i]["bbox_content"][1]),
                )
                for i in idxs
            }
            ordered = sorted(idxs, key=lambda i: spans[i])
            if spans[ordered[-1]] / max(spans[ordered[0]], 1e-6) < min_span_ratio:
                continue
            self.pairs.append((ordered[0], ordered[-1]))
        if not self.pairs:
            raise ValueError("MixedPairSampler needs at least one case with >= 2 frames")
        self.n_rows = len(rows)
        self.batch_size = int(batch_size)
        self.pair_batches = int(pair_batches)
        self.pairs_per_batch = max(1, self.batch_size // 2)
        self.n_pairs = self.pair_batches * self.pairs_per_batch
        self.seed = int(seed)
        self.epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        for _ in range(self.n_pairs):
            far, near = self.pairs[rng.randrange(len(self.pairs))]
            if rng.random() < 0.5:
                far, near = near, far
            yield far
            yield near
        order = list(range(self.n_rows))
        rng.shuffle(order)
        yield from order

    def __len__(self) -> int:
        return 2 * self.n_pairs + self.n_rows

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

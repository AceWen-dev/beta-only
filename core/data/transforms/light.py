"""轻量增广：水平翻转 + 亮度/对比度抖动 + 高斯噪声（D-0040）。

只作用于训练集（role=train 才装配），且只改 `rgb_views`：
- 翻转：所有视图同步左右翻转。`S`（框最大边长）在翻转下不变，无需改；
- 亮度/对比度：在 ImageNet 归一化后的张量上做 `v*a + b`，等价于轻微曝光/增益扰动；
- 噪声：小方差高斯，模拟传感器噪声。

不做旋转/裁剪：预裁剪视图旋转会引入黑角，裁剪会破坏与 `S` 的尺度一致性。
"""

from __future__ import annotations

import random

import torch

from .base import Transform


def build_light() -> Transform:
    def light(sample):
        views = sample.get("rgb_views")
        if views is None or not torch.is_tensor(views):
            return sample
        views = views.clone()
        if random.random() < 0.5:
            views = torch.flip(views, dims=[-1])          # 水平翻转
        gain = 1.0 + random.uniform(-0.15, 0.15)          # 对比度
        offset = random.uniform(-0.10, 0.10)              # 亮度（归一化空间）
        views = views * gain + offset
        if random.random() < 0.3:
            views = views + 0.02 * torch.randn_like(views)
        sample = dict(sample)
        sample["rgb_views"] = views
        return sample

    return Transform(name="light", fn=light)

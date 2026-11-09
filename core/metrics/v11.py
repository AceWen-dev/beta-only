"""v11 指标：总体/分层 MAE、10 mm 一致率、最差 1 mm 区间。

D-0077 起同时输出两个**选模诊断**（不改变既有验收指标的定义）：

- ``case_mae_lt10``：病例内帧误差先平均、再病例间平均（<10mm）；
- ``robust_worst_bin``：只在病例数 >= 8 的 1mm 档里取最差档的帧 MAE。

``v11/worst_bin`` 仍在 val 上由极少数 9–10 档病例决定，噪声大；训练 monitor
改用 ``ema/v11/robust_worst_bin``，``case_mae_lt10`` 作为护栏。
"""

from __future__ import annotations

import math
from collections import defaultdict

from dlef.contracts import MetricResult

MIN_CASES_PER_BIN = 8


class V11Metric:
    def __init__(self, min_cases: int = MIN_CASES_PER_BIN) -> None:
        self.min_cases = int(min_cases)
        self.reset()

    def reset(self) -> None:
        self._rows: list[tuple[float, float, str]] = []

    def update(self, output, batch) -> None:
        preds = output.predictions.detach().cpu().reshape(-1)
        targets = batch["y"].detach().cpu().reshape(-1)
        cases = batch.get("case_id")
        if cases is None:
            # 兼容没有 case_id 的合成/旧数据：每个样本当作一个独立病例，
            # case_* 指标退化为帧级；v11_frames 训练/评测始终提供 case_id。
            cases = [f"__row{i}" for i in range(len(preds))]
        for pred, target, case in zip(preds.tolist(), targets.tolist(), cases):
            self._rows.append((float(pred), float(target), str(case)))

    def compute(self) -> MetricResult:
        if not self._rows:
            return MetricResult(values={})
        preds = [p for p, _, _ in self._rows]
        targets = [t for _, t, _ in self._rows]
        n = len(self._rows)

        values: dict[str, float] = {
            "mae": sum(abs(p - t) for p, t in zip(preds, targets)) / n
        }
        lt10 = [(p, t) for p, t in zip(preds, targets) if t < 10]
        ge10 = [(p, t) for p, t in zip(preds, targets) if t >= 10]
        values["mae_lt10"] = (
            sum(abs(p - t) for p, t in lt10) / len(lt10) if lt10 else float("nan")
        )
        values["mae_ge10"] = (
            sum(abs(p - t) for p, t in ge10) / len(ge10) if ge10 else float("nan")
        )
        values["acc10"] = sum((p < 10) == (t < 10) for p, t in zip(preds, targets)) / n

        worst = None
        for k in range(0, 10):
            errs = [abs(p - t) for p, t in zip(preds, targets) if k <= t < k + 1]
            if errs:
                m = sum(errs) / len(errs)
                values[f"mae_{k}_{k+1}"] = m
                worst = m if worst is None else max(worst, m)
            else:
                values[f"mae_{k}_{k+1}"] = float("nan")
        values["worst_bin"] = worst if worst is not None else float("nan")

        by_case: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for pred, target, case in self._rows:
            by_case[case].append((pred, target))
        case_all: list[float] = []
        case_lt10: list[float] = []
        for case_rows in by_case.values():
            errs = [abs(p - t) for p, t in case_rows]
            case_all.append(sum(errs) / len(errs))
            lt = [abs(p - t) for p, t in case_rows if t < 10]
            if lt:
                case_lt10.append(sum(lt) / len(lt))
        values["case_mae"] = sum(case_all) / len(case_all)
        values["case_mae_lt10"] = (
            sum(case_lt10) / len(case_lt10) if case_lt10 else float("nan")
        )

        bin_err: dict[int, list[float]] = defaultdict(list)
        bin_cases: dict[int, set[str]] = defaultdict(set)
        for pred, target, case in self._rows:
            k = int(math.floor(target))
            if 0 <= k <= 9:
                bin_err[k].append(abs(pred - target))
                bin_cases[k].add(case)
        eligible = {
            k: sum(v) / len(v)
            for k, v in bin_err.items()
            if len(bin_cases[k]) >= self.min_cases
        }
        values["robust_worst_bin"] = (
            max(eligible.values()) if eligible else float("nan")
        )
        values["robust_bin_count"] = float(len(eligible))
        return MetricResult(values=values)


def build_v11_metrics() -> V11Metric:
    return V11Metric()

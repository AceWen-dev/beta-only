#!/usr/bin/env python3
"""单调标定诊断（D-0066）：预测的「压缩偏差」能否用 val 拟合的单调映射消除？

动机（来自 output/reports/smallbin_audit/）：
- company / R 的标签在 1mm 档内几乎是常数（整数 mm，档内 std=0.00）；
- final_ema 的预测均值只从 3.95（2–3mm 档）走到 6.54（9–10mm 档），真值却从 2 走到 9；
- 逐档 MAE ≈ |系统偏差|（差 ≤0.05mm）→ 误差主要是**压缩偏置**，不是随机噪声。

与旧结论的区别：isotonic 标定在 single-v2 上被证伪（STATE §4.1 item 6），但那次
valmerge 后的 val 只覆盖 4–6mm，档外只能 clip，压不到 2–3 与 9–10。company-v1 的
val 有 2,094 帧 / 182 例，覆盖 2–3（39 帧/10 例）到 ≥10（499 帧），第一次具备拟合
尾部单调映射的条件。

口径：
- **只在 val 上拟合**，绝不在 test 上拟合（AGENTS.md 禁止在 test 上调参）；
- val 自身也参与过选模，所以 val 侧数字用**病例级 2 折交叉拟合**（诚实估计）；
- 单调映射不改变排序 → **不改善分辨力**（档内 pred std 不变）。本脚本同时报告
  偏差消除量与分辨力，避免把「标定收益」误读成「测量能力提升」。

用法：
    python scripts/bias_recal_diagnosis.py --run <company-base run 目录>
产物：
    output/reports/recal_diagnosis/<run_id>.md / .json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = Path("/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/Betaregression_data")

from core.config import ModelConfig  # noqa: E402
from core.data.datasets.v11_frames import V11FramesDataset  # noqa: E402
from core.models.v11 import V11Net  # noqa: E402

BINS = tuple(range(1, 10))
THRESHOLD = 2.0
FIT = ("company-v1", "val")
EVAL = {
    "company": ("company-v1", "test_company"),
    "B": ("single-v2", "test_b"),
    "R": ("single-v2", "test_r"),
}


def pava(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Isotonic（递增）回归的 PAVA 实现；按 x 排序后拟合 y 的单调阶梯。"""
    order = np.argsort(x)
    x, y = x[order], y[order]
    blocks = [[1.0, v] for v in y]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][1] > blocks[i + 1][1] + 1e-12:
            w = blocks[i][0] + blocks[i + 1][0]
            v = (blocks[i][0] * blocks[i][1] + blocks[i + 1][0] * blocks[i + 1][1]) / w
            blocks[i:i + 2] = [[w, v]]
            i = max(0, i - 1)
        else:
            i += 1
    fitted = []
    for w, v in blocks:
        fitted.extend([v] * int(round(w)))
    return np.asarray(fitted)


class Isotonic:
    """val 上拟合的 pred→truth 单调映射；区间外 clip。"""

    def fit(self, pred: np.ndarray, truth: np.ndarray) -> "Isotonic":
        order = np.argsort(pred)
        x, y = pred[order], pava(pred, truth[order])
        # np.interp 要求 xp 严格递增：并列预测值合并为一档
        ux, inv = np.unique(x, return_inverse=True)
        sums = np.zeros(len(ux)); cnts = np.zeros(len(ux))
        np.add.at(sums, inv, y); np.add.at(cnts, inv, 1.0)
        self.x = ux
        self.y = sums / cnts
        return self

    def __call__(self, pred: np.ndarray) -> np.ndarray:
        return np.interp(pred, self.x, self.y, left=self.y[0], right=self.y[-1])


class LogLinear:
    """参数版：log(truth) = a + b·log(pred)。比 isotonic 更抗过拟合，但只能修幂律。"""

    def fit(self, pred: np.ndarray, truth: np.ndarray) -> "LogLinear":
        lp = np.log(np.clip(pred, 1e-3, None))
        lt = np.log(np.clip(truth, 1e-3, None))
        self.b, self.a = np.polyfit(lp, lt, 1)
        return self

    def __call__(self, pred: np.ndarray) -> np.ndarray:
        return np.exp(self.a + self.b * np.log(np.clip(pred, 1e-3, None)))


def rows_of(version: str, split: str, limit: int = 0) -> list[dict]:
    manifest = json.loads(
        (DATA / f"curated/v11_frames/{version}/manifest.json").read_text()
    )["rows"]
    rows = [r for r in manifest if r["split"] == split]
    rows.sort(key=lambda r: (str(r["case_id"]), str(r["sample_id"])))
    return rows[:limit] if limit else rows


def collate(items):
    return {
        **{k: torch.stack([x[k] for x in items])
           for k in ("rgb_views", "depth", "scalars", "D", "S", "y")},
        "case_id": [x["case_id"] for x in items],
    }


def infer(net, rows, device) -> tuple[np.ndarray, np.ndarray, list[str]]:
    ds = V11FramesDataset(rows, DATA, 2, None)
    ps, ts, cs = [], [], []
    with torch.inference_mode():
        for b in DataLoader(ds, batch_size=8, num_workers=2, collate_fn=collate):
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
            p, _ = net(b)
            ps.extend(p.cpu().tolist())
            ts.extend(b["y"].tolist())
            cs.extend(b["case_id"])
    return np.array(ps), np.array(ts), [str(c) for c in cs]


def per_bin_mae(pred: np.ndarray, truth: np.ndarray) -> dict[int, float]:
    out = {}
    for b in BINS:
        m = (truth >= b) & (truth < b + 1)
        if m.any():
            out[b] = float(np.mean(np.abs(pred[m] - truth[m])))
    return out


def lt10_mae(pred: np.ndarray, truth: np.ndarray) -> float:
    m = truth < 10
    return float(np.mean(np.abs(pred[m] - truth[m]))) if m.any() else float("nan")


def worst_bin(pred: np.ndarray, truth: np.ndarray) -> float:
    vals = list(per_bin_mae(pred, truth).values())
    return max(vals) if vals else float("nan")


def case_weighted_lt10(pred, truth, cases) -> float:
    per = defaultdict(list)
    for p, t, c in zip(pred, truth, cases):
        if t < 10:
            per[c].append(abs(float(p) - float(t)))
    return float(np.mean([np.mean(v) for v in per.values()])) if per else float("nan")


def corr(pred: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    m = truth < 10
    p, t = pred[m], truth[m]
    if len(p) < 3 or np.std(p) == 0:
        return float("nan"), float("nan")
    pear = float(np.corrcoef(p, t)[0, 1])
    rp = np.argsort(np.argsort(p)).astype(float)
    rt = np.argsort(np.argsort(t)).astype(float)
    return pear, float(np.corrcoef(rp, rt)[0, 1])


def within_bin_std(pred: np.ndarray, truth: np.ndarray) -> float:
    """档内预测离散度（分辨力代理）：各档 pred std 的加权均值。"""
    vals, ws = [], []
    for b in BINS:
        m = (truth >= b) & (truth < b + 1)
        if int(m.sum()) >= 5:
            vals.append(float(np.std(pred[m])))
            ws.append(int(m.sum()))
    return float(np.average(vals, weights=ws)) if vals else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", default="final_ema.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="output/reports/recal_diagnosis")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="每个 split 最多推理多少帧（0=全部；仅用于 CPU 冒烟）")
    args = ap.parse_args()

    if FIT[1].startswith("test"):
        print("refusing to fit on a test split"); return 2

    run = Path(args.run).resolve()
    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    resolved = json.loads((run / "config.resolved.json").read_text())
    net = V11Net(ModelConfig.from_dict(resolved["model"]))
    ck = torch.load(run / "checkpoints" / args.checkpoint, map_location="cpu",
                    weights_only=False)
    net.load_state_dict(ck.model)
    net = net.to(device).eval()
    print(f"run={run.name} ckpt={args.checkpoint} epoch={ck.epoch} device={device}")

    vp, vt, vc = infer(net, rows_of(*FIT, limit=args.max_frames), device)
    print(f"fit set: {FIT[0]}/{FIT[1]}  n={len(vp)} 例={len(set(vc))} "
          f"真值范围 [{vt.min():.1f}, {vt.max():.1f}]")

    iso_full = Isotonic().fit(vp, vt)
    lin_full = LogLinear().fit(vp, vt)

    # ---- val 侧：病例级 2 折交叉拟合（诚实估计，不含选模乐观） ----
    uniq = sorted(set(vc))
    fold_of = {c: i % 2 for i, c in enumerate(uniq)}
    fold = np.array([fold_of[c] for c in vc])
    cf = {"iso": np.zeros_like(vp), "lin": np.zeros_like(vp)}
    for f in (0, 1):
        tr, te = fold != f, fold == f
        cf["iso"][te] = Isotonic().fit(vp[tr], vt[tr])(vp[te])
        cf["lin"][te] = LogLinear().fit(vp[tr], vt[tr])(vp[te])

    report: dict = {
        "run": run.name,
        "checkpoint": args.checkpoint,
        "epoch": int(ck.epoch),
        "fit_set": {"version": FIT[0], "split": FIT[1], "n_frames": int(len(vp)),
                    "n_cases": len(uniq), "truth_min": float(vt.min()),
                    "truth_max": float(vt.max())},
        "log_linear": {"a": float(lin_full.a), "b": float(lin_full.b)},
        "val_crossfit": {},
        "test": {},
    }

    def block(before, after_iso, after_lin, truth, cases):
        return {
            "lt10_before": round(lt10_mae(before, truth), 4),
            "lt10_iso": round(lt10_mae(after_iso, truth), 4),
            "lt10_lin": round(lt10_mae(after_lin, truth), 4),
            "lt10_case_before": round(case_weighted_lt10(before, truth, cases), 4),
            "lt10_case_iso": round(case_weighted_lt10(after_iso, truth, cases), 4),
            "worst_bin_before": round(worst_bin(before, truth), 3),
            "worst_bin_iso": round(worst_bin(after_iso, truth), 3),
            "worst_bin_lin": round(worst_bin(after_lin, truth), 3),
            "bins_over2_before": sorted(b for b, v in per_bin_mae(before, truth).items()
                                        if v > THRESHOLD),
            "bins_over2_iso": sorted(b for b, v in per_bin_mae(after_iso, truth).items()
                                     if v > THRESHOLD),
            "bins_over2_lin": sorted(b for b, v in per_bin_mae(after_lin, truth).items()
                                     if v > THRESHOLD),
            "per_bin_before": {str(k): round(v, 3) for k, v in per_bin_mae(before, truth).items()},
            "per_bin_iso": {str(k): round(v, 3) for k, v in per_bin_mae(after_iso, truth).items()},
            "per_bin_lin": {str(k): round(v, 3) for k, v in per_bin_mae(after_lin, truth).items()},
            "pearson": round(corr(before, truth)[0], 4),
            "spearman": round(corr(before, truth)[1], 4),
            "spearman_iso": round(corr(after_iso, truth)[1], 4),
            "within_bin_pred_std": round(within_bin_std(before, truth), 3),
        }

    report["val_crossfit"] = block(vp, cf["iso"], cf["lin"], vt, vc)

    for name, (version, split) in EVAL.items():
        rows = rows_of(version, split, limit=args.max_frames)
        p, t, c = infer(net, rows, device)
        report["test"][name] = block(p, iso_full(p), lin_full(p), t, c)
        report["test"][name]["n_frames"] = len(p)
        report["test"][name]["version"] = version
        print(f"  {name:8s} n={len(p):5d} lt10 {report['test'][name]['lt10_before']:.3f} "
              f"→ iso {report['test'][name]['lt10_iso']:.3f} "
              f"/ lin {report['test'][name]['lt10_lin']:.3f}")

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{run.name}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- markdown ----
    def tbl(title: str, d: dict) -> list[str]:
        lines = [f"### {title}", "",
                 "| 档 | 标定前 | isotonic | log-linear |", "|---|---:|---:|---:|"]
        for b in BINS:
            k = str(b)
            if k not in d["per_bin_before"]:
                continue
            lines.append(f"| {b}–{b+1} | {d['per_bin_before'][k]:.2f} | "
                         f"{d['per_bin_iso'][k]:.2f} | {d['per_bin_lin'][k]:.2f} |")
        lines += ["",
                  f"- `<10mm` 帧级：**{d['lt10_before']:.3f}** → iso "
                  f"**{d['lt10_iso']:.3f}** / lin **{d['lt10_lin']:.3f}**",
                  f"- worst_bin：{d['worst_bin_before']:.2f} → iso "
                  f"{d['worst_bin_iso']:.2f} / lin {d['worst_bin_lin']:.2f}",
                  f"- 超标档（>2mm）：{d['bins_over2_before'] or '无'} → iso "
                  f"{d['bins_over2_iso'] or '无'} / lin {d['bins_over2_lin'] or '无'}",
                  f"- 分辨力（不因单调映射改变）：Pearson {d['pearson']:.3f}、"
                  f"Spearman {d['spearman']:.3f}→{d['spearman_iso']:.3f}、"
                  f"档内 pred std {d['within_bin_pred_std']:.2f}mm",
                  ""]
        return lines

    fs = report["fit_set"]
    L = [
        "# 单调标定诊断（D-0066）",
        "",
        f"- run `{report['run']}`，checkpoint `{report['checkpoint']}`"
        f"（epoch {report['epoch']}）",
        f"- 拟合集：`{fs['version']}/{fs['split']}`，{fs['n_frames']} 帧 / "
        f"{fs['n_cases']} 例，真值 [{fs['truth_min']:.1f}, {fs['truth_max']:.1f}] mm",
        f"- log-linear 参数：log(truth) = {report['log_linear']['a']:+.3f} "
        f"{report['log_linear']['b']:+.3f}·log(pred)",
        "- **口径**：只在 val 上拟合；val 侧数字用病例级 2 折交叉拟合；test 侧是"
        "把 val 拟合的映射直接套用（未在 test 上拟合任何参数）",
        "",
        "## 1. val（交叉拟合，诚实估计）",
        "",
        *tbl(f"{fs['version']}/{fs['split']}", report["val_crossfit"]),
        "## 2. 冻结测试集（套用 val 拟合的映射）",
        "",
    ]
    for name in EVAL:
        d = report["test"][name]
        L += tbl(f"{name}（{d['version']}，{d['n_frames']} 帧）", d)
        if "lt10_case_before" in d:
            L += [f"- 病例等权 `<10mm`：{d['lt10_case_before']:.3f} → iso "
                  f"{d['lt10_case_iso']:.3f}", ""]
    L += [
        "## 3. 怎么读这份结果",
        "",
        "- 若 test 逐档 MAE 明显下降且超标档减少 → **压缩偏差是可用 val 修正的**，"
        "标定应作为配方的一部分（并在 DECISIONS 里记为新决策）。",
        "- 若 val 交叉拟合改善、test 不改善 → 偏差曲线**跨队列不稳定**（域偏移），"
        "标定只对同域有效。",
        "- 无论哪种情况：Spearman 与档内 pred std **不会变**——标定修的是刻度，"
        "不是分辨力。要提升分辨力得换监督信号（例如序数/分档头），不是后处理。",
        "",
    ]
    (out / f"{run.name}.md").write_text("\n".join(L), encoding="utf-8")
    print(f"md:   {out / (run.name + '.md')}")
    print(f"json: {out / (run.name + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""取景远近漂移审计（D-0049 第 1 步）。

问题：同一颗息肉，镜头靠近（像素跨度 S 变大）时，模型预测是否跟着变大？
理想：真实尺寸不变 → 预测应随 S 变化保持稳定（log-log 斜率 0）。

物理分支 pred_beta = β·q·S，其中 q 是相对轴向深度读数（无物理单位）。
若 q 不随距离变化、β 不补偿，则 d log(pred_beta)/d log(S) ≈ 1（完全跟随表观大小）。
本脚本把**融合输出、β 分支、直接分支、门控**分开量化：

    d log(pred)/d log(S) = 0      理想
    = 1                           完全按表观大小缩放（无距离补偿）

并同时给出 d log(q)/d log(S)（理想 −1，距离翻倍时深度读数翻倍）与
d log(β)/d log(S)（理想 0）。

只读 checkpoint 与 dev（该 run 未被训练到的折），不碰任何测试集。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = Path("/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/Betaregression_data")

from core.config import ModelConfig  # noqa: E402
from core.data.datasets.v11_frames import (  # noqa: E402
    V11FramesDataset,
    assign_dev_folds,
)
from core.models.v11 import V11Net  # noqa: E402


def collate(items):
    return {
        **{
            k: torch.stack([x[k] for x in items])
            for k in ("rgb_views", "depth", "scalars", "D", "S", "y")
        },
        "case_id": [x["case_id"] for x in items],
        "sample_id": [x["sample_id"] for x in items],
    }


def slope(x: np.ndarray, y: np.ndarray) -> float:
    """最小二乘斜率 d y / d x；样本不足或无方差时返回 nan。"""
    if len(x) < 3 or np.std(x) < 1e-9:
        return float("nan")
    return float(np.polyfit(x, y, 1)[0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", default="final_ema.pt")
    ap.add_argument("--fold", type=int, default=0, help="dev 折（该 run 未训练的折）")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--min-frames", type=int, default=3)
    ap.add_argument("--min-ratio", type=float, default=1.10,
                    help="S 的 max/min 至少达到该倍数才纳入")
    ap.add_argument("--out", default="output/reports/framing_drift")
    args = ap.parse_args()

    run = Path(args.run).resolve()
    resolved = json.loads((run / "config.resolved.json").read_text())
    net = V11Net(ModelConfig.from_dict(resolved["model"]))
    ck = torch.load(
        run / "checkpoints" / args.checkpoint, map_location="cpu", weights_only=False
    )
    net.load_state_dict(ck.model)
    net = net.to("cuda").eval()

    rows_all = json.loads(
        (DATA / "curated/v11_frames/single-v2/manifest.json").read_text()
    )["rows"]
    dev = [r for r in rows_all if r["split"] in ("train", "val")]
    fold_of = assign_dev_folds(dev, args.folds)
    rows = [r for r in dev if fold_of[str(r["case_id"])] == args.fold]
    print(f"dev fold{args.fold}: {len(rows)} 帧 / {len({r['case_id'] for r in rows})} 病例")

    ds = V11FramesDataset(rows, DATA, 2, None)
    meta = {str(r["sample_id"]): r for r in rows}
    recs: list[dict] = []
    with torch.inference_mode():
        for b in DataLoader(ds, batch_size=16, num_workers=2, collate_fn=collate):
            b = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in b.items()}
            fused, _, br = net(b, return_branches=True)
            for i, sid in enumerate(b["sample_id"]):
                S = float(b["S"][i])
                q = float(b["D"][i])
                a = float(b["scalars"][i][0])  # p/W，归一化跨度
                pb = float(br["pred_beta"][i])
                pd = float(br["pred_direct"][i])
                wg = float(br["gate"][i])
                row = meta[str(sid)]
                w, h = (int(v) for v in row["content_size"])  # [h, w] 或 [w, h] 以框为准
                x1, y1, x2, y2 = (float(v) for v in row["bbox_content"])
                # content_size 在 manifest 里是 [h, w]
                hh, ww = (h, w) if w > h else (w, h)
                truncated = x1 <= 2 or y1 <= 2 or x2 >= ww - 2 or y2 >= hh - 2
                recs.append(
                    {
                        "case_id": str(b["case_id"][i]),
                        "sample_id": str(sid),
                        "truth": float(b["y"][i]),
                        "S_px": S,
                        "q": q,
                        "a": a,
                        "cx": float(b["scalars"][i][3]),
                        "cy": float(b["scalars"][i][4]),
                        "truncated": int(truncated),
                        "pred": float(fused[i]),
                        "pred_beta": pb,
                        "pred_direct": pd,
                        "gate": wg,
                        "beta_w": pb / max(q * a, 1e-9),
                        "beta_eff": pb / max(q * S, 1e-9),
                    }
                )

    by_case: dict[str, list[dict]] = {}
    for r in recs:
        by_case.setdefault(r["case_id"], []).append(r)

    def build_entry(case: str, rs: list[dict]) -> dict:
        rs = sorted(rs, key=lambda r: r["a"])
        loga = np.log([r["a"] for r in rs])
        ratio = rs[-1]["a"] / max(rs[0]["a"], 1e-9)
        entry = {
            "case_id": case,
            "n": len(rs),
            "truth": float(np.median([r["truth"] for r in rs])),
            "a_ratio": float(ratio),
            "a_min": float(rs[0]["a"]),
            "a_max": float(rs[-1]["a"]),
            "n_truncated": int(sum(r["truncated"] for r in rs)),
            "cx_span": float(max(r["cx"] for r in rs) - min(r["cx"] for r in rs)),
            "cy_span": float(max(r["cy"] for r in rs) - min(r["cy"] for r in rs)),
            "pred_mean": float(np.mean([r["pred"] for r in rs])),
            "pred_std": float(np.std([r["pred"] for r in rs])),
            "slope_pred": slope(loga, np.log([r["pred"] for r in rs])),
            "slope_beta": slope(loga, np.log([r["pred_beta"] for r in rs])),
            "slope_direct": slope(loga, np.log([r["pred_direct"] for r in rs])),
            "slope_q": slope(loga, np.log([r["q"] for r in rs])),
            "slope_beta_w": slope(loga, np.log([r["beta_w"] for r in rs])),
            "gate_mean": float(np.mean([r["gate"] for r in rs])),
            "gate_std": float(np.std([r["gate"] for r in rs])),
            "far_pred": float(rs[0]["pred"]),
            "near_pred": float(rs[-1]["pred"]),
            "far_beta": float(rs[0]["pred_beta"]),
            "near_beta": float(rs[-1]["pred_beta"]),
            "far_direct": float(rs[0]["pred_direct"]),
            "near_direct": float(rs[-1]["pred_direct"]),
        }
        entry["drift_mm"] = entry["near_pred"] - entry["far_pred"]
        # 恒等式：b_pred_beta = b_(βW) + b_q + 1（用于核对逐病例分解）
        if np.isfinite(entry["slope_beta_w"]) and np.isfinite(entry["slope_q"]):
            entry["identity_residual"] = (
                entry["slope_beta"] - (entry["slope_beta_w"] + entry["slope_q"] + 1.0)
            )
        else:
            entry["identity_residual"] = float("nan")
        return entry

    summary = []
    small = []
    for case, rs in by_case.items():
        if len(rs) < args.min_frames:
            continue
        # 剔除边界截断帧后若仍有 ≥min_frames 帧，优先用干净帧
        clean = [r for r in rs if not r["truncated"]]
        base = clean if len(clean) >= args.min_frames else rs
        e = build_entry(case, base)
        if e["a_ratio"] < args.min_ratio:
            continue
        summary.append(e)
        if e["truth"] < 3.0:
            small.append(e)

    if not summary:
        print("没有满足条件的病例（min_frames/min_ratio 放宽试试）")
        return 1

    summary.sort(key=lambda e: -abs(e["slope_pred"]))
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    with (out / "per_case.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        for row in summary:
            w.writerow(
                {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}
            )
    with (out / "frames.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0]))
        w.writeheader()
        for row in sorted(recs, key=lambda r: (r["case_id"], r["S_px"])):
            w.writerow(
                {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}
            )

    def med(key):
        vals = [e[key] for e in summary if np.isfinite(e[key])]
        return float(np.median(vals)) if vals else float("nan")

    print(f"\n纳入病例 {len(summary)}（≥{args.min_frames} 帧，p/W 变化 ≥{args.min_ratio:.2f}×）")
    print(f"{'case':26s} {'n':>3} {'a比':>5} {'真值':>5} {'预测跨度':>8} "
          f"{'斜率pred':>8} {'斜率β':>7} {'斜率直接':>8} {'斜率q':>6} {'门控':>5}")
    for e in summary[:20]:
        print(f"{e['case_id'][:26]:26s} {e['n']:>3} {e['a_ratio']:>5.2f} "
              f"{e['truth']:>5.1f} {e['pred_std']:>8.2f} "
              f"{e['slope_pred']:>8.2f} {e['slope_beta']:>7.2f} "
              f"{e['slope_direct']:>8.2f} {e['slope_q']:>6.2f} {e['gate_mean']:>5.2f}")

    if small:
        print(f"\n<3mm 病例（{len(small)} 例）：")
        print(f"{'case':26s} {'n':>3} {'a比':>5} {'真值':>5} "
              f"{'斜率pred':>8} {'斜率β':>7} {'斜率直接':>8} {'门控':>5}")
        for e in sorted(small, key=lambda x: -abs(x["slope_pred"])):
            print(f"{e['case_id'][:26]:26s} {e['n']:>3} {e['a_ratio']:>5.2f} "
                  f"{e['truth']:>5.1f} {e['slope_pred']:>8.2f} {e['slope_beta']:>7.2f} "
                  f"{e['slope_direct']:>8.2f} {e['gate_mean']:>5.2f}")

    def med_of(items, key):
        vals = [e[key] for e in items if np.isfinite(e[key])]
        return float(np.median(vals)) if vals else float("nan")

    agg = {
        "n_cases": len(summary),
        "median_slope_pred": med("slope_pred"),
        "median_slope_beta": med("slope_beta"),
        "median_slope_direct": med("slope_direct"),
        "median_slope_q": med("slope_q"),
        "median_slope_beta_w": med("slope_beta_w"),
        "median_identity_residual": med("identity_residual"),
        "median_abs_drift_mm": float(np.median([abs(e["drift_mm"]) for e in summary])),
        "median_pred_std": med("pred_std"),
        "n_small_cases": len(small),
        "median_slope_pred_small": med_of(small, "slope_pred"),
        "median_slope_beta_small": med_of(small, "slope_beta"),
        "median_slope_direct_small": med_of(small, "slope_direct"),
        "note": "斜率 = d log(pred)/d log(a)，a=p/W；恒等式 b_pred_β = b_(βW) + b_q + 1",
    }
    print("\n汇总（病例等权中位数）：")
    for k, v in agg.items():
        print(f"  {k}: {v if isinstance(v, str) else round(v, 3)}")
    (out / "summary.json").write_text(
        json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nCSV/JSON: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

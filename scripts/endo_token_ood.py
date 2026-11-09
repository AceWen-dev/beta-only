#!/usr/bin/env python3
"""Endo vs DA：B 上的深度 token / β 头输入是否 OOD（只读诊断，2026-11-06）。

三层证据：
1. 原始深度 token 的病灶结构（框内 vs 环形背景、对比度、梯度）；
2. depth_net 64 维嵌入与 β 头 128 维输入的 MMD / 线性探针 AUC；
3. B 帧的 OOD 分数与绝对误差的相关、按 OOD 四分位的 MAE。

不训练、不改数据、不写 run；只输出报告到 output/reports/endo_token_ood_20261106/。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/Betaregression_data")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from core.config import ModelConfig  # noqa: E402
from core.data.datasets.v11_frames import V11FramesDataset, _crop_box  # noqa: E402
from core.models.v11 import V11Net  # noqa: E402
import plot_per_bin as pp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STRUCT_KEYS = ("prom", "prom_rel", "prom_mad", "mad_rel", "range_rel",
               "grad_rel", "les_med", "valid", "clip")

SOURCES = {
    "endo": {
        "run": "output/runs/beta-regression/v11-company-betaonly-endo/"
               "20261106-172028_a24a260b_s20261020_663041",
        "cohorts": (
            ("company_test", "company-v1-endo", "test_company"),
            ("B", "b-r-endo", "test_b"),
            ("R", "b-r-endo", "test_r"),
        ),
    },
    "da": {
        "run": "output/runs/beta-regression/v11-company-diranchor40-betaonly/"
               "20261106-032421_9659caf2_s20261020_e4fafe",
        "cohorts": (
            ("company_test", "company-v1", "test_company"),
            ("B", "single-v2", "test_b"),
            ("R", "single-v2", "test_r"),
        ),
    },
}
COLORS = {"company_test": "#3572b0", "B": "#e07b39", "R": "#3a9d5d"}


# ---------------------------------------------------------------- raw tokens
def _view_masks(bbox, content_hw, view: int, size: int = 128):
    h, w = content_hw
    if view == 0:
        x1, y1, x2, y2 = bbox
        ch, cw = h, w
    else:
        cx1, cy1, cx2, cy2 = _crop_box(bbox, w, h, 1.25)
        x1, y1 = bbox[0] - cx1, bbox[1] - cy1
        x2, y2 = bbox[2] - cx1, bbox[3] - cy1
        ch, cw = cy2 - cy1, cx2 - cx1
    if ch <= 0 or cw <= 0:
        return None
    ratio = min(size / ch, size / cw)
    rh, rw = max(1, round(ch * ratio)), max(1, round(cw * ratio))
    top, left = (size - rh) // 2, (size - rw) // 2
    rect = np.zeros((size, size), bool)
    ix1 = max(0, int(np.floor(left + x1 * ratio)))
    ix2 = min(size, int(np.ceil(left + x2 * ratio)))
    iy1 = max(0, int(np.floor(top + y1 * ratio)))
    iy2 = min(size, int(np.ceil(top + y2 * ratio)))
    if ix2 - ix1 < 2 or iy2 - iy1 < 2:
        return None
    rect[iy1:iy2, ix1:ix2] = True
    pad = max(2, int(round(0.5 * min(ix2 - ix1, iy2 - iy1))))
    kernel = np.ones((2 * pad + 1, 2 * pad + 1), np.uint8)
    ring = cv2.dilate(rect.astype(np.uint8), kernel) > 0
    ring &= ~rect
    return rect, ring


def structure_stats(depth: np.ndarray, bbox, content_hw) -> dict:
    """depth: [views, 2, 128, 128]；通道 0=归一化深度，1=有效掩码。"""
    acc = defaultdict(list)
    for v in range(depth.shape[0]):
        d = depth[v, 0]
        valid = depth[v, 1] > 0.5
        masks = _view_masks(bbox, content_hw, v)
        if masks is None:
            continue
        rect0, ring0 = masks
        rect = rect0 & valid
        ring = ring0 & valid
        if rect.sum() < 4 or ring.sum() < 8:
            continue
        dv, rv = d[rect], d[ring]
        med = float(np.median(dv))
        rmed = float(np.median(rv))
        mad = float(np.median(np.abs(dv - med)))
        p10, p90 = np.percentile(dv, [10, 90])
        gy, gx = np.gradient(d)
        grad = float(np.mean(np.hypot(gx[rect], gy[rect])))
        eps = max(med, 1e-3)
        acc["prom"].append(rmed - med)
        acc["prom_rel"].append((rmed - med) / eps)
        acc["prom_mad"].append((rmed - med) / max(mad, 1e-3))
        acc["mad_rel"].append(mad / eps)
        acc["range_rel"].append((p90 - p10) / eps)
        acc["grad_rel"].append(grad / eps)
        acc["les_med"].append(med)
        acc["valid"].append(float(rect.sum() / max(rect0.sum(), 1)))
        acc["clip"].append(float(np.mean(dv >= 3.99)))
    if not acc:
        return {k: float("nan") for k in STRUCT_KEYS}
    return {k: float(np.mean(acc[k])) for k in STRUCT_KEYS}


# ---------------------------------------------------------------- inference
def _collate(items):
    keys = ("rgb_views", "depth", "scalars", "D", "S", "S_norm", "W", "y")
    out = {k: torch.stack([x[k] for x in items]) for k in keys}
    out["case_id"] = [x["case_id"] for x in items]
    out["sample_id"] = [x["sample_id"] for x in items]
    return out


def run_source(source: str, cfg: dict, out: Path, max_frames: int = 0):
    run = (ROOT / cfg["run"]).resolve()
    resolved = json.loads((run / "config.resolved.json").read_text())
    views = int(resolved["data"].get("views", 2))
    size = int(resolved["data"].get("input_size", 224))
    net = V11Net(ModelConfig.from_dict(resolved["model"]))
    ck = torch.load(run / "checkpoints/final_ema.pt", map_location="cpu",
                    weights_only=False)
    net.load_state_dict(ck.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device).eval()
    captured = []
    handle = net.head_beta[0].register_forward_hook(
        lambda _m, inp, _o: captured.append(inp[0].detach())
    )

    rows, deps, feats, sids, cids, cohs, truths, preds = [], [], [], [], [], [], [], []
    for cohort, version, split in cfg["cohorts"]:
        manifest = json.loads(
            (DATA / f"curated/v11_frames/{version}/manifest.json").read_text()
        )
        rows_m = [r for r in manifest["rows"]
                  if r["split"] == split and float(r["truth_mm"]) < 10.0]
        if max_frames:
            rows_m = rows_m[:max_frames]
        meta = {
            str(r["sample_id"]): ([float(v) for v in r["bbox_content"]],
                                  (int(r["content_size"][0]),
                                   int(r["content_size"][1])))
            for r in rows_m
        }
        ds = V11FramesDataset(rows_m, DATA, views, None, input_size=size)
        loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2,
                            collate_fn=_collate)
        print(f"[{source}] {cohort}: {len(rows_m)} frames", flush=True)
        with torch.inference_mode():
            for b in loader:
                captured.clear()
                gpu = {k: (v.to(device) if torch.is_tensor(v) else v)
                       for k, v in b.items()}
                pred, _ = net(gpu)
                feat = captured[0].cpu().numpy()
                bsz, nv, c, hd, wd = gpu["depth"].shape
                dep = net.depth_net(
                    gpu["depth"].reshape(bsz * nv, c, hd, wd)
                ).reshape(bsz, nv, 64).mean(dim=1).cpu().numpy()
                for j in range(bsz):
                    sid = b["sample_id"][j]
                    bbox, chw = meta[sid]
                    st = structure_stats(b["depth"][j].numpy(), bbox, chw)
                    rows.append({
                        "source": source, "cohort": cohort,
                        "sample_id": sid, "case_id": b["case_id"][j],
                        "truth": float(b["y"][j]),
                        "pred": float(pred[j].cpu()),
                        "abs_err": float(abs(pred[j].cpu() - b["y"][j])),
                        **{f"st_{k}": st[k] for k in STRUCT_KEYS},
                    })
                deps.append(dep)
                feats.append(feat)
                sids.extend(b["sample_id"])
                cids.extend(b["case_id"])
                cohs.extend([cohort] * bsz)
                truths.append(b["y"].numpy())
                preds.append(pred.cpu().numpy())
        print(f"  done {cohort} ({len(rows)} total)", flush=True)
    handle.remove()
    del net
    torch.cuda.empty_cache()
    np.savez_compressed(
        out / f"emb_{source}.npz",
        dep=np.concatenate(deps), feat=np.concatenate(feats),
        sample_id=np.array(sids, dtype=str),
        case_id=np.array(cids, dtype=str),
        cohort=np.array(cohs, dtype=str),
        truth=np.concatenate(truths), pred=np.concatenate(preds),
    )
    (out / f"rows_{source}.json").write_text(
        json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return rows


# ---------------------------------------------------------------- analysis
def _rank(x):
    order = np.argsort(x)
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    return r


def spearman(a, b):
    if len(a) < 3:
        return float("nan")
    return float(np.corrcoef(_rank(np.asarray(a)), _rank(np.asarray(b)))[0, 1])


def _sqdist(x, y):
    return (x * x).sum(1)[:, None] + (y * y).sum(1)[None, :] - 2 * x @ y.T


def mmd2(x, y, seed=0):
    rng = np.random.default_rng(seed)
    x = x[rng.permutation(len(x))]
    y = y[rng.permutation(len(y))]
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    z = np.concatenate([x, y])
    d2 = _sqdist(z, z)
    tri = d2[np.triu_indices(len(z), 1)]
    sigma2 = float(np.median(tri)) or 1.0
    kxx = np.exp(-_sqdist(x, x) / (2 * sigma2))
    kyy = np.exp(-_sqdist(y, y) / (2 * sigma2))
    kxy = np.exp(-_sqdist(x, y) / (2 * sigma2))
    return float(kxx.mean() + kyy.mean() - 2 * kxy.mean())


def _case_folds(cases, seed=0):
    uniq = sorted(set(cases))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    half = len(uniq) // 2
    f0 = set(uniq[:half])
    return np.array([c in f0 for c in cases])


def ridge_auc(x, y, fold, lam=1.0):
    aucs = []
    for f in (0, 1):
        tr, te = fold != f, fold == f
        if tr.sum() < 10 or te.sum() < 10 or len(set(y[tr])) < 2:
            continue
        mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-8
        xtr, xte = (x[tr] - mu) / sd, (x[te] - mu) / sd
        yc = y[tr].astype(float) - 0.5
        w = np.linalg.solve(xtr.T @ xtr + lam * np.eye(xtr.shape[1]),
                            xtr.T @ yc)
        s = xte @ w
        yt = y[te]
        n1, n0 = int(yt.sum()), int((1 - yt).sum())
        if n1 == 0 or n0 == 0:
            continue
        auc = (np.sum(_rank(s)[yt == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
        aucs.append(float(auc))
    return float(np.mean(aucs)) if aucs else float("nan")


def _mmd_one(z, a, b, n, seeds=range(5)):
    vals = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        aa = rng.choice(a, min(n, len(a)), replace=False)
        bb = rng.choice(b, min(n, len(b)), replace=False)
        vals.append(mmd2(z[aa], z[bb], seed))
    return float(np.mean(vals))


def _probe(z, cohort, target, cases):
    mask = (cohort == "company_test") | (cohort == target)
    y = (cohort[mask] == target).astype(int)
    fold = _case_folds(cases[mask], seed=1)
    return ridge_auc(z[mask], y, fold)


def _baseline_auc(z, cohort, cases):
    mask = cohort == "company_test"
    # 标签与 train/test 折独立，避免探针“预测自己的折”
    label = _case_folds(cases[mask], seed=2).astype(int)
    fold = _case_folds(cases[mask], seed=3)
    return ridge_auc(z[mask], label, fold)


def _ood(z, cohort):
    comp = z[cohort == "company_test"]
    mu, sd = comp.mean(0), comp.std(0) + 1e-8
    return np.sqrt(np.mean(((z - mu) / sd) ** 2, axis=1))


def _quartile_mae(score, err, n=4):
    idx = np.argsort(score)
    chunks = np.array_split(idx, n)
    return [float(np.mean(err[c])) if len(c) else float("nan") for c in chunks]


def analyse(source: str, out: Path) -> dict:
    rows = json.loads((out / f"rows_{source}.json").read_text())
    z = np.load(out / f"emb_{source}.npz", allow_pickle=False)
    cohort, cases = z["cohort"], z["case_id"]
    truth, pred = z["truth"], z["pred"]
    err = np.abs(pred - truth)
    dep, feat = z["dep"], z["feat"]

    # 用 company 标准化后再比较分布
    comp = cohort == "company_test"
    mu, sd = dep[comp].mean(0), dep[comp].std(0) + 1e-8
    zd = (dep - mu) / sd
    mu, sd = feat[comp].mean(0), feat[comp].std(0) + 1e-8
    zf = (feat - mu) / sd

    # case-level 折
    fold_dep = _case_folds(cases[comp], seed=0)
    cmp_idx = np.where(comp)[0]
    half0, half1 = cmp_idx[fold_dep], cmp_idx[~fold_dep]
    n_base = min(200, len(half0), len(half1))

    outd = {"mmd": {}, "probe": {}, "ood": {}}
    b_idx = np.where(cohort == "B")[0]
    r_idx = np.where(cohort == "R")[0]
    n_common = min(200, len(half0), len(half1), len(b_idx), len(r_idx))
    for name, zz in (("dep", zd), ("feat", zf)):
        base = _mmd_one(zz, half0, half1, n_common)
        outd["mmd"][name] = {
            "baseline": base,
            "B": _mmd_one(zz, cmp_idx, b_idx, n_common),
            "R": _mmd_one(zz, cmp_idx, r_idx, n_common),
        }
        outd["probe"][name] = {
            "baseline": _baseline_auc(zz, cohort, cases),
            "B": _probe(zz, cohort, "B", cases),
            "R": _probe(zz, cohort, "R", cases),
        }
        ood = _ood(zz, cohort)
        bmask = cohort == "B"
        outd["ood"][name] = {
            "company_med": float(np.median(ood[comp])),
            "B_med": float(np.median(ood[bmask])),
            "R_med": float(np.median(ood[cohort == "R"])),
            "B_spearman": spearman(ood[bmask], err[bmask]),
            "B_quartile_mae": _quartile_mae(ood[bmask], err[bmask]),
        }

    struct = {}
    for c in ("company_test", "B", "R"):
        sel = [r for r in rows if r["cohort"] == c]
        struct[c] = {}
        for k in STRUCT_KEYS:
            v = np.array([r[f"st_{k}"] for r in sel], float)
            v = v[np.isfinite(v)]
            if v.size:
                struct[c][k] = {
                    "median": float(np.median(v)),
                    "p25": float(np.percentile(v, 25)),
                    "p75": float(np.percentile(v, 75)),
                    "n": int(v.size),
                }
    outd["struct"] = struct

    ood_d_all, ood_f_all = _ood(zd, cohort), _ood(zf, cohort)
    with (out / f"ood_{source}.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cohort", "sample_id", "case_id", "truth", "pred",
                    "abs_err", "ood_dep", "ood_feat"])
        for i in range(len(cohort)):
            w.writerow([cohort[i], z["sample_id"][i], cases[i], truth[i],
                        pred[i], err[i], ood_d_all[i], ood_f_all[i]])
    return outd


# ---------------------------------------------------------------- plot
def _box(ax, values, positions, colors):
    if isinstance(colors, str):
        colors = [colors] * len(values)
    bp = ax.boxplot(values, positions=positions, widths=0.55,
                    patch_artist=True, showfliers=False, whis=(5, 95))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    for med in bp["medians"]:
        med.set_color("black")
        med.set_linewidth(1.4)


def draw(summary: dict, out: Path):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    # A/B：原始深度结构
    for ax, key, title in (
        (axes[0, 0], "prom_mad", "病灶突出度 / 局部 MAD（越大越像隆起）"),
        (axes[0, 1], "mad_rel", "病灶内深度对比度 MAD/median"),
    ):
        vals, pos, cols, labels = [], [], [], []
        p = 1
        for source in ("endo", "da"):
            for c in ("company_test", "B", "R"):
                v = [r for r in json.loads((out / f"rows_{source}.json").read_text())
                     if r["cohort"] == c]
                arr = np.array([r[f"st_{key}"] for r in v], float)
                vals.append(arr[np.isfinite(arr)])
                pos.append(p)
                cols.append(COLORS[c])
                labels.append(f"{source.upper()}\n{c}" if c == "company_test" else c)
                p += 1
            p += 0.6
        _box(ax, vals, pos, cols)
        ax.set_xticks(pos)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_title(title, fontsize=10)
        ax.axvline(3.8, color="#bbb", lw=1, ls="--")

    # C/D：MMD
    for ax, key, title in (
        (axes[0, 2], "dep", "depth_net 64-d 嵌入 MMD²"),
        (axes[1, 0], "feat", "β 头输入 128-d MMD²"),
    ):
        x = np.arange(2)
        w = 0.26
        for j, (name, color) in enumerate((("baseline", "#9e9e9e"),
                                           ("B", COLORS["B"]), ("R", COLORS["R"]))):
            vals = [summary[s]["mmd"][key][name] for s in ("endo", "da")]
            ax.bar(x + (j - 1) * w, vals, w, color=color, alpha=0.8, label=name)
            for xi, v in zip(x + (j - 1) * w, vals):
                ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(["Endo", "DA"])
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7)

    # E：B 上 OOD vs 误差
    for source, color in (("endo", "#3572b0"), ("da", "#e07b39")):
        rows = json.loads((out / f"rows_{source}.json").read_text())
        ood = np.load(out / f"emb_{source}.npz", allow_pickle=False)
        cohort = ood["cohort"]
        zf = ood["feat"]
        comp = cohort == "company_test"
        mu, sd = zf[comp].mean(0), zf[comp].std(0) + 1e-8
        score = np.sqrt(np.mean(((zf - mu) / sd) ** 2, axis=1))
        b = cohort == "B"
        err = np.abs(ood["pred"] - ood["truth"])
        axes[1, 1].scatter(score[b], err[b], s=10, alpha=0.45, color=color,
                           label=f"{source.upper()} ρ={spearman(score[b], err[b]):+.2f}")
    axes[1, 1].set_xlabel("β 输入 OOD 分数（到 company 的 RMS z）")
    axes[1, 1].set_ylabel("B 帧 |误差| (mm)")
    axes[1, 1].set_title("B：OOD 分数 vs 误差", fontsize=10)
    axes[1, 1].legend(fontsize=8)

    # F：B 的 OOD 四分位 MAE
    x = np.arange(4)
    w = 0.36
    for j, (source, color) in enumerate((("endo", "#3572b0"), ("da", "#e07b39"))):
        vals = summary[source]["ood"]["feat"]["B_quartile_mae"]
        axes[1, 2].bar(x + (j - 0.5) * w, vals, w, color=color, alpha=0.8,
                       label=source.upper())
        for xi, v in zip(x + (j - 0.5) * w, vals):
            axes[1, 2].text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(["Q1", "Q2", "Q3", "Q4"])
    axes[1, 2].set_xlabel("OOD 四分位（Q4=最 OOD）")
    axes[1, 2].set_ylabel("B 帧 MAE (mm)")
    axes[1, 2].set_title("B：按 β 输入 OOD 分位的 MAE", fontsize=10)
    axes[1, 2].legend(fontsize=8)

    fig.suptitle("Endo vs DA：B 的深度 token / β 输入 OOD 诊断（truth <10mm）",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(out / "token_ood.png", dpi=170)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "output/reports/endo_token_ood_20261106"))
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for source, cfg in SOURCES.items():
        if (not args.max_frames
                and (out / f"rows_{source}.json").is_file()
                and (out / f"emb_{source}.npz").is_file()):
            print(f"[{source}] cache hit", flush=True)
        else:
            run_source(source, cfg, out, args.max_frames)

    summary = {s: analyse(s, out) for s in SOURCES}
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# Endo vs DA：B 的深度 token / β 输入 OOD 诊断", "",
             "只读诊断；truth <10mm；company_test 为参照域。", ""]
    for source in ("endo", "da"):
        s = summary[source]
        lines += [f"## {source.upper()}", "",
                  "| 指标 | baseline（company 内部） | company vs B | company vs R |",
                  "|---|---:|---:|---:|"]
        for name, label in (("dep", "MMD² depth_emb"),
                            ("feat", "MMD² β 输入")):
            m = s["mmd"][name]
            lines.append(f"| {label} | {m['baseline']:.4f} | {m['B']:.4f} | {m['R']:.4f} |")
        for name, label in (("dep", "AUC depth_emb"),
                            ("feat", "AUC β 输入")):
            p = s["probe"][name]
            lines.append(f"| {label} | {p['baseline']:.3f} | {p['B']:.3f} | {p['R']:.3f} |")
        o = s["ood"]["feat"]
        lines += ["", f"- β 输入 OOD 中位：company {o['company_med']:.2f} / "
                      f"B {o['B_med']:.2f} / R {o['R_med']:.2f}",
                  f"- B 上 OOD–误差 Spearman：{o['B_spearman']:+.2f}",
                  "- B 按 OOD 四分位 MAE："
                  + " / ".join(f"{v:.2f}" for v in o["B_quartile_mae"]), ""]
        lines += ["| 队列 | prom_mad | mad_rel | grad_rel | valid | clip |",
                  "|---|---:|---:|---:|---:|---:|"]
        for c in ("company_test", "B", "R"):
            st = s["struct"][c]
            def g(k):
                return st.get(k, {}).get("median", float("nan"))
            lines.append(f"| {c} | {g('prom_mad'):.3f} | {g('mad_rel'):.3f} | "
                         f"{g('grad_rel'):.3f} | {g('valid'):.3f} | {g('clip'):.3f} |")
        lines.append("")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    draw(summary, out)
    print(f"done: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

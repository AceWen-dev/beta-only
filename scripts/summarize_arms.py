#!/usr/bin/env python3
"""解压缩两臂（D-0067）与 company-base 三 seed 噪声底的对比汇总。

数字全部来自已落盘的 JSON，不手抄：
- `output/reports/v11-company-base-3seed_evalmode.json`  三 seed 离线逐档（噪声底）
- `output/reports/v11-company-<arm>_evalmode.json`       各臂离线逐档
- `output/runs/beta-regression/v11-company-<arm>-eval/*/evals/*.json`  官方公司 test eval
  （eval 记录只带 checkpoint 摘要，用 final_ema.pt 的 sha256 绑回具体 run）

判定口径（预登记在 D-0067，达标 = benchmark2 的 company + B 所有 <10mm 1mm 档 MAE < 2mm）：
- company：9–10 档 < 2.0，且其余档不差于 base 三 seed 的最大值
- B：<10mm 总 MAE 不差于 base 三 seed 的最大值
- R 只作回归观察，不参与判定

用法：
    python scripts/summarize_arms.py \
        --arm gatefloor=<run_dir> --arm diranchor=<run_dir> \
        --out output/reports/arms_20261105.md
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "output" / "reports"
BINS = tuple(range(2, 10))
COHORT_LABEL = {"company": "公司 test", "b": "PolypSize B", "r": "REAL-colon R"}
JUDGED = ("company", "b")
THRESHOLD = 2.0
DEFAULT_TITLE = "6. 解压缩两臂（D-0067）对比 company-base 三 seed 噪声底"


def jload(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def resolve(p: str | Path) -> Path:
    q = Path(p)
    return q if q.is_absolute() else (ROOT / q).resolve()


def rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def fmt(v, nd=3):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "—"
    return f"{v:.{nd}f}"


def sfmt(v, nd=3):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "—"
    return f"{v:+.{nd}f}"


def ema_block(raw, name):
    """从 *_evalmode.json 取 final_ema.pt 那一块；三 seed 版是 list → 逐 seed 返回。"""
    if not raw:
        return []
    block = raw.get("final_ema.pt")
    if isinstance(block, list):
        return [b for b in block if isinstance(b, dict)]
    if isinstance(block, dict):
        return [block]
    raise SystemExit(f"{name}: 未找到 final_ema.pt 段")


def mean_block(blocks: list[dict]) -> dict:
    """多 seed 的 *_evalmode.json 是 list → 逐队列/逐档取均值（lt10 同样取均值）。"""
    if len(blocks) == 1:
        return blocks[0]
    out: dict = {}
    for cohort in {c for b in blocks for c in b}:
        per_bin: dict[str, list[float]] = {}
        lt10s, maes, epochs = [], [], []
        for b in blocks:
            st = b.get(cohort) or {}
            for k, v in (st.get("per_bin") or {}).items():
                if v is not None and math.isfinite(v):
                    per_bin.setdefault(str(k), []).append(v)
            for key, acc in (("lt10", lt10s), ("mae", maes), ("epoch", epochs)):
                v = st.get(key)
                if v is not None and math.isfinite(v):
                    acc.append(v)
        mean = lambda xs: sum(xs) / len(xs) if xs else None   # noqa: E731
        out[cohort] = {"per_bin": {k: mean(v) for k, v in per_bin.items()},
                       "lt10": mean(lt10s), "mae": mean(maes), "epoch": mean(epochs)}
    return out


def cohort_stats(block: dict, cohort: str) -> dict:
    c = block.get(cohort) or {}
    per_bin = {int(k): v for k, v in (c.get("per_bin") or {}).items()}
    return {"lt10": c.get("lt10"), "mae": c.get("mae"), "per_bin": per_bin,
            "epoch": c.get("epoch")}


def noise_floor(blocks: list[dict], cohort: str) -> dict:
    """三 seed 的 mean/min/max：逐档 + lt10。"""
    bins: dict[int, list[float]] = {}
    lt10: list[float] = []
    for b in blocks:
        st = cohort_stats(b, cohort)
        for bn, v in st["per_bin"].items():
            if v is not None and math.isfinite(v):
                bins.setdefault(bn, []).append(v)
        if st["lt10"] is not None and math.isfinite(st["lt10"]):
            lt10.append(st["lt10"])

    def agg(vs):
        return {"mean": sum(vs) / len(vs), "min": min(vs), "max": max(vs), "n": len(vs)}

    return {"n": len(blocks),
            "per_bin": {bn: agg(vs) for bn, vs in bins.items()},
            "lt10": {"lt10": agg(lt10)} if lt10 else {}}


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def official_eval(run_dir: Path) -> dict | None:
    """按 checkpoint 摘要把官方 eval 记录绑回该 run 的 final_ema.pt。

    eval run 目录名是 `<experiment>-eval`，experiment 直接取 run 目录的父目录名，
    这样 chart 的 --name（可能是多 seed 均值图名）不影响绑定。
    """
    experiment = run_dir.parent.name
    ck = run_dir / "checkpoints" / "final_ema.pt"
    if not ck.is_file():
        return None
    want = sha256_file(ck)
    best = None
    for p in sorted(glob.glob(str(ROOT / "output/runs/beta-regression/"
                                  f"{experiment}-eval/*/evals/*.json"))):
        rec = jload(Path(p))
        if not rec or rec.get("split") != "test":
            continue
        if rec.get("checkpoint_digest") != want:
            continue
        m = rec.get("metrics") or {}
        best = {"lt10": m.get("v11/mae_lt10"), "worst_bin": m.get("v11/worst_bin"),
                "mae": m.get("v11/mae"), "record": str(Path(p).relative_to(ROOT))}
    return best


def bin_table(floor: dict, arm: dict) -> tuple[list[str], dict]:
    rows, verdict = [], {"over": [], "worse": [], "ok": []}
    for bn in BINS:
        f = floor["per_bin"].get(bn)
        a = arm["per_bin"].get(bn)
        if a is None:
            continue
        delta = None if not f else a - f["mean"]
        worse = bool(f) and a > f["max"] + 1e-9
        over = a >= THRESHOLD
        if over:
            verdict["over"].append(bn)
        if worse:
            verdict["worse"].append(bn)
        if not over and not worse:
            verdict["ok"].append(bn)
        mark = "❌" if over else ("⚠️" if worse else "✅")
        rows.append(f"| {bn}–{bn+1} | {fmt(f['mean'] if f else None)} "
                    f"[{fmt(f['min'] if f else None)}–{fmt(f['max'] if f else None)}] "
                    f"| {fmt(a)} | {sfmt(delta)} | {mark} |")
    return rows, verdict


def render(arms: dict, base_path: Path, steps_log: Path | None,
           title: str = DEFAULT_TITLE) -> str:
    base_path = resolve(base_path)
    base_raw = jload(base_path)
    base_blocks = ema_block(base_raw, str(base_path))
    if not base_blocks:
        raise SystemExit(f"噪声底缺失：{base_path}")

    L = [f"## {title}", ""]
    L.append(f"- 噪声底：`{rel(base_path)}`，{len(base_blocks)} 个 seed（final_ema.pt）")
    L.append(f"- 判定：company + B 的每个 <10mm 1mm 档 MAE < {THRESHOLD:.1f}mm；"
             "且不比 base 三 seed 的最大值更差。R 只作回归观察。")
    L.append("- ✅=达标且不劣于噪声底 ⚠️=达标但超出噪声底最大值 ❌=未达标（≥2mm）")
    L.append("")

    summary: list[tuple[str, dict]] = []
    for idx, (arm_name, (experiment, run_dir)) in enumerate(arms.items(), start=1):
        rd = resolve(run_dir)
        raw = jload(REPORTS / f"{experiment}_evalmode.json")
        blocks = ema_block(raw, experiment)
        L.append(f"### 6.{idx} `{arm_name}`（experiment `{experiment}`）")
        L.append("")
        L.append(f"- run：`{rel(rd)}`")
        L.append("")
        if not blocks:
            L.append("- **离线逐档 JSON 缺失**（训练或出图失败？）")
            L.append("")
            summary.append((arm_name, {"missing": True}))
            continue
        arm_block = mean_block(blocks)
        arm = {c: cohort_stats(arm_block, c) for c in ("company", "b", "r")}
        if len(blocks) > 1:
            L.append(f"- 该臂为 **{len(blocks)} 个 seed 的均值**（同名 chart JSON）")
            L.append("")
        rec: dict = {"verdict": {}, "lt10": {}}

        for cohort in ("company", "b", "r"):
            floor = noise_floor(base_blocks, cohort)
            rows, verdict = bin_table(floor, arm[cohort])
            judged = cohort in JUDGED
            fl = floor["lt10"].get("lt10", {})
            L.append(f"**{COHORT_LABEL[cohort]}**（{'参与判定' if judged else '仅回归观察'}）："
                     f"lt10 base {fmt(fl.get('mean'))} [{fmt(fl.get('min'))}–{fmt(fl.get('max'))}]"
                     f" → arm **{fmt(arm[cohort]['lt10'])}**")
            L.append("")
            L.append("| 档(mm) | base 三 seed 均值 [min–max] | arm | Δ均值 | |")
            L.append("|---|---:|---:|---:|:--|")
            L.extend(rows)
            L.append("")
            if judged:
                rec["verdict"][cohort] = verdict
                rec["lt10"][cohort] = (arm[cohort]["lt10"], fl)

        ev = official_eval(rd)
        rec["official"] = ev
        if ev:
            L.append(f"- 官方公司 test eval（`{ev['record']}`）：lt10 {fmt(ev['lt10'])}、"
                     f"worst_bin {fmt(ev['worst_bin'])}、mae {fmt(ev['mae'])}")
        else:
            L.append("- 官方公司 test eval：**缺失**")
        L.append("")
        summary.append((arm_name, rec))

    def bins_str(x):
        return "、".join(f"{i}–{i+1}" for i in x) if x else "无"

    L.append("### 6.x 判定")
    L.append("")
    L.append("| arm | company 未达标档 | company 超噪声底档 | B 未达标档 | B 超噪声底档 "
             "| B lt10（base 上限） | 官方 eval | 结论 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for arm_name, rec in summary:
        if rec.get("missing"):
            L.append(f"| `{arm_name}` | — | — | — | — | — | — | 缺证据，不判定 |")
            continue
        v = rec["verdict"]
        co, b = v.get("company", {}), v.get("b", {})
        blt10, bfloor = rec["lt10"].get("b", (None, {}))
        b_lt10_worse = (blt10 is not None and bfloor
                        and blt10 > bfloor.get("max", 0.0) + 1e-9)
        passed = (not co.get("over") and not co.get("worse")
                  and not b.get("over") and not b.get("worse") and not b_lt10_worse)
        ev = rec.get("official") or {}
        L.append(f"| `{arm_name}` | {bins_str(co.get('over', []))} | {bins_str(co.get('worse', []))} "
                 f"| {bins_str(b.get('over', []))} | {bins_str(b.get('worse', []))} "
                 f"| {fmt(blt10)}（{fmt(bfloor.get('max'))}） | {fmt(ev.get('lt10'))} "
                 f"| {'**PASS**' if passed else 'FAIL'} |")
    L.append("")
    L.append("> 单 seed 对比，PASS 只代表「不比噪声底差且逐档达标」；"
             "采纳前仍需 val 选择 + 第二个 seed 复核（AGENTS.md：不报单个幸运 seed）。")
    L.append("")

    steps_log = resolve(steps_log) if steps_log else None
    if steps_log and steps_log.is_file():
        tail = [ln for ln in steps_log.read_text(errors="replace").splitlines()
                if ln.startswith("[") and ("OK" in ln or "FAIL" in ln or "STEP" in ln)]
        if tail:
            L.append("### 6.y 执行日志（最后 20 条）")
            L.append("")
            L.append("```")
            L.extend(tail[-20:])
            L.append("```")
            L.append("")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=EXPERIMENT:RUN_DIR",
                    help="如 gatefloor=v11-company-gatefloor:output/runs/.../<run_id>")
    ap.add_argument("--base", default=str(REPORTS / "v11-company-base-3seed_evalmode.json"))
    ap.add_argument("--steps-log", default=str(ROOT / "output/logs/arms_20261105.steps.log"))
    ap.add_argument("--title", default=DEFAULT_TITLE,
                    help="小节标题（追加到报告时用同一名做幂等替换）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--append-to", default=None)
    a = ap.parse_args()

    arms: dict = {}
    for spec in a.arm:
        name, _, rest = spec.partition("=")
        exp, _, rd = rest.partition(":")
        if not name or not exp or not rd:
            raise SystemExit(f"--arm 格式应为 NAME=EXPERIMENT:RUN_DIR，收到 {spec!r}")
        arms[name] = (exp, rd)

    text = render(arms, Path(a.base), Path(a.steps_log) if a.steps_log else None, a.title)
    out = resolve(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"wrote {out}")

    if a.append_to:
        tgt = resolve(a.append_to)
        old = tgt.read_text() if tgt.is_file() else ""
        head = text.split("\n", 1)[0]
        if head in old:  # 幂等：重跑覆盖同名小节
            pre, _, post = old.partition(head)
            rest = post.split("\n## ", 1)
            old = pre.rstrip() + "\n\n" + text.rstrip() + ("\n\n## " + rest[1] if len(rest) > 1 else "\n")
        else:
            old = (old.rstrip() + "\n\n" if old.strip() else "") + text
        tgt.write_text(old)
        print(f"appended to {tgt}")


if __name__ == "__main__":
    main()

"""TensorBoard callback: an optional mirror, not the experiment record.

RunStore / ``metrics.jsonl`` / ``logs/train.log`` remain the source of truth.
This callback only writes scalars and the resolved config under
``<run_dir>/<tensorboard.subdir>/`` so TensorBoard can plot them.

If TensorBoard is not installed the callback prints one warning and becomes a
no-op; training continues. Set ``tensorboard.enabled: false`` to skip it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from dlef.contracts import CallbackBase, RunContext, StepOutput, to_float


def ensure_tb_link(run_dir: str | Path, tb_root: str | Path) -> Path | None:
    """在 TensorBoard 的 logdir 下为新 run 建符号链接（D-0037）。

    TB 以 `--logdir output/tb` 运行；链接名 `<experiment>__<HHMMSS>__s<seed>`。
    幂等：已存在且指向同一目录时直接返回。
    """
    import json

    run_dir = Path(run_dir)
    tb_dir = run_dir / "tensorboard"
    if not tb_dir.is_dir():
        return None
    try:
        m = json.loads((run_dir / "manifest.json").read_text())
    except (OSError, json.JSONDecodeError):
        m = {}
    run_id = m.get("run_id", run_dir.name)
    parts = run_id.split("_")
    stamp = parts[0].split("-")[-1] if "-" in parts[0] else parts[0]
    seed = next((p for p in parts if p.startswith("s") and p[1:].isdigit()), "")
    name = f"{m.get('experiment', run_dir.parent.name)}__{stamp}__{seed}"
    root = Path(tb_root)
    root.mkdir(parents=True, exist_ok=True)
    link = root / name
    try:
        if link.is_symlink() and Path(link.resolve()) == tb_dir.resolve():
            pass
        else:
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(tb_dir.resolve())
    except OSError:
        return None
    return link


class TensorBoardCallback(CallbackBase):
    def __init__(
        self,
        *,
        subdir: str = "tensorboard",
        flush_secs: int = 10,
        tb_root: str | Path | None = None,
    ) -> None:
        self.subdir = subdir
        self.flush_secs = flush_secs
        self.tb_root = Path(tb_root) if tb_root else None
        self.writer: Any | None = None

    def on_run_start(self, ctx: RunContext) -> None:
        if not ctx.run_dir:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            print(
                "warning: tensorboard is enabled but not installed; "
                "run `pip install tensorboard` or set tensorboard.enabled=false",
                file=sys.stderr,
            )
            return

        log_dir = Path(ctx.run_dir) / self.subdir
        log_dir.mkdir(parents=True, exist_ok=True)
        if self.tb_root is not None:
            # 训练一开始就建链接：训练途中也能在 TB 里看到本 run（D-0037）
            ensure_tb_link(ctx.run_dir, self.tb_root)
        self.writer = SummaryWriter(log_dir=str(log_dir), flush_secs=self.flush_secs)
        self.writer.add_text(
            "run/config",
            json.dumps(ctx.config.to_dict(), indent=2, ensure_ascii=False, default=str),
            0,
        )
        if ctx.manifest is not None:
            self.writer.add_text(
                "run/manifest",
                json.dumps(
                    ctx.manifest.to_dict(),
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                ),
                0,
            )

    def on_step_end(self, ctx: RunContext, step: StepOutput) -> None:
        if self.writer is None:
            return
        if step.loss is not None:
            self.writer.add_scalar("train/loss", to_float(step.loss), ctx.step)
        for name, value in step.terms.items():
            self.writer.add_scalar(f"train/{name}", to_float(value), ctx.step)

    def on_epoch_end(self, ctx: RunContext) -> None:
        if self.writer is None:
            return
        for name, value in ctx.metrics.items():
            try:
                self.writer.add_scalar(f"val/{name}", float(value), ctx.epoch)
            except (TypeError, ValueError):
                continue

    def on_run_end(self, ctx: RunContext) -> None:
        self._close()

    def on_exception(self, ctx: RunContext, error: BaseException) -> None:
        self._close()

    def _close(self) -> None:
        if self.writer is None:
            return
        self.writer.flush()
        self.writer.close()
        self.writer = None


def write_eval_scalars(
    run_dir: str | Path,
    metrics: Mapping[str, Any],
    *,
    prefix: str,
    subdir: str = "tensorboard",
    flush_secs: int = 10,
) -> bool:
    """把一次评估的指标写成 TensorBoard 标量（eval.py 调用）。

    标签形如 ``<prefix>/v11/mae_2_3``，step=0（每个 eval run 一个点）。
    TensorBoard 未安装时静默跳过，返回 False。
    """
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        return False
    log_dir = Path(run_dir) / subdir
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir), flush_secs=flush_secs)
    try:
        for name, value in metrics.items():
            try:
                writer.add_scalar(f"{prefix}/{name}", float(value), 0)
            except (TypeError, ValueError):
                continue
        writer.flush()
    finally:
        writer.close()
    return True


def write_benchmark_scalars(
    run_dir: str | Path,
    metrics: Mapping[str, Any],
    *,
    cohort: str,
    checkpoint: str | None = None,
    subdir: str = "tensorboard",
    flush_secs: int = 10,
) -> bool:
    """把一个 benchmark 的指标写成 TensorBoard 曲线（D-0030）。

    - ``benchmark/<cohort>/mae_by_bin``：x=step=1mm 档序号(0..9)，y=该档 MAE
      → TensorBoard 里是一条逐档曲线（与逐区间图同义）；
    - ``benchmark/<cohort>/summary/<metric>``：整体指标单点（mae/lt10/worst_bin/acc10）。
    TensorBoard 未安装时静默跳过。
    """
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        return False
    log_dir = Path(run_dir) / subdir
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir), flush_secs=flush_secs)
    try:
        bins: list[tuple[int, float]] = []
        for k in range(12):
            key = f"v11/mae_{k}_{k + 1}"
            value = metrics.get(key)
            if value is None:
                continue
            try:
                v = float(value)
            except (TypeError, ValueError):
                continue
            if v != v:  # NaN
                continue
            bins.append((k, v))
        base = f"benchmark/{cohort}" + (f"/{checkpoint}" if checkpoint else "")
        # 2mm 达标线：TB 原生不能画参考线，这里用一条常数曲线实现（D-0033）
        for k in range(12):
            writer.add_scalar("benchmark/reference/2mm", 2.0, k)
        for k, v in bins:
            writer.add_scalar(f"{base}/mae_by_bin", v, k)
        for name in ("v11/mae", "v11/mae_lt10", "v11/mae_ge10", "v11/worst_bin", "v11/acc10"):
            value = metrics.get(name)
            try:
                v = float(value)
            except (TypeError, ValueError):
                continue
            if v != v:
                continue
            writer.add_scalar(f"{base}/summary/{name.split('/', 1)[1]}", v, 0)
        # 自定义布局：把 B/D/R/clean 与 2mm 参考线放进同一张图
        try:
            writer.add_custom_scalars_multilinechart(
                [f"benchmark/{c}/*/mae_by_bin" for c in ("B", "D", "R", "clean")]
                + ["benchmark/reference/2mm", "*/*/mae_by_bin"],
                category="benchmark per-bin MAE",
                title="per-bin MAE vs 2mm line (B / D / R / clean)",
            )
        except Exception:
            pass
        writer.flush()
    finally:
        writer.close()
    return True


def write_chart_image(
    run_dir: str | Path,
    image_path: str | Path,
    *,
    tag: str = "chart/per_bin_mae",
    subdir: str = "tensorboard",
    flush_secs: int = 10,
) -> bool:
    """把一张 matplotlib PNG 写进 TensorBoard 的 Images 页（D-0031）。

    这样 TB 里既能看标量曲线，也能看到带 2mm 达标线、四面板布局的原图。
    """
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        return False
    path = Path(image_path)
    if not path.is_file():
        return False
    try:
        import cv2
        import numpy as np
    except ImportError:
        return False
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return False
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # TB 卡片约 600px 宽：超过 1600px 的图先等比缩小，避免"要么糊要么超出屏幕"
    max_w = 1600
    if img.shape[1] > max_w:
        scale = max_w / img.shape[1]
        img = cv2.resize(img, (max_w, int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    log_dir = Path(run_dir) / subdir
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir), flush_secs=flush_secs)
    try:
        writer.add_image(tag, img, 0, dataformats="HWC")
        writer.flush()
    finally:
        writer.close()
    return True

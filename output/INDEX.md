# output/ 总览索引

> 自动生成。训练与评估是**独立 run**（eval 记录绑定 checkpoint_digest 与 benchmark_id，不可移入训练目录）。
> 已清理：全部训练中途 checkpoint（`epoch-*.pt`）与失败 run 的 checkpoint；保留 best/best_ema/last 与全部记录。

| 实验 | run/seed | <10mm MAE (B/D) | 磁盘 | 评估目录 | 图 |
|---|---:|---|---:|---|---|
| `v11-backbone-frozen-best` | 1 (20261020) | B 2.847 / D 2.197 | 0.4G | `v11-backbone-frozen-best-eval` | ✓ |
| `v11-backbone-partial-best` | 1 (20261020) | B 1.413 / D 1.285 | 0.7G | `v11-backbone-partial-best-eval` | ✓ |
| `v11-baseline-best` | 3 (20261020) | B 1.680 / D 1.339 | 1.9G | `v11-baseline-best-eval` | ✓ |
| `v11-baseline-relloss` | 3 (20261020,20261021,20261022) | B 1.439 / D 1.220 | 2.9G | `v11-baseline-relloss-eval` | ✓ |
| `v11-baseline-scaleaug` | 1 (20261020) | — | 1.0G | `v11-baseline-scaleaug-eval` | — |
| `v11-baseline-shuffle` | 4 (20261020,20261021,20261022) | B 1.661 / D 1.252 | 2.9G | `v11-baseline-shuffle-eval` | ✓ |
| `v11-baseline-valmerge` | 3 (20261020,20261021,20261022) | B 1.428 / D 1.288 | 2.9G | `v11-baseline-valmerge-eval` | ✓ |
| `v11-dev3-hybrid` | 3 (20261020) | — | 2.9G | — | — |
| `v11-dev3-relloss` | 6 (20261020) | — | 5.3G | — | — |
| `v11-proj-best` | 1 (20261020) | B 1.648 / D 1.307 | 1.0G | `v11-proj-best-eval` | ✓ |
| `v11-ranksim-align` | 3 (20261020,20261021,20261022) | B 1.777 / D 1.399 | 2.9G | `v11-ranksim-align-eval` | ✓ |
| `v11-ranksim-align40` | 1 (20261020) | — | 0.0G | — | — |
| `v11-ranksim-best` | 1 (20261020) | — | 0.7G | `v11-ranksim-best-eval` | ✓ |
| `v11-ranksim-relloss` | 1 (20261020) | — | 2.9G | — | — |
| `v11-ranksim-repro` | 1 (20261020) | — | 1.0G | `v11-ranksim-repro-eval` | ✓ |

## 说明

- 指标口径：BN 重估（D-0023 修正）优先；valmerge 为 recorded（无污染）。
- `output/artifacts/`（105G）是 checkpoint 副本，由 `dlef gc` 按 30 天 TTL 回收。
- 清理标准流程：`make gc-dry` → 核对 → `--apply`。

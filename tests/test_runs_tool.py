from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from dlef.eval import EvalRecord
from dlef.run import ArtifactStore, RunStore

from core.tools import runs as runs_tool


class RunsToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.runs_root = root / "runs"
        store = RunStore(self.runs_root, ArtifactStore(root / "artifacts"))
        self.runs = []
        for experiment, seed, tags, metric in (
            ("baseline", 1, {"baseline": "true"}, 0.5),
            ("baseline", 2, {}, 0.3),
            ("ablation", 1, {"debug": "true"}, None),
        ):
            run = store.create(
                project="p",
                experiment=experiment,
                config_digest="a" * 64,
                seed=seed,
                tags=tags,
                config_source=f"input/configs/experiments/{experiment}.yaml",
            )
            run.manifest.status = "completed"
            run.write_manifest()
            if metric is not None:
                run.save_json(
                    "evals/eval.json",
                    EvalRecord(
                        eval_id=f"{experiment}-{seed}-val",
                        split="val",
                        checkpoint_digest="checkpoint",
                        data_version="data-v1",
                        eval_spec_digest="spec",
                        metrics={"mae": metric},
                        created_at="2026-10-30T00:00:00+00:00",
                    ).to_dict(),
                )
            self.runs.append(run)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _call(self, *args: str) -> dict:
        output = io.StringIO()
        previous = sys.argv
        sys.argv = ["runs.py", "--runs", str(self.runs_root), *args]
        try:
            with redirect_stdout(output):
                code = runs_tool.main()
        finally:
            sys.argv = previous
        self.assertEqual(code, 0)
        return json.loads(output.getvalue())

    def test_summary_groups_by_experiment(self) -> None:
        payload = self._call("--json")
        self.assertEqual(payload["view"], "summary")
        self.assertEqual(payload["total"], 2)
        rows = {row["experiment"]: row for row in payload["rows"]}
        self.assertEqual(rows["baseline"]["runs"], 2)
        self.assertEqual(rows["baseline"]["best_metric"], 0.3)
        self.assertEqual(rows["ablation"]["runs"], 1)
        self.assertEqual(rows["ablation"]["best_metric"], None)

    def test_experiment_switches_to_run_view(self) -> None:
        payload = self._call("--experiment", "baseline", "--json")
        self.assertEqual(payload["view"], "runs")
        self.assertEqual(payload["total"], 2)
        self.assertEqual(payload["rows"][0]["experiment"], "baseline")

    def test_tag_filter(self) -> None:
        payload = self._call("--tag", "baseline", "--json")
        self.assertEqual(payload["view"], "summary")
        self.assertEqual(payload["total"], 1)
        self.assertEqual(payload["rows"][0]["experiment"], "baseline")
        self.assertEqual(payload["rows"][0]["runs"], 1)

    def test_all_limit(self) -> None:
        payload = self._call("--all", "--limit", "1", "--json")
        self.assertEqual(payload["view"], "runs")
        self.assertEqual(payload["total"], 3)
        self.assertEqual(payload["shown"], 1)


if __name__ == "__main__":
    unittest.main()

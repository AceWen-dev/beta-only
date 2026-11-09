from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    from core.data.paths import (
        curated_dir,
        data_root,
        split_dir,
        split_manifest_path,
    )

    PATHS_AVAILABLE = True
except ImportError:
    PATHS_AVAILABLE = False


@unittest.skipUnless(PATHS_AVAILABLE, "project dependencies are not installed")
class DataPathTests(unittest.TestCase):
    def test_data_root_requires_env(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DLEF_DATA_ROOT", None)
            with self.assertRaises(RuntimeError):
                data_root()

    def test_standard_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"DLEF_DATA_ROOT": tmp}):
                root = Path(tmp).resolve()
                self.assertEqual(data_root(), root)
                self.assertEqual(
                    curated_dir("mydata", "v1"),
                    root / "curated" / "mydata" / "v1",
                )
                self.assertEqual(
                    split_dir("mydata", "v1"),
                    root / "splits" / "mydata" / "v1",
                )
                self.assertEqual(
                    split_manifest_path("mydata", "v1", "train"),
                    root / "splits" / "mydata" / "v1" / "train.json",
                )


if __name__ == "__main__":
    unittest.main()

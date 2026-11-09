from __future__ import annotations

import unittest

try:
    import torch

    from dlef.config import ConfigError
    from core.config import DataConfig
    from core.data import make_loaders

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
class DataLoaderWorkerTests(unittest.TestCase):
    def test_num_workers_validation(self) -> None:
        DataConfig(num_workers=4).validate()
        with self.assertRaises(ConfigError):
            DataConfig(num_workers=-1).validate()
        with self.assertRaises(ConfigError):
            DataConfig(num_workers=33).validate()

    def test_workers_and_pin_memory_keep_batches(self) -> None:
        base = DataConfig(n_train=16, n_val=4, n_test=2, batch_size=4)
        parallel = DataConfig(
            n_train=16, n_val=4, n_test=2, batch_size=4,
            num_workers=2, pin_memory=True,
        )
        base.validate()
        parallel.validate()

        sequential_loader, _ = make_loaders(base, seed=7)
        parallel_loader, _ = make_loaders(parallel, seed=7)
        self.assertEqual(parallel_loader.loader.num_workers, 2)
        self.assertEqual(len(list(sequential_loader)), len(list(parallel_loader)))

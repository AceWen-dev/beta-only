from __future__ import annotations

from dataclasses import dataclass, field

from dlef.config import ConfigError, ConfigSchema, TrainConfig


@dataclass
class DataConfig(ConfigSchema):
    name: str = "regression"
    version: str = "synthetic"
    seed: int = 7
    split: str = "holdout"
    n_train: int = 512
    n_val: int = 128
    n_test: int = 128
    noise: float = 0.1
    batch_size: int = 32
    train_transform: str = "identity"
    eval_transform: str = "identity"
    views: int = 2
    test_set: str = "b"
    max_samples: int = 0

    def validate(self) -> None:
        if self.name not in {"regression", "v11_frames"}:
            raise ConfigError(f"unknown data.name: {self.name!r}")
        if not self.version.strip():
            raise ConfigError("data.version must not be empty")
        if self.seed < 0:
            raise ConfigError("data.seed must be >= 0")
        if self.split not in {"holdout", "manifest"}:
            raise ConfigError(f"unknown data.split: {self.split!r}")
        if self.train_transform not in {"identity"}:
            raise ConfigError(
                f"unknown data.train_transform: {self.train_transform!r}"
            )
        if self.eval_transform not in {"identity"}:
            raise ConfigError(
                f"unknown data.eval_transform: {self.eval_transform!r}"
            )
        if self.n_train <= 0 or self.n_val <= 0 or self.n_test <= 0:
            raise ConfigError("n_train/n_val/n_test must be > 0")
        if self.batch_size <= 0:
            raise ConfigError("batch_size must be > 0")
        if self.noise < 0:
            raise ConfigError("noise must be >= 0")
        if self.views not in {2, 3}:
            raise ConfigError("views must be 2 or 3")
        if self.test_set not in {"b", "d", "r", "clean"}:
            raise ConfigError(f"unknown test_set: {self.test_set!r}")
        if self.max_samples < 0:
            raise ConfigError("max_samples must be >= 0")


@dataclass
class ModelConfig(ConfigSchema):
    name: str = "mlp"
    hidden: int = 16
    views: int = 2
    d_model: int = 64
    heads: int = 4
    init_checkpoint: str = ""
    proj: bool = False
    backbone: str = "full"
    beta_ref: float = 0.07543606728391815
    truth_median: float = 5.0
    k_bound: float = 2.06

    def validate(self) -> None:
        if self.name not in {"mlp", "linear", "v11"}:
            raise ConfigError(f"unknown model.name: {self.name!r}")
        if self.hidden <= 0:
            raise ConfigError("hidden must be > 0")
        if self.views not in {2, 3}:
            raise ConfigError("model.views must be 2 or 3")
        if self.d_model <= 0 or self.heads <= 0:
            raise ConfigError("d_model/heads must be > 0")
        if self.backbone not in {"full", "partial", "frozen"}:
            raise ConfigError(f"unknown model.backbone: {self.backbone!r}")
        if self.beta_ref <= 0 or self.truth_median <= 0 or self.k_bound <= 0:
            raise ConfigError("beta_ref/truth_median/k_bound must be > 0")


@dataclass
class LossConfig(ConfigSchema):
    name: str = "mse"
    huber_beta: float = 1.0
    delta: float = 1.0
    rank_gamma: float = 1.0
    rank_interp: float = 1.0

    def validate(self) -> None:
        if self.name not in {"mse", "huber", "mm_case_bin", "mm_rank"}:
            raise ConfigError(f"unknown loss.name: {self.name!r}")
        if self.huber_beta <= 0:
            raise ConfigError("huber_beta must be > 0")
        if self.delta <= 0:
            raise ConfigError("loss.delta must be > 0")
        if self.rank_gamma < 0 or self.rank_interp < 0:
            raise ConfigError("rank_gamma/rank_interp must be >= 0")


@dataclass
class OptimConfig(ConfigSchema):
    name: str = "adamw"
    lr: float = 1e-2
    lr_backbone: float = 0.0
    weight_decay: float = 0.0
    momentum: float = 0.0

    def validate(self) -> None:
        if self.name not in {"adamw", "adam", "sgd", "v11_adamw"}:
            raise ConfigError(f"unknown optim.name: {self.name!r}")
        if self.lr <= 0:
            raise ConfigError("lr must be > 0")
        if self.lr_backbone < 0:
            raise ConfigError("lr_backbone must be >= 0")
        if self.weight_decay < 0:
            raise ConfigError("weight_decay must be >= 0")
        if self.momentum < 0:
            raise ConfigError("momentum must be >= 0")


@dataclass
class SchedulerConfig(ConfigSchema):
    name: str = "cosine"
    t_max: int | None = None
    eta_min: float = 0.0

    def validate(self) -> None:
        if self.name not in {"cosine", "constant"}:
            raise ConfigError(f"unknown scheduler.name: {self.name!r}")
        if self.t_max is not None and self.t_max < 1:
            raise ConfigError("scheduler.t_max must be >= 1 when set")
        if self.eta_min < 0:
            raise ConfigError("scheduler.eta_min must be >= 0")


@dataclass
class TensorBoardConfig(ConfigSchema):
    enabled: bool = True
    subdir: str = "tensorboard"
    flush_secs: int = 10

    def validate(self) -> None:
        if not self.subdir.strip():
            raise ConfigError("tensorboard.subdir must not be empty")
        if self.flush_secs < 1:
            raise ConfigError("tensorboard.flush_secs must be >= 1")


@dataclass
class BenchmarkConfig(ConfigSchema):
    name: str = "synthetic-test-v1"
    label_version: str = "v1"
    evaluator_version: str = "1"
    aggregation: str = "mean"
    metric_versions: dict[str, str] = field(default_factory=lambda: {"mae": "v1"})
    thresholds: dict[str, float] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.name.strip():
            raise ConfigError("benchmark.name must not be empty")
        if not self.metric_versions:
            raise ConfigError("benchmark.metric_versions must not be empty")
        for name, version in self.metric_versions.items():
            if not name or not version:
                raise ConfigError(
                    "benchmark.metric_versions keys/values must not be empty"
                )
        for name in self.thresholds:
            if not name:
                raise ConfigError("benchmark.thresholds keys must not be empty")


@dataclass
class ProjectConfig(TrainConfig):
    # Workspace path: where runs/artifacts/sweeps are written. Relative paths
    # resolve against the project root, not the current working directory.
    output_root: str = "output"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    tensorboard: TensorBoardConfig = field(default_factory=TensorBoardConfig)
    metrics: list[str] = field(default_factory=lambda: ["mae"])
    amp: bool = False
    grad_accum_steps: int = 1
    grad_clip: float | None = None

    def validate(self) -> None:
        super().validate()
        if not self.output_root.strip():
            raise ConfigError("output_root must not be empty")
        if not self.metrics:
            raise ConfigError("metrics must not be empty")
        missing = [name for name in self.metrics if name not in self.benchmark.metric_versions]
        if missing:
            raise ConfigError(
                f"benchmark.metric_versions missing metrics: {sorted(missing)}"
            )
        if self.grad_accum_steps < 1:
            raise ConfigError("grad_accum_steps must be >= 1")
        if self.grad_clip is not None and self.grad_clip <= 0:
            raise ConfigError("grad_clip must be > 0 when set")

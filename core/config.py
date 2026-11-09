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
    # DataLoader 并行度：0 = 主进程串行加载（旧行为）；>0 时多 worker 解码
    # JPEG/npz 并与 GPU 计算重叠。shuffle 仍由主进程 generator 决定，
    # 样本内容与顺序不变。pin_memory 需要 CUDA，无 CUDA 时自动关闭。
    num_workers: int = 0
    pin_memory: bool = False
    train_transform: str = "identity"
    eval_transform: str = "identity"
    views: int = 2
    # D-0080：视图 letterbox 边长。默认 224（所有历史 run）；448 用于特征
    # 分辨率对照。只改 RGB 视图，不改深度图（深度特征仍是 224）。
    input_size: int = 224
    test_set: str = "b"
    max_samples: int = 0
    sampler: str = "shuffle"
    # D-0050：真实远近成对采样（只作用于 train）。每个病例取表观跨度最大/最小
    # 的两帧组成一对，同批内相邻；配合 loss.pair_beta_weight 约束 β 分支在
    # 「同一息肉、不同取景」下输出一致。与 sampler=case_bin 互斥。
    pair_sampling: bool = False
    # 只让取景变化明显的病例参与配对（表观跨度 max/min ≥ 该值）。
    # 取景几乎不变的配对不提供一致性信号，1.0 = 不过滤。
    pair_min_span_ratio: float = 1.0
    # D-0056：混合采样——每轮先插入这么多个**配对批**（每批 8 对），
    # 再把全部帧 shuffle 一遍（100% 覆盖）。0 = 纯配对采样（D-0050 的 A/B 臂）。
    pair_batches: int = 0
    # D-0053：只保留真值 < 该值的**训练**帧（0 = 不过滤）。
    # 主指标是 0–10mm 逐档，≥10mm 的样本不进训练；val/test 不受影响。
    train_max_truth_mm: float = 0.0
    # S（框最大边长）的尺寸自洽修正：按训练集拟合 k=truth/S 随尺寸的变化，
    # 输入侧把 S 重标定为 S'=S·k(S)/k_ref，使 β·D·S' 跨档尺度一致。
    # 修正只依赖输入 S（部署可用），拟合只用 train 行（无泄漏）。见 D-0019。
    span_calibration: bool = False
    # scale-consistency 增广（只作用于 train）：随机把内容图与 bbox 同步缩放
    # α∈[1/scale_aug, scale_aug]，标签不变——模拟不同放大率下的同一息肉，
    # 教模型「表观大小 × 深度尺度 → mm」的尺度不变性。0 = 关闭，≥1 为最大倍率。
    scale_aug: float = 0.0
    # val 并入训练（协议变更，见 D-0020）：>0 时从 train 病例中确定性抽取
    # 该数量的 4–6mm 病例作为新 val，旧 val 的 205 帧（含 2–3/9–10 尾档）并入
    # 训练。0 = 使用 manifest val（默认）。测试集不受影响。
    val_from_train_cases: int = 0
    # 病例级三折 dev 交叉验证（D-0022，替代 valmerge 的选模协议）：
    # dev_folds>0 时 dev 池=train+val 全部病例，按（来源,中位尺寸档）分层轮转
    # 分配到 dev_folds 折；dev_fold 指定当前 run 用哪折做验证（其余训练）。
    # 每折 val 覆盖全部尺寸档（含 2-3 与 7-10 尾档），选模不再盲区。
    # 与 val_from_train_cases 互斥。0 = 关闭（用 manifest val）。
    dev_folds: int = 0
    dev_fold: int = 0
    # D-0039：把指定 test split 并入训练（逗号分隔，如 "test_r"）。
    # 仅实验用：被并入的测试集随即失效（同帧进了训练），其评测数字不可作为
    # 留出成绩；其他 cohort 不受影响。
    extra_train_splits: str = ""

    def validate(self) -> None:
        if self.name not in {"regression", "v11_frames"}:
            raise ConfigError(f"unknown data.name: {self.name!r}")
        if not self.version.strip():
            raise ConfigError("data.version must not be empty")
        if self.seed < 0:
            raise ConfigError("data.seed must be >= 0")
        if self.split not in {"holdout", "manifest"}:
            raise ConfigError(f"unknown data.split: {self.split!r}")
        if self.sampler not in {"shuffle", "case_bin"}:
            raise ConfigError(
                f"unknown data.sampler: {self.sampler!r}; "
                "available: ['shuffle', 'case_bin']"
            )
        if self.pair_batches < 0:
            raise ConfigError("data.pair_batches must be >= 0")
        if self.pair_batches and not self.pair_sampling:
            raise ConfigError("data.pair_batches requires data.pair_sampling=true")
        if self.train_max_truth_mm < 0:
            raise ConfigError("data.train_max_truth_mm must be >= 0")
        if self.pair_min_span_ratio < 1.0:
            raise ConfigError("data.pair_min_span_ratio must be >= 1.0")
        if self.pair_sampling:
            if self.name != "v11_frames":
                raise ConfigError(
                    "data.pair_sampling is only supported by data.name='v11_frames'"
                )
            if self.sampler == "case_bin":
                raise ConfigError(
                    "data.pair_sampling and data.sampler='case_bin' are mutually exclusive"
                )
        if self.scale_aug < 0 or 0 < self.scale_aug < 1:
            raise ConfigError(
                "data.scale_aug must be 0 (off) or >= 1 (max zoom factor)"
            )
        if self.val_from_train_cases < 0:
            raise ConfigError("data.val_from_train_cases must be >= 0")
        allowed = {"test_b", "test_d", "test_r", "test_clean"}
        for name in [x.strip() for x in self.extra_train_splits.split(",") if x.strip()]:
            if name not in allowed:
                raise ConfigError(
                    f"data.extra_train_splits has unknown split {name!r}; "
                    f"available: {sorted(allowed)}"
                )
        if self.dev_folds < 0:
            raise ConfigError("data.dev_folds must be >= 0")
        if self.dev_folds > 0:
            if not 0 <= self.dev_fold < self.dev_folds:
                raise ConfigError(
                    f"data.dev_fold must be in [0, {self.dev_folds}) when "
                    "data.dev_folds > 0"
                )
            if self.val_from_train_cases > 0:
                raise ConfigError(
                    "data.dev_folds and data.val_from_train_cases are mutually "
                    "exclusive"
                )
        if self.train_transform not in {"identity", "light"}:
            raise ConfigError(
                f"unknown data.train_transform: {self.train_transform!r}"
            )
        if self.eval_transform not in {"identity", "identity448"}:
            raise ConfigError(
                f"unknown data.eval_transform: {self.eval_transform!r}"
            )
        if self.eval_transform == "identity448" and self.input_size != 448:
            raise ConfigError(
                "data.eval_transform='identity448' requires data.input_size=448"
            )
        if self.n_train <= 0 or self.n_val <= 0 or self.n_test <= 0:
            raise ConfigError("n_train/n_val/n_test must be > 0")
        if self.batch_size <= 0:
            raise ConfigError("batch_size must be > 0")
        if not 0 <= self.num_workers <= 32:
            raise ConfigError("data.num_workers must be in [0, 32]")
        if self.noise < 0:
            raise ConfigError("noise must be >= 0")
        if self.views not in {1, 2, 3}:
            raise ConfigError("views must be 1, 2 or 3")
        if not 224 <= self.input_size <= 1024 or self.input_size % 32:
            raise ConfigError(
                "data.input_size must be a multiple of 32 in [224, 1024]"
            )
        # D-0057：company = 用户重划分的公司数据集（company-v1 版本的 test_company）
        if self.test_set not in {"b", "d", "r", "clean", "company"}:
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
    # D-0027：把 log(S)、log(D) 作为显式条件输入喂给 β 头（零初始化旁路）。
    # 现状 S 只作为乘数、D 仅藏在 geom 里，β 头看不到「框有多大」，无法学
    # 尺寸相关的先验修正——这正是 2–3mm 档系统性高估的结构性来源。
    size_input: bool = False
    # D-0055：7×7 背景空间分支。整图分支不再全局平均池化，而是保留 ResNet
    # layer4 的 7×7 网格 + 位置编码 + 病灶框关系，经交叉注意力读出「病灶旁边
    # 是褶皱、后方是肠腔」这类绝对尺度线索（相对深度归一化掉的信息）。
    # 零初始化注入 β 头，训练起点等价于原模型。
    spatial_context: bool = False
    # D-0067：解压缩两臂。逐档偏差单调（2–3 +1.95 … 9–10 −2.46，交叉点 5–6mm）说明
    # 输出动态范围被压扁；门控收敛到 0.13–0.26（初值 sigmoid(2)=0.88），即 74–87% 的
    # 输出走无几何锚定、天然向均值回归的 direct 分支。
    #   gate_beta_floor：wg = floor + (1−floor)·sigmoid(logit)，给 β 分支一个权重下界；
    #   direct_scale_anchor：direct 分支改学 pred = exp(head)·D·S（对数空间的无量纲
    #     乘性修正），初值与 β 分支相同（bias = log(beta_ref)）。
    # 两者默认关闭；关闭时与旧模型逐位一致，且都不新增参数（旧 checkpoint 仍可直接加载）。
    gate_beta_floor: float = 0.0
    direct_scale_anchor: bool = False
    # D-0077：融合选择器。`gate`（默认）= 原 β/direct/门控三头；
    # `none` = β-only（只保留 head_beta，初值仍为 βref·D·S）。
    # 默认 gate 保证旧 checkpoint 与所有历史 run 的结构不变。
    fusion: str = "gate"
    # D-0083：纯图像回归对照。只用 RGB 两视图，不读 depth/几何 token；
    # 直接回归 mm（softplus），用于判断深度/几何特征是否真的带来收益。
    image_only: bool = False
    # D-0078：span 旁路。保留 S，把 [log(S/W), log W] 零初始化注入 β 头；
    # 与 D-0027 size_input 同一模式，不替换 S（避免 D-0075 的交换）。
    span_input: bool = False
    span_ratio_ref: float = 0.24629
    span_width_ref: float = 1258.0
    # D-0069：分档头。用 11 类（0–9 各一档 + >=10mm 一档，与 `mm_case_bin.bin_of`
    # 一致）的交叉熵替代标量回归，输出 = 档左端 + sigmoid(档内偏移)。CE 的最优解是条件
    # 分布而不是条件均值，且分档正确时档内误差 <1mm，与逐档 MAE 判据直接对齐。
    # 纯 mm 参数化：D·S 只经 geom token 作为特征，不做硬乘性锚定——B 的 k 是 company 的
    # 2–8 倍，硬锚定跨队列不通用（diranchor 在 B 上退化最多即此因，见 D-0068）。
    # 开启时不构造 β/direct/门控三个头；关闭时与旧模型逐位一致。
    ordinal_head: bool = False
    beta_ref: float = 0.07543606728391815
    truth_median: float = 5.0
    k_bound: float = 2.06
    # D-0087：保留米制深度刻度。EndoStreamDepth 输出 mm，但 prepare_lesion 把
    # 深度通道按帧中位归一化（depth/global_q），模型看不到绝对尺度。开启后模型
    # 用 D=q 与 scalars[7]=q/global_q 反推 global_q，把通道 0 换成
    # log(depth_mm/metric_ref_mm)，保留米制信息；仅支持 β-only、非 image_only。
    metric_depth: bool = False
    metric_ref_mm: float = 20.0
    # D-0089：物理锚定头。k = k_phys(log D, log S) · exp(bound·tanh(z_feat/bound))；
    # k_phys 只看米制几何，特征残差被硬性压小。用于阻止 B 上的 head 漂移。
    metric_anchor: bool = False
    metric_residual_bound: float = 0.3
    metric_phys_bound: float = 1.0
    metric_span_ref_px: float = 300.0
    # D-0091：米制焦距头。size = D_mm·S/f，log f = log(1/beta_ref) +
    # bound·tanh(z_feat/bound)；紧界只允许跨队列有限焦距差，阻止 β 头压缩。
    metric_focal_head: bool = False
    metric_focal_bound: float = 0.15
    # D-0093：v2 双因子尺度。几何宽界负责相机/深度尺度，外观窄残差只修框误差。
    # pred = D·(S/W)/(f/W)·exp(r_app)；f/W = f_ref/W_ref·exp(c_phys·tanh(z_phys/c_phys))。
    metric_focal_v2: bool = False
    metric_focal_phys_bound: float = 2.0
    metric_focal_app_bound: float = 0.3
    metric_focal_width_ref: float = 1258.0
    metric_focal_height_ref: float = 1080.0

    def validate(self) -> None:
        if self.name not in {"mlp", "linear", "v11"}:
            raise ConfigError(f"unknown model.name: {self.name!r}")
        if self.hidden <= 0:
            raise ConfigError("hidden must be > 0")
        if self.views not in {1, 2, 3}:
            raise ConfigError("model.views must be 1, 2 or 3")
        if self.d_model <= 0 or self.heads <= 0:
            raise ConfigError("d_model/heads must be > 0")
        if self.backbone not in {"full", "partial", "frozen"}:
            raise ConfigError(f"unknown model.backbone: {self.backbone!r}")
        if self.beta_ref <= 0 or self.truth_median <= 0 or self.k_bound <= 0:
            raise ConfigError("beta_ref/truth_median/k_bound must be > 0")
        if self.spatial_context and self.name != "v11":
            raise ConfigError("model.spatial_context is only supported by model.name='v11'")
        if not 0.0 <= self.gate_beta_floor < 1.0:
            raise ConfigError("model.gate_beta_floor must be in [0, 1)")
        if (self.gate_beta_floor > 0.0 or self.direct_scale_anchor) and self.name != "v11":
            raise ConfigError(
                "model.gate_beta_floor/direct_scale_anchor are only supported by model.name='v11'"
            )
        if self.ordinal_head:
            if self.name != "v11":
                raise ConfigError("model.ordinal_head is only supported by model.name='v11'")
            # 单变量纪律：分档头必须是自己相对 base 的唯一改动，否则无法归因
            if (
                self.gate_beta_floor > 0.0
                or self.direct_scale_anchor
                or self.size_input
                or self.span_input
                or self.fusion != "gate"
            ):
                raise ConfigError(
                    "model.ordinal_head cannot be combined with gate_beta_floor/"
                    "direct_scale_anchor/size_input/span_input/fusion（单变量对照）"
                )
        if self.fusion not in {"gate", "none"}:
            raise ConfigError(f"unknown model.fusion: {self.fusion!r}")
        if self.image_only:
            if self.name != "v11":
                raise ConfigError("model.image_only is only supported by model.name='v11'")
            if (
                self.fusion != "none"
                or self.ordinal_head
                or self.size_input
                or self.span_input
                or self.direct_scale_anchor
                or self.gate_beta_floor > 0.0
                or self.spatial_context
            ):
                raise ConfigError(
                    "model.image_only must be isolated (fusion='none', no "
                    "ordinal/size/span/direct/gate/spatial branches)"
                )
        if self.fusion != "gate" and self.name != "v11":
            raise ConfigError("model.fusion is only supported by model.name='v11'")
        if self.fusion == "none" and (
            self.gate_beta_floor > 0.0 or self.direct_scale_anchor
        ):
            raise ConfigError(
                "model.fusion='none' cannot be combined with gate_beta_floor/"
                "direct_scale_anchor（没有门控/direct 分支）"
            )
        if self.span_input:
            if self.name != "v11":
                raise ConfigError("model.span_input is only supported by model.name='v11'")
            if self.size_input:
                raise ConfigError(
                    "model.span_input and model.size_input are mutually exclusive"
                )
        if self.span_ratio_ref <= 0 or self.span_width_ref <= 0:
            raise ConfigError("model.span_ratio_ref/span_width_ref must be > 0")
        if self.metric_depth:
            if self.name != "v11":
                raise ConfigError("model.metric_depth is only supported by model.name='v11'")
            if self.fusion != "none":
                raise ConfigError(
                    "model.metric_depth currently requires model.fusion='none' (β-only)"
                )
            if self.image_only:
                raise ConfigError("model.metric_depth cannot be combined with image_only")
        if self.metric_ref_mm <= 0:
            raise ConfigError("model.metric_ref_mm must be > 0")
        if self.metric_anchor:
            if self.name != "v11":
                raise ConfigError("model.metric_anchor is only supported by model.name='v11'")
            if self.fusion != "none":
                raise ConfigError(
                    "model.metric_anchor requires model.fusion='none' (β-only)"
                )
            if not self.metric_depth:
                raise ConfigError("model.metric_anchor requires model.metric_depth=true")
            if (
                self.metric_residual_bound <= 0
                or self.metric_phys_bound <= 0
                or self.metric_span_ref_px <= 0
            ):
                raise ConfigError(
                    "model.metric_* bounds/ref must be > 0"
                )
        if self.metric_focal_head:
            if self.name != "v11":
                raise ConfigError(
                    "model.metric_focal_head is only supported by model.name='v11'"
                )
            if self.fusion != "none":
                raise ConfigError(
                    "model.metric_focal_head requires model.fusion='none' (β-only)"
                )
            if not self.metric_depth:
                raise ConfigError(
                    "model.metric_focal_head requires model.metric_depth=true"
                )
            if self.metric_anchor:
                raise ConfigError(
                    "model.metric_focal_head and model.metric_anchor are "
                    "mutually exclusive"
                )
            if not 0.0 < self.metric_focal_bound <= 1.0:
                raise ConfigError(
                    "model.metric_focal_bound must be in (0, 1]"
                )
        if self.metric_focal_v2:
            if self.name != "v11":
                raise ConfigError(
                    "model.metric_focal_v2 is only supported by model.name='v11'"
                )
            if self.fusion != "none":
                raise ConfigError(
                    "model.metric_focal_v2 requires model.fusion='none' (β-only)"
                )
            if not self.metric_depth:
                raise ConfigError(
                    "model.metric_focal_v2 requires model.metric_depth=true"
                )
            if self.metric_anchor:
                raise ConfigError(
                    "model.metric_focal_v2 and model.metric_anchor are "
                    "mutually exclusive"
                )
            if self.metric_focal_head:
                raise ConfigError(
                    "model.metric_focal_v2 and model.metric_focal_head are "
                    "mutually exclusive"
                )
            if not 0.0 < self.metric_focal_phys_bound <= 3.0:
                raise ConfigError(
                    "model.metric_focal_phys_bound must be in (0, 3]"
                )
            if not 0.0 < self.metric_focal_app_bound <= 1.0:
                raise ConfigError(
                    "model.metric_focal_app_bound must be in (0, 1]"
                )
            if (
                self.metric_focal_width_ref <= 0
                or self.metric_focal_height_ref <= 0
            ):
                raise ConfigError(
                    "model.metric_focal_width_ref/height_ref must be > 0"
                )


@dataclass
class LossConfig(ConfigSchema):
    name: str = "mse"
    huber_beta: float = 1.0
    delta: float = 1.0
    rank_gamma: float = 1.0
    rank_interp: float = 1.0
    # mm_case_bin 用相对误差 |pred-truth|/clamp(truth,1mm)（仅 mm_case_bin 支持）：
    # 绝对 mm 误差下小息肉的梯度与大息肉相同，相对误差按比例放大小档压力。
    relative: bool = False
    # D-0022 对照 B：绝对为主 + 少量相对为辅的混合损失
    # per = (1-w)·|err| + w·hybrid_rel_scale·|err|/clamp(truth,1mm)。
    # 0 = 关闭；与 relative 互斥使用（relative 优先）。对照预注册值 w=0.25。
    hybrid_rel_weight: float = 0.0
    hybrid_rel_scale: float = 5.0
    # 对数空间（乘性）损失：per = SmoothL1(log(pred) − log(truth), beta=log_beta)。
    # 尺寸是乘性量，对数空间把「尺度压缩」变成可加的偏移；PM v11 默认配方即此口径，
    # 其逐档结果显示大档明显受益（D 8-9 1.86 vs 毫米空间 2.21）。
    log_space: bool = False
    log_beta: float = 0.1
    # D-0034：在相对/绝对主损失上**叠加**对数项：
    #   per = main + log_weight · SmoothL1(log(pred)−log(truth), beta=log_beta)
    # 相对项压小档、对数项压大档（两版各自实验证据），等权 1.0 为预注册值。
    # 与 log_space（纯对数）互斥。
    log_weight: float = 0.0
    # D-0045：分支独立监督（仅 mm_rank 支持）。总损失 =
    #   L_fused + γ·L_rank + branch_beta_weight·L_β + branch_direct_weight·L_direct
    # 其中 L_β / L_direct 与主损失同口径（log 空间、病例/档位等权）。
    branch_beta_weight: float = 0.0
    branch_direct_weight: float = 0.0
    # D-0050：β 分支配对一致性（仅 mm_rank 支持）。同一病例的远/近帧真实尺寸
    # 相同，故要求 β 分支输出相等：
    #   L_pair = SmoothL1(log d_β,near − log d_β,far, beta=1)
    # 直接约束 β 分支，避免模型只靠压低门控获得稳定输出。
    pair_beta_weight: float = 0.0
    # D-0069：分档头损失（loss.name=mm_ordinal）。总损失 =
    #   CE(bin) + ordinal_offset_weight·SmoothL1(档内偏移) + rank_gamma·RankSim
    # 两项都按 mm_case_bin 的病例/档位等权口径加权。档内偏移只在 0–9 档有定义，
    # >=10mm 档由模型输出 10+softplus(·)，不参与偏移监督。
    ordinal_offset_weight: float = 1.0
    ordinal_offset_beta: float = 0.1

    def validate(self) -> None:
        if self.name not in {"mse", "huber", "mm_case_bin", "mm_rank", "mm_ordinal"}:
            raise ConfigError(f"unknown loss.name: {self.name!r}")
        if not 0.0 <= self.hybrid_rel_weight <= 1.0:
            raise ConfigError("loss.hybrid_rel_weight must be in [0, 1]")
        if self.hybrid_rel_weight > 0 and self.relative:
            raise ConfigError(
                "loss.hybrid_rel_weight and loss.relative are mutually exclusive"
            )
        if self.log_beta <= 0:
            raise ConfigError("loss.log_beta must be > 0")
        if self.log_space and (self.relative or self.hybrid_rel_weight > 0):
            raise ConfigError(
                "loss.log_space is mutually exclusive with relative/hybrid_rel_weight"
            )
        if self.log_weight < 0:
            raise ConfigError("loss.log_weight must be >= 0")
        if self.branch_beta_weight < 0 or self.branch_direct_weight < 0:
            raise ConfigError("loss.branch_*_weight must be >= 0")
        if (self.branch_beta_weight or self.branch_direct_weight) and self.name != "mm_rank":
            raise ConfigError(
                "loss.branch_*_weight is only supported by loss.name='mm_rank'"
            )
        if self.pair_beta_weight < 0:
            raise ConfigError("loss.pair_beta_weight must be >= 0")
        if self.pair_beta_weight and self.name != "mm_rank":
            raise ConfigError(
                "loss.pair_beta_weight is only supported by loss.name='mm_rank'"
            )
        if self.log_space and self.log_weight > 0:
            raise ConfigError(
                "loss.log_weight is redundant with loss.log_space; use one of them"
            )
        if self.huber_beta <= 0:
            raise ConfigError("huber_beta must be > 0")
        if self.delta <= 0:
            raise ConfigError("loss.delta must be > 0")
        if self.rank_gamma < 0 or self.rank_interp < 0:
            raise ConfigError("rank_gamma/rank_interp must be >= 0")
        if self.ordinal_offset_weight < 0:
            raise ConfigError("loss.ordinal_offset_weight must be >= 0")
        if self.ordinal_offset_beta <= 0:
            raise ConfigError("loss.ordinal_offset_beta must be > 0")
        if self.name == "mm_ordinal" and (
            self.relative or self.log_space or self.log_weight > 0
            or self.hybrid_rel_weight > 0 or self.branch_beta_weight
            or self.branch_direct_weight or self.pair_beta_weight
        ):
            raise ConfigError(
                "loss.name='mm_ordinal' 只支持 CE + 档内偏移 + RankSim；"
                "relative/log_space/log_weight/hybrid/branch/pair 项都不适用"
            )


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
    warmup_epochs: int = 0

    def validate(self) -> None:
        if self.name not in {"cosine", "constant"}:
            raise ConfigError(f"unknown scheduler.name: {self.name!r}")
        if self.t_max is not None and self.t_max < 1:
            raise ConfigError("scheduler.t_max must be >= 1 when set")
        if self.eta_min < 0:
            raise ConfigError("scheduler.eta_min must be >= 0")
        if self.warmup_epochs < 0:
            raise ConfigError("scheduler.warmup_epochs must be >= 0")


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
        # 分档头与分档损失必须成对出现：头产出 aux['ord_logits']，损失才读得到
        if bool(self.model.ordinal_head) != (self.loss.name == "mm_ordinal"):
            raise ConfigError(
                "model.ordinal_head and loss.name='mm_ordinal' must be set together"
            )
        if self.model.metric_depth and "endo" not in self.data.version:
            raise ConfigError(
                "model.metric_depth requires an EndoStreamDepth metric version "
                "(data.version contains 'endo')"
            )

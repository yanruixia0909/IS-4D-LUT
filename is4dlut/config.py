"""Configuration objects for IS-4D LUT.

Paper settings (Sec. 4.1.3): Adam (beta1=0.9, beta2=0.999), batch size 1,
initial lr 1e-4, 1e5 iterations, N = 33 (Sec. 3.1 / Sec. 4.3).

Settings that are *not* stated explicitly in the paper are gathered in
``INFERRED_SETTINGS`` (also documented in README.md).
"""
from dataclasses import asdict, dataclass, field
import json
import os


@dataclass
class ModelConfig:
    # ---- 4D LUT ----
    n_grid: int = 33          # number of bins per axis (paper: N = 33)
    n_basis: int = 3          # number of basis LUTs M (paper uses M, value not given)
    # ---- encoder / decoder ----
    base_channels: int = 32   # channel width of the first level
    num_levels: int = 3       # down-sampling levels (H, H/2, H/4, H/8)
    # ---- intensity ----
    intensity_from: str = "prior"   # "prior" (luminance of F_p) or "intensity" (F_d)
    # ---- semantics ----
    n_regions: int = 3        # K semantic regions for the histogram loss


@dataclass
class LossConfig:
    # L_total1 (Eq. 11)
    alpha_smooth: float = 1e-4
    alpha_mono: float = 10.0
    alpha_mse: float = 1.0
    alpha_ssim: float = 0.1
    # L_total2 (Eq. 12)
    beta_recon: float = 1.0
    lambda_sc: float = 0.1
    lambda_sa: float = 0.05
    # details
    hist_bins: int = 32       # bins of the differentiable histogram
    gan_type: str = "hinge"   # {"hinge", "lsgan", "vanilla"}


@dataclass
class TrainConfig:
    data_root: str = "data"
    out_dir: str = "results"
    ckpt_dir: str = "checkpoints"
    # optimisation (paper: Adam, beta=(0.9, 0.999), lr=1e-4, batch=1, 1e5 iters)
    batch_size: int = 1
    num_workers: int = 0
    lr: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.0
    total_iters: int = 100000
    lr_decay_start: int = 50000
    grad_clip: float = 1.0
    amp: bool = True
    # data
    crop_size: int = 256
    use_flip: bool = True
    synthetic: bool = False
    synthetic_size: int = 128
    synthetic_train: int = 16
    synthetic_val: int = 4
    # logging / saving
    log_every: int = 20
    val_every: int = 1000
    save_every: int = 5000
    val_root: str = ""
    val_limit: int = 20
    val_images: int = 0       # >0: dump low/ours/GT grids at every validation
    n_critic: int = 1
    seed: int = 0
    device: str = "auto"      # {"auto", "cpu", "cuda"}


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> "Config":
        return cls(
            model=ModelConfig(**raw.get("model", {})),
            loss=LossConfig(**raw.get("loss", {})),
            train=TrainConfig(**raw.get("train", {})),
        )


# ----------------------------------------------------------------------
# Presets
# ----------------------------------------------------------------------
def paper_config() -> Config:
    """Faithful-to-paper configuration (N=33, lr=1e-4, batch=1, 1e5 iters)."""
    return Config()


def demo_config() -> Config:
    """Small configuration used by ``tools/demo.py`` so that the whole
    pipeline (training + inference + metrics) finishes in about a minute."""
    cfg = Config()
    cfg.model.n_grid = 9
    cfg.model.n_basis = 3
    cfg.model.base_channels = 16
    cfg.model.num_levels = 2
    cfg.train.batch_size = 1
    cfg.train.crop_size = 96
    cfg.train.synthetic = True
    cfg.train.synthetic_size = 96
    cfg.train.synthetic_train = 8
    cfg.train.synthetic_val = 2
    cfg.train.total_iters = 6
    cfg.train.log_every = 1
    cfg.train.val_every = 10 ** 9
    cfg.train.save_every = 10 ** 9
    cfg.train.amp = False
    cfg.train.out_dir = "results/demo"
    cfg.train.ckpt_dir = "checkpoints/demo"
    cfg.loss.hist_bins = 16
    return cfg


PRESETS = {
    "paper": paper_config,
    "demo": demo_config,
}


INFERRED_SETTINGS = {
    "n_basis": "M=3 basis LUTs (paper only says 'M context-driven base LUTs').",
    "n_grid": "N=33 from Sec. 3.1 / Sec. 4.3.",
    "intensity channel": "d_in = luminance(0.299R+0.587G+0.114B) of the lookup prior F_p; "
                         "switchable to the predicted intensity image F_d.",
    "encoder width": "base_channels=32 with 3 down-sampling levels.",
    "loss weights": "alpha_s=1e-4, alpha_m=10, alpha_mse=1, alpha_ssim=0.1, beta=1, "
                    "lambda_sc=0.1, lambda_sa=0.05 (anchored on the sweeps of Fig. 10).",
    "semantic branch": "Semantic features are produced by an internal encoder over "
                       "[I_r, I_L] (no external semantic labels / pretrained model is "
                       "specified in the paper).",
    "histogram loss": "Differentiable soft histogram + Wasserstein-1 (CDF L1) per "
                      "semantic region, plus a histogram-equalisation auxiliary term.",
    "adversarial loss": "PatchGAN global + local discriminators with hinge loss and "
                        "semantic saliency re-weighting.",
}

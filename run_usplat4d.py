"""run_usplat4d.py — CLI entry-point for USplat4D fine-tuning.

Loads an existing SoM checkpoint (produced by run_training.py) and applies the
USplat4D uncertainty-aware graph refinement on top.

Three directories are involved:
  --som-dir   : SoM output directory (INPUT) — must contain checkpoints/last.ckpt
  --out-dir   : USplat4D output directory (OUTPUT) — checkpoints, graph, config saved here
  --data.data-dir : original dataset directory (INPUT) — same as used for SoM training

Typical usage:
    conda activate som
    python run_usplat4d.py \\
        --som-dir /media/ee904/DATA1/Yun/Outputs/shape-of-motion/nvidia/Skating \\
        --out-dir /media/ee904/DATA1/Yun/Outputs/usplat4d/nvidia/Skating \\
        data:nvidia \\
        --data.data-dir /media/ee904/DATA1/Yun/Datasets/shape-of-motion/nvidia/Skating

All USplat4D hyperparameters (key-ratio, knn-k, …) can be overridden on the
command line, e.g. --key-ratio 0.03.

Checkpoints are written to <out-dir>/, with the final model saved as
<out-dir>/final.ckpt  (compatible with SoM's Trainer.init_from_checkpoint).
"""

import os
import sys
import os.path as osp
from dataclasses import asdict, dataclass
from typing import Annotated, Optional, Tuple

import torch
import tyro
import yaml
from loguru import logger as guru
from torch.utils.data import DataLoader

# ---- SoM imports ---------------------------------------------------------- #
sys.path.insert(0, osp.join(osp.dirname(__file__), "../shape-of-motion"))

from flow3d.configs import LossesConfig, OptimizerConfig, SceneLRConfig
from flow3d.data import (
    BaseDataset,
    DavisDataConfig,
    CustomDataConfig,
    get_train_val_datasets,
    iPhoneDataConfig,
    NvidiaDataConfig,
)
from flow3d.data.utils import to_device
from flow3d.trainer import Trainer as SoMTrainer

# ---- USplat4D imports ------------------------------------------------------ #
from usplat4d.trainer import USplat4DTrainer, USplat4DConfig


# --------------------------------------------------------------------------- #
#  CLI configuration                                                            #
# --------------------------------------------------------------------------- #

@dataclass
class USplat4DRunConfig:
    """Top-level configuration for USplat4D fine-tuning."""

    som_dir: str
    """SoM output directory (INPUT) — must contain checkpoints/last.ckpt."""

    out_dir: str
    """USplat4D output directory (OUTPUT) — checkpoints, graph, and cfg.yaml are saved here."""

    data: (
        Annotated[iPhoneDataConfig, tyro.conf.subcommand(name="iphone")]
        | Annotated[DavisDataConfig, tyro.conf.subcommand(name="davis")]
        | Annotated[CustomDataConfig, tyro.conf.subcommand(name="custom")]
        | Annotated[NvidiaDataConfig, tyro.conf.subcommand(name="nvidia")]
    )
    """Dataset config — must match the one used for SoM training."""

    # SoM model / optimizer configs (needed to reconstruct Trainer)
    # All sub-fields have defaults, so tyro resolves them without CLI input.
    lr: SceneLRConfig
    loss: LossesConfig
    optim: OptimizerConfig

    # USplat4D hyperparameters (all paper defaults)
    key_ratio: float = 0.02
    """Fraction of Gaussians designated as key nodes (top-uncertainty)."""
    spt_threshold: int = 5
    """Min frames below uncertainty threshold for SPT filter."""
    knn_k: int = 8
    """K nearest neighbors for key-key edges."""
    eta_c: float = 0.5
    """Convergence color error threshold (Section 4.1)."""
    phi: float = 1e6
    """Uncertainty ceiling for non-converged Gaussians."""
    r_scale: Tuple[float, float, float] = (1.0, 1.0, 0.01)
    """Anisotropic uncertainty scale [rx, ry, rz] (Eq. 8)."""

    # Loss weights
    lambda_iso: float = 1.0
    lambda_rigid: float = 1.0
    lambda_rot: float = 0.01
    lambda_vel: float = 0.01
    lambda_acc: float = 0.01
    lambda_key: float = 1.0
    lambda_non_key: float = 1.0

    # Training
    extra_epochs: int = 400
    batch_size: int = 8
    num_dl_workers: int = 4
    use_2dgs: bool = False

    # Misc
    resume_graph: bool = False
    """If True and <out_dir>/graph.pt exists, reload saved graph instead of recomputing."""
    port: Optional[int] = None


# --------------------------------------------------------------------------- #
#  Main                                                                         #
# --------------------------------------------------------------------------- #

def main(cfg: USplat4DRunConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    guru.info(f"Using device: {device}")

    ckpt_path = osp.join(cfg.som_dir, "checkpoints", "last.ckpt")
    if not osp.exists(ckpt_path):
        guru.error(f"SoM checkpoint not found at {ckpt_path}. "
                   "Run SoM training first with run_training.py.")
        raise FileNotFoundError(ckpt_path)

    # ---- Build dataset ------------------------------------------------------ #
    guru.info("Loading dataset …")
    train_dataset, _, _, _ = get_train_val_datasets(cfg.data, load_val=False)
    guru.info(f"  {train_dataset.num_frames} frames")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_dl_workers,
        persistent_workers=cfg.num_dl_workers > 0,
        collate_fn=BaseDataset.train_collate_fn,
        shuffle=True,
    )

    # ---- Load pretrained SoM Trainer --------------------------------------- #
    guru.info(f"Loading SoM checkpoint from {ckpt_path} …")
    som_trainer, start_epoch = SoMTrainer.init_from_checkpoint(
        ckpt_path,
        device,
        cfg.use_2dgs,
        cfg.lr,
        cfg.loss,
        cfg.optim,
        work_dir=cfg.out_dir,   # USplat4D output dir to avoid overwriting SoM checkpoint
        port=cfg.port,
    )
    guru.info(f"  SoM model loaded (epoch {start_epoch}, "
              f"step {som_trainer.global_step})")

    # ---- Build USplat4D config --------------------------------------------- #
    usplat_cfg = USplat4DConfig(
        key_ratio=cfg.key_ratio,
        spt_threshold=cfg.spt_threshold,
        knn_k=cfg.knn_k,
        eta_c=cfg.eta_c,
        phi=cfg.phi,
        r_scale=cfg.r_scale,
        lambda_iso=cfg.lambda_iso,
        lambda_rigid=cfg.lambda_rigid,
        lambda_rot=cfg.lambda_rot,
        lambda_vel=cfg.lambda_vel,
        lambda_acc=cfg.lambda_acc,
        lambda_key=cfg.lambda_key,
        lambda_non_key=cfg.lambda_non_key,
        extra_epochs=cfg.extra_epochs,
        batch_size=cfg.batch_size,
    )

    # Save config
    out_dir = cfg.out_dir
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(osp.join(out_dir, "checkpoints"), exist_ok=True)
    
    # Convert config to dict and recursively convert all tuples to lists for YAML compatibility
    def _tuples_to_lists(obj):
        """Recursively convert tuples to lists in dicts/lists."""
        if isinstance(obj, dict):
            return {k: _tuples_to_lists(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [_tuples_to_lists(item) for item in obj]
        else:
            return obj
    
    cfg_dict = _tuples_to_lists(asdict(cfg))
    
    with open(osp.join(out_dir, "cfg.yaml"), "w") as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)
    guru.info(f"Config saved to {out_dir}/cfg.yaml")

    # ---- Instantiate USplat4DTrainer --------------------------------------- #
    usplat_trainer = USplat4DTrainer(
        som_trainer=som_trainer,
        train_dataset=train_dataset,
        cfg=usplat_cfg,
        work_dir=out_dir,
        device=device,
    )

    # Optionally reload a previously saved graph (for resuming)
    graph_path = osp.join(out_dir, "graph.pt")
    if cfg.resume_graph and osp.exists(graph_path):
        guru.info(f"Reloading graph from {graph_path} …")
        usplat_trainer.load_graph(graph_path)
    else:
        usplat_trainer.save_graph(graph_path)

    # ---- Fine-tune --------------------------------------------------------- #
    guru.info(f"Starting USplat4D fine-tuning: {cfg.extra_epochs} extra epochs …")
    usplat_trainer.train(
        train_loader=train_loader,
        extra_epochs=cfg.extra_epochs,
    )

    guru.info(f"Done. Final checkpoint at: {osp.join(out_dir, 'final.ckpt')}")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    cfg = tyro.cli(USplat4DRunConfig)
    main(cfg)

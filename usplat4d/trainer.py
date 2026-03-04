"""USplat4D Trainer: extends SoM's Trainer with uncertainty-aware graph losses.

Usage flow:
    1. Load pretrained SoM checkpoint with Trainer.init_from_checkpoint()
    2. Instantiate USplat4DTrainer(som_trainer, train_dataset, ...)
       → pre-computes uncertainties, builds graph, freezes pretrained positions
    3. Call usplat4d_trainer.train() to run additional fine-tuning epochs

The trainer overrides compute_losses() to inject L_key and L_non_key on top of
SoM's existing RGB + tracking losses.

References:
    Paper Section 4.3 (Loss design) and Appendix A.2 (Hyperparameters)
"""

from __future__ import annotations

import copy
import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from loguru import logger as guru
from torch import Tensor
from tqdm import tqdm

# SoM imports
from flow3d.scene_model import SceneModel
from flow3d.trainer import Trainer as SoMTrainer

# USplat4D imports
from usplat4d.uncertainty import compute_uncertainty_all_frames, build_uncertainty_3d_matrices
from usplat4d.graph import build_graph, USplat4DGraph
from usplat4d.losses import key_node_loss, non_key_node_loss


class USplat4DConfig:
    """Hyperparameters from Section 4 and Appendix A.2 of the paper."""
    # Graph construction
    key_ratio: float = 0.02        # Top 2% most uncertain as key nodes
    spt_threshold: int = 5         # Min frames below uncertainty threshold
    knn_k: int = 8                 # K nearest neighbors for key-key edges
    # Uncertainty
    eta_c: float = 0.5             # Convergence color threshold
    phi: float = 1e6               # Max uncertainty for diverged Gaussians
    r_scale: Tuple[float, float, float] = (1.0, 1.0, 0.01)  # [rx, ry, rz]
    # Loss weights (Appendix A.2)
    lambda_iso: float = 1.0
    lambda_rigid: float = 1.0
    lambda_rot: float = 0.01
    lambda_vel: float = 0.01
    lambda_acc: float = 0.01
    # Training
    extra_epochs: int = 400
    batch_size: int = 8
    # Loss scale (relative to SoM losses)
    lambda_key: float = 1.0
    lambda_non_key: float = 1.0

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise ValueError(f"Unknown config key: {k}")
            setattr(self, k, v)

    def __repr__(self):
        attrs = {k: getattr(self, k) for k in vars(USplat4DConfig) if not k.startswith('_') and not callable(getattr(USplat4DConfig, k))}
        return f"USplat4DConfig({attrs})"


class USplat4DTrainer:
    """Wraps SoM's Trainer and adds USplat4D graph losses.

    The SoM model is fine-tuned with additional graph-based constraints.
    The pretrained positions (p^o) are frozen at initialization.

    Args:
        som_trainer: Already-loaded SoM Trainer instance.
        train_dataset: SoM training dataset (needed for uncertainty estimation).
        cfg: USplat4D configuration / hyperparameters.
        work_dir: Directory for saving USplat4D checkpoints (sub-folder 'usplat4d').
        device: torch.device.
    """

    def __init__(
        self,
        som_trainer: SoMTrainer,
        train_dataset,
        cfg: Optional[USplat4DConfig] = None,
        work_dir: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        self.som_trainer = som_trainer
        self.model: SceneModel = som_trainer.model
        self.train_dataset = train_dataset
        self.cfg = cfg or USplat4DConfig()
        self.device = device or next(self.model.parameters()).device
        self.work_dir = work_dir or som_trainer.work_dir

        os.makedirs(os.path.join(self.work_dir, "usplat4d"), exist_ok=True)

        # ------------------------------------------------------------------ #
        #  Step 1: pre-compute per-Gaussian uncertainties (Section 4.1)       #
        # ------------------------------------------------------------------ #
        guru.info("USplat4D: computing per-Gaussian uncertainties …")
        self.u_scalar = compute_uncertainty_all_frames(
            model=self.model,
            train_dataset=train_dataset,
            device=self.device,
            eta_c=self.cfg.eta_c,
            phi=self.cfg.phi,
        )
        # u_scalar: (G_fg, T)  -- only foreground Gaussians

        # ------------------------------------------------------------------ #
        #  Step 2: get per-Gaussian, per-time positions for graph building     #
        # ------------------------------------------------------------------ #
        T = self.model.num_frames
        G_fg = self.model.fg.num_gaussians

        guru.info("USplat4D: computing foreground positions for all frames …")
        ts_all = torch.arange(T, device=self.device)
        with torch.no_grad():
            means_all, quats_all = self.model.compute_poses_fg(ts_all)
            # means_all: (G_fg, T, 3)  quats_all: (G_fg, T, 4)

        # ------------------------------------------------------------------ #
        #  Step 3: build uncertainty-encoded graph (Section 4.2)              #
        # ------------------------------------------------------------------ #
        guru.info("USplat4D: building uncertainty-encoded graph …")
        self.graph: USplat4DGraph = build_graph(
            means_t=means_all,
            u_scalar=self.u_scalar,
            key_ratio=self.cfg.key_ratio,
            spt_threshold=self.cfg.spt_threshold,
            knn_k=self.cfg.knn_k,
        )
        guru.info(
            f"  → {len(self.graph.key_idx)} key nodes, "
            f"{len(self.graph.nonkey_idx)} non-key nodes"
        )

        # ------------------------------------------------------------------ #
        #  Step 4: freeze pretrained positions p^o (canonical → all frames)   #
        # ------------------------------------------------------------------ #
        guru.info("USplat4D: freezing pretrained positions p^o …")
        # Shape: (G_fg, T, 3) — frozen, used as Mahalanobis reference
        self.p_pretrained_fg: Tensor = means_all.detach().clone()
        # Canonical positions (t=0 basis, used for isometry loss)
        self.p_canonical_fg: Tensor = self.model.fg.params["means"].detach().clone()

        # Camera-to-world rotation matrices for all frames
        # w2cs: (T, 4, 4)  →  R_wc = R_{c2w} = R_{w2c}^T
        w2cs = self.model.w2cs.detach()  # (T, 4, 4)
        self.R_wc_all: Tensor = w2cs[:, :3, :3].transpose(-1, -2)  # (T, 3, 3)

        guru.info("USplat4D: initialization complete. Ready to train.")

        # Save a direct reference to SoM's original compute_losses so that
        # compute_losses_with_graph can call it without hitting the patch.
        self._som_compute_losses = som_trainer.compute_losses

    # ----------------------------------------------------------------------- #
    #  Core loss injection                                                      #
    # ----------------------------------------------------------------------- #

    def compute_usplat4d_losses(self, ts: Tensor) -> Tensor:
        """Compute USplat4D graph losses for a set of sampled frame indices.

        Args:
            ts: (B,) frame indices (int64) — same as SoM training batch 'ts'.

        Returns:
            Scalar tensor: L_key + L_non_key.
        """
        B = ts.shape[0]
        device = self.device

        key_idx = self.graph.key_idx        # (N_k,)
        nonkey_idx = self.graph.nonkey_idx  # (N_n,)

        # ---- Current positions (differentiable) ---- #
        # Recompute positions for the sampled frames so gradients flow.
        means_fg_t, quats_fg_t = self.model.compute_poses_fg(ts)
        # means_fg_t: (G_fg, B, 3)  quats_fg_t: (G_fg, B, 4)

        # Subset to key / non-key nodes
        pos_key_t  = means_fg_t[key_idx]    # (N_k, B, 3)
        quats_key_t = quats_fg_t[key_idx]   # (N_k, B, 4)
        pos_nk_t   = means_fg_t[nonkey_idx] # (N_n, B, 3)
        quats_nk_t = quats_fg_t[nonkey_idx] # (N_n, B, 4)

        # SE(3) transforms for key nodes (needed for rigid/DQB)
        transforms_key = self.model.compute_transforms(ts, inds=key_idx)  # (N_k, B, 3, 4)
        transforms_nk  = self.model.compute_transforms(ts, inds=nonkey_idx)  # (N_n, B, 3, 4)

        R_key_t = transforms_key[..., :3, :3]  # (N_k, B, 3, 3)
        t_key_t = transforms_key[..., :3, 3]   # (N_k, B, 3)

        # ---- Pretrained (frozen) reference positions ---- #
        # Index into (G_fg, T, 3) by [key_idx / nonkey_idx] and [ts]
        ts_cpu = ts.cpu()
        pos_key_pre = self.p_pretrained_fg[key_idx][:, ts_cpu].to(device)   # (N_k, B, 3)
        pos_nk_pre  = self.p_pretrained_fg[nonkey_idx][:, ts_cpu].to(device) # (N_n, B, 3)

        # ---- Uncertainty scalars for sampled frames ---- #
        u_key = self.u_scalar[key_idx][:, ts_cpu].to(device)   # (N_k, B)
        u_nk  = self.u_scalar[nonkey_idx][:, ts_cpu].to(device) # (N_n, B)

        # ---- Camera rotations for sampled frames ---- #
        R_wc_t = self.R_wc_all[ts_cpu].to(device)  # (B, 3, 3)

        # ---- Canonical positions for motion losses ---- #
        pos_o_key = self.p_canonical_fg[key_idx]    # (N_k, 3)
        pos_o_nk  = self.p_canonical_fg[nonkey_idx] # (N_n, 3)

        # ---- Key node neighbors (local indices within key_idx array) ---- #
        key_nbrs_local   = self.graph.key_nbrs          # (N_k, K) 0..N_k-1
        key_nbr_weights  = self.graph.key_nbr_weights   # (N_k, K)

        # ---- Non-key node neighbors (indices into key_idx array) ---- #
        nonkey_nbrs_local   = self.graph.nonkey_nbrs         # (N_n, K+1) indices into key nodes
        nonkey_nbr_weights  = self.graph.nonkey_nbr_weights  # (N_n, K+1)

        # ---- Key loss (Eq. 11) ---- #
        l_key = key_node_loss(
            pos_key_t=pos_key_t,
            quats_key_t=quats_key_t,
            transforms_key_t=transforms_key,
            pos_key_pretrained=pos_key_pre,
            u_key=u_key,
            R_wc_t=R_wc_t,
            pos_o=pos_o_key,
            key_nbrs_local=key_nbrs_local,
            key_nbr_weights=key_nbr_weights,
            r_scale=self.cfg.r_scale,
            lambda_iso=self.cfg.lambda_iso,
            lambda_rigid=self.cfg.lambda_rigid,
            lambda_rot=self.cfg.lambda_rot,
            lambda_vel=self.cfg.lambda_vel,
            lambda_acc=self.cfg.lambda_acc,
        )

        # ---- Non-key loss (Eq. 13) ---- #
        l_non_key = non_key_node_loss(
            pos_nk_t=pos_nk_t,
            quats_nk_t=quats_nk_t,
            transforms_nk_t=transforms_nk,
            pos_nk_pretrained=pos_nk_pre,
            u_nk=u_nk,
            R_wc_t=R_wc_t,
            pos_o_nk=pos_o_nk,
            R_key_t=R_key_t,         # current key node rotations
            t_key_t=t_key_t,         # current key node translations
            nonkey_nbrs_local=nonkey_nbrs_local,
            nonkey_nbr_weights=nonkey_nbr_weights,
            nonkey_nbrs_global=nonkey_nbrs_local,  # same (keys indexed 0..N_k-1)
            r_scale=self.cfg.r_scale,
            lambda_iso=self.cfg.lambda_iso,
            lambda_rigid=self.cfg.lambda_rigid,
            lambda_rot=self.cfg.lambda_rot,
            lambda_vel=self.cfg.lambda_vel,
            lambda_acc=self.cfg.lambda_acc,
        )

        return self.cfg.lambda_key * l_key + self.cfg.lambda_non_key * l_non_key, {
            "usplat4d/l_key": l_key.item(),
            "usplat4d/l_non_key": l_non_key.item(),
        }

    # ----------------------------------------------------------------------- #
    #  Patched train_step: inject graph losses into SoM's compute_losses       #
    # ----------------------------------------------------------------------- #

    def compute_losses_with_graph(self, batch):
        """compute_losses that appends USplat4D graph losses to SoM losses."""
        # Call the saved original — not self.som_trainer.compute_losses, which
        # has been replaced with this function (would cause infinite recursion).
        loss, stats, num_rays_per_step, num_rays_per_sec = \
            self._som_compute_losses(batch)

        # Inject graph losses using the same frame indices
        ts = batch["ts"].to(self.device)  # (B,)
        graph_loss, graph_stats = self.compute_usplat4d_losses(ts)
        loss = loss + graph_loss

        stats.update(graph_stats)
        return loss, stats, num_rays_per_step, num_rays_per_sec

    # ----------------------------------------------------------------------- #
    #  Main training entry-point                                               #
    # ----------------------------------------------------------------------- #

    def train(
        self,
        train_loader,
        extra_epochs: Optional[int] = None,
        disable_density_control_fraction: float = 0.1,
    ):
        """Fine-tune the SoM model with USplat4D graph losses.

        Monkey-patches som_trainer.compute_losses so density control and all
        existing SoM infrastructure continues to work unchanged.

        Args:
            train_loader: DataLoader yielding SoM-style batches.
            extra_epochs: Override cfg.extra_epochs if provided.
            disable_density_control_fraction: Fraction of epochs at the START
                (and last 20%) during which density control is paused.
        """
        n_epochs = extra_epochs or self.cfg.extra_epochs
        start_epoch = self.som_trainer.epoch

        # Patch compute_losses
        original_compute_losses = self.som_trainer.compute_losses
        self.som_trainer.compute_losses = self.compute_losses_with_graph

        guru.info(f"USplat4D: starting fine-tuning for {n_epochs} extra epochs "
                  f"(from epoch {start_epoch})")

        epoch_bar = tqdm(
            range(start_epoch, start_epoch + n_epochs),
            desc="USplat4D fine-tuning",
            unit="epoch",
            dynamic_ncols=True,
        )
        try:
            for epoch in epoch_bar:
                self.som_trainer.set_epoch(epoch)

                # Disable density control in first + last 20% of extra epochs
                frac = (epoch - start_epoch) / n_epochs
                disable_dc = frac < disable_density_control_fraction or frac > 0.8
                if disable_dc:
                    # Density control runs iff global_step % control_every == 0;
                    # temporarily set control_every to a very large number.
                    orig_ctrl = self.som_trainer.optim_cfg.control_every
                    self.som_trainer.optim_cfg.control_every = int(1e9)

                batch_bar = tqdm(
                    train_loader,
                    desc=f"Epoch {epoch}",
                    unit="batch",
                    leave=False,
                    dynamic_ncols=True,
                )
                for batch in batch_bar:
                    batch = {
                        k: [t.to(self.device) for t in v]
                        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Tensor)
                        else v.to(self.device) if isinstance(v, Tensor)
                        else v
                        for k, v in batch.items()
                    }
                    loss_val = self.som_trainer.train_step(batch)
                    batch_bar.set_postfix(loss=f"{loss_val:.4f}")

                epoch_bar.set_postfix(loss=f"{loss_val:.4f}", dc_off=disable_dc)

                if disable_dc:
                    self.som_trainer.optim_cfg.control_every = orig_ctrl

                if epoch % 50 == 0:
                    ckpt_path = os.path.join(
                        self.work_dir, "usplat4d", f"epoch_{epoch:04d}.ckpt"
                    )
                    self.som_trainer.save_checkpoint(ckpt_path)
                    guru.info(f"USplat4D checkpoint saved → {ckpt_path}")

        finally:
            # Restore original compute_losses even if an exception occurs
            self.som_trainer.compute_losses = original_compute_losses

        guru.info("USplat4D: fine-tuning complete.")
        final_path = os.path.join(self.work_dir, "usplat4d", "final.ckpt")
        self.som_trainer.save_checkpoint(final_path)
        guru.info(f"USplat4D final checkpoint → {final_path}")

    # ----------------------------------------------------------------------- #
    #  Convenience: save / load the graph for resuming                         #
    # ----------------------------------------------------------------------- #

    def save_graph(self, path: Optional[str] = None):
        path = path or os.path.join(self.work_dir, "usplat4d", "graph.pt")
        torch.save({
            "graph": {
                "key_idx":            self.graph.key_idx,
                "nonkey_idx":         self.graph.nonkey_idx,
                "key_nbrs":           self.graph.key_nbrs,
                "key_nbr_weights":    self.graph.key_nbr_weights,
                "nonkey_key_idx":     self.graph.nonkey_key_idx,
                "nonkey_nbrs":        self.graph.nonkey_nbrs,
                "nonkey_nbr_weights": self.graph.nonkey_nbr_weights,
            },
            "u_scalar":         self.u_scalar,
            "p_pretrained_fg":  self.p_pretrained_fg,
            "p_canonical_fg":   self.p_canonical_fg,
            "R_wc_all":         self.R_wc_all,
        }, path)
        guru.info(f"Graph saved → {path}")

    def load_graph(self, path: str):
        data = torch.load(path, map_location=self.device)
        g = data["graph"]
        from usplat4d.graph import USplat4DGraph
        self.graph = USplat4DGraph(
            key_idx=g["key_idx"],
            nonkey_idx=g["nonkey_idx"],
            key_nbrs=g["key_nbrs"],
            key_nbr_weights=g["key_nbr_weights"],
            nonkey_key_idx=g["nonkey_key_idx"],
            nonkey_nbrs=g["nonkey_nbrs"],
            nonkey_nbr_weights=g["nonkey_nbr_weights"],
        )
        self.u_scalar        = data["u_scalar"].to(self.device)
        self.p_pretrained_fg = data["p_pretrained_fg"].to(self.device)
        self.p_canonical_fg  = data["p_canonical_fg"].to(self.device)
        self.R_wc_all        = data["R_wc_all"].to(self.device)
        guru.info(f"Graph loaded ← {path}")

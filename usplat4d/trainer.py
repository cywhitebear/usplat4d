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
import csv
import os
from typing import Optional, Tuple, Dict, Any

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
from usplat4d.losses import (
    key_node_loss, non_key_node_loss,
    isometry_loss, rigidity_loss, rotation_loss, velocity_loss, acceleration_loss,
)


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
        self._first_batch_logged = False  # Flag for per-batch diagnostics

        os.makedirs(os.path.join(self.work_dir, "usplat4d"), exist_ok=True)

        # Initialize diagnostic log file
        self.log_file = os.path.join(self.work_dir, "usplat4d", "diagnostics.csv")
        self._init_log_file()

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
            f"  → {len(self.graph.key_idx)} key nodes ({100*len(self.graph.key_idx)/G_fg:.1f}%), "
            f"{len(self.graph.nonkey_idx)} non-key nodes"
        )
        # Diagnostic: key node uncertainty stats
        u_key_nodes = self.u_scalar[self.graph.key_idx]  # (N_k, T)
        u_all = self.u_scalar
        
        # Per-key-node: best (min) and worst (max) uncertainty across time
        u_key_min_per_node = u_key_nodes.min(dim=1).values  # (N_k,) - best frame
        u_key_max_per_node = u_key_nodes.max(dim=1).values  # (N_k,) - worst frame
        u_key_avg_per_node = u_key_nodes.mean(dim=1)  # (N_k,) - average across time
        
        # Ratio: how many (key, frame) pairs have u < tau (phi threshold)?
        phi_threshold = 1e6
        key_below_phi = (u_key_nodes < phi_threshold).sum().item()
        key_total = u_key_nodes.numel()
        key_phi_ratio = 100 * key_below_phi / key_total if key_total > 0 else 0
        
        guru.info(
            f"USplat4D Diagnostics:"
            f" u_all: [{u_all.min():.2e}, {u_all.median():.2e}, {u_all.max():.2e}] "
            f"| u_key (best frame): [{u_key_min_per_node.min():.2e}, {u_key_min_per_node.median():.2e}, {u_key_min_per_node.max():.2e}] "
            f"| u_key (avg time): [{u_key_avg_per_node.min():.2e}, {u_key_avg_per_node.median():.2e}, {u_key_avg_per_node.max():.2e}] "
            f"| key_ratio_target={self.cfg.key_ratio} "
            f"| {{key_frames < phi: {key_phi_ratio:.1f}%}}"
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
    #  Diagnostic logging                                                      #
    # ----------------------------------------------------------------------- #

    def _init_log_file(self):
        """Initialize the CSV log file with headers."""
        headers = [
            "epoch", "global_step", "loss_key", "loss_non_key",
            "l_iso_key", "l_rigid_key", "l_rot_key", "l_vel_key", "l_acc_key",
            "l_iso_nk", "l_rigid_nk", "l_rot_nk", "l_vel_nk", "l_acc_nk",
            "u_min", "u_median", "u_max", "u_key_min", "u_key_median", "u_key_max",
            "u_nk_min", "u_nk_median", "u_nk_max",
            "pos_dev_key_mean", "pos_dev_key_max", "pos_dev_nk_mean", "pos_dev_nk_max",
            "unconverged_count", "key_count", "nonkey_count"
        ]
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        guru.info(f"Diagnostic log initialized → {self.log_file}")

    def log_diagnostics(self, epoch: int, loss_dict: Dict[str, Any]):
        """Write comprehensive diagnostics to the log file.
        
        Args:
            epoch: Current epoch number.
            loss_dict: Dictionary of loss components (output from compute_usplat4d_losses_detailed).
        """
        row = [
            epoch,
            self.som_trainer.global_step,
            loss_dict.get("loss_key", 0.0),
            loss_dict.get("loss_non_key", 0.0),
            loss_dict.get("l_iso_key", 0.0),
            loss_dict.get("l_rigid_key", 0.0),
            loss_dict.get("l_rot_key", 0.0),
            loss_dict.get("l_vel_key", 0.0),
            loss_dict.get("l_acc_key", 0.0),
            loss_dict.get("l_iso_nk", 0.0),
            loss_dict.get("l_rigid_nk", 0.0),
            loss_dict.get("l_rot_nk", 0.0),
            loss_dict.get("l_vel_nk", 0.0),
            loss_dict.get("l_acc_nk", 0.0),
            loss_dict.get("u_min", 0.0),
            loss_dict.get("u_median", 0.0),
            loss_dict.get("u_max", 0.0),
            loss_dict.get("u_key_min", 0.0),
            loss_dict.get("u_key_median", 0.0),
            loss_dict.get("u_key_max", 0.0),
            loss_dict.get("u_nk_min", 0.0),
            loss_dict.get("u_nk_median", 0.0),
            loss_dict.get("u_nk_max", 0.0),
            loss_dict.get("pos_dev_key_mean", 0.0),
            loss_dict.get("pos_dev_key_max", 0.0),
            loss_dict.get("pos_dev_nk_mean", 0.0),
            loss_dict.get("pos_dev_nk_max", 0.0),
            loss_dict.get("unconverged_count", 0),
            loss_dict.get("key_count", 0),
            loss_dict.get("nonkey_count", 0),
        ]
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    # ----------------------------------------------------------------------- #
    #  Core loss injection                                                      #
    # ----------------------------------------------------------------------- #

    def compute_usplat4d_losses_detailed(self, ts: Tensor) -> Tuple[Tensor, Dict[str, float]]:
        """Compute USplat4D losses with detailed component breakdown.
        
        Returns:
            (total_loss, detailed_stats_dict)
        """
        B = ts.shape[0]
        device = self.device

        key_idx = self.graph.key_idx        # (N_k,)
        nonkey_idx = self.graph.nonkey_idx  # (N_n,)

        # ---- Current positions (differentiable) ---- #
        means_fg_t, quats_fg_t = self.model.compute_poses_fg(ts)
        
        pos_key_t  = means_fg_t[key_idx]    # (N_k, B, 3)
        quats_key_t = quats_fg_t[key_idx]   # (N_k, B, 4)
        pos_nk_t   = means_fg_t[nonkey_idx] # (N_n, B, 3)
        quats_nk_t = quats_fg_t[nonkey_idx] # (N_n, B, 4)

        transforms_key = self.model.compute_transforms(ts, inds=key_idx)  # (N_k, B, 3, 4)
        transforms_nk  = self.model.compute_transforms(ts, inds=nonkey_idx)  # (N_n, B, 3, 4)

        R_key_t = transforms_key[..., :3, :3]  # (N_k, B, 3, 3)
        t_key_t = transforms_key[..., :3, 3]   # (N_k, B, 3)

        # ---- Pretrained (frozen) reference positions ---- #
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

        # ---- Key node neighbors ---- #
        key_nbrs_local   = self.graph.key_nbrs          # (N_k, K)
        key_nbr_weights  = self.graph.key_nbr_weights   # (N_k, K)

        # ---- Non-key node neighbors ---- #
        nonkey_nbrs_local   = self.graph.nonkey_nbrs         # (N_n, K+1)
        nonkey_nbr_weights  = self.graph.nonkey_nbr_weights  # (N_n, K+1)

        # Compute key and non-key losses using the high-level functions
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

        l_non_key = non_key_node_loss(
            pos_nk_t=pos_nk_t,
            quats_nk_t=quats_nk_t,
            transforms_nk_t=transforms_nk,
            pos_nk_pretrained=pos_nk_pre,
            u_nk=u_nk,
            R_wc_t=R_wc_t,
            pos_o_nk=pos_o_nk,
            R_key_t=R_key_t,
            t_key_t=t_key_t,
            pos_key_t=pos_key_t,
            quats_key_t=quats_key_t,
            transforms_key_t=transforms_key,
            pos_o_key=pos_o_key,
            nonkey_nbrs_local=nonkey_nbrs_local,
            nonkey_nbr_weights=nonkey_nbr_weights,
            nonkey_nbrs_global=nonkey_nbrs_local,
            r_scale=self.cfg.r_scale,
            lambda_iso=self.cfg.lambda_iso,
            lambda_rigid=self.cfg.lambda_rigid,
            lambda_rot=self.cfg.lambda_rot,
            lambda_vel=self.cfg.lambda_vel,
            lambda_acc=self.cfg.lambda_acc,
        )

        # Compute individual motion loss components for key nodes
        with torch.no_grad():
            l_iso_key = isometry_loss(
                pos_t=pos_key_t, pos_o=pos_o_key,
                nbr_idx=key_nbrs_local, weights=key_nbr_weights
            )
            l_rigid_key = rigidity_loss(
                pos_t=pos_key_t, transforms_t=transforms_key,
                nbr_idx=key_nbrs_local, weights=key_nbr_weights
            )
            l_rot_key = rotation_loss(
                quats_t=quats_key_t, nbr_idx=key_nbrs_local, weights=key_nbr_weights
            )
            l_vel_key = velocity_loss(
                pos_t=pos_key_t, quats_t=quats_key_t, delta=1
            )
            l_acc_key = acceleration_loss(
                pos_t=pos_key_t, quats_t=quats_key_t, delta=1
            )

            # Compute individual motion loss components for non-key nodes
            l_iso_nk = isometry_loss(
                pos_t=pos_nk_t, pos_o=pos_o_nk,
                nbr_idx=nonkey_nbrs_local, weights=nonkey_nbr_weights,
                pos_nbr_t=pos_key_t, pos_o_nbr=pos_o_key
            )
            l_rigid_nk = rigidity_loss(
                pos_t=pos_nk_t, transforms_t=transforms_nk,
                nbr_idx=nonkey_nbrs_local, weights=nonkey_nbr_weights,
                pos_nbr_t=pos_key_t, transforms_nbr_t=transforms_key
            )
            l_rot_nk = rotation_loss(
                quats_t=quats_nk_t, nbr_idx=nonkey_nbrs_local, weights=nonkey_nbr_weights,
                quats_nbr_t=quats_key_t
            )
            l_vel_nk = velocity_loss(
                pos_t=pos_nk_t, quats_t=quats_nk_t, delta=1
            )
            l_acc_nk = acceleration_loss(
                pos_t=pos_nk_t, quats_t=quats_nk_t, delta=1
            )

            # Uncertainty statistics
            u_all = self.u_scalar
            unconverged_count = (self.u_scalar >= self.cfg.phi * 0.99).sum().item()

            # Position deviation statistics
            pos_dev_key = (pos_key_t - pos_key_pre).norm(dim=-1)  # (N_k, B)
            pos_dev_nk = (pos_nk_t - pos_nk_pre).norm(dim=-1)     # (N_n, B)

        total_loss = self.cfg.lambda_key * l_key + self.cfg.lambda_non_key * l_non_key

        stats = {
            "loss_total": total_loss.item(),
            "loss_key": l_key.item(),
            "loss_non_key": l_non_key.item(),
            "l_iso_key": l_iso_key.item(),
            "l_rigid_key": l_rigid_key.item(),
            "l_rot_key": l_rot_key.item(),
            "l_vel_key": l_vel_key.item(),
            "l_acc_key": l_acc_key.item(),
            "l_iso_nk": l_iso_nk.item(),
            "l_rigid_nk": l_rigid_nk.item(),
            "l_rot_nk": l_rot_nk.item(),
            "l_vel_nk": l_vel_nk.item(),
            "l_acc_nk": l_acc_nk.item(),
            "u_min": u_all.min().item(),
            "u_median": u_all.median().item(),
            "u_max": u_all.max().item(),
            "u_key_min": u_key.min().item(),
            "u_key_median": u_key.median().item(),
            "u_key_max": u_key.max().item(),
            "u_nk_min": u_nk.min().item(),
            "u_nk_median": u_nk.median().item(),
            "u_nk_max": u_nk.max().item(),
            "pos_dev_key_mean": pos_dev_key.mean().item(),
            "pos_dev_key_max": pos_dev_key.max().item(),
            "pos_dev_nk_mean": pos_dev_nk.mean().item(),
            "pos_dev_nk_max": pos_dev_nk.max().item(),
            "unconverged_count": unconverged_count,
            "key_count": len(key_idx),
            "nonkey_count": len(nonkey_idx),
        }

        return total_loss, stats

    def compute_usplat4d_losses(self, ts: Tensor) -> Tuple[Tensor, Dict[str, float]]:
        """Wrapper around compute_usplat4d_losses_detailed for backward compatibility."""
        return self.compute_usplat4d_losses_detailed(ts)

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
        log_every_n_steps: int = 10,
    ):
        """Fine-tune the SoM model with USplat4D graph losses.

        Monkey-patches som_trainer.compute_losses so density control and all
        existing SoM infrastructure continues to work unchanged.

        Args:
            train_loader: DataLoader yielding SoM-style batches.
            extra_epochs: Override cfg.extra_epochs if provided.
            disable_density_control_fraction: Fraction of epochs at the START
                (and last 20%) during which density control is paused.
            log_every_n_steps: Log diagnostics every N training steps.
        """
        n_epochs = extra_epochs or self.cfg.extra_epochs
        start_epoch = self.som_trainer.epoch

        # Patch compute_losses
        original_compute_losses = self.som_trainer.compute_losses
        self.som_trainer.compute_losses = self.compute_losses_with_graph

        guru.info(f"USplat4D: starting fine-tuning for {n_epochs} extra epochs "
                  f"(from epoch {start_epoch})")
        guru.info(f"USplat4D: diagnostic log → {self.log_file}")

        epoch_bar = tqdm(
            range(start_epoch, start_epoch + n_epochs),
            desc="USplat4D fine-tuning",
            unit="epoch",
            dynamic_ncols=True,
        )
        step_counter = 0
        
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

                    # Log diagnostics periodically
                    if step_counter % log_every_n_steps == 0:
                        ts = batch["ts"].to(self.device)
                        with torch.no_grad():
                            _, loss_dict = self.compute_usplat4d_losses_detailed(ts)
                        self.log_diagnostics(epoch, loss_dict)
                    
                    step_counter += 1

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

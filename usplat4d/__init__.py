"""USplat4D: Uncertainty-Aware Dynamic Gaussian Splatting (ICLR 2026).

This module refines 3D Gaussians from a pretrained Shape-of-Motion (SoM) model
using uncertainty-guided spatio-temporal graph optimization.

Paper: https://arxiv.org/abs/2510.12768
"""

from usplat4d.uncertainty import compute_uncertainty_all_frames
from usplat4d.graph import build_graph, USplat4DGraph
from usplat4d.losses import (
    key_node_loss,
    non_key_node_loss,
    motion_loss_key,
    motion_loss_non_key,
)
from usplat4d.trainer import USplat4DTrainer, USplat4DConfig

__all__ = [
    "compute_uncertainty_all_frames",
    "build_graph",
    "USplat4DGraph",
    "key_node_loss",
    "non_key_node_loss",
    "motion_loss_key",
    "motion_loss_non_key",
    "USplat4DTrainer",
    "USplat4DConfig",
]

"""Uncertainty-Encoded Graph Construction (Section 4.2 of USplat4D paper).

Key steps:
  1. Key node selection via 3D voxel gridization + significant period filter
  2. Key-key edges via Uncertainty-Aware kNN (UA-kNN, Eq. 9)
  3. Non-key → key assignment via minimum aggregated Mahalanobis distance (Eq. 10)

Graph outputs are stored in USplat4DGraph, which is used by the trainer
to compute graph losses for both key and non-key Gaussians.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class USplat4DGraph:
    """Stores the spatio-temporal graph for uncertainty-aware optimization.

    Attributes:
        key_idx:   (N_k,)       Gaussian indices of key nodes
        nonkey_idx:(N_n,)       Gaussian indices of non-key nodes
        key_nbrs:  (N_k, K)     For each key node, indices into key_idx of its K key neighbors
        key_nbr_weights: (N_k, K)   Edge weights for key-key graph (normalized)
        nonkey_key_idx:  (N_n,) For each non-key node, index into key_idx of its assigned key node
        nonkey_nbrs:     (N_n, K+1)  Neighbors inherited from assigned key + the key itself (indices into key_idx)
        nonkey_nbr_weights: (N_n, K+1)  Normalized DQB blending weights for non-key nodes
        num_key:   int
        num_nonkey:int
    """
    key_idx:            Tensor   # (N_k,)
    nonkey_idx:         Tensor   # (N_n,)
    key_nbrs:           Tensor   # (N_k, K)
    key_nbr_weights:    Tensor   # (N_k, K)
    nonkey_key_idx:     Tensor   # (N_n,)   index into key_idx
    nonkey_nbrs:        Tensor   # (N_n, K+1)  indices into key_idx
    nonkey_nbr_weights: Tensor   # (N_n, K+1)

    @property
    def num_key(self) -> int:
        return self.key_idx.shape[0]

    @property
    def num_nonkey(self) -> int:
        return self.nonkey_idx.shape[0]


def build_graph(
    means_t: Tensor,      # (G, T, 3)   per-Gaussian per-frame world positions
    u_scalar: Tensor,     # (G, T)      per-Gaussian per-frame uncertainties
    key_ratio: float = 0.02,       # fraction of Gaussians treated as key nodes (~top 2%)
    spt_threshold: int = 5,        # significant period threshold (min #frames with u < tau)
    voxel_size: Optional[float] = None,  # 3D voxel size; auto if None
    knn_k: int = 8,                # number of key-key neighbors per key node
    u_tau_percentile: float = 0.20, # uncertainty percentile below which a Gaussian is "low-u"
    device: Optional[torch.device] = None,
) -> USplat4DGraph:
    """Build the uncertainty-encoded spatio-temporal graph.

    Args:
        means_t:   (G, T, 3)  Gaussian positions (world space) for all frames
        u_scalar:  (G, T)     scalar uncertainties from compute_uncertainty_all_frames
        key_ratio: target fraction of key nodes relative to G
        spt_threshold: minimum number of frames where u_i < tau for a candidate to be key
        voxel_size: size of 3D voxels for candidate sampling; auto-computed if None
        knn_k: number of nearest key neighbors per key node (UA-kNN)
        u_tau_percentile: percentile of u used as low-uncertainty threshold tau

    Returns:
        USplat4DGraph  ready for use in the training loop
    """
    if device is None:
        device = means_t.device
    G, T, _ = means_t.shape
    means_t = means_t.to(device)
    u_scalar = u_scalar.to(device)

    # ---- 1. Key node selection ----
    # Low-uncertainty threshold tau: below this, a Gaussian is "reliably observed"
    finite_u = u_scalar[u_scalar.isfinite()]
    if finite_u.numel() == 0:
        raise RuntimeError("All uncertainties are non-finite; check rendering output.")
    tau = torch.quantile(finite_u, u_tau_percentile).item()

    low_u_mask = u_scalar < tau  # (G, T)  True where Gaussian is reliable

    # Stage 1: Candidate sampling via 3D voxel gridization
    # Use canonical positions (mean over all frames) for voxelization
    means_canon = means_t.mean(dim=1)   # (G, 3) approximate canonical position

    # Auto voxel size: scale scene so that ~target number of voxels are filled
    if voxel_size is None:
        # Aim for approximately G * key_ratio voxels, spread over the scene extent
        target_n_voxels = G * key_ratio * 5   # slightly larger to account for empty voxels
        scene_range = (means_canon.max(dim=0).values - means_canon.min(dim=0).values)
        scene_vol = (scene_range + 1e-6).prod().item()
        voxel_size = float((scene_vol / max(target_n_voxels, 1)) ** (1.0 / 3.0))
        voxel_size = max(voxel_size, 1e-4)

    # Assign each Gaussian to a voxel by discretizing its canonical position
    vox_min = means_canon.min(dim=0).values   # (3,)
    vox_ids = ((means_canon - vox_min) / voxel_size).long()  # (G, 3)

    # Hash voxel ID to a single integer: hash = x + D*y + D^2*z  (D large prime)
    D = 100003
    vox_hash = vox_ids[:, 0] + D * vox_ids[:, 1] + D * D * vox_ids[:, 2]  # (G,)

    # Per voxel: discard if only high-uncertainty Gaussians; sample 1 low-u Gaussian
    unique_vox, vox_inv = torch.unique(vox_hash, return_inverse=True)  # N_vox
    candidates = []
    for vid in range(unique_vox.shape[0]):
        in_vox = (vox_inv == vid).nonzero(as_tuple=True)[0]  # Gaussian indices in voxel
        # Check if any low-uncertainty Gaussian exists in this voxel (at any frame)
        low_u_in_vox = low_u_mask[in_vox].any(dim=1)  # (n_in_vox,) True if ever low-u
        good = in_vox[low_u_in_vox]
        if good.numel() == 0:
            continue   # all high-uncertainty → skip
        # Randomly pick one from this voxel
        pick = good[torch.randint(0, good.numel(), (1,), device=device)]
        candidates.append(pick)

    if len(candidates) == 0:
        raise RuntimeError("No key node candidates found; try relaxing u_tau_percentile.")
    candidates = torch.cat(candidates)  # (N_cand,)

    # Stage 2: Filter by significant period >= spt_threshold
    # Significant period = number of frames where u < tau
    sig_period = low_u_mask[candidates].sum(dim=1)  # (N_cand,)
    valid = sig_period >= spt_threshold
    candidates = candidates[valid]

    if candidates.numel() == 0:
        raise RuntimeError(
            f"No key nodes survive SPT={spt_threshold} filter. "
            "Try reducing spt_threshold."
        )

    # Keep at most G * key_ratio key nodes (top-confidence)
    max_key = max(int(G * key_ratio), 1)
    if candidates.numel() > max_key:
        # Rank by average uncertainty (ascending = most reliable first)
        mean_u = u_scalar[candidates].mean(dim=1)
        top = mean_u.topk(max_key, largest=False).indices
        candidates = candidates[top]

    key_idx = candidates   # (N_k,)

    # Non-key = all Gaussians not in key set
    is_key = torch.zeros(G, dtype=torch.bool, device=device)
    is_key[key_idx] = True
    nonkey_idx = (~is_key).nonzero(as_tuple=True)[0]  # (N_n,)

    # ---- 2. Key-key edges: UA-kNN (Eq. 9) ----
    # For each key node i, select K neighbors among other key nodes,
    # evaluated at best frame t_hat_i = argmin_t u_{i,t},
    # using Mahalanobis distance (weighted by U_i + U_j).
    N_k = key_idx.shape[0]
    k_eff = min(knn_k, N_k - 1)   # can't have more neighbors than other key nodes

    key_means_all = means_t[key_idx]    # (N_k, T, 3)
    key_u_all     = u_scalar[key_idx]   # (N_k, T)

    # Best frame per key node
    best_t = key_u_all.argmin(dim=1)   # (N_k,)
    key_pos_at_best = key_means_all[torch.arange(N_k, device=device), best_t]  # (N_k, 3)
    key_u_at_best   = key_u_all[torch.arange(N_k, device=device), best_t]       # (N_k,)

    # UA-kNN: distance(i, j) = ||p_i - p_j|| / sqrt(u_i + u_j)  (Eq. 9 — Mahalanobis)
    # We use isotropic approximation (scalar uncertainty) for efficiency.
    # Eq. 9 uses the full matrix U_i + U_j, here approximated as scalar u_i + u_j.
    diff = key_pos_at_best.unsqueeze(1) - key_pos_at_best.unsqueeze(0)  # (N_k, N_k, 3)
    dist_sq = (diff ** 2).sum(dim=-1)   # (N_k, N_k)
    u_sum = key_u_at_best.unsqueeze(1) + key_u_at_best.unsqueeze(0) + 1e-8  # (N_k, N_k)
    mahala_dist = dist_sq / u_sum       # (N_k, N_k)  UA-kNN distance

    # Exclude self (set diagonal to infinity)
    mahala_dist.fill_diagonal_(float("inf"))

    if k_eff > 0:
        _, key_nbr_local = mahala_dist.topk(k_eff, dim=-1, largest=False)  # (N_k, K)
    else:
        key_nbr_local = torch.zeros(N_k, 1, dtype=torch.long, device=device)
        k_eff = 1

    # Edge weights: inverse UA distance (softmax-normalized per node)
    nbr_dists = mahala_dist.gather(1, key_nbr_local)  # (N_k, K)
    nbr_dists_clamped = nbr_dists.clamp(min=1e-8)
    key_nbr_weights = F.softmax(-nbr_dists_clamped, dim=-1)  # (N_k, K)

    # ---- 3. Non-key → key assignment (Eq. 10) ----
    # Assign each non-key node i to key node j that minimizes
    # sum_t ||p_{i,t} - p_{j,t}||_{U_i^t + U_j^t} (scalar Mahalanobis approx).
    N_n = nonkey_idx.shape[0]
    nk_means_all = means_t[nonkey_idx]   # (N_n, T, 3)
    nk_u_all     = u_scalar[nonkey_idx]  # (N_n, T)

    # Compute aggregated distance: sum over time of Mahalanobis distance
    # dist[i, j] = sum_t (||nk_means_all[i,t] - key_means_all[j,t]||^2 / (nk_u[i,t] + key_u[j,t]))
    # Full (N_n, N_k, T) computation — may be large; process in chunks for memory
    CHUNK = 512
    agg_dist = torch.zeros(N_n, N_k, device=device)
    for start in range(0, N_n, CHUNK):
        end = min(start + CHUNK, N_n)
        # diff: (chunk, N_k, T, 3)
        d = (nk_means_all[start:end, None, :, :]       # (chunk, 1, T, 3)
             - key_means_all[None, :, :, :])            # (1, N_k, T, 3)
        d_sq = (d ** 2).sum(dim=-1)                    # (chunk, N_k, T)
        u_s = (nk_u_all[start:end, None, :]            # (chunk, 1, T)
               + key_u_all[None, :, :] + 1e-8)         # (1, N_k, T)
        agg_dist[start:end] = (d_sq / u_s).sum(dim=-1) # (chunk, N_k)

    nonkey_key_local = agg_dist.argmin(dim=-1)   # (N_n,)  index into key_idx

    # Non-key inherits assigned key node's edges + the key node itself (ε_i = ε_j ∪ {j})
    assigned_key_nbrs = key_nbr_local[nonkey_key_local]  # (N_n, K) neighbors of assigned key
    # Append the assigned key node itself as an extra neighbor
    extra = nonkey_key_local.unsqueeze(-1)               # (N_n, 1)
    nonkey_nbrs = torch.cat([assigned_key_nbrs, extra], dim=-1)  # (N_n, K+1)

    # DQB blending weights: inverse Mahalanobis distance to each neighbor
    # at the best frame of the non-key node
    nk_best_t = nk_u_all.argmin(dim=1)  # (N_n,)
    nk_pos_best = nk_means_all[torch.arange(N_n, device=device), nk_best_t]  # (N_n, 3)
    nk_u_best   = nk_u_all[torch.arange(N_n, device=device), nk_best_t]      # (N_n,)

    # Neighbor key positions at non-key node's best frame
    # nonkey_nbrs: (N_n, K+1) indices into key_idx (local)
    nbr_key_global = key_idx[nonkey_nbrs.reshape(-1)].reshape(N_n, k_eff + 1)  # (N_n, K+1) global idx
    nbr_t = nk_best_t.unsqueeze(-1).expand(-1, k_eff + 1)                      # (N_n, K+1)
    nbr_pos = means_t[nbr_key_global.reshape(-1), nbr_t.reshape(-1)].reshape(N_n, k_eff + 1, 3)  # (N_n, K+1, 3)
    nbr_u   = u_scalar[nbr_key_global.reshape(-1), nbr_t.reshape(-1)].reshape(N_n, k_eff + 1)    # (N_n, K+1)

    diff_nk = nk_pos_best.unsqueeze(1) - nbr_pos           # (N_n, K+1, 3)
    dist_sq_nk = (diff_nk ** 2).sum(dim=-1)                # (N_n, K+1)
    u_sum_nk = nk_u_best.unsqueeze(1) + nbr_u + 1e-8       # (N_n, K+1)
    mahala_nk = dist_sq_nk / u_sum_nk                      # (N_n, K+1)
    nonkey_nbr_weights = F.softmax(-mahala_nk.clamp(min=1e-8), dim=-1)  # (N_n, K+1)

    return USplat4DGraph(
        key_idx=key_idx,
        nonkey_idx=nonkey_idx,
        key_nbrs=key_nbr_local,           # (N_k, K) indices into key_idx
        key_nbr_weights=key_nbr_weights,  # (N_k, K)
        nonkey_key_idx=nonkey_key_local,  # (N_n,)   index into key_idx
        nonkey_nbrs=nonkey_nbrs,          # (N_n, K+1) indices into key_idx
        nonkey_nbr_weights=nonkey_nbr_weights,  # (N_n, K+1)
    )

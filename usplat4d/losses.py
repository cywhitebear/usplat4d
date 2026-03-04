"""Loss functions for USplat4D (Section 4.3 and Appendix A.2 of the paper).

Loss hierarchy:
    L_total = L_rgb + L_key + L_non_key                              (Eq. 14)

    L_key = sum_{t,i∈V_k} ||p_{i,t} - p^o_{i,t}||^2_{U^{-1}}
            + L_motion_key                                            (Eq. 11)

    L_non_key = sum_{t,i∈V_n} ||p_{i,t} - p^o_{i,t}||^2_{U^{-1}}
              + sum_{t,i∈V_n} ||p_{i,t} - p^DQB_{i,t}||^2_{U^{-1}}
              + L_motion_non_key                                      (Eq. 13)

Motion losses (Appendix A.2.1):
    L_motion = λ_iso * L_iso + λ_rigid * L_rigid + λ_rot * L_rot
             + λ_vel * L_vel + λ_acc * L_acc                        (Eq. S13)

Default weights: λ_iso = λ_rigid = 1.0,  λ_rot = λ_vel = λ_acc = 0.01

All Mahalanobis distances use the anisotropic U_{i,t} from uncertainty.py.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from usplat4d.uncertainty import mahalanobis_sq
from usplat4d.dqb import apply_dqb_to_batch


# --------------------------------------------------------------------------- #
#  Motion losses (Appendix A.2.1, Eqs. S8–S13)
# --------------------------------------------------------------------------- #

def isometry_loss(
    pos_t:   Tensor,    # (N, B, 3)   current positions for N Gaussians, B frames
    pos_o:   Tensor,    # (N, 3)      canonical (initial) positions
    nbr_idx: Tensor,    # (N, K)      neighbor indices (0..N-1 within the N Gaussians)
    weights: Tensor,    # (N, K)      edge weights
) -> Tensor:
    """Isometry loss L_iso (Eq. S8): canonical distances should be preserved over time.

    L_iso = 1/(K*N) * sum_t sum_i sum_j w_{ij} * (||p^o_j - p^o_i||^2 - ||p_{j,t} - p_{i,t}||^2)
    """
    # Canonical pairwise distances squared
    p_i_o = pos_o.unsqueeze(1)                   # (N, 1, 3)
    p_j_o = pos_o[nbr_idx]                       # (N, K, 3)
    dist_o_sq = ((p_i_o - p_j_o) ** 2).sum(-1)  # (N, K)

    # Time-varying pairwise distances squared
    # pos_t[nbr_idx]: (N, K, B, 3)  (fancy index on first dim)
    p_j_t = pos_t[nbr_idx].permute(0, 2, 1, 3)  # (N, B, K, 3)
    p_i_t = pos_t.unsqueeze(2)                   # (N, B, 1, 3)
    dist_t_sq = ((p_i_t - p_j_t) ** 2).sum(-1)  # (N, B, K)

    # (dist_o^2 - dist_t^2)  shape: need to broadcast dist_o_sq (N, K) → (N, 1, K)
    diff = dist_o_sq.unsqueeze(1) - dist_t_sq             # (N, B, K)
    loss = (weights.unsqueeze(1) * diff).abs().mean()
    return loss


def rigidity_loss(
    pos_t:    Tensor,       # (N, B, 3)   current positions
    transforms_t: Tensor,  # (N, B, 3, 4)  SE(3) transforms (canonical→world)
    nbr_idx:  Tensor,       # (N, K)
    weights:  Tensor,       # (N, K)
    delta: int = 1,         # frame interval Δ
) -> Tensor:
    """Rigidity loss L_rigid (Eq. S9): neighbors should move rigidly together.

    L_rigid = 1/(K*N) * sum_{t=Δ}^{T-1} sum_i sum_j w_{ij}
              * ||p_{j,t-Δ} - T_{i,t-Δ} T_{i,t}^{-1} p_{j,t}||^2

    T_{i,t-Δ} T_{i,t}^{-1} p_{j,t}
        = R_{i,t-Δ} @ (R_{i,t}^T @ (p_{j,t} - t_{i,t})) + t_{i,t-Δ}
    """
    B = pos_t.shape[1]
    if B <= delta:
        return pos_t.new_zeros(())

    # Split into (t-Δ) and t parts (along frame batch dimension)
    pos_prev = pos_t[:, :B - delta]          # (N, B-Δ, 3)
    T_prev = transforms_t[:, :B - delta]     # (N, B-Δ, 3, 4)
    T_curr = transforms_t[:, delta:]         # (N, B-Δ, 3, 4)
    pos_curr = pos_t[:, delta:]              # (N, B-Δ, 3)

    R_prev = T_prev[..., :3, :3]  # (N, B-Δ, 3, 3)
    t_prev = T_prev[..., :3, 3]   # (N, B-Δ, 3)
    R_curr = T_curr[..., :3, :3]  # (N, B-Δ, 3, 3)
    t_curr = T_curr[..., :3, 3]   # (N, B-Δ, 3)

    # For each neighbor j of i: compute T_{i,t-Δ} T_{i,t}^{-1} p_{j,t}
    # step 1: T_{i,t}^{-1} p_{j,t} = R_curr^T @ (p_{j,t} - t_curr)
    # step 2: T_{i,t-Δ} @ result = R_prev @ result + t_prev

    # Neighbor positions at frame t (pos_curr) and t-Δ (pos_prev)
    p_j_curr = pos_curr[nbr_idx]             # (N, K, B-Δ, 3)
    p_j_curr = p_j_curr.permute(0, 2, 1, 3) # (N, B-Δ, K, 3)
    p_j_prev = pos_prev[nbr_idx]             # (N, K, B-Δ, 3)
    p_j_prev = p_j_prev.permute(0, 2, 1, 3) # (N, B-Δ, K, 3)

    # Broadcast transforms: (N, B-Δ, 3, 3) → (N, B-Δ, 1, 3, 3) for K neighbors
    R_curr_exp = R_curr.unsqueeze(2)   # (N, B-Δ, 1, 3, 3)
    t_curr_exp = t_curr.unsqueeze(2)   # (N, B-Δ, 1, 3)
    R_prev_exp = R_prev.unsqueeze(2)   # (N, B-Δ, 1, 3, 3)
    t_prev_exp = t_prev.unsqueeze(2)   # (N, B-Δ, 1, 3)

    # T_curr^{-1} p_j_curr:
    step1 = (R_curr_exp.transpose(-1, -2) @
             (p_j_curr - t_curr_exp).unsqueeze(-1)).squeeze(-1)  # (N, B-Δ, K, 3)

    # T_prev @ step1:
    pred = (R_prev_exp @ step1.unsqueeze(-1)).squeeze(-1) + t_prev_exp  # (N, B-Δ, K, 3)

    diff_sq = ((p_j_prev - pred) ** 2).sum(-1)  # (N, B-Δ, K)
    loss = (weights.unsqueeze(1).expand_as(diff_sq) * diff_sq).mean()
    return loss


def rotation_loss(
    quats_t: Tensor,    # (N, B, 4)   quaternions (w,x,y,z) at sampled frames
    nbr_idx: Tensor,    # (N, K)
    weights: Tensor,    # (N, K)
    delta: int = 1,
) -> Tensor:
    """Relative rotation loss L_rot (Eq. S10).

    L_rot = 1/(K*N) * sum_{t=Δ}^{T-1} sum_i sum_j w_{ij}
            * ||q_{j,t} q_{j,t-Δ}^{-1} - q_{i,t} q_{i,t-Δ}^{-1}||^2

    Relative rotation of j (from t-Δ to t) should match that of i.
    """
    B = quats_t.shape[1]
    if B <= delta:
        return quats_t.new_zeros(())

    q_prev = quats_t[:, :B - delta]   # (N, B-Δ, 4)
    q_curr = quats_t[:, delta:]       # (N, B-Δ, 4)

    # Quaternion inverse (unit quat): conjugate
    q_prev_inv = torch.cat([q_prev[..., :1], -q_prev[..., 1:]], dim=-1)  # (N, B-Δ, 4)

    def quat_mul_batch(a, b):
        """Hamilton product for (..., 4) (w,x,y,z)."""
        w1, x1, y1, z1 = a.unbind(-1)
        w2, x2, y2, z2 = b.unbind(-1)
        return torch.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ], dim=-1)

    # Relative rotation for node i: q_i_rel = q_curr_i * q_prev_i^{-1}
    # (row: N, col: B-Δ, 4)
    q_i_rel = quat_mul_batch(q_curr, q_prev_inv)          # (N, B-Δ, 4)

    # Neighbor quats at t and t-Δ  (index q_curr/q_prev, not full quats_t)
    q_j_curr = q_curr[nbr_idx].permute(0, 2, 1, 3)        # (N, B-Δ, K, 4)
    q_j_prev = q_prev[nbr_idx].permute(0, 2, 1, 3)        # (N, B-Δ, K, 4)
    q_j_prev_inv = torch.cat([q_j_prev[..., :1], -q_j_prev[..., 1:]], dim=-1)
    q_j_rel = quat_mul_batch(q_j_curr, q_j_prev_inv)      # (N, B-Δ, K, 4)

    # Difference: ||q_j_rel - q_i_rel||^2   (broadcast i over K)
    diff = q_j_rel - q_i_rel.unsqueeze(2)                 # (N, B-Δ, K, 4)
    diff_sq = (diff ** 2).sum(-1)                         # (N, B-Δ, K)
    loss = (weights.unsqueeze(1).expand_as(diff_sq) * diff_sq).mean()
    return loss


def velocity_loss(
    pos_t:  Tensor,     # (N, B, 3)
    quats_t: Tensor,    # (N, B, 4)
    delta: int = 1,
) -> Tensor:
    """Velocity loss L_vel (Eq. S11): penalize large positional/rotational jumps.

    L_vel = sum_{t=1}^{T-1} sum_i ( ||p_{i,t-1} - p_{i,t}||_1
                                   + ||q_{i,t-1} * q_{i,t}^{-1}||_1 )
    """
    B = pos_t.shape[1]
    if B <= delta:
        return pos_t.new_zeros(())
    dp = (pos_t[:, :B - delta] - pos_t[:, delta:]).abs().sum(-1).mean()

    q_prev = quats_t[:, :B - delta]
    q_curr = quats_t[:, delta:]
    q_curr_inv = torch.cat([q_curr[..., :1], -q_curr[..., 1:]], dim=-1)
    w1, x1, y1, z1 = q_prev.unbind(-1)
    w2, x2, y2, z2 = q_curr_inv.unbind(-1)
    q_rel = torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)
    dq = q_rel.abs().sum(-1).mean()
    return dp + dq


def acceleration_loss(
    pos_t:   Tensor,    # (N, B, 3)
    quats_t: Tensor,    # (N, B, 4)
    delta: int = 1,
) -> Tensor:
    """Acceleration loss L_acc (Eq. S12): penalize large accelerations.

    L_acc = sum_{t=2}^{T-1} sum_i ( ||p_{i,t-2} - 2*p_{i,t-1} + p_{i,t}||_1
                                   + ||q_term||_1 )
    """
    B = pos_t.shape[1]
    if B < 3:
        return pos_t.new_zeros(())

    # Second-order finite difference for position
    p0 = pos_t[:, :B - 2 * delta]
    p1 = pos_t[:, delta:B - delta]
    p2 = pos_t[:, 2 * delta:]
    acc_p = (p0 - 2 * p1 + p2).abs().sum(-1).mean()

    # Second-order finite difference for rotation (using relative rotation)
    def qmul(a, b):
        w1, x1, y1, z1 = a.unbind(-1)
        w2, x2, y2, z2 = b.unbind(-1)
        return torch.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ], dim=-1)

    q0 = quats_t[:, :B - 2 * delta]
    q1 = quats_t[:, delta:B - delta]
    q2 = quats_t[:, 2 * delta:]

    q1_inv = torch.cat([q1[..., :1], -q1[..., 1:]], dim=-1)
    # term = q_{t-2} * q_{t-1}^{-1} * (q_{t-1} * q_t^{-1})^{-1}
    rel01 = qmul(q0, q1_inv)
    q2_inv = torch.cat([q2[..., :1], -q2[..., 1:]], dim=-1)
    rel12 = qmul(q1, q2_inv)
    rel12_inv = torch.cat([rel12[..., :1], -rel12[..., 1:]], dim=-1)
    acc_q = qmul(rel01, rel12_inv).abs().sum(-1).mean()

    return acc_p + acc_q


def motion_loss_key(
    pos_t:        Tensor,    # (N_k, B, 3)
    quats_t:      Tensor,    # (N_k, B, 4)
    transforms_t: Tensor,    # (N_k, B, 3, 4)  SE(3) transforms
    pos_o:        Tensor,    # (N_k, 3)  canonical positions
    key_nbrs_local: Tensor,  # (N_k, K)  neighbor indices (0..N_k-1)
    key_nbr_weights: Tensor, # (N_k, K)
    lambda_iso: float = 1.0,
    lambda_rigid: float = 1.0,
    lambda_rot: float = 0.01,
    lambda_vel: float = 0.01,
    lambda_acc: float = 0.01,
) -> Tensor:
    """Combined motion loss for key nodes (L_motion_key in Eq. 11)."""
    loss = pos_t.new_zeros(())
    if lambda_iso > 0:
        loss = loss + lambda_iso * isometry_loss(pos_t, pos_o, key_nbrs_local, key_nbr_weights)
    if lambda_rigid > 0:
        loss = loss + lambda_rigid * rigidity_loss(pos_t, transforms_t, key_nbrs_local, key_nbr_weights)
    if lambda_rot > 0:
        loss = loss + lambda_rot * rotation_loss(quats_t, key_nbrs_local, key_nbr_weights)
    if lambda_vel > 0:
        loss = loss + lambda_vel * velocity_loss(pos_t, quats_t)
    if lambda_acc > 0:
        loss = loss + lambda_acc * acceleration_loss(pos_t, quats_t)
    return loss


def motion_loss_non_key(
    pos_t:             Tensor,   # (N_n, B, 3)
    quats_t:           Tensor,   # (N_n, B, 4)
    transforms_t:      Tensor,   # (N_n, B, 3, 4)
    pos_o:             Tensor,   # (N_n, 3) canonical positions of non-key nodes
    nonkey_nbrs_local: Tensor,   # (N_n, K+1)
    nonkey_nbr_weights: Tensor,  # (N_n, K+1)
    lambda_iso: float = 1.0,
    lambda_rigid: float = 1.0,
    lambda_rot: float = 0.01,
    lambda_vel: float = 0.01,
    lambda_acc: float = 0.01,
) -> Tensor:
    """Combined motion loss for non-key nodes (L_motion_non_key in Eq. 13)."""
    loss = pos_t.new_zeros(())
    if lambda_iso > 0:
        loss = loss + lambda_iso * isometry_loss(pos_t, pos_o, nonkey_nbrs_local, nonkey_nbr_weights)
    if lambda_rigid > 0:
        loss = loss + lambda_rigid * rigidity_loss(pos_t, transforms_t, nonkey_nbrs_local, nonkey_nbr_weights)
    if lambda_rot > 0:
        loss = loss + lambda_rot * rotation_loss(quats_t, nonkey_nbrs_local, nonkey_nbr_weights)
    if lambda_vel > 0:
        loss = loss + lambda_vel * velocity_loss(pos_t, quats_t)
    if lambda_acc > 0:
        loss = loss + lambda_acc * acceleration_loss(pos_t, quats_t)
    return loss


# --------------------------------------------------------------------------- #
#  Key node position loss (Eq. 11, first term)
# --------------------------------------------------------------------------- #

def key_node_loss(
    pos_key_t:          Tensor,  # (N_k, B, 3)  current key positions at sampled frames
    quats_key_t:        Tensor,  # (N_k, B, 4)
    transforms_key_t:   Tensor,  # (N_k, B, 3, 4)
    pos_key_pretrained: Tensor,  # (N_k, B, 3)  frozen pretrained positions
    u_key:              Tensor,  # (N_k, B)     pre-computed scalar uncertainties
    R_wc_t:             Tensor,  # (B, 3, 3)    camera-to-world rotations per sampled frame
    pos_o:              Tensor,  # (N_k, 3)     canonical positions
    key_nbrs_local:     Tensor,  # (N_k, K)
    key_nbr_weights:    Tensor,  # (N_k, K)
    r_scale: Tuple[float, float, float] = (1.0, 1.0, 0.01),
    lambda_iso: float = 1.0,
    lambda_rigid: float = 1.0,
    lambda_rot: float = 0.01,
    lambda_vel: float = 0.01,
    lambda_acc: float = 0.01,
) -> Tensor:
    """Full key node loss L_key (Eq. 11).

    L_key = sum_{t,i∈V_k} ||p_{i,t} - p^o_{i,t}||^2_{U^{-1}_{w,t,i}}
          + L_motion_key
    """
    N_k, B = pos_key_t.shape[:2]

    # --- Mahalanobis deviation from pretrained positions ---
    delta = pos_key_t - pos_key_pretrained   # (N_k, B, 3)
    # R_wc_t: (B, 3, 3) → expand to (N_k, B, 3, 3)
    R_wc = R_wc_t.unsqueeze(0).expand(N_k, -1, -1, -1)  # (N_k, B, 3, 3)
    mah = mahalanobis_sq(delta, u_key, R_wc, r_scale)    # (N_k, B)
    pos_dev_loss = mah.mean()

    # --- Motion loss ---
    mot = motion_loss_key(
        pos_t=pos_key_t,
        quats_t=quats_key_t,
        transforms_t=transforms_key_t,
        pos_o=pos_o,
        key_nbrs_local=key_nbrs_local,
        key_nbr_weights=key_nbr_weights,
        lambda_iso=lambda_iso,
        lambda_rigid=lambda_rigid,
        lambda_rot=lambda_rot,
        lambda_vel=lambda_vel,
        lambda_acc=lambda_acc,
    )
    return pos_dev_loss + mot


# --------------------------------------------------------------------------- #
#  Non-key node position + DQB loss (Eq. 13)
# --------------------------------------------------------------------------- #

def non_key_node_loss(
    pos_nk_t:            Tensor,  # (N_n, B, 3)  current non-key positions
    quats_nk_t:          Tensor,  # (N_n, B, 4)
    transforms_nk_t:     Tensor,  # (N_n, B, 3, 4)
    pos_nk_pretrained:   Tensor,  # (N_n, B, 3)  frozen pretrained positions
    u_nk:                Tensor,  # (N_n, B)     scalar uncertainties (non-key)
    R_wc_t:              Tensor,  # (B, 3, 3)
    pos_o_nk:            Tensor,  # (N_n, 3)     canonical positions (non-key)
    # DQB inputs
    R_key_t:             Tensor,  # (N_k, B, 3, 3) current key node rotations
    t_key_t:             Tensor,  # (N_k, B, 3)    current key node translations
    nonkey_nbrs_local:   Tensor,  # (N_n, K+1)  indices into key_idx (local)
    nonkey_nbr_weights:  Tensor,  # (N_n, K+1)
    nonkey_nbrs_global:  Tensor,  # (N_n, K+1)  same but mapped to key-array indices (=nonkey_nbrs_local since we pass key subarrays)
    r_scale: Tuple[float, float, float] = (1.0, 1.0, 0.01),
    lambda_iso: float = 1.0,
    lambda_rigid: float = 1.0,
    lambda_rot: float = 0.01,
    lambda_vel: float = 0.01,
    lambda_acc: float = 0.01,
) -> Tensor:
    """Full non-key node loss L_non_key (Eq. 13).

    L_non_key = sum_{t,i∈V_n} ||p_{i,t} - p^o_{i,t}||^2_{U^{-1}}
              + sum_{t,i∈V_n} ||p_{i,t} - p^DQB_{i,t}||^2_{U^{-1}}
              + L_motion_non_key
    """
    N_n, B = pos_nk_t.shape[:2]
    N_k = R_key_t.shape[0]

    # --- Mahalanobis deviation from pretrained ---
    delta_pre = pos_nk_t - pos_nk_pretrained          # (N_n, B, 3)
    R_wc = R_wc_t.unsqueeze(0).expand(N_n, -1, -1, -1)
    mah_pre = mahalanobis_sq(delta_pre, u_nk, R_wc, r_scale)  # (N_n, B)
    loss_pre = mah_pre.mean()

    # --- Mahalanobis deviation from DQB interpolation ---
    # Compute DQB positions for each sampled frame
    pos_dqb_list = []
    for b in range(B):
        R_k_b = R_key_t[:, b]   # (N_k, 3, 3)
        t_k_b = t_key_t[:, b]   # (N_k, 3)
        p_dqb_b, _ = apply_dqb_to_batch(
            R_key_t=R_k_b,
            t_key_t=t_k_b,
            p_canon=pos_o_nk,
            nonkey_nbrs=nonkey_nbrs_local,
            nonkey_nbr_weights=nonkey_nbr_weights,
        )
        pos_dqb_list.append(p_dqb_b)
    pos_dqb = torch.stack(pos_dqb_list, dim=1)  # (N_n, B, 3)

    delta_dqb = pos_nk_t - pos_dqb.detach()     # (N_n, B, 3) — detach DQB (it's a target)
    mah_dqb = mahalanobis_sq(delta_dqb, u_nk, R_wc, r_scale)  # (N_n, B)
    loss_dqb = mah_dqb.mean()

    # --- Motion loss ---
    mot = motion_loss_non_key(
        pos_t=pos_nk_t,
        quats_t=quats_nk_t,
        transforms_t=transforms_nk_t,
        pos_o=pos_o_nk,
        nonkey_nbrs_local=nonkey_nbrs_local,
        nonkey_nbr_weights=nonkey_nbr_weights,
        lambda_iso=lambda_iso,
        lambda_rigid=lambda_rigid,
        lambda_rot=lambda_rot,
        lambda_vel=lambda_vel,
        lambda_acc=lambda_acc,
    )
    return loss_pre + loss_dqb + mot

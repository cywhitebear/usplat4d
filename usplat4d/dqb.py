"""Dual Quaternion Blending (DQB) for non-key node motion interpolation.

Reference: Kavan et al. 2007, "Skinning with Dual Quaternions"

DQB interpolates SE(3) transforms from key neighbors to produce a smooth
blended transform for each non-key Gaussian. The result is applied to the
canonical (initial) position to get the interpolated trajectory.

Quaternion convention: (w, x, y, z)  throughout this module.
PyTorch's roma library is available but we avoid the dependency here to
keep this self-contained; we only use standard torch ops.
"""

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


# --------------------------------------------------------------------------- #
#  Quaternion helpers  (w, x, y, z)
# --------------------------------------------------------------------------- #

def quat_mul(q1: Tensor, q2: Tensor) -> Tensor:
    """Hamilton product of two unit quaternions.

    Args:
        q1, q2: (..., 4)  (w, x, y, z)
    Returns:
        (..., 4)
    """
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)


def quat_conj(q: Tensor) -> Tensor:
    """Quaternion conjugate.  Args/returns: (..., 4) (w, x, y, z)."""
    w, xyz = q[..., :1], q[..., 1:]
    return torch.cat([w, -xyz], dim=-1)


def rotmat_to_quat(R: Tensor) -> Tensor:
    """Convert rotation matrix to unit quaternion (w, x, y, z).

    Args:
        R: (..., 3, 3)
    Returns:
        q: (..., 4)  (w, x, y, z), unit quaternion
    """
    # Shepperd's method
    batch = R.shape[:-2]
    R = R.reshape(-1, 3, 3)
    N = R.shape[0]

    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    q = torch.zeros(N, 4, device=R.device, dtype=R.dtype)

    # Case 1: trace > 0
    s1 = (0.5 / (trace + 1.0).clamp(min=1e-8).sqrt())
    q_c1 = torch.stack([
        0.25 / s1,
        (R[:, 2, 1] - R[:, 1, 2]) * s1,
        (R[:, 0, 2] - R[:, 2, 0]) * s1,
        (R[:, 1, 0] - R[:, 0, 1]) * s1,
    ], dim=-1)

    # Case 2: R[0,0] is largest diagonal
    s2 = (2.0 * (1.0 + R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2]).clamp(min=1e-8).sqrt())
    q_c2 = torch.stack([
        (R[:, 2, 1] - R[:, 1, 2]) / s2,
        0.25 * s2,
        (R[:, 0, 1] + R[:, 1, 0]) / s2,
        (R[:, 0, 2] + R[:, 2, 0]) / s2,
    ], dim=-1)

    # Case 3: R[1,1] is largest diagonal
    s3 = (2.0 * (1.0 - R[:, 0, 0] + R[:, 1, 1] - R[:, 2, 2]).clamp(min=1e-8).sqrt())
    q_c3 = torch.stack([
        (R[:, 0, 2] - R[:, 2, 0]) / s3,
        (R[:, 0, 1] + R[:, 1, 0]) / s3,
        0.25 * s3,
        (R[:, 1, 2] + R[:, 2, 1]) / s3,
    ], dim=-1)

    # Case 4: R[2,2] is largest diagonal
    s4 = (2.0 * (1.0 - R[:, 0, 0] - R[:, 1, 1] + R[:, 2, 2]).clamp(min=1e-8).sqrt())
    q_c4 = torch.stack([
        (R[:, 1, 0] - R[:, 0, 1]) / s4,
        (R[:, 0, 2] + R[:, 2, 0]) / s4,
        (R[:, 1, 2] + R[:, 2, 1]) / s4,
        0.25 * s4,
    ], dim=-1)

    cond1 = (trace > 0).unsqueeze(-1).expand_as(q)
    cond2 = (~(trace > 0) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
             ).unsqueeze(-1).expand_as(q)
    cond3 = (~(trace > 0) & ~(R[:, 0, 0] > R[:, 1, 1]) & (R[:, 1, 1] > R[:, 2, 2])
             ).unsqueeze(-1).expand_as(q)

    q = torch.where(cond1, q_c1, q)
    q = torch.where(cond2, q_c2, q)
    q = torch.where(~cond1 & ~cond2 & cond3, q_c3, q)
    q = torch.where(~cond1 & ~cond2 & ~cond3, q_c4, q)

    q = F.normalize(q, p=2, dim=-1)
    return q.reshape(batch + (4,))


def quat_to_rotmat(q: Tensor) -> Tensor:
    """Convert unit quaternion (w, x, y, z) to rotation matrix.

    Args:
        q: (..., 4)
    Returns:
        R: (..., 3, 3)
    """
    q = F.normalize(q, p=2, dim=-1)
    w, x, y, z = q.unbind(-1)
    R = torch.stack([
        1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y),
        2*(x*y + w*z),       1 - 2*(x*x + z*z),  2*(y*z - w*x),
        2*(x*z - w*y),       2*(y*z + w*x),       1 - 2*(x*x + y*y),
    ], dim=-1).reshape(q.shape[:-1] + (3, 3))
    return R


# --------------------------------------------------------------------------- #
#  Dual quaternion helpers  (q_r, q_d)  each (w, x, y, z)
# --------------------------------------------------------------------------- #

def se3_to_dual_quat(R: Tensor, t: Tensor) -> Tensor:
    """Encode SE(3) transform as dual quaternion.

    q_r = unit quaternion of R
    q_d = 0.5 * q_t * q_r   where q_t = (0, tx, ty, tz)

    Args:
        R: (..., 3, 3)
        t: (..., 3)
    Returns:
        dq: (..., 8)  concatenated (q_r, q_d) each (w,x,y,z)
    """
    q_r = rotmat_to_quat(R)                               # (..., 4)
    q_t = torch.cat([torch.zeros_like(t[..., :1]), t], dim=-1)  # (..., 4) pure quat with w=0
    q_d = 0.5 * quat_mul(q_t, q_r)                       # (..., 4)
    return torch.cat([q_r, q_d], dim=-1)                  # (..., 8)


def dual_quat_to_se3(dq: Tensor) -> Tuple[Tensor, Tensor]:
    """Decode dual quaternion to SE(3) (R, t).

    Args:
        dq: (..., 8)  (q_r, q_d) each length-4 (w,x,y,z)
    Returns:
        R: (..., 3, 3)
        t: (..., 3)
    """
    q_r = dq[..., :4]
    q_d = dq[..., 4:]
    q_r = F.normalize(q_r, p=2, dim=-1)
    # t = 2 * q_d * conj(q_r)   take imaginary part
    t_quat = 2.0 * quat_mul(q_d, quat_conj(q_r))  # (..., 4)
    t = t_quat[..., 1:]                             # (..., 3)  imaginary part
    R = quat_to_rotmat(q_r)
    return R, t


# --------------------------------------------------------------------------- #
#  DQB main function
# --------------------------------------------------------------------------- #

def dual_quaternion_blend(
    R_key: Tensor,       # (N_nbr, 3, 3)  rotation matrices of key neighbors
    t_key: Tensor,       # (N_nbr, 3)     translation vectors of key neighbors
    weights: Tensor,     # (N_nbr,)       normalized blending weights (sum to 1)
) -> Tuple[Tensor, Tensor]:
    """Blend SE(3) transforms using Dual Quaternion Blending (DQB).

    Implements "Skinning with Dual Quaternions" (Kavan et al. 2007).
    Produces a smooth interpolation of the provided SE(3) transforms.

    Args:
        R_key:   (N_nbr, 3, 3)
        t_key:   (N_nbr, 3)
        weights: (N_nbr,)  non-negative, should sum to 1

    Returns:
        R_blend: (3, 3)
        t_blend: (3,)
    """
    # Encode all key transforms as dual quaternions: (N_nbr, 8)
    dq = se3_to_dual_quat(R_key, t_key)   # (N_nbr, 8)

    # Ensure consistent orientation (shortest-path): flip q_r if dot < 0 w.r.t. first
    q_r_ref = dq[0, :4]                                    # (4,) reference
    dot = (dq[:, :4] * q_r_ref.unsqueeze(0)).sum(dim=-1)   # (N_nbr,)
    flip = (dot < 0).float() * (-2.0) + 1.0               # +1 or -1
    dq = dq * flip.unsqueeze(-1)                           # (N_nbr, 8)

    # Weighted sum
    w = weights.unsqueeze(-1)             # (N_nbr, 1)
    dq_blend = (dq * w).sum(dim=0)       # (8,)

    # Normalize: divide by ||q_r||
    norm_r = dq_blend[:4].norm().clamp(min=1e-8)
    dq_blend = dq_blend / norm_r

    return dual_quat_to_se3(dq_blend)    # (3, 3), (3,)


def apply_dqb_to_batch(
    R_key_t: Tensor,    # (N_k, 3, 3)   key node rotations at frame t
    t_key_t: Tensor,    # (N_k, 3)      key node translations at frame t
    p_canon: Tensor,    # (N_n, 3)      canonical (initial) positions of non-key nodes
    nonkey_nbrs: Tensor,        # (N_n, K+1)   neighbor indices into key idx (local)
    nonkey_nbr_weights: Tensor, # (N_n, K+1)   blending weights
) -> Tuple[Tensor, Tensor]:
    """Apply DQB for all non-key nodes at a single frame.

    Args:
        R_key_t:  (N_k, 3, 3)
        t_key_t:  (N_k, 3)
        p_canon:  (N_n, 3)      canonical positions of non-key Gaussians
        nonkey_nbrs:       (N_n, K+1)  per-non-key neighbor indices (into key nodes)
        nonkey_nbr_weights:(N_n, K+1)  blending weights

    Returns:
        p_dqb: (N_n, 3)   interpolated positions for non-key nodes at frame t
        q_dqb: (N_n, 4)   interpolated quaternions (w,x,y,z)
    """
    N_n, Kp1 = nonkey_nbrs.shape
    device = R_key_t.device

    # Collect neighbor R and t for all non-key nodes at once
    # nonkey_nbrs: (N_n, K+1) indices into key arrays
    flat_idx = nonkey_nbrs.reshape(-1)                           # (N_n*(K+1),)
    R_nbr = R_key_t[flat_idx].reshape(N_n, Kp1, 3, 3)          # (N_n, K+1, 3, 3)
    t_nbr = t_key_t[flat_idx].reshape(N_n, Kp1, 3)             # (N_n, K+1, 3)
    w_nbr = nonkey_nbr_weights                                   # (N_n, K+1)

    # Dual quaternion encoding: (N_n, K+1, 8)
    dq = se3_to_dual_quat(R_nbr, t_nbr)  # (N_n, K+1, 8)

    # Consistent orientation per non-key node (flip to same hemisphere as first neighbor)
    q_r_ref = dq[:, 0:1, :4]                              # (N_n, 1, 4)
    dot = (dq[:, :, :4] * q_r_ref).sum(dim=-1)            # (N_n, K+1)
    flip = (dot < 0).float() * (-2.0) + 1.0               # (N_n, K+1)
    dq = dq * flip.unsqueeze(-1)                           # (N_n, K+1, 8)

    # Weighted blend: (N_n, 8)
    dq_blend = (dq * w_nbr.unsqueeze(-1)).sum(dim=1)       # (N_n, 8)

    # Normalize
    norm_r = dq_blend[:, :4].norm(dim=-1, keepdim=True).clamp(min=1e-8)  # (N_n, 1)
    dq_blend = dq_blend / norm_r.expand(-1, 8)

    # Decode to R, t
    q_r = dq_blend[:, :4]    # (N_n, 4)
    q_d = dq_blend[:, 4:]    # (N_n, 4)
    q_r_norm = F.normalize(q_r, p=2, dim=-1)
    # t = 2 * q_d * conj(q_r)  [imaginary part]
    t_blend = 2.0 * quat_mul(q_d, quat_conj(q_r_norm))[..., 1:]  # (N_n, 3)
    R_blend = quat_to_rotmat(q_r_norm)                             # (N_n, 3, 3)

    # Apply transform to canonical positions: p_dqb = R_blend @ p_canon + t_blend
    p_dqb = (R_blend @ p_canon.unsqueeze(-1)).squeeze(-1) + t_blend  # (N_n, 3)

    # Convert quaternion back (w,x,y,z) for SoM compatibility
    q_dqb = q_r_norm    # (N_n, 4) already (w,x,y,z)

    return p_dqb, q_dqb

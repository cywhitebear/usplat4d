"""Dynamic Uncertainty Estimation (Section 4.1 of USplat4D paper).

For Gaussian i at frame t, the scalar uncertainty is:
    u_{i,t} = (sum_h (T_i^h * alpha_i)^2)^{-1}   if pixel residuals < eta_c
    u_{i,t} = phi                                   otherwise

We approximate sum_h (T_i^h * alpha_i)^2 using:
    - alpha_i^2 * pi * r_i^2  for visible (non-occluded) Gaussians
    - 0  (=> phi)             for occluded or out-of-frustum Gaussians

Occlusion is detected by comparing the Gaussian's camera-space depth against
the rendered depth map at the projected 2D pixel. Convergence is checked by
comparing rendered color vs. ground-truth.

Depth-aware anisotropic 3D uncertainty (Eq. 8):
    U_{i,t} = R_wc * diag(rx*u, ry*u, rz*u) * R_wc^T
with default [rx, ry, rz] = [1, 1, 0.01]  (depth axis down-weighted so that
depth deviations are penalized 100x more in the Mahalanobis loss).
"""

import math
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from gsplat.rendering import rasterization


@torch.no_grad()
def compute_uncertainty_single_frame(
    means_t: Tensor,       # (G, 3)  Gaussian positions in world space at frame t
    quats_t: Tensor,       # (G, 4)  Gaussian rotations (wxyz) at frame t
    scales: Tensor,        # (G, 3)  log-scales (exp gives true scales)
    opacities_raw: Tensor, # (G,)    pre-sigmoid opacities
    w2c: Tensor,           # (4, 4)  world-to-camera
    K: Tensor,             # (3, 3)  camera intrinsics
    img_wh: Tuple[int, int],  # (W, H)
    gt_img: Tensor,        # (H, W, 3)  ground-truth RGB in [0,1]
    eta_c: float = 0.5,    # color-error threshold for convergence check
    phi: float = 1e6,      # large constant for high-uncertainty Gaussians
    depth_margin_rel: float = 0.05,  # relative depth margin for occlusion test
) -> Tuple[Tensor, Tensor]:
    """Compute per-Gaussian uncertainty for a single frame.

    Returns:
        u_scalar: (G,)  scalar uncertainty per Gaussian, in [0, phi]
        radii:    (G,)  2D screen-space radii (pixels), 0 for invisible

    Implementation note:
        We approximate sum_h(v_i^h)^2 ≈ alpha_i^2 * pi * r_i^2 for visible
        Gaussians (where r_i is the screen-space radius from gsplat projection).
        This approximation holds when the Gaussian is the first in depth order
        at its covered pixels (T ≈ 1). Occluded Gaussians get u = phi.
    """
    device = means_t.device
    W, H = img_wh
    G = means_t.shape[0]

    # ---------- 1. Render scene to get depth + RGB + per-Gaussian projections ----------
    # Activate Gaussian parameters
    alpha = torch.sigmoid(opacities_raw)          # (G,)
    scale = torch.exp(scales)                      # (G, 3)
    # quats already in wxyz, gsplat expects wxyz
    # SoM stores canonical quats in wxyz convention

    # Use packed=False to get (1, G, ...) per-Gaussian info
    with torch.enable_grad():
        # Need enable_grad because gsplat's 2D projection creates a computation
        # graph; we only need forward values so we detach afterward.
        render_out, alphas, info = rasterization(
            means=means_t,
            quats=quats_t,
            scales=scale,
            opacities=alpha,
            colors=torch.sigmoid(  # dummy colors for rendering
                torch.zeros(G, 3, device=device)
            ),
            viewmats=w2c.unsqueeze(0),  # (1, 4, 4)
            Ks=K.unsqueeze(0),           # (1, 3, 3)
            width=W,
            height=H,
            packed=False,
            render_mode="RGB+ED",       # return depth too
            near_plane=0.01,
            far_plane=1e10,
        )

    # render_out: (1, H, W, 4) — [RGB, depth]
    rendered_depth = render_out[0, :, :, 3]   # (H, W) camera-space depth
    # rendered_rgb:  not needed here (we render again with real colors below)

    # Per-Gaussian from info (non-packed, shape (1, G)):
    g_radii   = info["radii"]      # (1, G)   or (1, G, 2) — screen-space radius
    g_depths  = info["depths"]     # (1, G)   camera-space depth per Gaussian
    g_means2d = info["means2d"]    # (1, G, 2) projected 2D pixel centers

    # Flatten camera dim
    g_radii   = g_radii[0]        # (G,) or (G, 2)
    g_depths  = g_depths[0]       # (G,)
    g_means2d = g_means2d[0]      # (G, 2)

    # Normalize to scalar radius (max of x/y if 2D)
    if g_radii.dim() == 2:
        g_radii = g_radii.max(dim=-1).values  # (G,)

    # ---------- 2. Compute per-Gaussian scalar uncertainty ----------
    u = torch.full((G,), fill_value=phi, dtype=torch.float32, device=device)

    # Gaussians with radii > 0 are within the camera frustum
    visible_mask = g_radii > 0  # (G,)

    if visible_mask.any():
        vis_idx = visible_mask.nonzero(as_tuple=True)[0]  # (N_vis,)

        # Projected pixel coords (integer, clamped to image bounds)
        px = g_means2d[vis_idx, 0].round().long().clamp(0, W - 1)  # (N_vis,)
        py = g_means2d[vis_idx, 1].round().long().clamp(0, H - 1)  # (N_vis,)

        # Depth comparison for occlusion detection
        d_gauss    = g_depths[vis_idx]                    # (N_vis,)
        d_rendered = rendered_depth[py, px]               # (N_vis,)
        margin     = depth_margin_rel * d_rendered.clamp(min=1e-3)
        not_occluded = d_gauss <= d_rendered + margin     # (N_vis,)

        # Projected 2D area × opacity² (approximation of sum_h (v_i^h)^2)
        r = g_radii[vis_idx]                              # (N_vis,)
        a = alpha[vis_idx]                                # (N_vis,)
        area = math.pi * r.clamp(min=0.5) ** 2           # (N_vis,) at least 1px circle
        inv_sum_sq_v = 1.0 / (a * a * area + 1e-8)       # (N_vis,)

        # Apply large constant where occluded
        u_vis = torch.where(not_occluded, inv_sum_sq_v,
                            torch.full_like(inv_sum_sq_v, phi))
        u[vis_idx] = u_vis

    # ---------- 3. Convergence filter (Eq. 6–7) ----------
    # If any pixel covered by Gaussian i has color error > eta_c, set u_i = phi.
    # We check the projected center pixel as a proxy for the entire footprint.
    if visible_mask.any():
        vis_idx = visible_mask.nonzero(as_tuple=True)[0]
        px = g_means2d[vis_idx, 0].round().long().clamp(0, W - 1)
        py = g_means2d[vis_idx, 1].round().long().clamp(0, H - 1)

        # Render with actual colors for convergence check
        with torch.enable_grad():
            render_rgb_out, _, _ = rasterization(
                means=means_t,
                quats=quats_t,
                scales=torch.exp(scales),
                opacities=torch.sigmoid(opacities_raw),
                colors=torch.zeros(G, 3, device=device),  # placeholder
                viewmats=w2c.unsqueeze(0),
                Ks=K.unsqueeze(0),
                width=W,
                height=H,
                packed=False,
                render_mode="RGB",
            )
        # render_rgb_out: (1, H, W, 3)
        rendered_rgb = render_rgb_out[0]  # (H, W, 3)
        color_err = (rendered_rgb[py, px] - gt_img[py, px]).abs().mean(dim=-1)  # (N_vis,)
        not_converged = color_err > eta_c
        u[vis_idx] = torch.where(not_converged, torch.full_like(u[vis_idx], phi),
                                 u[vis_idx])

    return u, g_radii


@torch.no_grad()
def compute_uncertainty_all_frames(
    model,                      # SoM SceneModel
    train_dataset,              # SoM training dataset
    device: torch.device,
    eta_c: float = 0.5,
    phi: float = 1e6,
    depth_margin_rel: float = 0.05,
) -> Tensor:
    """Compute per-Gaussian per-frame scalar uncertainties for all frames.

    Returns:
        u: (G, T)  uncertainty values.  Low = reliable, phi = unreliable.
    """
    G = model.num_fg_gaussians
    T = model.num_frames
    u_all = torch.full((G, T), fill_value=phi, device=device)

    img_wh = train_dataset.img_wh  # (W, H) — check dataset attribute name

    for t in range(T):
        # Get Gaussian positions at frame t
        with torch.no_grad():
            means_t, quats_t = model.compute_poses_fg(
                torch.tensor([t], device=device)
            )
            means_t = means_t[:, 0]   # (G, 3)
            quats_t = quats_t[:, 0]   # (G, 4)

        w2c = model.w2cs[t]  # (4, 4)
        K   = model.Ks[t]    # (3, 3)

        # Fetch ground-truth image for frame t from dataset
        item = train_dataset[t]
        gt_img = item["imgs"].to(device)      # (H, W, 3), float [0,1]
        if gt_img.dim() == 4:
            gt_img = gt_img[0]

        u_t, _ = compute_uncertainty_single_frame(
            means_t=means_t,
            quats_t=quats_t,
            scales=model.fg.params["scales"],
            opacities_raw=model.fg.params["opacities"],
            w2c=w2c,
            K=K,
            img_wh=img_wh,
            gt_img=gt_img,
            eta_c=eta_c,
            phi=phi,
            depth_margin_rel=depth_margin_rel,
        )
        u_all[:, t] = u_t

    return u_all  # (G, T)


def build_uncertainty_3d_matrices(
    u_scalar: Tensor,    # (G, T)  scalar uncertainties
    w2cs: Tensor,        # (T, 4, 4)  world-to-camera transforms
    r_scale: Tuple[float, float, float] = (1.0, 1.0, 0.01),
) -> Tensor:
    """Build anisotropic 3D uncertainty matrices (Eq. 8 in paper).

    U_{i,t} = R_wc * diag(rx*u, ry*u, rz*u) * R_wc^T

    where R_wc is the camera-to-world rotation at frame t,
    and [rx, ry, rz] = r_scale (default: depth axis down-weighted at 0.01).

    Returns:
        U: (G, T, 3, 3)  per-Gaussian per-frame uncertainty matrices
    """
    G, T = u_scalar.shape
    device = u_scalar.device
    rx, ry, rz = r_scale

    # Extract camera-to-world rotations: R_wc = w2c^{-1}[:3, :3] = w2c[:3,:3]^T
    # w2c[:3,:3] is R_cw, so R_wc = R_cw^T
    R_wc = w2cs[:, :3, :3].transpose(-1, -2)  # (T, 3, 3) = R cam-to-world

    # Build diagonal scale vectors: (3,) -> scaled by (rx, ry, rz)
    r = torch.tensor([rx, ry, rz], device=device, dtype=torch.float32)  # (3,)

    # U_{i,t} = R_wc[t] * diag(r * u[i,t]) * R_wc[t]^T
    # Shape maneuver: (G, T, 3, 3)
    # u_scalar: (G, T)
    # r: (3,)
    # diag values: (G, T, 3) = r[None,None,:] * u_scalar[:,:,None]
    diag_vals = r[None, None, :] * u_scalar[:, :, None]  # (G, T, 3)

    # U = R_wc @ diag(diag_vals) @ R_wc^T
    # Expand R_wc to (1, T, 3, 3) and diag to (G, T, 3, 3)
    R = R_wc[None, :, :, :]           # (1, T, 3, 3)
    D = torch.diag_embed(diag_vals)   # (G, T, 3, 3)

    U = R @ D @ R.transpose(-1, -2)   # (G, T, 3, 3)
    return U


def mahalanobis_sq(
    delta: Tensor,       # (..., 3)  displacement vectors in world space
    u_scalar: Tensor,    # (...,)    scalar uncertainty (per Gaussian per frame)
    R_wc: Tensor,        # (..., 3, 3)  camera-to-world rotation
    r_scale: Tuple[float, float, float] = (1.0, 1.0, 0.01),
) -> Tensor:
    """Compute Mahalanobis squared distance ||delta||^2_{U^{-1}}.

    U = R_wc * diag(rx*u, ry*u, rz*u) * R_wc^T
    U^{-1} = R_wc * diag(1/(rx*u), 1/(ry*u), 1/(rz*u)) * R_wc^T

    ||delta||^2_{U^{-1}} = delta_cam^T * diag(1/(r*u)) * delta_cam
                         = sum_d (delta_cam[d])^2 / (r[d] * u)

    where delta_cam = R_wc^T @ delta (transform to camera space).

    For [rx, ry, rz] = [1, 1, 0.01] and small u:
        x, y deviations penalized by 1/u
        z (depth) deviations penalized by 100/u   => depth errors corrected harder

    Returns:
        loss: (...)  per-element Mahalanobis squared distance
    """
    rx, ry, rz = r_scale
    device = delta.device

    # Transform delta to camera space: delta_cam = R_wc^T @ delta
    # R_wc^T has shape (..., 3, 3), delta has shape (..., 3)
    delta_cam = (R_wc.transpose(-1, -2) @ delta.unsqueeze(-1)).squeeze(-1)  # (..., 3)

    # Scale for U^{-1}: 1 / (r * u)  — large u → small penalty
    r = torch.tensor([rx, ry, rz], device=device, dtype=delta.dtype)
    inv_ru = 1.0 / (r * u_scalar.unsqueeze(-1).clamp(min=1e-8))  # (..., 3)

    # ||delta||^2_{U^{-1}} = sum_d (delta_cam_d)^2 * inv_ru_d
    return (delta_cam ** 2 * inv_ru).sum(dim=-1)  # (...,)
